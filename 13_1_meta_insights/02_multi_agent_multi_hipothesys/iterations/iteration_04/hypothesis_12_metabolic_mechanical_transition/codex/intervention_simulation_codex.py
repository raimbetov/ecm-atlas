from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

DATA_PATH = Path('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
VELOCITY_PATH = Path('../../../iteration_01/hypothesis_03_tissue_aging_clocks/codex/tissue_aging_velocity_codex.csv').resolve()
OUTPUT_DIR = Path(__file__).resolve().parent
EFFECT_PATH = OUTPUT_DIR / 'intervention_effects_codex.csv'
PLOT_PATH = OUTPUT_DIR / 'visualizations_codex/intervention_effects_codex.png'
THRESHOLD = 1.65

STATIC_METABOLIC_GENES = {
    'ATP5A1', 'ATP5B', 'ATP5C1', 'ATP5D', 'ATP5F1', 'COX4I1', 'COX5A', 'COX6A1',
    'NDUFA9', 'NDUFS1', 'NDUFS2', 'SDHB', 'SDHC', 'UQCRC1', 'UQCRC2', 'CS',
    'IDH3A', 'PDHA1', 'PDHB', 'HK1', 'HK2', 'PFKM', 'PFKP', 'ALDOA', 'GAPDH',
    'PGK1', 'ENO1', 'PKM', 'LDHA', 'LDHB', 'SOD2', 'PRDX3', 'ACLY', 'ACAT2'
}


def derive_metabolic_genes(df: pd.DataFrame):
    names = df[['Gene_Symbol', 'Protein_Name', 'Matrisome_Category_Simplified']].dropna().copy()
    names['Protein_Name'] = names['Protein_Name'].str.lower()
    names['Matrisome_Category_Simplified'] = names['Matrisome_Category_Simplified'].str.lower()
    genes = [g for g in STATIC_METABOLIC_GENES if g in names['Gene_Symbol'].values]
    if len(genes) < 5:
        metabolic_categories = names[names['Matrisome_Category_Simplified'].str.contains('regulator|secreted', na=False)]
        metabolic_keywords = ('peroxid', 'dehydrogenase', 'oxidase', 'glycol', 'nad', 'succinate', 'mitochond')
        metabolic_kw = names[names['Protein_Name'].str.contains('|'.join(metabolic_keywords), na=False)]
        genes = pd.concat([metabolic_categories['Gene_Symbol'], metabolic_kw['Gene_Symbol']]).unique().tolist()
    return genes


def load_features():
    df = pd.read_csv(DATA_PATH)
    velocity_df = pd.read_csv(VELOCITY_PATH)
    pivot = df.pivot_table(index='Tissue', columns='Gene_Symbol', values='Zscore_Delta', aggfunc='mean')
    pivot = pivot.reindex(velocity_df['Tissue']).fillna(0.0)
    scaler = StandardScaler()
    features = scaler.fit_transform(pivot)
    y = velocity_df['Velocity'].values
    phases = np.where(y < THRESHOLD, 'Phase_I', 'Phase_II')
    return df, pivot, features, y, phases, scaler


def simulate_rescue():
    df, pivot, features, y, phases, scaler = load_features()
    metabolic_gene_list = derive_metabolic_genes(df)
    model = GradientBoostingRegressor(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    predicted = model.predict(features)

    metabolic_set = set(metabolic_gene_list)
    metabolic_indices = [idx for idx, gene in enumerate(pivot.columns) if gene in metabolic_set]
    delta_matrix = np.zeros_like(features)
    if metabolic_indices:
        delta_matrix[:, metabolic_indices] = 0.5
    rescued_features = features + delta_matrix
    rescued_pred = model.predict(rescued_features)

    effect = rescued_pred - predicted
    results = pd.DataFrame({
        'Tissue': pivot.index,
        'Phase': phases,
        'Baseline_Velocity_Pred': predicted,
        'Rescued_Velocity_Pred': rescued_pred,
        'Delta_Velocity': effect
    })
    results.to_csv(EFFECT_PATH, index=False)

    plt.figure(figsize=(6, 4))
    sns = __import__('seaborn')
    sns.violinplot(data=results, x='Phase', y='Delta_Velocity')
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel('Δ Velocity (rescued - baseline)')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    phase_effect = results.groupby('Phase')['Delta_Velocity'].agg(['mean', 'std', 'count'])
    stats = phase_effect.to_dict()
    with open(OUTPUT_DIR / 'intervention_effects_summary_codex.json', 'w') as f:
        import json
        json.dump({k: {kk: float(vv) for kk, vv in val.items()} for k, val in stats.items()}, f, indent=2)


if __name__ == '__main__':
    simulate_rescue()
