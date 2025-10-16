#!/usr/bin/env python3
"""
AGENT 17: Collagen Crosslinking & Matrix Entropy Analyzer

Mission: Connect crosslinking enzymes to matrix stiffening in DEATh theorem context.
Hypothesis: LOX/PLOD enzymes create crosslinks → reduced matrix entropy → cellular aging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/10_insights")
OUTPUT_DIR.mkdir(exist_ok=True)

# Define target proteins
CROSSLINKING_ENZYMES = {
    'LOX': 'Lysyl oxidase',
    'LOXL1': 'Lysyl oxidase homolog 1',
    'LOXL2': 'Lysyl oxidase homolog 2',
    'LOXL3': 'Lysyl oxidase homolog 3',
    'LOXL4': 'Lysyl oxidase homolog 4',
    'PLOD1': 'Procollagen-lysine,2-oxoglutarate 5-dioxygenase 1',
    'PLOD2': 'Procollagen-lysine,2-oxoglutarate 5-dioxygenase 2',
    'PLOD3': 'Procollagen-lysine,2-oxoglutarate 5-dioxygenase 3',
    'P4HA1': 'Prolyl 4-hydroxylase subunit alpha-1',
    'P4HA2': 'Prolyl 4-hydroxylase subunit alpha-2',
    'TGM2': 'Transglutaminase 2'
}

DEGRADATION_ENZYMES = {
    'MMP1': 'Matrix metalloproteinase-1',
    'MMP2': 'Matrix metalloproteinase-2',
    'MMP3': 'Matrix metalloproteinase-3',
    'MMP9': 'Matrix metalloproteinase-9',
    'MMP13': 'Matrix metalloproteinase-13',
    'MMP14': 'Matrix metalloproteinase-14',
    'CTSK': 'Cathepsin K',
    'CTSL': 'Cathepsin L',
    'CTSB': 'Cathepsin B',
    'ELANE': 'Neutrophil elastase',
    'ELN': 'Elastin (elastase substrate)'
}

INFLAMMATORY_MARKERS = {
    'IL6': 'Interleukin-6',
    'IL1B': 'Interleukin-1 beta',
    'TNF': 'Tumor necrosis factor',
    'CXCL8': 'Interleukin-8',
    'CCL2': 'C-C motif chemokine 2',
    'CCL3': 'C-C motif chemokine 3',
    'CCL5': 'C-C motif chemokine 5',
    'PTGS2': 'Prostaglandin G/H synthase 2 (COX-2)',
    'NOS2': 'Nitric oxide synthase 2'
}

SENESCENCE_MARKERS = {
    'CDKN1A': 'Cyclin-dependent kinase inhibitor 1A (p21)',
    'CDKN2A': 'Cyclin-dependent kinase inhibitor 2A (p16)',
    'TP53': 'Tumor protein p53',
    'RB1': 'Retinoblastoma protein',
    'SERPINE1': 'Plasminogen activator inhibitor 1 (PAI-1)',
    'IL6': 'Interleukin-6 (SASP)',
    'IL1A': 'Interleukin-1 alpha',
    'CXCL1': 'C-X-C motif chemokine 1'
}

LONGEVITY_PROTEINS = {
    'SIRT1': 'Sirtuin 1',
    'SIRT3': 'Sirtuin 3',
    'FOXO1': 'Forkhead box protein O1',
    'FOXO3': 'Forkhead box protein O3',
    'KLOTHO': 'Klotho',
    'SOD1': 'Superoxide dismutase 1',
    'SOD2': 'Superoxide dismutase 2',
    'CAT': 'Catalase',
    'GPX1': 'Glutathione peroxidase 1'
}

COLLAGENS = ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL5A1', 'COL6A1']

TGFB_PATHWAY = {
    'TGFB1': 'Transforming growth factor beta-1',
    'TGFB2': 'Transforming growth factor beta-2',
    'TGFB3': 'Transforming growth factor beta-3',
    'TGFBR1': 'TGF-beta receptor type-1',
    'TGFBR2': 'TGF-beta receptor type-2',
    'SMAD2': 'SMAD family member 2',
    'SMAD3': 'SMAD family member 3',
    'SMAD4': 'SMAD family member 4'
}


def load_data():
    """Load and prepare ECM dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records from {df['Study_ID'].nunique()} studies")
    print(f"Tissues: {df['Tissue'].nunique()}")
    print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    return df


def extract_protein_group(df, protein_dict, group_name):
    """Extract data for specific protein group."""
    genes = list(protein_dict.keys())
    subset = df[df['Canonical_Gene_Symbol'].isin(genes)].copy()
    subset['Protein_Group'] = group_name
    print(f"\n{group_name}: Found {subset['Canonical_Gene_Symbol'].nunique()} proteins")
    print(f"  Genes found: {sorted(subset['Canonical_Gene_Symbol'].unique())}")
    return subset


def calculate_crosslinking_index(df):
    """Calculate crosslinking index per tissue/study."""
    results = []

    for (tissue, study), group_df in df.groupby(['Tissue', 'Study_ID']):
        # Get mean z-scores for each category
        crosslink_data = group_df[group_df['Canonical_Gene_Symbol'].isin(CROSSLINKING_ENZYMES.keys())]
        degrad_data = group_df[group_df['Canonical_Gene_Symbol'].isin(DEGRADATION_ENZYMES.keys())]

        if len(crosslink_data) > 0 and len(degrad_data) > 0:
            crosslink_old = crosslink_data['Zscore_Old'].mean()
            crosslink_young = crosslink_data['Zscore_Young'].mean()
            degrad_old = degrad_data['Zscore_Old'].mean()
            degrad_young = degrad_data['Zscore_Young'].mean()

            # Calculate index (higher = more crosslinking relative to degradation)
            index_old = crosslink_old - degrad_old
            index_young = crosslink_young - degrad_young
            delta_index = index_old - index_young

            results.append({
                'Tissue': tissue,
                'Study_ID': study,
                'Crosslink_Old': crosslink_old,
                'Crosslink_Young': crosslink_young,
                'Degrad_Old': degrad_old,
                'Degrad_Young': degrad_young,
                'Index_Old': index_old,
                'Index_Young': index_young,
                'Delta_Index': delta_index,
                'N_Crosslink_Proteins': len(crosslink_data['Canonical_Gene_Symbol'].unique()),
                'N_Degrad_Proteins': len(degrad_data['Canonical_Gene_Symbol'].unique())
            })

    return pd.DataFrame(results)


def analyze_protein_trajectories(df, protein_dict, title):
    """Analyze aging trajectories for protein group."""
    genes = list(protein_dict.keys())
    subset = df[df['Canonical_Gene_Symbol'].isin(genes)].copy()

    if len(subset) == 0:
        print(f"No data for {title}")
        return None

    # Calculate delta z-scores
    subset = subset.dropna(subset=['Zscore_Old', 'Zscore_Young'])
    subset['Zscore_Delta'] = subset['Zscore_Old'] - subset['Zscore_Young']

    # Summary statistics
    summary = subset.groupby('Canonical_Gene_Symbol').agg({
        'Zscore_Delta': ['mean', 'std', 'count'],
        'Zscore_Old': 'mean',
        'Zscore_Young': 'mean'
    }).round(3)

    print(f"\n{title} Trajectories:")
    print(summary.to_string())

    # Statistical test
    if len(subset) > 0:
        t_stat, p_val = stats.ttest_1samp(subset['Zscore_Delta'].dropna(), 0)
        print(f"\nOne-sample t-test (H0: delta=0): t={t_stat:.3f}, p={p_val:.4f}")

    return subset


def test_correlation(df, genes1, genes2, label1, label2):
    """Test correlation between two protein groups."""
    # Get aggregated z-scores per tissue/study
    results = []

    for (tissue, study), group_df in df.groupby(['Tissue', 'Study_ID']):
        data1 = group_df[group_df['Canonical_Gene_Symbol'].isin(genes1)]
        data2 = group_df[group_df['Canonical_Gene_Symbol'].isin(genes2)]

        if len(data1) > 0 and len(data2) > 0:
            results.append({
                'Tissue': tissue,
                'Study_ID': study,
                label1: data1['Zscore_Delta'].mean(),
                label2: data2['Zscore_Delta'].mean()
            })

    if len(results) == 0:
        print(f"\nNo overlapping data for {label1} vs {label2}")
        return None

    corr_df = pd.DataFrame(results)

    # Calculate correlation
    r, p = stats.pearsonr(corr_df[label1].dropna(), corr_df[label2].dropna())
    print(f"\n{label1} vs {label2}:")
    print(f"  Pearson r = {r:.3f}, p = {p:.4f}")
    print(f"  N tissues = {len(corr_df)}")

    return corr_df


def plot_trajectories(crosslink_data, degrad_data, output_path):
    """Plot enzyme trajectories across tissues."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Crosslinking enzymes by tissue
    ax = axes[0, 0]
    if len(crosslink_data) > 0:
        pivot = crosslink_data.pivot_table(
            values='Zscore_Delta',
            index='Canonical_Gene_Symbol',
            columns='Tissue',
            aggfunc='mean'
        )
        sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Z-score Change (Old - Young)'}, ax=ax)
        ax.set_title('Crosslinking Enzymes: Aging Trajectories by Tissue', fontsize=12, weight='bold')
        ax.set_xlabel('Tissue')
        ax.set_ylabel('Enzyme')

    # 2. Degradation enzymes by tissue
    ax = axes[0, 1]
    if len(degrad_data) > 0:
        pivot = degrad_data.pivot_table(
            values='Zscore_Delta',
            index='Canonical_Gene_Symbol',
            columns='Tissue',
            aggfunc='mean'
        )
        sns.heatmap(pivot, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                   cbar_kws={'label': 'Z-score Change (Old - Young)'}, ax=ax)
        ax.set_title('Degradation Enzymes: Aging Trajectories by Tissue', fontsize=12, weight='bold')
        ax.set_xlabel('Tissue')
        ax.set_ylabel('Enzyme')

    # 3. Crosslinking enzyme distribution
    ax = axes[1, 0]
    if len(crosslink_data) > 0:
        crosslink_summary = crosslink_data.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean().sort_values()
        colors = ['red' if x > 0 else 'blue' for x in crosslink_summary.values]
        crosslink_summary.plot(kind='barh', color=colors, ax=ax)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Mean Z-score Change (Old - Young)', fontsize=11)
        ax.set_ylabel('Enzyme', fontsize=11)
        ax.set_title('Crosslinking Enzymes: Mean Change with Aging', fontsize=12, weight='bold')
        ax.grid(axis='x', alpha=0.3)

    # 4. Degradation enzyme distribution
    ax = axes[1, 1]
    if len(degrad_data) > 0:
        degrad_summary = degrad_data.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean().sort_values()
        colors = ['red' if x > 0 else 'blue' for x in degrad_summary.values]
        degrad_summary.plot(kind='barh', color=colors, ax=ax)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Mean Z-score Change (Old - Young)', fontsize=11)
        ax.set_ylabel('Enzyme', fontsize=11)
        ax.set_title('Degradation Enzymes: Mean Change with Aging', fontsize=12, weight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved trajectory plot: {output_path}")
    plt.close()


def plot_crosslinking_index(index_df, output_path):
    """Plot crosslinking index by tissue."""
    if len(index_df) == 0:
        print("No crosslinking index data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Delta index by tissue
    ax = axes[0, 0]
    tissue_summary = index_df.groupby('Tissue')['Delta_Index'].mean().sort_values()
    colors = ['red' if x > 0 else 'blue' for x in tissue_summary.values]
    tissue_summary.plot(kind='barh', color=colors, ax=ax)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Crosslinking Index Change', fontsize=11)
    ax.set_ylabel('Tissue', fontsize=11)
    ax.set_title('Matrix Rigidification Index by Tissue\n(Positive = Increased Crosslinking)',
                fontsize=12, weight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 2. Old vs Young comparison
    ax = axes[0, 1]
    x = np.arange(len(index_df))
    width = 0.35
    ax.bar(x - width/2, index_df['Index_Young'], width, label='Young', alpha=0.7, color='blue')
    ax.bar(x + width/2, index_df['Index_Old'], width, label='Old', alpha=0.7, color='red')
    ax.set_xlabel('Tissue', fontsize=11)
    ax.set_ylabel('Crosslinking Index', fontsize=11)
    ax.set_title('Crosslinking Index: Young vs Old', fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(index_df['Tissue'], rotation=45, ha='right')
    ax.legend()
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    # 3. Crosslink vs Degradation enzymes (Old)
    ax = axes[1, 0]
    ax.scatter(index_df['Degrad_Old'], index_df['Crosslink_Old'],
              s=100, alpha=0.6, c='red', edgecolors='black')
    for idx, row in index_df.iterrows():
        ax.annotate(row['Tissue'], (row['Degrad_Old'], row['Crosslink_Old']),
                   fontsize=8, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Degradation Enzymes Z-score (Old)', fontsize=11)
    ax.set_ylabel('Crosslinking Enzymes Z-score (Old)', fontsize=11)
    ax.set_title('Crosslinking vs Degradation: Old Samples', fontsize=12, weight='bold')
    ax.grid(alpha=0.3)

    # 4. Index delta distribution
    ax = axes[1, 1]
    ax.hist(index_df['Delta_Index'], bins=15, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(index_df['Delta_Index'].mean(), color='orange', linestyle='-',
              linewidth=2, label=f'Mean = {index_df["Delta_Index"].mean():.3f}')
    ax.set_xlabel('Crosslinking Index Change', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Matrix Rigidification\n(Positive = Stiffening with Age)',
                fontsize=12, weight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved crosslinking index plot: {output_path}")
    plt.close()


def plot_correlations(corr_results, output_path):
    """Plot correlation analyses."""
    n_plots = len([x for x in corr_results.values() if x is not None])
    if n_plots == 0:
        print("No correlation data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_idx = 0

    # Plot each correlation
    for (label, data) in corr_results.items():
        if data is not None and plot_idx < 4:
            ax = axes[plot_idx]
            cols = [c for c in data.columns if c not in ['Tissue', 'Study_ID']]

            if len(cols) == 2:
                x_col, y_col = cols
                ax.scatter(data[x_col], data[y_col], s=100, alpha=0.6, edgecolors='black')

                # Add tissue labels
                for idx, row in data.iterrows():
                    ax.annotate(row['Tissue'], (row[x_col], row[y_col]),
                               fontsize=8, alpha=0.7)

                # Add trend line
                z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
                p = np.poly1d(z)
                x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

                # Calculate correlation
                r, p_val = stats.pearsonr(data[x_col].dropna(), data[y_col].dropna())

                ax.set_xlabel(x_col.replace('_', ' '), fontsize=11)
                ax.set_ylabel(y_col.replace('_', ' '), fontsize=11)
                ax.set_title(f'{label}\nr = {r:.3f}, p = {p_val:.4f}',
                           fontsize=12, weight='bold')
                ax.grid(alpha=0.3)
                ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

                plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, 4):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation plot: {output_path}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("AGENT 17: Collagen Crosslinking & Matrix Entropy Analyzer")
    print("="*80)

    # Load data
    df = load_data()

    # Extract protein groups
    print("\n" + "="*80)
    print("EXTRACTING PROTEIN GROUPS")
    print("="*80)

    crosslink_data = extract_protein_group(df, CROSSLINKING_ENZYMES, "Crosslinking")
    degrad_data = extract_protein_group(df, DEGRADATION_ENZYMES, "Degradation")
    inflam_data = extract_protein_group(df, INFLAMMATORY_MARKERS, "Inflammatory")
    senesc_data = extract_protein_group(df, SENESCENCE_MARKERS, "Senescence")
    longevity_data = extract_protein_group(df, LONGEVITY_PROTEINS, "Longevity")
    collagen_data = df[df['Canonical_Gene_Symbol'].isin(COLLAGENS)].copy()
    tgfb_data = extract_protein_group(df, TGFB_PATHWAY, "TGF-beta")

    # Analyze trajectories
    print("\n" + "="*80)
    print("ANALYZING PROTEIN TRAJECTORIES")
    print("="*80)

    crosslink_traj = analyze_protein_trajectories(crosslink_data, CROSSLINKING_ENZYMES,
                                                  "Crosslinking Enzymes")
    degrad_traj = analyze_protein_trajectories(degrad_data, DEGRADATION_ENZYMES,
                                               "Degradation Enzymes")

    # Calculate crosslinking index
    print("\n" + "="*80)
    print("CALCULATING MATRIX RIGIDIFICATION INDEX")
    print("="*80)

    index_df = calculate_crosslinking_index(df)
    print(f"\nCalculated index for {len(index_df)} tissue/study combinations")
    print("\nTop 5 tissues with increased rigidification:")
    print(index_df.nlargest(5, 'Delta_Index')[['Tissue', 'Delta_Index',
                                                'N_Crosslink_Proteins', 'N_Degrad_Proteins']])

    # Test correlations
    print("\n" + "="*80)
    print("TESTING CORRELATIONS")
    print("="*80)

    corr_results = {}

    # 1. Crosslinking vs Inflammatory
    corr_results['Crosslinking vs Inflammation'] = test_correlation(
        df, list(CROSSLINKING_ENZYMES.keys()), list(INFLAMMATORY_MARKERS.keys()),
        'Crosslinking_Delta', 'Inflammation_Delta'
    )

    # 2. Crosslinking vs Senescence
    corr_results['Crosslinking vs Senescence'] = test_correlation(
        df, list(CROSSLINKING_ENZYMES.keys()), list(SENESCENCE_MARKERS.keys()),
        'Crosslinking_Delta', 'Senescence_Delta'
    )

    # 3. Crosslinking vs Longevity
    corr_results['Crosslinking vs Longevity'] = test_correlation(
        df, list(CROSSLINKING_ENZYMES.keys()), list(LONGEVITY_PROTEINS.keys()),
        'Crosslinking_Delta', 'Longevity_Delta'
    )

    # 4. Crosslinking vs Collagen
    corr_results['Crosslinking vs Collagen'] = test_correlation(
        df, list(CROSSLINKING_ENZYMES.keys()), COLLAGENS,
        'Crosslinking_Delta', 'Collagen_Delta'
    )

    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plot_trajectories(
        crosslink_data, degrad_data,
        OUTPUT_DIR / "agent_17_enzyme_trajectories.png"
    )

    plot_crosslinking_index(
        index_df,
        OUTPUT_DIR / "agent_17_crosslinking_index.png"
    )

    plot_correlations(
        corr_results,
        OUTPUT_DIR / "agent_17_correlations.png"
    )

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    if crosslink_traj is not None and len(crosslink_traj) > 0:
        crosslink_traj.to_csv(OUTPUT_DIR / "agent_17_crosslinking_trajectories.csv", index=False)
        print(f"Saved: agent_17_crosslinking_trajectories.csv ({len(crosslink_traj)} rows)")

    if len(index_df) > 0:
        index_df.to_csv(OUTPUT_DIR / "agent_17_crosslinking_index.csv", index=False)
        print(f"Saved: agent_17_crosslinking_index.csv ({len(index_df)} rows)")

    # Save correlation results
    for label, data in corr_results.items():
        if data is not None:
            filename = f"agent_17_correlation_{label.lower().replace(' ', '_')}.csv"
            data.to_csv(OUTPUT_DIR / filename, index=False)
            print(f"Saved: {filename} ({len(data)} rows)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
