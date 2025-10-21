#!/usr/bin/env python3
"""
PCOLCE Tissue Compartment Analysis - Agent 2

Purpose: Analyze tissue-specific and compartment-specific patterns of PCOLCE
and related collagen processing proteins to test mechanistic hypotheses.

Hypotheses:
H1: Mechanical loading correlation - PCOLCE decrease stronger in high-load tissues
H2: Compartment divergence - Different compartments show different responses
H3: Species conservation - PCOLCE downregulation conserved across human/mouse
H4: Compensatory mechanisms - BMP1, PCOLCE2, LOX enzymes compensate

Author: Agent 2
Date: 2025-10-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
DB_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/PCOLCE research anomaly/agent_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Proteins of interest
TARGET_PROTEINS = {
    'PCOLCE': 'Procollagen C-endopeptidase enhancer 1',
    'PCOLCE2': 'Procollagen C-endopeptidase enhancer 2',
    'BMP1': 'Bone morphogenetic protein 1 (C-proteinase)',
    'LOX': 'Lysyl oxidase',
    'LOXL1': 'Lysyl oxidase-like 1',
    'LOXL2': 'Lysyl oxidase-like 2',
    'LOXL3': 'Lysyl oxidase-like 3',
    'LOXL4': 'Lysyl oxidase-like 4',
    'P4HA1': 'Prolyl 4-hydroxylase subunit alpha 1',
    'P4HA2': 'Prolyl 4-hydroxylase subunit alpha 2',
    'P4HA3': 'Prolyl 4-hydroxylase subunit alpha 3',
    'P4HB': 'Prolyl 4-hydroxylase subunit beta',
    'PLOD1': 'Procollagen-lysine 1,2-oxoglutarate 5-dioxygenase 1',
    'PLOD2': 'Procollagen-lysine 1,2-oxoglutarate 5-dioxygenase 2',
    'PLOD3': 'Procollagen-lysine 1,2-oxoglutarate 5-dioxygenase 3',
    'ADAMTS2': 'ADAM metallopeptidase with thrombospondin type 1 motif 2',
    'ADAMTS3': 'ADAM metallopeptidase with thrombospondin type 1 motif 3',
    'ADAMTS14': 'ADAM metallopeptidase with thrombospondin type 1 motif 14'
}

# Tissue mechanical loading classification
MECHANICAL_LOADING = {
    'Intervertebral_disc': 10,  # Very high - compression
    'Skeletal_muscle': 9,        # Very high - contraction
    'Heart': 8,                  # High - constant contraction
    'Skin dermis': 7,            # Moderate-high - tension
    'Lung': 5,                   # Moderate - breathing
    'Ovary': 2                   # Low - minimal mechanical stress
}


def load_data():
    """Load merged ECM database."""
    print("Loading database...")
    df = pd.read_csv(DB_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Unique genes: {df['Canonical_Gene_Symbol'].nunique()}")
    return df


def filter_target_proteins(df, proteins):
    """Extract only proteins of interest."""
    # Try both Canonical_Gene_Symbol and Gene_Symbol columns
    if 'Canonical_Gene_Symbol' in df.columns:
        mask = df['Canonical_Gene_Symbol'].isin(proteins)
    else:
        mask = df['Gene_Symbol'].isin(proteins)

    filtered = df[mask].copy()
    print(f"\nFiltered to {len(filtered)} rows for {proteins}")
    return filtered


def calculate_protein_summary(df):
    """Calculate summary statistics per protein."""
    summary_rows = []

    # Determine gene symbol column
    gene_col = 'Canonical_Gene_Symbol' if 'Canonical_Gene_Symbol' in df.columns else 'Gene_Symbol'

    for protein in TARGET_PROTEINS.keys():
        prot_data = df[df[gene_col] == protein]

        if len(prot_data) == 0:
            continue

        # Calculate statistics
        mean_dz = prot_data['Zscore_Delta'].mean()
        median_dz = prot_data['Zscore_Delta'].median()
        std_dz = prot_data['Zscore_Delta'].std()
        n_studies = prot_data['Study_ID'].nunique()
        n_tissues = prot_data['Tissue'].nunique()

        # Directional consistency (% negative if mean is negative, % positive if mean is positive)
        if mean_dz < 0:
            consistency = (prot_data['Zscore_Delta'] < 0).sum() / len(prot_data)
        else:
            consistency = (prot_data['Zscore_Delta'] > 0).sum() / len(prot_data)

        # Species
        species = prot_data['Species'].unique()

        summary_rows.append({
            'Protein': protein,
            'Description': TARGET_PROTEINS[protein],
            'N_Measurements': len(prot_data),
            'N_Studies': n_studies,
            'N_Tissues': n_tissues,
            'Mean_Zscore_Delta': mean_dz,
            'Median_Zscore_Delta': median_dz,
            'StdDev_Zscore_Delta': std_dz,
            'Directional_Consistency': consistency,
            'Species': ', '.join(species)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Mean_Zscore_Delta')
    return summary_df


def analyze_pcolce_tissue_specificity(df):
    """Detailed tissue-specific analysis for PCOLCE."""
    gene_col = 'Canonical_Gene_Symbol' if 'Canonical_Gene_Symbol' in df.columns else 'Gene_Symbol'
    pcolce_data = df[df[gene_col] == 'PCOLCE'].copy()

    if len(pcolce_data) == 0:
        print("WARNING: No PCOLCE data found!")
        return None

    tissue_summary = []

    # Group by Study and Tissue/Compartment
    for (study, tissue, compartment), group in pcolce_data.groupby(['Study_ID', 'Tissue', 'Tissue_Compartment']):
        tissue_summary.append({
            'Study': study,
            'Tissue': tissue,
            'Compartment': compartment,
            'Species': group['Species'].iloc[0],
            'N': len(group),
            'Mean_Zscore_Delta': group['Zscore_Delta'].mean(),
            'Median_Zscore_Delta': group['Zscore_Delta'].median(),
            'Min_Zscore_Delta': group['Zscore_Delta'].min(),
            'Max_Zscore_Delta': group['Zscore_Delta'].max(),
            'Mechanical_Loading': MECHANICAL_LOADING.get(tissue, 5)
        })

    tissue_df = pd.DataFrame(tissue_summary)
    tissue_df = tissue_df.sort_values('Mean_Zscore_Delta')

    return tissue_df


def test_mechanical_loading_hypothesis(tissue_df):
    """H1: Test if PCOLCE decrease correlates with mechanical loading."""
    print("\n=== H1: Mechanical Loading Hypothesis ===")

    if tissue_df is None or len(tissue_df) == 0:
        print("No data for testing")
        return None

    # Correlation between mechanical loading and |Zscore_Delta|
    tissue_df['Abs_Zscore_Delta'] = tissue_df['Mean_Zscore_Delta'].abs()

    corr, pval = stats.pearsonr(tissue_df['Mechanical_Loading'],
                                 tissue_df['Abs_Zscore_Delta'])

    print(f"Correlation (Mechanical Loading vs |Δz|): r={corr:.3f}, p={pval:.4f}")

    if pval < 0.05:
        if corr > 0:
            print("✓ SUPPORTED: Higher mechanical loading → stronger PCOLCE decrease")
        else:
            print("✗ REJECTED: Inverse correlation (unexpected)")
    else:
        print("? INCONCLUSIVE: No significant correlation")

    return {'correlation': corr, 'p_value': pval, 'result': tissue_df}


def test_compartment_divergence(df):
    """H2: Test if different compartments within same tissue show divergent patterns."""
    print("\n=== H2: Compartment Divergence Hypothesis ===")

    gene_col = 'Canonical_Gene_Symbol' if 'Canonical_Gene_Symbol' in df.columns else 'Gene_Symbol'
    pcolce_data = df[df[gene_col] == 'PCOLCE'].copy()

    # Focus on Tam_2020 (has NP, IAF, OAF compartments)
    tam_data = pcolce_data[pcolce_data['Study_ID'] == 'Tam_2020']

    if len(tam_data) == 0:
        print("No Tam_2020 data available")
        return None

    compartment_stats = tam_data.groupby('Tissue_Compartment')['Zscore_Delta'].agg(['mean', 'std', 'count'])
    print("\nTam_2020 Intervertebral Disc Compartments:")
    print(compartment_stats)

    # Check if compartments differ significantly
    compartments = tam_data['Tissue_Compartment'].unique()
    if len(compartments) >= 2:
        groups = [tam_data[tam_data['Tissue_Compartment'] == c]['Zscore_Delta'].values
                  for c in compartments]

        if all(len(g) > 0 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"\nANOVA across compartments: F={f_stat:.3f}, p={p_val:.4f}")

            if p_val < 0.05:
                print("✓ SUPPORTED: Significant compartment-specific differences")
            else:
                print("? INCONCLUSIVE: No significant differences between compartments")
        else:
            print("Insufficient data for ANOVA")

    return compartment_stats


def test_species_conservation(df):
    """H3: Test if PCOLCE downregulation is conserved across species."""
    print("\n=== H3: Species Conservation Hypothesis ===")

    gene_col = 'Canonical_Gene_Symbol' if 'Canonical_Gene_Symbol' in df.columns else 'Gene_Symbol'
    pcolce_data = df[df[gene_col] == 'PCOLCE'].copy()

    species_stats = pcolce_data.groupby('Species')['Zscore_Delta'].agg(['mean', 'median', 'std', 'count'])
    print("\nPCOLCE by Species:")
    print(species_stats)

    # Check if both species show negative Δz
    human_mean = species_stats.loc['Homo sapiens', 'mean'] if 'Homo sapiens' in species_stats.index else np.nan
    mouse_mean = species_stats.loc['Mus musculus', 'mean'] if 'Mus musculus' in species_stats.index else np.nan

    if not np.isnan(human_mean) and not np.isnan(mouse_mean):
        if human_mean < 0 and mouse_mean < 0:
            print(f"✓ SUPPORTED: Both human (Δz={human_mean:.2f}) and mouse (Δz={mouse_mean:.2f}) show PCOLCE decrease")
        else:
            print(f"✗ REJECTED: Discordant directions (Human: {human_mean:.2f}, Mouse: {mouse_mean:.2f})")
    else:
        print("? INCONCLUSIVE: Missing data for one or both species")

    return species_stats


def test_compensatory_mechanisms(protein_summary):
    """H4: Test if other proteins compensate for PCOLCE decrease."""
    print("\n=== H4: Compensatory Mechanisms Hypothesis ===")

    if protein_summary is None or len(protein_summary) == 0:
        print("No data for testing")
        return None

    # Check PCOLCE2, BMP1, LOX enzymes
    pcolce_dz = protein_summary[protein_summary['Protein'] == 'PCOLCE']['Mean_Zscore_Delta'].values[0] if 'PCOLCE' in protein_summary['Protein'].values else np.nan

    compensatory_candidates = ['PCOLCE2', 'BMP1', 'LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4']

    print(f"\nPCOLCE Mean Δz: {pcolce_dz:.2f}")
    print("\nCompensatory candidates:")

    compensators = []
    for protein in compensatory_candidates:
        if protein in protein_summary['Protein'].values:
            prot_dz = protein_summary[protein_summary['Protein'] == protein]['Mean_Zscore_Delta'].values[0]
            n_studies = protein_summary[protein_summary['Protein'] == protein]['N_Studies'].values[0]

            print(f"  {protein}: Δz={prot_dz:+.2f} ({n_studies} studies)")

            # Compensation = opposite sign to PCOLCE
            if prot_dz > 0 and pcolce_dz < 0:
                compensators.append(protein)

    if compensators:
        print(f"\n✓ POTENTIAL COMPENSATORS (upregulated): {', '.join(compensators)}")
    else:
        print("\n✗ NO COMPENSATORS: No candidate proteins show upregulation")

    return compensators


def create_visualizations(df, protein_summary, tissue_df):
    """Generate comprehensive visualizations."""
    print("\n=== Creating Visualizations ===")

    fig = plt.figure(figsize=(20, 12))

    # 1. Protein summary heatmap
    ax1 = plt.subplot(2, 3, 1)
    if protein_summary is not None and len(protein_summary) > 0:
        proteins_plot = protein_summary.head(15)  # Top 15
        y_pos = np.arange(len(proteins_plot))
        colors = ['red' if x < 0 else 'blue' for x in proteins_plot['Mean_Zscore_Delta']]

        ax1.barh(y_pos, proteins_plot['Mean_Zscore_Delta'], color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(proteins_plot['Protein'], fontsize=9)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Mean Zscore Delta', fontsize=10)
        ax1.set_title('Collagen Processing Proteins\nMean Δz During Aging', fontsize=11, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

    # 2. PCOLCE tissue-specific patterns
    ax2 = plt.subplot(2, 3, 2)
    if tissue_df is not None and len(tissue_df) > 0:
        tissues_plot = tissue_df.sort_values('Mean_Zscore_Delta')
        y_pos = np.arange(len(tissues_plot))

        ax2.barh(y_pos, tissues_plot['Mean_Zscore_Delta'], color='crimson', alpha=0.7)
        ax2.set_yticks(y_pos)
        labels = [f"{row['Tissue']}\n{row['Compartment']}" for _, row in tissues_plot.iterrows()]
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('Mean Zscore Delta', fontsize=10)
        ax2.set_title('PCOLCE by Tissue/Compartment', fontsize=11, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

    # 3. Mechanical loading correlation
    ax3 = plt.subplot(2, 3, 3)
    if tissue_df is not None and len(tissue_df) > 0:
        ax3.scatter(tissue_df['Mechanical_Loading'],
                   tissue_df['Mean_Zscore_Delta'].abs(),
                   s=100, alpha=0.6, c='darkgreen')

        # Add labels
        for _, row in tissue_df.iterrows():
            ax3.annotate(row['Tissue'][:10],
                        (row['Mechanical_Loading'], abs(row['Mean_Zscore_Delta'])),
                        fontsize=7, alpha=0.7)

        ax3.set_xlabel('Mechanical Loading Score', fontsize=10)
        ax3.set_ylabel('|Mean Zscore Delta|', fontsize=10)
        ax3.set_title('H1: Mechanical Loading\nvs PCOLCE Decrease', fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3)

    # 4. Species comparison
    ax4 = plt.subplot(2, 3, 4)
    gene_col = 'Canonical_Gene_Symbol' if 'Canonical_Gene_Symbol' in df.columns else 'Gene_Symbol'
    pcolce_data = df[df[gene_col] == 'PCOLCE']

    if len(pcolce_data) > 0:
        species_data = [pcolce_data[pcolce_data['Species'] == sp]['Zscore_Delta'].values
                       for sp in pcolce_data['Species'].unique()]
        species_labels = pcolce_data['Species'].unique()

        bp = ax4.boxplot(species_data, labels=species_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax4.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_ylabel('Zscore Delta', fontsize=10)
        ax4.set_title('H3: Species Conservation\nPCOLCE Δz Distribution', fontsize=11, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

    # 5. Compensatory network
    ax5 = plt.subplot(2, 3, 5)
    if protein_summary is not None and len(protein_summary) > 0:
        # Focus on key proteins
        key_proteins = ['PCOLCE', 'PCOLCE2', 'BMP1', 'LOX', 'LOXL1', 'LOXL2',
                       'P4HA1', 'P4HA2', 'PLOD1', 'PLOD2', 'PLOD3']
        key_data = protein_summary[protein_summary['Protein'].isin(key_proteins)]

        if len(key_data) > 0:
            x_pos = np.arange(len(key_data))
            colors = ['darkred' if p == 'PCOLCE' else 'gray' for p in key_data['Protein']]

            ax5.bar(x_pos, key_data['Mean_Zscore_Delta'], color=colors, alpha=0.7)
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(key_data['Protein'], rotation=45, ha='right', fontsize=8)
            ax5.axhline(0, color='black', linewidth=0.5)
            ax5.set_ylabel('Mean Zscore Delta', fontsize=10)
            ax5.set_title('H4: Compensatory Network\nCollagen Processing Enzymes', fontsize=11, fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)

    # 6. Study-level consistency
    ax6 = plt.subplot(2, 3, 6)
    if len(pcolce_data) > 0:
        study_means = pcolce_data.groupby('Study_ID')['Zscore_Delta'].mean().sort_values()

        ax6.barh(np.arange(len(study_means)), study_means.values, color='steelblue', alpha=0.7)
        ax6.set_yticks(np.arange(len(study_means)))
        ax6.set_yticklabels(study_means.index, fontsize=9)
        ax6.axvline(0, color='red', linestyle='--', linewidth=1)
        ax6.axvline(-1.41, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Expected=-1.41')
        ax6.set_xlabel('Mean Zscore Delta', fontsize=10)
        ax6.set_title('PCOLCE by Study\nDirectional Consistency', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "pcolce_tissue_analysis_agent_2.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")

    plt.close()


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("PCOLCE Tissue Compartment Analysis - Agent 2")
    print("="*60)

    # Load data
    df = load_data()

    # Filter to target proteins
    df_proteins = filter_target_proteins(df, TARGET_PROTEINS.keys())

    # Calculate protein summary
    print("\n=== Protein Summary Statistics ===")
    protein_summary = calculate_protein_summary(df_proteins)
    print(protein_summary.to_string(index=False))

    # Save protein summary
    protein_summary.to_csv(OUTPUT_DIR / "protein_summary_agent_2.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'protein_summary_agent_2.csv'}")

    # PCOLCE tissue-specific analysis
    print("\n=== PCOLCE Tissue-Specific Analysis ===")
    tissue_df = analyze_pcolce_tissue_specificity(df_proteins)
    if tissue_df is not None:
        print(tissue_df.to_string(index=False))
        tissue_df.to_csv(OUTPUT_DIR / "pcolce_tissue_compartment_agent_2.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'pcolce_tissue_compartment_agent_2.csv'}")

    # Test hypotheses
    h1_result = test_mechanical_loading_hypothesis(tissue_df)
    h2_result = test_compartment_divergence(df_proteins)
    h3_result = test_species_conservation(df_proteins)
    h4_result = test_compensatory_mechanisms(protein_summary)

    # Create visualizations
    create_visualizations(df_proteins, protein_summary, tissue_df)

    # Summary report
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print(f"1. PCOLCE detected in {protein_summary[protein_summary['Protein']=='PCOLCE']['N_Studies'].values[0] if 'PCOLCE' in protein_summary['Protein'].values else 0} studies")
    print(f"2. Mean PCOLCE Δz: {protein_summary[protein_summary['Protein']=='PCOLCE']['Mean_Zscore_Delta'].values[0]:.2f}")
    print(f"3. Directional consistency: {protein_summary[protein_summary['Protein']=='PCOLCE']['Directional_Consistency'].values[0]*100:.1f}%")

    print("\nOutputs generated:")
    print(f"  - {OUTPUT_DIR / 'protein_summary_agent_2.csv'}")
    print(f"  - {OUTPUT_DIR / 'pcolce_tissue_compartment_agent_2.csv'}")
    print(f"  - {OUTPUT_DIR / 'pcolce_tissue_analysis_agent_2.png'}")


if __name__ == "__main__":
    main()
