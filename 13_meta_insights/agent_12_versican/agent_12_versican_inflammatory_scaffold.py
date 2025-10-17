#!/usr/bin/env python3
"""
AGENT 12: VERSICAN-HYALURONAN INFLAMMATORY SCAFFOLD ANALYZER

Dissects the versican-hyaluronan-ITIH complex inflammatory scaffold and its role in aging.
Tests hypothesis: VCAN accumulation creates pro-inflammatory matrix that traps immune cells.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load dataset
print("Loading ECM aging dataset...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Define proteins of interest for the inflammatory scaffold
SCAFFOLD_PROTEINS = {
    'VCAN': 'Versican (core proteoglycan)',
    'TNFAIP6': 'TSG-6 (inflammatory mediator)',
    'ITIH1': 'Inter-alpha-trypsin inhibitor heavy chain H1',
    'ITIH2': 'Inter-alpha-trypsin inhibitor heavy chain H2',
    'ITIH3': 'Inter-alpha-trypsin inhibitor heavy chain H3',
    'ITIH4': 'Inter-alpha-trypsin inhibitor heavy chain H4',
    'ITIH5': 'Inter-alpha-trypsin inhibitor heavy chain H5',
    'HAS1': 'Hyaluronan synthase 1',
    'HAS2': 'Hyaluronan synthase 2',
    'HAS3': 'Hyaluronan synthase 3',
    'HYAL1': 'Hyaluronidase 1',
    'HYAL2': 'Hyaluronidase 2',
    'CD44': 'CD44 (HA receptor)',
    'RHAMM': 'HMMR (HA-mediated motility receptor)'
}

# TGF-beta pathway proteins (for temporal analysis)
TGFB_PROTEINS = {
    'TGFB1': 'TGF-beta 1',
    'TGFB2': 'TGF-beta 2',
    'TGFB3': 'TGF-beta 3',
    'TGFBR1': 'TGF-beta receptor type-1',
    'TGFBR2': 'TGF-beta receptor type-2',
    'SMAD2': 'SMAD2',
    'SMAD3': 'SMAD3',
    'LTBP1': 'Latent TGF-beta-binding protein 1',
    'LTBP2': 'Latent TGF-beta-binding protein 2'
}

# Other fibrotic markers
FIBROTIC_MARKERS = {
    'COL1A1': 'Collagen type I alpha 1',
    'COL1A2': 'Collagen type I alpha 2',
    'COL3A1': 'Collagen type III alpha 1',
    'FN1': 'Fibronectin',
    'ACTA2': 'Alpha-smooth muscle actin',
    'CTGF': 'Connective tissue growth factor',
    'LOX': 'Lysyl oxidase',
    'LOXL2': 'Lysyl oxidase-like 2'
}

# Inflammatory cytokines/factors
INFLAMMATORY_MARKERS = {
    'IL6': 'Interleukin-6',
    'IL1B': 'Interleukin-1 beta',
    'TNF': 'Tumor necrosis factor',
    'CCL2': 'C-C motif chemokine 2',
    'CCL3': 'C-C motif chemokine 3',
    'CXCL8': 'C-X-C motif chemokine 8',
    'S100A8': 'S100 calcium-binding protein A8',
    'S100A9': 'S100 calcium-binding protein A9',
    'LCN2': 'Lipocalin-2'
}

def extract_protein_data(df, gene_symbols):
    """Extract data for specific proteins."""
    mask = df['Canonical_Gene_Symbol'].isin(gene_symbols)
    protein_df = df[mask].copy()
    return protein_df

def calculate_zscore_delta_stats(protein_df):
    """Calculate statistics for z-score deltas by protein."""
    stats_dict = {}

    for gene in protein_df['Canonical_Gene_Symbol'].unique():
        gene_data = protein_df[protein_df['Canonical_Gene_Symbol'] == gene]['Zscore_Delta'].dropna()

        if len(gene_data) > 0:
            stats_dict[gene] = {
                'mean_delta': gene_data.mean(),
                'median_delta': gene_data.median(),
                'std_delta': gene_data.std(),
                'n_observations': len(gene_data),
                'n_positive': (gene_data > 0).sum(),
                'n_negative': (gene_data < 0).sum(),
                'pct_increasing': (gene_data > 0).sum() / len(gene_data) * 100
            }

    return pd.DataFrame(stats_dict).T

def create_correlation_matrix(df, proteins, save_path):
    """Create correlation matrix of z-score changes across tissues."""
    # Create wide format: tissues x proteins
    tissue_protein_matrix = []
    tissues = []

    for tissue in df['Tissue_Compartment'].unique():
        tissue_data = df[df['Tissue_Compartment'] == tissue]
        row = []
        has_data = False

        for protein in proteins:
            protein_data = tissue_data[tissue_data['Canonical_Gene_Symbol'] == protein]['Zscore_Delta']
            if len(protein_data) > 0:
                row.append(protein_data.mean())
                has_data = True
            else:
                row.append(np.nan)

        if has_data:
            tissue_protein_matrix.append(row)
            tissues.append(tissue)

    # Create DataFrame
    matrix_df = pd.DataFrame(tissue_protein_matrix, columns=proteins, index=tissues)

    # Calculate correlation
    corr_matrix = matrix_df.corr(method='pearson')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    plt.title('Correlation Matrix: Versican-Hyaluronan Scaffold Proteins\n(Z-score Delta across Tissues)',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return corr_matrix, matrix_df

def calculate_inflammatory_scaffold_score(df):
    """Calculate composite inflammatory scaffold score per tissue."""
    core_proteins = ['VCAN', 'TNFAIP6', 'ITIH1', 'ITIH2', 'ITIH4']

    scores = []

    for tissue in df['Tissue_Compartment'].unique():
        tissue_data = df[df['Tissue_Compartment'] == tissue]

        # Calculate mean z-score delta for core proteins
        deltas = []
        protein_counts = 0

        for protein in core_proteins:
            protein_data = tissue_data[tissue_data['Canonical_Gene_Symbol'] == protein]['Zscore_Delta']
            if len(protein_data) > 0:
                deltas.append(protein_data.mean())
                protein_counts += 1

        if protein_counts >= 2:  # Require at least 2 proteins
            scaffold_score = np.mean(deltas)
            scores.append({
                'Tissue': tissue,
                'Scaffold_Score': scaffold_score,
                'N_Proteins': protein_counts,
                'Protein_Coverage': protein_counts / len(core_proteins) * 100
            })

    return pd.DataFrame(scores).sort_values('Scaffold_Score', ascending=False)

def analyze_temporal_cascade(df, save_path):
    """Analyze if VCAN increase precedes or follows TGF-beta pathway activation."""
    # Get VCAN data
    vcan_data = df[df['Canonical_Gene_Symbol'] == 'VCAN'].copy()

    # Get TGF-beta pathway data
    tgfb_genes = list(TGFB_PROTEINS.keys())
    tgfb_data = df[df['Canonical_Gene_Symbol'].isin(tgfb_genes)].copy()

    # For each tissue, compare mean z-score changes
    comparison = []

    for tissue in df['Tissue_Compartment'].unique():
        vcan_tissue = vcan_data[vcan_data['Tissue_Compartment'] == tissue]['Zscore_Delta']
        tgfb_tissue = tgfb_data[tgfb_data['Tissue_Compartment'] == tissue]['Zscore_Delta']

        if len(vcan_tissue) > 0 and len(tgfb_tissue) > 0:
            comparison.append({
                'Tissue': tissue,
                'VCAN_Delta': vcan_tissue.mean(),
                'TGFB_Delta': tgfb_tissue.mean(),
                'Difference': vcan_tissue.mean() - tgfb_tissue.mean()
            })

    comparison_df = pd.DataFrame(comparison)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    if len(comparison_df) > 0:
        axes[0].scatter(comparison_df['TGFB_Delta'], comparison_df['VCAN_Delta'],
                       s=100, alpha=0.6, edgecolors='black', linewidth=1)

        # Add diagonal line
        lims = [min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
                max(axes[0].get_xlim()[1], axes[0].get_ylim()[1])]
        axes[0].plot(lims, lims, 'k--', alpha=0.3, zorder=0)

        # Add tissue labels
        for idx, row in comparison_df.iterrows():
            axes[0].annotate(row['Tissue'], (row['TGFB_Delta'], row['VCAN_Delta']),
                           fontsize=8, alpha=0.7)

        axes[0].set_xlabel('TGF-β Pathway Z-score Delta', fontsize=12)
        axes[0].set_ylabel('VCAN Z-score Delta', fontsize=12)
        axes[0].set_title('VCAN vs TGF-β Pathway Changes', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Calculate correlation
        if len(comparison_df) > 2:
            corr, pval = stats.pearsonr(comparison_df['TGFB_Delta'], comparison_df['VCAN_Delta'])
            axes[0].text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.3e}',
                        transform=axes[0].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Bar plot of differences
    if len(comparison_df) > 0:
        comparison_sorted = comparison_df.sort_values('Difference')
        colors = ['red' if x > 0 else 'blue' for x in comparison_sorted['Difference']]
        axes[1].barh(range(len(comparison_sorted)), comparison_sorted['Difference'], color=colors, alpha=0.6)
        axes[1].set_yticks(range(len(comparison_sorted)))
        axes[1].set_yticklabels(comparison_sorted['Tissue'], fontsize=8)
        axes[1].set_xlabel('VCAN - TGF-β Pathway (Z-score Delta)', fontsize=12)
        axes[1].set_title('Temporal Pattern: VCAN vs TGF-β', fontsize=12, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[1].grid(True, alpha=0.3, axis='x')

        # Add legend
        axes[1].text(0.05, 0.95, 'Red: VCAN precedes\nBlue: TGF-β precedes',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df

def identify_vcan_negative_tissues(scaffold_scores):
    """Identify tissues where VCAN doesn't increase (protective phenotype)."""
    # Tissues with negative or minimal scaffold scores
    protective = scaffold_scores[scaffold_scores['Scaffold_Score'] < 0.1].copy()
    return protective

def create_comprehensive_heatmap(df, proteins, save_path):
    """Create comprehensive heatmap of all scaffold proteins across tissues."""
    # Create tissue x protein matrix
    tissues = sorted(df['Tissue_Compartment'].unique())
    matrix_data = []

    for tissue in tissues:
        tissue_data = df[df['Tissue_Compartment'] == tissue]
        row = []

        for protein in proteins:
            protein_data = tissue_data[tissue_data['Canonical_Gene_Symbol'] == protein]['Zscore_Delta']
            if len(protein_data) > 0:
                row.append(protein_data.mean())
            else:
                row.append(np.nan)

        matrix_data.append(row)

    # Create DataFrame
    heatmap_df = pd.DataFrame(matrix_data, columns=proteins, index=tissues)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(heatmap_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, cbar_kws={'label': 'Mean Z-score Delta'}, ax=ax)
    plt.title('Inflammatory Scaffold Proteins Across Tissues\n(Aging-Related Changes)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Protein', fontsize=12)
    plt.ylabel('Tissue/Compartment', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return heatmap_df

def analyze_ha_metabolism(df, save_path):
    """Analyze balance between HA synthesis (HAS) and degradation (HYAL)."""
    ha_genes = ['HAS1', 'HAS2', 'HAS3', 'HYAL1', 'HYAL2']
    ha_data = df[df['Canonical_Gene_Symbol'].isin(ha_genes)].copy()

    # Calculate synthesis vs degradation score per tissue
    balance_scores = []

    for tissue in df['Tissue_Compartment'].unique():
        tissue_data = ha_data[ha_data['Tissue_Compartment'] == tissue]

        # Synthesis (HAS)
        has_delta = tissue_data[tissue_data['Canonical_Gene_Symbol'].str.contains('HAS')]['Zscore_Delta'].mean()

        # Degradation (HYAL)
        hyal_delta = tissue_data[tissue_data['Canonical_Gene_Symbol'].str.contains('HYAL')]['Zscore_Delta'].mean()

        if not np.isnan(has_delta) or not np.isnan(hyal_delta):
            balance_scores.append({
                'Tissue': tissue,
                'HAS_Delta': has_delta if not np.isnan(has_delta) else 0,
                'HYAL_Delta': hyal_delta if not np.isnan(hyal_delta) else 0,
                'Balance_Score': (has_delta if not np.isnan(has_delta) else 0) -
                                (hyal_delta if not np.isnan(hyal_delta) else 0)
            })

    balance_df = pd.DataFrame(balance_scores).sort_values('Balance_Score', ascending=False)

    # Plot
    if len(balance_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        x = range(len(balance_df))
        width = 0.35

        ax.bar([i - width/2 for i in x], balance_df['HAS_Delta'], width,
               label='HA Synthesis (HAS)', alpha=0.8, color='blue')
        ax.bar([i + width/2 for i in x], balance_df['HYAL_Delta'], width,
               label='HA Degradation (HYAL)', alpha=0.8, color='red')

        ax.set_xlabel('Tissue/Compartment', fontsize=12)
        ax.set_ylabel('Mean Z-score Delta', fontsize=12)
        ax.set_title('Hyaluronan Metabolism: Synthesis vs Degradation Balance',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(balance_df['Tissue'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return balance_df

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("AGENT 12: VERSICAN-HYALURONAN INFLAMMATORY SCAFFOLD ANALYSIS")
print("="*80)

# 1. Extract scaffold protein data
print("\n1. Extracting inflammatory scaffold proteins...")
scaffold_genes = list(SCAFFOLD_PROTEINS.keys())
scaffold_df = extract_protein_data(df, scaffold_genes)

print(f"\nFound {len(scaffold_df)} observations for scaffold proteins")
print(f"Proteins detected: {sorted(scaffold_df['Canonical_Gene_Symbol'].unique())}")
print(f"Tissues analyzed: {len(scaffold_df['Tissue_Compartment'].unique())}")

# Calculate statistics
scaffold_stats = calculate_zscore_delta_stats(scaffold_df)
print("\nScaffold Protein Statistics:")
print(scaffold_stats.to_string())

# 2. Correlation analysis
print("\n2. Analyzing protein-protein correlations...")
detected_proteins = [p for p in scaffold_genes if p in scaffold_df['Canonical_Gene_Symbol'].unique()]
if len(detected_proteins) >= 2:
    corr_matrix, tissue_protein_matrix = create_correlation_matrix(
        scaffold_df, detected_proteins,
        '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_correlation_matrix.png'
    )
    print("\nCorrelation matrix created successfully")
    print(f"Strong positive correlations (r > 0.6):")
    corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if not np.isnan(corr_matrix.iloc[i, j]) and corr_matrix.iloc[i, j] > 0.6:
                corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    for p1, p2, r in sorted(corr_pairs, key=lambda x: x[2], reverse=True):
        print(f"  {p1} <-> {p2}: r = {r:.3f}")

# 3. Calculate inflammatory scaffold scores
print("\n3. Calculating inflammatory scaffold scores by tissue...")
scaffold_scores = calculate_inflammatory_scaffold_score(df)
print("\nTop 10 tissues by inflammatory scaffold score:")
print(scaffold_scores.head(10).to_string())

# Save scores
scaffold_scores.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_scaffold_scores.csv', index=False)

# 4. Identify protective (VCAN-negative) tissues
print("\n4. Identifying VCAN-negative (protective) tissues...")
protective_tissues = identify_vcan_negative_tissues(scaffold_scores)
print(f"\nProtective tissues (low scaffold score, n={len(protective_tissues)}):")
print(protective_tissues.to_string())

# 5. Temporal cascade analysis (VCAN vs TGF-beta)
print("\n5. Analyzing temporal cascade: VCAN vs TGF-β pathway...")
temporal_df = analyze_temporal_cascade(
    df,
    '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_temporal_cascade.png'
)
print("\nTemporal analysis completed")
if len(temporal_df) > 0:
    vcan_first = temporal_df[temporal_df['Difference'] > 0]
    tgfb_first = temporal_df[temporal_df['Difference'] < 0]
    print(f"  Tissues where VCAN precedes TGF-β: {len(vcan_first)}")
    print(f"  Tissues where TGF-β precedes VCAN: {len(tgfb_first)}")

# 6. Create comprehensive heatmap
print("\n6. Creating comprehensive tissue x protein heatmap...")
heatmap_df = create_comprehensive_heatmap(
    scaffold_df, detected_proteins,
    '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_tissue_protein_heatmap.png'
)

# 7. Hyaluronan metabolism analysis
print("\n7. Analyzing hyaluronan metabolism balance...")
ha_balance = analyze_ha_metabolism(
    df,
    '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_ha_metabolism.png'
)
if len(ha_balance) > 0:
    print("\nHA metabolism balance (top 5 tissues with highest synthesis):")
    print(ha_balance.head().to_string())

# 8. Additional analyses: fibrotic markers
print("\n8. Correlating with fibrotic markers...")
fibrotic_genes = list(FIBROTIC_MARKERS.keys())
fibrotic_df = extract_protein_data(df, fibrotic_genes)
print(f"Fibrotic markers detected: {sorted(fibrotic_df['Canonical_Gene_Symbol'].unique())}")

# 9. Inflammatory markers correlation
print("\n9. Checking inflammatory markers...")
inflammatory_genes = list(INFLAMMATORY_MARKERS.keys())
inflammatory_df = extract_protein_data(df, inflammatory_genes)
detected_inflammatory = sorted(inflammatory_df['Canonical_Gene_Symbol'].unique())
print(f"Inflammatory markers detected: {detected_inflammatory}")

if len(detected_inflammatory) > 0:
    inflammatory_stats = calculate_zscore_delta_stats(inflammatory_df)
    print("\nInflammatory Marker Statistics:")
    print(inflammatory_stats.to_string())

# 10. Create summary statistics table
print("\n10. Generating summary statistics...")

summary_data = {
    'Protein': [],
    'Category': [],
    'Mean_Zscore_Delta': [],
    'Median_Zscore_Delta': [],
    'N_Tissues': [],
    'Pct_Increasing': []
}

for category, proteins_dict in [
    ('Scaffold Core', SCAFFOLD_PROTEINS),
    ('TGF-β Pathway', TGFB_PROTEINS),
    ('Fibrotic Markers', FIBROTIC_MARKERS),
    ('Inflammatory Markers', INFLAMMATORY_MARKERS)
]:
    for gene in proteins_dict.keys():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene]['Zscore_Delta'].dropna()
        if len(gene_data) > 0:
            summary_data['Protein'].append(gene)
            summary_data['Category'].append(category)
            summary_data['Mean_Zscore_Delta'].append(gene_data.mean())
            summary_data['Median_Zscore_Delta'].append(gene_data.median())
            summary_data['N_Tissues'].append(len(gene_data))
            summary_data['Pct_Increasing'].append((gene_data > 0).sum() / len(gene_data) * 100)

summary_df = pd.DataFrame(summary_data).sort_values(['Category', 'Mean_Zscore_Delta'], ascending=[True, False])
summary_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_summary_statistics.csv', index=False)

print("\nSummary statistics saved")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nOutput files generated:")
print("  - agent_12_correlation_matrix.png")
print("  - agent_12_scaffold_scores.csv")
print("  - agent_12_temporal_cascade.png")
print("  - agent_12_tissue_protein_heatmap.png")
print("  - agent_12_ha_metabolism.png")
print("  - agent_12_summary_statistics.csv")
