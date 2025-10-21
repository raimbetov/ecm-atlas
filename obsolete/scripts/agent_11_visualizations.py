#!/usr/bin/env python3
"""
Agent 11: Cross-Species Visualization Script

Generate publication-quality figures:
1. Scatterplot: Human vs Mouse Zscore_Delta
2. Venn diagram: Species protein overlap
3. Conservation heatmap
4. Lifespan correlation plot

Author: Claude Code
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn2
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_data():
    """Load dataset"""
    df_path = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
    df = pd.read_csv(df_path)
    return df

def plot_human_vs_mouse_correlation(df, output_path):
    """Scatterplot: Human Zscore_Delta vs Mouse Zscore_Delta for shared genes"""
    print("\nGenerating Human vs Mouse correlation plot...")

    # Get valid data
    valid_df = df.dropna(subset=['Zscore_Delta'])

    # Get shared genes
    gene_species = valid_df.groupby('Canonical_Gene_Symbol')['Species'].apply(set).to_dict()
    shared_genes = [gene for gene, sp_set in gene_species.items()
                    if 'Homo sapiens' in sp_set and 'Mus musculus' in sp_set]

    if len(shared_genes) == 0:
        print("  No shared genes between human and mouse!")
        return

    # Extract z-scores
    human_data = valid_df[valid_df['Species'] == 'Homo sapiens'].groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean()
    mouse_data = valid_df[valid_df['Species'] == 'Mus musculus'].groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean()

    # Get shared data points
    human_z = []
    mouse_z = []
    gene_labels = []

    for gene in shared_genes:
        if gene in human_data.index and gene in mouse_data.index:
            human_z.append(human_data.loc[gene])
            mouse_z.append(mouse_data.loc[gene])
            gene_labels.append(gene)

    if len(human_z) < 2:
        print("  Insufficient data for correlation plot")
        return

    # Calculate correlation
    r, p = pearsonr(human_z, mouse_z)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(human_z, mouse_z, s=200, alpha=0.7, c='steelblue', edgecolors='black', linewidth=1.5)

    # Add gene labels
    for i, gene in enumerate(gene_labels):
        ax.annotate(gene, (human_z[i], mouse_z[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold')

    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add diagonal line (perfect conservation)
    lims = [min(human_z + mouse_z), max(human_z + mouse_z)]
    ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect conservation (R=1.0)')

    # Labels and title
    ax.set_xlabel('Human Zscore_Delta (Aging Effect)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mouse Zscore_Delta (Aging Effect)', fontsize=14, fontweight='bold')
    ax.set_title(f'Cross-Species Conservation of ECM Aging\nHomo sapiens vs Mus musculus (N={len(gene_labels)} genes)\nPearson R = {r:.3f}, p = {p:.3f}',
                fontsize=16, fontweight='bold', pad=20)

    # Add correlation annotation
    textstr = f'Conservation Score: R = {r:.3f}\n'
    textstr += f'P-value: {p:.3f}\n'
    textstr += f'Interpretation: {"Conserved" if r > 0.3 and p < 0.05 else "Divergent" if r < -0.3 and p < 0.05 else "Neutral"}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_species_venn(df, output_path):
    """Venn diagram showing protein overlap across species"""
    print("\nGenerating species Venn diagram...")

    valid_df = df.dropna(subset=['Zscore_Delta'])

    # Get unique genes per species
    human_genes = set(valid_df[valid_df['Species'] == 'Homo sapiens']['Canonical_Gene_Symbol'].unique())
    mouse_genes = set(valid_df[valid_df['Species'] == 'Mus musculus']['Canonical_Gene_Symbol'].unique())
    cow_genes = set(valid_df[valid_df['Species'] == 'Bos taurus']['Canonical_Gene_Symbol'].unique())

    print(f"  Human genes: {len(human_genes)}")
    print(f"  Mouse genes: {len(mouse_genes)}")
    print(f"  Cow genes: {len(cow_genes)}")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create Venn diagram
    venn = venn3([human_genes, mouse_genes, cow_genes],
                 set_labels=('Homo sapiens\n(Human)', 'Mus musculus\n(Mouse)', 'Bos taurus\n(Cow)'),
                 ax=ax)

    # Color the circles
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    if venn.patches:
        for patch in venn.patches:
            if patch:
                patch.set_alpha(0.6)
                patch.set_edgecolor('black')
                patch.set_linewidth(2)

    # Title
    ax.set_title('ECM Protein Gene Overlap Across Species\nECM-Atlas Dataset',
                fontsize=18, fontweight='bold', pad=20)

    # Add annotation
    total_unique = len(human_genes | mouse_genes | cow_genes)
    textstr = f'Total unique genes: {total_unique}\n'
    textstr += f'Species-specific genes: {total_unique - len(human_genes & mouse_genes) - len(human_genes & cow_genes) - len(mouse_genes & cow_genes)}\n'
    textstr += f'Ortholog overlap: {(len(human_genes & mouse_genes) + len(human_genes & cow_genes) + len(mouse_genes & cow_genes)) / total_unique * 100:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.5, -0.15, textstr, transform=ax.transAxes, fontsize=14,
           horizontalalignment='center', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_lifespan_correlation(df, output_path):
    """Lifespan vs ECM aging rate correlation"""
    print("\nGenerating lifespan correlation plot...")

    SPECIES_LIFESPAN = {
        'Homo sapiens': 122,
        'Mus musculus': 4,
        'Bos taurus': 22,
    }

    valid_df = df.dropna(subset=['Zscore_Delta'])

    # Calculate mean absolute z-score delta per species
    species_aging_rates = {}
    for species in valid_df['Species'].unique():
        species_data = valid_df[valid_df['Species'] == species]
        mean_abs_delta = species_data['Zscore_Delta'].abs().mean()
        species_aging_rates[species] = mean_abs_delta

    # Prepare data
    lifespans = []
    aging_rates = []
    labels = []

    for species in species_aging_rates.keys():
        if species in SPECIES_LIFESPAN:
            lifespans.append(SPECIES_LIFESPAN[species])
            aging_rates.append(species_aging_rates[species])
            labels.append(species.split()[1])  # Genus name

    # Calculate correlation
    r, p = pearsonr(lifespans, aging_rates)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    for i, (life, rate, label) in enumerate(zip(lifespans, aging_rates, labels)):
        ax.scatter(life, rate, s=500, alpha=0.7, c=colors[i], edgecolors='black', linewidth=2)
        ax.annotate(label, (life, rate), fontsize=14, fontweight='bold',
                   ha='center', va='center')

    # Add trend line
    z = np.polyfit(lifespans, aging_rates, 1)
    p_trend = np.poly1d(z)
    x_trend = np.linspace(min(lifespans), max(lifespans), 100)
    ax.plot(x_trend, p_trend(x_trend), "r--", alpha=0.5, linewidth=2, label=f'Trend line (R={r:.3f})')

    # Labels and title
    ax.set_xlabel('Maximum Lifespan (years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('ECM Aging Rate (Mean |Zscore_Delta|)', fontsize=14, fontweight='bold')
    ax.set_title('Lifespan vs ECM Aging Rate Hypothesis\nDo long-lived species age slower?',
                fontsize=16, fontweight='bold', pad=20)

    # Add correlation annotation
    textstr = f'Pearson R = {r:.3f}\n'
    textstr += f'P-value: {p:.3f}\n'
    textstr += f'Result: {"SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"}\n'
    textstr += f'Conclusion: {"Lifespan predicts ECM aging" if p < 0.05 else "Lifespan does NOT predict ECM aging"}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=props)

    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_conserved_protein_heatmap(df, output_path):
    """Heatmap of conserved protein aging across species"""
    print("\nGenerating conserved protein heatmap...")

    valid_df = df.dropna(subset=['Zscore_Delta'])

    # Get genes in multiple species
    gene_species = valid_df.groupby('Canonical_Gene_Symbol')['Species'].apply(set).to_dict()
    multi_species_genes = [gene for gene, sp_set in gene_species.items() if len(sp_set) > 1]

    if len(multi_species_genes) == 0:
        print("  No multi-species genes found!")
        return

    # Create matrix
    species_list = sorted(valid_df['Species'].unique())
    matrix_data = []
    gene_labels = []

    for gene in multi_species_genes:
        row = []
        for species in species_list:
            gene_data = valid_df[(valid_df['Canonical_Gene_Symbol'] == gene) &
                                 (valid_df['Species'] == species)]
            if len(gene_data) > 0:
                row.append(gene_data['Zscore_Delta'].mean())
            else:
                row.append(np.nan)
        matrix_data.append(row)
        gene_labels.append(gene)

    matrix_df = pd.DataFrame(matrix_data, columns=species_list, index=gene_labels)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(multi_species_genes) * 0.5)))

    # Heatmap
    sns.heatmap(matrix_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               cbar_kws={'label': 'Zscore_Delta (Aging Effect)'},
               linewidths=1, linecolor='black',
               vmin=-2, vmax=2, ax=ax)

    ax.set_title('Cross-Species Conservation Heatmap\nOrthologous Protein Aging Patterns',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Species', fontsize=14, fontweight='bold')
    ax.set_ylabel('Gene Symbol', fontsize=14, fontweight='bold')

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def main():
    """Generate all visualizations"""
    print("="*80)
    print("AGENT 11: CROSS-SPECIES VISUALIZATION GENERATOR")
    print("="*80)

    df = load_data()
    output_dir = "./visualizations/"

    # Generate figures
    plot_human_vs_mouse_correlation(df, f"{output_dir}agent_11_fig1_human_mouse_correlation.png")
    plot_species_venn(df, f"{output_dir}agent_11_fig2_species_venn.png")
    plot_lifespan_correlation(df, f"{output_dir}agent_11_fig3_lifespan_correlation.png")
    plot_conserved_protein_heatmap(df, f"{output_dir}agent_11_fig4_conservation_heatmap.png")

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
