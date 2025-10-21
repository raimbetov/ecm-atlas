#!/usr/bin/env python3
"""
Visualization script for universal aging markers
Creates heatmaps showing cross-tissue consistency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
DATA_FILE = "../../10_insights/agent_01_universal_markers_data.csv"
MERGED_CSV = "../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = "./visualizations/"

def create_heatmap_top_candidates(merged_df, results_df, top_n=20):
    """Create heatmap for top N universal candidates across all tissues"""

    # Get top candidates
    top_genes = results_df.nlargest(top_n, 'Universality_Score')['Gene_Symbol'].tolist()

    # Filter merged data to these genes
    data = merged_df[merged_df['Gene_Symbol'].isin(top_genes)].copy()

    # Create pivot table: genes x tissues
    pivot = data.pivot_table(
        values='Zscore_Delta',
        index='Gene_Symbol',
        columns='Tissue_Compartment',
        aggfunc='mean'
    )

    # Reorder rows by universality score
    gene_order = top_genes
    pivot = pivot.reindex([g for g in gene_order if g in pivot.index])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0,
        vmin=-2,
        vmax=2,
        cbar_kws={'label': 'Z-score Delta (Old - Young)'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_title(f'Top {top_n} Universal ECM Aging Markers: Cross-Tissue Z-score Changes',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tissue Compartment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene Symbol', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR + "heatmap_top20_universal_markers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved: {output_path}")

    plt.close()

def create_consistency_plot(results_df):
    """Create scatter plot: tissue breadth vs consistency"""

    # Filter to proteins in ≥3 tissues
    filtered = results_df[results_df['N_Tissues'] >= 3].copy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with color by universality score
    scatter = ax.scatter(
        filtered['N_Tissues'],
        filtered['Direction_Consistency'] * 100,
        c=filtered['Universality_Score'],
        s=filtered['Abs_Mean_Zscore_Delta'] * 50,  # Size by effect size
        alpha=0.6,
        cmap='viridis'
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Universality Score', fontsize=12, fontweight='bold')

    # Reference lines
    ax.axhline(70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='70% consistency threshold')
    ax.axvline(3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='3 tissue threshold')

    # Labels
    ax.set_xlabel('Number of Tissues', fontsize=12, fontweight='bold')
    ax.set_ylabel('Directional Consistency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Universal Marker Criteria: Tissue Breadth vs Consistency\n(size = effect size)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR + "scatter_tissue_consistency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Scatter plot saved: {output_path}")

    plt.close()

def create_direction_distribution(results_df):
    """Create bar chart showing up vs down regulation"""

    # Filter universal candidates
    candidates = results_df[
        (results_df['N_Tissues'] >= 3) &
        (results_df['Direction_Consistency'] >= 0.7)
    ].copy()

    # Count by matrisome category and direction
    category_counts = candidates.groupby(['Matrisome_Category', 'Predominant_Direction']).size().unstack(fill_value=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    category_counts.plot(kind='barh', stacked=False, ax=ax, color=['#3498db', '#e74c3c'])

    ax.set_xlabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax.set_ylabel('Matrisome Category', fontsize=12, fontweight='bold')
    ax.set_title('Universal Markers by Category and Direction\n(≥3 tissues, ≥70% consistency)',
                fontsize=14, fontweight='bold')
    ax.legend(['Downregulated', 'Upregulated'], loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR + "barplot_category_direction.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Bar plot saved: {output_path}")

    plt.close()

def main():
    """Main visualization workflow"""

    print("\n" + "="*80)
    print("VISUALIZING UNIVERSAL MARKERS")
    print("="*80 + "\n")

    # Load data
    print("Loading data...")
    results_df = pd.read_csv(DATA_FILE)
    merged_df = pd.read_csv(MERGED_CSV)

    # Filter merged to valid aging data
    merged_df = merged_df.dropna(subset=['Zscore_Delta'])

    print(f"Loaded {len(results_df)} proteins from analysis")
    print(f"Loaded {len(merged_df)} measurements from merged dataset\n")

    # Create visualizations
    print("Creating visualizations...")
    create_heatmap_top_candidates(merged_df, results_df, top_n=20)
    create_consistency_plot(results_df)
    create_direction_distribution(results_df)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("- heatmap_top20_universal_markers.png")
    print("- scatter_tissue_consistency.png")
    print("- barplot_category_direction.png\n")

if __name__ == "__main__":
    main()
