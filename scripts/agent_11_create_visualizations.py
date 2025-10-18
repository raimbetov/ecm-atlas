#!/usr/bin/env python3
"""
Create visualizations for basement membrane collapse analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load analysis results."""
    base_dir = Path('/Users/Kravtsovd/projects/ecm-atlas/10_insights')

    protein_stats = pd.read_csv(base_dir / 'bm_protein_statistics.csv', index_col=0)
    therapeutic_targets = pd.read_csv(base_dir / 'bm_therapeutic_targets.csv', index_col=0)
    heatmap_data = pd.read_csv(base_dir / 'bm_tissue_heatmap.csv', index_col=0)
    breach_correlation = pd.read_csv(base_dir / 'bm_breach_correlation.csv')

    return protein_stats, therapeutic_targets, heatmap_data, breach_correlation

def create_figure1_protein_ranking(protein_stats):
    """Figure 1: BM proteins ranked by z-score delta."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Sort and plot
    data = protein_stats.sort_values('Mean_Delta').head(20)

    colors = []
    for family in data['BM_Family']:
        if family == 'Collagen IV':
            colors.append('#E74C3C')
        elif family == 'Laminins':
            colors.append('#3498DB')
        elif family == 'Nidogens':
            colors.append('#2ECC71')
        elif family == 'Perlecan':
            colors.append('#F39C12')
        elif family == 'Agrin':
            colors.append('#9B59B6')
        else:
            colors.append('#95A5A6')

    bars = ax.barh(range(len(data)), data['Mean_Delta'], color=colors)

    # Add error bars
    ax.errorbar(data['Mean_Delta'], range(len(data)),
                xerr=data['Std_Delta'], fmt='none',
                ecolor='black', capsize=3, alpha=0.5)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_xlabel('Mean Z-score Delta (Aging)', fontsize=12, fontweight='bold')
    ax.set_title('Basement Membrane Proteins: Loss During Aging\n(Top 20 Decreasing)',
                 fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Collagen IV'),
        Patch(facecolor='#3498DB', label='Laminins'),
        Patch(facecolor='#2ECC71', label='Nidogens'),
        Patch(facecolor='#F39C12', label='Perlecan'),
        Patch(facecolor='#9B59B6', label='Agrin'),
        Patch(facecolor='#95A5A6', label='Other BM')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('./visualizations/fig1_bm_protein_ranking.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig1_bm_protein_ranking.png")
    plt.close()

def create_figure2_heatmap(heatmap_data):
    """Figure 2: Cross-tissue heatmap of BM degradation."""
    # Filter to proteins with data in multiple tissues
    coverage = (~heatmap_data.isna()).sum(axis=1)
    filtered = heatmap_data[coverage >= 4]

    # Sort by mean delta
    mean_delta = filtered.mean(axis=1).sort_values()
    filtered = filtered.loc[mean_delta.index]

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(filtered, cmap='RdBu_r', center=0, vmin=-1.5, vmax=1.5,
                cbar_kws={'label': 'Z-score Delta (Aging)'},
                linewidths=0.5, linecolor='gray',
                ax=ax, annot=False, fmt='.2f')

    ax.set_title('Basement Membrane Degradation: Cross-Tissue Patterns',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tissue', fontsize=12, fontweight='bold')
    ax.set_ylabel('BM Protein', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./visualizations/fig2_bm_tissue_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig2_bm_tissue_heatmap.png")
    plt.close()

def create_figure3_col4a3_analysis(protein_stats):
    """Figure 3: COL4A3 vs other Collagen IV chains."""
    col4_proteins = ['COL4A1', 'COL4A2', 'COL4A3', 'COL4A4', 'COL4A5', 'COL4A6']
    col4_data = protein_stats.loc[protein_stats.index.isin(col4_proteins)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Mean deltas
    colors = ['#E74C3C' if prot == 'COL4A3' else '#95A5A6' for prot in col4_data.index]
    ax1.bar(range(len(col4_data)), col4_data['Mean_Delta'], color=colors)
    ax1.errorbar(range(len(col4_data)), col4_data['Mean_Delta'],
                 yerr=col4_data['Std_Delta'], fmt='none',
                 ecolor='black', capsize=5, alpha=0.5)
    ax1.set_xticks(range(len(col4_data)))
    ax1.set_xticklabels(col4_data.index, rotation=45, ha='right')
    ax1.set_ylabel('Mean Z-score Delta', fontsize=11, fontweight='bold')
    ax1.set_title('A) Collagen IV Chains: Aging Changes', fontsize=12, fontweight='bold')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)

    # Panel B: Tissue coverage
    ax2.bar(range(len(col4_data)), col4_data['N_Tissues'], color=colors)
    ax2.set_xticks(range(len(col4_data)))
    ax2.set_xticklabels(col4_data.index, rotation=45, ha='right')
    ax2.set_ylabel('Number of Tissues', fontsize=11, fontweight='bold')
    ax2.set_title('B) Tissue Coverage', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle('COL4A3 Analysis (red = COL4A3)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./visualizations/fig3_col4a3_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig3_col4a3_analysis.png")
    plt.close()

def create_figure4_breach_correlation(breach_correlation):
    """Figure 4: BM loss vs plasma infiltration."""
    if breach_correlation is None or len(breach_correlation) < 3:
        print("⊘ Skipping fig4: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(breach_correlation['BM_Mean_Delta'],
               breach_correlation['Plasma_Mean_Delta'],
               s=200, alpha=0.6, c='#3498DB', edgecolors='black', linewidth=1.5)

    # Add tissue labels
    for idx, row in breach_correlation.iterrows():
        ax.annotate(row['Tissue'],
                   (row['BM_Mean_Delta'], row['Plasma_Mean_Delta']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    # Add regression line
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(breach_correlation['BM_Mean_Delta'],
                                                   breach_correlation['Plasma_Mean_Delta'])
    x_line = np.linspace(breach_correlation['BM_Mean_Delta'].min(),
                         breach_correlation['BM_Mean_Delta'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
            label=f'r={r:.3f}, p={p:.3f}')

    ax.set_xlabel('BM Protein Mean Δz (Loss →)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Plasma Protein Mean Δz (← Infiltration)', fontsize=12, fontweight='bold')
    ax.set_title('Basement Membrane Breach Hypothesis\n(BM Loss vs Plasma Infiltration)',
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig('./visualizations/fig4_breach_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig4_breach_correlation.png")
    plt.close()

def create_figure5_therapeutic_targets(therapeutic_targets):
    """Figure 5: Top therapeutic targets."""
    top15 = therapeutic_targets.head(15).copy()
    top15 = top15.sort_values('Therapeutic_Score')

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = []
    for idx, row in top15.iterrows():
        if idx == 'COL4A3':
            colors.append('#E74C3C')  # Red for COL4A3
        elif row['BM_Family'] == 'Collagen IV':
            colors.append('#F39C12')
        elif row['BM_Family'] == 'Laminins':
            colors.append('#3498DB')
        else:
            colors.append('#95A5A6')

    bars = ax.barh(range(len(top15)), top15['Therapeutic_Score'], color=colors)

    # Highlight COL4A3
    for i, (idx, row) in enumerate(top15.iterrows()):
        if idx == 'COL4A3':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)

    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15.index, fontsize=10)
    ax.set_xlabel('Therapeutic Score (Composite)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Basement Membrane Therapeutic Targets\n(Red = COL4A3, Orange = Other Col IV, Blue = Laminins)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add score values
    for i, score in enumerate(top15['Therapeutic_Score']):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('./visualizations/fig5_therapeutic_targets.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig5_therapeutic_targets.png")
    plt.close()

def create_figure6_family_comparison(protein_stats):
    """Figure 6: Comparison across BM protein families."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Box plot by family
    family_data = []
    family_labels = []
    for family in protein_stats['BM_Family'].unique():
        family_proteins = protein_stats[protein_stats['BM_Family'] == family]
        family_data.append(family_proteins['Mean_Delta'].values)
        family_labels.append(f"{family}\n(n={len(family_proteins)})")

    bp = ax1.boxplot(family_data, labels=family_labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))

    ax1.set_ylabel('Mean Z-score Delta', fontsize=11, fontweight='bold')
    ax1.set_title('A) BM Protein Families: Aging Distribution', fontsize=12, fontweight='bold')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Panel B: Mean by family
    family_means = protein_stats.groupby('BM_Family')['Mean_Delta'].mean().sort_values()
    colors_family = ['#E74C3C' if 'Collagen IV' in idx else '#95A5A6' for idx in family_means.index]

    ax2.barh(range(len(family_means)), family_means.values, color=colors_family)
    ax2.set_yticks(range(len(family_means)))
    ax2.set_yticklabels(family_means.index, fontsize=10)
    ax2.set_xlabel('Mean Z-score Delta', fontsize=11, fontweight='bold')
    ax2.set_title('B) Family-Level Mean Changes', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('./visualizations/fig6_family_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: fig6_family_comparison.png")
    plt.close()

def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("GENERATING BASEMENT MEMBRANE VISUALIZATIONS")
    print("="*70 + "\n")

    # Load data
    protein_stats, therapeutic_targets, heatmap_data, breach_correlation = load_results()

    # Create figures
    create_figure1_protein_ranking(protein_stats)
    create_figure2_heatmap(heatmap_data)
    create_figure3_col4a3_analysis(protein_stats)
    create_figure4_breach_correlation(breach_correlation)
    create_figure5_therapeutic_targets(therapeutic_targets)
    create_figure6_family_comparison(protein_stats)

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
