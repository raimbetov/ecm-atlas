#!/usr/bin/env python3
"""
Visualization Script for Cross-Tissue ECM Aging Analysis
Agent 1 - Q1.1.3

Creates publication-quality figures for the analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Paths
base_path = "/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/"

# ============================================================================
# FIGURE 1: TISSUE CORRELATION HEATMAP
# ============================================================================
print("\n[1/6] Creating tissue correlation heatmap...")

tissue_corr = pd.read_csv(base_path + 'tissue_correlation_matrix.csv', index_col=0)
tissue_clusters = pd.read_csv(base_path + 'tissue_clusters.csv')

# Create clustered heatmap
fig, ax = plt.subplots(figsize=(14, 12))

# Hierarchical clustering
corr_distance = 1 - tissue_corr.abs()
linkage_matrix = hierarchy.linkage(squareform(corr_distance), method='average')

# Plot clustermap
g = sns.clustermap(
    tissue_corr,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation (r)'},
    figsize=(14, 12),
    row_linkage=linkage_matrix,
    col_linkage=linkage_matrix,
    dendrogram_ratio=0.15,
    cbar_pos=(0.02, 0.83, 0.03, 0.15)
)

g.ax_heatmap.set_xlabel('Tissue', fontsize=12, fontweight='bold')
g.ax_heatmap.set_ylabel('Tissue', fontsize=12, fontweight='bold')
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

g.fig.suptitle('Cross-Tissue ECM Aging Signature Correlations\nHierarchical Clustering by Aging Pattern Similarity',
               fontsize=14, fontweight='bold', y=0.98)

plt.savefig(base_path + 'fig1_tissue_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: fig1_tissue_correlation_heatmap.png")
plt.close()

# ============================================================================
# FIGURE 2: PCA BIPLOT
# ============================================================================
print("\n[2/6] Creating PCA biplot...")

pca_scores = pd.read_csv(base_path + 'pca_scores.csv', index_col=0)
pca_loadings = pd.read_csv(base_path + 'pca_loadings.csv', index_col=0)
pca_variance = pd.read_csv(base_path + 'pca_variance.csv')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PC1 vs PC2
ax = axes[0]
scatter = ax.scatter(
    pca_scores['PC1'],
    pca_scores['PC2'],
    s=100,
    alpha=0.7,
    edgecolors='black',
    linewidths=1.5,
    c=range(len(pca_scores)),
    cmap='tab20'
)

# Add tissue labels
for idx, tissue in enumerate(pca_scores.index):
    ax.annotate(
        tissue,
        (pca_scores.loc[tissue, 'PC1'], pca_scores.loc[tissue, 'PC2']),
        fontsize=8,
        ha='right',
        va='bottom',
        alpha=0.8
    )

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel(f"PC1 ({pca_variance.loc[0, 'Variance_Explained']*100:.1f}% variance)", fontweight='bold')
ax.set_ylabel(f"PC2 ({pca_variance.loc[1, 'Variance_Explained']*100:.1f}% variance)", fontweight='bold')
ax.set_title('PCA: Tissue Clustering by ECM Aging Patterns', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# PC1 vs PC3
ax = axes[1]
scatter = ax.scatter(
    pca_scores['PC1'],
    pca_scores['PC3'],
    s=100,
    alpha=0.7,
    edgecolors='black',
    linewidths=1.5,
    c=range(len(pca_scores)),
    cmap='tab20'
)

for idx, tissue in enumerate(pca_scores.index):
    ax.annotate(
        tissue,
        (pca_scores.loc[tissue, 'PC1'], pca_scores.loc[tissue, 'PC3']),
        fontsize=8,
        ha='right',
        va='bottom',
        alpha=0.8
    )

ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel(f"PC1 ({pca_variance.loc[0, 'Variance_Explained']*100:.1f}% variance)", fontweight='bold')
ax.set_ylabel(f"PC3 ({pca_variance.loc[2, 'Variance_Explained']*100:.1f}% variance)", fontweight='bold')
ax.set_title('PCA: PC1 vs PC3', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Principal Component Analysis of Cross-Tissue ECM Aging', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(base_path + 'fig2_pca_biplot.png', dpi=300, bbox_inches='tight')
print("  Saved: fig2_pca_biplot.png")
plt.close()

# ============================================================================
# FIGURE 3: PCA VARIANCE EXPLAINED (SCREE PLOT)
# ============================================================================
print("\n[3/6] Creating scree plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Variance explained per component
ax = axes[0]
n_components = min(15, len(pca_variance))
components = range(1, n_components + 1)
variance = pca_variance['Variance_Explained'].iloc[:n_components] * 100

ax.bar(components, variance, color='steelblue', alpha=0.7, edgecolor='black')
ax.plot(components, variance, 'o-', color='darkred', linewidth=2, markersize=6)
ax.set_xlabel('Principal Component', fontweight='bold')
ax.set_ylabel('Variance Explained (%)', fontweight='bold')
ax.set_title('Scree Plot: Variance per Component', fontsize=12, fontweight='bold')
ax.set_xticks(components)
ax.grid(axis='y', alpha=0.3)

# Add percentage labels
for i, (comp, var) in enumerate(zip(components, variance)):
    if i < 10:  # Label first 10
        ax.text(comp, var + 1, f'{var:.1f}%', ha='center', fontsize=8)

# Cumulative variance
ax = axes[1]
cumulative = pca_variance['Cumulative_Variance'].iloc[:n_components] * 100

ax.fill_between(components, cumulative, alpha=0.3, color='steelblue')
ax.plot(components, cumulative, 'o-', color='darkblue', linewidth=2, markersize=6)
ax.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='80% threshold')
ax.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, label='90% threshold')
ax.set_xlabel('Number of Components', fontweight='bold')
ax.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold')
ax.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax.set_xticks(components)
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('PCA Variance Decomposition: Universal vs. Tissue-Specific Components',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(base_path + 'fig3_pca_variance.png', dpi=300, bbox_inches='tight')
print("  Saved: fig3_pca_variance.png")
plt.close()

# ============================================================================
# FIGURE 4: PC1 LOADINGS (UNIVERSAL AGING SIGNATURE)
# ============================================================================
print("\n[4/6] Creating PC1 loadings plot...")

pc1_loadings = pca_loadings['PC1'].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 8))

# Top positive and negative loadings
top_n = 20
top_positive = pc1_loadings.head(top_n)
top_negative = pc1_loadings.tail(top_n)
top_loadings = pd.concat([top_positive, top_negative]).sort_values()

colors = ['darkred' if x > 0 else 'darkblue' for x in top_loadings.values]
ax.barh(range(len(top_loadings)), top_loadings.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_loadings)))
ax.set_yticklabels(top_loadings.index, fontsize=9)
ax.set_xlabel('PC1 Loading', fontweight='bold')
ax.set_title(f'PC1 Loadings: Top {top_n} Proteins Driving Universal Aging Component\n(Red = Upregulated with Age, Blue = Downregulated with Age)',
             fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(base_path + 'fig4_pc1_loadings.png', dpi=300, bbox_inches='tight')
print("  Saved: fig4_pc1_loadings.png")
plt.close()

# ============================================================================
# FIGURE 5: TISSUE-PROTEIN HEATMAP (TOP PROTEINS)
# ============================================================================
print("\n[5/6] Creating tissue-protein heatmap...")

tissue_protein_matrix = pd.read_csv(base_path + 'tissue_protein_matrix.csv', index_col=0)

# Select top proteins by variance
protein_variance = tissue_protein_matrix.var(axis=0).sort_values(ascending=False)
top_var_proteins = protein_variance.head(30).index

fig, ax = plt.subplots(figsize=(16, 10))

# Create heatmap
g = sns.clustermap(
    tissue_protein_matrix[top_var_proteins],
    cmap='RdBu_r',
    center=0,
    vmin=-3,
    vmax=3,
    linewidths=0.5,
    cbar_kws={'label': 'Z-score Delta (Aging Effect)'},
    figsize=(16, 10),
    row_cluster=True,
    col_cluster=True,
    dendrogram_ratio=0.1,
    yticklabels=True,
    xticklabels=True,
    cbar_pos=(0.02, 0.85, 0.02, 0.12)
)

g.ax_heatmap.set_xlabel('Protein', fontsize=12, fontweight='bold')
g.ax_heatmap.set_ylabel('Tissue', fontsize=12, fontweight='bold')
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='center', fontsize=9)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

g.fig.suptitle('Cross-Tissue ECM Protein Aging Signatures\nTop 30 Proteins by Variance',
               fontsize=14, fontweight='bold', y=0.98)

plt.savefig(base_path + 'fig5_tissue_protein_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: fig5_tissue_protein_heatmap.png")
plt.close()

# ============================================================================
# FIGURE 6: VARIANCE PARTITION (UNIVERSAL vs TISSUE-SPECIFIC)
# ============================================================================
print("\n[6/6] Creating variance partition plot...")

variance_partition = pd.read_csv(base_path + 'variance_partition.csv')

# Sort by absolute grand mean
variance_partition['Abs_Mean'] = variance_partition['Grand_Mean_Zscore'].abs()
variance_partition_sorted = variance_partition.sort_values('Abs_Mean', ascending=False).head(30)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Grand mean vs variance
ax = axes[0]
scatter = ax.scatter(
    variance_partition['Grand_Mean_Zscore'],
    variance_partition['Total_Variance'],
    s=variance_partition['N_Tissues'] * 10,
    alpha=0.6,
    c=variance_partition['N_Tissues'],
    cmap='viridis',
    edgecolors='black',
    linewidths=0.5
)

# Annotate top proteins
for _, row in variance_partition_sorted.head(15).iterrows():
    ax.annotate(
        row['Protein'],
        (row['Grand_Mean_Zscore'], row['Total_Variance']),
        fontsize=7,
        alpha=0.8
    )

ax.set_xlabel('Grand Mean Z-score Delta (Universal Component)', fontweight='bold')
ax.set_ylabel('Total Variance Across Tissues', fontweight='bold')
ax.set_title('Universal vs. Tissue-Specific Aging Patterns\n(Size = # Tissues)', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Tissues', fontweight='bold')

# Plot 2: Consistency across tissues
ax = axes[1]
# Calculate coefficient of variation (normalized variance)
variance_partition['CV'] = np.sqrt(variance_partition['Total_Variance']) / (variance_partition['Grand_Mean_Zscore'].abs() + 0.01)

# Universal proteins: high absolute mean, low CV
universal_proteins = variance_partition[
    (variance_partition['Abs_Mean'] > variance_partition['Abs_Mean'].quantile(0.75)) &
    (variance_partition['CV'] < variance_partition['CV'].quantile(0.5))
]

# Tissue-specific proteins: low absolute mean, high variance
tissue_specific_proteins = variance_partition[
    (variance_partition['Abs_Mean'] < variance_partition['Abs_Mean'].quantile(0.5)) &
    (variance_partition['Total_Variance'] > variance_partition['Total_Variance'].quantile(0.75))
]

ax.scatter(
    variance_partition['Abs_Mean'],
    variance_partition['Total_Variance'],
    s=50,
    alpha=0.4,
    color='gray',
    label='All proteins'
)

ax.scatter(
    universal_proteins['Abs_Mean'],
    universal_proteins['Total_Variance'],
    s=100,
    alpha=0.8,
    color='darkred',
    edgecolors='black',
    linewidths=1,
    label=f'Universal candidates (n={len(universal_proteins)})'
)

ax.scatter(
    tissue_specific_proteins['Abs_Mean'],
    tissue_specific_proteins['Total_Variance'],
    s=100,
    alpha=0.8,
    color='darkblue',
    edgecolors='black',
    linewidths=1,
    label=f'Tissue-specific (n={len(tissue_specific_proteins)})'
)

# Annotate universal candidates
for _, row in universal_proteins.iterrows():
    ax.annotate(
        row['Protein'],
        (row['Abs_Mean'], row['Total_Variance']),
        fontsize=8,
        fontweight='bold',
        color='darkred'
    )

ax.set_xlabel('|Mean Z-score Delta| (Universal Signal Strength)', fontweight='bold')
ax.set_ylabel('Variance Across Tissues (Tissue-Specificity)', fontweight='bold')
ax.set_title('Classification: Universal vs. Tissue-Specific Proteins', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(base_path + 'fig6_variance_partition.png', dpi=300, bbox_inches='tight')
print("  Saved: fig6_variance_partition.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE")
print("="*80)
