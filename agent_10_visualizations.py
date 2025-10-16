#!/usr/bin/env python3
"""
Create visualizations for weak signal analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load results
df_proteins = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/all_protein_statistics.csv')
weak_signals = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_proteins.csv')
df_full = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
df_full = df_full[df_full['Zscore_Delta'].notna()]

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. FOREST PLOT - Meta-analysis visualization
print("Creating forest plots...")
fig, axes = plt.subplots(4, 1, figsize=(12, 16))

top_proteins = weak_signals.head(12)['Gene_Symbol'].values

for idx, gene in enumerate(top_proteins[:4]):
    ax = axes[idx]
    gene_data = df_full[df_full['Canonical_Gene_Symbol'] == gene].copy()

    # Group by study
    study_means = gene_data.groupby('Study_ID')['Zscore_Delta'].agg(['mean', 'std', 'count']).reset_index()
    study_means['se'] = study_means['std'] / np.sqrt(study_means['count'])
    study_means = study_means.sort_values('mean')

    # Plot each study
    y_pos = np.arange(len(study_means))
    ax.errorbar(study_means['mean'], y_pos, xerr=study_means['se']*1.96,
                fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.7)

    # Add overall meta-analytic mean
    overall_mean = gene_data['Zscore_Delta'].mean()
    overall_se = gene_data['Zscore_Delta'].std() / np.sqrt(len(gene_data))
    ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2, label=f'Meta-mean: {overall_mean:.3f}')
    ax.axvspan(overall_mean - 1.96*overall_se, overall_mean + 1.96*overall_se,
               alpha=0.2, color='red')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(study_means['Study_ID'])
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Z-score Delta (95% CI)', fontsize=11)
    ax.set_title(f'{gene} - Forest Plot (n={len(gene_data)} measurements)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_forest_plots.png', dpi=300, bbox_inches='tight')
print("Saved: weak_signal_forest_plots.png")
plt.close()

# 2. EFFECT SIZE vs CONSISTENCY SCATTER
print("Creating effect size vs consistency plot...")
fig, ax = plt.subplots(figsize=(12, 8))

# Color by number of studies
scatter = ax.scatter(df_proteins['Abs_Mean_Zscore_Delta'],
                    df_proteins['Direction_Consistency'],
                    c=df_proteins['N_Studies'],
                    s=df_proteins['N_Measurements']*10,
                    alpha=0.6,
                    cmap='viridis',
                    edgecolors='black',
                    linewidths=0.5)

# Highlight weak signals
weak_mask = df_proteins['Gene_Symbol'].isin(weak_signals['Gene_Symbol'])
ax.scatter(df_proteins[weak_mask]['Abs_Mean_Zscore_Delta'],
          df_proteins[weak_mask]['Direction_Consistency'],
          s=df_proteins[weak_mask]['N_Measurements']*10,
          facecolors='none',
          edgecolors='red',
          linewidths=3,
          label='Weak Signals')

# Draw weak signal zone
ax.axvspan(0.3, 1.0, ymin=0.65, ymax=1.0, alpha=0.1, color='red', label='Weak Signal Zone')

# Annotations for top weak signals
for idx, row in weak_signals.head(8).iterrows():
    ax.annotate(row['Gene_Symbol'],
                xy=(row['Abs_Mean_Zscore_Delta'], row['Direction_Consistency']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Studies', fontsize=11)

ax.set_xlabel('Absolute Mean Z-score Delta', fontsize=12)
ax.set_ylabel('Direction Consistency', fontsize=12)
ax.set_title('Weak Signal Landscape: Effect Size vs Consistency\n(size = number of measurements)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_landscape.png', dpi=300, bbox_inches='tight')
print("Saved: weak_signal_landscape.png")
plt.close()

# 3. P-VALUE DISTRIBUTION & POWER ANALYSIS
print("Creating p-value and power analysis plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3a. Combined P-value distribution
ax = axes[0, 0]
p_values = df_proteins['Combined_P_Value'].dropna()
ax.hist(p_values, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
ax.set_xlabel('Combined P-value (Fisher\'s Method)', fontsize=11)
ax.set_ylabel('Number of Proteins', fontsize=11)
ax.set_title('Distribution of Meta-Analytic P-values', fontsize=12, fontweight='bold')
ax.legend()
ax.set_yscale('log')

# 3b. Statistical power curves
ax = axes[0, 1]
n_samples = np.arange(2, 20)
effect_sizes = [0.3, 0.5, 0.8, 1.0]
alpha = 0.05

for effect in effect_sizes:
    # Power = 1 - β, approximate for one-sample t-test
    powers = []
    for n in n_samples:
        ncp = effect * np.sqrt(n)  # Non-centrality parameter
        critical_t = stats.t.ppf(1 - alpha/2, n-1)
        power = 1 - stats.nct.cdf(critical_t, n-1, ncp) + stats.nct.cdf(-critical_t, n-1, ncp)
        powers.append(power)

    ax.plot(n_samples, powers, marker='o', linewidth=2, label=f'δ={effect}')

ax.axhline(0.80, color='red', linestyle='--', linewidth=1, label='80% Power')
ax.set_xlabel('Number of Measurements', fontsize=11)
ax.set_ylabel('Statistical Power', fontsize=11)
ax.set_title('Power Analysis: Sample Size Requirements', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3c. Variance vs Effect Size
ax = axes[1, 0]
scatter = ax.scatter(df_proteins['Abs_Mean_Zscore_Delta'],
                    df_proteins['Std_Zscore_Delta'],
                    c=df_proteins['N_Measurements'],
                    s=50,
                    alpha=0.6,
                    cmap='plasma',
                    edgecolors='black',
                    linewidths=0.5)

# Weak signal zone: low variance, moderate effect
ax.axhspan(0, 0.4, xmin=0.15, xmax=0.5, alpha=0.1, color='red')
ax.axvline(0.3, color='red', linestyle='--', alpha=0.5)
ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
ax.axhline(0.4, color='red', linestyle='--', alpha=0.5)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('N Measurements', fontsize=10)

ax.set_xlabel('Absolute Mean Z-score Delta', fontsize=11)
ax.set_ylabel('Standard Deviation', fontsize=11)
ax.set_title('Variance vs Effect Size (Low variance = High confidence)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3d. Cumulative effect distribution
ax = axes[1, 1]
cumulative_effects = df_proteins['Cumulative_Effect'].dropna()
ax.hist(cumulative_effects, bins=50, edgecolor='black', alpha=0.7, color='teal')
ax.axvline(cumulative_effects.median(), color='red', linestyle='--', linewidth=2,
          label=f'Median: {cumulative_effects.median():.2f}')
ax.set_xlabel('Cumulative Z-score Effect', fontsize=11)
ax.set_ylabel('Number of Proteins', fontsize=11)
ax.set_title('Cumulative Effect Distribution\n(Sum of all Z-score deltas per protein)',
            fontsize=12, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_statistical_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: weak_signal_statistical_analysis.png")
plt.close()

# 4. MATRISOME ENRICHMENT
print("Creating matrisome enrichment plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Get matrisome info for weak signals
weak_with_cat = []
for gene in weak_signals['Gene_Symbol']:
    cats = df_full[df_full['Canonical_Gene_Symbol'] == gene]['Matrisome_Division'].unique()
    if len(cats) > 0:
        weak_with_cat.append({'Gene': gene, 'Category': cats[0]})

df_weak_cat = pd.DataFrame(weak_with_cat)

if len(df_weak_cat) > 0:
    category_counts = df_weak_cat['Category'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']

    ax1.barh(category_counts.index, category_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Number of Weak Signal Proteins', fontsize=11)
    ax1.set_title('Weak Signals by Matrisome Category', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)

# Compare to background
background_cats = df_full['Matrisome_Division'].value_counts(normalize=True)
weak_cats_norm = df_weak_cat['Category'].value_counts(normalize=True)

enrichment_data = []
for cat in background_cats.index:
    bg_prop = background_cats[cat]
    weak_prop = weak_cats_norm.get(cat, 0)
    enrichment = weak_prop / bg_prop if bg_prop > 0 else 0
    enrichment_data.append({'Category': cat, 'Enrichment': enrichment})

df_enrich = pd.DataFrame(enrichment_data).sort_values('Enrichment', ascending=True)

colors_enrich = ['green' if x > 1 else 'gray' for x in df_enrich['Enrichment']]
ax2.barh(df_enrich['Category'], df_enrich['Enrichment'], color=colors_enrich, edgecolor='black', linewidth=1.5)
ax2.axvline(1, color='red', linestyle='--', linewidth=2, label='No enrichment')
ax2.set_xlabel('Fold Enrichment (vs Background)', fontsize=11)
ax2.set_title('Category Enrichment in Weak Signals', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_matrisome_enrichment.png', dpi=300, bbox_inches='tight')
print("Saved: weak_signal_matrisome_enrichment.png")
plt.close()

# 5. TISSUE DIVERSITY HEATMAP
print("Creating tissue diversity heatmap...")
top_weak = weak_signals.head(12)

tissue_matrix = []
tissues_list = df_full['Tissue'].unique()

for gene in top_weak['Gene_Symbol']:
    gene_data = df_full[df_full['Canonical_Gene_Symbol'] == gene]
    tissue_deltas = []

    for tissue in tissues_list:
        tissue_data = gene_data[gene_data['Tissue'] == tissue]
        if len(tissue_data) > 0:
            tissue_deltas.append(tissue_data['Zscore_Delta'].mean())
        else:
            tissue_deltas.append(np.nan)

    tissue_matrix.append(tissue_deltas)

df_tissue_matrix = pd.DataFrame(tissue_matrix,
                                index=top_weak['Gene_Symbol'],
                                columns=tissues_list)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(df_tissue_matrix, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f',
            cbar_kws={'label': 'Mean Z-score Delta'},
            linewidths=0.5, linecolor='gray',
            ax=ax)

ax.set_title('Weak Signal Proteins Across Tissues\n(Consistent small changes)',
            fontsize=14, fontweight='bold')
ax.set_xlabel('Tissue', fontsize=11)
ax.set_ylabel('Gene Symbol', fontsize=11)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_tissue_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: weak_signal_tissue_heatmap.png")
plt.close()

print("\nAll visualizations complete!")
