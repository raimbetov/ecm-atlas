#!/usr/bin/env python3
"""
HYPOTHESIS 01: INVERSE ABUNDANCE PARADOX ANALYSIS

Nobel Prize Hypothesis: ECM proteins that DECREASE with aging correlate with pathology
- "Quality over Quantity" paradigm shift
- Less protein = more disease (inverse correlation)

Analysis Pipeline:
1. Filter universal proteins (≥3 tissues, ≥70% consistency)
2. Focus on DOWNREGULATED proteins (Direction == DOWN)
3. Find inverse paradox candidates (strong down + high universality)
4. Analyze biological significance
5. Generate publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style for publication-quality figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Directories
BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights")
DATA_FILE = BASE_DIR / "agent_01_universal_markers" / "agent_01_universal_markers_data.csv"
OUTPUT_DIR = BASE_DIR / "age_related_proteins" / "hypothesis_01_inverse_paradox"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INVERSE ABUNDANCE PARADOX ANALYSIS")
print("=" * 80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv(DATA_FILE)
print(f"   Total proteins: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# Basic statistics
print("\n2. DATASET OVERVIEW:")
print(f"   - Core matrisome: {len(df[df['Matrisome_Division'] == 'Core matrisome'])}")
print(f"   - Matrisome-associated: {len(df[df['Matrisome_Division'] == 'Matrisome-associated'])}")
print(f"   - Upregulated: {len(df[df['Predominant_Direction'] == 'UP'])}")
print(f"   - Downregulated: {len(df[df['Predominant_Direction'] == 'DOWN'])}")

# STEP 1: Filter for universal proteins
print("\n3. FILTERING UNIVERSAL PROTEINS...")
universal_filter = (df['N_Tissues'] >= 3) & (df['Direction_Consistency'] >= 0.70)
df_universal = df[universal_filter].copy()
print(f"   Proteins with ≥3 tissues & ≥70% consistency: {len(df_universal)}")

# STEP 2: Focus on downregulated proteins
print("\n4. FILTERING DOWNREGULATED PROTEINS...")
df_down = df_universal[df_universal['Predominant_Direction'] == 'DOWN'].copy()
print(f"   Universal downregulated proteins: {len(df_down)}")

# STEP 3: Identify inverse paradox candidates
print("\n5. IDENTIFYING INVERSE PARADOX CANDIDATES...")
# Criteria:
# - Strong downregulation: Mean_Zscore_Delta < -0.8
# - High universality: Universality_Score > 0.7
# - Good consistency: Direction_Consistency >= 0.8

inverse_paradox = df_down[
    (df_down['Mean_Zscore_Delta'] < -0.8) &
    (df_down['Universality_Score'] > 0.7) &
    (df_down['Direction_Consistency'] >= 0.8)
].copy()

print(f"\n   INVERSE PARADOX CANDIDATES (n={len(inverse_paradox)}):")
print("   " + "-" * 76)

if len(inverse_paradox) > 0:
    # Sort by Mean_Zscore_Delta (most negative first)
    inverse_paradox_sorted = inverse_paradox.sort_values('Mean_Zscore_Delta')

    for idx, row in inverse_paradox_sorted.iterrows():
        print(f"   {row['Gene_Symbol']:15s} | Δz={row['Mean_Zscore_Delta']:+.3f} | "
              f"Univ={row['Universality_Score']:.3f} | Cons={row['Direction_Consistency']:.1%} | "
              f"{row['N_Tissues']} tissues | {row['Matrisome_Category']}")
else:
    print("   No proteins meet strict criteria. Relaxing thresholds...")

    # Relaxed criteria
    inverse_paradox_relaxed = df_down[
        (df_down['Mean_Zscore_Delta'] < -0.5) &
        (df_down['Universality_Score'] > 0.6) &
        (df_down['Direction_Consistency'] >= 0.75)
    ].copy()

    inverse_paradox = inverse_paradox_relaxed
    print(f"\n   RELAXED CRITERIA CANDIDATES (n={len(inverse_paradox)}):")
    print("   " + "-" * 76)

    if len(inverse_paradox) > 0:
        inverse_paradox_sorted = inverse_paradox.sort_values('Mean_Zscore_Delta')

        for idx, row in inverse_paradox_sorted.iterrows():
            print(f"   {row['Gene_Symbol']:15s} | Δz={row['Mean_Zscore_Delta']:+.3f} | "
                  f"Univ={row['Universality_Score']:.3f} | Cons={row['Direction_Consistency']:.1%} | "
                  f"{row['N_Tissues']} tissues | {row['Matrisome_Category']}")

# Save inverse paradox candidates
inverse_paradox.to_csv(OUTPUT_DIR / "inverse_paradox_candidates.csv", index=False)
print(f"\n   Saved: inverse_paradox_candidates.csv")

# STEP 4: Top downregulated proteins (for broader analysis)
print("\n6. TOP 20 DOWNREGULATED UNIVERSAL PROTEINS:")
print("   " + "-" * 76)
top20_down = df_down.nsmallest(20, 'Mean_Zscore_Delta')
for idx, row in top20_down.iterrows():
    sig = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
    print(f"   {row['Gene_Symbol']:15s} | Δz={row['Mean_Zscore_Delta']:+.3f} | "
          f"p={row['P_Value']:.4f}{sig:3s} | {row['N_Tissues']:2d} tissues | "
          f"{row['Matrisome_Category']}")

top20_down.to_csv(OUTPUT_DIR / "top20_downregulated_proteins.csv", index=False)

# STEP 5: Category enrichment analysis
print("\n7. MATRISOME CATEGORY ANALYSIS:")
print("   " + "-" * 76)

# All downregulated
category_counts = df_down['Matrisome_Category'].value_counts()
print("\n   All downregulated proteins by category:")
for cat, count in category_counts.items():
    pct = (count / len(df_down)) * 100
    print(f"   {cat:30s}: {count:3d} ({pct:.1f}%)")

# Inverse paradox (if any)
if len(inverse_paradox) > 0:
    paradox_categories = inverse_paradox['Matrisome_Category'].value_counts()
    print("\n   Inverse paradox candidates by category:")
    for cat, count in paradox_categories.items():
        pct = (count / len(inverse_paradox)) * 100
        print(f"   {cat:30s}: {count:3d} ({pct:.1f}%)")

# STEP 6: Statistical summary
print("\n8. STATISTICAL SUMMARY:")
print("   " + "-" * 76)

print(f"\n   All universal downregulated (n={len(df_down)}):")
print(f"   Mean Δz: {df_down['Mean_Zscore_Delta'].mean():.3f} ± {df_down['Mean_Zscore_Delta'].std():.3f}")
print(f"   Median Δz: {df_down['Mean_Zscore_Delta'].median():.3f}")
print(f"   Range: [{df_down['Mean_Zscore_Delta'].min():.3f}, {df_down['Mean_Zscore_Delta'].max():.3f}]")
print(f"   Universality: {df_down['Universality_Score'].mean():.3f} ± {df_down['Universality_Score'].std():.3f}")

if len(inverse_paradox) > 0:
    print(f"\n   Inverse paradox candidates (n={len(inverse_paradox)}):")
    print(f"   Mean Δz: {inverse_paradox['Mean_Zscore_Delta'].mean():.3f} ± {inverse_paradox['Mean_Zscore_Delta'].std():.3f}")
    print(f"   Median Δz: {inverse_paradox['Mean_Zscore_Delta'].median():.3f}")
    print(f"   Range: [{inverse_paradox['Mean_Zscore_Delta'].min():.3f}, {inverse_paradox['Mean_Zscore_Delta'].max():.3f}]")
    print(f"   Universality: {inverse_paradox['Universality_Score'].mean():.3f} ± {inverse_paradox['Universality_Score'].std():.3f}")

# Proteins with p < 0.01
significant = df_down[df_down['P_Value'] < 0.01]
print(f"\n   Statistically significant (p<0.01): {len(significant)} proteins")

# ============================================================================
# VISUALIZATION 1: Volcano Plot
# ============================================================================
print("\n9. GENERATING VISUALIZATIONS...")
print("   Creating volcano plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot all universal proteins
scatter_all = ax.scatter(
    df_universal['Mean_Zscore_Delta'],
    -np.log10(df_universal['P_Value'] + 1e-10),
    c='lightgray',
    s=30,
    alpha=0.5,
    label='All universal proteins'
)

# Highlight downregulated
scatter_down = ax.scatter(
    df_down['Mean_Zscore_Delta'],
    -np.log10(df_down['P_Value'] + 1e-10),
    c='steelblue',
    s=50,
    alpha=0.7,
    label=f'Downregulated (n={len(df_down)})'
)

# Highlight inverse paradox (if any)
if len(inverse_paradox) > 0:
    scatter_paradox = ax.scatter(
        inverse_paradox['Mean_Zscore_Delta'],
        -np.log10(inverse_paradox['P_Value'] + 1e-10),
        c='darkred',
        s=100,
        alpha=0.9,
        edgecolors='black',
        linewidth=1.5,
        label=f'Inverse paradox (n={len(inverse_paradox)})',
        marker='D'
    )

    # Annotate inverse paradox proteins
    for idx, row in inverse_paradox.iterrows():
        ax.annotate(
            row['Gene_Symbol'].split(';')[0],  # First gene name if multiple
            xy=(row['Mean_Zscore_Delta'], -np.log10(row['P_Value'] + 1e-10)),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='darkred'
        )

# Add threshold lines
ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=-np.log10(0.01), color='red', linestyle='--', alpha=0.5, linewidth=1, label='p=0.01')
ax.axhline(y=-np.log10(0.05), color='orange', linestyle='--', alpha=0.3, linewidth=1, label='p=0.05')

ax.set_xlabel('Mean Z-score Delta (Δz)', fontsize=14, fontweight='bold')
ax.set_ylabel('-log₁₀(p-value)', fontsize=14, fontweight='bold')
ax.set_title('Inverse Abundance Paradox: Downregulated ECM Proteins',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_volcano_plot_inverse_paradox.png", dpi=300)
print(f"   Saved: 01_volcano_plot_inverse_paradox.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Heatmap of Top 20 Downregulated
# ============================================================================
print("   Creating heatmap of top 20 downregulated proteins...")

# Create heatmap data
heatmap_data = []
for idx, row in top20_down.iterrows():
    heatmap_data.append({
        'Gene': row['Gene_Symbol'].split(';')[0],
        'Mean Δz': row['Mean_Zscore_Delta'],
        'Universality': row['Universality_Score'],
        'Consistency': row['Direction_Consistency'],
        'N_Tissues': row['N_Tissues'],
        'Category': row['Matrisome_Category'][:20]  # Truncate for display
    })

df_heatmap = pd.DataFrame(heatmap_data)

fig, ax = plt.subplots(figsize=(10, 12))

# Create color mapping
colors = ['Mean Δz', 'Universality', 'Consistency']
data_for_heatmap = df_heatmap[colors].T

im = ax.imshow(data_for_heatmap, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(df_heatmap)))
ax.set_yticks(np.arange(len(colors)))
ax.set_xticklabels(df_heatmap['Gene'], rotation=45, ha='right')
ax.set_yticklabels(colors)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Score', rotation=270, labelpad=20)

# Add values as text
for i in range(len(colors)):
    for j in range(len(df_heatmap)):
        text = ax.text(j, i, f'{data_for_heatmap.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Top 20 Downregulated ECM Proteins\n(Age-Related Decline)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_heatmap_top20_downregulated.png", dpi=300)
print(f"   Saved: 02_heatmap_top20_downregulated.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Category Enrichment Bar Plot
# ============================================================================
print("   Creating category enrichment plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Count categories
categories = df_down['Matrisome_Category'].value_counts()
x_pos = np.arange(len(categories))

bars = ax.bar(x_pos, categories.values, color='steelblue', alpha=0.7, edgecolor='black')

# Add percentages
for i, (cat, count) in enumerate(categories.items()):
    pct = (count / len(df_down)) * 100
    ax.text(i, count + 0.5, f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.set_xlabel('Matrisome Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Proteins', fontsize=14, fontweight='bold')
ax.set_title(f'Matrisome Category Distribution\nUniversal Downregulated Proteins (n={len(df_down)})',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(categories.index, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_category_enrichment.png", dpi=300)
print(f"   Saved: 03_category_enrichment.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Universality vs Downregulation Scatter
# ============================================================================
print("   Creating universality vs downregulation scatter plot...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot all downregulated
scatter = ax.scatter(
    df_down['Mean_Zscore_Delta'],
    df_down['Universality_Score'],
    c=df_down['N_Tissues'],
    s=100,
    alpha=0.6,
    cmap='viridis',
    edgecolors='black',
    linewidth=0.5
)

# Highlight inverse paradox
if len(inverse_paradox) > 0:
    ax.scatter(
        inverse_paradox['Mean_Zscore_Delta'],
        inverse_paradox['Universality_Score'],
        s=200,
        alpha=0.9,
        edgecolors='red',
        linewidth=3,
        facecolors='none',
        marker='o'
    )

    # Annotate
    for idx, row in inverse_paradox.iterrows():
        ax.annotate(
            row['Gene_Symbol'].split(';')[0],
            xy=(row['Mean_Zscore_Delta'], row['Universality_Score']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='red'
        )

# Add threshold lines
ax.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5, label='Strong decline threshold')
ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='High universality threshold')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Number of Tissues', rotation=270, labelpad=20)

ax.set_xlabel('Mean Z-score Delta (Δz)', fontsize=14, fontweight='bold')
ax.set_ylabel('Universality Score', fontsize=14, fontweight='bold')
ax.set_title('Universality vs Downregulation\n(Inverse Paradox Candidate Space)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_universality_vs_downregulation.png", dpi=300)
print(f"   Saved: 04_universality_vs_downregulation.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Distribution comparison
# ============================================================================
print("   Creating distribution comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Z-score delta distribution
ax1 = axes[0, 0]
ax1.hist(df_down['Mean_Zscore_Delta'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
if len(inverse_paradox) > 0:
    ax1.axvline(inverse_paradox['Mean_Zscore_Delta'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Inverse paradox mean')
ax1.set_xlabel('Mean Z-score Delta', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Distribution of Downregulation', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Universality score distribution
ax2 = axes[0, 1]
ax2.hist(df_down['Universality_Score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
if len(inverse_paradox) > 0:
    ax2.axvline(inverse_paradox['Universality_Score'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Inverse paradox mean')
ax2.set_xlabel('Universality Score', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('Distribution of Universality', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Number of tissues
ax3 = axes[1, 0]
tissue_counts = df_down['N_Tissues'].value_counts().sort_index()
ax3.bar(tissue_counts.index, tissue_counts.values, color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of Tissues', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Tissue Coverage', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: P-value distribution
ax4 = axes[1, 1]
ax4.hist(-np.log10(df_down['P_Value'] + 1e-10), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(-np.log10(0.05), color='orange', linestyle='--', linewidth=2, label='p=0.05')
ax4.axvline(-np.log10(0.01), color='red', linestyle='--', linewidth=2, label='p=0.01')
ax4.set_xlabel('-log₁₀(p-value)', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('Statistical Significance', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Statistical Distributions of Downregulated ECM Proteins',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_statistical_distributions.png", dpi=300)
print(f"   Saved: 05_statistical_distributions.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - inverse_paradox_candidates.csv")
print("  - top20_downregulated_proteins.csv")
print("  - 01_volcano_plot_inverse_paradox.png")
print("  - 02_heatmap_top20_downregulated.png")
print("  - 03_category_enrichment.png")
print("  - 04_universality_vs_downregulation.png")
print("  - 05_statistical_distributions.png")

print("\n" + "=" * 80)
print("NEXT STEP: Generate discovery report (02_discovery_report.md)")
print("=" * 80)
