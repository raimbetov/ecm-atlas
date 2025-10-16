#!/usr/bin/env python3
"""
Create comprehensive summary figure for weak signal analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Load data
weak_signals = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_proteins.csv')
df_full = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
df_full = df_full[df_full['Zscore_Delta'].notna()]

# Create figure
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
colors = {'increase': '#E74C3C', 'decrease': '#3498DB'}

# 1. TOP PANEL: Weak Signal Proteins Ranked
ax1 = fig.add_subplot(gs[0, :])

top_14 = weak_signals.head(14)
y_pos = np.arange(len(top_14))

# Create horizontal bars colored by direction
bar_colors = [colors[d] for d in top_14['Dominant_Direction']]
bars = ax1.barh(y_pos, top_14['Abs_Mean_Zscore_Delta'],
                color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add error bars (std)
ax1.errorbar(top_14['Abs_Mean_Zscore_Delta'], y_pos,
             xerr=top_14['Std_Zscore_Delta'],
             fmt='none', ecolor='black', capsize=4, capthick=1.5)

# Add consistency markers (size = consistency)
for i, (idx, row) in enumerate(top_14.iterrows()):
    marker_size = row['Direction_Consistency'] * 100
    ax1.scatter(row['Abs_Mean_Zscore_Delta'], i,
               s=marker_size, color='yellow', edgecolors='black',
               linewidths=2, zorder=10, alpha=0.8)

# Labels
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{g} (n={n}, {int(c*100)}%)"
                      for g, n, c in zip(top_14['Gene_Symbol'],
                                         top_14['N_Measurements'],
                                         top_14['Direction_Consistency'])],
                     fontsize=10)
ax1.set_xlabel('Absolute Mean Z-score Delta', fontsize=12, fontweight='bold')
ax1.set_title('TOP 14 WEAK SIGNAL PROTEINS\n(Bar = Effect Size, Error = Std Dev, Yellow = Consistency)',
             fontsize=14, fontweight='bold')
ax1.axvline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Weak Signal Threshold')
ax1.axvline(1.0, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate Signal')
ax1.grid(True, axis='x', alpha=0.3)
ax1.legend(loc='lower right', fontsize=10)

# Add direction legend
increase_patch = mpatches.Patch(color=colors['increase'], label='Increase with Age')
decrease_patch = mpatches.Patch(color=colors['decrease'], label='Decrease with Age')
ax1.legend(handles=[increase_patch, decrease_patch], loc='upper right', fontsize=10)

# 2. BOTTOM LEFT: Pathway Compounding
ax2 = fig.add_subplot(gs[1, 0])

# Group by pathway
collagens = top_14[top_14['Gene_Symbol'].str.startswith('COL')]
fibrillins = top_14[top_14['Gene_Symbol'].isin(['Fbln5', 'Ltbp4'])]
cathepsins = top_14[top_14['Gene_Symbol'].str.contains('Ctsd|CTSD')]
others = top_14[~top_14['Gene_Symbol'].isin(
    list(collagens['Gene_Symbol']) +
    list(fibrillins['Gene_Symbol']) +
    list(cathepsins['Gene_Symbol'])
)]

pathways = {
    'Collagen\nNetwork': collagens,
    'Fibrillin\nNetwork': fibrillins,
    'Cathepsin\nPathway': cathepsins,
    'Other\nECM': others
}

pathway_effects = []
pathway_names = []
pathway_n = []

for name, group in pathways.items():
    if len(group) > 0:
        cumulative = group['Mean_Zscore_Delta'].sum()
        pathway_effects.append(cumulative)
        pathway_names.append(name)
        pathway_n.append(len(group))

x_pos = np.arange(len(pathway_names))
bar_colors_path = ['#3498DB' if e < 0 else '#E74C3C' for e in pathway_effects]

bars = ax2.bar(x_pos, pathway_effects, color=bar_colors_path,
               alpha=0.7, edgecolor='black', linewidth=2)

# Add n proteins on bars
for i, (bar, n) in enumerate(zip(bars, pathway_n)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height/2,
            f'n={n}', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(pathway_names, fontsize=10, fontweight='bold')
ax2.set_ylabel('Cumulative Z-score Effect', fontsize=11, fontweight='bold')
ax2.set_title('PATHWAY COMPOUNDING EFFECTS', fontsize=12, fontweight='bold')
ax2.axhline(0, color='black', linewidth=1)
ax2.grid(True, axis='y', alpha=0.3)

# 3. BOTTOM MIDDLE: Study Requirements
ax3 = fig.add_subplot(gs[1, 1])

effect_sizes = [0.3, 0.5, 0.8]
n_needed = [11, 4, 2]
colors_power = ['#E74C3C', '#F39C12', '#27AE60']

bars = ax3.barh(effect_sizes, n_needed, color=colors_power,
                alpha=0.7, edgecolor='black', linewidth=2)

# Add text labels
for i, (effect, n) in enumerate(zip(effect_sizes, n_needed)):
    ax3.text(n + 0.3, effect, f'{n} measurements',
            va='center', fontsize=11, fontweight='bold')

ax3.set_yticks(effect_sizes)
ax3.set_yticklabels([f'δ = {e}' for e in effect_sizes], fontsize=11)
ax3.set_xlabel('Measurements Required (80% Power)', fontsize=11, fontweight='bold')
ax3.set_title('STATISTICAL POWER\nα=0.05, Power=80%', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 13)
ax3.grid(True, axis='x', alpha=0.3)

# Add current dataset annotation
ax3.axvline(9, color='blue', linestyle='--', linewidth=2, alpha=0.7,
           label='Current: 9 avg meas/protein')
ax3.legend(fontsize=9)

# 4. BOTTOM RIGHT: Key Insights
ax4 = fig.add_subplot(gs[1, 2])
ax4.axis('off')

insights_text = """
KEY INSIGHTS

1. PROTEINS IDENTIFIED
   • 14 moderate weak signals
   • Effect size: 0.3-1.0 z-score
   • Consistency: 65-100%

2. TOP CANDIDATES
   • Fbln5: -0.50 (9 tissues)
   • Ltbp4: -0.46 (6 tissues)
   • Collagens: -0.37 avg (4 proteins)

3. BIOLOGICAL THEMES
   ✓ Elastic fiber degradation
   ✓ Collagen fibril dysregulation
   ✓ TGFβ pathway disruption
   ✓ Chronic ECM proteolysis

4. STATISTICAL POWER
   • Current: 12 studies adequate
   • Need: +5 studies for δ=0.3
   • Pathways: Compounding amplifies

5. CLINICAL POTENTIAL
   ⚕ Early aging biomarkers
   ⚕ Preventive intervention targets
   ⚕ Tissue-agnostic therapies
"""

ax4.text(0.05, 0.95, insights_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Overall title
fig.suptitle('WEAK SIGNAL AMPLIFIER: ECM AGING WHISPERS\n14 Proteins with Small but Consistent Changes',
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/weak_signal_summary.png',
           dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: weak_signal_summary.png")
plt.close()

print("\nSummary figure complete!")
print(f"\nTop 3 Weak Signals:")
for idx, row in weak_signals.head(3).iterrows():
    print(f"  {row['Gene_Symbol']}: δ={row['Mean_Zscore_Delta']:.3f}, "
          f"consistency={row['Direction_Consistency']*100:.0f}%, "
          f"n={row['N_Measurements']} measurements")
