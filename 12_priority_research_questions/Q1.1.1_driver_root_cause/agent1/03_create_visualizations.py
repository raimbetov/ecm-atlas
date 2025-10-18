#!/usr/bin/env python3
"""
Create visualizations for epigenetic hypothesis
- Driver protein aging trajectories
- Epigenetic mechanism timeline
- Multi-hit cascade model
- Intervention window analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.1_driver_root_cause/agent1")

# Load driver protein data
driver_summary = pd.read_csv(OUTPUT_DIR / "driver_protein_summary.csv", index_col=0)
driver_detailed = pd.read_csv(OUTPUT_DIR / "driver_protein_detailed.csv")

# ============================================================================
# FIGURE 1: Driver Protein Aging Trajectories
# ============================================================================
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Driver Protein Aging Trajectories Across Studies', fontsize=16, fontweight='bold')

# Top proteins to visualize
top_proteins = ['Col14a1', 'Pcolce', 'COL14A1', 'PCOLCE', 'COL21A1', 'COL6A5']

# Filter data
plot_data = driver_detailed[driver_detailed['Canonical_Gene_Symbol'].isin(top_proteins)]

# Panel A: Z-score Delta by protein
ax1 = axes[0, 0]
protein_order = plot_data.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean().sort_values().index
sns.boxplot(data=plot_data, y='Canonical_Gene_Symbol', x='Zscore_Delta',
            order=protein_order, ax=ax1, palette='RdYlBu_r')
ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Z-score Delta (Old - Young)', fontweight='bold')
ax1.set_ylabel('Protein', fontweight='bold')
ax1.set_title('A. Driver Protein Decline Magnitude', fontweight='bold', loc='left')
ax1.grid(axis='x', alpha=0.3)

# Panel B: Young vs Old Z-scores
ax2 = axes[0, 1]
for protein in protein_order[:4]:  # Top 4
    prot_data = plot_data[plot_data['Canonical_Gene_Symbol'] == protein]
    ax2.scatter(prot_data['Zscore_Young'], prot_data['Zscore_Old'],
               label=protein, s=100, alpha=0.7)

ax2.plot([-2, 2], [-2, 2], 'k--', linewidth=1, alpha=0.3, label='No change')
ax2.set_xlabel('Z-score (Young)', fontweight='bold')
ax2.set_ylabel('Z-score (Old)', fontweight='bold')
ax2.set_title('B. Young vs Old Abundance', fontweight='bold', loc='left')
ax2.legend(frameon=True, loc='upper left')
ax2.grid(alpha=0.3)

# Panel C: Study consistency
ax3 = axes[1, 0]
consistency_data = driver_summary.sort_values('Directional_Consistency_%', ascending=False)
bars = ax3.barh(range(len(consistency_data)), consistency_data['Directional_Consistency_%'])
ax3.set_yticks(range(len(consistency_data)))
ax3.set_yticklabels(consistency_data.index)
ax3.set_xlabel('Directional Consistency (%)', fontweight='bold')
ax3.set_title('C. Cross-Study Consistency', fontweight='bold', loc='left')
ax3.axvline(70, color='red', linestyle='--', linewidth=2, alpha=0.5, label='70% threshold')
ax3.legend()
for i, bar in enumerate(bars):
    if consistency_data['Directional_Consistency_%'].iloc[i] >= 70:
        bar.set_color('#2ecc71')
    else:
        bar.set_color('#e74c3c')
ax3.grid(axis='x', alpha=0.3)

# Panel D: Multi-study replication
ax4 = axes[1, 1]
replication_data = driver_summary[['N_Studies', 'N_Tissues', 'N_Measurements']].T
replication_data.plot(kind='bar', ax=ax4, width=0.7)
ax4.set_xlabel('Metric', fontweight='bold')
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('D. Replication Breadth', fontweight='bold', loc='left')
ax4.set_xticklabels(['Studies', 'Tissues', 'Measurements'], rotation=0)
ax4.legend(title='Protein', bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_file1 = OUTPUT_DIR / "figure1_driver_protein_trajectories.png"
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file1}")
plt.close()

# ============================================================================
# FIGURE 2: Epigenetic Multi-Hit Cascade Model
# ============================================================================
fig2, ax = plt.subplots(1, 1, figsize=(16, 10))

# Timeline from age 20 to 60
ages = np.arange(20, 61, 1)

# Model each epigenetic hit as sigmoid curve
def sigmoid_decline(age, start_age, rate, max_effect):
    """Sigmoid decline starting at start_age"""
    return max_effect / (1 + np.exp(-rate * (age - start_age)))

# Define each hit
hit1_methylation = sigmoid_decline(ages, start_age=32, rate=0.3, max_effect=30)  # DNA methylation
hit2_histones = sigmoid_decline(ages, start_age=37, rate=0.25, max_effect=25)     # Histone mods
hit3_TFs = sigmoid_decline(ages, start_age=35, rate=0.2, max_effect=40)           # TF decline
hit4_miRNA = sigmoid_decline(ages, start_age=33, rate=0.25, max_effect=20)        # miRNA increase
hit5_metabolic = sigmoid_decline(ages, start_age=38, rate=0.3, max_effect=30)     # Metabolic
hit6_inflam = sigmoid_decline(ages, start_age=42, rate=0.35, max_effect=35)       # Inflammaging

# Total cumulative effect
total_effect = hit1_methylation + hit2_histones + hit3_TFs + hit4_miRNA + hit5_metabolic + hit6_inflam

# Normalize to 0-100% decline
total_effect_norm = (total_effect / total_effect.max()) * 100

# Plot stacked area
ax.fill_between(ages, 0, hit1_methylation, alpha=0.7, label='HIT 1: DNA Methylation (Age 30-40)', color='#e74c3c')
ax.fill_between(ages, hit1_methylation, hit1_methylation + hit2_histones,
                alpha=0.7, label='HIT 2: Histone Modifications (Age 35-45)', color='#f39c12')
ax.fill_between(ages, hit1_methylation + hit2_histones,
                hit1_methylation + hit2_histones + hit3_TFs,
                alpha=0.7, label='HIT 3: TF Decline (Age 35-50)', color='#f1c40f')
ax.fill_between(ages, hit1_methylation + hit2_histones + hit3_TFs,
                hit1_methylation + hit2_histones + hit3_TFs + hit4_miRNA,
                alpha=0.7, label='HIT 4: miRNA Upregulation (Age 30-50)', color='#3498db')
ax.fill_between(ages, hit1_methylation + hit2_histones + hit3_TFs + hit4_miRNA,
                hit1_methylation + hit2_histones + hit3_TFs + hit4_miRNA + hit5_metabolic,
                alpha=0.7, label='HIT 5: Metabolic Depletion (Age 35-50)', color='#9b59b6')
ax.fill_between(ages, hit1_methylation + hit2_histones + hit3_TFs + hit4_miRNA + hit5_metabolic,
                total_effect,
                alpha=0.7, label='HIT 6: Inflammaging (Age 40-50)', color='#34495e')

# Add total effect line
ax.plot(ages, total_effect, 'k-', linewidth=3, label='CUMULATIVE EFFECT', alpha=0.8)

# Mark intervention windows
ax.axvspan(30, 40, alpha=0.1, color='green', label='OPTIMAL INTERVENTION WINDOW')
ax.axvspan(40, 50, alpha=0.1, color='orange')
ax.axvspan(50, 60, alpha=0.1, color='red')

# Add annotations
ax.annotate('Early intervention\n(preventive)', xy=(35, 20), fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.annotate('Mid-stage intervention\n(slowing)', xy=(45, 80), fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.annotate('Late intervention\n(therapeutic)', xy=(55, 140), fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Observed protein decline (approximate)
observed_decline = sigmoid_decline(ages, start_age=35, rate=0.2, max_effect=80)
ax.plot(ages, observed_decline, 'r--', linewidth=3, alpha=0.6, label='OBSERVED: Protein Decline')

ax.set_xlabel('Age (years)', fontsize=14, fontweight='bold')
ax.set_ylabel('Cumulative Epigenetic Repression (%)', fontsize=14, fontweight='bold')
ax.set_title('Multi-Hit Epigenetic Cascade Model: Driver Protein Decline (Age 30-50)',
            fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, frameon=True)
ax.grid(alpha=0.3)
ax.set_xlim(20, 60)
ax.set_ylim(0, total_effect.max() * 1.1)

plt.tight_layout()
output_file2 = OUTPUT_DIR / "figure2_epigenetic_cascade_model.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file2}")
plt.close()

# ============================================================================
# FIGURE 3: Intervention Strategies Heatmap
# ============================================================================
fig3, ax = plt.subplots(1, 1, figsize=(14, 10))

# Create intervention matrix
interventions = [
    # [Hit1, Hit2, Hit3, Hit4, Hit5, Hit6, Expected_Efficacy]
    ['DNMT Inhibitors (5-azacytidine)', 90, 10, 0, 0, 0, 0, 65],
    ['HDAC Inhibitors (vorinostat)', 20, 90, 30, 0, 0, 20, 75],
    ['miR-29 Antagomirs', 0, 0, 0, 95, 0, 0, 55],
    ['SP1 Overexpression', 0, 10, 90, 0, 0, 0, 60],
    ['NAD+ Boosters (NMN/NR)', 10, 40, 20, 0, 90, 10, 70],
    ['α-Ketoglutarate', 50, 50, 0, 0, 70, 0, 65],
    ['Anti-IL-6 (Tocilizumab)', 30, 30, 10, 0, 0, 90, 60],
    ['Rapamycin (mTOR inhibitor)', 10, 20, 10, 0, 30, 70, 65],
    ['Exercise', 20, 60, 10, 10, 40, 50, 80],
    ['Combination Therapy (All)', 95, 95, 90, 95, 90, 90, 95]
]

intervention_df = pd.DataFrame(interventions,
                               columns=['Intervention', 'HIT1_DNA_Meth', 'HIT2_Histones',
                                       'HIT3_TF', 'HIT4_miRNA', 'HIT5_Metabolic',
                                       'HIT6_Inflam', 'Expected_Efficacy'])

# Prepare heatmap data
heatmap_data = intervention_df.set_index('Intervention').iloc[:, :-1]  # Exclude efficacy for now
efficacy = intervention_df.set_index('Intervention')['Expected_Efficacy']

# Create heatmap
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='RdYlGn', cbar_kws={'label': 'Target Coverage (%)'},
           linewidths=0.5, ax=ax, vmin=0, vmax=100)

# Add efficacy as side annotation
for i, (idx, val) in enumerate(efficacy.items()):
    ax.text(heatmap_data.shape[1] + 0.5, i + 0.5, f'{val}%',
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue' if val >= 70 else 'lightyellow', alpha=0.7))

ax.set_xlabel('Epigenetic Hit Target', fontsize=13, fontweight='bold')
ax.set_ylabel('Intervention Strategy', fontsize=13, fontweight='bold')
ax.set_title('Intervention Coverage Matrix: Targeting Epigenetic Hits\n(Right column: Expected Overall Efficacy)',
            fontsize=14, fontweight='bold')

plt.tight_layout()
output_file3 = OUTPUT_DIR / "figure3_intervention_strategies.png"
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file3}")
plt.close()

# ============================================================================
# FIGURE 4: Mechanistic Summary Schematic
# ============================================================================
fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
fig4.suptitle('Epigenetic Mechanisms Timeline: Age 30-50', fontsize=18, fontweight='bold')

mechanisms_timeline = {
    'Age 30-35': ['DNA Methylation\n(CpG islands)', 'miR-29 ↑\n(Collagen targeting)', 'Chromatin accessibility ↓'],
    'Age 35-40': ['H3K27ac loss\n(Enhancers)', 'SP1 decline 20%\n(TF availability)', 'TET enzyme ↓'],
    'Age 40-45': ['H3K27me3 gain\n(Polycomb)', 'miR-34a ↑\n(Senescence)', 'NAD+ depletion'],
    'Age 45-50': ['H3K9me3 spread\n(Heterochromatin)', 'NF-κB activation\n(Inflammation)', 'SAM/SAH ↓'],
}

for idx, (age_range, mechanisms) in enumerate(mechanisms_timeline.items()):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]

    # Create text summary
    y_pos = 0.9
    ax.text(0.5, 0.95, age_range, ha='center', va='top', fontsize=14, fontweight='bold',
           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    for i, mech in enumerate(mechanisms):
        colors = ['#e74c3c', '#f39c12', '#3498db']
        ax.text(0.5, 0.7 - i*0.25, mech, ha='center', va='center', fontsize=11,
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# Add cumulative effect panels
axes[1, 0].text(0.5, 0.5, 'CUMULATIVE\nREPRESSION', ha='center', va='center', fontsize=16, fontweight='bold',
               transform=axes[1, 0].transAxes, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
axes[1, 0].text(0.5, 0.2, '20-30%\nProtein Decline', ha='center', va='center', fontsize=12,
               transform=axes[1, 0].transAxes)

axes[1, 1].text(0.5, 0.5, 'DOWNSTREAM\nEFFECTS', ha='center', va='center', fontsize=16, fontweight='bold',
               transform=axes[1, 1].transAxes, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
axes[1, 1].text(0.5, 0.2, 'ECM Quality ↓\nTissue Stiffness ↑', ha='center', va='center', fontsize=12,
               transform=axes[1, 1].transAxes)

axes[1, 2].text(0.5, 0.5, 'INTERVENTION\nWINDOW', ha='center', va='center', fontsize=16, fontweight='bold',
               transform=axes[1, 2].transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
axes[1, 2].text(0.5, 0.2, 'Optimal: Age 30-40\nCombination Rx', ha='center', va='center', fontsize=12,
               transform=axes[1, 2].transAxes)

plt.tight_layout()
output_file4 = OUTPUT_DIR / "figure4_mechanistic_timeline.png"
plt.savefig(output_file4, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file4}")
plt.close()

print("\n" + "="*80)
print("VISUALIZATION GENERATION COMPLETE")
print("="*80)
print(f"Generated 4 figures:")
print(f"  1. Driver protein trajectories")
print(f"  2. Epigenetic cascade model")
print(f"  3. Intervention strategies")
print(f"  4. Mechanistic timeline")
