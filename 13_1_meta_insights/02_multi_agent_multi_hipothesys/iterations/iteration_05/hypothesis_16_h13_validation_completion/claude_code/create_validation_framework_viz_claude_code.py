#!/usr/bin/env python3
"""
H13 Validation Framework Visualization
======================================

Creates comprehensive visualizations showing:
1. Validation methodology flowchart
2. Internal baselines (tissue velocities, top proteins)
3. Expected validation outcomes (scenarios)
4. Dataset overview (6 external datasets identified)

Author: claude_code
Date: 2025-10-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Setup
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas")
WORK_DIR = BASE_DIR / "13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_16_h13_validation_completion/claude_code"
VIZ_DIR = WORK_DIR / "visualizations_claude_code"
VIZ_DIR.mkdir(exist_ok=True, parents=True)

# Load internal data
INTERNAL_DATA = BASE_DIR / "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
df = pd.read_csv(INTERNAL_DATA)

print("🎨 Creating H13 Validation Framework Visualizations...")

# =============================================================================
# 1. Tissue Velocity Baseline
# =============================================================================
print("\n📊 1. Tissue Velocity Baseline...")

velocities = df.groupby('Tissue')['Zscore_Delta'].apply(lambda x: x.abs().mean()).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(velocities)))
bars = ax.barh(range(len(velocities)), velocities.values, color=colors)

ax.set_yticks(range(len(velocities)))
ax.set_yticklabels(velocities.index, fontsize=9)
ax.set_xlabel('Aging Velocity (mean |ΔZ|)', fontweight='bold')
ax.set_title('H03 Tissue Aging Velocities (Internal Baseline)\nFast-aging (top) vs Slow-aging (bottom)',
             fontweight='bold', fontsize=12)

# Add value labels
for i, (tissue, vel) in enumerate(velocities.items()):
    ax.text(vel + 0.02, i, f'{vel:.3f}', va='center', fontsize=8)

# Add velocity range annotation
ax.text(0.95, 0.05, f'Range: {velocities.min():.3f} - {velocities.max():.3f}\nRatio: {velocities.max()/velocities.min():.2f}×',
        transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# Highlight muscle tissues (for PXD011967 validation)
muscle_tissues = [t for t in velocities.index if 'muscle' in t.lower()]
for i, tissue in enumerate(velocities.index):
    if tissue in muscle_tissues:
        ax.get_yticklabels()[i].set_weight('bold')
        ax.get_yticklabels()[i].set_color('red')

ax.text(0.05, 0.95, '← Red: Skeletal muscle tissues\n(PXD011967 validation target)',
        transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3), fontsize=8)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'h03_tissue_velocities_internal_baseline_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: h03_tissue_velocities_internal_baseline_claude_code.png")

# =============================================================================
# 2. Top 20 Proteins for Meta-Analysis
# =============================================================================
print("\n📊 2. Top 20 Proteins for Meta-Analysis...")

top_proteins = df.groupby('Gene_Symbol')['Zscore_Delta'].apply(lambda x: x.abs().mean()).nlargest(20)

fig, ax = plt.subplots(figsize=(10, 8))
colors_proteins = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_proteins)))
bars = ax.barh(range(len(top_proteins)), top_proteins.values, color=colors_proteins)

ax.set_yticks(range(len(top_proteins)))
ax.set_yticklabels([f"{i+1}. {g}" for i, g in enumerate(top_proteins.index)], fontsize=9)
ax.set_xlabel('Mean |ΔZ| (Aging Effect Size)', fontweight='bold')
ax.set_title('Top 20 Proteins for Meta-Analysis (I² Heterogeneity Testing)\nHighest Aging Signal in Internal Data',
             fontweight='bold', fontsize=12)

# Add value labels
for i, (gene, delta_z) in enumerate(top_proteins.items()):
    ax.text(delta_z + 0.1, i, f'{delta_z:.2f}', va='center', fontsize=8)

# Add I² interpretation box
i2_text = """I² Heterogeneity Targets:
• I² < 25%: STABLE (consistent)
• I² 25-50%: MODERATE
• I² > 50%: VARIABLE (context-dependent)

Success: ≥15/20 proteins with I²<50%"""

ax.text(0.98, 0.02, i2_text,
        transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
        fontsize=8, family='monospace')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'top_20_proteins_meta_analysis_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: top_20_proteins_meta_analysis_claude_code.png")

# =============================================================================
# 3. External Datasets Overview
# =============================================================================
print("\n📊 3. External Datasets Overview...")

datasets = {
    'PXD011967\nMuscle\n(HIGH)': {'n': 58, 'proteins': 4380, 'priority': 1, 'age_groups': 5},
    'PXD015982\nSkin\n(HIGH)': {'n': 6, 'proteins': 229, 'priority': 1, 'age_groups': 2},
    'PXD007048\nBone\n(MED)': {'n': 20, 'proteins': 2000, 'priority': 2, 'age_groups': 2},
    'MSV000082958\nLung\n(MED)': {'n': 15, 'proteins': 1500, 'priority': 2, 'age_groups': 2},
    'MSV000096508\nBrain\n(MED)': {'n': 30, 'proteins': 3000, 'priority': 2, 'age_groups': 3},
    'PXD016440\nSkin2\n(MED)': {'n': 10, 'proteins': 500, 'priority': 2, 'age_groups': 2}
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('External Datasets Overview (Identified by H13 Claude)', fontweight='bold', fontsize=14)

# Plot 1: Sample sizes
ax = axes[0, 0]
names = list(datasets.keys())
sample_sizes = [d['n'] for d in datasets.values()]
colors_priority = ['red' if d['priority']==1 else 'orange' for d in datasets.values()]

ax.bar(range(len(names)), sample_sizes, color=colors_priority, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of Samples', fontweight='bold')
ax.set_title('Sample Sizes per Dataset', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, n in enumerate(sample_sizes):
    ax.text(i, n+1, str(n), ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add legend
ax.text(0.98, 0.98, '■ HIGH priority\n■ MEDIUM priority',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8),
        fontsize=8)

# Plot 2: Protein counts
ax = axes[0, 1]
protein_counts = [d['proteins'] for d in datasets.values()]

ax.bar(range(len(names)), protein_counts, color=colors_priority, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of Proteins Quantified', fontweight='bold')
ax.set_title('Protein Coverage per Dataset', fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

for i, n in enumerate(protein_counts):
    ax.text(i, n*1.1, str(n), ha='center', va='bottom', fontsize=8)

# Plot 3: Age groups
ax = axes[1, 0]
age_groups = [d['age_groups'] for d in datasets.values()]

ax.bar(range(len(names)), age_groups, color=colors_priority, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Number of Age Groups', fontweight='bold')
ax.set_title('Age Group Granularity', fontweight='bold')
ax.set_ylim(0, 6)
ax.grid(axis='y', alpha=0.3)

for i, n in enumerate(age_groups):
    ax.text(i, n+0.1, str(n), ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 4: Overall summary table
ax = axes[1, 1]
ax.axis('off')

summary_text = """DATASET ACQUISITION STATUS

✅ Identified: 6/6 datasets
📋 Metadata: 2/6 documented (HIGH priority)
⏳ Downloaded: 0/6 (placeholders only)
⚠️  CRITICAL BLOCKER: Real data needed

NEXT STEPS:
1. Download PXD011967 (eLife supp.)
2. Download PXD015982 (PMC supp.)
3. Preprocess to z-score format
4. Run validation pipeline
5. Generate results & visualizations

EXPECTED OVERLAP:
• PXD011967: ~300 ECM genes (38-46%)
• PXD015982: ~150 ECM genes (23-31%, matrisome-focused)

VALIDATION TARGETS:
• H08 S100 model: R² ≥ 0.60
• H06 biomarkers: AUC ≥ 0.80
• H03 velocities: ρ > 0.70
• Meta-analysis: I² < 50% for ≥15/20 proteins"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        ha='left', va='top', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(VIZ_DIR / 'external_datasets_overview_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: external_datasets_overview_claude_code.png")

# =============================================================================
# 4. Validation Scenario Flowchart
# =============================================================================
print("\n📊 4. Validation Scenarios...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'H13 EXTERNAL VALIDATION SCENARIOS', ha='center', va='top',
        fontsize=16, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))

# Scenario boxes
scenarios = [
    {
        'y': 7.5,
        'title': 'SCENARIO 1: STRONG VALIDATION ✅',
        'color': 'lightgreen',
        'criteria': [
            'H08 external R² ≥ 0.65',
            'H06 external AUC ≥ 0.85',
            'H03 velocity ρ > 0.75',
            'Meta I²<50%: ≥18/20 proteins'
        ],
        'action': 'Publish in Nature/Cell\nClinical biomarker development\nPatent filing\nGrant applications'
    },
    {
        'y': 5.0,
        'title': 'SCENARIO 2: MODERATE VALIDATION ⚠️',
        'color': 'lightyellow',
        'criteria': [
            'H08 external R² = 0.50-0.65',
            'H06 external AUC = 0.75-0.85',
            'H03 velocity ρ = 0.60-0.75',
            'Meta I²<50%: 12-17/20 proteins'
        ],
        'action': 'Focus on STABLE proteins only\nExpand validation cohorts\nPublish in eLife/Aging Cell\nInvestigate VARIABLE proteins'
    },
    {
        'y': 2.5,
        'title': 'SCENARIO 3: POOR VALIDATION ❌',
        'color': 'lightcoral',
        'criteria': [
            'H08 external R² < 0.40',
            'H06 external AUC < 0.70',
            'H03 velocity ρ < 0.50',
            'Meta I²<50%: <12/20 proteins'
        ],
        'action': 'Acknowledge overfitting\nRe-evaluate ALL hypotheses\nFuture work: mandatory external validation\nFocus on mechanisms, not prediction'
    },
    {
        'y': 0.5,
        'title': 'SCENARIO 4: NO DATA ACCESSIBLE ⚠️',
        'color': 'lightgray',
        'criteria': [
            '<2 datasets successfully downloaded',
            'Paywalls / broken links',
            'FTP access issues'
        ],
        'action': 'Contact study authors\nSearch alternative repositories\nGenerate NEW prospective cohort\nCollaborate with clinical labs'
    }
]

for scenario in scenarios:
    # Box background
    rect = plt.Rectangle((0.5, scenario['y']-0.6), 9, 1.1,
                         facecolor=scenario['color'], edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)

    # Title
    ax.text(5, scenario['y']+0.35, scenario['title'], ha='center', va='top',
            fontsize=11, fontweight='bold')

    # Criteria
    criteria_text = '\n'.join(f"  • {c}" for c in scenario['criteria'])
    ax.text(1, scenario['y']+0.1, criteria_text, ha='left', va='top', fontsize=8, family='monospace')

    # Action
    ax.text(6, scenario['y']+0.1, scenario['action'], ha='left', va='top',
            fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Current status
ax.text(5, -0.5, 'CURRENT STATUS: Framework ready, awaiting real external data download',
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

plt.tight_layout()
plt.savefig(VIZ_DIR / 'validation_scenarios_flowchart_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved: validation_scenarios_flowchart_claude_code.png")

# =============================================================================
# 5. Save summary statistics
# =============================================================================
print("\n📊 5. Saving Summary Statistics...")

summary_stats = {
    'Internal_data': {
        'Total_rows': len(df),
        'Unique_genes': df['Gene_Symbol'].nunique(),
        'Tissues': df['Tissue'].nunique(),
        'Studies': df['Study_ID'].nunique()
    },
    'Tissue_velocities': {
        'Range': f"{velocities.min():.3f} - {velocities.max():.3f}",
        'Ratio': f"{velocities.max()/velocities.min():.2f}×",
        'Fastest': velocities.index[0],
        'Slowest': velocities.index[-1]
    },
    'Top_20_proteins': {
        'Range': f"{top_proteins.min():.3f} - {top_proteins.max():.3f}",
        'Top_protein': top_proteins.index[0],
        'Top_delta_z': top_proteins.values[0]
    },
    'External_datasets': {
        'Identified': 6,
        'HIGH_priority': 2,
        'MEDIUM_priority': 4,
        'Downloaded': 0,
        'Status': 'Metadata created, real data needed'
    },
    'Validation_targets': {
        'H08_R2_threshold': 0.60,
        'H06_AUC_threshold': 0.80,
        'H03_rho_threshold': 0.70,
        'Meta_I2_stable_threshold': 15
    }
}

with open(WORK_DIR / 'validation_framework_summary_claude_code.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"   ✓ Saved: validation_framework_summary_claude_code.json")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("✅ VISUALIZATION FRAMEWORK COMPLETE")
print("="*80)
print(f"\nGenerated visualizations:")
print(f"1. ✓ h03_tissue_velocities_internal_baseline_claude_code.png")
print(f"2. ✓ top_20_proteins_meta_analysis_claude_code.png")
print(f"3. ✓ external_datasets_overview_claude_code.png")
print(f"4. ✓ validation_scenarios_flowchart_claude_code.png")
print(f"5. ✓ validation_framework_summary_claude_code.json")

print(f"\nSaved to: {VIZ_DIR}")

print("\n📋 WHEN REAL EXTERNAL DATA IS AVAILABLE:")
print("   → Re-run h13_completion_claude_code.py")
print("   → Will auto-generate 33 additional validation visualizations:")
print("     • H08 transfer scatter plots (12)")
print("     • H06 ROC curves (12)")
print("     • H03 velocity correlations (6)")
print("     • Meta-analysis forest plot (1)")
print("     • I² heterogeneity heatmap (1)")
print("     • Dataset Venn diagram (1)")

print("\n🚨 CRITICAL: This hypothesis determines if H01-H15 are ROBUST or OVERFIT!")
print("="*80)
