#!/usr/bin/env python3
"""
Comprehensive Phase I Metabolic Hypothesis Analysis
Combines: (1) Literature-based metabolic simulation, (2) Proteomics proxy analysis, (3) Visualization
Conclusion: Phase I NOT validated due to data unavailability, but provides roadmap for future study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup
import os
np.random.seed(42)
sns.set_style('whitegrid')
os.makedirs('visualizations_claude_code', exist_ok=True)
os.makedirs('metabolomics_data', exist_ok=True)

print("="*90)
print("🧪 COMPREHENSIVE PHASE I METABOLIC HYPOTHESIS ANALYSIS")
print("="*90)

# ============================================================================
# PART 1: Load Proteomic Data and Tissue Velocities
# ============================================================================

print("\n📊 PART 1: Loading ECM Proteomics and Tissue Velocities...")
print("-" * 90)

# Load main proteom dataset
df_proteomics = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Load velocities
df_velocity = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_01/hypothesis_03_tissue_aging_clocks/claude_code/tissue_aging_velocity_claude_code.csv')

print(f"✅ Proteomics: {len(df_proteomics)} observations, {df_proteomics['Gene_Symbol'].nunique()} unique proteins")
print(f"✅ Velocities: {len(df_velocity)} tissues, velocity range {df_velocity['Velocity'].min():.2f}-{df_velocity['Velocity'].max():.2f}")

# Define phases (from H12)
df_velocity['Phase'] = pd.cut(df_velocity['Velocity'],
                               bins=[-np.inf, 1.65, 2.17, np.inf],
                               labels=['Phase I (v<1.65)', 'Transition (1.65-2.17)', 'Phase II (v>2.17)'])

print(f"\n📊 Phase Distribution:")
print(df_velocity['Phase'].value_counts().sort_index())

# ============================================================================
# PART 2: Search Proteomics for Metabolic Proxy Markers
# ============================================================================

print("\n\n📊 PART 2: Searching Proteomics for Metabolic Proxy Markers...")
print("-" * 90)

# Known metabolic proteins (if present in ECM dataset)
metabolic_markers = {
    'Mitochondrial': ['ATP5A1', 'ATP5B', 'COX4I1', 'COX5A', 'NDUFA9', 'NDUFB8', 'SDHB', 'UQCRC2'],
    'Glycolysis': ['GAPDH', 'PKM', 'LDHA', 'LDHB', 'ENO1', 'ENO2', 'PGK1', 'ALDOA'],
    'TCA_Cycle': ['IDH1', 'IDH2', 'MDH1', 'MDH2', 'CS', 'ACO2', 'FH', 'SDHA'],
    'NAD_Metabolism': ['NAMPT', 'NMNAT1', 'NMNAT2', 'SIRT1', 'SIRT3', 'PARP1'],
    'Fatty_Acid': ['CPT1A', 'CPT2', 'ACADM', 'ACADVL', 'HADHA', 'HADHB']
}

found_metabolic = {}
for category, markers in metabolic_markers.items():
    found = df_proteomics[df_proteomics['Gene_Symbol'].isin(markers)]['Gene_Symbol'].unique().tolist()
    found_metabolic[category] = found
    print(f"\n{category}: {len(found)}/{len(markers)} found")
    if found:
        print(f"  Found: {', '.join(found)}")
    else:
        print(f"  ❌ None present in ECM dataset")

total_found = sum(len(v) for v in found_metabolic.values())
total_searched = sum(len(v) for v in metabolic_markers.values())

print(f"\n🎯 RESULT: {total_found}/{total_searched} metabolic proteins found in ECM dataset ({total_found/total_searched*100:.1f}%)")

if total_found > 0:
    print(f"\n✅ Analyzing {total_found} metabolic proxy markers...")

    # Analyze found metabolic markers
    metabolic_genes = [gene for genes in found_metabolic.values() for gene in genes]
    df_metabolic_proteins = df_proteomics[df_proteomics['Gene_Symbol'].isin(metabolic_genes)]

    # Aggregate by tissue
    df_metabolic_tissue = df_metabolic_proteins.groupby('Tissue').agg({
        'Zscore_Delta': 'mean',
        'Gene_Symbol': 'count'
    }).rename(columns={'Gene_Symbol': 'N_Metabolic_Proteins'})

    # Merge with velocities
    df_metabolic_velocity = df_velocity.merge(df_metabolic_tissue, left_on='Tissue', right_index=True, how='left')

    # Correlation with velocity
    rho, p_val = spearmanr(df_metabolic_velocity.dropna()['Velocity'], df_metabolic_velocity.dropna()['Zscore_Delta'])
    print(f"\n📊 Metabolic protein expression vs velocity:")
    print(f"   Spearman ρ = {rho:.3f}, p = {p_val:.4f}")
else:
    print(f"\n❌ CRITICAL LIMITATION: ECM dataset contains NO metabolic proteins!")
    print(f"   This confirms H19's core premise: proteomics cannot see Phase I metabolic changes")

# ============================================================================
# PART 3: Literature-Based Metabolic Signature Simulation
# ============================================================================

print("\n\n📊 PART 3: Simulating Expected Metabolic Signatures (Literature-Based)...")
print("-" * 90)
print("📚 References:")
print("  - NAD+ depletion ~40% in aging: Rajman et al. 2018 Cell Metabolism")
print("  - ATP decline ~30% in aging: López-Otín et al. 2023 Cell")
print("  - Lactate/pyruvate ratio ↑2× : Stout et al. 2017 Aging Cell")

# Create simulated metabolomics based on published aging signatures
metabolomics_sim = []

for _, row in df_velocity.iterrows():
    tissue = row['Tissue']
    velocity = row['Velocity']
    phase = row['Phase']

    # Expected pattern from literature:
    # - Metabolic dysfunction happens EARLY (Phase I) but is invisible to ECM proteomics
    # - Should show ATP↓, NAD↓, Lactate↑ in Phase I
    # - By Phase II, metabolic problems plateau, mechanical dominates

    # Simulate Phase I having WORSE metabolic state than Phase II
    # (This is the hypothesis we're testing)

    if phase == 'Phase I (v<1.65)':
        # Phase I: High metabolic dysfunction
        atp_mult = np.random.normal(0.70, 0.05)  # 30% depletion
        nad_mult = np.random.normal(0.60, 0.06)  # 40% depletion
        lactate_mult = np.random.normal(2.00, 0.15)  # 2× increase
        pyruvate_mult = np.random.normal(0.80, 0.08)  # 20% decrease
    elif phase == 'Transition (1.65-2.17)':
        # Transition: Intermediate
        atp_mult = np.random.normal(0.80, 0.05)
        nad_mult = np.random.normal(0.70, 0.06)
        lactate_mult = np.random.normal(1.50, 0.12)
        pyruvate_mult = np.random.normal(0.85, 0.07)
    else:  # Phase II
        # Phase II: Metabolic problems stabilized, mechanical dominant
        atp_mult = np.random.normal(0.85, 0.05)  # Less depleted
        nad_mult = np.random.normal(0.75, 0.06)
        lactate_mult = np.random.normal(1.30, 0.10)
        pyruvate_mult = np.random.normal(0.90, 0.06)

    metabolomics_sim.append({
        'Tissue': tissue,
        'Velocity': velocity,
        'Phase': phase,
        'ATP': 100 * atp_mult,
        'NAD': 100 * nad_mult,
        'NADH': 20 * (1 / atp_mult),  # Inverse relationship
        'Lactate': 50 * lactate_mult,
        'Pyruvate': 30 * pyruvate_mult,
        'Lactate_Pyruvate_Ratio': (50 * lactate_mult) / (30 * pyruvate_mult)
    })

df_metabolomics_sim = pd.DataFrame(metabolomics_sim)

# Test Phase I vs Phase II
phase1_sim = df_metabolomics_sim[df_metabolomics_sim['Phase'] == 'Phase I (v<1.65)']
phase2_sim = df_metabolomics_sim[df_metabolomics_sim['Phase'] == 'Phase II (v>2.17)']

print(f"\n🎯 SIMULATED Phase I (n={len(phase1_sim)}) vs Phase II (n={len(phase2_sim)}):")

for metabolite in ['ATP', 'NAD', 'Lactate', 'Lactate_Pyruvate_Ratio']:
    p1_mean = phase1_sim[metabolite].mean()
    p2_mean = phase2_sim[metabolite].mean()
    pct_change = ((p1_mean - p2_mean) / p2_mean) * 100

    t_stat, p_val = ttest_ind(phase1_sim[metabolite], phase2_sim[metabolite])

    print(f"\n{metabolite}:")
    print(f"  Phase I: {p1_mean:.1f}")
    print(f"  Phase II: {p2_mean:.1f}")
    print(f"  Δ = {pct_change:+.1f}%, p = {p_val:.4f}")

# Save simulation
df_metabolomics_sim.to_csv('metabolomics_data/simulated_metabolomics_literature_based_claude_code.csv', index=False)

print(f"\n✅ Simulation demonstrates EXPECTED pattern if metabolomics data were available")

# ============================================================================
# PART 4: Visualizations
# ============================================================================

print("\n\n📊 PART 4: Generating Visualizations...")
print("-" * 90)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Phase I Metabolic Hypothesis: Simulated Metabolomics vs Tissue Velocity', fontsize=16, fontweight='bold')

# Plot 1: ATP vs Velocity (simulated)
ax = axes[0, 0]
colors = {'Phase I (v<1.65)': '#2E86AB', 'Transition (1.65-2.17)': '#A23B72', 'Phase II (v>2.17)': '#F18F01'}
for phase in df_metabolomics_sim['Phase'].unique():
    data = df_metabolomics_sim[df_metabolomics_sim['Phase'] == phase]
    ax.scatter(data['Velocity'], data['ATP'], label=phase, c=colors.get(phase, 'gray'), s=100, alpha=0.7)

ax.axvline(1.65, color='red', linestyle='--', alpha=0.5, label='Phase I threshold')
ax.axvline(2.17, color='orange', linestyle='--', alpha=0.5, label='Phase II threshold')
ax.set_xlabel('Tissue Aging Velocity', fontsize=12)
ax.set_ylabel('ATP (Simulated, AU)', fontsize=12)
ax.set_title('ATP Depletion in Phase I', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: NAD vs Velocity
ax = axes[0, 1]
for phase in df_metabolomics_sim['Phase'].unique():
    data = df_metabolomics_sim[df_metabolomics_sim['Phase'] == phase]
    ax.scatter(data['Velocity'], data['NAD'], label=phase, c=colors.get(phase, 'gray'), s=100, alpha=0.7)

ax.axvline(1.65, color='red', linestyle='--', alpha=0.5)
ax.axvline(2.17, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('Tissue Aging Velocity', fontsize=12)
ax.set_ylabel('NAD+ (Simulated, AU)', fontsize=12)
ax.set_title('NAD+ Depletion in Phase I', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 3: Lactate/Pyruvate Ratio
ax = axes[0, 2]
for phase in df_metabolomics_sim['Phase'].unique():
    data = df_metabolomics_sim[df_metabolomics_sim['Phase'] == phase]
    ax.scatter(data['Velocity'], data['Lactate_Pyruvate_Ratio'], label=phase, c=colors.get(phase, 'gray'), s=100, alpha=0.7)

ax.axvline(1.65, color='red', linestyle='--', alpha=0.5)
ax.axvline(2.17, color='orange', linestyle='--', alpha=0.5)
ax.set_xlabel('Tissue Aging Velocity', fontsize=12)
ax.set_ylabel('Lactate/Pyruvate Ratio', fontsize=12)
ax.set_title('Glycolytic Shift in Phase I', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 4: Box plots - ATP by Phase
ax = axes[1, 0]
phases_order = ['Phase I (v<1.65)', 'Transition (1.65-2.17)', 'Phase II (v>2.17)']
df_metabolomics_sim_sorted = df_metabolomics_sim.copy()
df_metabolomics_sim_sorted['Phase'] = pd.Categorical(df_metabolomics_sim_sorted['Phase'], categories=phases_order, ordered=True)

sns.boxplot(data=df_metabolomics_sim_sorted, x='Phase', y='ATP', ax=ax, palette=colors)
ax.set_xlabel('')
ax.set_ylabel('ATP (Simulated, AU)', fontsize=12)
ax.set_title('ATP: Phase I vs Phase II', fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# Plot 5: Box plots - NAD by Phase
ax = axes[1, 1]
sns.boxplot(data=df_metabolomics_sim_sorted, x='Phase', y='NAD', ax=ax, palette=colors)
ax.set_xlabel('')
ax.set_ylabel('NAD+ (Simulated, AU)', fontsize=12)
ax.set_title('NAD+: Phase I vs Phase II', fontweight='bold')
ax.tick_params(axis='x', rotation=15)

# Plot 6: Box plots - Lactate/Pyruvate by Phase
ax = axes[1, 2]
sns.boxplot(data=df_metabolomics_sim_sorted, x='Phase', y='Lactate_Pyruvate_Ratio', ax=ax, palette=colors)
ax.set_xlabel('')
ax.set_ylabel('Lactate/Pyruvate Ratio', fontsize=12)
ax.set_title('Lactate/Pyruvate: Phase I vs Phase II', fontweight='bold')
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('visualizations_claude_code/phase1_metabolic_signatures_simulated_claude_code.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: visualizations_claude_code/phase1_metabolic_signatures_simulated_claude_code.png")

# ============================================================================
# PART 5: Summary Statistics
# ============================================================================

print("\n\n📊 PART 5: Summary Statistics...")
print("-" * 90)

# Correlation summary
corr_summary = []
for metabolite in ['ATP', 'NAD', 'Lactate', 'Lactate_Pyruvate_Ratio']:
    rho, p_val = spearmanr(df_metabolomics_sim['Velocity'], df_metabolomics_sim[metabolite])
    corr_summary.append({
        'Metabolite': metabolite,
        'Spearman_rho': rho,
        'p_value': p_val,
        'Expected_Direction': 'Negative' if metabolite in ['ATP', 'NAD'] else 'Positive',
        'Hypothesis_Supported': 'Yes' if (rho < 0 and metabolite in ['ATP', 'NAD']) or (rho > 0 and metabolite in ['Lactate', 'Lactate_Pyruvate_Ratio']) else 'No'
    })

df_corr_summary = pd.DataFrame(corr_summary)
df_corr_summary.to_csv('metabolite_velocity_correlation_simulated_claude_code.csv', index=False)

print("\n🔬 Metabolite-Velocity Correlations (Simulated):")
print(df_corr_summary.to_string(index=False))

# Phase comparison
phase_stats = []
for metabolite in ['ATP', 'NAD', 'Lactate_Pyruvate_Ratio']:
    p1_vals = phase1_sim[metabolite]
    p2_vals = phase2_sim[metabolite]

    t_stat, p_val = ttest_ind(p1_vals, p2_vals)
    pct_change = ((p1_vals.mean() - p2_vals.mean()) / p2_vals.mean()) * 100

    phase_stats.append({
        'Metabolite': metabolite,
        'Phase1_Mean': p1_vals.mean(),
        'Phase2_Mean': p2_vals.mean(),
        'Percent_Change': pct_change,
        'p_value': p_val,
        'Cohen_d': (p1_vals.mean() - p2_vals.mean()) / np.sqrt((p1_vals.std()**2 + p2_vals.std()**2) / 2)
    })

df_phase_stats = pd.DataFrame(phase_stats)
df_phase_stats.to_csv('phase1_vs_phase2_simulated_claude_code.csv', index=False)

print("\n\n🔬 Phase I vs Phase II Comparison (Simulated):")
print(df_phase_stats.to_string(index=False))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "="*90)
print("✅ ANALYSIS COMPLETE")
print("="*90)

print(f"\n📁 Outputs Generated:")
print(f"   1. metabolomics_data/simulated_metabolomics_literature_based_claude_code.csv")
print(f"   2. metabolite_velocity_correlation_simulated_claude_code.csv")
print(f"   3. phase1_vs_phase2_simulated_claude_code.csv")
print(f"   4. visualizations_claude_code/phase1_metabolic_signatures_simulated_claude_code.png")

print(f"\n🎯 KEY FINDINGS:")
print(f"   ❌ VALIDATION STATUS: NOT VALIDATED (no real metabolomics data available)")
print(f"   ✅ SIMULATION: Demonstrates expected Phase I signature (ATP↓30%, NAD↓40%, Lactate/Pyr↑2×)")
print(f"   ❌ ECM PROTEOMICS: Contains {total_found}/{total_searched} metabolic proteins ({total_found/total_searched*100:.1f}%)")
print(f"   🎯 CONCLUSION: Phase I hypothesis CANNOT be validated with proteomics alone")

print(f"\n📋 RECOMMENDATIONS:")
print(f"   1. URGENT: Generate tissue-level metabolomics data (LC-MS/GC-MS)")
print(f"   2. Target metabolites: ATP, NAD+, NADH, Lactate, Pyruvate, TCA intermediates")
print(f"   3. Priority tissues: Liver (v=1.02), Muscle (v=1.95), Lung (v=4.29)")
print(f"   4. Validate Phase I metabolic signature in v<1.65 tissues")
print(f"   5. Test NAD+ boosters (NMN, NR) for intervention window validation")

print("\n" + "="*90)
