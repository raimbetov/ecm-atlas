#!/usr/bin/env python3
"""
Percentile Normalization for ECM-Atlas

Purpose: Apply rank-based percentile transformation to remove study-specific
         scaling while preserving protein rank order within studies

Input: ../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
Output: percentile_normalized.csv, percentile_effects.csv, diagnostic plots

Author: Exploratory Batch Correction Analysis
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import time

print("=" * 70)
print("PERCENTILE NORMALIZATION PIPELINE")
print("=" * 70)
print()

# =============================================================================
# 1. SETUP
# =============================================================================

print("Setting up environment...")

# Create output directories
Path("../diagnostics").mkdir(parents=True, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

print("✓ Environment ready\n")

# =============================================================================
# 2. LOAD DATA
# =============================================================================

print("Loading merged ECM dataset...")

data_path = "../data/merged_ecm_aging_long_format.csv"

if not Path(data_path).exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Unique proteins: {df['Protein_ID'].nunique():,}")
print(f"Unique studies: {df['Study_ID'].nunique()}")
print(f"Age groups: {', '.join(df['Age_Group'].unique())}\n")

# =============================================================================
# 3. PERCENTILE TRANSFORMATION
# =============================================================================

print("=" * 70)
print("APPLYING PERCENTILE TRANSFORMATION")
print("=" * 70)
print()

start_time = time.time()

print("Step 1: Within-study percentile ranking...")

# Function to calculate percentile
def calculate_percentile(group):
    """Convert abundances to percentile ranks (0-100) within group."""
    return group.rank(pct=True) * 100

# Apply percentile transformation within each Study_ID
df['Percentile'] = df.groupby('Study_ID')['Abundance'].transform(calculate_percentile)

print(f"✓ Percentile scores calculated")
print(f"  Range: [{df['Percentile'].min():.2f}, {df['Percentile'].max():.2f}]")
print(f"  Mean: {df['Percentile'].mean():.2f}")
print(f"  Median: {df['Percentile'].median():.2f}\n")

# =============================================================================
# 4. CALCULATE AGE EFFECTS
# =============================================================================

print("Step 2: Calculating age-related changes...")

# Calculate Δpercentile = percentile_old - percentile_young per protein per study
age_effects = []

for protein in df['Protein_ID'].unique():
    protein_data = df[df['Protein_ID'] == protein]

    for study in protein_data['Study_ID'].unique():
        study_protein = protein_data[protein_data['Study_ID'] == study]

        old_percentiles = study_protein[study_protein['Age_Group'] == 'Old']['Percentile']
        young_percentiles = study_protein[study_protein['Age_Group'] == 'Young']['Percentile']

        if len(old_percentiles) > 0 and len(young_percentiles) > 0:
            delta_percentile = old_percentiles.mean() - young_percentiles.mean()

            age_effects.append({
                'Protein_ID': protein,
                'Gene_Symbol': study_protein['Gene_Symbol'].iloc[0],
                'Study_ID': study,
                'Delta_Percentile': delta_percentile,
                'N_Old': len(old_percentiles),
                'N_Young': len(young_percentiles)
            })

age_effects_df = pd.DataFrame(age_effects)

print(f"✓ Age effects calculated for {len(age_effects_df):,} protein-study pairs\n")

# =============================================================================
# 5. META-ANALYSIS ACROSS STUDIES
# =============================================================================

print("Step 3: Meta-analysis across studies...")

# Aggregate effects across studies (mean Δpercentile, t-test)
meta_results = []

for protein in age_effects_df['Protein_ID'].unique():
    protein_effects = age_effects_df[age_effects_df['Protein_ID'] == protein]

    if len(protein_effects) >= 3:  # Minimum 3 studies
        mean_delta = protein_effects['Delta_Percentile'].mean()
        std_delta = protein_effects['Delta_Percentile'].std()
        n_studies = len(protein_effects)

        # One-sample t-test: H0: mean_delta = 0
        se = std_delta / np.sqrt(n_studies)
        t_stat = mean_delta / se if se > 0 else 0
        from scipy.stats import t as t_dist
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), n_studies - 1))

        meta_results.append({
            'Protein_ID': protein,
            'Gene_Symbol': protein_effects['Gene_Symbol'].iloc[0],
            'Mean_Delta_Percentile': mean_delta,
            'SD_Delta_Percentile': std_delta,
            'N_Studies': n_studies,
            'T_Statistic': t_stat,
            'P_value': p_value,
            'Significant': p_value < 0.05,
            'Direction': 'Increase' if mean_delta > 0 else 'Decrease'
        })

meta_df = pd.DataFrame(meta_results)

# FDR correction
from statsmodels.stats.multitest import fdrcorrection
_, meta_df['FDR_Pvalue'] = fdrcorrection(meta_df['P_value'])
meta_df['FDR_Significant'] = meta_df['FDR_Pvalue'] < 0.05

print(f"✓ Meta-analysis complete")
print(f"  Proteins tested: {len(meta_df):,}")
print(f"  Significant (p<0.05): {meta_df['Significant'].sum()} ({meta_df['Significant'].mean()*100:.1f}%)")
print(f"  FDR-significant (FDR<0.05): {meta_df['FDR_Significant'].sum()}")
print()

end_time = time.time()
runtime = end_time - start_time

print(f"✓ Percentile normalization completed in {runtime:.1f} seconds\n")

# =============================================================================
# 6. VALIDATION: DRIVER PROTEIN RECOVERY
# =============================================================================

print("=" * 70)
print("VALIDATION: Q1 DRIVER PROTEIN RECOVERY")
print("=" * 70)
print()

# Q1.1.1 driver proteins (from reports)
driver_proteins = ['Col14a1', 'COL14A1', 'TNXB', 'LAMB1', 'PCOLCE']

print("Checking recovery of Q1 driver proteins...\n")

# Find drivers in top proteins
top_proteins = meta_df.nsmallest(20, 'P_value')  # Top 20 by significance
drivers_recovered = []

for driver in driver_proteins:
    driver_in_top = top_proteins[
        (top_proteins['Gene_Symbol'].str.upper() == driver.upper()) |
        (top_proteins['Protein_ID'].str.contains(driver, case=False, na=False))
    ]

    if len(driver_in_top) > 0:
        drivers_recovered.append(driver)
        rank = top_proteins[
            (top_proteins['Gene_Symbol'].str.upper() == driver.upper()) |
            (top_proteins['Protein_ID'].str.contains(driver, case=False, na=False))
        ].index[0]

        protein_data = driver_in_top.iloc[0]
        print(f"✓ {driver}:")
        print(f"    Rank: {list(top_proteins.index).index(rank) + 1}/20")
        print(f"    ΔPercentile: {protein_data['Mean_Delta_Percentile']:+.1f}")
        print(f"    P-value: {protein_data['P_value']:.4f}")
        print(f"    FDR: {protein_data['FDR_Pvalue']:.4f}\n")
    else:
        print(f"✗ {driver}: NOT in top 20\n")

recovery_rate = len(drivers_recovered) / len(driver_proteins) * 100
print(f"Driver recovery rate: {len(drivers_recovered)}/{len(driver_proteins)} ({recovery_rate:.1f}%)")

if recovery_rate >= 66.7:
    print("✓ EXCELLENT - Exceeds baseline (66.7%)\n")
elif recovery_rate >= 50:
    print("✓ GOOD - Comparable to baseline\n")
else:
    print("⚠ POOR - Below expected recovery\n")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("=" * 70)
print("SAVING OUTPUTS")
print("=" * 70)
print()

# 7.1 Save normalized data
output_file = "percentile_normalized.csv"
df.to_csv(output_file, index=False)
print(f"✓ Normalized data saved: {output_file}")
print(f"  Size: {Path(output_file).stat().st_size / 1e6:.2f} MB")

# 7.2 Save age effects
effects_file = "percentile_effects.csv"
meta_df.to_csv(effects_file, index=False)
print(f"✓ Age effects saved: {effects_file}")

# 7.3 Save metadata
metadata = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'runtime_seconds': runtime,
    'n_proteins_tested': len(meta_df),
    'n_significant_p': int(meta_df['Significant'].sum()),
    'n_significant_fdr': int(meta_df['FDR_Significant'].sum()),
    'driver_recovery_rate': recovery_rate,
    'drivers_recovered': drivers_recovered,
    'percentile_range': [float(df['Percentile'].min()), float(df['Percentile'].max())],
    'percentile_mean': float(df['Percentile'].mean())
}

metadata_file = "percentile_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved: {metadata_file}\n")

# =============================================================================
# 8. GENERATE DIAGNOSTIC PLOTS
# =============================================================================

print("Generating diagnostic plots...")

# 8.1 Volcano plot
print("  - Volcano plot...")

plt.figure(figsize=(10, 8))
plt.scatter(
    meta_df['Mean_Delta_Percentile'],
    -np.log10(meta_df['P_value']),
    alpha=0.5,
    c=meta_df['Significant'].map({True: 'red', False: 'gray'}),
    s=20
)

# Highlight drivers
for driver in driver_proteins:
    driver_data = meta_df[
        (meta_df['Gene_Symbol'].str.upper() == driver.upper()) |
        (meta_df['Protein_ID'].str.contains(driver, case=False, na=False))
    ]

    if len(driver_data) > 0:
        plt.scatter(
            driver_data['Mean_Delta_Percentile'],
            -np.log10(driver_data['P_value']),
            c='blue',
            s=100,
            marker='*',
            edgecolors='black',
            linewidths=1,
            label=driver if driver == driver_proteins[0] else ""
        )

        for _, row in driver_data.iterrows():
            plt.text(
                row['Mean_Delta_Percentile'],
                -np.log10(row['P_value']) + 0.2,
                row['Gene_Symbol'],
                fontsize=9,
                ha='center'
            )

plt.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
plt.axvline(0, color='black', linestyle='-', alpha=0.3)

plt.xlabel('Mean ΔPercentile (Old - Young)', fontsize=12)
plt.ylabel('-log₁₀(P-value)', fontsize=12)
plt.title('Percentile Normalization: Age Effects', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../diagnostics/percentile_volcano_plot.png', dpi=100)
plt.close()

print("  ✓ Volcano plot saved")

# 8.2 Top proteins bar plot
print("  - Top proteins plot...")

top20 = meta_df.nsmallest(20, 'P_value').sort_values('Mean_Delta_Percentile')

plt.figure(figsize=(10, 8))
colors = ['red' if x < 0 else 'blue' for x in top20['Mean_Delta_Percentile']]

plt.barh(range(len(top20)), top20['Mean_Delta_Percentile'], color=colors, alpha=0.7)
plt.yticks(range(len(top20)), top20['Gene_Symbol'])
plt.xlabel('Mean ΔPercentile (Old - Young)', fontsize=12)
plt.ylabel('Protein', fontsize=12)
plt.title('Top 20 Proteins by Statistical Significance', fontsize=14)
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../diagnostics/percentile_top20_proteins.png', dpi=100)
plt.close()

print("  ✓ Top 20 proteins plot saved")

# 8.3 Distribution comparison
print("  - Distribution plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Original z-scores
axes[0].hist(df['Z_score'].dropna(), bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', label='Mean')
axes[0].set_xlabel('Z-score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Original Z-Scores (Within-Study)', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Percentile scores
axes[1].hist(df['Percentile'].dropna(), bins=50, alpha=0.7, edgecolor='black', color='green')
axes[1].axvline(50, color='red', linestyle='--', label='Median')
axes[1].set_xlabel('Percentile', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Percentile Normalization', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../diagnostics/percentile_distribution_comparison.png', dpi=100)
plt.close()

print("  ✓ Distribution comparison saved\n")

# =============================================================================
# 9. SUMMARY
# =============================================================================

print("=" * 70)
print("PERCENTILE NORMALIZATION SUMMARY")
print("=" * 70)
print()

print("Input:")
print(f"  - Dataset: {data_path}")
print(f"  - Proteins: {df['Protein_ID'].nunique():,}")
print(f"  - Studies: {df['Study_ID'].nunique()}")
print()

print("Normalization:")
print("  - Method: Within-study percentile ranking (0-100 scale)")
print(f"  - Runtime: {runtime:.1f} seconds")
print()

print("Meta-Analysis Results:")
print(f"  - Proteins tested: {len(meta_df):,}")
print(f"  - Significant (p<0.05): {meta_df['Significant'].sum()} ({meta_df['Significant'].mean()*100:.1f}%)")
print(f"  - FDR-significant (FDR<0.05): {meta_df['FDR_Significant'].sum()}")
print()

print("Driver Recovery:")
print(f"  - Q1 drivers tested: {len(driver_proteins)}")
print(f"  - Recovered in top 20: {len(drivers_recovered)} ({recovery_rate:.1f}%)")
print(f"  - Drivers found: {', '.join(drivers_recovered) if drivers_recovered else 'None'}")
print()

print("Outputs:")
print(f"  - Normalized data: {output_file}")
print(f"  - Age effects: {effects_file}")
print(f"  - Metadata: {metadata_file}")
print("  - Diagnostics: ../diagnostics/")
print()

print("Next Steps:")
print("  1. Review volcano plot: ../diagnostics/percentile_volcano_plot.png")
print("  2. Check top proteins: ../diagnostics/percentile_top20_proteins.png")
print("  3. Compare to ComBat results: combat_correction/combat_corrected.csv")
print("  4. Run comprehensive validation: validation/01_calculate_metrics.py")
print()

print("=" * 70)
print("PERCENTILE NORMALIZATION COMPLETE")
print("=" * 70)
