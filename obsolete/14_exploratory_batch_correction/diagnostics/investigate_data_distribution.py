#!/usr/bin/env python3
"""
Investigate Data Distribution and Normalization Impact

Purpose: Diagnose whether poor driver recovery in percentile normalization is due to:
1. Using z-scores (already normalized) instead of raw abundances
2. Log-transformation masking biological signal
3. Different skewness handling across studies

Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

# Load both datasets
print("="*70)
print("DATA DISTRIBUTION INVESTIGATION")
print("="*70)
print()

# 1. Load wide format data (has raw abundances AND z-scores)
print("Loading wide format data...")
df_wide = pd.read_csv('../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
print(f"Loaded {len(df_wide):,} rows")
print(f"Columns: {df_wide.columns.tolist()}")
print()

# 2. Load long format data (what we used for percentile)
print("Loading long format data (used in percentile analysis)...")
df_long = pd.read_csv('../data/merged_ecm_aging_long_format.csv')
print(f"Loaded {len(df_long):,} rows")
print(f"Columns: {df_long.columns.tolist()}")
print()

# 3. Check what data the percentile analysis actually used
print("="*70)
print("CRITICAL QUESTION: Did percentile normalization use raw or z-scores?")
print("="*70)
print()

# Check if 'Abundance' in long format is raw or z-score
# Sample a few proteins and compare
sample_proteins = ['TNXB', 'COL14A1', 'LAMB1', 'VTN']

for protein in sample_proteins:
    if protein in df_wide['Gene_Symbol'].values:
        # Get wide format values
        wide_subset = df_wide[df_wide['Gene_Symbol'] == protein].iloc[0]

        # Get long format values
        long_subset = df_long[df_long['Gene_Symbol'] == protein]

        print(f"\n{protein}:")
        print(f"  Wide format - Abundance_Old: {wide_subset.get('Abundance_Old', 'N/A')}")
        print(f"  Wide format - Zscore_Old: {wide_subset.get('Zscore_Old', 'N/A')}")

        if not long_subset.empty:
            old_long = long_subset[long_subset['Age_Group'] == 'Old']
            if not old_long.empty:
                print(f"  Long format - Abundance (Old): {old_long['Abundance'].iloc[0]}")
                print(f"  Long format - Z_score (Old): {old_long.get('Z_score', old_long.get('Zscore_Old', 'N/A')).iloc[0] if 'Z_score' in old_long.columns or 'Zscore_Old' in old_long.columns else 'N/A'}")

print("\n" + "="*70)
print("DISTRIBUTION ANALYSIS")
print("="*70)
print()

# 4. Analyze distribution of raw abundances
print("Analyzing raw abundance distributions...")

# Get all non-null abundances
abundances_old = df_wide['Abundance_Old'].dropna()
abundances_young = df_wide['Abundance_Young'].dropna()
all_abundances = pd.concat([abundances_old, abundances_young])

# Calculate statistics
print(f"\nRaw Abundances (Abundance_Old/Young):")
print(f"  Count: {len(all_abundances):,}")
print(f"  Range: [{all_abundances.min():.2f}, {all_abundances.max():.2f}]")
print(f"  Mean: {all_abundances.mean():.2f}")
print(f"  Median: {all_abundances.median():.2f}")
print(f"  Std: {all_abundances.std():.2f}")
print(f"  Skewness: {stats.skew(all_abundances):.2f}")
print(f"  Kurtosis: {stats.kurtosis(all_abundances):.2f}")

# Test for normality
_, p_shapiro = stats.shapiro(all_abundances.sample(min(5000, len(all_abundances)), random_state=42))
print(f"  Shapiro-Wilk p-value: {p_shapiro:.6f} {'(NORMAL)' if p_shapiro > 0.05 else '(NOT NORMAL)'}")

# 5. Check if abundances are already log-transformed
print(f"\n📊 Are raw abundances already log-transformed?")
print(f"  Typical LFQ range: 10^15 - 10^21 (linear)")
print(f"  Typical log2(LFQ) range: 20-35 (log-transformed)")
print(f"  Observed range: [{all_abundances.min():.2f}, {all_abundances.max():.2f}]")

if all_abundances.min() > 10 and all_abundances.max() < 50:
    print(f"  ✅ Abundances appear to be LOG2-TRANSFORMED already")
    log_transformed = True
else:
    print(f"  ⚠️  Abundances may be in LINEAR scale")
    log_transformed = False

# 6. Analyze Z-score distributions
print(f"\nZ-score Distributions (Zscore_Old/Young):")
zscores_old = df_wide['Zscore_Old'].dropna()
zscores_young = df_wide['Zscore_Young'].dropna()
all_zscores = pd.concat([zscores_old, zscores_young])

print(f"  Count: {len(all_zscores):,}")
print(f"  Range: [{all_zscores.min():.2f}, {all_zscores.max():.2f}]")
print(f"  Mean: {all_zscores.mean():.6f}")
print(f"  Std: {all_zscores.std():.6f}")
print(f"  Skewness: {stats.skew(all_zscores):.2f}")

# 7. Check what percentile normalization actually calculated on
print("\n" + "="*70)
print("PERCENTILE NORMALIZATION INPUT CHECK")
print("="*70)
print()

# Read the transformation metadata
with open('../data/transformation_metadata.json', 'r') as f:
    transform_meta = json.load(f)

print("Data transformation metadata:")
print(f"  Input file: {transform_meta['input_file']}")
print(f"  Original rows (wide): {transform_meta['n_rows_wide']:,}")
print(f"  Output rows (long): {transform_meta['n_rows_long']:,}")
print(f"  Columns created: {', '.join(transform_meta['columns_created'])}")

# 8. Create diagnostic plots
print("\nGenerating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Raw abundance distribution
axes[0, 0].hist(all_abundances, bins=100, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Abundance (raw)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title(f'Raw Abundance Distribution\n(Skewness={stats.skew(all_abundances):.2f})')
axes[0, 0].axvline(all_abundances.mean(), color='red', linestyle='--', label=f'Mean={all_abundances.mean():.1f}')
axes[0, 0].axvline(all_abundances.median(), color='blue', linestyle='--', label=f'Median={all_abundances.median():.1f}')
axes[0, 0].legend()

# Plot 2: Log2 of raw abundances (if not already log)
if not log_transformed:
    log_abundances = np.log2(all_abundances + 1)
    axes[0, 1].hist(log_abundances, bins=100, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('log2(Abundance + 1)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Log2-Transformed Distribution\n(Skewness={stats.skew(log_abundances):.2f})')
else:
    axes[0, 1].text(0.5, 0.5, 'Abundances already\nlog2-transformed',
                    ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
    axes[0, 1].set_title('Log2 Transform (N/A)')

# Plot 3: Z-score distribution
axes[0, 2].hist(all_zscores, bins=100, alpha=0.7, edgecolor='black', color='green')
axes[0, 2].set_xlabel('Z-score')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title(f'Z-score Distribution\n(Mean={all_zscores.mean():.3f}, Std={all_zscores.std():.3f})')
axes[0, 2].axvline(0, color='red', linestyle='--', label='Zero')
axes[0, 2].legend()

# Plot 4: Q-Q plot for raw abundances
stats.probplot(all_abundances.sample(min(1000, len(all_abundances)), random_state=42), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Raw Abundances\nvs Normal Distribution')

# Plot 5: Q-Q plot for log abundances
if not log_transformed:
    stats.probplot(log_abundances.sample(min(1000, len(log_abundances)), random_state=42), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Log2(Abundances)\nvs Normal Distribution')
else:
    stats.probplot(all_abundances.sample(min(1000, len(all_abundances)), random_state=42), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Abundances (already log2)\nvs Normal Distribution')

# Plot 6: Comparison of driver proteins
driver_proteins = ['TNXB', 'COL14A1', 'LAMB1', 'PCOLCE']
driver_abundances = []
driver_zscores = []
driver_labels = []

for protein in driver_proteins:
    if protein in df_wide['Gene_Symbol'].values:
        subset = df_wide[df_wide['Gene_Symbol'] == protein]
        # Get old-young differences
        for _, row in subset.iterrows():
            if pd.notna(row.get('Abundance_Old')) and pd.notna(row.get('Abundance_Young')):
                driver_abundances.append(row['Abundance_Old'] - row['Abundance_Young'])
                driver_zscores.append(row.get('Zscore_Delta', row.get('Zscore_Old', 0) - row.get('Zscore_Young', 0)))
                driver_labels.append(protein)

if driver_abundances:
    axes[1, 2].scatter(driver_abundances, driver_zscores, s=100, alpha=0.6)
    for i, label in enumerate(driver_labels):
        axes[1, 2].annotate(label, (driver_abundances[i], driver_zscores[i]),
                           fontsize=8, alpha=0.7)
    axes[1, 2].set_xlabel('Abundance Delta (Old - Young)')
    axes[1, 2].set_ylabel('Z-score Delta')
    axes[1, 2].set_title('Driver Proteins: Abundance vs Z-score Changes')
    axes[1, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(0, color='gray', linestyle='--', alpha=0.5)
else:
    axes[1, 2].text(0.5, 0.5, 'No driver protein data',
                    ha='center', va='center', transform=axes[1, 2].transAxes)

plt.tight_layout()
plt.savefig('data_distribution_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: data_distribution_diagnostics.png")

# 9. Key finding: What did percentile normalization actually use?
print("\n" + "="*70)
print("KEY FINDING")
print("="*70)
print()

# Check if 'Abundance' column in long format matches raw or z-score
if 'Abundance' in df_long.columns:
    long_abundances = df_long['Abundance'].dropna()

    print("Long format 'Abundance' column statistics:")
    print(f"  Range: [{long_abundances.min():.2f}, {long_abundances.max():.2f}]")
    print(f"  Mean: {long_abundances.mean():.2f}")
    print(f"  Std: {long_abundances.std():.2f}")

    # Compare to wide format
    print("\nComparison to wide format:")
    print(f"  Wide Abundance_Old/Young range: [{all_abundances.min():.2f}, {all_abundances.max():.2f}]")
    print(f"  Wide Zscore_Old/Young range: [{all_zscores.min():.2f}, {all_zscores.max():.2f}]")

    # Determine which one matches
    if abs(long_abundances.mean() - all_abundances.mean()) < 5:
        print(f"\n✅ Long format 'Abundance' = RAW ABUNDANCES (log2-transformed LFQ)")
        print(f"   Percentile normalization was calculated on RAW data ✓")
        used_raw = True
    elif abs(long_abundances.mean() - all_zscores.mean()) < 0.5:
        print(f"\n⚠️  Long format 'Abundance' = Z-SCORES")
        print(f"   Percentile normalization was calculated on ALREADY-NORMALIZED data ✗")
        print(f"   THIS IS THE PROBLEM!")
        used_raw = False
    else:
        print(f"\n❓ Long format 'Abundance' source unclear")
        used_raw = None

# 10. Generate summary report
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)
print()

summary = {
    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_characteristics": {
        "raw_abundances_log_transformed": log_transformed,
        "raw_abundance_range": [float(all_abundances.min()), float(all_abundances.max())],
        "raw_abundance_skewness": float(stats.skew(all_abundances)),
        "raw_abundance_normal_test_pvalue": float(p_shapiro),
        "zscore_mean": float(all_zscores.mean()),
        "zscore_std": float(all_zscores.std())
    },
    "percentile_input": {
        "used_raw_abundances": used_raw,
        "long_format_range": [float(long_abundances.min()), float(long_abundances.max())],
        "long_format_mean": float(long_abundances.mean())
    }
}

# Check if the issue is using z-scores instead of raw
if not used_raw:
    print("🔴 CRITICAL ISSUE IDENTIFIED:")
    print("   Percentile normalization was applied to Z-SCORES, not RAW abundances!")
    print()
    print("Why this is a problem:")
    print("  1. Z-scores are already normalized (mean=0, std=1 within each study-compartment)")
    print("  2. Converting z-scores to percentiles DOUBLE-NORMALIZES the data")
    print("  3. This removes biological signal and explains poor driver recovery (20%)")
    print()
    print("Solution:")
    print("  Re-run percentile normalization on RAW abundance values (Abundance_Old/Young)")
    print("  NOT on pre-calculated z-scores (Zscore_Old/Young)")

    summary["diagnosis"] = "DOUBLE_NORMALIZATION"
    summary["recommendation"] = "Re-run percentile on raw abundances"
elif used_raw:
    print("✅ Percentile normalization correctly used RAW abundances")
    print()
    print("Then why did driver recovery fail?")
    print("  Possible reasons:")
    print("  1. Raw abundances already log2-transformed before z-score calculation")
    print("  2. Log transformation reduces dynamic range, making percentile ranks similar")
    print("  3. Different studies have different protein coverage → incomparable ranks")
    print("  4. Statistical power insufficient (N=3-5 studies per protein)")

    summary["diagnosis"] = "LOW_POWER_OR_LOG_TRANSFORM_ISSUE"
    summary["recommendation"] = "Try percentile on LINEAR scale abundances OR accept low cross-study power"

# Save summary
with open('distribution_diagnosis.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Diagnosis saved: distribution_diagnosis.json")
print("\n" + "="*70)
print("INVESTIGATION COMPLETE")
print("="*70)
