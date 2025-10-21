#!/usr/bin/env python3
"""
Standardize Abundance Data to Consistent Log2 Scale

Purpose: Fix mixed linear/log2 scale issue by converting all abundances to log2.

Background:
- Current database contains mixed scales (some studies linear, some log2)
- Conditional log-transformation (skewness > 1) created incompatible scales
- This prevents cross-study comparisons and causes poor driver recovery (20%)

Solution:
- Detect scale per abundance value (linear vs log2)
- Standardize all values to log2 scale
- Preserve NaN values (missing data)

Date: 2025-10-18
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path
import time

# Start timer
start_time = time.time()

print("=" * 70)
print("LOG2 SCALE STANDARDIZATION")
print("=" * 70)
print()

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("Loading merged ECM dataset...")
data_path = "../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

if not Path(data_path).exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Unique proteins: {df['Protein_ID'].nunique():,}")
print(f"Unique studies: {df['Study_ID'].nunique()}")
print()

# =============================================================================
# 2. ANALYZE CURRENT SCALE DISTRIBUTION
# =============================================================================

print("=" * 70)
print("CURRENT SCALE ANALYSIS")
print("=" * 70)
print()

# Combine all abundances for distribution analysis
abundances_old = df['Abundance_Old'].dropna()
abundances_young = df['Abundance_Young'].dropna()
all_abundances = pd.concat([abundances_old, abundances_young])

print("Current abundance distribution:")
print(f"  Count: {len(all_abundances):,}")
print(f"  Range: [{all_abundances.min():.2e}, {all_abundances.max():.2e}]")
print(f"  Median: {all_abundances.median():.2f}")
print(f"  Mean: {all_abundances.mean():.2e}")
print(f"  Std: {all_abundances.std():.2e}")
print()

# Categorize by likely scale
def detect_scale(abundance):
    """Detect if abundance is likely in linear or log2 scale."""
    if pd.isna(abundance):
        return 'NaN'
    elif abundance > 100:
        return 'LINEAR'
    elif 10 < abundance <= 100:
        return 'AMBIGUOUS'
    else:
        return 'LOG2'

# Count by scale type
old_scales = df['Abundance_Old'].apply(detect_scale).value_counts()
young_scales = df['Abundance_Young'].apply(detect_scale).value_counts()

print("Scale distribution (Old samples):")
for scale, count in old_scales.items():
    pct = count / len(df) * 100
    print(f"  {scale:12s}: {count:5,} ({pct:5.1f}%)")

print()
print("Scale distribution (Young samples):")
for scale, count in young_scales.items():
    pct = count / len(df) * 100
    print(f"  {scale:12s}: {count:5,} ({pct:5.1f}%)")

print()

# Study-specific analysis
print("Study-specific median abundances:")
print(f"{'Study':<20s} {'Median':>12s} {'Scale':>12s}")
print("-" * 50)

study_scales = {}
for study in sorted(df['Study_ID'].unique()):
    study_data = df[df['Study_ID'] == study]
    study_abundances = pd.concat([
        study_data['Abundance_Old'].dropna(),
        study_data['Abundance_Young'].dropna()
    ])

    if len(study_abundances) > 0:
        median_val = study_abundances.median()
        scale = detect_scale(median_val)
        study_scales[study] = scale
        print(f"{study:<20s} {median_val:>12.2f} {scale:>12s}")

print()

# =============================================================================
# 3. STANDARDIZE TO LOG2 SCALE
# =============================================================================

print("=" * 70)
print("STANDARDIZING TO LOG2 SCALE")
print("=" * 70)
print()

def standardize_to_log2(abundance):
    """
    Convert abundance to consistent log2 scale.

    Logic:
    - If abundance > 100: Definitely linear scale → log2(x + 1)
    - If 10 < abundance ≤ 100: Ambiguous → log2(x + 1) to be safe
    - If abundance ≤ 10: Likely already log2 OR very low → keep as-is
    - If NaN: Keep as NaN (missing data)

    Note: Adding +1 prevents log2(0) = -inf for zero abundances
    """
    if pd.isna(abundance):
        return np.nan
    elif abundance > 100:  # Definitely linear scale
        return np.log2(abundance + 1)
    elif abundance > 10:   # Ambiguous, standardize to be safe
        return np.log2(abundance + 1)
    else:  # Likely already log2 (range 0-10) or very low abundance
        return abundance

print("Applying log2 standardization...")

# Apply to both Old and Young abundances
df['Abundance_Old_log2'] = df['Abundance_Old'].apply(standardize_to_log2)
df['Abundance_Young_log2'] = df['Abundance_Young'].apply(standardize_to_log2)

print("✓ Standardization complete")
print()

# =============================================================================
# 4. VALIDATE STANDARDIZATION
# =============================================================================

print("=" * 70)
print("VALIDATION")
print("=" * 70)
print()

# Get standardized abundances
standardized_old = df['Abundance_Old_log2'].dropna()
standardized_young = df['Abundance_Young_log2'].dropna()
all_standardized = pd.concat([standardized_old, standardized_young])

print("Standardized abundance distribution:")
print(f"  Count: {len(all_standardized):,}")
print(f"  Range: [{all_standardized.min():.2f}, {all_standardized.max():.2f}]")
print(f"  Median: {all_standardized.median():.2f}")
print(f"  Mean: {all_standardized.mean():.2f}")
print(f"  Std: {all_standardized.std():.2f}")
print(f"  Skewness: {stats.skew(all_standardized):.2f}")
print()

# Check if in expected log2 range
expected_min = 0  # log2(1) = 0
expected_max = 40  # log2(1 trillion) ≈ 40

if all_standardized.min() >= expected_min and all_standardized.max() <= expected_max:
    print(f"✓ All values in expected log2 range [{expected_min}, {expected_max}]")
else:
    print(f"⚠ Some values outside expected range [{expected_min}, {expected_max}]")
    print(f"  Min: {all_standardized.min():.2f} (expected ≥{expected_min})")
    print(f"  Max: {all_standardized.max():.2f} (expected ≤{expected_max})")

print()

# Check for reasonable proteomics range (typical log2 LFQ: 15-35)
typical_min = 10
typical_max = 40
in_range = all_standardized[(all_standardized >= typical_min) &
                            (all_standardized <= typical_max)]

print(f"Values in typical LFQ range [{typical_min}, {typical_max}]: "
      f"{len(in_range):,} / {len(all_standardized):,} ({len(in_range)/len(all_standardized)*100:.1f}%)")
print()

# Study-specific validation
print("Study-specific standardized medians:")
print(f"{'Study':<20s} {'Original':>12s} {'Standardized':>12s} {'Status':>12s}")
print("-" * 60)

for study in sorted(df['Study_ID'].unique()):
    study_data = df[df['Study_ID'] == study]

    original_abundances = pd.concat([
        study_data['Abundance_Old'].dropna(),
        study_data['Abundance_Young'].dropna()
    ])

    standardized_abundances = pd.concat([
        study_data['Abundance_Old_log2'].dropna(),
        study_data['Abundance_Young_log2'].dropna()
    ])

    if len(original_abundances) > 0 and len(standardized_abundances) > 0:
        orig_med = original_abundances.median()
        std_med = standardized_abundances.median()

        # Check if transformation was applied
        if orig_med > 100:
            status = "✓ Converted"
        elif abs(orig_med - std_med) < 1:
            status = "✓ Preserved"
        else:
            status = "⚠ Changed"

        print(f"{study:<20s} {orig_med:>12.2f} {std_med:>12.2f} {status:>12s}")

print()

# Check for NaN preservation
original_nan_old = df['Abundance_Old'].isna().sum()
original_nan_young = df['Abundance_Young'].isna().sum()
standardized_nan_old = df['Abundance_Old_log2'].isna().sum()
standardized_nan_young = df['Abundance_Young_log2'].isna().sum()

print("NaN preservation check:")
print(f"  Abundance_Old: {original_nan_old} NaN (original) → {standardized_nan_old} NaN (standardized)")
print(f"  Abundance_Young: {original_nan_young} NaN (original) → {standardized_nan_young} NaN (standardized)")

if original_nan_old == standardized_nan_old and original_nan_young == standardized_nan_young:
    print("  ✓ NaN values correctly preserved")
else:
    print("  ⚠ Warning: NaN count changed during transformation")

print()

# =============================================================================
# 5. SAVE STANDARDIZED DATA
# =============================================================================

print("=" * 70)
print("SAVING STANDARDIZED DATA")
print("=" * 70)
print()

output_path = "../data/merged_ecm_standardized_log2.csv"

# Save full dataset with standardized columns
df.to_csv(output_path, index=False)

file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
print(f"✓ Standardized data saved: {output_path}")
print(f"  Size: {file_size:.2f} MB")
print()

# Save metadata
metadata = {
    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "input_file": data_path,
    "output_file": output_path,
    "n_rows": len(df),
    "n_proteins": df['Protein_ID'].nunique(),
    "n_studies": df['Study_ID'].nunique(),
    "original_scale_distribution": {
        "LINEAR": int(old_scales.get('LINEAR', 0) + young_scales.get('LINEAR', 0)),
        "LOG2": int(old_scales.get('LOG2', 0) + young_scales.get('LOG2', 0)),
        "AMBIGUOUS": int(old_scales.get('AMBIGUOUS', 0) + young_scales.get('AMBIGUOUS', 0)),
        "NaN": int(old_scales.get('NaN', 0) + young_scales.get('NaN', 0))
    },
    "standardized_range": [float(all_standardized.min()), float(all_standardized.max())],
    "standardized_median": float(all_standardized.median()),
    "standardized_mean": float(all_standardized.mean()),
    "standardized_std": float(all_standardized.std()),
    "validation": {
        "in_expected_range": bool(all_standardized.min() >= expected_min and
                                   all_standardized.max() <= expected_max),
        "typical_lfq_percentage": float(len(in_range) / len(all_standardized) * 100),
        "nan_preserved": bool(original_nan_old == standardized_nan_old and
                             original_nan_young == standardized_nan_young)
    }
}

metadata_path = "../data/standardization_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved: {metadata_path}")
print()

# =============================================================================
# 6. SUMMARY
# =============================================================================

runtime = time.time() - start_time

print("=" * 70)
print("STANDARDIZATION SUMMARY")
print("=" * 70)
print()

print("Input:")
print(f"  - Dataset: {data_path}")
print(f"  - Rows: {len(df):,}")
print(f"  - Original scale: Mixed (LINEAR + LOG2)")
print()

print("Transformation:")
print(f"  - Method: Conditional log2 standardization")
print(f"  - Linear scale (>100): log2(x + 1)")
print(f"  - Ambiguous (10-100): log2(x + 1)")
print(f"  - Log2 scale (≤10): preserved")
print(f"  - Runtime: {runtime:.1f} seconds")
print()

print("Output:")
print(f"  - Standardized range: [{all_standardized.min():.2f}, {all_standardized.max():.2f}]")
print(f"  - Median: {all_standardized.median():.2f}")
print(f"  - Typical LFQ range coverage: {len(in_range)/len(all_standardized)*100:.1f}%")
print(f"  - NaN preserved: {'Yes' if metadata['validation']['nan_preserved'] else 'No'}")
print()

print("Files Generated:")
print(f"  - {output_path}")
print(f"  - {metadata_path}")
print()

print("Next Steps:")
print("  1. Transform to long format: python preprocessing/01_transform_to_long_format.py")
print("  2. Re-run percentile: python percentile_normalization/01_apply_percentile.py")
print("  3. Compare driver recovery: original 20% vs standardized")
print()

print("=" * 70)
print("STANDARDIZATION COMPLETE")
print("=" * 70)
