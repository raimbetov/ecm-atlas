#!/usr/bin/env python3
"""
Z-Score Normalization by Tissue Compartment - Randles 2021

This script performs compartment-specific z-score normalization for kidney ECM proteins,
separating Glomerular and Tubulointerstitial data to preserve biological differences.

Author: Claude Code
Date: 2025-10-12
Project: ECM Atlas
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import skew
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1.0 DATA LOADING & VALIDATION
# ==============================================================================

print("=" * 80)
print("Z-SCORE NORMALIZATION BY TISSUE COMPARTMENT")
print("Study: Randles 2021 (PMID: 34049963)")
print("=" * 80)
print()

# Load wide-format CSV
input_file = "/Users/Kravtsovd/projects/ecm-atlas/05_Randles_paper_to_csv/claude_code/Randles_2021_wide_format.csv"
print(f"Loading data from: {input_file}")

df = pd.read_csv(input_file)

print(f"\nInput data summary:")
print(f"  Total rows: {len(df):,}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Columns: {', '.join(df.columns[:8])}...")
print()

# Validation checks
print("Validating data structure...")
required_columns = ['Tissue_Compartment', 'Abundance_Young', 'Abundance_Old',
                   'Protein_ID', 'Gene_Symbol']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    raise ValueError(f"❌ Missing required columns: {missing_cols}")

print("✅ All required columns present")
print()

# Check compartments
compartment_counts = df['Tissue_Compartment'].value_counts()
print(f"Tissue compartments found:")
for comp, count in compartment_counts.items():
    print(f"  {comp}: {count:,} proteins")
print()

if len(compartment_counts) != 2:
    raise ValueError(f"❌ Expected 2 compartments, found {len(compartment_counts)}")

print("✅ Compartment validation passed")
print()

# ==============================================================================
# 2.0 COMPARTMENT SEPARATION
# ==============================================================================

print("-" * 80)
print("PHASE 2: COMPARTMENT SEPARATION")
print("-" * 80)
print()

# Split into separate dataframes
df_glomerular = df[df['Tissue_Compartment'] == 'Glomerular'].copy()
df_tubulointerstitial = df[df['Tissue_Compartment'] == 'Tubulointerstitial'].copy()

print(f"Split results:")
print(f"  Glomerular: {len(df_glomerular):,} proteins")
print(f"  Tubulointerstitial: {len(df_tubulointerstitial):,} proteins")
print(f"  Total: {len(df_glomerular) + len(df_tubulointerstitial):,} proteins")
print()

# Verify no data loss
assert len(df_glomerular) + len(df_tubulointerstitial) == len(df), \
    "❌ Data loss during compartment split"

print("✅ No data loss during split")
print()

# ==============================================================================
# 3.0 QUALITY CHECKS
# ==============================================================================

def validate_compartment_data(df_compartment, compartment_name):
    """Validate data quality for a single compartment."""

    print(f"Validating {compartment_name} data quality...")

    issues = []

    # Check for nulls
    null_young = df_compartment['Abundance_Young'].isna().sum()
    null_old = df_compartment['Abundance_Old'].isna().sum()

    if null_young > 0:
        issues.append(f"⚠️  {null_young} null Abundance_Young values")
    if null_old > 0:
        issues.append(f"⚠️  {null_old} null Abundance_Old values")

    # Check for zeros
    zero_young = (df_compartment['Abundance_Young'] == 0).sum()
    zero_old = (df_compartment['Abundance_Old'] == 0).sum()

    if zero_young > 0:
        issues.append(f"⚠️  {zero_young} zero Abundance_Young values")
    if zero_old > 0:
        issues.append(f"⚠️  {zero_old} zero Abundance_Old values")

    # Check for negatives
    neg_young = (df_compartment['Abundance_Young'] < 0).sum()
    neg_old = (df_compartment['Abundance_Old'] < 0).sum()

    if neg_young > 0:
        issues.append(f"❌ CRITICAL: {neg_young} negative Abundance_Young values")
    if neg_old > 0:
        issues.append(f"❌ CRITICAL: {neg_old} negative Abundance_Old values")

    if issues:
        print(f"\n{compartment_name} validation issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅ All validation checks passed")

    print()
    return len(issues) == 0

# Validate both compartments
glom_valid = validate_compartment_data(df_glomerular, "Glomerular")
tubu_valid = validate_compartment_data(df_tubulointerstitial, "Tubulointerstitial")

if not (glom_valid and tubu_valid):
    print("⚠️  WARNING: Data quality issues found - proceeding with caution")
    print()

# ==============================================================================
# 4.0 DISTRIBUTION ANALYSIS
# ==============================================================================

print("-" * 80)
print("PHASE 3: DISTRIBUTION ANALYSIS")
print("-" * 80)
print()

def analyze_distribution(df_compartment, compartment_name):
    """
    Check if data is right-skewed and needs log-transformation.
    Rule: If skewness > 1, apply log2-transformation before z-score.
    """

    # Remove zeros and nulls for skewness calculation
    young_values = df_compartment['Abundance_Young'].replace(0, np.nan).dropna()
    old_values = df_compartment['Abundance_Old'].replace(0, np.nan).dropna()

    skew_young = skew(young_values)
    skew_old = skew(old_values)

    print(f"{compartment_name} distribution analysis:")
    print(f"  Abundance_Young skewness: {skew_young:.3f}")
    print(f"  Abundance_Old skewness:   {skew_old:.3f}")

    needs_log_transform = (skew_young > 1) or (skew_old > 1)

    if needs_log_transform:
        print(f"  ✅ Recommendation: Apply log2-transformation (skewness > 1)")
    else:
        print(f"  ℹ️  No log-transformation needed (skewness ≤ 1)")

    print()
    return needs_log_transform

# Check both compartments
glom_needs_log = analyze_distribution(df_glomerular, "Glomerular")
tubu_needs_log = analyze_distribution(df_tubulointerstitial, "Tubulointerstitial")

# ==============================================================================
# 5.0 LOG2-TRANSFORMATION
# ==============================================================================

print("-" * 80)
print("PHASE 4: LOG2-TRANSFORMATION")
print("-" * 80)
print()

def apply_log2_transform(df_compartment, compartment_name, apply_transform=True):
    """
    Apply log2(x + 1) transformation to handle zeros.
    Formula: log2(Abundance + 1)
    """

    print(f"Processing {compartment_name}:")

    if not apply_transform:
        # No transformation - use raw abundances
        df_compartment['Abundance_Young_transformed'] = df_compartment['Abundance_Young']
        df_compartment['Abundance_Old_transformed'] = df_compartment['Abundance_Old']
        print("  ℹ️  Using raw abundances (no transformation)")
    else:
        # Log2-transformation
        df_compartment['Abundance_Young_transformed'] = np.log2(df_compartment['Abundance_Young'] + 1)
        df_compartment['Abundance_Old_transformed'] = np.log2(df_compartment['Abundance_Old'] + 1)
        print("  ✅ Applied log2(x + 1) transformation")

    print()
    return df_compartment

# Apply transformation if needed
df_glomerular = apply_log2_transform(df_glomerular, "Glomerular",
                                     apply_transform=glom_needs_log)
df_tubulointerstitial = apply_log2_transform(df_tubulointerstitial, "Tubulointerstitial",
                                              apply_transform=tubu_needs_log)

# ==============================================================================
# 6.0 Z-SCORE CALCULATION
# ==============================================================================

print("-" * 80)
print("PHASE 5: Z-SCORE CALCULATION")
print("-" * 80)
print()

def calculate_zscore(df_compartment, compartment_name):
    """
    Calculate z-scores separately for Young and Old abundance values.
    Formula: Z = (X - Mean) / StdDev
    """

    # Calculate statistics for Young
    mean_young = df_compartment['Abundance_Young_transformed'].mean()
    std_young = df_compartment['Abundance_Young_transformed'].std()

    # Calculate statistics for Old
    mean_old = df_compartment['Abundance_Old_transformed'].mean()
    std_old = df_compartment['Abundance_Old_transformed'].std()

    print(f"{compartment_name} normalization parameters:")
    print(f"  Young: Mean = {mean_young:.4f}, StdDev = {std_young:.4f}")
    print(f"  Old:   Mean = {mean_old:.4f}, StdDev = {std_old:.4f}")

    # Calculate z-scores
    df_compartment['Zscore_Young'] = (
        (df_compartment['Abundance_Young_transformed'] - mean_young) / std_young
    )
    df_compartment['Zscore_Old'] = (
        (df_compartment['Abundance_Old_transformed'] - mean_old) / std_old
    )

    # Calculate delta (aging signature)
    df_compartment['Zscore_Delta'] = (
        df_compartment['Zscore_Old'] - df_compartment['Zscore_Young']
    )

    # Store normalization parameters
    df_compartment.attrs['norm_params'] = {
        'mean_young': float(mean_young),
        'std_young': float(std_young),
        'mean_old': float(mean_old),
        'std_old': float(std_old)
    }

    print()
    return df_compartment

# Calculate z-scores for both compartments
df_glomerular = calculate_zscore(df_glomerular, "Glomerular")
df_tubulointerstitial = calculate_zscore(df_tubulointerstitial, "Tubulointerstitial")

# ==============================================================================
# 7.0 Z-SCORE VALIDATION
# ==============================================================================

print("-" * 80)
print("PHASE 6: Z-SCORE VALIDATION")
print("-" * 80)
print()

def validate_zscores(df_compartment, compartment_name):
    """Validate that z-scores have mean ≈ 0 and std ≈ 1."""

    print(f"{compartment_name} z-score validation:")

    # Young z-scores
    mean_zy = df_compartment['Zscore_Young'].mean()
    std_zy = df_compartment['Zscore_Young'].std()
    print(f"  Zscore_Young:  Mean = {mean_zy:.6f}, StdDev = {std_zy:.6f}")

    # Old z-scores
    mean_zo = df_compartment['Zscore_Old'].mean()
    std_zo = df_compartment['Zscore_Old'].std()
    print(f"  Zscore_Old:    Mean = {mean_zo:.6f}, StdDev = {std_zo:.6f}")

    # Check criteria
    mean_ok = abs(mean_zy) < 1e-10 and abs(mean_zo) < 1e-10
    std_ok = 0.99 < std_zy < 1.01 and 0.99 < std_zo < 1.01

    if mean_ok and std_ok:
        print(f"  ✅ Validation PASSED (means ≈ 0, stds ≈ 1)")
    else:
        print(f"  ⚠️  Validation WARNING (check tolerance)")

    print()

validate_zscores(df_glomerular, "Glomerular")
validate_zscores(df_tubulointerstitial, "Tubulointerstitial")

# ==============================================================================
# 8.0 TOP CHANGERS ANALYSIS
# ==============================================================================

print("-" * 80)
print("PHASE 7: TOP AGING MARKERS")
print("-" * 80)
print()

def show_top_changers(df_compartment, compartment_name, n=5):
    """Display proteins with largest z-score changes from Young to Old."""

    print(f"{compartment_name} - Top {n} proteins with largest z-score INCREASES (aging):")
    top_increases = df_compartment.nlargest(n, 'Zscore_Delta')[
        ['Gene_Symbol', 'Zscore_Young', 'Zscore_Old', 'Zscore_Delta']
    ]
    print(top_increases.to_string(index=False))
    print()

    print(f"{compartment_name} - Top {n} proteins with largest z-score DECREASES (aging):")
    top_decreases = df_compartment.nsmallest(n, 'Zscore_Delta')[
        ['Gene_Symbol', 'Zscore_Young', 'Zscore_Old', 'Zscore_Delta']
    ]
    print(top_decreases.to_string(index=False))
    print()

show_top_changers(df_glomerular, "Glomerular")
show_top_changers(df_tubulointerstitial, "Tubulointerstitial")

# ==============================================================================
# 9.0 CSV EXPORT
# ==============================================================================

print("-" * 80)
print("PHASE 8: CSV EXPORT")
print("-" * 80)
print()

# Define output columns
output_columns = [
    # Protein identifiers
    'Protein_ID',
    'Protein_Name',
    'Gene_Symbol',

    # Tissue metadata
    'Tissue',
    'Tissue_Compartment',
    'Species',

    # Original abundance values (for reference)
    'Abundance_Young',
    'Abundance_Old',

    # Transformed values
    'Abundance_Young_transformed',
    'Abundance_Old_transformed',

    # Z-scores (main output)
    'Zscore_Young',
    'Zscore_Old',
    'Zscore_Delta',

    # Study metadata
    'Method',
    'Study_ID',

    # Annotation (if present)
    'Canonical_Gene_Symbol',
    'Matrisome_Category',
    'Matrisome_Division',
    'Match_Level',
    'Match_Confidence'
]

# Filter to only existing columns
output_columns = [col for col in output_columns if col in df_glomerular.columns]

# Export Glomerular
output_file_glom = "Randles_2021_Glomerular_zscore.csv"
df_glomerular[output_columns].to_csv(output_file_glom, index=False)
print(f"✅ Exported: {output_file_glom} ({len(df_glomerular):,} proteins)")

# Export Tubulointerstitial
output_file_tubu = "Randles_2021_Tubulointerstitial_zscore.csv"
df_tubulointerstitial[output_columns].to_csv(output_file_tubu, index=False)
print(f"✅ Exported: {output_file_tubu} ({len(df_tubulointerstitial):,} proteins)")

# Show file sizes
import os
glom_size = os.path.getsize(output_file_glom) / (1024**2)
tubu_size = os.path.getsize(output_file_tubu) / (1024**2)
print()
print(f"File sizes:")
print(f"  Glomerular:          {glom_size:.2f} MB")
print(f"  Tubulointerstitial:  {tubu_size:.2f} MB")
print()

# ==============================================================================
# 10.0 VALIDATION REPORT
# ==============================================================================

print("-" * 80)
print("PHASE 9: VALIDATION REPORT GENERATION")
print("-" * 80)
print()

def generate_validation_report(df_glom, df_tubu, glom_log, tubu_log):
    """Generate comprehensive validation report with distribution statistics."""

    report = []
    report.append("# Z-Score Normalization Validation Report")
    report.append("")
    report.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Study:** Randles 2021 (PMID: 34049963)")
    report.append(f"**Dataset:** Kidney ECM proteins by tissue compartment")
    report.append("")
    report.append("---")
    report.append("")

    # Summary statistics
    report.append("## 1. Summary Statistics")
    report.append("")
    report.append("| Compartment | Proteins | Log2-Transform | Young Mean | Young StdDev | Old Mean | Old StdDev |")
    report.append("|-------------|----------|----------------|------------|--------------|----------|------------|")

    for df_comp, name, log_applied in [(df_glom, "Glomerular", glom_log),
                                         (df_tubu, "Tubulointerstitial", tubu_log)]:
        mean_y = df_comp['Zscore_Young'].mean()
        std_y = df_comp['Zscore_Young'].std()
        mean_o = df_comp['Zscore_Old'].mean()
        std_o = df_comp['Zscore_Old'].std()
        log_str = "Yes" if log_applied else "No"
        report.append(f"| {name} | {len(df_comp):,} | {log_str} | {mean_y:.6f} | {std_y:.6f} | {mean_o:.6f} | {std_o:.6f} |")

    report.append("")
    report.append("**Expected:** Z-score means ≈ 0, standard deviations ≈ 1")
    report.append("")

    # Distribution checks
    report.append("## 2. Distribution Validation")
    report.append("")

    for df_comp, name in [(df_glom, "Glomerular"), (df_tubu, "Tubulointerstitial")]:
        report.append(f"### {name}")
        report.append("")

        # Z-score Young distribution
        zmin_y = df_comp['Zscore_Young'].min()
        zmax_y = df_comp['Zscore_Young'].max()
        z_gt2_y = (df_comp['Zscore_Young'].abs() > 2).sum()
        z_gt3_y = (df_comp['Zscore_Young'].abs() > 3).sum()

        report.append(f"**Zscore_Young:**")
        report.append(f"- Range: [{zmin_y:.2f}, {zmax_y:.2f}]")
        report.append(f"- Proteins with |Z| > 2: {z_gt2_y} ({100*z_gt2_y/len(df_comp):.1f}%)")
        report.append(f"- Proteins with |Z| > 3: {z_gt3_y} ({100*z_gt3_y/len(df_comp):.1f}%)")
        report.append("")

        # Z-score Old distribution
        zmin_o = df_comp['Zscore_Old'].min()
        zmax_o = df_comp['Zscore_Old'].max()
        z_gt2_o = (df_comp['Zscore_Old'].abs() > 2).sum()
        z_gt3_o = (df_comp['Zscore_Old'].abs() > 3).sum()

        report.append(f"**Zscore_Old:**")
        report.append(f"- Range: [{zmin_o:.2f}, {zmax_o:.2f}]")
        report.append(f"- Proteins with |Z| > 2: {z_gt2_o} ({100*z_gt2_o/len(df_comp):.1f}%)")
        report.append(f"- Proteins with |Z| > 3: {z_gt3_o} ({100*z_gt3_o/len(df_comp):.1f}%)")
        report.append("")

        # Zscore Delta
        zdelta_mean = df_comp['Zscore_Delta'].mean()
        zdelta_std = df_comp['Zscore_Delta'].std()

        report.append(f"**Zscore_Delta (Old - Young):**")
        report.append(f"- Mean: {zdelta_mean:.3f}")
        report.append(f"- StdDev: {zdelta_std:.3f}")
        report.append("")

    # Known ECM markers check
    report.append("## 3. Known ECM Markers Validation")
    report.append("")

    ecm_markers = ['COL1A1', 'COL1A2', 'COL4A1', 'FN1', 'LAMA1', 'LAMB2', 'LAMC1']

    report.append("| Gene | Compartment | Zscore_Young | Zscore_Old | Zscore_Delta |")
    report.append("|------|-------------|--------------|------------|--------------|")

    for marker in ecm_markers:
        for df_comp, name in [(df_glom, "Glomerular"), (df_tubu, "Tubulointerstitial")]:
            marker_rows = df_comp[df_comp['Gene_Symbol'] == marker]

            if len(marker_rows) > 0:
                z_y = marker_rows.iloc[0]['Zscore_Young']
                z_o = marker_rows.iloc[0]['Zscore_Old']
                z_d = marker_rows.iloc[0]['Zscore_Delta']
                report.append(f"| {marker} | {name} | {z_y:.2f} | {z_o:.2f} | {z_d:.2f} |")

    report.append("")

    # Success criteria validation
    report.append("## 4. Success Criteria Validation")
    report.append("")
    report.append("### Tier 1: Critical (ALL required)")
    report.append("")

    criteria_met = 0
    total_criteria = 0

    # Criterion 1: 2 output files
    total_criteria += 1
    if os.path.exists("Randles_2021_Glomerular_zscore.csv") and \
       os.path.exists("Randles_2021_Tubulointerstitial_zscore.csv"):
        report.append("- ✅ 2 output CSV files created")
        criteria_met += 1
    else:
        report.append("- ❌ 2 output CSV files NOT created")

    # Criterion 2: Row count preservation
    total_criteria += 1
    total_rows = len(df_glom) + len(df_tubu)
    if total_rows == 5220:
        report.append(f"- ✅ Total proteins = {total_rows:,} (preserved)")
        criteria_met += 1
    else:
        report.append(f"- ❌ Total proteins = {total_rows:,} (expected 5,220)")

    # Criterion 3: Z-score means
    total_criteria += 1
    means_ok = all([
        abs(df_glom['Zscore_Young'].mean()) < 1e-8,
        abs(df_glom['Zscore_Old'].mean()) < 1e-8,
        abs(df_tubu['Zscore_Young'].mean()) < 1e-8,
        abs(df_tubu['Zscore_Old'].mean()) < 1e-8
    ])
    if means_ok:
        report.append("- ✅ Z-score means within [-0.01, +0.01]")
        criteria_met += 1
    else:
        report.append("- ⚠️  Z-score means outside tolerance")

    # Criterion 4: Z-score stds
    total_criteria += 1
    stds_ok = all([
        0.99 < df_glom['Zscore_Young'].std() < 1.01,
        0.99 < df_glom['Zscore_Old'].std() < 1.01,
        0.99 < df_tubu['Zscore_Young'].std() < 1.01,
        0.99 < df_tubu['Zscore_Old'].std() < 1.01
    ])
    if stds_ok:
        report.append("- ✅ Z-score standard deviations within [0.99, 1.01]")
        criteria_met += 1
    else:
        report.append("- ⚠️  Z-score standard deviations outside tolerance")

    # Criterion 5: No nulls
    total_criteria += 1
    nulls_glom = df_glom[['Zscore_Young', 'Zscore_Old']].isna().sum().sum()
    nulls_tubu = df_tubu[['Zscore_Young', 'Zscore_Old']].isna().sum().sum()
    if nulls_glom == 0 and nulls_tubu == 0:
        report.append("- ✅ No null values in z-score columns")
        criteria_met += 1
    else:
        report.append(f"- ❌ Found {nulls_glom + nulls_tubu} null values in z-score columns")

    report.append("")
    report.append("### Tier 2: Quality")
    report.append("")

    # Outlier check
    outliers_glom = ((df_glom['Zscore_Young'].abs() > 3).sum() +
                     (df_glom['Zscore_Old'].abs() > 3).sum())
    outliers_tubu = ((df_tubu['Zscore_Young'].abs() > 3).sum() +
                     (df_tubu['Zscore_Old'].abs() > 3).sum())
    total_values = (len(df_glom) + len(df_tubu)) * 2
    outlier_pct = 100 * (outliers_glom + outliers_tubu) / total_values

    report.append(f"- Extreme outliers (|Z| > 3): {outlier_pct:.2f}% of values")
    if outlier_pct < 5:
        report.append("  ✅ Below 5% threshold")
    else:
        report.append("  ⚠️  Above 5% threshold")

    report.append("")

    # Final score
    report.append("## 5. Final Validation Score")
    report.append("")
    report.append(f"**Tier 1 Critical Criteria:** {criteria_met}/{total_criteria} passed")
    report.append("")

    if criteria_met == total_criteria:
        report.append("✅ **VALIDATION PASSED** - All critical criteria met")
    else:
        report.append("⚠️  **VALIDATION WARNING** - Some criteria not met")

    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by zscore_normalization.py*")

    # Save report
    report_text = "\n".join(report)
    with open("zscore_validation_report.md", "w") as f:
        f.write(report_text)

    print("✅ Generated: zscore_validation_report.md")

    return report_text

# Generate validation report
validation_report = generate_validation_report(
    df_glomerular, df_tubulointerstitial,
    glom_needs_log, tubu_needs_log
)

# ==============================================================================
# 11.0 METADATA JSON
# ==============================================================================

print()
print("-" * 80)
print("PHASE 10: METADATA EXPORT")
print("-" * 80)
print()

def generate_metadata(df_glom, df_tubu, glom_log, tubu_log):
    """Generate metadata JSON with normalization parameters."""

    metadata = {
        "dataset_id": "Randles_2021_zscore",
        "normalization_timestamp": pd.Timestamp.now().isoformat(),
        "source_file": "../05_Randles_paper_to_csv/claude_code/Randles_2021_wide_format.csv",
        "method": "Z-score normalization (compartment-specific)",
        "rationale": "Compartment-specific normalization preserves biological differences between Glomerular and Tubulointerstitial tissues",
        "reference": "01_TASK_DATA_STANDARDIZATION.md section 3.0",

        "glomerular": {
            "protein_count": int(len(df_glom)),
            "log2_transformed": bool(glom_log),
            "normalization_parameters": {
                "mean_young": float(df_glom.attrs.get('norm_params', {}).get('mean_young', 0)),
                "std_young": float(df_glom.attrs.get('norm_params', {}).get('std_young', 1)),
                "mean_old": float(df_glom.attrs.get('norm_params', {}).get('mean_old', 0)),
                "std_old": float(df_glom.attrs.get('norm_params', {}).get('std_old', 1))
            },
            "zscore_statistics": {
                "mean_zscore_young": float(df_glom['Zscore_Young'].mean()),
                "std_zscore_young": float(df_glom['Zscore_Young'].std()),
                "mean_zscore_old": float(df_glom['Zscore_Old'].mean()),
                "std_zscore_old": float(df_glom['Zscore_Old'].std())
            },
            "output_file": "Randles_2021_Glomerular_zscore.csv"
        },

        "tubulointerstitial": {
            "protein_count": int(len(df_tubu)),
            "log2_transformed": bool(tubu_log),
            "normalization_parameters": {
                "mean_young": float(df_tubu.attrs.get('norm_params', {}).get('mean_young', 0)),
                "std_young": float(df_tubu.attrs.get('norm_params', {}).get('std_young', 1)),
                "mean_old": float(df_tubu.attrs.get('norm_params', {}).get('mean_old', 0)),
                "std_old": float(df_tubu.attrs.get('norm_params', {}).get('std_old', 1))
            },
            "zscore_statistics": {
                "mean_zscore_young": float(df_tubu['Zscore_Young'].mean()),
                "std_zscore_young": float(df_tubu['Zscore_Young'].std()),
                "mean_zscore_old": float(df_tubu['Zscore_Old'].mean()),
                "std_zscore_old": float(df_tubu['Zscore_Old'].std())
            },
            "output_file": "Randles_2021_Tubulointerstitial_zscore.csv"
        },

        "validation": {
            "expected_zscore_mean": 0.0,
            "expected_zscore_std": 1.0,
            "tolerance_mean": 0.01,
            "tolerance_std": 0.01,
            "reference_document": "01_TASK_DATA_STANDARDIZATION.md section 3.0"
        }
    }

    # Export metadata
    with open("zscore_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Generated: zscore_metadata.json")

    return metadata

# Generate and export metadata
metadata = generate_metadata(df_glomerular, df_tubulointerstitial,
                            glom_needs_log, tubu_needs_log)

# ==============================================================================
# 12.0 COMPLETION SUMMARY
# ==============================================================================

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()
print("Output files generated:")
print("  1. Randles_2021_Glomerular_zscore.csv")
print("  2. Randles_2021_Tubulointerstitial_zscore.csv")
print("  3. zscore_validation_report.md")
print("  4. zscore_metadata.json")
print()
print("✅ Z-score normalization completed successfully")
print()
