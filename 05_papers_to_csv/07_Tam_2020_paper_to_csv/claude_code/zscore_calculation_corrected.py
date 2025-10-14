"""
Z-Score Calculation with Missing Value Handling
================================================

This script calculates compartment-specific z-scores with proper NaN handling.

Key principles:
1. Missing values (NaN) are EXCLUDED from mean/std calculation
2. Missing values are PRESERVED in output (as NaN)
3. Z-scores calculated separately for Young and Old
4. Log2-transformation applied if skewness > 1
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from pathlib import Path

def calculate_zscore_with_nan_handling(df_wide, compartment_name):
    """
    Calculate z-scores with proper missing value handling.

    Parameters:
    -----------
    df_wide : pd.DataFrame
        Wide-format data for one compartment
    compartment_name : str
        Name of the compartment (for logging)

    Returns:
    --------
    pd.DataFrame with added z-score columns
    """

    print(f"\n{'='*60}")
    print(f"Processing: {compartment_name}")
    print(f"{'='*60}")

    # Check missing values
    n_total = len(df_wide)
    n_missing_young = df_wide['Abundance_Young'].isna().sum()
    n_missing_old = df_wide['Abundance_Old'].isna().sum()

    print(f"\nMissing values:")
    print(f"  Abundance_Young: {n_missing_young}/{n_total} ({n_missing_young/n_total*100:.1f}%)")
    print(f"  Abundance_Old: {n_missing_old}/{n_total} ({n_missing_old/n_total*100:.1f}%)")

    # Calculate skewness (excluding NaN)
    skew_young = skew(df_wide['Abundance_Young'].dropna())
    skew_old = skew(df_wide['Abundance_Old'].dropna())

    print(f"\nSkewness (calculated on non-missing values only):")
    print(f"  Young: {skew_young:.3f}")
    print(f"  Old: {skew_old:.3f}")

    # Apply log2 transformation if needed
    needs_log = (skew_young > 1) or (skew_old > 1)

    if needs_log:
        print(f"\n✅ Applying log2(x + 1) transformation")
        # Log2 transform, preserving NaN
        young_values = np.log2(df_wide['Abundance_Young'] + 1)
        old_values = np.log2(df_wide['Abundance_Old'] + 1)
    else:
        print(f"\nℹ️  No log-transformation needed")
        young_values = df_wide['Abundance_Young']
        old_values = df_wide['Abundance_Old']

    # Calculate mean and std (EXCLUDING NaN - this is critical!)
    mean_young = young_values.mean(skipna=True)  # skipna=True by default, but explicit
    std_young = young_values.std(skipna=True)
    mean_old = old_values.mean(skipna=True)
    std_old = old_values.std(skipna=True)

    print(f"\nNormalization parameters (calculated on non-missing values):")
    print(f"  Young: Mean={mean_young:.4f}, StdDev={std_young:.4f} (n={young_values.notna().sum()})")
    print(f"  Old:   Mean={mean_old:.4f}, StdDev={std_old:.4f} (n={old_values.notna().sum()})")

    # Calculate z-scores (NaN values will remain NaN)
    df_wide = df_wide.copy()
    df_wide['Zscore_Young'] = (young_values - mean_young) / std_young
    df_wide['Zscore_Old'] = (old_values - mean_old) / std_old
    df_wide['Zscore_Delta'] = df_wide['Zscore_Old'] - df_wide['Zscore_Young']

    # Validate z-scores (on non-NaN values only)
    z_young_valid = df_wide['Zscore_Young'].dropna()
    z_old_valid = df_wide['Zscore_Old'].dropna()

    z_mean_young = z_young_valid.mean()
    z_std_young = z_young_valid.std()
    z_mean_old = z_old_valid.mean()
    z_std_old = z_old_valid.std()

    print(f"\nZ-score validation (on non-missing values):")
    print(f"  Zscore_Young: Mean={z_mean_young:.6f}, StdDev={z_std_young:.6f} (n={len(z_young_valid)})")
    print(f"  Zscore_Old:   Mean={z_mean_old:.6f}, StdDev={z_std_old:.6f} (n={len(z_old_valid)})")

    # Check validation thresholds
    if abs(z_mean_young) < 0.01 and abs(z_mean_old) < 0.01:
        print(f"  ✅ Z-score means ≈ 0")
    else:
        print(f"  ⚠️ Z-score means deviate from 0")

    if abs(z_std_young - 1.0) < 0.01 and abs(z_std_old - 1.0) < 0.01:
        print(f"  ✅ Z-score standard deviations ≈ 1")
    else:
        print(f"  ⚠️ Z-score standard deviations deviate from 1")

    # Count outliers (|z| > 3)
    outliers_young = (z_young_valid.abs() > 3).sum()
    outliers_old = (z_old_valid.abs() > 3).sum()

    print(f"\nOutliers (|z| > 3):")
    print(f"  Young: {outliers_young} ({outliers_young/len(z_young_valid)*100:.1f}%)")
    print(f"  Old: {outliers_old} ({outliers_old/len(z_old_valid)*100:.1f}%)")

    # Check missing values in z-scores
    n_missing_z_young = df_wide['Zscore_Young'].isna().sum()
    n_missing_z_old = df_wide['Zscore_Old'].isna().sum()

    print(f"\nMissing z-scores (preserved from input):")
    print(f"  Zscore_Young: {n_missing_z_young}/{n_total} ({n_missing_z_young/n_total*100:.1f}%)")
    print(f"  Zscore_Old: {n_missing_z_old}/{n_total} ({n_missing_z_old/n_total*100:.1f}%)")

    return df_wide


def main():
    """
    Recalculate z-scores for Tam 2020 with proper NaN handling.
    """

    base_dir = Path('/Users/Kravtsovd/projects/ecm-atlas/07_Tam_2020_paper_to_csv/claude_code')

    # Load wide-format file
    wide_file = base_dir / 'Tam_2020_wide_format.csv'
    print(f"Loading: {wide_file}")
    df_wide = pd.read_csv(wide_file)

    print(f"\nWide-format loaded: {len(df_wide)} rows")

    # Define output columns
    output_cols = [
        'Protein_ID', 'Protein_Name', 'Gene_Symbol',
        'Tissue', 'Tissue_Compartment', 'Species',
        'Abundance_Young', 'Abundance_Old',
        'Zscore_Young', 'Zscore_Old', 'Zscore_Delta',
        'Method', 'Study_ID',
        'Canonical_Gene_Symbol', 'Matrisome_Category', 'Matrisome_Division',
        'Match_Level', 'Match_Confidence',
        'N_Profiles_Young', 'N_Profiles_Old'
    ]

    # Process each compartment
    compartments = ['NP', 'IAF', 'OAF']

    for comp in compartments:
        # Filter compartment
        df_comp = df_wide[df_wide['Tissue_Compartment'] == comp].copy()

        # Calculate z-scores
        df_comp = calculate_zscore_with_nan_handling(df_comp, comp)

        # Save
        output_file = base_dir / f'Tam_2020_{comp}_zscore.csv'
        df_comp[output_cols].to_csv(output_file, index=False)
        print(f"\n✅ Saved: {output_file}")
        print(f"   Rows: {len(df_comp)}")

    print(f"\n{'='*60}")
    print(f"✅ All compartments processed successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
