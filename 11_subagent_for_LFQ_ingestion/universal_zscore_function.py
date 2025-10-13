#!/usr/bin/env python3
"""
Universal Z-Score Calculation Function
=======================================

Calculate compartment-specific z-scores for newly added study in unified CSV.

Usage:
    python universal_zscore_function.py <study_id> <groupby_column1> [groupby_column2 ...]

Examples:
    python universal_zscore_function.py Randles_2021 Tissue
    python universal_zscore_function.py Tam_2020 Tissue
    python universal_zscore_function.py MultiAge_2024 Tissue Age_Category
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from datetime import datetime
from pathlib import Path
import shutil
import json


def calculate_study_zscores(
    study_id: str,
    groupby_columns: list,
    csv_path: str = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/ECM_Atlas_Unified.csv',
    backup: bool = True
):
    """
    Calculate z-scores for ONE study in unified CSV.

    Parameters:
    -----------
    study_id : str
        Study to process (e.g., 'Randles_2021', 'Tam_2020')
    groupby_columns : list
        Columns to group by for z-score calculation
        Examples:
        - ['Tissue'] → most common (Kidney_Glomerular, Kidney_Tubulointerstitial)
        - ['Tissue_Compartment'] → compartment only (Glomerular, Tubulointerstitial)
        - ['Matrisome_Category'] → by ECM category (Collagens, Proteoglycans, etc.)
        - ['Tissue', 'Age_Category'] → for multi-age studies
    csv_path : str
        Path to unified CSV file
    backup : bool
        Create backup before updating (default: True)

    Returns:
    --------
    pd.DataFrame: Updated unified dataframe
    dict: Metadata about z-score calculation

    Example Usage:
    --------------
    # Simple case: Randles 2021 with 2 compartments
    calculate_study_zscores(
        study_id='Randles_2021',
        groupby_columns=['Tissue']
    )
    # Groups: Kidney_Glomerular, Kidney_Tubulointerstitial

    # Simple case: Tam 2020 with 3 compartments
    calculate_study_zscores(
        study_id='Tam_2020',
        groupby_columns=['Tissue']
    )
    # Groups: Intervertebral_disc_NP, Intervertebral_disc_IAF, Intervertebral_disc_OAF

    # Complex case: Multi-age study
    calculate_study_zscores(
        study_id='MultiAge_2024',
        groupby_columns=['Tissue', 'Age_Category']
    )
    # Groups: (Lung, Young), (Lung, Middle), (Lung, Old)
    """

    print(f"\n{'='*70}")
    print(f"Z-SCORE CALCULATION FOR: {study_id}")
    print(f"Grouping by: {groupby_columns}")
    print(f"{'='*70}")

    # === STEP 1: Load and filter ===
    print(f"\nStep 1: Loading unified dataset...")
    df_unified = pd.read_csv(csv_path)
    print(f"✅ Loaded: {len(df_unified)} total rows")
    print(f"   Studies present: {df_unified['Study_ID'].unique().tolist()}")

    df_study = df_unified[df_unified['Study_ID'] == study_id].copy()

    if len(df_study) == 0:
        raise ValueError(f"❌ Study '{study_id}' not found in unified CSV!")

    print(f"✅ Found {len(df_study)} rows for study '{study_id}'")

    # === STEP 2: Initialize z-score columns ===
    for col in ['Zscore_Young', 'Zscore_Old', 'Zscore_Delta']:
        if col not in df_study.columns:
            df_study[col] = np.nan

    # === STEP 3: Group and process ===
    print(f"\nStep 2: Grouping by {groupby_columns}...")

    # Validate groupby columns exist
    for col in groupby_columns:
        if col not in df_study.columns:
            raise ValueError(f"❌ Column '{col}' not found in dataset!")

    grouped = df_study.groupby(groupby_columns, dropna=False)
    print(f"✅ Created {len(grouped)} groups")

    results = []
    metadata = {}

    # === STEP 4: Process each group ===
    for group_key, df_group in grouped:
        # Format group name
        group_name = group_key if isinstance(group_key, str) else '_'.join(map(str, group_key))

        print(f"\n{'─'*70}")
        print(f"Group: {group_name}")
        print(f"{'─'*70}")
        print(f"  Rows: {len(df_group)}")

        # Copy to avoid warnings
        df_group = df_group.copy()

        # --- Substep 4.1: Count missing values ---
        n_missing_young = df_group['Abundance_Young'].isna().sum()
        n_missing_old = df_group['Abundance_Old'].isna().sum()

        pct_missing_young = n_missing_young / len(df_group) * 100
        pct_missing_old = n_missing_old / len(df_group) * 100

        print(f"  Missing values:")
        print(f"    Abundance_Young: {n_missing_young}/{len(df_group)} ({pct_missing_young:.1f}%)")
        print(f"    Abundance_Old: {n_missing_old}/{len(df_group)} ({pct_missing_old:.1f}%)")

        # --- Substep 4.2: Calculate skewness (on non-NaN) ---
        young_notna = df_group['Abundance_Young'].notna().sum()
        old_notna = df_group['Abundance_Old'].notna().sum()

        skew_young = skew(df_group['Abundance_Young'].dropna()) if young_notna > 0 else 0
        skew_old = skew(df_group['Abundance_Old'].dropna()) if old_notna > 0 else 0

        print(f"  Skewness (non-NaN values only):")
        print(f"    Young: {skew_young:.3f} (n={young_notna})")
        print(f"    Old: {skew_old:.3f} (n={old_notna})")

        # --- Substep 4.3: Transform if needed ---
        needs_log = (abs(skew_young) > 1) or (abs(skew_old) > 1)

        if needs_log:
            print(f"  ✅ Applying log2(x + 1) transformation")
            young_values = np.log2(df_group['Abundance_Young'] + 1)  # NaN preserved
            old_values = np.log2(df_group['Abundance_Old'] + 1)      # NaN preserved
        else:
            print(f"  ℹ️  No log-transformation needed")
            young_values = df_group['Abundance_Young']
            old_values = df_group['Abundance_Old']

        # --- Substep 4.4: Calculate mean and std (EXCLUDING NaN) ---
        mean_young = young_values.mean()  # skipna=True by default in pandas
        std_young = young_values.std()
        mean_old = old_values.mean()
        std_old = old_values.std()

        print(f"  Normalization parameters (non-NaN values):")
        print(f"    Young: μ={mean_young:.4f}, σ={std_young:.4f}")
        print(f"    Old:   μ={mean_old:.4f}, σ={std_old:.4f}")

        # --- Substep 4.5: Calculate z-scores (NaN → NaN) ---
        df_group['Zscore_Young'] = (young_values - mean_young) / std_young
        df_group['Zscore_Old'] = (old_values - mean_old) / std_old
        df_group['Zscore_Delta'] = df_group['Zscore_Old'] - df_group['Zscore_Young']

        # --- Substep 4.6: Validate (on non-NaN) ---
        z_mean_young = df_group['Zscore_Young'].mean()
        z_std_young = df_group['Zscore_Young'].std()
        z_mean_old = df_group['Zscore_Old'].mean()
        z_std_old = df_group['Zscore_Old'].std()

        print(f"  Z-score validation (non-NaN values):")
        print(f"    Zscore_Young: μ={z_mean_young:.6f}, σ={z_std_young:.6f}")
        print(f"    Zscore_Old:   μ={z_mean_old:.6f}, σ={z_std_old:.6f}")

        # Check validation thresholds
        valid_mean = abs(z_mean_young) < 0.01 and abs(z_mean_old) < 0.01
        valid_std = abs(z_std_young - 1.0) < 0.01 and abs(z_std_old - 1.0) < 0.01

        if valid_mean and valid_std:
            print(f"  ✅ Validation PASSED (μ ≈ 0, σ ≈ 1)")
        else:
            print(f"  ⚠️  Validation WARNING (check parameters)")

        # --- Substep 4.7: Count outliers ---
        z_young_valid = df_group['Zscore_Young'].dropna()
        z_old_valid = df_group['Zscore_Old'].dropna()

        outliers_young = (z_young_valid.abs() > 3).sum() if len(z_young_valid) > 0 else 0
        outliers_old = (z_old_valid.abs() > 3).sum() if len(z_old_valid) > 0 else 0

        pct_outliers_young = outliers_young / len(z_young_valid) * 100 if len(z_young_valid) > 0 else 0
        pct_outliers_old = outliers_old / len(z_old_valid) * 100 if len(z_old_valid) > 0 else 0

        print(f"  Outliers (|z| > 3):")
        print(f"    Young: {outliers_young} ({pct_outliers_young:.1f}%)")
        print(f"    Old: {outliers_old} ({pct_outliers_old:.1f}%)")

        # --- Substep 4.8: Store metadata ---
        metadata[group_name] = {
            'n_rows': len(df_group),
            'missing_young_%': round(pct_missing_young, 1),
            'missing_old_%': round(pct_missing_old, 1),
            'skew_young': round(skew_young, 3),
            'skew_old': round(skew_old, 3),
            'log2_transformed': bool(needs_log),
            'mean_young': round(float(mean_young), 4),
            'std_young': round(float(std_young), 4),
            'mean_old': round(float(mean_old), 4),
            'std_old': round(float(std_old), 4),
            'z_mean_young': round(float(z_mean_young), 6),
            'z_std_young': round(float(z_std_young), 6),
            'z_mean_old': round(float(z_mean_old), 6),
            'z_std_old': round(float(z_std_old), 6),
            'outliers_young': int(outliers_young),
            'outliers_old': int(outliers_old),
            'validation_passed': bool(valid_mean and valid_std)
        }

        results.append(df_group)

    # === STEP 5: Combine results ===
    df_study_with_zscores = pd.concat(results, ignore_index=True)

    print(f"\n{'='*70}")
    print(f"✅ Processed {len(grouped)} groups")
    print(f"{'='*70}")

    # === STEP 6: Backup original unified CSV ===
    if backup:
        csv_dir = Path(csv_path).parent
        backup_dir = csv_dir / 'backups'
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        backup_path = backup_dir / f"ECM_Atlas_Unified_{timestamp}.csv"
        shutil.copy(csv_path, backup_path)
        print(f"\n✅ Backup created: {backup_path}")

    # === STEP 7: Update unified CSV ===
    print(f"\nStep 3: Updating unified CSV...")

    # Find rows for this study
    mask = df_unified['Study_ID'] == study_id

    # Initialize z-score columns in unified if they don't exist
    for col in ['Zscore_Young', 'Zscore_Old', 'Zscore_Delta']:
        if col not in df_unified.columns:
            df_unified[col] = np.nan

    # Update ONLY the target study rows
    df_unified.loc[mask, 'Zscore_Young'] = df_study_with_zscores['Zscore_Young'].values
    df_unified.loc[mask, 'Zscore_Old'] = df_study_with_zscores['Zscore_Old'].values
    df_unified.loc[mask, 'Zscore_Delta'] = df_study_with_zscores['Zscore_Delta'].values

    print(f"✅ Updated {mask.sum()} rows for study '{study_id}'")
    print(f"   Other studies remain unchanged")

    # Save updated CSV
    df_unified.to_csv(csv_path, index=False)
    print(f"✅ Saved updated unified CSV: {csv_path}")

    # === STEP 8: Save metadata ===
    csv_dir = Path(csv_path).parent
    metadata_path = csv_dir / f"zscore_metadata_{study_id}.json"

    full_metadata = {
        'study_id': study_id,
        'groupby_columns': groupby_columns,
        'timestamp': datetime.now().isoformat(),
        'n_groups': len(grouped),
        'total_rows_processed': len(df_study_with_zscores),
        'groups': metadata
    }

    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)

    print(f"✅ Saved metadata: {metadata_path}")

    print(f"\n{'='*70}")
    print(f"✅ Z-SCORE CALCULATION COMPLETE")
    print(f"{'='*70}")

    return df_unified, metadata


if __name__ == '__main__':
    import sys

    # Command-line usage
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nError: Missing required arguments")
        print("\nUsage:")
        print("  python universal_zscore_function.py <study_id> <groupby_column1> [groupby_column2 ...]")
        print("\nExamples:")
        print("  python universal_zscore_function.py Randles_2021 Tissue")
        print("  python universal_zscore_function.py Tam_2020 Tissue")
        print("  python universal_zscore_function.py MultiAge_2024 Tissue Age_Category")
        sys.exit(1)

    study_id = sys.argv[1]
    groupby_columns = sys.argv[2:]

    print(f"Running z-score calculation...")
    print(f"Study ID: {study_id}")
    print(f"Groupby columns: {groupby_columns}")

    df_updated, metadata = calculate_study_zscores(
        study_id=study_id,
        groupby_columns=groupby_columns
    )

    print("\n✅ Done!")
    print(f"\nNext steps:")
    print(f"1. Check unified CSV: 08_merged_ecm_dataset/ECM_Atlas_Unified.csv")
    print(f"2. Review metadata: 08_merged_ecm_dataset/zscore_metadata_{study_id}.json")
    print(f"3. Verify backup created: 08_merged_ecm_dataset/backups/")
