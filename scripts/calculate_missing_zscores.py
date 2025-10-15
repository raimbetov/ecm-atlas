#!/usr/bin/env python3
"""
Calculate z-scores for studies that are missing them
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from pathlib import Path
import shutil
from datetime import datetime

def calculate_zscores_for_study(df, study_id, groupby_col='Tissue'):
    """Calculate z-scores for one study."""

    print(f"\n{'='*70}")
    print(f"Processing: {study_id}")
    print(f"{'='*70}")

    df_study = df[df['Study_ID'] == study_id].copy()
    print(f"Rows: {len(df_study)}")

    # Group by tissue (or other column)
    groups = df_study[groupby_col].unique()
    print(f"Groups: {groups}")

    results = []

    for group in groups:
        df_group = df_study[df_study[groupby_col] == group].copy()
        print(f"\n  Group: {group} ({len(df_group)} rows)")

        # Get abundance values
        young_vals = df_group['Abundance_Young'].dropna()
        old_vals = df_group['Abundance_Old'].dropna()

        if len(young_vals) == 0 or len(old_vals) == 0:
            print(f"    ⚠️ Skipping (no data)")
            df_group['Zscore_Young'] = np.nan
            df_group['Zscore_Old'] = np.nan
            df_group['Zscore_Delta'] = np.nan
            results.append(df_group)
            continue

        # Check if already log2-transformed (values < 30 suggest log2)
        young_max = young_vals.max()
        old_max = old_vals.max()
        already_log2 = (young_max < 30 and old_max < 30)

        if already_log2:
            print(f"    ℹ️  Data already log2-transformed (max: Y={young_max:.1f}, O={old_max:.1f})")
            young_transformed = df_group['Abundance_Young']
            old_transformed = df_group['Abundance_Old']
        else:
            # Check skewness
            skew_y = skew(young_vals)
            skew_o = skew(old_vals)
            print(f"    Skewness: Y={skew_y:.2f}, O={skew_o:.2f}")

            if abs(skew_y) > 1 or abs(skew_o) > 1:
                print(f"    ✅ Applying log2(x+1) transform")
                young_transformed = np.log2(df_group['Abundance_Young'] + 1)
                old_transformed = np.log2(df_group['Abundance_Old'] + 1)
            else:
                print(f"    ℹ️  No transform needed")
                young_transformed = df_group['Abundance_Young']
                old_transformed = df_group['Abundance_Old']

        # Calculate z-scores
        mean_y = young_transformed.mean()
        std_y = young_transformed.std()
        mean_o = old_transformed.mean()
        std_o = old_transformed.std()

        print(f"    Young: μ={mean_y:.2f}, σ={std_y:.2f}")
        print(f"    Old:   μ={mean_o:.2f}, σ={std_o:.2f}")

        df_group['Zscore_Young'] = (young_transformed - mean_y) / std_y
        df_group['Zscore_Old'] = (old_transformed - mean_o) / std_o
        df_group['Zscore_Delta'] = df_group['Zscore_Old'] - df_group['Zscore_Young']

        # Validation
        z_mean_y = df_group['Zscore_Young'].mean()
        z_std_y = df_group['Zscore_Young'].std()
        z_mean_o = df_group['Zscore_Old'].mean()
        z_std_o = df_group['Zscore_Old'].std()

        print(f"    Validation: Y(μ={z_mean_y:.4f}, σ={z_std_y:.4f}), O(μ={z_mean_o:.4f}, σ={z_std_o:.4f})")

        if abs(z_mean_y) < 0.01 and abs(z_std_y - 1) < 0.01:
            print(f"    ✅ Validation passed")
        else:
            print(f"    ⚠️  Validation warning")

        results.append(df_group)

    return pd.concat(results, ignore_index=True)

def main():
    csv_path = "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

    # Backup
    backup_dir = Path("08_merged_ecm_dataset/backups")
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    backup_path = backup_dir / f"merged_ecm_aging_zscore_{timestamp}.csv"
    shutil.copy(csv_path, backup_path)
    print(f"✅ Backup created: {backup_path}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded: {len(df)} rows")

    # Studies to process
    studies_to_process = ['Angelidis_2019', 'Dipali_2023', 'LiDermis_2021']

    for study_id in studies_to_process:
        # Check if already has z-scores
        df_study = df[df['Study_ID'] == study_id]
        has_zscores = df_study['Zscore_Young'].notna().sum()

        if has_zscores > 0:
            print(f"\n⏭️  Skipping {study_id} (already has z-scores)")
            continue

        # Calculate z-scores
        df_updated = calculate_zscores_for_study(df, study_id)

        # Update in main dataframe
        mask = df['Study_ID'] == study_id
        df.loc[mask, 'Zscore_Young'] = df_updated['Zscore_Young'].values
        df.loc[mask, 'Zscore_Old'] = df_updated['Zscore_Old'].values
        df.loc[mask, 'Zscore_Delta'] = df_updated['Zscore_Delta'].values

        print(f"\n✅ Updated {mask.sum()} rows for {study_id}")

    # Save
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"✅ SAVED: {csv_path}")
    print(f"{'='*70}")

    # Summary
    print("\n=== Final Z-Score Status ===")
    for study in df['Study_ID'].unique():
        df_study = df[df['Study_ID'] == study]
        has_z = df_study['Zscore_Young'].notna().sum()
        total = len(df_study)
        print(f"{study}: {has_z}/{total} ({has_z/total*100:.1f}%)")

if __name__ == '__main__':
    main()
