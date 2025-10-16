#!/usr/bin/env python3
"""
Zero Value Analysis for Angelidis_2019 Dataset
Purpose: Audit impact of converting zeros to NaN
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
SOURCE_FILE = "../../data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx"
SHEET_NAME = "Proteome"
SAMPLE_COLUMNS = ["young_1", "young_2", "young_3", "young_4",
                  "old_1", "old_2", "old_3", "old_4"]

def analyze_zeros():
    """Analyze zero values in source dataset."""

    print("=" * 80)
    print("ZERO VALUE ANALYSIS: Angelidis_2019")
    print("=" * 80)

    # Load source data
    print(f"\n1. Loading source file: {SOURCE_FILE}")
    df = pd.read_excel(SOURCE_FILE, sheet_name=SHEET_NAME)
    print(f"   Loaded: {len(df)} proteins, {len(df.columns)} columns")

    # Verify sample columns exist
    missing = [col for col in SAMPLE_COLUMNS if col not in df.columns]
    if missing:
        print(f"\n   ERROR: Missing columns: {missing}")
        return

    print(f"   All {len(SAMPLE_COLUMNS)} sample columns found")

    # Extract abundance data
    abundance_data = df[SAMPLE_COLUMNS]

    print("\n2. Zero Statistics:")
    print("-" * 80)

    total_values = abundance_data.size
    total_zeros = (abundance_data == 0).sum().sum()
    total_nans = abundance_data.isna().sum().sum()
    total_nonzero = ((abundance_data != 0) & abundance_data.notna()).sum().sum()

    print(f"   Total proteins: {len(df)}")
    print(f"   Total values (8 samples × {len(df)} proteins): {total_values}")
    print(f"   Current NaN values: {total_nans} ({total_nans/total_values*100:.2f}%)")
    print(f"   Zero values: {total_zeros} ({total_zeros/total_values*100:.2f}%)")
    print(f"   Non-zero values: {total_nonzero} ({total_nonzero/total_values*100:.2f}%)")

    print("\n3. Per-Column Zero Analysis:")
    print("-" * 80)

    for col in SAMPLE_COLUMNS:
        zero_count = (df[col] == 0).sum()
        nan_count = df[col].isna().sum()
        nonzero_count = ((df[col] != 0) & df[col].notna()).sum()
        total = len(df)

        print(f"   {col:12s}: Zeros={zero_count:4d} ({zero_count/total*100:5.2f}%), "
              f"NaN={nan_count:4d} ({nan_count/total*100:5.2f}%), "
              f"Non-zero={nonzero_count:4d} ({nonzero_count/total*100:5.2f}%)")

    print("\n4. Technical Replicate Analysis:")
    print("-" * 80)

    # Check young replicates
    young_cols = SAMPLE_COLUMNS[:4]
    old_cols = SAMPLE_COLUMNS[4:]

    # Find proteins with zeros in replicates
    young_data = df[young_cols]
    old_data = df[old_cols]

    # Proteins with at least one zero in young samples
    young_with_zeros = (young_data == 0).any(axis=1).sum()
    # Proteins with at least one zero in old samples
    old_with_zeros = (old_data == 0).any(axis=1).sum()

    print(f"   Proteins with ≥1 zero in young samples: {young_with_zeros} ({young_with_zeros/len(df)*100:.1f}%)")
    print(f"   Proteins with ≥1 zero in old samples: {old_with_zeros} ({old_with_zeros/len(df)*100:.1f}%)")

    # Find examples where zeros affect averaging
    print("\n5. Example Cases (zeros affecting replicate averaging):")
    print("-" * 80)

    examples_found = 0
    for idx, row in df.head(50).iterrows():  # Check first 50 proteins
        young_values = row[young_cols].values
        old_values = row[old_cols].values

        # Check if there are zeros mixed with non-zeros
        young_has_zeros = (young_values == 0).any()
        young_has_nonzeros = ((young_values != 0) & pd.notna(young_values)).any()

        old_has_zeros = (old_values == 0).any()
        old_has_nonzeros = ((old_values != 0) & pd.notna(old_values)).any()

        if (young_has_zeros and young_has_nonzeros) or (old_has_zeros and old_has_nonzeros):
            if examples_found < 3:  # Show first 3 examples
                protein_id = row.get('Protein IDs', 'N/A')
                gene = row.get('Gene names', 'N/A')

                print(f"\n   Example {examples_found + 1}: {protein_id} ({gene})")
                print(f"   Young replicates: {young_values}")
                print(f"   Old replicates: {old_values}")

                # Show what happens with current averaging (0 included)
                young_mean_with_zero = np.nanmean(young_values)
                old_mean_with_zero = np.nanmean(old_values)

                # Show what would happen if 0 -> NaN
                young_values_no_zero = np.where(young_values == 0, np.nan, young_values)
                old_values_no_zero = np.where(old_values == 0, np.nan, old_values)
                young_mean_no_zero = np.nanmean(young_values_no_zero)
                old_mean_no_zero = np.nanmean(old_values_no_zero)

                print(f"   Current mean (0 included): Young={young_mean_with_zero:.2f}, Old={old_mean_with_zero:.2f}")
                print(f"   After 0→NaN conversion: Young={young_mean_no_zero:.2f}, Old={old_mean_no_zero:.2f}")

                examples_found += 1

    if examples_found == 0:
        print("   No examples found in first 50 proteins")

    print("\n6. Impact Assessment:")
    print("-" * 80)

    # Calculate how many proteins will have their averages changed
    proteins_affected = 0
    for idx, row in df.iterrows():
        young_values = row[young_cols].values
        old_values = row[old_cols].values

        young_has_zeros = (young_values == 0).any()
        young_has_nonzeros = ((young_values != 0) & pd.notna(young_values)).any()

        old_has_zeros = (old_values == 0).any()
        old_has_nonzeros = ((old_values != 0) & pd.notna(old_values)).any()

        if (young_has_zeros and young_has_nonzeros) or (old_has_zeros and old_has_nonzeros):
            proteins_affected += 1

    print(f"   Total proteins: {len(df)}")
    print(f"   Proteins with zeros: {(abundance_data == 0).any(axis=1).sum()}")
    print(f"   Proteins affected by 0→NaN (zeros mixed with non-zeros): {proteins_affected}")
    print(f"   Percentage affected: {proteins_affected/len(df)*100:.2f}%")

    # After conversion stats
    print(f"\n   AFTER 0→NaN conversion:")
    print(f"   - Total NaN values will be: {total_nans + total_zeros} ({(total_nans + total_zeros)/total_values*100:.2f}%)")
    print(f"   - Non-zero/non-NaN values: {total_nonzero} ({total_nonzero/total_values*100:.2f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Converting {total_zeros} zeros to NaN will:")
    print(f"  1. Increase missing data from {total_nans/total_values*100:.2f}% to {(total_nans + total_zeros)/total_values*100:.2f}%")
    print(f"  2. Affect averaging for {proteins_affected} proteins ({proteins_affected/len(df)*100:.2f}%)")
    print(f"  3. Exclude true zero measurements from statistical analysis")
    print(f"  4. Change mean abundance calculations for affected proteins")
    print("=" * 80)

if __name__ == '__main__':
    analyze_zeros()
