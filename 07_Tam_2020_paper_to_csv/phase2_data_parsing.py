#!/usr/bin/env python3
"""
Phase 2: Data Parsing
Extract and reshape data from Excel to long-format table
"""

import pandas as pd
import numpy as np
import sys

def parse_data():
    """Parse Excel data to long format"""

    print("=" * 80)
    print("PHASE 2: DATA PARSING")
    print("=" * 80)

    excel_file = "../data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx"

    # Step 1: Load and prepare data
    print("\n1. Loading Excel sheets...")
    df_raw = pd.read_excel(excel_file, sheet_name="Raw data", header=1)
    df_metadata = pd.read_excel(excel_file, sheet_name="Sample information", header=1)

    print(f"   Raw data: {df_raw.shape}")
    print(f"   Metadata: {df_metadata.shape}")

    # Strip "T: " prefix from protein identifier column names
    df_raw.columns = [col.replace('T: ', '') for col in df_raw.columns]

    # Extract LFQ intensity columns
    lfq_columns = [col for col in df_raw.columns if col.startswith('LFQ intensity ')]
    id_cols = ['Majority protein IDs', 'Protein names', 'Gene names']

    print(f"\n   Protein ID columns: {id_cols}")
    print(f"   LFQ intensity columns: {len(lfq_columns)}")

    # Step 2: Reshape to long format
    print(f"\n2. Reshaping to long format...")
    df_filtered = df_raw[id_cols + lfq_columns].copy()

    df_long = df_filtered.melt(
        id_vars=id_cols,
        value_vars=lfq_columns,
        var_name='LFQ_Column',
        value_name='Abundance'
    )

    # Extract profile name from LFQ column name
    df_long['Profile_Name'] = df_long['LFQ_Column'].str.replace('LFQ intensity ', '')

    print(f"   Long-format shape: {df_long.shape}")
    expected_rows = len(df_raw) * len(lfq_columns)
    print(f"   Expected: ({expected_rows}, {len(id_cols) + 3})")

    if len(df_long) != expected_rows:
        print(f"   ⚠️  WARNING: Row count mismatch")
    else:
        print(f"   ✅ Row count correct")

    # Step 3: Join with metadata
    print(f"\n3. Joining with sample metadata...")

    # First, let's check the profile name format in metadata
    print(f"   Sample profile names from metadata:")
    print(f"   {df_metadata['Profile names'].head(5).tolist()}")

    print(f"\n   Sample profile names from Raw data:")
    print(f"   {df_long['Profile_Name'].unique()[:5].tolist()}")

    # The metadata HAS "LFQ intensity " prefix in Profile names, so we need to add it back for join
    df_long['Profile_Name_Full'] = 'LFQ intensity ' + df_long['Profile_Name']

    df_long = df_long.merge(
        df_metadata,
        left_on='Profile_Name_Full',
        right_on='Profile names',
        how='left'
    )

    print(f"   After metadata join: {df_long.shape}")

    # Check for unmatched profiles
    unmatched = df_long[df_long['Age-group'].isna()]
    if len(unmatched) > 0:
        unmatched_profiles = unmatched['Profile_Name'].unique()
        print(f"   ⚠️  WARNING: {len(unmatched_profiles)} unique profiles with missing metadata")
        print(f"   Unmatched profiles (first 5): {unmatched_profiles[:5].tolist()}")

        # Show profile names from both sources for debugging
        print(f"\n   Debug: Comparing profile names...")
        raw_profiles = set(df_long['Profile_Name'].unique())
        meta_profiles = set(df_metadata['Profile names'].unique())
        print(f"   Raw profiles count: {len(raw_profiles)}")
        print(f"   Meta profiles count: {len(meta_profiles)}")

        # This shouldn't happen now that we fixed the join
        print(f"   ❌ ERROR: Still {len(unmatched)} rows with missing metadata")
        print(f"   Sample unmatched: {unmatched[['Profile_Name', 'Majority protein IDs']].head(10)}")
        sys.exit(1)

    else:
        print(f"   ✅ All profiles successfully joined with metadata")

    # Step 4: Parse age and compartment
    print(f"\n4. Parsing age and compartment information...")

    # Map age-group to numeric age (aged = old = 59yr)
    age_map = {
        'young': 16,
        'aged': 59,
        'old': 59  # Just in case
    }
    df_long['Age'] = df_long['Age-group'].map(age_map)

    # Map age to age bin
    df_long['Age_Bin'] = df_long['Age-group'].map({
        'young': 'Young',
        'aged': 'Old',
        'old': 'Old'
    })

    # Rename compartment column for clarity
    df_long['Tissue_Compartment'] = df_long['Disc Compartments']

    # Create Sample_ID from Profile_Name
    df_long['Sample_ID'] = df_long['Profile_Name']

    print(f"   Age distribution:")
    print(df_long['Age'].value_counts())
    print(f"\n   Age bin distribution:")
    print(df_long['Age_Bin'].value_counts())
    print(f"\n   Compartment distribution:")
    print(df_long['Tissue_Compartment'].value_counts())

    # Step 5: Add tissue metadata
    print(f"\n5. Creating tissue metadata...")

    # Create Tissue column combining organ and compartment
    # Format: "Intervertebral_disc_NP", "Intervertebral_disc_IAF", "Intervertebral_disc_OAF"
    # Note: We'll handle transition zones (NP/IAF) separately
    df_long['Tissue'] = 'Intervertebral_disc_' + df_long['Tissue_Compartment']

    print(f"   Tissue types:")
    tissue_counts = df_long['Tissue'].value_counts()
    print(tissue_counts)

    # Note: We have transition zones like "NP/IAF" which are spatial regions
    # These will be kept separate initially and handled during aggregation

    # Step 6: Validation
    print(f"\n6. Parsing validation...")
    print(f"   Total rows: {len(df_long)}")
    print(f"   Expected: ~{3157 * 66} (3,157 proteins × 66 profiles)")
    print(f"   Unique proteins: {df_long['Majority protein IDs'].nunique()}")
    print(f"   Expected: ~3,157")
    print(f"   Unique profiles: {df_long['Profile_Name'].nunique()}")
    print(f"   Expected: 66")

    # Check for null abundances
    null_abundance = df_long['Abundance'].isna().sum()
    null_pct = (null_abundance / len(df_long)) * 100
    print(f"   Null abundances: {null_abundance} ({null_pct:.1f}%)")
    print(f"   Note: 40-70% nulls is typical for LFQ proteomics (not all proteins detected in all profiles)")

    # Save intermediate result
    output_file = "Tam_2020_long_format.csv"
    print(f"\n7. Saving long-format data...")
    df_long.to_csv(output_file, index=False)
    print(f"   ✅ Saved: {output_file}")

    # Show summary
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY")
    print("=" * 80)
    print(f"✅ Data parsing completed")
    print(f"   - Long-format rows: {len(df_long)}")
    print(f"   - Unique proteins: {df_long['Majority protein IDs'].nunique()}")
    print(f"   - Unique profiles: {df_long['Profile_Name'].nunique()}")
    print(f"   - Age bins: {df_long['Age_Bin'].unique().tolist()}")
    print(f"   - Compartments: {sorted(df_long['Tissue_Compartment'].unique().tolist())}")
    print(f"   - Null abundances: {null_pct:.1f}%")
    print(f"\n✅ PROCEED to Phase 3: Schema Standardization")
    print("=" * 80)

    return df_long

if __name__ == "__main__":
    df_long = parse_data()
