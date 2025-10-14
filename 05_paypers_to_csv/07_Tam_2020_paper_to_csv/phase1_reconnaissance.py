#!/usr/bin/env python3
"""
Phase 1: File Reconnaissance
Validate data availability and structure before parsing
"""

import pandas as pd
import sys

def reconnaissance():
    """Perform file reconnaissance checks"""

    print("=" * 80)
    print("PHASE 1: FILE RECONNAISSANCE")
    print("=" * 80)

    excel_file = "../data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx"

    # Load Excel file
    print(f"\n1. Loading Excel file: {excel_file}")
    try:
        xl = pd.ExcelFile(excel_file)
        print(f"   ✅ File loaded successfully")
    except Exception as e:
        print(f"   ❌ ERROR: Could not load file: {e}")
        sys.exit(1)

    # Check sheet names
    print(f"\n2. Sheet names: {xl.sheet_names}")
    required_sheets = ['Raw data', 'Sample information']
    for sheet in required_sheets:
        if sheet in xl.sheet_names:
            print(f"   ✅ '{sheet}' sheet found")
        else:
            print(f"   ❌ ERROR: '{sheet}' sheet missing")
            sys.exit(1)

    # Load Raw data sheet
    print(f"\n3. Loading 'Raw data' sheet...")
    df_raw = pd.read_excel(excel_file, sheet_name="Raw data", header=1)
    print(f"   Shape: {df_raw.shape}")
    print(f"   Expected: (~3158, ~80)")

    if df_raw.shape[0] < 3000 or df_raw.shape[1] < 70:
        print(f"   ⚠️  WARNING: Dimensions differ from expected")
    else:
        print(f"   ✅ Dimensions look correct")

    # Check columns
    print(f"\n4. Checking column structure...")
    print(f"   First 10 columns: {list(df_raw.columns[:10])}")

    # Check for protein identifier columns
    id_columns = ['T: Majority protein IDs', 'T: Protein names', 'T: Gene names']
    for col in id_columns:
        if col in df_raw.columns:
            print(f"   ✅ Found: '{col}'")
        else:
            print(f"   ❌ ERROR: Missing column '{col}'")
            sys.exit(1)

    # Check for LFQ intensity columns
    lfq_columns = [col for col in df_raw.columns if col.startswith('LFQ intensity ')]
    print(f"\n5. LFQ intensity columns: {len(lfq_columns)}")
    print(f"   Expected: 66")
    if len(lfq_columns) != 66:
        print(f"   ⚠️  WARNING: Expected 66 LFQ columns, found {len(lfq_columns)}")
    else:
        print(f"   ✅ Correct number of LFQ columns")

    # Sample a few LFQ column names
    print(f"   Sample LFQ columns (first 5):")
    for col in lfq_columns[:5]:
        print(f"     - {col}")

    # Load Sample information sheet
    print(f"\n6. Loading 'Sample information' sheet...")
    df_metadata = pd.read_excel(excel_file, sheet_name="Sample information", header=1)
    print(f"   Shape: {df_metadata.shape}")
    print(f"   Expected: (66, ~8)")

    if df_metadata.shape[0] != 66:
        print(f"   ⚠️  WARNING: Expected 66 profiles, found {df_metadata.shape[0]}")
    else:
        print(f"   ✅ Correct number of profiles")

    print(f"\n7. Sample information columns:")
    print(f"   {list(df_metadata.columns)}")

    # Check key columns
    required_metadata_cols = ['Profile names', 'Age-group', 'Disc Compartments']
    for col in required_metadata_cols:
        if col in df_metadata.columns:
            print(f"   ✅ Found: '{col}'")
        else:
            print(f"   ❌ ERROR: Missing column '{col}'")
            sys.exit(1)

    # Check age groups
    print(f"\n8. Age group distribution:")
    age_counts = df_metadata['Age-group'].value_counts()
    print(age_counts)
    # Note: the file uses "aged" instead of "old"
    if 'young' in age_counts and ('old' in age_counts or 'aged' in age_counts):
        print(f"   ✅ Both age groups present")
    else:
        print(f"   ❌ ERROR: Missing age groups")
        sys.exit(1)

    # Check compartments
    print(f"\n9. Compartment distribution:")
    compartment_counts = df_metadata['Disc Compartments'].value_counts()
    print(compartment_counts)
    expected_compartments = {'NP', 'IAF', 'OAF'}
    actual_compartments = set(df_metadata['Disc Compartments'].unique())
    if expected_compartments.issubset(actual_compartments):
        print(f"   ✅ All 3 compartments present (NP, IAF, OAF)")
    else:
        missing = expected_compartments - actual_compartments
        print(f"   ❌ ERROR: Missing compartments: {missing}")
        sys.exit(1)

    # Validate metadata join
    print(f"\n10. Validating metadata join...")
    # Extract profile names from Raw data LFQ columns
    profile_names_raw = [col.replace('LFQ intensity ', '') for col in lfq_columns]
    profile_names_metadata = df_metadata['Profile names'].tolist()

    print(f"    Profile count in Raw data: {len(profile_names_raw)}")
    print(f"    Profile count in Sample information: {len(profile_names_metadata)}")

    # Find mismatches
    set_raw = set(profile_names_raw)
    set_metadata = set(profile_names_metadata)

    mismatches_raw = set_raw - set_metadata
    mismatches_metadata = set_metadata - set_raw

    if mismatches_raw:
        print(f"    ⚠️  WARNING: {len(mismatches_raw)} profiles in Raw data but not in Sample information")
        print(f"    Examples: {list(mismatches_raw)[:3]}")

    if mismatches_metadata:
        print(f"    ⚠️  WARNING: {len(mismatches_metadata)} profiles in Sample information but not in Raw data")
        print(f"    Examples: {list(mismatches_metadata)[:3]}")

    if not mismatches_raw and not mismatches_metadata:
        print(f"    ✅ All profiles match between sheets")

    # Check for duplicate profile names
    if len(profile_names_metadata) != len(set_metadata):
        print(f"    ❌ ERROR: Duplicate profile names in Sample information")
        sys.exit(1)
    else:
        print(f"    ✅ Profile names are unique in Sample information")

    # Data quality checks
    print(f"\n11. Data quality checks...")

    # Check for null protein IDs
    null_ids = df_raw['T: Majority protein IDs'].isna().sum()
    if null_ids > 0:
        pct = (null_ids / len(df_raw)) * 100
        print(f"    ⚠️  {null_ids} rows missing Protein IDs ({pct:.1f}%)")
        if pct > 10:
            print(f"    ❌ ERROR: >10% missing Protein IDs")
            sys.exit(1)
    else:
        print(f"    ✅ No missing Protein IDs")

    # Check for null gene symbols
    null_genes = df_raw['T: Gene names'].isna().sum()
    if null_genes > 0:
        pct = (null_genes / len(df_raw)) * 100
        print(f"    ⚠️  {null_genes} rows missing Gene names ({pct:.1f}%)")
    else:
        print(f"    ✅ No missing Gene names")

    # Sample LFQ intensity columns for null data
    print(f"\n12. Checking LFQ intensity columns (sampling first 5)...")
    for col in lfq_columns[:5]:
        null_count = df_raw[col].isna().sum()
        pct = (null_count / len(df_raw)) * 100
        if null_count == len(df_raw):
            print(f"    ❌ ERROR: Column {col} is entirely empty")
            sys.exit(1)
        else:
            print(f"    ℹ️  {col}: {null_count} nulls ({pct:.1f}%)")

    # Check for missing compartment labels
    null_compartments = df_metadata['Disc Compartments'].isna().sum()
    if null_compartments > 0:
        print(f"    ⚠️  {null_compartments} profiles missing Compartment labels")
    else:
        print(f"    ✅ No missing Compartment labels")

    # Final decision
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)
    print("✅ File reconnaissance completed successfully")
    print(f"   - File accessible: ✅")
    print(f"   - Both sheets present: ✅")
    print(f"   - Protein count: {df_raw.shape[0]}")
    print(f"   - Profile count: {len(lfq_columns)}")
    print(f"   - Age groups: {list(age_counts.index)}")
    print(f"   - Compartments: {sorted(actual_compartments)}")
    print(f"   - Missing Protein IDs: {null_ids} ({null_ids/len(df_raw)*100:.1f}%)")
    print(f"\n✅ PROCEED to Phase 2: Data Parsing")
    print("=" * 80)

if __name__ == "__main__":
    reconnaissance()
