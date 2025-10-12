#!/usr/bin/env python3
"""
Phase 3: Schema Standardization
Map columns to unified 14-column format
"""

import pandas as pd
import numpy as np
import sys

def standardize_schema():
    """Standardize to unified schema"""

    print("=" * 80)
    print("PHASE 3: SCHEMA STANDARDIZATION")
    print("=" * 80)

    # Load long-format data
    print("\n1. Loading long-format data...")
    df_long = pd.read_csv("Tam_2020_long_format.csv")
    print(f"   Shape: {df_long.shape}")
    print(f"   Columns: {list(df_long.columns)}")

    # Step 2: Create standardized dataframe
    print("\n2. Creating standardized schema...")

    df_standardized = pd.DataFrame({
        # Protein identifiers
        'Protein_ID': df_long['Majority protein IDs'],
        'Protein_Name': df_long['Protein names'],
        'Gene_Symbol': df_long['Gene names'],
        'Canonical_Gene_Symbol': None,  # Filled in annotation phase
        'Matrisome_Category': None,      # Filled in annotation phase
        'Matrisome_Division': None,      # Filled in annotation phase

        # Tissue metadata - COMPARTMENTS KEPT SEPARATE
        'Tissue': df_long['Tissue'],  # "Intervertebral_disc_NP" or "Intervertebral_disc_IAF" or "Intervertebral_disc_OAF" or "Intervertebral_disc_NP/IAF"
        'Tissue_Compartment': df_long['Tissue_Compartment'],  # Explicit compartment for filtering

        # Species
        'Species': 'Homo sapiens',

        # Age information
        'Age': df_long['Age'],
        'Age_Unit': 'years',
        'Age_Bin': df_long['Age_Bin'],

        # Abundance
        'Abundance': df_long['Abundance'],
        'Abundance_Unit': 'LFQ_intensity',

        # Method
        'Method': 'Label-free LC-MS/MS (MaxQuant LFQ)',

        # Study identifiers
        'Study_ID': 'Tam_2020',
        'Sample_ID': df_long['Sample_ID'],

        # Spatial metadata (unique to this study)
        'Disc_Level': df_long['Disc level'],
        'Anatomical_Direction': df_long['Direction'],

        # Additional notes
        'Parsing_Notes': (
            'Spatially-resolved proteomics; ' +
            'Compartment=' + df_long['Tissue_Compartment'].astype(str) + '; ' +
            'Disc_level=' + df_long['Disc level'].astype(str) + '; ' +
            'Profile=' + df_long['Profile_Name'].astype(str)
        )
    })

    print(f"   Standardized shape: {df_standardized.shape}")
    print(f"   Columns: {len(df_standardized.columns)}")

    # Step 3: Data cleaning
    print("\n3. Data cleaning...")

    # Remove rows with null Protein_ID
    initial_rows = len(df_standardized)
    df_standardized = df_standardized[df_standardized['Protein_ID'].notna()].copy()
    removed = initial_rows - len(df_standardized)
    if removed > 0:
        print(f"   ⚠️  Removed {removed} rows with missing Protein_ID")
    else:
        print(f"   ✅ No rows with missing Protein_ID")

    # Remove rows with null Abundance
    initial_rows = len(df_standardized)
    df_standardized = df_standardized[df_standardized['Abundance'].notna()].copy()
    removed = initial_rows - len(df_standardized)
    if removed > 0:
        pct = (removed / initial_rows) * 100
        print(f"   ⚠️  Removed {removed} rows with missing Abundance ({pct:.1f}%)")
    else:
        print(f"   ✅ No rows with missing Abundance")

    # Convert data types
    df_standardized['Age'] = df_standardized['Age'].astype(int)
    df_standardized['Abundance'] = df_standardized['Abundance'].astype(float)

    print(f"   Final shape after cleaning: {df_standardized.shape}")

    # Step 4: Validate schema compliance
    print("\n4. Validating schema compliance...")

    required_cols = ['Protein_ID', 'Gene_Symbol', 'Species', 'Tissue', 'Tissue_Compartment',
                     'Age', 'Age_Unit', 'Abundance', 'Abundance_Unit', 'Method', 'Study_ID', 'Sample_ID']

    all_passed = True
    for col in required_cols:
        null_count = df_standardized[col].isna().sum()
        if null_count > 0:
            print(f"   ❌ FAIL: {null_count} null values in required column '{col}'")
            all_passed = False
        else:
            print(f"   ✅ PASS: Column '{col}' has no nulls")

    if not all_passed:
        print(f"\n   ❌ ERROR: Schema validation failed")
        sys.exit(1)

    # Step 5: Validate compartment separation
    print("\n5. Validating compartment structure...")

    compartment_counts = df_standardized['Tissue'].value_counts()
    print(f"   Tissue distribution:")
    print(compartment_counts)

    unique_tissues = df_standardized['Tissue'].nunique()
    print(f"\n   Unique tissue values: {unique_tissues}")
    print(f"   Expected: 4 (NP, IAF, OAF, NP/IAF transition zone)")

    # Check core compartments are present
    core_compartments = {'Intervertebral_disc_NP', 'Intervertebral_disc_IAF', 'Intervertebral_disc_OAF'}
    actual_compartments = set(df_standardized['Tissue'].unique())

    if core_compartments.issubset(actual_compartments):
        print(f"   ✅ All 3 core compartments present and separate (NP, IAF, OAF)")
    else:
        missing = core_compartments - actual_compartments
        print(f"   ❌ ERROR: Missing compartments: {missing}")
        sys.exit(1)

    # Note about transition zones
    transition_zones = actual_compartments - core_compartments
    if transition_zones:
        print(f"   ℹ️  Transition zones detected: {transition_zones}")
        print(f"   Note: These will be included in analyses as spatially-resolved data")

    # Step 6: Save standardized data
    print("\n6. Saving standardized data...")
    output_file = "Tam_2020_standardized.csv"
    df_standardized.to_csv(output_file, index=False)
    print(f"   ✅ Saved: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY")
    print("=" * 80)
    print(f"✅ Schema standardization completed")
    print(f"   - Standardized rows: {len(df_standardized)}")
    print(f"   - Unique proteins: {df_standardized['Protein_ID'].nunique()}")
    print(f"   - Tissue types: {unique_tissues}")
    print(f"   - Core compartments: {sorted(core_compartments)}")
    print(f"   - Transition zones: {sorted(transition_zones) if transition_zones else 'None'}")
    print(f"   - Schema validation: {'PASSED' if all_passed else 'FAILED'}")
    print(f"\n✅ PROCEED to Phase 4: Protein Annotation")
    print("=" * 80)

    return df_standardized

if __name__ == "__main__":
    df_standardized = standardize_schema()
