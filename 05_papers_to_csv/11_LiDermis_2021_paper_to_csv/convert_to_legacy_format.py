#!/usr/bin/env python3
"""
Step 3: Filter ECM and convert to LEGACY format for LiDermis 2021
"""

import pandas as pd
import numpy as np
from datetime import datetime

def log(message, log_file="agent_log.md"):
    """Append message to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def main():
    log("\n---\n")
    log("## PHASE 1 (continued): Convert to LEGACY Format")
    log("\n### Step 3.1: Load annotated data and filter ECM")

    # Load annotated long format
    df_annotated = pd.read_csv("LiDermis_2021_long_annotated.csv")
    log(f"Loaded annotated data: {len(df_annotated)} rows")

    # Filter ECM only
    df_ecm = df_annotated[df_annotated['Match_Confidence'] > 0].copy()
    log(f"Filtered to ECM proteins: {len(df_ecm)} rows")
    log(f"Unique ECM proteins: {df_ecm['Protein_ID'].nunique()}")

    log("\n### Step 3.2: Aggregate by Age group WITH N_Profiles count")

    # Aggregate Young samples (LiDermis: Age = 2yr OR 14yr)
    # Count non-NaN values per protein = N_Profiles
    df_young = df_ecm[df_ecm['Age'].isin([2, 14])].groupby('Protein_ID').agg({
        'Protein_Name': 'first',
        'Gene_Symbol': 'first',
        'Canonical_Gene_Symbol': 'first',
        'Matrisome Category': 'first',
        'Matrisome Division': 'first',
        'Tissue': 'first',
        'Species': 'first',
        'Method': 'first',
        'Study_ID': 'first',
        'Match_Level': 'first',
        'Match_Confidence': 'first',
        'Abundance': [
            ('mean', lambda x: x.mean(skipna=True)),
            ('count', lambda x: x.notna().sum())  # N_Profiles_Young
        ]
    }).reset_index()

    # Flatten column names
    df_young.columns = [
        'Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Canonical_Gene_Symbol',
        'Matrisome Category', 'Matrisome Division', 'Tissue', 'Species',
        'Method', 'Study_ID', 'Match_Level', 'Match_Confidence',
        'Abundance_Young', 'N_Profiles_Young'
    ]

    log(f"Young aggregation: {len(df_young)} proteins")
    log(f"  N_Profiles_Young range: {df_young['N_Profiles_Young'].min()}-{df_young['N_Profiles_Young'].max()}")

    # Aggregate Old samples (LiDermis: Age = 65yr)
    df_old = df_ecm[df_ecm['Age'] == 65].groupby('Protein_ID').agg({
        'Abundance': [
            ('mean', lambda x: x.mean(skipna=True)),
            ('count', lambda x: x.notna().sum())  # N_Profiles_Old
        ]
    }).reset_index()

    # Flatten column names
    df_old.columns = ['Protein_ID', 'Abundance_Old', 'N_Profiles_Old']

    log(f"Old aggregation: {len(df_old)} proteins")
    log(f"  N_Profiles_Old range: {df_old['N_Profiles_Old'].min()}-{df_old['N_Profiles_Old'].max()}")

    log("\n### Step 3.3: Merge Young and Old")

    # Merge
    df_wide = df_young.merge(df_old, on='Protein_ID', how='outer')
    log(f"Merged wide format: {len(df_wide)} proteins")

    # Check for missing values
    missing_young = df_wide['Abundance_Young'].isna().sum()
    missing_old = df_wide['Abundance_Old'].isna().sum()
    log(f"Missing Abundance_Young: {missing_young} ({missing_young/len(df_wide)*100:.1f}%)")
    log(f"Missing Abundance_Old: {missing_old} ({missing_old/len(df_wide)*100:.1f}%)")

    log("\n### Step 3.4: Add LEGACY format columns")

    # 1. Dataset_Name (duplicate of Study_ID)
    df_wide['Dataset_Name'] = df_wide['Study_ID']

    # 2. Organ (from Tissue - for LiDermis it's "Skin dermis")
    df_wide['Organ'] = 'Skin dermis'  # From config

    # 3. Compartment (for LiDermis without compartments = same as Tissue)
    df_wide['Compartment'] = 'Skin dermis'  # No compartments

    # 4. Tissue_Compartment (same as Compartment)
    df_wide['Tissue_Compartment'] = df_wide['Compartment']

    log(f"Added Dataset_Name, Organ, Compartment, Tissue_Compartment")

    log("\n### Step 3.5: Keep abundances as-is (ALREADY log2-transformed)")

    # Data is ALREADY log2-normalized (from source Table S2)
    # Just copy to _transformed columns
    df_wide['Abundance_Young_transformed'] = df_wide['Abundance_Young']
    df_wide['Abundance_Old_transformed'] = df_wide['Abundance_Old']

    log(f"Added log2-transformed abundances")

    # Sample values
    sample = df_wide[['Abundance_Young', 'Abundance_Young_transformed']].head(3)
    log(f"\nSample log2 transform:")
    for i, row in sample.iterrows():
        if pd.notna(row['Abundance_Young']):
            log(f"  {row['Abundance_Young']:.2f} → {row['Abundance_Young_transformed']:.2f}")

    log("\n### Step 3.6: Rename columns to LEGACY format (underscore)")

    df_wide.rename(columns={
        'Matrisome Category': 'Matrisome_Category',
        'Matrisome Division': 'Matrisome_Division'
    }, inplace=True)

    log(f"Renamed Matrisome columns (space → underscore)")

    log("\n### Step 3.7: Add placeholder z-score columns")

    # Z-scores will be calculated later
    df_wide['Zscore_Young'] = np.nan
    df_wide['Zscore_Old'] = np.nan
    df_wide['Zscore_Delta'] = np.nan

    log(f"Added placeholder z-score columns (will be filled by universal_zscore_function.py)")

    log("\n### Step 3.8: Reorder columns to match LEGACY format")

    # LEGACY column order (from merged_ecm_aging_zscore.csv)
    column_order = [
        'Dataset_Name', 'Organ', 'Compartment',
        'Abundance_Old', 'Abundance_Old_transformed',
        'Abundance_Young', 'Abundance_Young_transformed',
        'Canonical_Gene_Symbol', 'Gene_Symbol',
        'Match_Confidence', 'Match_Level',
        'Matrisome_Category', 'Matrisome_Division',
        'Method',
        'N_Profiles_Old', 'N_Profiles_Young',
        'Protein_ID', 'Protein_Name', 'Species', 'Study_ID',
        'Tissue', 'Tissue_Compartment',
        'Zscore_Delta', 'Zscore_Old', 'Zscore_Young'
    ]

    df_wide = df_wide[column_order]

    log(f"Reordered columns to match legacy format")

    log("\n### Step 3.9: Validation")

    log(f"Final schema check:")
    log(f"  Columns: {len(df_wide.columns)} (expected: 25)")
    log(f"  Rows: {len(df_wide)}")
    log(f"  Unique proteins: {df_wide['Protein_ID'].nunique()}")

    # Sample data
    log(f"\nSample rows (first 3):")
    for i, row in df_wide.head(3).iterrows():
        young_val = f"{row['Abundance_Young']:.2f}" if pd.notna(row['Abundance_Young']) else 'NaN'
        old_val = f"{row['Abundance_Old']:.2f}" if pd.notna(row['Abundance_Old']) else 'NaN'
        log(f"  {row['Protein_ID']}: {row['Gene_Symbol']} - Young={young_val}, Old={old_val}, N_Profiles={row['N_Profiles_Young']:.0f}/{row['N_Profiles_Old']:.0f}")

    log("\n### Step 3.10: Save LEGACY format CSV")

    output_file = "LiDermis_2021_LEGACY_format.csv"
    df_wide.to_csv(output_file, index=False)
    log(f"✅ Saved: {output_file}")

    log("\n✅ PHASE 1 COMPLETE - LEGACY format ready")
    log("\nNext steps:")
    log("  1. Review output file")
    log("  2. Append to merged_ecm_aging_zscore.csv manually OR use merge script")
    log("  3. Calculate z-scores using universal_zscore_function.py")

    return df_wide

if __name__ == '__main__':
    try:
        df_result = main()
        print("\n✅ Success!")
    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(f"\n```\n{traceback.format_exc()}\n```")
        raise
