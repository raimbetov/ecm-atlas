#!/usr/bin/env python3
"""
Step 3: Filter ECM and convert to wide format
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
    log("## PHASE 1 (continued): Wide Format Conversion")
    log("\n### Step 3.1: Load annotated data and filter ECM")

    # Load annotated long format
    df_annotated = pd.read_csv("Angelidis_2019_long_annotated.csv")
    log(f"Loaded annotated data: {len(df_annotated)} rows")

    # Filter ECM only
    df_ecm = df_annotated[df_annotated['Match_Confidence'] > 0].copy()
    log(f"Filtered to ECM proteins: {len(df_ecm)} rows")
    log(f"Unique ECM proteins: {df_ecm['Protein_ID'].nunique()}")

    log("\n### Step 3.2: Aggregate by Age group")

    # Aggregate Young samples (mean across replicates, skipna=True)
    df_young = df_ecm[df_ecm['Age'] == 3].groupby('Protein_ID').agg({
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
        'Abundance': lambda x: x.mean(skipna=True)  # Mean of young_1-4
    }).reset_index()

    df_young.rename(columns={'Abundance': 'Abundance_Young'}, inplace=True)
    log(f"Young aggregation: {len(df_young)} proteins")

    # Aggregate Old samples
    df_old = df_ecm[df_ecm['Age'] == 24].groupby('Protein_ID').agg({
        'Abundance': lambda x: x.mean(skipna=True)  # Mean of old_1-4
    }).reset_index()

    df_old.rename(columns={'Abundance': 'Abundance_Old'}, inplace=True)
    log(f"Old aggregation: {len(df_old)} proteins")

    log("\n### Step 3.3: Merge Young and Old")

    # Merge
    df_wide = df_young.merge(df_old, on='Protein_ID', how='outer')
    log(f"Merged wide format: {len(df_wide)} proteins")

    # Check for missing values
    missing_young = df_wide['Abundance_Young'].isna().sum()
    missing_old = df_wide['Abundance_Old'].isna().sum()
    log(f"Missing Abundance_Young: {missing_young} ({missing_young/len(df_wide)*100:.1f}%)")
    log(f"Missing Abundance_Old: {missing_old} ({missing_old/len(df_wide)*100:.1f}%)")

    log("\n### Step 3.4: Add Tissue_Compartment column")

    # No compartments for Angelidis (whole lung tissue)
    df_wide['Tissue_Compartment'] = df_wide['Tissue']
    log(f"Tissue_Compartment set to: {df_wide['Tissue_Compartment'].unique().tolist()}")

    log("\n### Step 3.5: Reorder columns to match expected schema")

    column_order = [
        'Protein_ID', 'Protein_Name', 'Gene_Symbol',
        'Canonical_Gene_Symbol', 'Matrisome Category', 'Matrisome Division',
        'Tissue', 'Tissue_Compartment', 'Species',
        'Abundance_Young', 'Abundance_Old',
        'Method', 'Study_ID',
        'Match_Level', 'Match_Confidence'
    ]

    df_wide = df_wide[column_order]

    log("\n### Step 3.6: Validation")

    log(f"Final schema check:")
    log(f"  Columns: {df_wide.columns.tolist()}")
    log(f"  Rows: {len(df_wide)}")
    log(f"  Unique proteins: {df_wide['Protein_ID'].nunique()}")

    # Sample data
    log(f"\nSample rows (first 3):")
    for i, row in df_wide.head(3).iterrows():
        young_val = f"{row['Abundance_Young']:.2f}" if pd.notna(row['Abundance_Young']) else 'NaN'
        old_val = f"{row['Abundance_Old']:.2f}" if pd.notna(row['Abundance_Old']) else 'NaN'
        log(f"  {row['Protein_ID']}: {row['Gene_Symbol']} - Young={young_val}, Old={old_val}")

    log("\n### Step 3.7: Save wide format CSV")

    output_file = "Angelidis_2019_wide_format.csv"
    df_wide.to_csv(output_file, index=False)
    log(f"✅ Saved: {output_file}")

    log("\n✅ PHASE 1 COMPLETE - Wide format ready for merge")

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
