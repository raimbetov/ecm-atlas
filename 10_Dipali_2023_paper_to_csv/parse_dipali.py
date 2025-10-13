#!/usr/bin/env python3
"""
Dipali 2023 Data Parser
Based on: 04_compilation_of_papers/05_Dipali_2023_comprehensive_analysis.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Configuration from comprehensive analysis
CONFIG = {
    "study_id": "Dipali_2023",
    "data_file": "data_raw/Dipali et al. - 2023/Report_Birgit_Protein+Quant_Pivot+(Pivot).xls",
    "species": "Mus musculus",
    "tissue": "Ovary",
    "method": "Label-free DIA (directDIA)",
    "young_age": 2.25,  # Mean of 6-12 weeks
    "old_age": 11,      # 10-12 months
    "age_unit": "months",
    "protein_id_col": "PG.UniProtIds",
    "protein_name_col": "PG.ProteinDescriptions",
    "gene_symbol_col": "PG.Genes",
    "quantity_suffix": ".PG.Quantity"
}

def log(message, log_file="agent_log.md"):
    """Append message to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def extract_sample_id(column_name):
    """Extract sample ID from column name like '[1] 210910_0013_SD7_11_Y1L_S6_800ng_180min_DIA.raw.PG.Quantity'"""
    # Look for Y1L, O1L, etc. pattern
    match = re.search(r'_([YO]\d+L)_', column_name)
    if match:
        return match.group(1)
    return None

def determine_age(sample_id):
    """Determine age from sample ID (Y = young, O = old)"""
    if sample_id.startswith('Y'):
        return CONFIG['young_age']
    elif sample_id.startswith('O'):
        return CONFIG['old_age']
    return None

def main():
    log("# Dipali 2023 Processing Log (LEGACY Format)")
    log("")
    log("**Study ID:** Dipali_2023")
    log(f"**Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("**Input:** data_raw/Dipali et al. - 2023/")
    log("**Output Format:** LEGACY (merged_ecm_aging_zscore.csv compatible)")
    log("")
    log("---")
    log("")
    log("## PHASE 1: Data Normalization (LEGACY Format)")
    log("")
    log("### Step 1.1: Load TSV file (despite .xls extension)")

    # Find project root
    project_root = Path.cwd()
    while not (project_root / "references" / "mouse_matrisome_v2.csv").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError("Could not find project root")

    log(f"Project root: {project_root}")

    # Load data (tab-separated despite .xls extension)
    data_path = project_root / CONFIG["data_file"]
    log(f"Loading: {data_path}")

    df = pd.read_csv(data_path, sep='\t')
    log(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Show columns
    log(f"\nColumns found:")
    for i, col in enumerate(df.columns[:10]):
        log(f"  {i}: {col}")
    if len(df.columns) > 10:
        log(f"  ... and {len(df.columns) - 10} more")

    log("\n### Step 1.2: Find and verify .PG.Quantity columns")

    # Find all columns ending with .PG.Quantity
    quantity_cols = [col for col in df.columns if col.endswith(CONFIG['quantity_suffix'])]
    log(f"Found {len(quantity_cols)} quantity columns")

    # Extract sample IDs and map to ages
    sample_mapping = {}
    for col in quantity_cols:
        sample_id = extract_sample_id(col)
        if sample_id:
            age = determine_age(sample_id)
            sample_mapping[col] = {
                'sample_id': sample_id,
                'age': age,
                'age_group': 'Young' if sample_id.startswith('Y') else 'Old'
            }
            log(f"  {sample_id}: {age} months ({sample_mapping[col]['age_group']})")

    if len(sample_mapping) != 10:
        log(f"⚠️  WARNING: Expected 10 samples, found {len(sample_mapping)}")

    log(f"\n✅ Sample mapping complete:")
    young_count = sum(1 for s in sample_mapping.values() if s['age_group'] == 'Young')
    old_count = sum(1 for s in sample_mapping.values() if s['age_group'] == 'Old')
    log(f"  Young: {young_count} samples")
    log(f"  Old: {old_count} samples")

    log("\n### Step 1.3: Extract and process protein IDs")

    # Process UniProt IDs (take first if semicolon-separated)
    df['Protein_ID_processed'] = df[CONFIG['protein_id_col']].apply(
        lambda x: str(x).split(';')[0] if pd.notna(x) and ';' in str(x) else x
    )

    log(f"Sample Protein IDs (first 5):")
    for i, pid in enumerate(df['Protein_ID_processed'].head(5)):
        log(f"  {pid}")

    log("\n### Step 1.4: Transform to long format")

    # Create rows for long format
    rows = []

    for idx, row in df.iterrows():
        protein_id = row['Protein_ID_processed']
        protein_name = row[CONFIG['protein_name_col']]
        gene_symbol = row[CONFIG['gene_symbol_col']]

        # Process each quantity column
        for col, mapping in sample_mapping.items():
            abundance = row[col]

            # Note about age deviation
            age_note = ""
            if mapping['age'] == 11:
                age_note = " (Note: 11mo below standard 18mo geriatric cutoff, but biologically relevant for reproductive aging)"

            rows.append({
                'Protein_ID': protein_id,
                'Protein_Name': protein_name,
                'Gene_Symbol': gene_symbol,
                'Tissue': CONFIG['tissue'],
                'Species': CONFIG['species'],
                'Age': mapping['age'],
                'Age_Unit': CONFIG['age_unit'],
                'Abundance': abundance,
                'Abundance_Unit': 'DIA_intensity',
                'Method': CONFIG['method'],
                'Study_ID': CONFIG['study_id'],
                'Sample_ID': mapping['sample_id'],
                'Parsing_Notes': f"Age={mapping['age']}mo from sample {mapping['sample_id']}; DIA intensity from Spectronaut{age_note}; Decellularized ECM from mouse ovary"
            })

    df_long = pd.DataFrame(rows)

    log(f"✅ Long format created: {len(df_long)} rows")
    log(f"   Expected: {len(df)} proteins × 10 samples = {len(df) * 10} rows")
    log(f"   Match: {'✅' if len(df_long) == len(df) * 10 else '❌'}")

    log("\n### Step 1.5: Validate schema")

    expected_cols = [
        'Protein_ID', 'Protein_Name', 'Gene_Symbol',
        'Tissue', 'Species', 'Age', 'Age_Unit',
        'Abundance', 'Abundance_Unit', 'Method',
        'Study_ID', 'Sample_ID', 'Parsing_Notes'
    ]

    log(f"Schema columns: {df_long.columns.tolist()}")

    all_present = all(col in df_long.columns for col in expected_cols)
    log(f"All expected columns present: {'✅' if all_present else '❌'}")

    log("\n### Step 1.6: Basic statistics")

    log(f"Total rows: {len(df_long)}")
    log(f"Unique proteins: {df_long['Protein_ID'].nunique()}")
    log(f"Young samples: {young_count}")
    log(f"Old samples: {old_count}")
    log(f"Missing abundances: {df_long['Abundance'].isna().sum()} ({df_long['Abundance'].isna().sum() / len(df_long) * 100:.1f}%)")

    # Check abundance range
    valid_abundances = df_long['Abundance'].dropna()
    if len(valid_abundances) > 0:
        log(f"Abundance range: {valid_abundances.min():.2e} to {valid_abundances.max():.2e}")

    log("\n### Step 1.7: Save long format CSV")

    output_file = "Dipali_2023_long_format.csv"
    df_long.to_csv(output_file, index=False)
    log(f"✅ Saved: {output_file}")

    log("\n✅ PHASE 1 COMPLETE - Long format created")

    return df_long

if __name__ == '__main__':
    try:
        df_result = main()
        print("\n✅ Success!")
    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(f"\n```\n{traceback.format_exc()}\n```")
        raise
