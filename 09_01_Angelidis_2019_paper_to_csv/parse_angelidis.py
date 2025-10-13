#!/usr/bin/env python3
"""
Angelidis 2019 Data Parser
Based on: 04_compilation_of_papers/01_Angelidis_2019_comprehensive_analysis.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration from comprehensive analysis
CONFIG = {
    "study_id": "Angelidis_2019",
    "data_file": "data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx",
    "sheet_name": "Proteome",
    "species": "Mus musculus",
    "tissue": "Lung",
    "method": "Label-free LC-MS/MS (MaxQuant LFQ)",
    "young_age": 3,
    "old_age": 24,
    "age_unit": "months",
    "young_columns": ["young_1", "young_2", "young_3", "young_4"],
    "old_columns": ["old_1", "old_2", "old_3", "old_4"],
    "protein_id_col": "Protein IDs",
    "protein_name_col": "Protein names",
    "gene_symbol_col": "Gene names"
}

def log(message, log_file="agent_log.md"):
    """Append message to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def main():
    log("### Step 1.1: Load Excel file")

    # Find project root
    project_root = Path.cwd()
    while not (project_root / "references" / "human_matrisome_v2.csv").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError("Could not find project root")

    log(f"Project root: {project_root}")

    # Load data
    data_path = project_root / CONFIG["data_file"]
    log(f"Loading: {data_path}")
    log(f"Sheet: {CONFIG['sheet_name']}")

    df = pd.read_excel(data_path, sheet_name=CONFIG["sheet_name"])
    log(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Show columns
    log(f"\nColumns found:")
    for i, col in enumerate(df.columns[:10]):
        log(f"  {i}: {col}")
    if len(df.columns) > 10:
        log(f"  ... and {len(df.columns) - 10} more")

    log("\n### Step 1.2: Verify sample columns exist")

    sample_cols = CONFIG["young_columns"] + CONFIG["old_columns"]
    missing_cols = [col for col in sample_cols if col not in df.columns]

    if missing_cols:
        log(f"❌ ERROR: Missing sample columns: {missing_cols}")
        log(f"Available columns: {df.columns.tolist()}")
        return False

    log(f"✅ All 8 sample columns found")

    log("\n### Step 1.3: Extract and process protein IDs")

    # Process Protein IDs (take first if semicolon-separated)
    df['Protein_ID_processed'] = df[CONFIG['protein_id_col']].apply(
        lambda x: x.split(';')[0] if pd.notna(x) and ';' in str(x) else x
    )

    log(f"Sample Protein IDs (first 5):")
    for i, pid in enumerate(df['Protein_ID_processed'].head(5)):
        log(f"  {pid}")

    log("\n### Step 1.4: Transform to long format")

    # Create base dataframe with protein info
    protein_info = df[[
        CONFIG['protein_id_col'],
        CONFIG['protein_name_col'],
        CONFIG['gene_symbol_col']
    ]].copy()

    protein_info.columns = ['Original_Protein_ID', 'Protein_Name', 'Gene_Symbol']
    protein_info['Protein_ID'] = df['Protein_ID_processed']

    # Reshape to long format
    rows = []

    for idx, row in df.iterrows():
        protein_id = row['Protein_ID_processed']
        protein_name = row[CONFIG['protein_name_col']]
        gene_symbol = row[CONFIG['gene_symbol_col']]

        # Young samples
        for col in CONFIG['young_columns']:
            rows.append({
                'Protein_ID': protein_id,
                'Protein_Name': protein_name,
                'Gene_Symbol': gene_symbol,
                'Tissue': CONFIG['tissue'],
                'Species': CONFIG['species'],
                'Age': CONFIG['young_age'],
                'Age_Unit': CONFIG['age_unit'],
                'Abundance': row[col],
                'Abundance_Unit': 'LFQ_intensity',
                'Method': CONFIG['method'],
                'Study_ID': CONFIG['study_id'],
                'Sample_ID': col,
                'Parsing_Notes': f"Age={CONFIG['young_age']}mo from column '{col}'; LFQ intensity from MaxQuant; C57BL/6J cohorts"
            })

        # Old samples
        for col in CONFIG['old_columns']:
            rows.append({
                'Protein_ID': protein_id,
                'Protein_Name': protein_name,
                'Gene_Symbol': gene_symbol,
                'Tissue': CONFIG['tissue'],
                'Species': CONFIG['species'],
                'Age': CONFIG['old_age'],
                'Age_Unit': CONFIG['age_unit'],
                'Abundance': row[col],
                'Abundance_Unit': 'LFQ_intensity',
                'Method': CONFIG['method'],
                'Study_ID': CONFIG['study_id'],
                'Sample_ID': col,
                'Parsing_Notes': f"Age={CONFIG['old_age']}mo from column '{col}'; LFQ intensity from MaxQuant; C57BL/6J cohorts"
            })

    df_long = pd.DataFrame(rows)

    log(f"✅ Long format created: {len(df_long)} rows")
    log(f"   Expected: {len(df)} proteins × 8 samples = {len(df) * 8} rows")
    log(f"   Match: {'✅' if len(df_long) == len(df) * 8 else '❌'}")

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
    log(f"Young samples: {len(CONFIG['young_columns'])}")
    log(f"Old samples: {len(CONFIG['old_columns'])}")
    log(f"Missing abundances: {df_long['Abundance'].isna().sum()} ({df_long['Abundance'].isna().sum() / len(df_long) * 100:.1f}%)")

    log("\n### Step 1.7: Save long format CSV")

    output_file = "Angelidis_2019_long_format.csv"
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
