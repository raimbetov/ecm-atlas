#!/usr/bin/env python3
"""
LiDermis 2021 Data Parser
Based on: 04_compilation_of_papers/06_LiDermis_2021_comprehensive_analysis.md

Key features:
- 4 age groups → 2 bins (exclude Adult 40yr middle-aged)
- 10 samples total: Toddler(2), Teenager(3), Adult(2), Elderly(3)
- Protein_Name via UniProt lookup (not in source file)
- Already log2-normalized data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Configuration from comprehensive analysis
CONFIG = {
    "study_id": "LiDermis_2021",
    "data_file": "data_raw/Li et al. - 2021 | dermis/Table 2.xlsx",
    "sheet_name": "Table S2",
    "species": "Homo sapiens",
    "tissue": "Skin dermis",
    "method": "Label-free LC-MS/MS",
    "age_unit": "years",
    # Age midpoints from comprehensive analysis
    "age_mapping": {
        "Toddler": 2,      # Range: 1-3 years
        "Teenager": 14,    # Range: 8-20 years
        "Adult": 40,       # Range: 30-50 years (EXCLUDED)
        "Elderly": 65      # Range: >60 years
    },
    # Age groups to INCLUDE (exclude Adult 40yr)
    "young_groups": ["Toddler", "Teenager"],
    "old_groups": ["Elderly"],
    # Human age cutoffs
    "young_cutoff": 30,
    "old_cutoff": 55
}

def log(message, log_file="agent_log.md"):
    """Append message to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def fetch_protein_name_from_uniprot(protein_id, retry=3):
    """Fetch protein name from UniProt API."""
    import requests

    for attempt in range(retry):
        try:
            url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.txt"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Parse UniProt text format for protein name
                for line in response.text.split('\n'):
                    if line.startswith('DE   RecName: Full='):
                        name = line.replace('DE   RecName: Full=', '').strip().rstrip(';')
                        return name
                    elif line.startswith('DE   SubName: Full='):
                        name = line.replace('DE   SubName: Full=', '').strip().rstrip(';')
                        return name
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            if attempt == retry - 1:
                log(f"  ⚠️  Failed to fetch {protein_id}: {e}")
    return None

def main():
    log("# LiDermis 2021 Processing Log (LEGACY Format)")
    log("")
    log("**Study ID:** LiDermis_2021")
    log(f"**Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("**Input:** data_raw/Li et al. - 2021 | dermis/")
    log("**Output Format:** LEGACY (merged_ecm_aging_zscore.csv compatible)")
    log("")
    log("---")
    log("")
    log("## PHASE 1: Data Normalization (LEGACY Format)")
    log("")
    log("### Step 1.1: Load Excel file with complex headers")

    # Find project root
    project_root = Path.cwd()
    while not (project_root / "references" / "human_matrisome_v2.csv").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError("Could not find project root")

    log(f"Project root: {project_root}")

    # Load data - row 3 contains sample names
    data_path = project_root / CONFIG["data_file"]
    log(f"Loading: {data_path}")
    log(f"Sheet: {CONFIG['sheet_name']}")

    # Read raw to get row 3 (sample names)
    df_raw = pd.read_excel(data_path, sheet_name=CONFIG["sheet_name"], header=None)
    sample_row = df_raw.iloc[3]  # Row 3 has sample names

    log(f"\n=== Sample names from row 3 ===")
    sample_names = []
    for i, val in enumerate(sample_row):
        if pd.notna(val) and any(group in str(val) for group in ["Toddler", "Teenager", "Adult", "Elderly"]):
            sample_names.append((i, str(val).strip()))
            log(f"  Column {i}: {val}")

    log(f"\n✅ Found {len(sample_names)} sample columns")

    # Now load data starting from row 4 (after headers)
    df = pd.read_excel(data_path, sheet_name=CONFIG["sheet_name"], skiprows=4)
    log(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Set proper column names from row 3
    new_columns = df.columns.tolist()
    for col_idx, sample_name in sample_names:
        if col_idx < len(new_columns):
            new_columns[col_idx] = sample_name
    df.columns = new_columns

    log(f"\nColumns (first 10):")
    for i, col in enumerate(df.columns[:10]):
        log(f"  {i}: {col}")

    log("\n### Step 1.2: Filter samples (exclude Adult 40yr)")

    # Identify columns by age group (EXCLUDE Ave_ averages, only individual samples)
    young_cols = []
    old_cols = []
    adult_cols = []

    for col in df.columns:
        col_str = str(col)
        # Skip average columns (Ave_) - only want individual Sample columns
        if col_str.startswith('Ave_') or 'Ave' in col_str[:3]:
            continue
        # Only include columns with "Sample" in name
        if 'Sample' not in col_str:
            continue

        if any(group in col_str for group in CONFIG["young_groups"]):
            young_cols.append(col)
        elif any(group in col_str for group in CONFIG["old_groups"]):
            old_cols.append(col)
        elif "Adult" in col_str:
            adult_cols.append(col)

    log(f"Young columns (Toddler + Teenager): {len(young_cols)}")
    for col in young_cols:
        log(f"  - {col}")

    log(f"\nOld columns (Elderly): {len(old_cols)}")
    for col in old_cols:
        log(f"  - {col}")

    log(f"\n❌ Excluded Adult columns (40yr, middle-aged): {len(adult_cols)}")
    for col in adult_cols:
        log(f"  - {col}")

    retention_pct = (len(young_cols) + len(old_cols)) / len(sample_names) * 100
    log(f"\n✅ Retention: {len(young_cols) + len(old_cols)}/{len(sample_names)} = {retention_pct:.1f}%")

    log("\n### Step 1.3: Extract protein identifiers")

    # Find Protein ID and Gene symbol columns
    protein_id_col = None
    gene_symbol_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'protein' in col_lower and 'id' in col_lower:
            protein_id_col = col
        elif 'gene' in col_lower and 'symbol' in col_lower:
            gene_symbol_col = col

    if not protein_id_col or not gene_symbol_col:
        # Try first two columns as fallback
        protein_id_col = df.columns[0]
        gene_symbol_col = df.columns[1]

    log(f"Protein ID column: '{protein_id_col}'")
    log(f"Gene Symbol column: '{gene_symbol_col}'")

    log(f"\nSample Protein IDs (first 5):")
    for i, pid in enumerate(df[protein_id_col].head(5)):
        log(f"  {pid}")

    log("\n### Step 1.4: Protein_Name lookup via UniProt API")
    log("⚠️  Warning: This may take 5-10 minutes for ~260 proteins")

    # Fetch protein names
    protein_names = {}
    unique_ids = df[protein_id_col].dropna().unique()

    log(f"Fetching names for {len(unique_ids)} unique proteins...")

    for i, protein_id in enumerate(unique_ids):
        if i > 0 and i % 50 == 0:
            log(f"  Progress: {i}/{len(unique_ids)} ({i/len(unique_ids)*100:.1f}%)")

        name = fetch_protein_name_from_uniprot(protein_id)
        if name:
            protein_names[protein_id] = name
        else:
            protein_names[protein_id] = f"Unknown protein ({protein_id})"

    log(f"✅ Fetched {len([v for v in protein_names.values() if 'Unknown' not in v])} protein names")
    log(f"⚠️  {len([v for v in protein_names.values() if 'Unknown' in v])} proteins without names")

    log("\n### Step 1.5: Transform to long format")

    rows = []

    for idx, row in df.iterrows():
        protein_id = row[protein_id_col]
        if pd.isna(protein_id):
            continue

        gene_symbol = row[gene_symbol_col]
        protein_name = protein_names.get(protein_id, f"Unknown protein ({protein_id})")

        # Process young samples (Toddler + Teenager)
        for col in young_cols:
            abundance = row[col]
            age_group = None
            for group in ["Toddler", "Teenager"]:
                if group in str(col):
                    age_group = group
                    break

            age = CONFIG["age_mapping"][age_group]
            sample_num = str(col).split('Sample')[-1].strip() if 'Sample' in str(col) else "1"

            rows.append({
                'Protein_ID': protein_id,
                'Protein_Name': protein_name,
                'Gene_Symbol': gene_symbol,
                'Tissue': CONFIG['tissue'],
                'Species': CONFIG['species'],
                'Age': age,
                'Age_Unit': CONFIG['age_unit'],
                'Abundance': abundance,
                'Abundance_Unit': 'log2_normalized_intensity',
                'Method': CONFIG['method'],
                'Study_ID': CONFIG['study_id'],
                'Sample_ID': f"Dermis_{age_group}_{sample_num}",
                'Parsing_Notes': f"Age={age}yr (midpoint of {age_group} group, range varies); Column '{col}' in Table 2 sheet 'Table S2'; Abundance: log2-normalized intensity (FOT fraction of total); Method: LC-MS/MS on decellularized dermal scaffold; Adult group (40yr, n=2) excluded as middle-aged (30<40<55); Retained {retention_pct:.1f}% samples"
            })

        # Process old samples (Elderly)
        for col in old_cols:
            abundance = row[col]
            age = CONFIG["age_mapping"]["Elderly"]
            sample_num = str(col).split('Sample')[-1].strip() if 'Sample' in str(col) else "1"

            rows.append({
                'Protein_ID': protein_id,
                'Protein_Name': protein_name,
                'Gene_Symbol': gene_symbol,
                'Tissue': CONFIG['tissue'],
                'Species': CONFIG['species'],
                'Age': age,
                'Age_Unit': CONFIG['age_unit'],
                'Abundance': abundance,
                'Abundance_Unit': 'log2_normalized_intensity',
                'Method': CONFIG['method'],
                'Study_ID': CONFIG['study_id'],
                'Sample_ID': f"Dermis_Elderly_{sample_num}",
                'Parsing_Notes': f"Age={age}yr (midpoint of Elderly group, >60yr); Column '{col}' in Table 2 sheet 'Table S2'; Abundance: log2-normalized intensity (FOT fraction of total); Method: LC-MS/MS on decellularized dermal scaffold; Adult group (40yr, n=2) excluded as middle-aged (30<40<55); Retained {retention_pct:.1f}% samples"
            })

    df_long = pd.DataFrame(rows)

    log(f"✅ Long format created: {len(df_long)} rows")
    expected_rows = len(df) * (len(young_cols) + len(old_cols))
    log(f"   Expected: {len(df)} proteins × {len(young_cols) + len(old_cols)} samples = {expected_rows} rows")
    log(f"   Match: {'✅' if len(df_long) == expected_rows else '❌'}")

    log("\n### Step 1.6: Validate schema")

    expected_cols = [
        'Protein_ID', 'Protein_Name', 'Gene_Symbol',
        'Tissue', 'Species', 'Age', 'Age_Unit',
        'Abundance', 'Abundance_Unit', 'Method',
        'Study_ID', 'Sample_ID', 'Parsing_Notes'
    ]

    log(f"Schema columns: {df_long.columns.tolist()}")
    all_present = all(col in df_long.columns for col in expected_cols)
    log(f"All expected columns present: {'✅' if all_present else '❌'}")

    log("\n### Step 1.7: Basic statistics")

    log(f"Total rows: {len(df_long)}")
    log(f"Unique proteins: {df_long['Protein_ID'].nunique()}")
    log(f"Young samples: {len(young_cols)} (Toddler: 2yr × 2, Teenager: 14yr × 3)")
    log(f"Old samples: {len(old_cols)} (Elderly: 65yr × 3)")
    log(f"Missing abundances: {df_long['Abundance'].isna().sum()} ({df_long['Abundance'].isna().sum() / len(df_long) * 100:.1f}%)")

    # Age distribution
    log(f"\nAge distribution:")
    for age, count in df_long.groupby('Age').size().items():
        log(f"  {age}yr: {count} rows")

    log("\n### Step 1.8: Save long format CSV")

    output_file = "LiDermis_2021_long_format.csv"
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
