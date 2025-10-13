#!/usr/bin/env python3
"""
Step 2: ECM Annotation for Dipali 2023
Following 02_TASK_PROTEIN_ANNOTATION_GUIDELINES.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def log(message, log_file="agent_log.md"):
    """Append message to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"\n[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def main():
    log("\n---\n")
    log("## PHASE 1 (continued): ECM Annotation")
    log("\n### Step 2.1: Load long format and matrisome reference")

    # Find project root
    project_root = Path.cwd()
    while not (project_root / "references" / "mouse_matrisome_v2.csv").exists():
        project_root = project_root.parent
        if project_root == project_root.parent:
            raise FileNotFoundError("Could not find project root")

    # Load long format
    df_long = pd.read_csv("Dipali_2023_long_format.csv")
    log(f"Loaded long format: {len(df_long)} rows")
    log(f"Unique proteins: {df_long['Protein_ID'].nunique()}")

    # Load mouse matrisome
    matrisome_path = project_root / "references" / "mouse_matrisome_v2.csv"
    df_matrisome = pd.read_csv(matrisome_path)
    log(f"Loaded mouse matrisome: {len(df_matrisome)} ECM proteins")

    log("\n### Step 2.2: 4-level annotation hierarchy")

    # Level 1: Gene Symbol match
    log("\nLevel 1: Gene Symbol match")
    log(f"  Matrisome columns: {df_matrisome.columns.tolist()[:5]}...")

    df_level1 = df_long.merge(
        df_matrisome[['Gene Symbol', 'Matrisome Category', 'Matrisome Division']],
        left_on='Gene_Symbol',
        right_on='Gene Symbol',
        how='left'
    )
    matched_level1 = df_level1['Matrisome Category'].notna().sum()
    log(f"  Matched: {matched_level1} / {len(df_long)} rows ({matched_level1/len(df_long)*100:.1f}%)")

    # Level 2: UniProt ID match (for unmatched)
    log("\nLevel 2: UniProt ID match (for unmatched)")
    unmatched_mask = df_level1['Matrisome Category'].isna()

    df_level2 = df_level1.copy()

    # UniProt_IDs column may contain multiple IDs separated by colons
    # Create lookup from all UniProt IDs to matrisome data
    uniprot_lookup = {}
    for _, row in df_matrisome.iterrows():
        uniprots = str(row['UniProt_IDs']).split(':')
        for up_id in uniprots:
            if up_id and up_id != 'nan':
                uniprot_lookup[up_id.strip()] = {
                    'Matrisome Category': row['Matrisome Category'],
                    'Matrisome Division': row['Matrisome Division'],
                    'Gene Symbol': row['Gene Symbol']
                }

    # Match unmatched proteins by UniProt ID
    for idx in df_long[unmatched_mask].index:
        protein_id = df_long.loc[idx, 'Protein_ID']
        if protein_id in uniprot_lookup:
            df_level2.loc[idx, 'Matrisome Category'] = uniprot_lookup[protein_id]['Matrisome Category']
            df_level2.loc[idx, 'Matrisome Division'] = uniprot_lookup[protein_id]['Matrisome Division']
            df_level2.loc[idx, 'Gene Symbol'] = uniprot_lookup[protein_id]['Gene Symbol']

    matched_level2 = df_level2['Matrisome Category'].notna().sum() - matched_level1
    log(f"  Additional matches: {matched_level2} rows")

    # Level 3: Synonym match (skip for now - needs synonym list)
    log("\nLevel 3: Synonym match (skipped - would need synonym mapping)")

    # Level 4: Unmatched
    log("\nLevel 4: Unmatched (non-ECM proteins)")
    unmatched_final = df_level2['Matrisome Category'].isna().sum()
    log(f"  Unmatched: {unmatched_final} rows ({unmatched_final/len(df_long)*100:.1f}%)")

    log("\n### Step 2.3: Add annotation columns")

    df_annotated = df_level2.copy()

    # Rename matched columns
    df_annotated.rename(columns={
        'Gene Symbol': 'Canonical_Gene_Symbol'
    }, inplace=True)

    # Fill unmatched with "Non-ECM"
    df_annotated['Canonical_Gene_Symbol'].fillna(df_annotated['Gene_Symbol'], inplace=True)
    df_annotated['Matrisome Category'].fillna('Non-ECM', inplace=True)
    df_annotated['Matrisome Division'].fillna('Non-ECM', inplace=True)

    # Add Match Level
    df_annotated['Match_Level'] = 'Unmatched'
    df_annotated.loc[df_annotated['Matrisome Category'] != 'Non-ECM', 'Match_Level'] = 'Gene_Symbol_or_UniProt'

    # Add Match Confidence (0 = non-ECM, 1 = ECM)
    df_annotated['Match_Confidence'] = 0
    df_annotated.loc[df_annotated['Matrisome Category'] != 'Non-ECM', 'Match_Confidence'] = 1

    log(f"✅ Annotation complete")
    log(f"   ECM proteins: {(df_annotated['Match_Confidence'] == 1).sum()} rows")
    log(f"   Non-ECM: {(df_annotated['Match_Confidence'] == 0).sum()} rows")

    log("\n### Step 2.4: ECM coverage statistics")

    df_ecm = df_annotated[df_annotated['Match_Confidence'] == 1]
    unique_ecm = df_ecm['Protein_ID'].nunique()
    total_proteins = df_annotated['Protein_ID'].nunique()

    log(f"ECM protein coverage:")
    log(f"  Unique ECM proteins: {unique_ecm}")
    log(f"  Total unique proteins: {total_proteins}")
    log(f"  ECM percentage: {unique_ecm/total_proteins*100:.1f}%")

    log("\n### Step 2.5: Save annotated long format")

    output_file = "Dipali_2023_long_annotated.csv"
    df_annotated.to_csv(output_file, index=False)
    log(f"✅ Saved: {output_file}")

    log("\n✅ ECM Annotation complete")

    return df_annotated

if __name__ == '__main__':
    try:
        df_result = main()
        print("\n✅ Success!")
    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(f"\n```\n{traceback.format_exc()}\n```")
        raise
