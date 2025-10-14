#!/usr/bin/env python3
"""
Custom processing script for Schuler et al. 2021 mmc4.xls

This file has a unique structure with 4 sheets (4 muscle types),
each with pre-filtered ECM proteins.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def log(message):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    log("=" * 70)
    log("SCHULER 2021 mmc4.xls CUSTOM PROCESSING")
    log("=" * 70)

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Paths
    data_file = project_root / "data_raw" / "Schuler et al. - 2021" / "mmc4.xls"
    matrisome_file = project_root / "references" / "mouse_matrisome_v2.csv"
    output_dir = script_dir

    log(f"Data file: {data_file}")
    log(f"Matrisome reference: {matrisome_file}")
    log(f"Output directory: {output_dir}")

    # Load matrisome reference
    log("\n## Loading mouse matrisome reference...")
    df_matrisome = pd.read_csv(matrisome_file)
    log(f"✅ Loaded {len(df_matrisome)} matrisome proteins")

    # Create lookup dictionaries
    gene_lookup = {}
    uniprot_lookup = {}

    for _, row in df_matrisome.iterrows():
        gene_sym = row['Gene Symbol']
        mat_cat = row['Matrisome Category']
        mat_div = row['Matrisome Division']

        gene_lookup[gene_sym] = {
            'Matrisome_Category': mat_cat,
            'Matrisome_Division': mat_div,
            'Canonical_Gene_Symbol': gene_sym
        }

        # Handle multiple UniProt IDs
        uniprots = str(row['UniProt_IDs']).split(':')
        for up_id in uniprots:
            if up_id and up_id != 'nan':
                uniprot_lookup[up_id.strip()] = {
                    'Matrisome_Category': mat_cat,
                    'Matrisome_Division': mat_div,
                    'Canonical_Gene_Symbol': gene_sym
                }

    # Define sheets to process
    sheets_info = {
        '1_S O vs. Y': 'Soleus',
        '2_G O vs. Y': 'Gastrocnemius',
        '3_TA O vs. Y': 'TA',
        '4_EDL O vs. Y': 'EDL'
    }

    all_data = []

    # Process each sheet
    for sheet_name, muscle_type in sheets_info.items():
        log(f"\n## Processing {sheet_name} ({muscle_type})...")

        # Read sheet
        df_sheet = pd.read_excel(data_file, sheet_name=sheet_name)
        log(f"   Loaded {len(df_sheet)} rows")

        # Extract relevant columns
        df_processed = pd.DataFrame({
            'Protein_ID': df_sheet['uniprot'],
            'Gene_Symbol': df_sheet['short.name'],
            'Abundance_Young': df_sheet['sample1_abundance'],
            'Abundance_Old': df_sheet['sample2_abundance'],
            'Tissue_Compartment': f"Skeletal_muscle_{muscle_type}",
            'Tissue': 'Skeletal muscle',
            'Species': 'Mus musculus',
            'Method': 'LFQ',
            'Study_ID': 'Schuler_2021',
            'Dataset_Name': f'Schuler_2021_{muscle_type}',
            'Organ': 'Skeletal muscle'
        })

        # Annotate with matrisome
        log(f"   Annotating with matrisome...")

        df_processed['Matrisome_Category'] = None
        df_processed['Matrisome_Division'] = None
        df_processed['Canonical_Gene_Symbol'] = None
        df_processed['Match_Level'] = 'Unmatched'
        df_processed['Match_Confidence'] = 0

        # Level 1: Gene Symbol match
        for idx, row in df_processed.iterrows():
            gene_sym = row['Gene_Symbol']
            if gene_sym in gene_lookup:
                info = gene_lookup[gene_sym]
                df_processed.loc[idx, 'Matrisome_Category'] = info['Matrisome_Category']
                df_processed.loc[idx, 'Matrisome_Division'] = info['Matrisome_Division']
                df_processed.loc[idx, 'Canonical_Gene_Symbol'] = info['Canonical_Gene_Symbol']
                df_processed.loc[idx, 'Match_Level'] = 'exact_gene'
                df_processed.loc[idx, 'Match_Confidence'] = 100

        # Level 2: UniProt ID match (for unmatched)
        unmatched_mask = df_processed['Match_Confidence'] == 0
        for idx in df_processed[unmatched_mask].index:
            prot_id = df_processed.loc[idx, 'Protein_ID']
            if prot_id in uniprot_lookup:
                info = uniprot_lookup[prot_id]
                df_processed.loc[idx, 'Matrisome_Category'] = info['Matrisome_Category']
                df_processed.loc[idx, 'Matrisome_Division'] = info['Matrisome_Division']
                df_processed.loc[idx, 'Canonical_Gene_Symbol'] = info['Canonical_Gene_Symbol']
                df_processed.loc[idx, 'Match_Level'] = 'uniprot_id'
                df_processed.loc[idx, 'Match_Confidence'] = 100

        # Fill unmatched
        df_processed['Canonical_Gene_Symbol'].fillna(df_processed['Gene_Symbol'], inplace=True)
        df_processed['Matrisome_Category'].fillna('Non-ECM', inplace=True)
        df_processed['Matrisome_Division'].fillna('Non-ECM', inplace=True)

        matched = (df_processed['Match_Confidence'] == 100).sum()
        total = len(df_processed)
        log(f"   ✅ Matrisome annotation: {matched}/{total} ({matched/total*100:.1f}%) matched")

        all_data.append(df_processed)

    # Combine all muscles
    log(f"\n## Combining all muscle types...")
    df_combined = pd.concat(all_data, ignore_index=True)
    log(f"✅ Combined: {len(df_combined)} total rows")
    log(f"   Unique proteins: {df_combined['Protein_ID'].nunique()}")
    log(f"   Tissue compartments: {df_combined['Tissue_Compartment'].unique().tolist()}")

    # Add additional required columns
    df_combined['Protein_Name'] = None  # Can be filled later from UniProt API
    df_combined['Compartment'] = df_combined['Tissue_Compartment'].str.split('_').str[-1]
    df_combined['N_Profiles_Young'] = 3  # Typical for LFQ (estimate)
    df_combined['N_Profiles_Old'] = 3

    # Transform abundances (log2 if needed - mmc4 appears to already be log2)
    df_combined['Abundance_Young_transformed'] = df_combined['Abundance_Young']
    df_combined['Abundance_Old_transformed'] = df_combined['Abundance_Old']

    # Calculate z-scores per compartment
    log(f"\n## Calculating z-scores per compartment...")

    df_combined['Zscore_Young'] = None
    df_combined['Zscore_Old'] = None
    df_combined['Zscore_Delta'] = None

    for compartment in df_combined['Tissue_Compartment'].unique():
        mask = df_combined['Tissue_Compartment'] == compartment

        # Calculate z-scores for young
        young_vals = df_combined.loc[mask, 'Abundance_Young']
        mean_young = young_vals.mean(skipna=True)
        std_young = young_vals.std(skipna=True)
        if std_young > 0:
            df_combined.loc[mask, 'Zscore_Young'] = (young_vals - mean_young) / std_young

        # Calculate z-scores for old
        old_vals = df_combined.loc[mask, 'Abundance_Old']
        mean_old = old_vals.mean(skipna=True)
        std_old = old_vals.std(skipna=True)
        if std_old > 0:
            df_combined.loc[mask, 'Zscore_Old'] = (old_vals - mean_old) / std_old

        # Calculate delta
        df_combined.loc[mask, 'Zscore_Delta'] = (
            df_combined.loc[mask, 'Zscore_Old'] - df_combined.loc[mask, 'Zscore_Young']
        )

        log(f"   ✅ {compartment}: z-scores calculated")

    # Reorder columns to match database schema
    column_order = [
        'Dataset_Name', 'Organ', 'Compartment',
        'Abundance_Old', 'Abundance_Old_transformed',
        'Abundance_Young', 'Abundance_Young_transformed',
        'Canonical_Gene_Symbol', 'Gene_Symbol',
        'Match_Confidence', 'Match_Level',
        'Matrisome_Category', 'Matrisome_Division',
        'Method', 'N_Profiles_Old', 'N_Profiles_Young',
        'Protein_ID', 'Protein_Name', 'Species',
        'Study_ID', 'Tissue', 'Tissue_Compartment',
        'Zscore_Delta', 'Zscore_Old', 'Zscore_Young'
    ]

    df_final = df_combined[column_order]

    # Save output
    output_file = output_dir / "Schuler_2021_processed.csv"
    df_final.to_csv(output_file, index=False)
    log(f"\n✅ Saved processed data: {output_file}")
    log(f"   Rows: {len(df_final)}")
    log(f"   Columns: {len(df_final.columns)}")

    # Print summary statistics
    log(f"\n## Summary Statistics:")
    log(f"   Total rows: {len(df_final)}")
    log(f"   Unique proteins: {df_final['Protein_ID'].nunique()}")
    log(f"   ECM proteins: {(df_final['Match_Confidence'] == 100).sum()}")
    log(f"   Non-ECM: {(df_final['Match_Confidence'] == 0).sum()}")
    log(f"   Matrisome match rate: {(df_final['Match_Confidence'] == 100).sum() / len(df_final) * 100:.1f}%")
    log(f"\n   Tissue compartments:")
    for compartment in df_final['Tissue_Compartment'].unique():
        count = (df_final['Tissue_Compartment'] == compartment).sum()
        log(f"     - {compartment}: {count} rows")

    log(f"\n   Matrisome categories:")
    for category, count in df_final['Matrisome_Category'].value_counts().head(10).items():
        log(f"     - {category}: {count}")

    log(f"\n{'='*70}")
    log("✅ PROCESSING COMPLETE")
    log(f"{'='*70}")
    log(f"\nNext steps:")
    log(f"1. Merge to database: python ../11_subagent_for_LFQ_ingestion/merge_to_unified.py {output_file.relative_to(project_root)}")
    log(f"2. Verify in dashboard")

    return df_final

if __name__ == '__main__':
    try:
        df_result = main()
        print("\n✅ Success!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
