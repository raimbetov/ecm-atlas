#!/usr/bin/env python3
"""
TMT Adapter for Ouni et al. 2022
Transforms TMTpro 16-plex data to unified wide-format schema

Study: Proteome-wide and matrisome-specific atlas of the human ovary
PMID: 35341935
Method: DC-MaP + TMTpro 16-plex LC-MS/MS
Tissue: Human ovarian cortex
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def process_ouni2022():
    """Process Ouni 2022 TMT data to wide format."""

    print("="*80)
    print("TMT Adapter for Ouni et al. 2022")
    print("="*80)

    # 1. Load TMT data
    data_file = '../../data_raw/Ouni et al. - 2022/Supp Table 3.xlsx'
    print(f"\n[1/6] Loading TMT data from: {data_file}")

    df = pd.read_excel(
        data_file,
        sheet_name='Matrisome Proteins'
    )

    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Shape: {df.shape}")

    # Filter out empty rows (missing Accession)
    df = df[df['Accession'].notna()].copy()
    print(f"   ✓ After filtering empty rows: {len(df)} matrisome proteins")

    # 2. Define sample columns for each age group
    print("\n[2/6] Identifying TMT sample columns...")

    prepub_cols = [f'Q. Norm. of TOT_prepub{i}' for i in range(1, 6)]
    repro_cols = [f'Q. Norm. of TOT_repro{i}' for i in range(1, 6)]
    meno_cols = [f'Q. Norm. of TOT_meno{i}' for i in range(1, 6)]

    print(f"   ✓ Prepubertal samples (n=5): {prepub_cols[0]} ... {prepub_cols[-1]}")
    print(f"   ✓ Reproductive samples (n=5): {repro_cols[0]} ... {repro_cols[-1]}")
    print(f"   ✓ Menopausal samples (n=5): {meno_cols[0]} ... {meno_cols[-1]}")

    # 3. Calculate means per age group
    print("\n[3/6] Calculating mean abundances per age group...")

    df['Abundance_Prepubertal'] = df[prepub_cols].mean(axis=1)
    df['Abundance_Reproductive'] = df[repro_cols].mean(axis=1)
    df['Abundance_Menopausal'] = df[meno_cols].mean(axis=1)

    print(f"   ✓ Prepubertal mean: {df['Abundance_Prepubertal'].mean():.2f} (range: {df['Abundance_Prepubertal'].min():.2f}-{df['Abundance_Prepubertal'].max():.2f})")
    print(f"   ✓ Reproductive mean: {df['Abundance_Reproductive'].mean():.2f} (range: {df['Abundance_Reproductive'].min():.2f}-{df['Abundance_Reproductive'].max():.2f})")
    print(f"   ✓ Menopausal mean: {df['Abundance_Menopausal'].mean():.2f} (range: {df['Abundance_Menopausal'].min():.2f}-{df['Abundance_Menopausal'].max():.2f})")

    # 4. Map to Young/Old binary groups
    print("\n[4/6] Mapping age groups to Young/Old...")
    print("   Strategy: Reproductive (26±5 years) = Young")
    print("            Menopausal (59±8 years) = Old")
    print("            (Prepubertal excluded for comparability with other studies)")

    df['Abundance_Young'] = df['Abundance_Reproductive']
    df['Abundance_Old'] = df['Abundance_Menopausal']

    # 5. Map to unified schema
    print("\n[5/6] Mapping to unified schema...")

    df_wide = pd.DataFrame({
        'Protein_ID': df['Accession'],
        'Protein_Name': df['EntryName'],
        'Gene_Symbol': df['EntryGeneSymbol'],
        'Canonical_Gene_Symbol': df['EntryGeneSymbol'],  # Already canonical
        'Matrisome_Category': df['Category'],
        'Matrisome_Division': df['Division'],
        'Tissue': 'Ovary_Cortex',
        'Tissue_Compartment': 'Cortex',
        'Species': 'Homo sapiens',
        'Abundance_Young': df['Abundance_Young'],
        'Abundance_Old': df['Abundance_Old'],
        'Method': 'DC-MaP + TMTpro 16-plex',
        'Study_ID': 'Ouni_2022',
        'Match_Level': 1,  # ECM classification already done by authors
        'Match_Confidence': 100.0
    })

    print(f"   ✓ Created wide format with {len(df_wide)} rows")
    print(f"   ✓ Columns: {list(df_wide.columns)}")

    # Check for missing values
    nan_young = df_wide['Abundance_Young'].isna().sum()
    nan_old = df_wide['Abundance_Old'].isna().sum()
    print(f"   ✓ Missing values: Young={nan_young}, Old={nan_old}")

    # 6. Save wide format
    output_file = 'Ouni_2022_wide_format.csv'
    print(f"\n[6/6] Saving wide format to: {output_file}")

    df_wide.to_csv(output_file, index=False)

    print(f"   ✓ Saved successfully!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Study ID: Ouni_2022")
    print(f"Species: Homo sapiens")
    print(f"Tissue: Ovary_Cortex")
    print(f"Method: DC-MaP + TMTpro 16-plex")
    print(f"Proteins: {len(df_wide)}")
    print(f"Young age group: Reproductive (26±5 years, n=5)")
    print(f"Old age group: Menopausal (59±8 years, n=5)")
    print(f"\nOutput: {output_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Merge to unified database:")
    print(f"   cd ../../")
    print(f"   python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \\")
    print(f"       05_papers_to_csv/08_Ouni_2022_paper_to_csv/{output_file}")
    print("\n2. Calculate Z-scores:")
    print(f"   python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \\")
    print(f"       'Ouni_2022' 'Tissue'")
    print("="*80)

    return df_wide

if __name__ == '__main__':
    try:
        df = process_ouni2022()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
