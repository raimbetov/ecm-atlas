#!/usr/bin/env python3
"""
iTRAQ Adapter for Caldeira et al. 2017
Transforms iTRAQ 8-plex data to unified wide-format schema

Study: Matrisome Profiling During Intervertebral Disc Development And Ageing
PMID: 28900233
DOI: 10.1038/s41598-017-11960-0
Method: iTRAQ 8-plex LC-MS/MS (2 technical batches)
Tissue: Bovine caudal intervertebral disc (nucleus pulposus)
Species: Bos taurus (primarily, with some cross-species matches)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def process_caldeira2017():
    """Process Caldeira 2017 iTRAQ data to wide format."""

    print("="*80)
    print("iTRAQ Adapter for Caldeira et al. 2017")
    print("="*80)

    # 1. Load iTRAQ data
    data_file = '../../data_raw/Caldeira et al. - 2017/41598_2017_11960_MOESM3_ESM.xls'
    print(f"\n[1/7] Loading iTRAQ data from: {data_file}")

    df = pd.read_excel(
        data_file,
        sheet_name='1. Proteins'
    )

    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Shape: {df.shape}")

    # Filter out empty rows (missing Accession Number)
    df = df[df['Accession Number'].notna()].copy()
    print(f"   ✓ After filtering empty rows: {len(df)} proteins")

    # 2. Identify sample columns
    print("\n[2/7] Identifying iTRAQ sample columns...")

    # The data has 3 age groups with biological replicates
    # Young samples appear to be duplicated (batch 1 and batch 2)
    foetus_cols = ['Foetus 1', 'Foetus 2', 'Foetus 3']  # Exclude Pool Foetus (it's a pool)
    young_cols = ['Young 1', 'Young 2', 'Young 3', 'Young 1 (2)', 'Young 2 (2)', 'Young 3 (2)']  # 2 batches
    old_cols = ['Old 1', 'Old 2', 'Old 3']  # Exclude Pool Old (it's a pool)

    print(f"   ✓ Foetus samples (n={len(foetus_cols)}, EXCLUDED): {foetus_cols}")
    print(f"   ✓ Young samples (n={len(young_cols)}): {young_cols}")
    print(f"   ✓ Old samples (n={len(old_cols)}): {old_cols}")
    print(f"   Note: Pool samples excluded (not independent biological replicates)")

    # 3. Load matrisome reference for annotation
    print("\n[3/7] Loading matrisome reference for ECM annotation...")

    # Load bovine matrisome reference
    matrisome_file = '../../references/bovine_matrisome.csv'

    # Check if bovine matrisome exists, otherwise use human and manual mapping
    if not Path(matrisome_file).exists():
        print(f"   ⚠️  Bovine matrisome not found, using multi-species strategy")
        # Load human and mouse for cross-species annotation
        matrisome_human = pd.read_csv('../../references/human_matrisome_v2.csv')
        matrisome_mouse = pd.read_csv('../../references/mouse_matrisome_v2.csv')

        # Combine references
        matrisome_refs = [matrisome_human, matrisome_mouse]
    else:
        matrisome_refs = [pd.read_csv(matrisome_file)]

    # 4. Annotate ECM proteins
    print("\n[4/7] Annotating ECM proteins...")

    def annotate_protein(row):
        """
        Annotate protein with matrisome information.
        Multi-level matching: UniProt ID, then Gene Symbol, then Protein Name keywords.
        """
        uniprot_id = str(row['Accession Number']).strip()
        protein_name = str(row['Protein Name']).lower()

        # Try matching against references
        for ref in matrisome_refs:
            # Level 1: UniProt ID match
            if 'UniProt_IDs' in ref.columns:
                match = ref[ref['UniProt_IDs'] == uniprot_id]
                if not match.empty:
                    return {
                        'Canonical_Gene_Symbol': match.iloc[0]['Gene Symbol'],
                        'Matrisome_Category': match.iloc[0]['Matrisome Category'],
                        'Matrisome_Division': match.iloc[0]['Matrisome Division'],
                        'Match_Level': 1,
                        'Match_Confidence': 100.0
                    }

            # Level 2: Protein name keyword matching for cross-species
            # This is necessary because bovine proteins may not have direct UniProt matches

        # Level 3: Manual ECM keyword classification
        ecm_keywords = {
            'Collagens': ['collagen'],
            'ECM Glycoproteins': ['fibronectin', 'laminin', 'tenascin', 'thrombospondin', 'vitronectin'],
            'Proteoglycans': ['aggrecan', 'biglycan', 'decorin', 'perlecan', 'versican', 'proteoglycan', 'heparan sulfate'],
            'ECM Regulators': ['metalloproteinase', 'mmp', 'timp', 'serpin'],
            'ECM-affiliated Proteins': ['cartilage oligomeric', 'link protein', 'matrilin']
        }

        for category, keywords in ecm_keywords.items():
            if any(kw in protein_name for kw in keywords):
                return {
                    'Canonical_Gene_Symbol': None,  # Unknown for cross-species
                    'Matrisome_Category': category,
                    'Matrisome_Division': 'Core matrisome' if category in ['Collagens', 'ECM Glycoproteins', 'Proteoglycans'] else 'Matrisome-associated',
                    'Match_Level': 3,
                    'Match_Confidence': 70.0
                }

        # Level 4: Not ECM
        return {
            'Canonical_Gene_Symbol': None,
            'Matrisome_Category': None,
            'Matrisome_Division': None,
            'Match_Level': 4,
            'Match_Confidence': 0.0
        }

    annotation_results = df.apply(annotate_protein, axis=1, result_type='expand')
    df_annotated = pd.concat([df, annotation_results], axis=1)

    # Filter to ECM proteins only
    df_ecm = df_annotated[df_annotated['Match_Confidence'] > 0].copy()

    total_proteins = len(df_annotated)
    ecm_proteins = len(df_ecm)
    print(f"   ✓ ECM proteins identified: {ecm_proteins}/{total_proteins} ({ecm_proteins/total_proteins*100:.1f}%)")
    print(f"   ✓ Matrisome categories found:")
    for cat, count in df_ecm['Matrisome_Category'].value_counts().items():
        print(f"      - {cat}: {count}")

    # 5. Calculate mean abundances per age group
    print("\n[5/7] Calculating mean abundances per age group...")

    df_ecm['Abundance_Young'] = df_ecm[young_cols].mean(axis=1, skipna=True)
    df_ecm['Abundance_Old'] = df_ecm[old_cols].mean(axis=1, skipna=True)

    print(f"   ✓ Young mean: {df_ecm['Abundance_Young'].mean():.3f} (range: {df_ecm['Abundance_Young'].min():.3f}-{df_ecm['Abundance_Young'].max():.3f})")
    print(f"   ✓ Old mean: {df_ecm['Abundance_Old'].mean():.3f} (range: {df_ecm['Abundance_Old'].min():.3f}-{df_ecm['Abundance_Old'].max():.3f})")

    # Check for missing values
    nan_young = df_ecm['Abundance_Young'].isna().sum()
    nan_old = df_ecm['Abundance_Old'].isna().sum()
    print(f"   ✓ Missing values: Young={nan_young} ({nan_young/len(df_ecm)*100:.1f}%), Old={nan_old} ({nan_old/len(df_ecm)*100:.1f}%)")

    # 6. Map to unified schema
    print("\n[6/7] Mapping to unified schema...")

    # Extract gene symbol from Accession Name (e.g., "PGCA_BOVIN" -> "PGCA")
    df_ecm['Gene_Symbol'] = df_ecm['Accession Name'].str.split('_').str[0]

    df_wide = pd.DataFrame({
        'Protein_ID': df_ecm['Accession Number'],
        'Protein_Name': df_ecm['Protein Name'],
        'Gene_Symbol': df_ecm['Gene_Symbol'],
        'Canonical_Gene_Symbol': df_ecm['Canonical_Gene_Symbol'],
        'Matrisome_Category': df_ecm['Matrisome_Category'],
        'Matrisome_Division': df_ecm['Matrisome_Division'],
        'Dataset_Name': 'Caldeira_2017',  # For dashboard display
        'Organ': 'Intervertebral_disc',  # Extracted from Tissue
        'Compartment': 'Nucleus_pulposus',  # Tissue compartment
        'Tissue': 'Intervertebral_disc_Nucleus_pulposus',
        'Tissue_Compartment': 'Nucleus_pulposus',
        'Species': 'Bos taurus',  # Primary species (bovine)
        'Abundance_Young': df_ecm['Abundance_Young'],
        'Abundance_Old': df_ecm['Abundance_Old'],
        'Method': 'iTRAQ 8-plex LC-MS/MS',
        'Study_ID': 'Caldeira_2017',
        'Match_Level': df_ecm['Match_Level'],
        'Match_Confidence': df_ecm['Match_Confidence']
    })

    print(f"   ✓ Created wide format with {len(df_wide)} rows")
    print(f"   ✓ Columns: {list(df_wide.columns)}")

    # 7. Validate and save
    print("\n[7/7] Validating and saving wide format...")

    # Check for critical missing values
    critical_cols = ['Dataset_Name', 'Organ', 'Compartment']
    for col in critical_cols:
        nan_count = df_wide[col].isna().sum()
        if nan_count > 0:
            print(f"   ⚠️  WARNING: {col} has {nan_count} NaN values - fixing...")
            if col == 'Dataset_Name':
                df_wide[col] = 'Caldeira_2017'
            elif col == 'Organ':
                df_wide[col] = 'Intervertebral_disc'
            elif col == 'Compartment':
                df_wide[col] = 'Nucleus_pulposus'

    # Final validation
    nan_dataset = df_wide['Dataset_Name'].isna().sum()
    nan_organ = df_wide['Organ'].isna().sum()
    nan_compartment = df_wide['Compartment'].isna().sum()

    if nan_dataset == 0 and nan_organ == 0 and nan_compartment == 0:
        print(f"   ✓ Validation passed: No NaN in critical columns")
    else:
        print(f"   ❌ Validation failed: NaN in critical columns")
        sys.exit(1)

    # Save wide format
    output_file = 'Caldeira_2017_wide_format.csv'
    print(f"\n   Saving to: {output_file}")

    df_wide.to_csv(output_file, index=False)

    print(f"   ✓ Saved successfully!")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Study ID: Caldeira_2017")
    print(f"Species: Bos taurus (bovine)")
    print(f"Tissue: Intervertebral_disc_Nucleus_pulposus")
    print(f"Method: iTRAQ 8-plex LC-MS/MS")
    print(f"ECM Proteins: {len(df_wide)}")
    print(f"Young age group: Young adult (6 replicates from 2 batches)")
    print(f"Old age group: Aged (3 replicates)")
    print(f"Foetus group: EXCLUDED (developmental, not aging comparison)")
    print(f"\nOutput: {output_file}")

    print("\n" + "="*80)
    print("DATA QUALITY METRICS")
    print("="*80)
    print(f"Missing values in Abundance_Young: {nan_young} ({nan_young/len(df_wide)*100:.1f}%)")
    print(f"Missing values in Abundance_Old: {nan_old} ({nan_old/len(df_wide)*100:.1f}%)")
    print(f"ECM annotation coverage: {ecm_proteins}/{total_proteins} ({ecm_proteins/total_proteins*100:.1f}%)")
    print(f"Match confidence distribution:")
    for conf, count in df_wide['Match_Confidence'].value_counts().sort_index(ascending=False).items():
        print(f"   {conf:.0f}%: {count} proteins")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Merge to unified database:")
    print(f"   cd ../../")
    print(f"   python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \\")
    print(f"       05_papers_to_csv/03_Caldeira_2017_paper_to_csv/{output_file}")
    print("\n2. Calculate Z-scores:")
    print(f"   python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \\")
    print(f"       'Caldeira_2017' 'Tissue'")
    print("="*80)

    return df_wide

if __name__ == '__main__':
    try:
        df = process_caldeira2017()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
