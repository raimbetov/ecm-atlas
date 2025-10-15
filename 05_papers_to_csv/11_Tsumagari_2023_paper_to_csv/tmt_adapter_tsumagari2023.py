#!/usr/bin/env python3
"""
TMT Adapter for Tsumagari et al. 2023
Transforms TMT proteomics data to unified wide-format schema

Study: Age-related proteomic changes in mouse brain
PMID: 38086838
Method: TMT 6-plex LC-MS/MS
Tissue: Mouse brain (Cortex and Hippocampus)
Species: Mus musculus
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def load_matrisome_reference(species='Mus musculus'):
    """Load matrisome reference for annotation."""
    if species == 'Mus musculus':
        ref_file = '../../references/mouse_matrisome_v2.csv'
    else:
        ref_file = '../../references/human_matrisome_v2.csv'

    ref = pd.read_csv(ref_file)
    print(f"   Loaded {len(ref)} matrisome proteins from {ref_file}")
    return ref

def annotate_proteins(df, matrisome_ref):
    """
    Annotate proteins with matrisome classification.
    4-level matching hierarchy:
    Level 1: Gene symbol match (confidence 100%)
    Level 2: UniProt ID match (confidence 95%)
    Level 3: Synonym match (confidence 80%)
    Level 4: Unmatched (confidence 0%)
    """

    # Create lookup dictionaries
    ref_by_gene = {}
    for idx, row in matrisome_ref.iterrows():
        gene = str(row['Gene Symbol']).strip().upper()
        ref_by_gene[gene] = row

    ref_by_uniprot = {}
    for idx, row in matrisome_ref.iterrows():
        uniprot = str(row['UniProt_IDs']).strip()
        if uniprot and uniprot != 'nan':
            ref_by_uniprot[uniprot] = row

    # Parse synonyms
    ref_by_synonym = {}
    for idx, row in matrisome_ref.iterrows():
        synonyms = str(row.get('Synonyms', ''))
        if synonyms and synonyms != 'nan':
            for syn in synonyms.split('|'):
                syn_clean = syn.strip().upper()
                ref_by_synonym[syn_clean] = row

    print(f"   Lookup tables: {len(ref_by_gene)} genes, {len(ref_by_uniprot)} UniProt, {len(ref_by_synonym)} synonyms")

    # Annotate each protein
    annotations = []
    for idx, row in df.iterrows():
        gene = str(row['Gene name']).strip().upper() if pd.notna(row['Gene name']) else ''
        uniprot = str(row['UniProt accession']).strip() if pd.notna(row['UniProt accession']) else ''

        matched = None
        match_level = 4
        match_confidence = 0

        # Level 1: Gene symbol
        if gene and gene in ref_by_gene:
            matched = ref_by_gene[gene]
            match_level = 1
            match_confidence = 100
        # Level 2: UniProt
        elif uniprot and uniprot in ref_by_uniprot:
            matched = ref_by_uniprot[uniprot]
            match_level = 2
            match_confidence = 95
        # Level 3: Synonym
        elif gene and gene in ref_by_synonym:
            matched = ref_by_synonym[gene]
            match_level = 3
            match_confidence = 80

        if matched is not None:
            annotations.append({
                'Canonical_Gene_Symbol': matched['Gene Symbol'],
                'Matrisome_Category': matched['Matrisome Category'],
                'Matrisome_Division': matched['Matrisome Division'],
                'Match_Level': match_level,
                'Match_Confidence': match_confidence
            })
        else:
            annotations.append({
                'Canonical_Gene_Symbol': None,
                'Matrisome_Category': None,
                'Matrisome_Division': None,
                'Match_Level': 4,
                'Match_Confidence': 0
            })

    return pd.DataFrame(annotations)

def process_brain_region(data_file, region_name, region_prefix, matrisome_ref):
    """Process one brain region (Cortex or Hippocampus)."""

    print(f"\n{'='*70}")
    print(f"Processing {region_name}")
    print(f"{'='*70}")

    # 1. Load data
    print(f"\n[1/7] Loading TMT data from: {data_file.split('/')[-1]}")
    df = pd.read_excel(data_file, sheet_name='expression')
    print(f"   Loaded {len(df)} proteins")

    # 2. Identify sample columns
    print(f"\n[2/7] Identifying sample columns...")
    # Fix: use 'mo_' not '_mo_' (column names are like 'Cx_3mo_1' not 'Cx_3_mo_1')
    sample_cols = [col for col in df.columns if region_prefix in str(col) and 'mo_' in str(col)]

    age_3mo = [col for col in sample_cols if '3mo_' in col]
    age_15mo = [col for col in sample_cols if '15mo_' in col]
    age_24mo = [col for col in sample_cols if '24mo_' in col]

    print(f"   3 months (Young): n={len(age_3mo)}")
    print(f"   15 months (Middle): n={len(age_15mo)}")
    print(f"   24 months (Old): n={len(age_24mo)}")

    # 3. Calculate mean abundances per age group
    print(f"\n[3/7] Calculating mean abundances per age group...")
    df['Abundance_3mo'] = df[age_3mo].mean(axis=1, skipna=True)
    df['Abundance_15mo'] = df[age_15mo].mean(axis=1, skipna=True)
    df['Abundance_24mo'] = df[age_24mo].mean(axis=1, skipna=True)

    print(f"   3mo mean: {df['Abundance_3mo'].mean():.2f} (range: {df['Abundance_3mo'].min():.2f}-{df['Abundance_3mo'].max():.2f})")
    print(f"   15mo mean: {df['Abundance_15mo'].mean():.2f} (range: {df['Abundance_15mo'].min():.2f}-{df['Abundance_15mo'].max():.2f})")
    print(f"   24mo mean: {df['Abundance_24mo'].mean():.2f} (range: {df['Abundance_24mo'].min():.2f}-{df['Abundance_24mo'].max():.2f})")

    # 4. Map to Young/Old binary groups
    print(f"\n[4/7] Mapping age groups to Young/Old...")
    print(f"   Strategy: 3 months = Young (adult)")
    print(f"            24 months = Old (aged)")
    print(f"            (15 months excluded - middle age)")

    df['Abundance_Young'] = df['Abundance_3mo']
    df['Abundance_Old'] = df['Abundance_24mo']

    # 5. Annotate with matrisome reference
    print(f"\n[5/7] Annotating proteins with matrisome classification...")
    annotations = annotate_proteins(df, matrisome_ref)
    df = pd.concat([df, annotations], axis=1)

    total_proteins = len(df)
    ecm_proteins = (df['Match_Confidence'] > 0).sum()
    print(f"   Annotation coverage: {100*ecm_proteins/total_proteins:.1f}% ({ecm_proteins}/{total_proteins})")

    # 6. Filter to ECM proteins only
    print(f"\n[6/7] Filtering to ECM proteins only...")
    df_ecm = df[df['Match_Confidence'] > 0].copy()
    print(f"   Filtered to {len(df_ecm)} ECM proteins")

    if len(df_ecm) == 0:
        print("   WARNING: No ECM proteins found!")
        return None

    # 7. Map to unified schema
    print(f"\n[7/7] Mapping to unified schema...")

    df_wide = pd.DataFrame({
        'Protein_ID': df_ecm['UniProt accession'],
        'Protein_Name': None,  # Will be enriched if needed
        'Gene_Symbol': df_ecm['Gene name'],
        'Canonical_Gene_Symbol': df_ecm['Canonical_Gene_Symbol'],
        'Matrisome_Category': df_ecm['Matrisome_Category'],
        'Matrisome_Division': df_ecm['Matrisome_Division'],
        'Dataset_Name': 'Tsumagari_2023',  # For dashboard display
        'Organ': 'Brain',  # Extracted from Tissue
        'Compartment': region_name,  # Cortex or Hippocampus
        'Tissue': f'Brain_{region_name}',
        'Tissue_Compartment': region_name,
        'Species': 'Mus musculus',
        'Abundance_Young': df_ecm['Abundance_Young'],
        'Abundance_Old': df_ecm['Abundance_Old'],
        'Method': 'TMT 6-plex LC-MS/MS',
        'Study_ID': 'Tsumagari_2023',
        'Match_Level': df_ecm['Match_Level'],
        'Match_Confidence': df_ecm['Match_Confidence']
    })

    # Check for missing values
    nan_young = df_wide['Abundance_Young'].isna().sum()
    nan_old = df_wide['Abundance_Old'].isna().sum()
    print(f"   Missing values: Young={nan_young} ({100*nan_young/len(df_wide):.1f}%), Old={nan_old} ({100*nan_old/len(df_wide):.1f}%)")

    return df_wide

def process_tsumagari2023():
    """Process Tsumagari 2023 TMT data to wide format."""

    print("="*80)
    print("TMT Adapter for Tsumagari et al. 2023")
    print("="*80)

    # Load matrisome reference
    print("\n[0/7] Loading matrisome reference...")
    matrisome_ref = load_matrisome_reference('Mus musculus')

    # Process both brain regions
    data_files = {
        'Cortex': {
            'file': '../../data_raw/Tsumagari et al. - 2023/41598_2023_45570_MOESM3_ESM.xlsx',
            'prefix': 'Cx_'
        },
        'Hippocampus': {
            'file': '../../data_raw/Tsumagari et al. - 2023/41598_2023_45570_MOESM4_ESM.xlsx',
            'prefix': 'Hipp_'
        }
    }

    all_regions = []
    for region_name, config in data_files.items():
        df_region = process_brain_region(
            config['file'],
            region_name,
            config['prefix'],
            matrisome_ref
        )
        if df_region is not None:
            all_regions.append(df_region)

    # Combine both regions
    print(f"\n{'='*80}")
    print("COMBINING BRAIN REGIONS")
    print(f"{'='*80}")

    df_combined = pd.concat(all_regions, ignore_index=True)
    print(f"\nCombined dataset:")
    print(f"   Total rows: {len(df_combined)}")
    print(f"   Unique proteins: {df_combined['Protein_ID'].nunique()}")
    print(f"   Regions: {df_combined['Tissue_Compartment'].unique().tolist()}")

    # Matrisome category breakdown
    print(f"\nMatrisome categories:")
    for cat, count in df_combined['Matrisome_Category'].value_counts().items():
        print(f"   {cat}: {count}")

    # Save wide format
    output_file = 'Tsumagari_2023_wide_format.csv'
    print(f"\n{'='*80}")
    print(f"SAVING WIDE FORMAT")
    print(f"{'='*80}")

    df_combined.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Validate output
    print(f"\nValidation checks:")
    checks = [
        ('No null Protein_ID', df_combined['Protein_ID'].isna().sum() == 0),
        ('No null Organ', df_combined['Organ'].isna().sum() == 0),
        ('No null Compartment', df_combined['Compartment'].isna().sum() == 0),
        ('No null Dataset_Name', df_combined['Dataset_Name'].isna().sum() == 0),
        ('All rows are ECM proteins', df_combined['Match_Confidence'].min() > 0),
        ('Both regions present', len(df_combined['Tissue_Compartment'].unique()) == 2)
    ]

    all_passed = True
    for check_name, passed in checks:
        status = '✓' if passed else '✗'
        print(f"   [{status}] {check_name}")
        if not passed:
            all_passed = False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Study ID: Tsumagari_2023")
    print(f"Species: Mus musculus")
    print(f"Tissue: Brain (Cortex + Hippocampus)")
    print(f"Method: TMT 6-plex LC-MS/MS")
    print(f"ECM proteins: {len(df_combined)}")
    print(f"Regions: {df_combined['Tissue_Compartment'].unique().tolist()}")
    print(f"Young age group: 3 months (adult, n=6 per region)")
    print(f"Old age group: 24 months (aged, n=6 per region)")
    print(f"\nOutput: {output_file}")
    print(f"Validation: {'PASSED' if all_passed else 'FAILED'}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Merge to unified database:")
    print(f"   cd ../../")
    print(f"   python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \\")
    print(f"       05_papers_to_csv/11_Tsumagari_2023_paper_to_csv/{output_file}")
    print("\n2. Calculate Z-scores:")
    print(f"   python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \\")
    print(f"       'Tsumagari_2023' 'Tissue'")
    print("="*80)

    return df_combined

if __name__ == '__main__':
    try:
        os.chdir('/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/11_Tsumagari_2023_paper_to_csv')
        df = process_tsumagari2023()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
