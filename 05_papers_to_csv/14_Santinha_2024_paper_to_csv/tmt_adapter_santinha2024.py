#!/usr/bin/env python3
"""
TMT Adapter for Santinha et al. 2024
Transforms TMT-10plex dual-species cardiac aging data to unified wide-format schema

Study: Dual-species (Mouse + Human) cardiac aging with HGPS model
Tissue: Left Ventricle (LV)
Method: TMT-10plex LC-MS/MS
Data format: Differential expression (logFC, AveExpr) - need to back-calculate abundances

Processing Strategy:
- Process 3 datasets separately:
  1. Mouse Native Tissue (NT) - mmc2
  2. Mouse Decellularized Tissue (DT) - mmc6
  3. Human - mmc5
- Back-calculate Young/Old abundances from logFC and AveExpr
- Annotate with species-specific matrisome references
- Combine into single wide-format CSV with separate Study_IDs per dataset
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data_raw' / 'Santinha et al. - 2024'
OUTPUT_DIR = PROJECT_ROOT / '05_papers_to_csv' / '14_Santinha_2024_paper_to_csv'
REF_DIR = PROJECT_ROOT / 'references'


def load_matrisome_reference(species):
    """Load matrisome reference for species."""
    if species == 'Mus musculus':
        ref_file = REF_DIR / 'mouse_matrisome_v2.csv'
    elif species == 'Homo sapiens':
        ref_file = REF_DIR / 'human_matrisome_v2.csv'
    else:
        raise ValueError(f"Unknown species: {species}")

    if not ref_file.exists():
        raise FileNotFoundError(f"Matrisome reference not found: {ref_file}")

    return pd.read_csv(ref_file)


def back_calculate_abundances(df):
    """
    Back-calculate Young and Old abundances from logFC and AveExpr.

    Given:
        logFC = log2(Old) - log2(Young)
        AveExpr = (log2(Young) + log2(Old)) / 2

    Solve:
        log2(Old) = AveExpr + (logFC / 2)
        log2(Young) = AveExpr - (logFC / 2)

    Returns log2-transformed abundances (suitable for z-score normalization).
    """
    df = df.copy()
    df['Abundance_Young'] = df['AveExpr'] - (df['logFC'] / 2)
    df['Abundance_Old'] = df['AveExpr'] + (df['logFC'] / 2)
    return df


def annotate_ecm_proteins(df, species):
    """
    Annotate proteins with matrisome classification.

    Uses hierarchical matching:
    Level 1: Gene symbol match (confidence 100%)
    Level 2: UniProt ID match (confidence 95%)
    Level 3: Synonym match (confidence 80%)
    Level 4: Unmatched (confidence 0%)
    """
    ref = load_matrisome_reference(species)

    # Create lookup dictionaries
    ref_by_gene = ref.set_index('Gene Symbol').to_dict('index')

    # Handle duplicate UniProt IDs by keeping first occurrence
    ref_dedup = ref.drop_duplicates(subset='UniProt_IDs', keep='first')
    ref_by_uniprot = ref_dedup.set_index('UniProt_IDs').to_dict('index')

    # Parse synonyms (pipe-separated)
    ref_by_synonym = {}
    for idx, row in ref.iterrows():
        if pd.notna(row['Synonyms']):
            for syn in row['Synonyms'].split('|'):
                syn_clean = syn.strip().upper()
                if syn_clean:
                    ref_by_synonym[syn_clean] = row.to_dict()

    # Annotation function
    def annotate_row(row):
        gene = str(row['gene.name']).strip().upper() if pd.notna(row['gene.name']) else ''
        uniprot = str(row['ID']).strip() if pd.notna(row['ID']) else ''

        # Level 1: Gene symbol
        if gene in [g.upper() for g in ref_by_gene.keys()]:
            # Find exact match (case-insensitive)
            for ref_gene in ref_by_gene.keys():
                if ref_gene.upper() == gene:
                    match = ref_by_gene[ref_gene]
                    return pd.Series({
                        'Canonical_Gene_Symbol': ref_gene,
                        'Matrisome_Category': match['Matrisome Category'],
                        'Matrisome_Division': match['Matrisome Division'],
                        'Match_Level': 1,
                        'Match_Confidence': 100.0
                    })

        # Level 2: UniProt ID
        if uniprot in ref_by_uniprot:
            match = ref_by_uniprot[uniprot]
            return pd.Series({
                'Canonical_Gene_Symbol': match['Gene Symbol'],
                'Matrisome_Category': match['Matrisome Category'],
                'Matrisome_Division': match['Matrisome Division'],
                'Match_Level': 2,
                'Match_Confidence': 95.0
            })

        # Level 3: Synonym
        if gene in ref_by_synonym:
            match = ref_by_synonym[gene]
            return pd.Series({
                'Canonical_Gene_Symbol': match['Gene Symbol'],
                'Matrisome_Category': match['Matrisome Category'],
                'Matrisome_Division': match['Matrisome Division'],
                'Match_Level': 3,
                'Match_Confidence': 80.0
            })

        # Level 4: Unmatched
        return pd.Series({
            'Canonical_Gene_Symbol': None,
            'Matrisome_Category': None,
            'Matrisome_Division': None,
            'Match_Level': 4,
            'Match_Confidence': 0.0
        })

    # Apply annotation
    annotation_cols = df.apply(annotate_row, axis=1)
    df = pd.concat([df, annotation_cols], axis=1)

    return df


def process_dataset(file_path, sheet_name, study_id, species, tissue_compartment, organ='Heart'):
    """
    Process a single TMT dataset to wide format.

    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name containing data
        study_id: Unique study identifier
        species: 'Mus musculus' or 'Homo sapiens'
        tissue_compartment: Compartment name (e.g., 'Native_Tissue', 'Decellularized_Tissue')
        organ: Organ name (default 'Heart')

    Returns:
        DataFrame in unified wide format
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING: {study_id}")
    print(f"{'='*70}")

    # Load data
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Loaded {len(df)} proteins from {sheet_name}")

    # Back-calculate abundances
    df = back_calculate_abundances(df)
    print(f"Back-calculated Young and Old abundances from logFC and AveExpr")

    # Annotate ECM proteins
    df = annotate_ecm_proteins(df, species)

    # Filter to ECM proteins only
    df_ecm = df[df['Match_Confidence'] > 0].copy()
    total_proteins = len(df)
    ecm_proteins = len(df_ecm)
    coverage = (ecm_proteins / total_proteins) * 100

    print(f"ECM annotation: {ecm_proteins}/{total_proteins} proteins ({coverage:.1f}%)")
    print(f"  Level 1 (Gene): {(df_ecm['Match_Level'] == 1).sum()}")
    print(f"  Level 2 (UniProt): {(df_ecm['Match_Level'] == 2).sum()}")
    print(f"  Level 3 (Synonym): {(df_ecm['Match_Level'] == 3).sum()}")

    # Map to unified schema
    df_wide = pd.DataFrame({
        'Protein_ID': df_ecm['ID'],
        'Protein_Name': df_ecm['description'],
        'Gene_Symbol': df_ecm['gene.name'],
        'Canonical_Gene_Symbol': df_ecm['Canonical_Gene_Symbol'],
        'Matrisome_Category': df_ecm['Matrisome_Category'],
        'Matrisome_Division': df_ecm['Matrisome_Division'],
        'Dataset_Name': f"{study_id}",
        'Organ': organ,
        'Compartment': tissue_compartment,
        'Tissue': f"{organ}_{tissue_compartment}",
        'Tissue_Compartment': tissue_compartment,
        'Species': species,
        'Abundance_Young': df_ecm['Abundance_Young'],  # Log2-transformed
        'Abundance_Old': df_ecm['Abundance_Old'],      # Log2-transformed
        'Method': 'TMT-10plex LC-MS/MS',
        'Study_ID': study_id,
        'Match_Level': df_ecm['Match_Level'],
        'Match_Confidence': df_ecm['Match_Confidence']
    })

    # Validation
    missing_young = df_wide['Abundance_Young'].isna().sum()
    missing_old = df_wide['Abundance_Old'].isna().sum()

    print(f"\nData quality:")
    print(f"  Total ECM proteins: {len(df_wide)}")
    print(f"  Missing Young: {missing_young} (0% expected for TMT)")
    print(f"  Missing Old: {missing_old} (0% expected for TMT)")
    print(f"  Matrisome categories: {df_wide['Matrisome_Category'].nunique()}")

    return df_wide


def main():
    """Process all Santinha 2024 datasets."""

    print("="*70)
    print("SANTINHA ET AL. 2024 - TMT ADAPTER")
    print("Dual-species cardiac aging proteomics")
    print("="*70)

    # Dataset configurations
    datasets = [
        {
            'file': DATA_DIR / 'mmc2.xlsx',
            'sheet': 'MICE_NT_old_vs_young',
            'study_id': 'Santinha_2024_Mouse_NT',
            'species': 'Mus musculus',
            'compartment': 'Native_Tissue',
            'description': 'Mouse native left ventricle tissue, 3mo vs 20mo'
        },
        {
            'file': DATA_DIR / 'mmc6.xlsx',
            'sheet': 'MICE_DT old_vs_young',
            'study_id': 'Santinha_2024_Mouse_DT',
            'species': 'Mus musculus',
            'compartment': 'Decellularized_Tissue',
            'description': 'Mouse decellularized left ventricle tissue, 3mo vs 20mo'
        },
        {
            'file': DATA_DIR / 'mmc5.xlsx',
            'sheet': 'Human_old vs young',
            'study_id': 'Santinha_2024_Human',
            'species': 'Homo sapiens',
            'compartment': 'Native_Tissue',
            'description': 'Human left ventricle tissue, age information TBD'
        }
    ]

    # Process each dataset
    all_dfs = []
    for config in datasets:
        print(f"\n{config['description']}")
        df = process_dataset(
            file_path=config['file'],
            sheet_name=config['sheet'],
            study_id=config['study_id'],
            species=config['species'],
            tissue_compartment=config['compartment']
        )
        all_dfs.append(df)

    # Combine all datasets
    df_combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*70}")
    print("COMBINED DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total ECM proteins: {len(df_combined)}")
    print(f"Unique proteins: {df_combined['Protein_ID'].nunique()}")
    print(f"Studies: {df_combined['Study_ID'].unique().tolist()}")
    print(f"Species breakdown:")
    for species in df_combined['Species'].unique():
        count = (df_combined['Species'] == species).sum()
        print(f"  {species}: {count} rows")
    print(f"Compartments: {df_combined['Tissue_Compartment'].unique().tolist()}")

    # Save wide format
    output_file = OUTPUT_DIR / 'Santinha_2024_wide_format.csv'
    df_combined.to_csv(output_file, index=False)

    print(f"\n✅ Created wide format: {output_file}")
    print(f"   Total rows: {len(df_combined)}")
    print(f"   Columns: {len(df_combined.columns)}")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("Phase 2: Merge to unified database")
    print(f"  cd {PROJECT_ROOT}")
    print(f"  python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \\")
    print(f"      {output_file.relative_to(PROJECT_ROOT)}")
    print()
    print("Phase 3: Calculate z-scores (run separately per study)")
    print("  python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py 'Santinha_2024_Mouse_NT' 'Tissue'")
    print("  python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py 'Santinha_2024_Mouse_DT' 'Tissue'")
    print("  python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py 'Santinha_2024_Human' 'Tissue'")

    return df_combined


if __name__ == '__main__':
    try:
        df = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
