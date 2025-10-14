#!/usr/bin/env python3
"""
Phase 4: Protein Annotation for Tam 2020 Dataset

This script annotates proteins from the Tam 2020 dataset with matrisome information
using a hierarchical matching strategy against the human matrisome reference.

Matching Strategy:
- Level 1: Exact gene symbol match (confidence 100%)
- Level 2: UniProt ID match (confidence 95%)
- Level 3: Synonym match (confidence 80%)
- Level 4: Unmatched (confidence 0)

Target: ≥90% annotation coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import warnings

# Constants
STANDARDIZED_FILE = "Tam_2020_standardized.csv"
REFERENCE_FILE = "../references/human_matrisome_v2.csv"
OUTPUT_FILE = "Tam_2020_annotated.csv"
REPORT_FILE = "Tam_2020_annotation_report.md"
TARGET_COVERAGE = 0.90

# Known markers for validation
# Note: In the reference file, "Matrisome Category" is the specific type (e.g., "Collagens")
# and "Matrisome Division" is the broader category (e.g., "Core matrisome")
KNOWN_MARKERS = {
    "COL1A1": {"category": "Collagens", "division": "Core matrisome"},
    "COL2A1": {"category": "Collagens", "division": "Core matrisome"},
    "FN1": {"category": "ECM Glycoproteins", "division": "Core matrisome"},
    "ACAN": {"category": "Proteoglycans", "division": "Core matrisome"},
    "MMP2": {"category": "ECM Regulators", "division": "Matrisome-associated"}
}


def load_standardized_data(file_path: str) -> pd.DataFrame:
    """Load the standardized Tam 2020 dataset."""
    print(f"Loading standardized data from {file_path}...")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Standardized file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    return df


def load_matrisome_reference(file_path: str) -> pd.DataFrame:
    """Load the human matrisome reference dataset."""
    print(f"\nLoading matrisome reference from {file_path}...")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Reference file not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} matrisome proteins")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    return df


def build_lookup_dictionaries(reference_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    Build lookup dictionaries for gene symbols, UniProt IDs, and synonyms.

    Returns:
        Tuple of (gene_symbol_dict, uniprot_dict, synonym_dict)
    """
    print("\nBuilding lookup dictionaries...")

    gene_symbol_dict = {}
    uniprot_dict = {}
    synonym_dict = {}

    for idx, row in reference_df.iterrows():
        gene_symbol = str(row['Gene Symbol']).strip().upper()
        matrisome_info = {
            'Canonical_Gene_Symbol': row['Gene Symbol'],
            'Matrisome_Category': row['Matrisome Category'],
            'Matrisome_Division': row['Matrisome Division']
        }

        # Level 1: Gene symbol lookup (case-insensitive)
        gene_symbol_dict[gene_symbol] = matrisome_info

        # Level 2: UniProt ID lookup
        # Handle colon-separated UniProt IDs in reference
        uniprot_ids = str(row['UniProt_IDs']).strip()
        if uniprot_ids and uniprot_ids != 'nan':
            for uniprot_id in uniprot_ids.split(':'):
                uniprot_id = uniprot_id.strip().upper()
                if uniprot_id:
                    uniprot_dict[uniprot_id] = matrisome_info

        # Level 3: Synonym lookup
        # Handle pipe-separated synonyms in reference
        synonyms = str(row['Synonyms']).strip()
        if synonyms and synonyms != 'nan':
            for synonym in synonyms.split('|'):
                synonym = synonym.strip().upper()
                if synonym and synonym != gene_symbol:
                    synonym_dict[synonym] = matrisome_info

    print(f"  Gene symbols: {len(gene_symbol_dict)}")
    print(f"  UniProt IDs: {len(uniprot_dict)}")
    print(f"  Synonyms: {len(synonym_dict)}")

    return gene_symbol_dict, uniprot_dict, synonym_dict


def match_protein(
    gene_symbol: str,
    uniprot_ids: str,
    gene_symbol_dict: Dict,
    uniprot_dict: Dict,
    synonym_dict: Dict
) -> Tuple[str, str, str, str, int]:
    """
    Match a protein using hierarchical strategy.

    Returns:
        Tuple of (canonical_gene_symbol, category, division, match_level, confidence)
    """
    # Normalize inputs
    gene_symbol_norm = str(gene_symbol).strip().upper()
    uniprot_ids_norm = str(uniprot_ids).strip().upper()

    # Level 1: Exact gene symbol match (confidence 100%)
    if gene_symbol_norm in gene_symbol_dict:
        info = gene_symbol_dict[gene_symbol_norm]
        return (
            info['Canonical_Gene_Symbol'],
            info['Matrisome_Category'],
            info['Matrisome_Division'],
            'Level 1: Gene Symbol',
            100
        )

    # Level 2: UniProt ID match (confidence 95%)
    # Handle semicolon-separated UniProt IDs in dataset
    if uniprot_ids_norm and uniprot_ids_norm != 'NAN':
        for uniprot_id in uniprot_ids_norm.split(';'):
            uniprot_id = uniprot_id.strip()
            if uniprot_id in uniprot_dict:
                info = uniprot_dict[uniprot_id]
                return (
                    info['Canonical_Gene_Symbol'],
                    info['Matrisome_Category'],
                    info['Matrisome_Division'],
                    'Level 2: UniProt ID',
                    95
                )

    # Level 3: Synonym match (confidence 80%)
    if gene_symbol_norm in synonym_dict:
        info = synonym_dict[gene_symbol_norm]
        return (
            info['Canonical_Gene_Symbol'],
            info['Matrisome_Category'],
            info['Matrisome_Division'],
            'Level 3: Synonym',
            80
        )

    # Level 4: Unmatched (confidence 0)
    return (gene_symbol, '', '', 'Level 4: Unmatched', 0)


def annotate_proteins(
    df: pd.DataFrame,
    gene_symbol_dict: Dict,
    uniprot_dict: Dict,
    synonym_dict: Dict
) -> pd.DataFrame:
    """Apply hierarchical matching to annotate all proteins."""
    print("\nAnnotating proteins...")

    # Initialize annotation columns
    df['Canonical_Gene_Symbol'] = ''
    df['Matrisome_Category'] = ''
    df['Matrisome_Division'] = ''
    df['Match_Level'] = ''
    df['Match_Confidence'] = 0

    # Apply matching for each row
    for idx, row in df.iterrows():
        canonical, category, division, level, confidence = match_protein(
            row['Gene_Symbol'],
            row['Protein_ID'],
            gene_symbol_dict,
            uniprot_dict,
            synonym_dict
        )

        df.at[idx, 'Canonical_Gene_Symbol'] = canonical
        df.at[idx, 'Matrisome_Category'] = category
        df.at[idx, 'Matrisome_Division'] = division
        df.at[idx, 'Match_Level'] = level
        df.at[idx, 'Match_Confidence'] = confidence

    print(f"  Annotated {len(df)} proteins")

    return df


def calculate_coverage_statistics(df: pd.DataFrame) -> Dict:
    """Calculate annotation coverage statistics."""
    print("\nCalculating coverage statistics...")

    total_proteins = len(df)
    matched_proteins = len(df[df['Match_Confidence'] > 0])
    coverage_rate = matched_proteins / total_proteins if total_proteins > 0 else 0

    # Count by match level
    level_counts = df['Match_Level'].value_counts().to_dict()

    # Count by category
    category_counts = df[df['Matrisome_Category'] != '']['Matrisome_Category'].value_counts().to_dict()

    # Count by division
    division_counts = df[df['Matrisome_Division'] != '']['Matrisome_Division'].value_counts().to_dict()

    stats = {
        'total_proteins': total_proteins,
        'matched_proteins': matched_proteins,
        'unmatched_proteins': total_proteins - matched_proteins,
        'coverage_rate': coverage_rate,
        'level_counts': level_counts,
        'category_counts': category_counts,
        'division_counts': division_counts
    }

    print(f"  Total proteins: {total_proteins}")
    print(f"  Matched: {matched_proteins} ({coverage_rate:.1%})")
    print(f"  Unmatched: {total_proteins - matched_proteins}")

    return stats


def validate_known_markers(df: pd.DataFrame) -> Dict:
    """Validate that known markers are correctly annotated."""
    print("\nValidating known markers...")

    validation_results = {}

    for marker, expected in KNOWN_MARKERS.items():
        # Find the marker in the dataset (case-insensitive)
        marker_rows = df[df['Gene_Symbol'].str.upper() == marker.upper()]

        if len(marker_rows) == 0:
            validation_results[marker] = {
                'found': False,
                'status': 'NOT_FOUND',
                'message': f"Marker {marker} not found in dataset"
            }
            print(f"  {marker}: NOT FOUND in dataset")
        else:
            marker_row = marker_rows.iloc[0]
            category_match = marker_row['Matrisome_Category'] == expected['category']
            division_match = marker_row['Matrisome_Division'] == expected['division']

            if category_match and division_match:
                validation_results[marker] = {
                    'found': True,
                    'status': 'PASS',
                    'category': marker_row['Matrisome_Category'],
                    'division': marker_row['Matrisome_Division'],
                    'match_level': marker_row['Match_Level'],
                    'message': f"Correctly annotated"
                }
                print(f"  {marker}: PASS ({marker_row['Match_Level']})")
            else:
                validation_results[marker] = {
                    'found': True,
                    'status': 'FAIL',
                    'expected_category': expected['category'],
                    'actual_category': marker_row['Matrisome_Category'],
                    'expected_division': expected['division'],
                    'actual_division': marker_row['Matrisome_Division'],
                    'message': f"Annotation mismatch"
                }
                print(f"  {marker}: FAIL - Expected {expected['category']}/{expected['division']}, "
                      f"Got {marker_row['Matrisome_Category']}/{marker_row['Matrisome_Division']}")

    return validation_results


def generate_annotation_report(
    stats: Dict,
    validation_results: Dict,
    output_file: str
) -> None:
    """Generate a markdown annotation report."""
    print(f"\nGenerating annotation report: {output_file}...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Tam 2020 Protein Annotation Report

**Generated:** {timestamp}

## Summary

- **Total Proteins:** {stats['total_proteins']:,}
- **Matched Proteins:** {stats['matched_proteins']:,}
- **Unmatched Proteins:** {stats['unmatched_proteins']:,}
- **Coverage Rate:** {stats['coverage_rate']:.2%}
- **Target Coverage:** {TARGET_COVERAGE:.0%}
- **Status:** {"✅ PASS" if stats['coverage_rate'] >= TARGET_COVERAGE else "⚠️ WARNING - Below target"}

## Match Level Distribution

| Match Level | Count | Percentage |
|-------------|-------|------------|
"""

    for level in ['Level 1: Gene Symbol', 'Level 2: UniProt ID', 'Level 3: Synonym', 'Level 4: Unmatched']:
        count = stats['level_counts'].get(level, 0)
        percentage = count / stats['total_proteins'] * 100 if stats['total_proteins'] > 0 else 0
        report += f"| {level} | {count:,} | {percentage:.1f}% |\n"

    report += f"""
## Matrisome Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
"""

    for category, count in sorted(stats['category_counts'].items()):
        percentage = count / stats['matched_proteins'] * 100 if stats['matched_proteins'] > 0 else 0
        report += f"| {category} | {count:,} | {percentage:.1f}% |\n"

    report += f"""
## Matrisome Division Distribution

| Division | Count | Percentage |
|----------|-------|------------|
"""

    for division, count in sorted(stats['division_counts'].items()):
        percentage = count / stats['matched_proteins'] * 100 if stats['matched_proteins'] > 0 else 0
        report += f"| {division} | {count:,} | {percentage:.1f}% |\n"

    report += """
## Known Marker Validation

| Marker | Status | Category | Division | Match Level | Notes |
|--------|--------|----------|----------|-------------|-------|
"""

    for marker, result in validation_results.items():
        if result['found']:
            if result['status'] == 'PASS':
                report += f"| {marker} | ✅ PASS | {result['category']} | {result['division']} | {result['match_level']} | {result['message']} |\n"
            else:
                report += f"| {marker} | ❌ FAIL | {result['actual_category']} | {result['actual_division']} | - | Expected: {result['expected_category']}/{result['expected_division']} |\n"
        else:
            report += f"| {marker} | ⚠️ NOT FOUND | - | - | - | {result['message']} |\n"

    report += """
## Methodology

### Hierarchical Matching Strategy

1. **Level 1: Gene Symbol Match (100% confidence)**
   - Exact match on gene symbol (case-insensitive)

2. **Level 2: UniProt ID Match (95% confidence)**
   - Match on UniProt accession numbers
   - Handles multiple IDs (semicolon-separated in dataset, colon-separated in reference)

3. **Level 3: Synonym Match (80% confidence)**
   - Match on known gene synonyms
   - Pipe-separated in reference

4. **Level 4: Unmatched (0% confidence)**
   - No match found in reference

### Data Sources

- **Dataset:** Tam_2020_standardized.csv
- **Reference:** human_matrisome_v2.csv

### Quality Metrics

- **Target Coverage:** ≥90% of proteins annotated
- **Validation:** Known ECM markers correctly classified
"""

    if stats['coverage_rate'] < TARGET_COVERAGE:
        report += f"""
## ⚠️ Coverage Warning

The annotation coverage ({stats['coverage_rate']:.2%}) is below the target threshold ({TARGET_COVERAGE:.0%}).

**Unmatched Proteins:** {stats['unmatched_proteins']:,}

This may indicate:
- Novel ECM proteins not in the reference database
- Non-ECM proteins in the dataset
- Gene symbol mismatches requiring manual curation
- Dataset-specific naming conventions

**Recommendation:** Review unmatched proteins for manual curation.
"""

    # Write report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"  Report saved to {output_file}")


def save_annotated_data(df: pd.DataFrame, output_file: str) -> None:
    """Save the annotated dataset to CSV."""
    print(f"\nSaving annotated data to {output_file}...")

    df.to_csv(output_file, index=False)
    print(f"  Saved {len(df)} rows")
    print(f"  Columns: {', '.join(df.columns.tolist())}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Phase 4: Protein Annotation - Tam 2020 Dataset")
    print("=" * 80)

    try:
        # Load data
        standardized_df = load_standardized_data(STANDARDIZED_FILE)
        reference_df = load_matrisome_reference(REFERENCE_FILE)

        # Build lookup dictionaries
        gene_symbol_dict, uniprot_dict, synonym_dict = build_lookup_dictionaries(reference_df)

        # Annotate proteins
        annotated_df = annotate_proteins(
            standardized_df,
            gene_symbol_dict,
            uniprot_dict,
            synonym_dict
        )

        # Calculate coverage statistics
        stats = calculate_coverage_statistics(annotated_df)

        # Validate known markers
        validation_results = validate_known_markers(annotated_df)

        # Generate report
        generate_annotation_report(stats, validation_results, REPORT_FILE)

        # Save annotated data
        save_annotated_data(annotated_df, OUTPUT_FILE)

        # Final status
        print("\n" + "=" * 80)
        if stats['coverage_rate'] >= TARGET_COVERAGE:
            print(f"✅ SUCCESS: Annotation coverage {stats['coverage_rate']:.2%} meets target {TARGET_COVERAGE:.0%}")
        else:
            print(f"⚠️ WARNING: Annotation coverage {stats['coverage_rate']:.2%} below target {TARGET_COVERAGE:.0%}")
            warnings.warn(
                f"Annotation coverage ({stats['coverage_rate']:.2%}) is below target ({TARGET_COVERAGE:.0%}). "
                f"Review {REPORT_FILE} for details."
            )

        # Check validation results
        failed_validations = [m for m, r in validation_results.items() if r['status'] != 'PASS']
        if failed_validations:
            print(f"⚠️ WARNING: {len(failed_validations)} known marker(s) failed validation: {', '.join(failed_validations)}")
        else:
            print("✅ All known markers validated successfully")

        print("=" * 80)
        print(f"\nOutputs:")
        print(f"  - Annotated data: {OUTPUT_FILE}")
        print(f"  - Annotation report: {REPORT_FILE}")
        print("\nPhase 4 complete!")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
