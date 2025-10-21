#!/usr/bin/env python3
"""
Study Configuration Template
=============================

Fill in study-specific parameters here before running the processing pipeline.

This file serves as a central configuration for PHASE 1 processing.
Copy this template for each new study and fill in the values.
"""

# ==============================================================================
# STUDY METADATA
# ==============================================================================

STUDY_CONFIG = {
    # Study identification
    "study_id": "Randles_2021",  # Format: AuthorLastName_Year
    "paper_pmid": "34049963",    # PubMed ID
    "paper_doi": "10.1681/ASN.2020101442",  # DOI (optional)

    # Biological metadata
    "species": "Homo sapiens",   # Options: "Homo sapiens", "Mus musculus"
    "tissue": "Kidney",          # Main tissue/organ
    "method": "Label-free LC-MS/MS (Progenesis + Mascot)",  # Proteomics method

    # Age groups (customize for your study)
    "young_ages": [15, 29, 37],  # List of ages considered "Young"
    "old_ages": [61, 67, 69],    # List of ages considered "Old"
    "age_unit": "years",         # Options: "years", "months", "weeks"

    # Compartments (if applicable)
    "compartments": {
        "G": "Glomerular",       # Mapping from abbreviation to full name
        "T": "Tubulointerstitial"
    },
    # If no compartments, use: "compartments": None

    # File paths (relative to project root)
    "data_file": "data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx",
    "data_sheet": "Human data matrix fraction",  # Main data sheet name
    "metadata_sheet": None,  # If exists: "Sample information", else None

    # Column mapping (customize for your Excel file)
    "column_mapping": {
        # Protein identifier columns
        "protein_id": "Accession",        # UniProt ID column
        "protein_name": "Description",    # Protein name column
        "gene_symbol": "Gene name",       # Gene symbol column

        # LFQ intensity columns (list all sample columns)
        "intensity_columns": [
            "G15", "T15", "G29", "T29", "G37", "T37",
            "G61", "T61", "G67", "T67", "G69", "T69"
        ],

        # If metadata sheet exists, specify join column
        "metadata_join_column": None  # e.g., "Profile_Name"
    },

    # Sample parsing strategy
    "parse_sample_info": {
        # How to extract age from column name
        # Example: "G15" → compartment="G", age=15
        "method": "regex",  # Options: "regex", "manual", "metadata_sheet"
        "compartment_pattern": r"^([GT])",  # Regex to extract compartment
        "age_pattern": r"(\d+)$",           # Regex to extract age
    },

    # Output directory (relative to project root)
    "output_dir": "05_Randles_paper_to_csv",

    # Reference files (relative to project root)
    "matrisome_reference": "references/human_matrisome_v2.csv",  # Auto-selected based on species
}


# ==============================================================================
# EXAMPLE CONFIGURATIONS FOR DIFFERENT STUDY TYPES
# ==============================================================================

# Example 1: Study with separate metadata sheet (Tam 2020)
TAM_2020_CONFIG = {
    "study_id": "Tam_2020",
    "paper_pmid": "33382035",
    "species": "Homo sapiens",
    "tissue": "Intervertebral disc",
    "method": "Label-free LC-MS/MS (MaxQuant LFQ)",
    "young_ages": [16],
    "old_ages": [59],
    "age_unit": "years",
    "compartments": {
        "NP": "Nucleus Pulposus",
        "IAF": "Inner Annulus Fibrosus",
        "OAF": "Outer Annulus Fibrosus",
        "NP/IAF": "Transition zone"
    },
    "data_file": "data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx",
    "data_sheet": "Raw data",
    "metadata_sheet": "Sample information",  # Separate metadata sheet
    "column_mapping": {
        "protein_id": "T: Majority protein IDs",
        "protein_name": "T: Protein names",
        "gene_symbol": "T: Gene names",
        "intensity_columns": None,  # Auto-detect all LFQ columns
        "metadata_join_column": "Profile"  # Join on this column
    },
    "parse_sample_info": {
        "method": "metadata_sheet",  # Use metadata sheet
    },
    "output_dir": "07_Tam_2020_paper_to_csv",
    "matrisome_reference": "references/human_matrisome_v2.csv",
}


# Example 2: Mouse study
MOUSE_STUDY_CONFIG = {
    "study_id": "Mouse_2024",
    "paper_pmid": "12345678",
    "species": "Mus musculus",  # Mouse
    "tissue": "Lung",
    "method": "TMT labeling",
    "young_ages": [3, 6],  # months
    "old_ages": [18, 24],  # months
    "age_unit": "months",
    "compartments": None,  # No compartments
    "data_file": "data_raw/Mouse et al. - 2024/supplementary_data.xlsx",
    "data_sheet": "Protein abundances",
    "metadata_sheet": None,
    "column_mapping": {
        "protein_id": "UniProt_ID",
        "protein_name": "Protein_Name",
        "gene_symbol": "Gene",
        "intensity_columns": [
            "Young_3mo_1", "Young_3mo_2", "Young_6mo_1", "Young_6mo_2",
            "Old_18mo_1", "Old_18mo_2", "Old_24mo_1", "Old_24mo_2"
        ],
    },
    "parse_sample_info": {
        "method": "regex",
        "age_pattern": r"_(\d+)mo",  # Extract age in months
    },
    "output_dir": "XX_Mouse_2024_paper_to_csv",
    "matrisome_reference": "references/mouse_matrisome_v2.csv",  # Mouse reference
}


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_config(config):
    """
    Validate study configuration before processing.

    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []

    # Required fields
    required = ['study_id', 'species', 'tissue', 'data_file', 'data_sheet']
    for field in required:
        if field not in config or not config[field]:
            errors.append(f"Missing required field: {field}")

    # Species validation
    if config.get('species') not in ['Homo sapiens', 'Mus musculus']:
        errors.append(f"Invalid species: {config.get('species')}")

    # Age groups validation
    if not config.get('young_ages') or not config.get('old_ages'):
        errors.append("Must specify young_ages and old_ages")

    # Data retention check
    young_count = len(config.get('young_ages', []))
    old_count = len(config.get('old_ages', []))
    total = young_count + old_count
    if total > 0 and (young_count + old_count) / total < 0.66:
        errors.append(f"Data retention {(young_count + old_count) / total:.1%} < 66%")

    # File path validation
    from pathlib import Path
    data_file = Path(config.get('data_file', ''))
    if not data_file.exists():
        errors.append(f"Data file not found: {config.get('data_file')}")

    # Auto-select matrisome reference if not specified
    if 'matrisome_reference' not in config:
        if config.get('species') == 'Homo sapiens':
            config['matrisome_reference'] = 'references/human_matrisome_v2.csv'
        elif config.get('species') == 'Mus musculus':
            config['matrisome_reference'] = 'references/mouse_matrisome_v2.csv'

    return len(errors) == 0, errors


def print_config_summary(config):
    """Print a summary of the study configuration."""
    print("=" * 70)
    print("STUDY CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Study ID: {config['study_id']}")
    print(f"Species: {config['species']}")
    print(f"Tissue: {config['tissue']}")
    print(f"Method: {config.get('method', 'Not specified')}")
    print(f"Young ages: {config['young_ages']} {config.get('age_unit', 'years')}")
    print(f"Old ages: {config['old_ages']} {config.get('age_unit', 'years')}")
    print(f"Compartments: {config.get('compartments', 'None')}")
    print(f"Data file: {config['data_file']}")
    print(f"Data sheet: {config['data_sheet']}")
    print(f"Metadata sheet: {config.get('metadata_sheet', 'None')}")
    print(f"Output directory: {config.get('output_dir', 'Not specified')}")
    print("=" * 70)


if __name__ == '__main__':
    # Validate the main configuration
    is_valid, errors = validate_config(STUDY_CONFIG)

    if is_valid:
        print("✅ Configuration is valid")
        print_config_summary(STUDY_CONFIG)
    else:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the errors and try again.")
