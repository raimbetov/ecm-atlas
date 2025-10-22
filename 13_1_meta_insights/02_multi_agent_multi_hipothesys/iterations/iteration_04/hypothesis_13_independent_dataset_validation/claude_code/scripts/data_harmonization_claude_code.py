#!/usr/bin/env python3
"""
H13 Independent Dataset Validation - Data Harmonization Script
Agent: claude_code

Purpose: Process external datasets into unified z-score format matching our merged_ecm_aging_zscore.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import json
from typing import Dict, List, Tuple
import sys

# Add path to universal zscore function
sys.path.append("/Users/Kravtsovd/projects/ecm-atlas")

#============================================================================
# CONFIGURATION
#============================================================================

WORKSPACE = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code")
EXTERNAL_DATA_DIR = WORKSPACE / "external_datasets"

# Our ECM gene list (648 genes from merged dataset)
OUR_MERGED_DATA = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# Target schema
TARGET_COLUMNS = [
    "Protein_ID", "Gene_Symbol", "Tissue", "Species", "Age", "Age_Group",
    "Abundance", "Z_score", "Study_ID", "Compartment", "Method"
]

#============================================================================
# UNIPROT MAPPING
#============================================================================

def map_protein_ids_to_genes(protein_ids: List[str], species: str = "human") -> Dict[str, str]:
    """
    Map UniProt IDs to Gene Symbols using UniProt API
    """
    print(f"\nMapping {len(protein_ids)} protein IDs to gene symbols...")

    # UniProt ID mapping API
    url = "https://rest.uniprot.org/idmapping/run"

    # Batch process (max 500 per request)
    batch_size = 500
    mapping = {}

    for i in range(0, len(protein_ids), batch_size):
        batch = protein_ids[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{len(protein_ids)//batch_size + 1}...")

        try:
            # Submit job
            response = requests.post(
                url,
                data={
                    "ids": ",".join(batch),
                    "from": "UniProtKB_AC-ID",
                    "to": "Gene_Name"
                }
            )

            if response.status_code == 200:
                job_id = response.json()["jobId"]

                # Poll for results
                result_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
                result_response = requests.get(result_url)

                if result_response.status_code == 200:
                    results = result_response.json().get("results", [])
                    for item in results:
                        from_id = item["from"]
                        to_gene = item.get("to", {}).get("geneName", "UNKNOWN")
                        mapping[from_id] = to_gene

        except Exception as e:
            print(f"    Warning: Failed to map batch: {e}")

    print(f"  Successfully mapped {len(mapping)}/{len(protein_ids)} IDs")
    return mapping

#============================================================================
# Z-SCORE CALCULATION (MATCHING OUR METHOD)
#============================================================================

def calculate_zscore_universal(
    df: pd.DataFrame,
    value_col: str = "Abundance",
    group_by: List[str] = ["Gene_Symbol", "Tissue"]
) -> pd.DataFrame:
    """
    Calculate z-scores using same method as our merged dataset

    Z = (X - mean) / std per protein per tissue
    Handles NaN and zero values appropriately
    """
    print("\nCalculating z-scores...")

    df = df.copy()
    df["Z_score"] = np.nan

    for group_keys, group_df in df.groupby(group_by):
        values = group_df[value_col]

        # Remove NaN for mean/std calculation
        non_nan_values = values.dropna()

        if len(non_nan_values) >= 2:
            mean_val = non_nan_values.mean()
            std_val = non_nan_values.std()

            if std_val > 0:
                z_scores = (values - mean_val) / std_val
                df.loc[group_df.index, "Z_score"] = z_scores
            else:
                # Zero variance - all values same
                df.loc[group_df.index, "Z_score"] = 0.0

    return df

#============================================================================
# DATASET-SPECIFIC PROCESSORS
#============================================================================

def process_pxd011967_muscle(input_file: Path) -> pd.DataFrame:
    """
    Process PXD011967: Discovery proteomics in aging human skeletal muscle

    Expected format: Excel file with TMT abundances
    Columns: Protein IDs, Gene Symbols, TMT reporter intensities, Sample annotations
    """
    print("\n" + "="*80)
    print("PROCESSING PXD011967: Skeletal Muscle Aging")
    print("="*80)

    # Read supplementary data
    # Adjust based on actual file structure
    df = pd.read_excel(input_file, sheet_name=0)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)[:10]}...")

    # Expected structure (adjust based on actual data):
    # - Protein ID column
    # - Gene Symbol column
    # - TMT abundance columns (one per sample)
    # - Age group annotation

    # Melt to long format
    # df_long = df.melt(
    #     id_vars=["Protein_ID", "Gene_Symbol"],
    #     var_name="Sample_ID",
    #     value_name="Abundance"
    # )

    # Add metadata
    # Age groups from paper: 20-34, 35-49, 50-64, 65-79, 80+
    # age_mapping = {...}  # Sample ID -> Age group

    # df_long["Tissue"] = "Muscle"
    # df_long["Species"] = "Human"
    # df_long["Study_ID"] = "PXD011967"
    # df_long["Method"] = "TMT"

    # Calculate z-scores
    # df_long = calculate_zscore_universal(df_long)

    # return df_long

    print("⚠️  PLACEHOLDER: Awaiting actual data file structure")
    print("   Returning empty DataFrame")
    return pd.DataFrame(columns=TARGET_COLUMNS)

def process_pxd015982_skin(input_file: Path) -> pd.DataFrame:
    """
    Process PXD015982: Alterations in ECM composition during skin aging

    Expected format: Quantification table with QconCAT absolute abundances
    Young vs Aged groups, 3 anatomical sites
    """
    print("\n" + "="*80)
    print("PROCESSING PXD015982: Skin ECM Aging")
    print("="*80)

    # Read data
    # df = pd.read_csv(input_file)  # or read_excel

    # Expected: ECM proteins with absolute abundances
    # Groups: Young (26.7 yr), Aged (84.0 yr)
    # Sites: Hip, Underarm, Forearm

    # Process to long format
    # df_long = ...

    # Add metadata
    # df_long["Tissue"] = "Skin"
    # df_long["Species"] = "Human"
    # df_long["Study_ID"] = "PXD015982"
    # df_long["Method"] = "QconCAT"

    # Calculate z-scores
    # df_long = calculate_zscore_universal(df_long)

    # return df_long

    print("⚠️  PLACEHOLDER: Awaiting actual data file structure")
    return pd.DataFrame(columns=TARGET_COLUMNS)

#============================================================================
# GENE OVERLAP ANALYSIS
#============================================================================

def analyze_gene_overlap(external_df: pd.DataFrame, our_df: pd.DataFrame) -> Dict:
    """
    Calculate overlap between external dataset and our 648 ECM genes
    """
    external_genes = set(external_df["Gene_Symbol"].dropna().unique())
    our_genes = set(our_df["Gene_Symbol"].dropna().unique())

    overlap = external_genes & our_genes
    external_only = external_genes - our_genes
    our_only = our_genes - external_genes

    overlap_pct = len(overlap) / len(our_genes) * 100 if len(our_genes) > 0 else 0

    return {
        "n_external_genes": len(external_genes),
        "n_our_genes": len(our_genes),
        "n_overlap": len(overlap),
        "overlap_percentage": overlap_pct,
        "overlap_genes": sorted(list(overlap)),
        "external_only": sorted(list(external_only)),
        "target_met": overlap_pct >= 70  # Success criterion: ≥70% overlap
    }

#============================================================================
# MAIN HARMONIZATION PIPELINE
#============================================================================

def harmonize_external_dataset(
    dataset_id: str,
    input_file: Path,
    output_dir: Path
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main harmonization pipeline for external dataset
    """
    print("\n" + "="*80)
    print(f"HARMONIZING EXTERNAL DATASET: {dataset_id}")
    print("="*80)

    # Process dataset (dataset-specific)
    if dataset_id == "PXD011967":
        df_processed = process_pxd011967_muscle(input_file)
    elif dataset_id == "PXD015982":
        df_processed = process_pxd015982_skin(input_file)
    else:
        raise ValueError(f"Unknown dataset ID: {dataset_id}")

    # Load our merged dataset for comparison
    print("\nLoading our merged dataset for comparison...")
    our_df = pd.read_csv(OUR_MERGED_DATA)

    # Analyze overlap
    print("\nAnalyzing gene overlap...")
    overlap_analysis = analyze_gene_overlap(df_processed, our_df)

    print(f"\n📊 GENE OVERLAP RESULTS:")
    print(f"  External genes: {overlap_analysis['n_external_genes']}")
    print(f"  Our ECM genes: {overlap_analysis['n_our_genes']}")
    print(f"  Overlap: {overlap_analysis['n_overlap']} ({overlap_analysis['overlap_percentage']:.1f}%)")
    print(f"  Target (≥70%): {'✅ MET' if overlap_analysis['target_met'] else '❌ NOT MET'}")

    # Save processed data
    output_file = output_dir / f"{dataset_id}_processed_zscore.csv"
    df_processed.to_csv(output_file, index=False)
    print(f"\n💾 Saved processed data to: {output_file}")

    # Save overlap analysis
    overlap_file = output_dir / f"{dataset_id}_gene_overlap.json"
    with open(overlap_file, "w") as f:
        json.dump(overlap_analysis, f, indent=2)
    print(f"💾 Saved overlap analysis to: {overlap_file}")

    return df_processed, overlap_analysis

#============================================================================
# MAIN EXECUTION
#============================================================================

def main():
    """Main execution"""

    print("\n" + "="*80)
    print("H13 INDEPENDENT DATASET VALIDATION - DATA HARMONIZATION")
    print("Agent: claude_code")
    print("="*80)

    # Create output directories
    (EXTERNAL_DATA_DIR / "PXD011967").mkdir(parents=True, exist_ok=True)
    (EXTERNAL_DATA_DIR / "PXD015982").mkdir(parents=True, exist_ok=True)

    print("\n⚠️  DATA ACQUISITION STATUS:")
    print("   PXD011967: Not yet downloaded (awaiting supplementary files)")
    print("   PXD015982: Not yet downloaded (awaiting PRIDE/PMC access)")
    print("\n📋 NEXT STEPS:")
    print("   1. Download supplementary data from eLife and PMC")
    print("   2. Place files in external_datasets/ folders")
    print("   3. Re-run this script with actual data")
    print("\n   See DATA_ACQUISITION_PLAN.md for detailed instructions")

    # Example usage (when data is available):
    # df_muscle, overlap_muscle = harmonize_external_dataset(
    #     "PXD011967",
    #     EXTERNAL_DATA_DIR / "PXD011967" / "source_data_1.xlsx",
    #     EXTERNAL_DATA_DIR / "PXD011967"
    # )

if __name__ == "__main__":
    main()
