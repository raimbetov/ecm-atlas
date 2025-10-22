#!/usr/bin/env python3
"""
H13 Independent Dataset Validation - Dataset Search Script
Agent: claude_code
Purpose: Systematically search proteomics repositories for aging ECM datasets
"""

import requests
import json
import pandas as pd
from typing import Dict, List
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKSPACE = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code")

# Known datasets from literature search
VALIDATED_DATASETS = {
    "PXD011967": {
        "title": "Discovery proteomics in aging human skeletal muscle",
        "tissue": "Skeletal muscle",
        "species": "Human",
        "ftp": "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2019/11/PXD011967",
        "priority": "HIGH",
        "age_groups": "20-34, 35-49, 50-64, 65-79, 80+",
        "sample_size": 58,
        "proteins": 4380
    },
    "PXD015982": {
        "title": "Alterations in ECM composition during skin aging and photoaging",
        "tissue": "Skin",
        "species": "Human",
        "ftp": "https://www.ebi.ac.uk/pride/archive/projects/PXD015982",
        "priority": "HIGH",
        "age_groups": "Young (26.7), Aged (84.0)",
        "sample_size": 6,
        "proteins": 229
    },
    "PXD007048": {
        "title": "Cell-specific bone marrow proteomes and aging",
        "tissue": "Bone marrow",
        "species": "Human",
        "ftp": "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2018/09/PXD007048",
        "priority": "MEDIUM",
        "age_groups": "Young (20-30), Old (60-70)",
        "sample_size": "Multiple donors",
        "proteins": "Thousands"
    },
    "PXD016440": {
        "title": "Time-resolved ECM atlas of developing human skin dermis",
        "tissue": "Skin dermis",
        "species": "Human",
        "ftp": "https://www.ebi.ac.uk/pride/archive/projects/PXD016440",
        "priority": "LOW",
        "age_groups": "Developmental (fetal to adult)",
        "sample_size": "Multiple stages",
        "proteins": "Comprehensive matrisome"
    }
}

MASSIVE_DATASETS = {
    "MSV000082958": {
        "title": "Quantitative proteomic profiling of ECM in lung fibrosis model",
        "tissue": "Lung (in vitro)",
        "species": "Human",
        "priority": "MEDIUM",
        "note": "In vitro fibrosis model, excellent for collagen PTMs"
    },
    "MSV000096508": {
        "title": "Brain ECM and cognitive aging",
        "tissue": "Brain (hippocampus, cortex)",
        "species": "Mouse",
        "priority": "MEDIUM",
        "note": "Mouse model, cognitive aging focus"
    }
}

# Our 13 studies to EXCLUDE (avoid duplication)
OUR_STUDY_IDS = [
    # Add known study IDs from our merged dataset if available
    # Placeholder for now
]

# ============================================================================
# PRIDE API FUNCTIONS
# ============================================================================

def search_pride_api(keyword: str, page_size: int = 50, page: int = 0) -> Dict:
    """
    Search PRIDE database via REST API
    API docs: https://www.ebi.ac.uk/pride/ws/archive/v2/
    """
    base_url = "https://www.ebi.ac.uk/pride/ws/archive/v2/projects"
    params = {
        "keyword": keyword,
        "pageSize": page_size,
        "page": page,
        "sortDirection": "DESC",
        "sortFields": "submission_date"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error searching PRIDE API: {e}")
        return {}

def get_pride_project_details(accession: str) -> Dict:
    """Get detailed information for a specific PRIDE project"""
    url = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{accession}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {accession}: {e}")
        return {}

# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def search_aging_ecm_datasets() -> List[Dict]:
    """
    Comprehensive search for aging ECM proteomics datasets
    """
    results = []

    # Search terms from task requirements
    search_terms = [
        "aging extracellular matrix",
        "aging tissue proteomics",
        "senescence collagen",
        "fibrosis proteome",
        "ECM aging",
        "matrisome aging"
    ]

    print("=" * 80)
    print("SEARCHING PRIDE DATABASE FOR AGING ECM DATASETS")
    print("=" * 80)

    for term in search_terms:
        print(f"\nSearching for: '{term}'...")
        data = search_pride_api(term, page_size=20)

        if "_embedded" in data and "projects" in data["_embedded"]:
            projects = data["_embedded"]["projects"]
            print(f"  Found {len(projects)} projects")

            for proj in projects:
                accession = proj.get("accession", "")
                title = proj.get("title", "")

                # Filter criteria
                organisms = proj.get("organisms", [])
                is_human_or_mouse = any(
                    "sapiens" in str(org).lower() or "musculus" in str(org).lower()
                    for org in organisms
                )

                # Check if already in our validated list
                if accession in VALIDATED_DATASETS or accession in OUR_STUDY_IDS:
                    continue

                if is_human_or_mouse:
                    results.append({
                        "accession": accession,
                        "title": title,
                        "organisms": organisms,
                        "submission_date": proj.get("submissionDate", ""),
                        "publication_date": proj.get("publicationDate", ""),
                        "search_term": term
                    })

        time.sleep(0.5)  # Rate limiting

    # Remove duplicates
    unique_results = {r["accession"]: r for r in results}.values()
    return list(unique_results)

def assess_dataset_suitability(accession: str) -> Dict:
    """
    Assess if a dataset meets our inclusion criteria
    """
    details = get_pride_project_details(accession)

    if not details:
        return {"suitable": False, "reason": "Failed to fetch details"}

    # Inclusion criteria checklist
    criteria = {
        "has_age_comparison": False,
        "is_tissue": False,
        "is_human_or_mouse": False,
        "has_quantification": False,
        "likely_ecm_proteins": False
    }

    # Check description for aging keywords
    description = details.get("projectDescription", "").lower()
    keywords = [kw.get("name", "").lower() for kw in details.get("keywords", [])]

    age_keywords = ["aging", "age", "old", "young", "elderly", "senescence"]
    criteria["has_age_comparison"] = any(kw in description or kw in " ".join(keywords) for kw in age_keywords)

    tissue_keywords = ["tissue", "organ", "muscle", "skin", "heart", "liver", "lung", "brain"]
    criteria["is_tissue"] = any(kw in description or kw in " ".join(keywords) for kw in tissue_keywords)

    organisms = [org.get("name", "").lower() for org in details.get("organisms", [])]
    criteria["is_human_or_mouse"] = any("sapiens" in org or "musculus" in org for org in organisms)

    # Check for quantitative methods
    quant_methods = details.get("quantificationMethods", [])
    exp_types = [et.get("name", "").lower() for et in details.get("experimentTypes", [])]
    criteria["has_quantification"] = len(quant_methods) > 0 or "quantitative" in " ".join(exp_types)

    # Check for ECM-related keywords
    ecm_keywords = ["extracellular matrix", "ecm", "collagen", "fibronectin", "matrisome", "fibrosis"]
    criteria["likely_ecm_proteins"] = any(kw in description or kw in " ".join(keywords) for kw in ecm_keywords)

    # Overall suitability score
    score = sum(criteria.values())
    suitable = score >= 3  # At least 3 out of 5 criteria

    return {
        "suitable": suitable,
        "score": score,
        "criteria": criteria,
        "title": details.get("title", ""),
        "description": description[:200] + "...",
        "organisms": [org.get("name", "") for org in details.get("organisms", [])],
        "submission_date": details.get("submissionDate", "")
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("H13 INDEPENDENT DATASET VALIDATION - DATASET SEARCH")
    print("Agent: claude_code")
    print("=" * 80 + "\n")

    # Step 1: Display validated datasets
    print("STEP 1: VALIDATED DATASETS FROM LITERATURE REVIEW")
    print("-" * 80)
    print(f"Found {len(VALIDATED_DATASETS)} PRIDE datasets:")
    for acc, info in VALIDATED_DATASETS.items():
        print(f"  {acc}: {info['title']}")
        print(f"    - Tissue: {info['tissue']}, Priority: {info['priority']}")
        print(f"    - FTP: {info['ftp']}")

    print(f"\nFound {len(MASSIVE_DATASETS)} MassIVE datasets:")
    for acc, info in MASSIVE_DATASETS.items():
        print(f"  {acc}: {info['title']}")
        print(f"    - Tissue: {info['tissue']}, Priority: {info['priority']}")

    # Step 2: Search for additional datasets
    print("\n" + "=" * 80)
    print("STEP 2: SEARCHING FOR ADDITIONAL DATASETS")
    print("-" * 80)

    additional_datasets = search_aging_ecm_datasets()
    print(f"\nFound {len(additional_datasets)} additional candidate datasets")

    # Step 3: Assess suitability of top candidates
    print("\n" + "=" * 80)
    print("STEP 3: ASSESSING DATASET SUITABILITY")
    print("-" * 80)

    suitable_datasets = []
    for dataset in additional_datasets[:10]:  # Assess top 10
        acc = dataset["accession"]
        print(f"\nAssessing {acc}...")
        assessment = assess_dataset_suitability(acc)

        if assessment["suitable"]:
            print(f"  ✓ SUITABLE (Score: {assessment['score']}/5)")
            print(f"    {assessment['title']}")
            suitable_datasets.append({
                "accession": acc,
                **assessment
            })
        else:
            print(f"  ✗ Not suitable (Score: {assessment['score']}/5)")

        time.sleep(1)  # Rate limiting

    # Step 4: Save results
    print("\n" + "=" * 80)
    print("STEP 4: SAVING RESULTS")
    print("-" * 80)

    # Save summary
    summary = {
        "validated_pride": list(VALIDATED_DATASETS.keys()),
        "validated_massive": list(MASSIVE_DATASETS.keys()),
        "additional_suitable": [d["accession"] for d in suitable_datasets],
        "total_datasets_found": len(VALIDATED_DATASETS) + len(MASSIVE_DATASETS) + len(suitable_datasets)
    }

    summary_path = WORKSPACE / "dataset_search_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    # Save detailed results
    all_results = []
    for acc, info in VALIDATED_DATASETS.items():
        all_results.append({"accession": acc, "repository": "PRIDE", **info})
    for acc, info in MASSIVE_DATASETS.items():
        all_results.append({"accession": acc, "repository": "MassIVE", **info})
    for d in suitable_datasets:
        all_results.append({"repository": "PRIDE", **d})

    df_results = pd.DataFrame(all_results)
    results_path = WORKSPACE / "discovered_datasets_claude_code.csv"
    df_results.to_csv(results_path, index=False)
    print(f"Saved detailed results to: {results_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("DATASET SEARCH COMPLETE")
    print("=" * 80)
    print(f"Total datasets identified: {len(all_results)}")
    print(f"  - PRIDE: {len([d for d in all_results if d['repository'] == 'PRIDE'])}")
    print(f"  - MassIVE: {len([d for d in all_results if d['repository'] == 'MassIVE'])}")
    print(f"\nHigh priority datasets for download: {len([d for d in all_results if d.get('priority') == 'HIGH'])}")
    print("\nNext steps:")
    print("  1. Download high-priority datasets (PXD011967, PXD015982)")
    print("  2. Process and harmonize data")
    print("  3. Test H08, H06, H03 models on external data")

if __name__ == "__main__":
    main()
