#!/usr/bin/env python3
"""
Metabolomics Database Search for Phase I Validation
Search Metabolomics Workbench and MetaboLights for tissue aging studies
Target: ATP, NAD+, lactate, pyruvate measurements
"""

import requests
import pandas as pd
import json
import time
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_URL_MW = "https://www.metabolomicsworkbench.org/rest"
BASE_URL_ML = "https://www.ebi.ac.uk/metabolights/ws"
OUTPUT_DIR = "metabolomics_data"

# Target metabolites
TARGET_METABOLITES = ['ATP', 'NAD', 'NADH', 'lactate', 'pyruvate', 'glucose', 'citrate']

print("="*80)
print("🔍 METABOLOMICS DATABASE SEARCH - PHASE I VALIDATION")
print("="*80)

# ============================================================================
# PART 1: Search Metabolomics Workbench
# ============================================================================

print("\n📊 PART 1: Searching Metabolomics Workbench...")
print("-" * 80)

def search_metabolomics_workbench():
    """Search Metabolomics Workbench for aging tissue studies"""

    print("\n🔎 Fetching all studies from Metabolomics Workbench...")

    # Get all study IDs
    try:
        response = requests.get(f"{BASE_URL_MW}/study/study_id/all/summary")

        if response.status_code != 200:
            print(f"❌ Error fetching studies: {response.status_code}")
            return pd.DataFrame()

        # Parse response - expecting format like "study_id:ST000001\tstudy_title:..."
        lines = response.text.strip().split('\n')
        studies = []

        for line in lines[1:]:  # Skip header
            parts = line.split('\t')
            if len(parts) < 2:
                continue

            study_dict = {}
            for part in parts:
                if ':' in part:
                    key, value = part.split(':', 1)
                    study_dict[key.strip()] = value.strip()

            if study_dict:
                studies.append(study_dict)

        print(f"✅ Found {len(studies)} total studies")

        # Filter for aging/fibrosis studies
        aging_keywords = ['aging', 'age', 'senescence', 'fibrosis', 'elderly', 'longevity']
        tissue_keywords = ['tissue', 'muscle', 'liver', 'heart', 'lung', 'brain', 'kidney']

        filtered_studies = []
        for study in studies:
            title = study.get('study_title', '').lower()
            summary = study.get('study_summary', '').lower()

            has_aging = any(kw in title or kw in summary for kw in aging_keywords)
            has_tissue = any(kw in title or kw in summary for kw in tissue_keywords)

            if has_aging and has_tissue:
                filtered_studies.append(study)

        print(f"✅ Filtered to {len(filtered_studies)} aging tissue studies")

        return pd.DataFrame(filtered_studies)

    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

# Load previous search results from H12
print("\n📁 Loading previous search results from H12...")
try:
    h12_hits = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_12_metabolic_mechanical_transition/codex/metabolomics_workbench_hits_codex.csv')
    print(f"✅ Loaded {len(h12_hits)} previous hits from H12")

    # Filter for tissue studies only
    tissue_studies_h12 = h12_hits[h12_hits['Title'].str.contains('tissue|muscle|liver|heart|lung|brain|kidney', case=False, na=False)]
    print(f"✅ Found {len(tissue_studies_h12)} tissue studies from H12")

    # Prioritize studies from aging search
    aging_studies_h12 = tissue_studies_h12[tissue_studies_h12['Search_Term'] == 'aging']
    print(f"✅ Found {len(aging_studies_h12)} aging tissue studies from H12")

    # Focus on mouse studies (match our proteomic data)
    priority_studies = aging_studies_h12[
        aging_studies_h12['Study_ID'].isin(['ST001699', 'ST001701', 'ST001702', 'ST001703', 'ST001637'])
    ]

    print(f"\n🎯 Priority studies (mouse tissue aging):")
    for _, row in priority_studies.iterrows():
        print(f"  - {row['Study_ID']}: {row['Title']}")

except FileNotFoundError:
    print("⚠️  H12 results not found, will search from scratch")
    aging_studies_h12 = pd.DataFrame()

# ============================================================================
# PART 2: Check Study Details for Target Metabolites
# ============================================================================

print("\n\n📊 PART 2: Checking Studies for Target Metabolites...")
print("-" * 80)

def check_study_metabolites(study_id: str) -> Dict:
    """Check if study contains target metabolites"""

    print(f"\n🔍 Checking {study_id}...")

    try:
        # Get study data
        response = requests.get(f"{BASE_URL_MW}/study/study_id/{study_id}/metabolites")
        time.sleep(0.5)  # Rate limiting

        if response.status_code != 200:
            print(f"  ❌ Cannot fetch metabolite list")
            return None

        # Parse metabolites
        lines = response.text.strip().split('\n')
        metabolites = []

        for line in lines[1:]:  # Skip header
            parts = line.split('\t')
            if len(parts) >= 2:
                metabolite_name = parts[1] if len(parts) > 1 else parts[0]
                metabolites.append(metabolite_name.strip())

        # Check for target metabolites
        found_metabolites = []
        for target in TARGET_METABOLITES:
            matches = [m for m in metabolites if target.lower() in m.lower()]
            if matches:
                found_metabolites.extend(matches)

        found_metabolites_unique = list(set(found_metabolites))

        result = {
            'Study_ID': study_id,
            'Total_Metabolites': len(metabolites),
            'Target_Metabolites_Found': len(found_metabolites_unique),
            'Found_Metabolites': ', '.join(found_metabolites_unique[:10]),  # First 10
            'Has_ATP': any('atp' in m.lower() for m in metabolites),
            'Has_NAD': any('nad' in m.lower() for m in metabolites),
            'Has_Lactate': any('lactate' in m.lower() or 'lactic' in m.lower() for m in metabolites),
            'Has_Pyruvate': any('pyruvate' in m.lower() or 'pyruvic' in m.lower() for m in metabolites),
        }

        print(f"  ✅ {result['Total_Metabolites']} metabolites total")
        print(f"  ✅ {result['Target_Metabolites_Found']} target metabolites found")
        if result['Target_Metabolites_Found'] > 0:
            print(f"     → {result['Found_Metabolites']}")

        return result

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# Check priority studies
if not aging_studies_h12.empty:
    priority_study_ids = ['ST001699', 'ST001701', 'ST001702', 'ST001703', 'ST001637', 'ST001888']

    metabolite_results = []
    for study_id in priority_study_ids:
        result = check_study_metabolites(study_id)
        if result:
            metabolite_results.append(result)

    df_metabolite_check = pd.DataFrame(metabolite_results)

    if df_metabolite_check.empty:
        print("\n⚠️  No metabolite data retrieved from studies")
    else:
        # Save results
        df_metabolite_check.to_csv(f'{OUTPUT_DIR}/metabolite_coverage_claude_code.csv', index=False)

        print("\n\n📋 SUMMARY OF METABOLITE COVERAGE:")
        print("=" * 80)
        print(df_metabolite_check[['Study_ID', 'Total_Metabolites', 'Has_ATP', 'Has_NAD', 'Has_Lactate', 'Has_Pyruvate']].to_string(index=False))

        # Identify best studies
        df_metabolite_check['Score'] = (
            df_metabolite_check['Has_ATP'].astype(int) +
            df_metabolite_check['Has_NAD'].astype(int) +
            df_metabolite_check['Has_Lactate'].astype(int) +
            df_metabolite_check['Has_Pyruvate'].astype(int)
        )

        best_studies = df_metabolite_check.nlargest(3, 'Score')

        print("\n\n🏆 TOP 3 STUDIES FOR PHASE I VALIDATION:")
        print("=" * 80)
        for _, row in best_studies.iterrows():
            print(f"\n{row['Study_ID']}: Score = {row['Score']}/4")
            print(f"  ATP: {'✅' if row['Has_ATP'] else '❌'}")
            print(f"  NAD: {'✅' if row['Has_NAD'] else '❌'}")
            print(f"  Lactate: {'✅' if row['Has_Lactate'] else '❌'}")
            print(f"  Pyruvate: {'✅' if row['Has_Pyruvate'] else '❌'}")
            print(f"  Total metabolites: {row['Total_Metabolites']}")

        # Save selected studies
        best_studies.to_csv(f'{OUTPUT_DIR}/selected_studies_claude_code.csv', index=False)

        print(f"\n\n✅ Results saved to {OUTPUT_DIR}/")
        print(f"   - metabolite_coverage_claude_code.csv")
        print(f"   - selected_studies_claude_code.csv")

# ============================================================================
# PART 3: MetaboLights Search
# ============================================================================

print("\n\n📊 PART 3: Searching MetaboLights...")
print("-" * 80)

def search_metabolights():
    """Search MetaboLights for aging studies"""

    try:
        # MetaboLights API search
        response = requests.get(f"{BASE_URL_ML}/studies")

        if response.status_code != 200:
            print(f"❌ Error fetching MetaboLights studies: {response.status_code}")
            return pd.DataFrame()

        data = response.json()
        studies = data.get('content', [])

        print(f"✅ Found {len(studies)} total MetaboLights studies")

        # Filter for aging
        aging_keywords = ['aging', 'age', 'senescence', 'elderly', 'longevity']
        tissue_keywords = ['tissue', 'muscle', 'liver', 'heart', 'lung', 'brain', 'kidney']

        filtered = []
        for study in studies:
            title = study.get('title', '').lower()
            has_aging = any(kw in title for kw in aging_keywords)
            has_tissue = any(kw in title for kw in tissue_keywords)

            if has_aging or has_tissue:
                filtered.append({
                    'Accession': study.get('accession', ''),
                    'Title': study.get('title', ''),
                    'Release_Date': study.get('releaseDate', ''),
                    'Status': study.get('status', '')
                })

        print(f"✅ Filtered to {len(filtered)} relevant studies")

        df_ml = pd.DataFrame(filtered)
        df_ml.to_csv(f'{OUTPUT_DIR}/metabolights_hits_claude_code.csv', index=False)

        print(f"\n📝 Top MetaboLights studies:")
        for _, row in df_ml.head(10).iterrows():
            print(f"  - {row['Accession']}: {row['Title'][:70]}...")

        return df_ml

    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

df_metabolights = search_metabolights()

print("\n\n" + "="*80)
print("✅ DATABASE SEARCH COMPLETED")
print("="*80)
print(f"\n📊 Summary:")
print(f"  - Metabolomics Workbench: {len(metabolite_results) if 'metabolite_results' in locals() else 0} studies checked")
print(f"  - MetaboLights: {len(df_metabolights)} relevant studies found")
print(f"  - Output directory: {OUTPUT_DIR}/")
print("\n🎯 Next step: Download data from top-ranked studies")
print("="*80)
