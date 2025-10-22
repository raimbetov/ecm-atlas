#!/usr/bin/env python3
"""
External Dataset Search: CALM and CAMK Proteins in Aging/ECM Studies

Search GEO, PRIDE, and ProteomeXchange for datasets containing the missing
mediator proteins (CALM1/2/3, CAMK family).

Author: Claude (claude_code agent)
Date: 2025-10-21
"""

import requests
import json
import time
from datetime import datetime
import re

class DatasetSearcher:
    """Search external proteomics databases for calcium signaling proteins"""

    def __init__(self):
        self.results = {
            'GEO': [],
            'PRIDE': [],
            'Misc': []
        }

    def search_geo(self, keywords, max_results=20):
        """
        Search GEO (Gene Expression Omnibus) via NCBI E-utilities

        Args:
            keywords: Search terms
            max_results: Maximum datasets to retrieve
        """
        print(f"\n🔍 Searching GEO: '{keywords}'")

        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        # Search
        search_url = f"{base_url}esearch.fcgi"
        params = {
            'db': 'gds',
            'term': keywords,
            'retmax': max_results,
            'retmode': 'json'
        }

        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            ids = data.get('esearchresult', {}).get('idlist', [])

            print(f"   Found {len(ids)} datasets")

            if not ids:
                return []

            # Fetch details
            time.sleep(0.5)
            fetch_url = f"{base_url}esummary.fcgi"
            fetch_params = {
                'db': 'gds',
                'id': ','.join(ids),
                'retmode': 'json'
            }

            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            response.raise_for_status()
            data = response.json()

            datasets = []
            for gds_id, info in data.get('result', {}).items():
                if gds_id == 'uids':
                    continue

                try:
                    datasets.append({
                        'id': info.get('accession', gds_id),
                        'title': info.get('title', 'N/A'),
                        'summary': info.get('summary', 'N/A'),
                        'type': info.get('entrytype', 'N/A'),
                        'organism': info.get('taxon', 'N/A'),
                        'pubdate': info.get('pdat', 'N/A'),
                        'url': f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={info.get('accession', '')}",
                        'keywords': keywords
                    })
                except Exception as e:
                    print(f"   Warning: Failed to parse dataset {gds_id}: {e}")

            print(f"   Retrieved {len(datasets)} dataset records")
            return datasets

        except Exception as e:
            print(f"   ERROR: {e}")
            return []

    def search_pride(self, keywords, max_results=20):
        """
        Search PRIDE proteomics repository

        Note: PRIDE API requires specific formatting
        """
        print(f"\n🔍 Searching PRIDE: '{keywords}'")

        # PRIDE REST API
        base_url = "https://www.ebi.ac.uk/pride/ws/archive/v2/search/projects"

        params = {
            'keyword': keywords,
            'pageSize': max_results,
            'page': 0
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            projects = data.get('_embedded', {}).get('projects', [])
            print(f"   Found {len(projects)} projects")

            datasets = []
            for proj in projects:
                datasets.append({
                    'id': proj.get('accession', 'N/A'),
                    'title': proj.get('title', 'N/A'),
                    'summary': proj.get('projectDescription', 'N/A')[:500],
                    'organisms': ', '.join([org.get('name', '') for org in proj.get('organisms', [])]),
                    'instruments': ', '.join([inst.get('name', '') for inst in proj.get('instruments', [])]),
                    'pubdate': proj.get('publicationDate', 'N/A'),
                    'url': f"https://www.ebi.ac.uk/pride/archive/projects/{proj.get('accession', '')}",
                    'keywords': keywords
                })

            print(f"   Retrieved {len(datasets)} project records")
            return datasets

        except Exception as e:
            print(f"   ERROR: {e}")
            return []

    def run_geo_searches(self):
        """Execute GEO searches for calcium signaling + aging"""

        queries = [
            "calmodulin aging proteomics",
            "CALM1 human aging",
            "CAMK calcium aging",
            "calcium signaling extracellular matrix aging",
            "(calmodulin OR CAMK) AND (collagen OR ECM) AND aging"
        ]

        all_datasets = []
        for query in queries:
            datasets = self.search_geo(query, max_results=15)
            all_datasets.extend(datasets)
            time.sleep(1)

        # Deduplicate by ID
        unique = {}
        for ds in all_datasets:
            ds_id = ds['id']
            if ds_id not in unique:
                unique[ds_id] = ds

        self.results['GEO'] = list(unique.values())
        print(f"\n✅ GEO: {len(self.results['GEO'])} unique datasets")

    def run_pride_searches(self):
        """Execute PRIDE searches"""

        queries = [
            "calmodulin aging",
            "calcium signaling ECM",
            "CAMK fibrosis",
            "calmodulin extracellular matrix"
        ]

        all_datasets = []
        for query in queries:
            datasets = self.search_pride(query, max_results=15)
            all_datasets.extend(datasets)
            time.sleep(1)

        # Deduplicate
        unique = {}
        for ds in all_datasets:
            ds_id = ds['id']
            if ds_id not in unique:
                unique[ds_id] = ds

        self.results['PRIDE'] = list(unique.values())
        print(f"\n✅ PRIDE: {len(self.results['PRIDE'])} unique datasets")

    def filter_relevant_datasets(self):
        """Filter datasets for likely CALM/CAMK content"""

        print(f"\n{'='*70}")
        print("📊 FILTERING FOR CALM/CAMK RELEVANCE")
        print(f"{'='*70}\n")

        target_proteins = [
            'calm', 'calmodulin',
            'camk', 'calcium/calmodulin-dependent kinase',
            'camkii', 'camk2'
        ]

        target_contexts = [
            'aging', 'age', 'elderly', 'senescence',
            'ecm', 'extracellular matrix', 'collagen', 'fibrosis',
            'tissue', 'organ'
        ]

        for db_name, datasets in self.results.items():
            if not datasets:
                continue

            print(f"\n{db_name} Datasets ({len(datasets)} total):")
            print("-" * 70)

            relevant = []
            for ds in datasets:
                text = (ds.get('title', '') + ' ' + ds.get('summary', '')).lower()

                # Check for protein mentions
                protein_match = any(prot in text for prot in target_proteins)

                # Check for context mentions
                context_match = any(ctx in text for ctx in target_contexts)

                # Check for human
                is_human = 'human' in text.lower() or 'homo sapiens' in text.lower()

                if protein_match and (context_match or is_human):
                    relevant.append(ds)

            print(f"\n  ✅ {len(relevant)} RELEVANT datasets found:\n")

            for ds in relevant[:5]:  # Show top 5
                print(f"  [{ds['id']}]")
                print(f"  {ds['title'][:100]}")
                print(f"  {ds['url']}")
                print()

            if len(relevant) > 5:
                print(f"  ... and {len(relevant) - 5} more\n")

    def save_results(self, output_dir):
        """Save search results to JSON and markdown"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON
        json_file = f"{output_dir}/external_datasets_claude.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        total = sum(len(datasets) for datasets in self.results.values())
        print(f"\n💾 Saved {total} datasets to {json_file}")

        # Markdown
        md_file = f"{output_dir}/external_datasets_claude.md"
        with open(md_file, 'w') as f:
            f.write("# External Dataset Search: CALM/CAMK Proteins\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**Objective:** Find proteomics datasets containing calmodulin (CALM) and ")
            f.write("calcium/calmodulin-dependent kinases (CAMK) to fill gaps in ECM-Atlas data.\n\n")

            for db_name, datasets in self.results.items():
                if not datasets:
                    continue

                f.write(f"\n## {db_name} ({len(datasets)} datasets)\n\n")

                for ds in datasets:
                    f.write(f"### {ds['id']}: {ds['title']}\n\n")

                    if 'organisms' in ds:
                        f.write(f"- **Organism:** {ds['organisms']}\n")
                    elif 'organism' in ds:
                        f.write(f"- **Organism:** {ds['organism']}\n")

                    f.write(f"- **Date:** {ds.get('pubdate', 'N/A')}\n")
                    f.write(f"- **URL:** {ds['url']}\n\n")

                    summary = ds.get('summary', 'N/A')
                    if len(summary) > 500:
                        summary = summary[:500] + "..."
                    f.write(f"**Summary:** {summary}\n\n")
                    f.write("---\n\n")

        print(f"💾 Saved dataset list to {md_file}")

        # CSV for easy filtering
        csv_file = f"{output_dir}/external_datasets_claude.csv"
        import csv

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Database', 'Accession', 'Title', 'Organism', 'Date', 'URL'])

            for db_name, datasets in self.results.items():
                for ds in datasets:
                    organism = ds.get('organisms', ds.get('organism', 'N/A'))
                    writer.writerow([
                        db_name,
                        ds['id'],
                        ds['title'][:100],
                        organism,
                        ds.get('pubdate', 'N/A'),
                        ds['url']
                    ])

        print(f"💾 Saved CSV table to {csv_file}")


def main():
    """Main execution"""

    print("="*70)
    print("  EXTERNAL DATASET SEARCH: CALM/CAMK Proteins")
    print("="*70)

    searcher = DatasetSearcher()

    # Search GEO
    print("\n" + "="*70)
    print("  GENE EXPRESSION OMNIBUS (GEO)")
    print("="*70)
    searcher.run_geo_searches()

    # Search PRIDE
    print("\n" + "="*70)
    print("  PRIDE PROTEOMICS REPOSITORY")
    print("="*70)
    searcher.run_pride_searches()

    # Filter for relevance
    searcher.filter_relevant_datasets()

    # Save
    output_dir = "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_10_calcium_signaling_cascade/claude_code/external_datasets"
    searcher.save_results(output_dir)

    print("\n" + "="*70)
    print("  DATASET SEARCH COMPLETE")
    print("="*70)

    # Print summary
    total = sum(len(datasets) for datasets in searcher.results.values())
    print(f"\n📊 SUMMARY:")
    print(f"   GEO: {len(searcher.results['GEO'])} datasets")
    print(f"   PRIDE: {len(searcher.results['PRIDE'])} datasets")
    print(f"   Total: {total} datasets found")
    print(f"\n⚠️  NOTE: Download and integration of actual data files requires manual review")
    print(f"   and approval due to data licensing and size constraints.\n")


if __name__ == "__main__":
    main()
