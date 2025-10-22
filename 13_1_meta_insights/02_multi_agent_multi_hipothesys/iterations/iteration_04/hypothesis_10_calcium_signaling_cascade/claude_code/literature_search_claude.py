#!/usr/bin/env python3
"""
Literature Search for S100-CALM-CAMK-LOX Calcium Signaling Pathway

This script searches PubMed and other databases for evidence of the complete
calcium signaling cascade in ECM regulation.

Author: Claude (claude_code agent)
Date: 2025-10-21
"""

import requests
import json
import time
from datetime import datetime
import xml.etree.ElementTree as ET

class LiteratureSearcher:
    """Search PubMed and compile evidence for calcium signaling pathway"""

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.results = []

    def search_pubmed(self, query, max_results=50, year_from=2015):
        """
        Search PubMed for articles matching query

        Args:
            query: Search terms
            max_results: Maximum papers to retrieve
            year_from: Earliest publication year
        """
        print(f"\n🔍 Searching PubMed: '{query}'")

        # Step 1: Search and get PMIDs
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': f'{query} AND {year_from}:3000[PDAT]',
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }

        try:
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])

            print(f"   Found {len(pmids)} articles")

            if not pmids:
                return []

            # Step 2: Fetch details for each PMID
            time.sleep(0.5)  # Be nice to NCBI
            fetch_url = f"{self.base_url}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids[:20]),  # Get top 20
                'retmode': 'xml'
            }

            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)
            papers = []

            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract metadata
                    pmid = article.find('.//PMID').text

                    title_elem = article.find('.//ArticleTitle')
                    title = ''.join(title_elem.itertext()) if title_elem is not None else 'N/A'

                    abstract_elem = article.find('.//AbstractText')
                    abstract = ''.join(abstract_elem.itertext()) if abstract_elem is not None else 'N/A'

                    year_elem = article.find('.//PubDate/Year')
                    year = year_elem.text if year_elem is not None else 'N/A'

                    # Authors
                    authors = []
                    for author in article.findall('.//Author')[:3]:  # First 3 authors
                        last = author.find('.//LastName')
                        init = author.find('.//Initials')
                        if last is not None and init is not None:
                            authors.append(f"{last.text} {init.text}")
                    authors_str = ', '.join(authors) + (' et al.' if len(article.findall('.//Author')) > 3 else '')

                    # Journal
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else 'N/A'

                    papers.append({
                        'pmid': pmid,
                        'title': title,
                        'authors': authors_str,
                        'year': year,
                        'journal': journal,
                        'abstract': abstract,
                        'query': query,
                        'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'
                    })

                except Exception as e:
                    print(f"   Warning: Failed to parse article: {e}")
                    continue

            print(f"   Retrieved {len(papers)} complete records")
            return papers

        except Exception as e:
            print(f"   ERROR: {e}")
            return []

    def run_all_searches(self):
        """Execute all literature search queries"""

        queries = [
            "S100 calcium signaling aging extracellular matrix",
            "S100 calmodulin binding interaction",
            "calmodulin CAMK extracellular matrix",
            "calcium calmodulin-dependent kinase collagen crosslinking",
            "LOX lysyl oxidase calcium regulation",
            "transglutaminase calcium calmodulin",
            "CAMK2 fibrosis tissue stiffness",
            "S100A10 ECM remodeling",
            "S100B aging fibrosis",
            "calcium signaling tissue aging"
        ]

        all_papers = []

        for i, query in enumerate(queries, 1):
            print(f"\n{'='*70}")
            print(f"Query {i}/{len(queries)}")
            papers = self.search_pubmed(query, max_results=30)
            all_papers.extend(papers)
            time.sleep(1)  # Rate limiting

        # Remove duplicates by PMID
        unique_papers = {}
        for paper in all_papers:
            pmid = paper['pmid']
            if pmid not in unique_papers:
                unique_papers[pmid] = paper

        self.results = list(unique_papers.values())
        print(f"\n{'='*70}")
        print(f"✅ Total unique papers found: {len(self.results)}")

        return self.results

    def analyze_relevance(self):
        """Analyze papers for mentions of key proteins"""

        key_proteins = {
            'S100': ['S100A', 'S100B', 'S100 protein'],
            'CALM': ['calmodulin', 'CALM1', 'CALM2', 'CALM3', 'CaM'],
            'CAMK': ['CAMK', 'CaMK', 'calcium/calmodulin-dependent kinase', 'CaMKII'],
            'LOX': ['lysyl oxidase', 'LOX', 'LOXL'],
            'TGM': ['transglutaminase', 'TGM2', 'TG2']
        }

        print(f"\n{'='*70}")
        print("📊 RELEVANCE ANALYSIS")
        print(f"{'='*70}\n")

        for category, terms in key_proteins.items():
            matching_papers = []

            for paper in self.results:
                text = (paper['title'] + ' ' + paper['abstract']).lower()
                if any(term.lower() in text for term in terms):
                    matching_papers.append(paper)

            print(f"{category}: {len(matching_papers)} papers mention {terms[0]}")

            if matching_papers:
                print(f"  Top papers:")
                for paper in matching_papers[:3]:
                    print(f"    • {paper['authors']} ({paper['year']}). {paper['title'][:80]}...")
                print()

        # Find papers mentioning MULTIPLE pathway components
        print("\n🔗 Papers mentioning MULTIPLE pathway components:")
        multi_component = []

        for paper in self.results:
            text = (paper['title'] + ' ' + paper['abstract']).lower()
            components_found = []

            for category, terms in key_proteins.items():
                if any(term.lower() in text for term in terms):
                    components_found.append(category)

            if len(components_found) >= 2:
                multi_component.append({
                    'paper': paper,
                    'components': components_found
                })

        # Sort by number of components
        multi_component.sort(key=lambda x: len(x['components']), reverse=True)

        for item in multi_component[:10]:
            paper = item['paper']
            comps = ' + '.join(item['components'])
            print(f"\n  [{comps}]")
            print(f"  {paper['authors']} ({paper['year']})")
            print(f"  {paper['title']}")
            print(f"  {paper['url']}")

    def save_results(self, output_file):
        """Save results to JSON and markdown"""

        # Save JSON
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Saved {len(self.results)} papers to {json_file}")

        # Save Markdown
        with open(output_file, 'w') as f:
            f.write("# Literature Review: S100→CALM→CAMK→LOX/TGM Calcium Signaling Pathway\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Papers:** {len(self.results)}\n\n")
            f.write("---\n\n")

            # Group by query
            queries_seen = {}
            for paper in self.results:
                query = paper.get('query', 'Unknown')
                if query not in queries_seen:
                    queries_seen[query] = []
                queries_seen[query].append(paper)

            for query, papers in queries_seen.items():
                f.write(f"## Query: \"{query}\"\n\n")
                f.write(f"**Papers found:** {len(papers)}\n\n")

                for paper in papers:
                    f.write(f"### {paper['title']}\n\n")
                    f.write(f"- **Authors:** {paper['authors']}\n")
                    f.write(f"- **Year:** {paper['year']}\n")
                    f.write(f"- **Journal:** {paper['journal']}\n")
                    f.write(f"- **PMID:** [{paper['pmid']}]({paper['url']})\n\n")

                    if paper['abstract'] != 'N/A':
                        f.write(f"**Abstract:** {paper['abstract'][:500]}...\n\n")

                    f.write("---\n\n")

        print(f"💾 Saved literature review to {output_file}")


def main():
    """Main execution"""

    print("="*70)
    print("  LITERATURE SEARCH: Calcium Signaling Pathway in ECM Aging")
    print("="*70)

    searcher = LiteratureSearcher()

    # Run searches
    papers = searcher.run_all_searches()

    # Analyze relevance
    if papers:
        searcher.analyze_relevance()

        # Save results
        output_dir = "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_10_calcium_signaling_cascade/claude_code/literature"
        searcher.save_results(f"{output_dir}/literature_review_claude.md")
    else:
        print("\n⚠️  No papers found!")

    print("\n" + "="*70)
    print("  LITERATURE SEARCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
