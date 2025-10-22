import json
import time
from pathlib import Path
import requests

QUERIES = [
    "metabolic aging extracellular matrix",
    "mechanotransduction aging transition",
    "mitochondrial dysfunction fibrosis",
    "YAP TAZ mechanical aging",
    "reversible aging interventions metabolic",
    "fibrosis point of no return crosslinking",
    "Warburg effect aging collagen"
]

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / 'literature_findings_codex.csv'
MD_PATH = OUTPUT_DIR / 'literature_metabolic_mechanical.md'
API_BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
EMAIL = 'codex_agent@example.com'
MAX_PER_QUERY = 5


def esearch(query: str):
    params = {
        'db': 'pubmed',
        'term': query,
        'retmode': 'json',
        'retmax': MAX_PER_QUERY,
        'sort': 'relevance',
        'email': EMAIL
    }
    response = requests.get(f'{API_BASE}/esearch.fcgi', params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data.get('esearchresult', {}).get('idlist', [])


def esummary(pmids):
    if not pmids:
        return {}
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'json',
        'email': EMAIL
    }
    response = requests.get(f'{API_BASE}/esummary.fcgi', params=params, timeout=15)
    response.raise_for_status()
    return response.json().get('result', {})


def fetch_details():
    rows = []
    md_lines = ['# Literature Evidence: Metabolic ⇄ Mechanical Transition\n']
    seen_pmids = set()
    for query in QUERIES:
        pmids = esearch(query)
        time.sleep(0.34)  # rate limit courtesy
        summary = esummary(pmids)
        query_lines = [f'## Query: {query}']
        query_entries = []
        for pmid in pmids:
            if pmid == 'uids':
                continue
            record = summary.get(pmid)
            if not record or pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)
            title = record.get('title', '').rstrip('.')
            journal = record.get('fulljournalname', '')
            pubdate = record.get('pubdate', '')
            authors = ', '.join([a['name'] for a in record.get('authors', [])[:5]])
            summary_text = record.get('summary', '') or record.get('elocationid', '')
            query_entries.append((pmid, title, journal, pubdate, authors, summary_text))
            rows.append({
                'Query': query,
                'PMID': pmid,
                'Title': title,
                'Journal': journal,
                'Publication_Date': pubdate,
                'Lead_Authors': authors,
                'Summary': summary_text
            })
        if query_entries:
            for pmid, title, journal, pubdate, authors, summary_text in query_entries:
                query_lines.append(f'- **PMID {pmid}** — {title} ({journal}, {pubdate}). Authors: {authors}. {summary_text}')
        else:
            query_lines.append('- No new hits (already captured).')
        md_lines.extend(query_lines)
        md_lines.append('')
        time.sleep(0.34)
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    with open(MD_PATH, 'w') as f:
        f.write('\n'.join(md_lines))


if __name__ == '__main__':
    fetch_details()
