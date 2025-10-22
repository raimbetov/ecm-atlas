import json
from pathlib import Path
import requests
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent
WORKBENCH_URL = 'https://www.metabolomicsworkbench.org/rest/study/study_title/{term}/summary'
WORKBENCH_DATASETS = OUTPUT_DIR / 'metabolomics_workbench_hits_codex.csv'

KEYWORDS = {
    'proteomics', 'proteomic', 'paired', 'multi-omics', 'multiomics',
    'transcriptomics', 'mechanical', 'mechanotransduction', 'yap', 'fibrosis',
    'collagen', 'matrix', 'ecm'
}
SEARCH_TERMS = ['aging', 'fibrosis', 'extracellular matrix']


def fetch_metabolomics_workbench(term: str = 'aging'):
    response = requests.get(WORKBENCH_URL.format(term=term), timeout=20)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        entries = data.values()
    else:
        entries = data
    rows = []
    for entry in entries:
        study_id = entry.get('study_id') or entry.get('STUDY_ID')
        title = entry.get('study_title', '') or entry.get('STUDY_TITLE', '')
        design = entry.get('study_design', '') or entry.get('STUDY_DESIGN', '')
        summary = entry.get('study_summary', '') or entry.get('STUDY_SUMMARY', '')
        text = ' '.join([title.lower(), design.lower(), summary.lower()])
        keyword_hits = [kw for kw in KEYWORDS if kw in text]
        rows.append({
            'Study_ID': study_id,
            'Title': title,
            'Design': design,
            'Summary': summary,
            'Keyword_Hits': ';'.join(keyword_hits)
        })
    return pd.DataFrame(rows)


def main():
    frames = []
    for term in SEARCH_TERMS:
        df = fetch_metabolomics_workbench(term)
        df['Search_Term'] = term
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['Study_ID'])
    combined.to_csv(WORKBENCH_DATASETS, index=False)
    priority = combined[combined['Keyword_Hits'] != '']
    md_lines = ['# Metabolomics Workbench Candidate Studies\n']
    for _, row in priority.head(10).iterrows():
        md_lines.append(
            f"- **{row['Study_ID']}** ({row['Search_Term']}): {row['Title']} — keywords: {row['Keyword_Hits']}"
        )
    if priority.empty:
        md_lines.append('- No clear paired metabolomics-proteomics hits found; follow-up needed.')
    with open(OUTPUT_DIR / 'metabolomics_workbench_summary_codex.md', 'w') as f:
        f.write('\n'.join(md_lines))


if __name__ == '__main__':
    main()
