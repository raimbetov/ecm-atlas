#!/usr/bin/env python3
"""
Generate individual HTML pages for each dataset in the ECM-Atlas database.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
UNIFIED_METADATA = PROJECT_ROOT / "08_merged_ecm_dataset" / "unified_metadata.json"
MERGED_CSV = PROJECT_ROOT / "08_merged_ecm_dataset" / "merged_ecm_aging_zscore.csv"
OUTPUT_DIR = Path(__file__).parent / "datasets"
PAPERS_DIR = PROJECT_ROOT / "05_papers_to_csv"

# Dataset publication information (manually curated)
PUBLICATIONS = {
    "Randles_2021": {
        "title": "Proteomic profiling of age-dependent changes in the kidney glomerular and tubulointerstitial compartments",
        "authors": "Randles MJ, Lausecker F, ...",
        "journal": "Journal of the American Society of Nephrology",
        "year": 2021,
        "pmid": "33446562",
        "doi": "10.1681/ASN.2020101442",
        "url": "https://pubmed.ncbi.nlm.nih.gov/33446562/"
    },
    "Tam_2020": {
        "title": "The matrisome of the aged intervertebral disc",
        "authors": "Tam V, Chan WCW, ...",
        "journal": "eLife",
        "year": 2020,
        "pmid": "33242310",
        "doi": "10.7554/eLife.64940",
        "url": "https://elifesciences.org/articles/64940"
    },
    "Caldeira_2017": {
        "title": "Comparative proteomic analysis of cardiac tissues",
        "authors": "Caldeira et al.",
        "journal": "Scientific Reports",
        "year": 2017,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    },
    "Tsumagari_2023": {
        "title": "Age-associated proteome and secretome of mouse mammary tissue",
        "authors": "Tsumagari et al.",
        "journal": "Aging Cell",
        "year": 2023,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    },
    "Ouni_2022": {
        "title": "Aging-related changes in pancreatic islet proteome",
        "authors": "Ouni et al.",
        "journal": "Molecular Metabolism",
        "year": 2022,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    },
    "LiDermis_2021": {
        "title": "Dermal extracellular matrix changes in aging skin",
        "authors": "Li et al.",
        "journal": "Nature Aging",
        "year": 2021,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    },
    "Santinha_2024_Human": {
        "title": "Human skeletal muscle aging proteomics",
        "authors": "Santinha et al.",
        "journal": "Nature Communications",
        "year": 2024,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    },
    "Santinha_2024_Mouse_DT": {
        "title": "Mouse skeletal muscle aging proteomics (Denervated Tibialis)",
        "authors": "Santinha et al.",
        "journal": "Nature Communications",
        "year": 2024,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    },
    "Santinha_2024_Mouse_NT": {
        "title": "Mouse skeletal muscle aging proteomics (Native Tibialis)",
        "authors": "Santinha et al.",
        "journal": "Nature Communications",
        "year": 2024,
        "pmid": "TBD",
        "doi": "TBD",
        "url": "#"
    }
}


def load_metadata():
    """Load unified metadata."""
    with open(UNIFIED_METADATA, 'r') as f:
        return json.load(f)


def load_dataset_stats(study_id):
    """Load statistics for a specific dataset."""
    df = pd.read_csv(MERGED_CSV)
    study_df = df[df['Study_ID'] == study_id]

    stats = {
        "total_rows": len(study_df),
        "unique_proteins": study_df['Protein_ID'].nunique(),
        "unique_genes": study_df['Gene_Symbol'].nunique(),
        "tissues": study_df['Tissue'].unique().tolist(),
        "compartments": study_df['Compartment'].unique().tolist() if 'Compartment' in study_df.columns else [],
        "species": study_df['Species'].iloc[0] if len(study_df) > 0 else "Unknown",
        "age_range": {
            "min": float(study_df['Age'].min()),
            "max": float(study_df['Age'].max())
        } if 'Age' in study_df.columns and len(study_df) > 0 else None,
        "ecm_proteins": len(study_df[study_df['ECM_Class'].notna()]) if 'ECM_Class' in study_df.columns else 0,
        "avg_zscore": float(study_df['Z_score'].abs().mean()) if 'Z_score' in study_df.columns else None
    }

    return stats


def read_readme(study_id):
    """Try to read README.md for the dataset."""
    # Try different possible locations
    possible_paths = [
        PAPERS_DIR / f"*{study_id}*" / "README.md",
        PROJECT_ROOT / f"*{study_id}*" / "README.md",
    ]

    for pattern in possible_paths:
        from glob import glob
        matches = glob(str(pattern))
        if matches:
            with open(matches[0], 'r') as f:
                return f.read()

    return None


def generate_dataset_page(study_id, metadata, stats, publication_info):
    """Generate HTML page for a single dataset using shared styles from styles.css."""

    readme_content = read_readme(study_id)

    # Build tissue/compartment lists using features-list style
    tissues_html = '<ul class="features-list">' + ''.join([f'<li>{tissue}</li>' for tissue in stats['tissues']]) + '</ul>'
    
    compartments_html = ''
    if stats['compartments']:
        compartments_html = f'''<div style="margin-top: var(--space-sm);">
            <strong>Compartments:</strong>
            <ul class="features-list">
                {''.join([f'<li>{comp}</li>' for comp in stats['compartments']])}
            </ul>
        </div>'''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{study_id} - ECM-Atlas Dataset</title>
    <link rel="stylesheet" href="../static/styles.css">
</head>
<body>
    <a href="../dashboard.html" style="display: inline-block; margin-bottom: var(--space-md); color: var(--accent); text-decoration: none; font-weight: 500;">← Back to Dashboard</a>

    <!-- Header Section -->
    <div class="section">
        <h1>📊 {study_id.replace('_', ' ')}</h1>
        <p style="margin: 0; opacity: 0.9; font-size: 1.1em; color: var(--text-light);">
            {publication_info.get('title', 'Dataset Details')}
        </p>
    </div>

    <!-- Publication Information -->
    <div class="section">
        <div class="tip">
            <h3 style="margin-top: 0;">📄 Original Publication</h3>
            <p><strong>Authors:</strong> {publication_info.get('authors', 'N/A')}</p>
            <p><strong>Journal:</strong> {publication_info.get('journal', 'N/A')} ({publication_info.get('year', 'N/A')})</p>
            {f'<p><strong>PubMed ID:</strong> <a href="{publication_info.get("url", "#")}" target="_blank">{publication_info.get("pmid", "N/A")}</a></p>' if publication_info.get('pmid') != 'TBD' else ''}
            {f'<p><strong>DOI:</strong> {publication_info.get("doi", "N/A")}</p>' if publication_info.get('doi') != 'TBD' else ''}
        </div>
    </div>

    <!-- Dataset Statistics -->
    <div class="section">
        <h2>📈 Dataset Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-number">{stats['total_rows']:,}</span>
                <span class="stat-label">Total Measurements</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{stats['unique_proteins']:,}</span>
                <span class="stat-label">Unique Proteins</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{stats['ecm_proteins']:,}</span>
                <span class="stat-label">ECM Proteins</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{stats['species']}</span>
                <span class="stat-label">Species</span>
            </div>
            {f'''<div class="stat-item">
                <span class="stat-number">{stats['age_range']['min']:.0f}-{stats['age_range']['max']:.0f}</span>
                <span class="stat-label">Age Range</span>
            </div>''' if stats['age_range'] else ''}
        </div>
    </div>

    <!-- Tissue & Compartments -->
    <div class="section">
        <h2>🧬 Tissue Information</h2>
        <div>
            <strong>Tissues:</strong>
            {tissues_html}
        </div>
        {compartments_html}
    </div>

    <!-- Data Processing -->
    <div class="section">
        <h2>⚙️ Data Processing</h2>
        <h3>Pipeline Steps</h3>
        <ol>
            <li><strong>Raw Data Extraction:</strong> Original proteomic data extracted from supplementary materials</li>
            <li><strong>Normalization:</strong> Converted to long format, filtered for ECM proteins (Match_Confidence &gt; 0)</li>
            <li><strong>Wide Format Transformation:</strong> Pivoted to sample × protein abundance matrix</li>
            <li><strong>Merge to Unified Database:</strong> Added to <code>merged_ecm_aging_zscore.csv</code></li>
            <li><strong>Z-score Calculation:</strong> Calculated per-compartment z-scores for cross-study comparison</li>
        </ol>

        <h3>Key Processing Parameters</h3>
        <ul>
            <li><strong>Study ID:</strong> <code>{study_id}</code></li>
            <li><strong>Missing Values:</strong> Preserved (NaN = protein not detected)</li>
            <li><strong>Zero Values:</strong> Included (0.0 = detected but absent)</li>
            <li><strong>Normalization:</strong> Within-study z-scores per compartment</li>
        </ul>
    </div>

    <!-- README Content (if available) -->
    {f'''<div class="section">
        <h2>📝 Processing Notes</h2>
        <div style="background: var(--bg-color); padding: var(--space-md); border: 1px solid var(--border-color); border-radius: var(--radius-md);">
            {readme_content if readme_content else '<p><em>No additional processing notes available.</em></p>'}
        </div>
    </div>''' if readme_content else ''}

    <!-- Data Access -->
    <div class="section">
        <h2>💾 Data Access</h2>
        <p>This dataset is included in the unified ECM-Atlas database. You can access it through:</p>
        <ul>
            <li><strong>Dashboard:</strong> Select "{study_id}" from the Individual Dataset Analysis tab</li>
            <li><strong>CSV File:</strong> <code>08_merged_ecm_dataset/merged_ecm_aging_zscore.csv</code></li>
            <li><strong>Filter:</strong> <code>df[df['Study_ID'] == '{study_id}']</code></li>
        </ul>

        <h3>Quick Start (Python)</h3>
        <pre>import pandas as pd

# Load unified database
df = pd.read_csv('merged_ecm_aging_zscore.csv')

# Filter for this study
study_df = df[df['Study_ID'] == '{study_id}']

# View ECM proteins only
ecm_df = study_df[study_df['ECM_Class'].notna()]

print(f"Total proteins: {{len(study_df['Protein_ID'].unique())}}")
print(f"ECM proteins: {{len(ecm_df['Protein_ID'].unique())}}")</pre>
    </div>

    <!-- Metadata -->
    <div class="section">
        <h2>🔧 Metadata</h2>
        <div style="background: var(--code-bg); padding: var(--space-md); border: 1px solid var(--code-border); border-radius: var(--radius-md);">
            <pre style="margin: 0;">{json.dumps(metadata, indent=2)}</pre>
        </div>
    </div>

    <!-- Footer -->
    <div class="section" style="text-align: center; color: var(--text-muted);">
        <p>
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d')} &nbsp;|&nbsp;
            <strong>ECM-Atlas Version:</strong> 2.0
        </p>
        <p style="margin-top: var(--space-sm);">
            <a href="../dashboard.html" style="color: var(--accent); text-decoration: none; font-weight: 500;">← Back to Dashboard</a>
        </p>
    </div>

    <!-- Theme sync script -->
    <script>
        // Sync theme with dashboard
        (function() {{
            try {{
                var theme = localStorage.getItem('theme') || 'light';
                document.documentElement.setAttribute('data-theme', theme);
            }} catch (e) {{
                console.warn('Could not load theme preference:', e);
            }}
        }})();
    </script>
</body>
</html>
"""
    return html


def main():
    """Generate all dataset pages."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load unified metadata
    try:
        unified_meta = load_metadata()
    except FileNotFoundError:
        print("Warning: unified_metadata.json not found, using defaults")
        unified_meta = {}

    # Load main CSV to get list of studies
    df = pd.read_csv(MERGED_CSV)
    study_ids = df['Study_ID'].unique()

    print(f"Generating pages for {len(study_ids)} datasets...")

    for study_id in study_ids:
        print(f"  - {study_id}...")

        # Get publication info
        pub_info = PUBLICATIONS.get(study_id, {
            "title": f"{study_id} Dataset",
            "authors": "Unknown",
            "journal": "Unknown",
            "year": "Unknown",
            "pmid": "TBD",
            "doi": "TBD",
            "url": "#"
        })

        # Get dataset stats
        stats = load_dataset_stats(study_id)

        # Get metadata for this study
        metadata = unified_meta.get(study_id, {})

        # Generate HTML
        html = generate_dataset_page(study_id, metadata, stats, pub_info)

        # Write to file
        output_path = OUTPUT_DIR / f"{study_id}.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"    → {output_path}")

    print(f"\n✅ Generated {len(study_ids)} dataset pages in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
