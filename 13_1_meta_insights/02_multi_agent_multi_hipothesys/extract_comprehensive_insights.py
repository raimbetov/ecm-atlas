#!/usr/bin/env python3
"""
Extract comprehensive insights from all hypothesis results files.
Creates master insight database for synthesis document.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Base directory
BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys")
ITERATIONS_DIR = BASE_DIR / "iterations"

# Insight storage
insights = []
insight_id_counter = 1

# Pattern matching for key findings
KEY_PATTERNS = {
    'confirmed': r'(?i)(confirmed|validated|significant|p\s*[<≤]\s*0\.0[0-5]|r²\s*[>≥]\s*0\.[6-9]|auc\s*[>≥]\s*0\.[8-9])',
    'rejected': r'(?i)(rejected|failed|not\s+significant|p\s*[>≥]\s*0\.05|no\s+evidence)',
    'biomarker': r'(?i)(biomarker|clinical|therapeutic|drug\s+target|treatment)',
    'mechanism': r'(?i)(pathway|mechanism|cascade|signaling|drives|causes)',
    'method': r'(?i)(lstm|gnn|autoencoder|pca|random\s+forest|xgboost|neural|machine\s+learning)',
}

def extract_hypothesis_info(filepath):
    """Extract hypothesis ID and agent from filepath."""
    parts = filepath.parts

    # Find iteration number
    iteration = None
    for part in parts:
        if 'iteration_' in part:
            iteration = part.split('_')[1]
            break

    # Find hypothesis ID
    hypothesis = None
    for part in parts:
        match = re.search(r'hypothesis_(\d+)', part)
        if match:
            hypothesis = f"H{match.group(1).zfill(2)}"
            break

    # Find agent
    agent = 'CLAUDE' if 'claude_code' in str(filepath) else 'CODEX'

    return iteration, hypothesis, agent

def read_file_safely(filepath):
    """Read file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def extract_score(content):
    """Extract self-evaluation score if present."""
    score_patterns = [
        r'(?i)final\s+score:?\s*(\d+)/100',
        r'(?i)total\s+score:?\s*(\d+)/100',
        r'(?i)score:?\s*(\d+)/100',
    ]
    for pattern in score_patterns:
        match = re.search(pattern, content)
        if match:
            return int(match.group(1))
    return None

def extract_key_findings(content):
    """Extract key findings section."""
    findings = []

    # Try to find key findings section
    sections = re.split(r'\n#{1,3}\s+', content)
    for section in sections:
        if any(keyword in section.lower()[:100] for keyword in ['key finding', 'main result', 'breakthrough', 'discovery']):
            # Extract bullet points or numbered items
            bullets = re.findall(r'(?:^|\n)(?:\*\*?|-|\d+\.)\s+(.+?)(?=\n(?:\*\*?|-|\d+\.)|$)', section, re.MULTILINE)
            findings.extend([b.strip() for b in bullets if len(b.strip()) > 20])

    return findings[:10]  # Top 10 findings

def extract_r_squared_values(content):
    """Extract R² values."""
    r2_values = re.findall(r'[rR]²\s*=\s*([-\d.]+)', content)
    valid_r2 = []
    for v in r2_values:
        if not v:
            continue
        try:
            # Remove trailing periods
            v_clean = v.rstrip('.')
            val = float(v_clean)
            if abs(val) <= 1.5:
                valid_r2.append(val)
        except ValueError:
            continue
    return valid_r2

def extract_p_values(content):
    """Extract p-values."""
    p_values = re.findall(r'p\s*[<≤=]\s*([\d.e-]+)', content)
    valid_p_values = []
    for v in p_values:
        if not v:
            continue
        try:
            val = float(v)
            if val <= 1.0:
                valid_p_values.append(val)
        except ValueError:
            # Skip invalid values like ranges
            continue
    return valid_p_values

def categorize_insight(content, findings):
    """Categorize insight based on content."""
    categories = []

    if re.search(KEY_PATTERNS['confirmed'], content):
        categories.append('CONFIRMED')
    if re.search(KEY_PATTERNS['rejected'], content):
        categories.append('REJECTED')
    if re.search(KEY_PATTERNS['biomarker'], content):
        categories.append('BIOMARKER')
    if re.search(KEY_PATTERNS['mechanism'], content):
        categories.append('MECHANISM')
    if re.search(KEY_PATTERNS['method'], content):
        categories.append('METHODOLOGY')

    # Check for specific patterns
    if any('s100' in f.lower() for f in findings):
        categories.append('S100_PATHWAY')
    if any('serpin' in f.lower() for f in findings):
        categories.append('SERPIN_CASCADE')
    if any('coagulation' in f.lower() for f in findings):
        categories.append('COAGULATION')
    if any(w in content.lower() for w in ['velocity', 'aging rate', 'tissue-specific']):
        categories.append('TISSUE_VELOCITY')

    return list(set(categories))

def extract_proteins_mentioned(content):
    """Extract protein names mentioned."""
    # Common ECM protein patterns
    protein_patterns = [
        r'\b([A-Z][A-Z0-9]{2,}[A-Z]?\d*)\b',  # Generic protein names
        r'\b(COL\d+A\d+)\b',  # Collagens
        r'\b(S100A\d+)\b',     # S100 proteins
        r'\b(SERPIN[A-Z]\d+)\b',  # Serpins
        r'\b(TGM\d+)\b',       # Transglutaminases
        r'\b(LOX[L]?\d*)\b',   # Lysyl oxidases
    ]

    proteins = set()
    for pattern in protein_patterns:
        matches = re.findall(pattern, content)
        proteins.update(matches)

    # Filter out common false positives
    exclude = {'LSTM', 'HTML', 'HTTP', 'JSON', 'UMAP', 'SHAP', 'AUC', 'MSE'}
    proteins = {p for p in proteins if p not in exclude and len(p) >= 3}

    return list(proteins)[:50]  # Top 50

def process_results_file(filepath):
    """Process a single results file and extract insights."""
    global insight_id_counter

    iteration, hypothesis, agent = extract_hypothesis_info(filepath)
    content = read_file_safely(filepath)

    if not content or len(content) < 100:
        return None

    # Extract components
    score = extract_score(content)
    findings = extract_key_findings(content)
    categories = categorize_insight(content, findings)
    r2_values = extract_r_squared_values(content)
    p_values = extract_p_values(content)
    proteins = extract_proteins_mentioned(content)

    # Get title from first line
    first_lines = content.split('\n')[:5]
    title = None
    for line in first_lines:
        if line.strip() and not line.startswith('#'):
            title = line.strip()[:100]
            break

    if not title:
        title = f"{hypothesis} - {agent} Results"

    # Create insight record
    insight = {
        'id': f"INS-{insight_id_counter:03d}",
        'iteration': iteration,
        'hypothesis': hypothesis,
        'agent': agent,
        'title': title,
        'filepath': str(filepath),
        'score': score,
        'categories': categories,
        'key_findings': findings,
        'r2_values': r2_values,
        'p_values': p_values,
        'proteins': proteins,
        'content_length': len(content),
        'has_confirmed_evidence': 'CONFIRMED' in categories,
        'has_rejected_evidence': 'REJECTED' in categories,
        'has_biomarker': 'BIOMARKER' in categories,
        'has_mechanism': 'MECHANISM' in categories,
    }

    insight_id_counter += 1
    return insight

def main():
    """Main extraction process."""
    print("🔍 Extracting insights from all hypothesis results...")

    # Find all results files
    results_files = list(ITERATIONS_DIR.glob("**/90_results*.md"))
    print(f"📁 Found {len(results_files)} results files")

    # Process each file
    all_insights = []
    for i, filepath in enumerate(results_files, 1):
        print(f"Processing {i}/{len(results_files)}: {filepath.name}")
        insight = process_results_file(filepath)
        if insight:
            all_insights.append(insight)

    print(f"\n✅ Extracted {len(all_insights)} insights")

    # Save to JSON
    output_file = BASE_DIR / "comprehensive_insights_extraction.json"
    with open(output_file, 'w') as f:
        json.dump(all_insights, f, indent=2)

    print(f"💾 Saved to: {output_file}")

    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # By hypothesis
    by_hypo = defaultdict(list)
    for ins in all_insights:
        by_hypo[ins['hypothesis']].append(ins)

    print(f"\n📊 Insights by Hypothesis:")
    for hypo in sorted(by_hypo.keys()):
        agents = [i['agent'] for i in by_hypo[hypo]]
        print(f"  {hypo}: {len(by_hypo[hypo])} insights ({', '.join(set(agents))})")

    # By agent
    by_agent = defaultdict(list)
    for ins in all_insights:
        by_agent[ins['agent']].append(ins)

    print(f"\n🤖 Insights by Agent:")
    for agent, items in by_agent.items():
        avg_score = sum(i['score'] for i in items if i['score']) / len([i for i in items if i['score']]) if items else 0
        print(f"  {agent}: {len(items)} insights (avg score: {avg_score:.1f})")

    # By category
    category_counts = defaultdict(int)
    for ins in all_insights:
        for cat in ins['categories']:
            category_counts[cat] += 1

    print(f"\n🏷️  Top Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat}: {count}")

    # Top proteins
    protein_counts = defaultdict(int)
    for ins in all_insights:
        for prot in ins['proteins']:
            protein_counts[prot] += 1

    print(f"\n🧬 Top 20 Proteins Mentioned:")
    for prot, count in sorted(protein_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {prot}: {count} insights")

    # Agreement analysis
    print(f"\n🤝 Agent Agreement Analysis:")
    both_tested = defaultdict(lambda: {'claude': None, 'codex': None})
    for ins in all_insights:
        hypo = ins['hypothesis']
        agent = ins['agent'].lower()
        both_tested[hypo][agent] = ins

    agreements = {'BOTH': 0, 'CLAUDE_ONLY': 0, 'CODEX_ONLY': 0, 'DISAGREE': 0}
    for hypo, agents in both_tested.items():
        if agents['claude'] and agents['codex']:
            # Check if both confirmed or both rejected
            c_conf = agents['claude']['has_confirmed_evidence']
            c_rej = agents['claude']['has_rejected_evidence']
            x_conf = agents['codex']['has_confirmed_evidence']
            x_rej = agents['codex']['has_rejected_evidence']

            if (c_conf and x_conf) or (c_rej and x_rej):
                agreements['BOTH'] += 1
            else:
                agreements['DISAGREE'] += 1
        elif agents['claude']:
            agreements['CLAUDE_ONLY'] += 1
        elif agents['codex']:
            agreements['CODEX_ONLY'] += 1

    for k, v in agreements.items():
        print(f"  {k}: {v}")

    print("\n" + "="*60)
    print("✅ Extraction complete!")
    print("="*60)

    return all_insights

if __name__ == "__main__":
    main()
