#!/usr/bin/env python3
"""
Extract key metrics from all hypothesis results files (H01-H21)
Creates structured data for comprehensive analysis
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations")

def extract_score(text: str) -> float:
    """Extract self-evaluation score from results file"""
    # Look for patterns like "Total Score: 96/100", "Score: 84/100", "88/100"
    patterns = [
        r'(?:Total\s+)?Score:\s*(\d+)/100',
        r'(?:Final\s+)?Score:\s*(\d+)/100',
        r'\*\*(\d+)/100\*\*',
        r'(\d+)/100\s*points?'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    return None

def extract_status(text: str) -> str:
    """Extract hypothesis status/verdict"""
    text_lower = text.lower()

    # Check for explicit verdicts
    if 'breakthrough' in text_lower or 'strongly confirmed' in text_lower:
        return 'BREAKTHROUGH'
    elif 'validated' in text_lower or 'confirmed' in text_lower:
        return 'VALIDATED'
    elif 'rejected' in text_lower or 'not supported' in text_lower:
        return 'REJECTED'
    elif 'blocked' in text_lower:
        return 'BLOCKED'
    elif 'incomplete' in text_lower or 'partial' in text_lower:
        return 'PARTIAL'
    else:
        return 'UNKNOWN'

def extract_thesis(text: str) -> str:
    """Extract thesis statement (first line after '**Thesis:**')"""
    match = re.search(r'\*\*Thesis:\*\*\s*(.+?)(?:\n\n|\*\*Overview)', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Alternative: Look for "Thesis:" at start of line
    match = re.search(r'^Thesis:\s*(.+?)$', text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    return ""

def extract_key_findings(text: str) -> List[str]:
    """Extract bullet points from key findings or conclusions"""
    findings = []

    # Look for sections like "Key Findings", "Conclusions", "Primary Conclusions"
    patterns = [
        r'(?:Key Findings|Primary Conclusions|Main Results):?\s*\n((?:[-•*]\s*.+\n?)+)',
        r'¶\d+\s+\*\*(.+?):\*\*',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            for match in matches[:5]:  # Limit to first 5
                findings.append(match.strip())

    return findings

def extract_metrics(text: str) -> Dict[str, Any]:
    """Extract quantitative metrics (p-values, R², AUC, etc.)"""
    metrics = {}

    # P-values
    p_matches = re.findall(r'p\s*[=<>]\s*([0-9.e-]+)', text, re.IGNORECASE)
    if p_matches:
        metrics['p_values'] = [float(p) for p in p_matches[:3]]

    # R² values
    r2_matches = re.findall(r'R[²2]\s*[=:]\s*([0-9.]+)', text)
    if r2_matches:
        metrics['R2'] = [float(r) for r in r2_matches[:3]]

    # AUC
    auc_matches = re.findall(r'AUC\s*[=:]\s*([0-9.]+)', text, re.IGNORECASE)
    if auc_matches:
        metrics['AUC'] = [float(a) for a in auc_matches[:3]]

    # Correlation
    corr_matches = re.findall(r'(?:ρ|rho|correlation)\s*[=:]\s*([0-9.-]+)', text, re.IGNORECASE)
    if corr_matches:
        metrics['correlation'] = [float(c) for c in corr_matches[:3]]

    return metrics

def process_results_file(filepath: Path) -> Dict[str, Any]:
    """Process a single results file and extract all relevant data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract agent name from path
        agent = 'claude_code' if 'claude_code' in str(filepath) else 'codex'

        # Extract hypothesis ID from path
        hyp_match = re.search(r'hypothesis_(\d+)_', str(filepath))
        hyp_id = f"H{hyp_match.group(1)}" if hyp_match else "UNKNOWN"

        return {
            'hypothesis_id': hyp_id,
            'agent': agent,
            'filepath': str(filepath),
            'thesis': extract_thesis(content),
            'score': extract_score(content),
            'status': extract_status(content),
            'key_findings': extract_key_findings(content),
            'metrics': extract_metrics(content),
            'file_size': len(content)
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    """Main extraction function"""
    results_files = list(BASE_DIR.glob("*/hypothesis_*/*/90_results_*.md"))

    print(f"Found {len(results_files)} results files")

    all_data = []
    for filepath in sorted(results_files):
        print(f"Processing: {filepath.name}")
        data = process_results_file(filepath)
        if data:
            all_data.append(data)

    # Save to JSON
    output_json = BASE_DIR.parent / "extracted_results_data.json"
    with open(output_json, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\nExtracted data from {len(all_data)} files")
    print(f"Saved to: {output_json}")

    # Create summary DataFrame
    summary_data = []
    for item in all_data:
        summary_data.append({
            'Hypothesis_ID': item['hypothesis_id'],
            'Agent': item['agent'],
            'Score': item['score'],
            'Status': item['status'],
            'Thesis_Length': len(item['thesis']),
            'N_Findings': len(item['key_findings']),
            'Has_Metrics': len(item['metrics']) > 0
        })

    df = pd.DataFrame(summary_data)

    # Pivot to show both agents side by side
    pivot = df.pivot_table(
        index='Hypothesis_ID',
        columns='Agent',
        values=['Score', 'Status'],
        aggfunc='first'
    )

    output_csv = BASE_DIR.parent / "results_summary.csv"
    pivot.to_csv(output_csv)
    print(f"Summary saved to: {output_csv}")

    # Print quick stats
    print(f"\n=== SUMMARY STATS ===")
    print(f"Total hypotheses: {df['Hypothesis_ID'].nunique()}")
    print(f"Hypotheses with 2 agents: {df.groupby('Hypothesis_ID').size().eq(2).sum()}")
    print(f"Hypotheses with 1 agent: {df.groupby('Hypothesis_ID').size().eq(1).sum()}")
    print(f"\nMean score (Claude): {df[df['Agent']=='claude_code']['Score'].mean():.1f}")
    print(f"Mean score (Codex): {df[df['Agent']=='codex']['Score'].mean():.1f}")

    print(f"\nStatus distribution:")
    print(df.groupby(['Status', 'Agent']).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()
