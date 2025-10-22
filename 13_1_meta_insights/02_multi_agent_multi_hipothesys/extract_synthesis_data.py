#!/usr/bin/env python3
"""
Extract key metrics from all hypothesis results files for synthesis report.
Generates master ranking table and Claude vs Codex comparison.
"""

import os
import re
import pandas as pd
import glob
from pathlib import Path

# Base directory
BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations")

def extract_thesis(content):
    """Extract thesis statement from markdown"""
    match = re.search(r'\*\*Thesis:\*\*\s*(.+?)(?:\n\n|\*\*Overview)', content, re.DOTALL)
    if match:
        return match.group(1).strip()[:200]  # First 200 chars
    # Try alternate format
    match = re.search(r'^Thesis:\s*(.+?)(?:\n\n|Overview)', content, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()[:200]
    return "N/A"

def extract_score(content):
    """Extract self-evaluation score"""
    # Look for patterns like "Score: 96/100", "Total Score: 84/100"
    patterns = [
        r'Total Score:\s*\*?\*?(\d+)/100',
        r'Score:\s*\*?\*?(\d+)/100',
        r'Final Score:\s*\*?\*?(\d+)/100',
        r'Self-Evaluation.*?(\d+)/100',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1))
    return None

def extract_metrics(content):
    """Extract performance metrics (R2, AUC, accuracy, etc.)"""
    metrics = {}

    def safe_float(match):
        """Safely extract float from regex match"""
        try:
            val = match.group(1).strip()
            # Remove trailing periods
            val = val.rstrip('.')
            return float(val)
        except (ValueError, AttributeError):
            return None

    # R-squared / R²
    r2_match = re.search(r'R[²2]?\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if r2_match:
        val = safe_float(r2_match)
        if val is not None:
            metrics['R2'] = val

    # AUC
    auc_match = re.search(r'AUC\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if auc_match:
        val = safe_float(auc_match)
        if val is not None:
            metrics['AUC'] = val

    # Accuracy
    acc_match = re.search(r'Accuracy\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if acc_match:
        val = safe_float(acc_match)
        if val is not None:
            metrics['Accuracy'] = val

    # F1 Score
    f1_match = re.search(r'F1[- ]?[Ss]core\s*[=:]\s*([0-9.]+)', content)
    if f1_match:
        val = safe_float(f1_match)
        if val is not None:
            metrics['F1'] = val

    # MSE
    mse_match = re.search(r'MSE\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if mse_match:
        val = safe_float(mse_match)
        if val is not None:
            metrics['MSE'] = val

    # ARI (Adjusted Rand Index)
    ari_match = re.search(r'ARI\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if ari_match:
        val = safe_float(ari_match)
        if val is not None:
            metrics['ARI'] = val

    # p-value
    p_match = re.search(r'p\s*[=:]\s*([0-9.e-]+)', content, re.IGNORECASE)
    if p_match:
        val = safe_float(p_match)
        if val is not None:
            metrics['p_value'] = val

    return metrics

def extract_status(content, score):
    """Determine hypothesis status (SUCCESS/PARTIAL/FAILURE)"""
    # Look for explicit verdicts
    if re.search(r'CONFIRMED|SUCCESS|SUPPORTED|VALIDATED', content, re.IGNORECASE):
        return "SUCCESS"
    if re.search(r'REJECTED|FAILED|NOT SUPPORTED|HYPOTHESIS.*REJECTED', content, re.IGNORECASE):
        return "FAILURE"
    if re.search(r'PARTIAL|MIXED EVIDENCE|PARTIALLY', content, re.IGNORECASE):
        return "PARTIAL"

    # Use score if available
    if score:
        if score >= 90:
            return "SUCCESS"
        elif score >= 70:
            return "PARTIAL"
        else:
            return "FAILURE"

    return "UNKNOWN"

def extract_key_finding(content):
    """Extract one-sentence key finding"""
    # Look for Key Finding, Main Finding, Primary Conclusion patterns
    patterns = [
        r'Key Finding[s]?:\s*(.+?)(?:\n\n|\*\*)',
        r'Primary Conclusion:\s*(.+?)(?:\n\n|\*\*)',
        r'Main Finding:\s*(.+?)(?:\n\n|\*\*)',
        r'Hypothesis Verdict:\s*(.+?)(?:\n\n|\*\*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            finding = match.group(1).strip()
            # Take first sentence
            first_sent = re.split(r'[.!?]\s', finding)[0] + '.'
            return first_sent[:200]
    return "N/A"

def process_all_results():
    """Process all results files and create synthesis data"""
    results = []

    # Find all results files
    result_files = glob.glob(str(BASE_DIR / "**" / "90_results*.md"), recursive=True)

    for filepath in sorted(result_files):
        path = Path(filepath)

        # Extract hypothesis number and agent
        hypothesis_match = re.search(r'hypothesis_(\d+)_(.+)', str(path))
        if not hypothesis_match:
            continue

        h_num = int(hypothesis_match.group(1))
        h_name = hypothesis_match.group(2).replace('_', ' ').title()

        # Determine agent
        agent = "claude" if "claude" in path.stem else "codex"

        print(f"Processing H{h_num:02d} ({agent}): {path.stem}")

        # Read content
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"  ERROR reading {filepath}: {e}")
            continue

        # Extract data
        thesis = extract_thesis(content)
        score = extract_score(content)
        metrics = extract_metrics(content)
        status = extract_status(content, score)
        key_finding = extract_key_finding(content)

        results.append({
            'Hypothesis_ID': f"H{h_num:02d}",
            'Hypothesis_Name': h_name,
            'Agent': agent,
            'Thesis': thesis,
            'Key_Finding': key_finding,
            'Status': status,
            'Score': score,
            **metrics,  # Add all extracted metrics
            'File': str(path.relative_to(BASE_DIR))
        })

    return pd.DataFrame(results)

def create_master_ranking(df):
    """Create master ranking table aggregating both agents"""
    # Group by hypothesis, aggregate agent results
    ranking_data = []

    for h_id in sorted(df['Hypothesis_ID'].unique()):
        h_df = df[df['Hypothesis_ID'] == h_id]

        # Get data from both agents
        claude_row = h_df[h_df['Agent'] == 'claude'].iloc[0] if len(h_df[h_df['Agent'] == 'claude']) > 0 else None
        codex_row = h_df[h_df['Agent'] == 'codex'].iloc[0] if len(h_df[h_df['Agent'] == 'codex']) > 0 else None

        # Determine agreement
        if claude_row is not None and codex_row is not None:
            if claude_row['Status'] == codex_row['Status']:
                agreement = "AGREE"
            elif (claude_row['Status'] in ['SUCCESS', 'PARTIAL'] and codex_row['Status'] in ['SUCCESS', 'PARTIAL']) or \
                 (claude_row['Status'] == 'FAILURE' and codex_row['Status'] == 'FAILURE'):
                agreement = "PARTIAL"
            else:
                agreement = "DISAGREE"
            both_completed = True
        else:
            agreement = "N/A"
            both_completed = False

        # Use Claude results as primary (or Codex if Claude missing)
        primary = claude_row if claude_row is not None else codex_row
        if primary is None:
            continue

        # Determine clinical potential (heuristic based on hypothesis type)
        clinical_potential = "MEDIUM"  # Default
        if any(word in primary['Hypothesis_Name'].lower() for word in ['biomarker', 'therapeutic', 'clinical', 'drug']):
            clinical_potential = "HIGH"
        elif any(word in primary['Hypothesis_Name'].lower() for word in ['network', 'embedding', 'cluster']):
            clinical_potential = "LOW"

        # Extract best metric for ranking
        best_metric = None
        metric_type = None
        for m in ['R2', 'AUC', 'Accuracy', 'ARI', 'F1']:
            if m in primary and primary[m] is not None:
                best_metric = primary[m]
                metric_type = m
                break

        ranking_data.append({
            'Rank': 0,  # Will be filled later
            'Hypothesis_ID': h_id,
            'Hypothesis_Name': primary['Hypothesis_Name'],
            'Question': primary['Thesis'][:150] + '...' if len(primary['Thesis']) > 150 else primary['Thesis'],
            'Key_Finding': primary['Key_Finding'],
            'Metrics': f"{metric_type}={best_metric:.3f}" if best_metric else "N/A",
            'Status': primary['Status'],
            'Claude_Codex_Agreement': agreement,
            'Clinical_Potential': clinical_potential,
            'Claude_Score': claude_row['Score'] if claude_row is not None else None,
            'Codex_Score': codex_row['Score'] if codex_row is not None else None,
            'Both_Completed': both_completed
        })

    ranking_df = pd.DataFrame(ranking_data)

    # Rank by: 1) Status (SUCCESS > PARTIAL > FAILURE), 2) Score (average), 3) Clinical potential
    status_order = {'SUCCESS': 3, 'PARTIAL': 2, 'FAILURE': 1, 'UNKNOWN': 0}
    clinical_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

    ranking_df['status_rank'] = ranking_df['Status'].map(status_order)
    ranking_df['clinical_rank'] = ranking_df['Clinical_Potential'].map(clinical_order)
    ranking_df['avg_score'] = ranking_df[['Claude_Score', 'Codex_Score']].mean(axis=1, skipna=True)

    ranking_df = ranking_df.sort_values(['status_rank', 'avg_score', 'clinical_rank'],
                                        ascending=[False, False, False])
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)

    # Drop temporary columns
    ranking_df = ranking_df.drop(columns=['status_rank', 'clinical_rank', 'avg_score'])

    return ranking_df

def create_agent_comparison(df):
    """Create Claude vs Codex comparison table"""
    claude_df = df[df['Agent'] == 'claude']
    codex_df = df[df['Agent'] == 'codex']

    comparison = {
        'Metric': [
            'Total Hypotheses Attempted',
            'Successfully Completed',
            'Partial Success',
            'Failures',
            'Average Score',
            'Median Score',
            'Hypotheses with Score >= 90',
            'Hypotheses with Score < 70',
        ],
        'Claude': [
            len(claude_df),
            len(claude_df[claude_df['Status'] == 'SUCCESS']),
            len(claude_df[claude_df['Status'] == 'PARTIAL']),
            len(claude_df[claude_df['Status'] == 'FAILURE']),
            claude_df['Score'].mean() if 'Score' in claude_df else 0,
            claude_df['Score'].median() if 'Score' in claude_df else 0,
            len(claude_df[claude_df['Score'] >= 90]) if 'Score' in claude_df else 0,
            len(claude_df[claude_df['Score'] < 70]) if 'Score' in claude_df else 0,
        ],
        'Codex': [
            len(codex_df),
            len(codex_df[codex_df['Status'] == 'SUCCESS']),
            len(codex_df[codex_df['Status'] == 'PARTIAL']),
            len(codex_df[codex_df['Status'] == 'FAILURE']),
            codex_df['Score'].mean() if 'Score' in codex_df else 0,
            codex_df['Score'].median() if 'Score' in codex_df else 0,
            len(codex_df[codex_df['Score'] >= 90]) if 'Score' in codex_df else 0,
            len(codex_df[codex_df['Score'] < 70]) if 'Score' in codex_df else 0,
        ]
    }

    comparison_df = pd.DataFrame(comparison)

    # Add agreement statistics
    hypotheses = df['Hypothesis_ID'].unique()
    agree_count = 0
    partial_count = 0
    disagree_count = 0

    for h_id in hypotheses:
        h_df = df[df['Hypothesis_ID'] == h_id]
        if len(h_df) == 2:  # Both agents completed
            claude_status = h_df[h_df['Agent'] == 'claude'].iloc[0]['Status']
            codex_status = h_df[h_df['Agent'] == 'codex'].iloc[0]['Status']

            if claude_status == codex_status:
                agree_count += 1
            elif (claude_status in ['SUCCESS', 'PARTIAL'] and codex_status in ['SUCCESS', 'PARTIAL']):
                partial_count += 1
            else:
                disagree_count += 1

    total_both = agree_count + partial_count + disagree_count

    agreement_stats = pd.DataFrame({
        'Metric': ['Agreement Rate (%)', 'Partial Agreement (%)', 'Disagreement (%)'],
        'Value': [
            100 * agree_count / total_both if total_both > 0 else 0,
            100 * partial_count / total_both if total_both > 0 else 0,
            100 * disagree_count / total_both if total_both > 0 else 0
        ]
    })

    return comparison_df, agreement_stats

if __name__ == "__main__":
    print("Extracting synthesis data from all results files...")
    print("=" * 80)

    # Process all results
    all_results = process_all_results()

    print(f"\nProcessed {len(all_results)} results files")
    print(f"Unique hypotheses: {all_results['Hypothesis_ID'].nunique()}")

    # Create master ranking
    master_ranking = create_master_ranking(all_results)

    # Create agent comparison
    agent_comparison, agreement_stats = create_agent_comparison(all_results)

    # Save outputs
    output_dir = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys")

    all_results.to_csv(output_dir / "raw_extraction_data.csv", index=False)
    master_ranking.to_csv(output_dir / "synthesis_master_ranking_table.csv", index=False)
    agent_comparison.to_csv(output_dir / "claude_vs_codex_comparison.csv", index=False)
    agreement_stats.to_csv(output_dir / "agreement_statistics.csv", index=False)

    print("\n" + "=" * 80)
    print("OUTPUT FILES CREATED:")
    print(f"  1. raw_extraction_data.csv ({len(all_results)} rows)")
    print(f"  2. synthesis_master_ranking_table.csv ({len(master_ranking)} hypotheses)")
    print(f"  3. claude_vs_codex_comparison.csv")
    print(f"  4. agreement_statistics.csv")
    print("\n" + "=" * 80)
    print("\nTOP 10 HYPOTHESES (by rank):")
    print(master_ranking[['Rank', 'Hypothesis_ID', 'Hypothesis_Name', 'Status', 'Metrics']].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("\nCLAUDE VS CODEX COMPARISON:")
    print(agent_comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("\nAGREEMENT STATISTICS:")
    print(agreement_stats.to_string(index=False))

    print("\n" + "=" * 80)
    print("DONE!")
