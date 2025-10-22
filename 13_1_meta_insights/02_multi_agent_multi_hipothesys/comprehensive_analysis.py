#!/usr/bin/env python3
"""
Comprehensive Multi-Hypothesis Analysis Generator
Creates all required deliverables for H01-H21 analysis
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import re

BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys")
ITERATIONS_DIR = BASE_DIR / "iterations"

# Load extracted data
with open(BASE_DIR / "extracted_results_data.json", 'r') as f:
    ALL_DATA = json.load(f)

# Create output directories
(BASE_DIR / "detailed_hypothesis_comparisons").mkdir(exist_ok=True)
(BASE_DIR / "iteration_synthesis").mkdir(exist_ok=True)

def get_hypothesis_title(hyp_id: str) -> str:
    """Map hypothesis ID to human-readable title"""
    titles = {
        "H01": "Compartment Mechanical Stress Antagonism",
        "H02": "Serpin Cascade Dysregulation",
        "H03": "Tissue Aging Velocity Clocks",
        "H04": "Deep Protein Embeddings",
        "H05": "GNN Aging Networks",
        "H06": "ML Ensemble Biomarkers",
        "H07": "Coagulation Central Hub",
        "H08": "S100 Calcium Signaling",
        "H09": "Temporal RNN Trajectories",
        "H10": "Calcium Signaling Cascade",
        "H11": "Standardized Temporal Trajectories",
        "H12": "Metabolic-Mechanical Transition",
        "H13": "Independent Dataset Validation",
        "H14": "Serpin Centrality Resolution",
        "H15": "Ovary-Heart Transition Biology",
        "H16": "H13 Validation Completion",
        "H17": "SERPINE1 Precision Target",
        "H18": "Multi-Modal AI Integration",
        "H19": "Metabolomics Phase 1",
        "H20": "Cross-Species Conservation",
        "H21": "Browser Automation (Unblocks H16)"
    }
    return titles.get(hyp_id, f"Hypothesis {hyp_id}")

def get_iteration(hyp_id: str) -> int:
    """Map hypothesis to iteration number"""
    iteration_map = {
        "H01": 1, "H02": 1, "H03": 1,
        "H04": 2, "H05": 2, "H06": 2,
        "H07": 3, "H08": 3, "H09": 3,
        "H10": 4, "H11": 4, "H12": 4, "H13": 4, "H14": 4, "H15": 4,
        "H16": 5, "H17": 5, "H18": 5, "H19": 5, "H20": 5,
        "H21": 6
    }
    return iteration_map.get(hyp_id, 0)

def analyze_agreement(claude_data: Dict, codex_data: Dict) -> Dict:
    """Analyze agreement between Claude and Codex"""
    if not claude_data or not codex_data:
        return {"agreement": "N/A", "reason": "Missing data"}

    # Score difference
    claude_score = claude_data.get('score')
    codex_score = codex_data.get('score')

    score_diff = None
    if claude_score and codex_score:
        score_diff = abs(claude_score - codex_score)

    # Status agreement
    claude_status = claude_data.get('status', 'UNKNOWN')
    codex_status = codex_data.get('status', 'UNKNOWN')

    status_agree = claude_status == codex_status

    # Determine agreement level
    if score_diff is not None:
        if score_diff < 10 and status_agree:
            agreement = "FULL AGREEMENT"
        elif score_diff < 20:
            agreement = "PARTIAL AGREEMENT"
        else:
            agreement = "DISAGREEMENT"
    else:
        agreement = "INCOMPLETE DATA"

    return {
        "agreement": agreement,
        "score_diff": score_diff,
        "claude_score": claude_score,
        "codex_score": codex_score,
        "claude_status": claude_status,
        "codex_status": codex_status,
        "status_agree": status_agree
    }

def create_hypothesis_comparison(hyp_id: str):
    """Create detailed comparison document for one hypothesis"""

    # Get data for both agents
    claude_data = next((d for d in ALL_DATA if d['hypothesis_id'] == hyp_id and d['agent'] == 'claude_code'), None)
    codex_data = next((d for d in ALL_DATA if d['hypothesis_id'] == hyp_id and d['agent'] == 'codex'), None)

    if not claude_data and not codex_data:
        return  # Skip if no data

    agreement_analysis = analyze_agreement(claude_data, codex_data)

    # Create markdown document
    title = get_hypothesis_title(hyp_id)
    md_content = f"""# {hyp_id}: {title}

## Hypothesis Statement

*[Extracted from task files]*

## Claude Code Results

"""

    if claude_data:
        md_content += f"""- **Status:** {claude_data['status']} (Score: {claude_data['score']}/100)
- **Thesis:** {claude_data['thesis'][:500]}...

### Key Metrics
{json.dumps(claude_data.get('metrics', {}), indent=2)}

### Verdict
{claude_data['status']}
"""
    else:
        md_content += "- **Status:** NO DATA\n"

    md_content += f"""

## Codex Results

"""

    if codex_data:
        md_content += f"""- **Status:** {codex_data['status']} (Score: {codex_data['score']}/100)
- **Thesis:** {codex_data['thesis'][:500]}...

### Key Metrics
{json.dumps(codex_data.get('metrics', {}), indent=2)}

### Verdict
{codex_data['status']}
"""
    else:
        md_content += "- **Status:** NO DATA\n"

    md_content += f"""

## CONSENSUS ANALYSIS

- **Agreement Level:** {agreement_analysis['agreement']}
- **Score Difference:** {agreement_analysis.get('score_diff', 'N/A')}
- **Status Match:** {'YES' if agreement_analysis.get('status_agree') else 'NO'}

### Resolution
Claude Status: {agreement_analysis.get('claude_status', 'N/A')}
Codex Status: {agreement_analysis.get('codex_status', 'N/A')}

### Final Verdict
*[Based on combined evidence]*

### Follow-up Needed
*[To be determined]*

"""

    # Save to file
    filename = BASE_DIR / "detailed_hypothesis_comparisons" / f"{hyp_id}_{title.lower().replace(' ', '_').replace('-', '_')}.md"
    with open(filename, 'w') as f:
        f.write(md_content)

    print(f"Created: {filename.name}")

def create_master_table():
    """Create master significance table with all hypotheses"""

    rows = []
    for hyp_id in [f"H{i:02d}" for i in range(1, 22)]:
        # Get data
        claude_data = next((d for d in ALL_DATA if d['hypothesis_id'] == hyp_id and d['agent'] == 'claude_code'), None)
        codex_data = next((d for d in ALL_DATA if d['hypothesis_id'] == hyp_id and d['agent'] == 'codex'), None)

        agreement = analyze_agreement(claude_data, codex_data)

        title = get_hypothesis_title(hyp_id)

        rows.append({
            'Rank': 0,  # Will assign later
            'Hypothesis_ID': hyp_id,
            'Title': title,
            'Status': agreement.get('claude_status', 'UNKNOWN'),
            'Claude_Score': agreement.get('claude_score'),
            'Codex_Score': agreement.get('codex_score'),
            'Avg_Score': (agreement.get('claude_score', 0) + agreement.get('codex_score', 0)) / 2 if agreement.get('claude_score') and agreement.get('codex_score') else agreement.get('claude_score') or agreement.get('codex_score') or 0,
            'Agreement': agreement['agreement'],
            'Iteration': get_iteration(hyp_id),
        })

    df = pd.DataFrame(rows)

    # Sort by average score (descending)
    df = df.sort_values('Avg_Score', ascending=False)
    df['Rank'] = range(1, len(df) + 1)

    # Save
    output_file = BASE_DIR / "master_significance_table.csv"
    df.to_csv(output_file, index=False)
    print(f"\nCreated master table: {output_file}")

    return df

def create_dependency_tree():
    """Create hypothesis dependency tree"""

    dependencies = {
        "H01": {"parents": [], "children": ["H21"]},
        "H02": {"parents": [], "children": ["H07", "H14"]},
        "H03": {"parents": [], "children": ["H08", "H11", "H15"]},
        "H04": {"parents": [], "children": ["H18"]},
        "H05": {"parents": [], "children": ["H18"]},
        "H06": {"parents": [], "children": ["H16"]},
        "H07": {"parents": ["H02"], "children": []},
        "H08": {"parents": ["H03"], "children": ["H10", "H18"]},
        "H09": {"parents": [], "children": []},
        "H10": {"parents": ["H08"], "children": []},
        "H11": {"parents": ["H03"], "children": ["H12", "H18"]},
        "H12": {"parents": ["H11"], "children": []},
        "H13": {"parents": [], "children": ["H16"]},
        "H14": {"parents": ["H02"], "children": ["H17"]},
        "H15": {"parents": ["H03"], "children": []},
        "H16": {"parents": ["H06", "H13"], "children": ["H21"]},
        "H17": {"parents": ["H14"], "children": []},
        "H18": {"parents": ["H04", "H05", "H08", "H11"], "children": []},
        "H19": {"parents": [], "children": []},
        "H20": {"parents": [], "children": []},
        "H21": {"parents": ["H01", "H16"], "children": []},
    }

    # Save as JSON
    json_file = BASE_DIR / "hypothesis_dependency_tree.json"
    with open(json_file, 'w') as f:
        json.dump(dependencies, f, indent=2)

    # Create text tree
    tree_text = """HYPOTHESIS DEPENDENCY TREE

ROOT HYPOTHESES (No parents):
├── H01: Compartment Mechanical Stress (REJECTED) → H21
├── H02: Serpin Cascade → H07, H14
├── H03: Tissue Aging Velocities (BREAKTHROUGH) → H08, H11, H15
├── H04: Deep Protein Embeddings → H18
├── H05: GNN Aging Networks → H18
├── H06: ML Ensemble Biomarkers → H16
├── H09: Temporal RNN Trajectories
├── H13: Independent Dataset Validation → H16
├── H19: Metabolomics Phase 1 (BLOCKED)
└── H20: Cross-Species Conservation

DEPENDENT HYPOTHESES:
├── H07: Coagulation Hub ← H02
├── H08: S100 Calcium Signaling ← H03 → H10, H18
├── H10: Calcium Cascade ← H08
├── H11: Temporal Trajectories ← H03 → H12, H18
├── H12: Metabolic-Mechanical Transition ← H11
├── H14: Serpin Centrality ← H02 → H17
├── H15: Ovary-Heart Transition ← H03
├── H16: External Validation ← H06, H13 → H21
├── H17: SERPINE1 Drug Target ← H14
└── H18: Multi-Modal Integration ← H04, H05, H08, H11

BLOCKING CHAIN:
H13 (incomplete) → H16 (blocked) → H21 (browser automation) → UNBLOCK

"""

    text_file = BASE_DIR / "hypothesis_dependency_tree.txt"
    with open(text_file, 'w') as f:
        f.write(tree_text)

    print(f"Created dependency tree: {json_file} and {text_file}")

def main():
    """Main execution"""
    print("=" * 60)
    print("COMPREHENSIVE MULTI-HYPOTHESIS ANALYSIS")
    print("=" * 60)

    # 1. Create per-hypothesis comparisons
    print("\n1. Creating detailed hypothesis comparisons...")
    for hyp_id in [f"H{i:02d}" for i in range(1, 22)]:
        create_hypothesis_comparison(hyp_id)

    # 2. Create master table
    print("\n2. Creating master significance table...")
    master_df = create_master_table()
    print(master_df[['Rank', 'Hypothesis_ID', 'Title', 'Avg_Score', 'Status', 'Agreement']].to_string(index=False))

    # 3. Create dependency tree
    print("\n3. Creating dependency tree...")
    create_dependency_tree()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
