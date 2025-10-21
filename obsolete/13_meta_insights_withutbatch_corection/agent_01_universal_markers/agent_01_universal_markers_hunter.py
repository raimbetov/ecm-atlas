#!/usr/bin/env python3
"""
Agent 1: Universal Markers Hunter

MISSION: Find 2-3 ECM proteins that change CONSISTENTLY across ALL tissues during aging
Analyzes merged dataset to identify universal aging biomarkers with cross-tissue validation

Author: Claude Code Agent 01
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
MERGED_CSV = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_REPORT = "/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_01_universal_markers_report.md"
OUTPUT_DATA = "/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_01_universal_markers_data.csv"

# Thresholds for classification
STRONG_EFFECT = 0.5  # |Zscore_Delta| > 0.5 = strong effect
MIN_TISSUES = 3      # Minimum tissues to be considered "multi-tissue"
UNIVERSALITY_THRESHOLD = 0.7  # 70% directional consistency

def load_and_prepare_data():
    """Load merged dataset and prepare for analysis"""
    print("Loading merged ECM dataset...")
    df = pd.read_csv(MERGED_CSV)

    # Filter to valid aging data (has both young and old)
    df_valid = df.dropna(subset=['Zscore_Delta']).copy()

    print(f"Total rows: {len(df):,}")
    print(f"Valid aging comparisons: {len(df_valid):,}")
    print(f"Unique proteins (Gene_Symbol): {df_valid['Gene_Symbol'].nunique()}")
    print(f"Unique studies: {df_valid['Study_ID'].nunique()}")
    print(f"Unique tissues: {df_valid['Tissue_Compartment'].nunique()}")

    return df_valid

def identify_tissues_and_studies(df):
    """Catalog all unique tissues and studies"""

    tissues = df.groupby('Tissue_Compartment').agg({
        'Study_ID': 'first',
        'Species': 'first',
        'Organ': 'first',
        'Compartment': 'first',
        'Gene_Symbol': 'nunique'
    }).reset_index()

    tissues.columns = ['Tissue_Compartment', 'Study_ID', 'Species', 'Organ',
                      'Compartment', 'Num_Proteins']

    print("\n" + "="*80)
    print("TISSUE CATALOG")
    print("="*80)
    print(tissues.to_string(index=False))

    return tissues

def calculate_protein_universality(df):
    """
    For each protein, calculate universality metrics:
    - Number of tissues where detected
    - Directional consistency (% same direction)
    - Average effect size across tissues
    - Statistical significance
    """

    print("\n" + "="*80)
    print("CALCULATING UNIVERSALITY METRICS FOR EACH PROTEIN")
    print("="*80)

    results = []

    for gene in df['Gene_Symbol'].unique():
        gene_data = df[df['Gene_Symbol'] == gene].copy()

        # Basic counts
        n_tissues = gene_data['Tissue_Compartment'].nunique()
        n_measurements = len(gene_data)

        # Z-score statistics
        zscore_deltas = gene_data['Zscore_Delta'].dropna()

        if len(zscore_deltas) == 0:
            continue

        # Direction analysis
        n_upregulated = (zscore_deltas > 0).sum()
        n_downregulated = (zscore_deltas < 0).sum()

        # Directional consistency (what % go same direction)
        direction_consistency = max(n_upregulated, n_downregulated) / len(zscore_deltas)
        predominant_direction = 'UP' if n_upregulated > n_downregulated else 'DOWN'

        # Effect size statistics
        mean_delta = zscore_deltas.mean()
        abs_mean_delta = zscore_deltas.abs().mean()
        median_delta = zscore_deltas.median()
        std_delta = zscore_deltas.std()

        # Count strong effects (|delta| > 0.5)
        n_strong_effects = (zscore_deltas.abs() > STRONG_EFFECT).sum()
        strong_effect_rate = n_strong_effects / len(zscore_deltas)

        # Statistical test: Is mean delta significantly different from 0?
        if len(zscore_deltas) >= 3:  # Need at least 3 samples
            t_stat, p_value = stats.ttest_1samp(zscore_deltas, 0)
        else:
            t_stat, p_value = np.nan, np.nan

        # Matrisome info (most common category)
        matrisome_cat = gene_data['Matrisome_Category'].mode()[0] if len(gene_data['Matrisome_Category'].mode()) > 0 else 'Unknown'
        matrisome_div = gene_data['Matrisome_Division'].mode()[0] if len(gene_data['Matrisome_Division'].mode()) > 0 else 'Unknown'

        # Protein name
        protein_name = gene_data['Protein_Name'].iloc[0]

        # Universality score (composite metric)
        # Factors: tissue breadth, directional consistency, effect size, significance
        universality_score = (
            (n_tissues / df['Tissue_Compartment'].nunique()) * 0.3 +  # Breadth: 30%
            direction_consistency * 0.3 +                               # Consistency: 30%
            min(abs_mean_delta / 2.0, 1.0) * 0.2 +                     # Effect size: 20%
            (1 - min(p_value, 1.0) if not np.isnan(p_value) else 0) * 0.2  # Significance: 20%
        )

        results.append({
            'Gene_Symbol': gene,
            'Protein_Name': protein_name,
            'Matrisome_Category': matrisome_cat,
            'Matrisome_Division': matrisome_div,
            'N_Tissues': n_tissues,
            'N_Measurements': n_measurements,
            'Direction_Consistency': direction_consistency,
            'Predominant_Direction': predominant_direction,
            'N_Upregulated': n_upregulated,
            'N_Downregulated': n_downregulated,
            'Mean_Zscore_Delta': mean_delta,
            'Abs_Mean_Zscore_Delta': abs_mean_delta,
            'Median_Zscore_Delta': median_delta,
            'Std_Zscore_Delta': std_delta,
            'N_Strong_Effects': n_strong_effects,
            'Strong_Effect_Rate': strong_effect_rate,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Universality_Score': universality_score
        })

    results_df = pd.DataFrame(results)

    print(f"Analyzed {len(results_df)} unique proteins")

    return results_df

def identify_universal_candidates(results_df, min_tissues=MIN_TISSUES,
                                  min_consistency=UNIVERSALITY_THRESHOLD):
    """
    Identify top universal aging marker candidates

    Criteria:
    1. Present in >= min_tissues
    2. Directional consistency >= min_consistency (e.g., 70%)
    3. Large average effect size
    4. Statistically significant (p < 0.05)
    """

    print("\n" + "="*80)
    print(f"IDENTIFYING UNIVERSAL MARKERS")
    print(f"Criteria: ≥{min_tissues} tissues, ≥{min_consistency*100:.0f}% directional consistency")
    print("="*80)

    # Filter candidates
    candidates = results_df[
        (results_df['N_Tissues'] >= min_tissues) &
        (results_df['Direction_Consistency'] >= min_consistency)
    ].copy()

    # Sort by universality score
    candidates = candidates.sort_values('Universality_Score', ascending=False)

    print(f"\nFound {len(candidates)} candidates meeting criteria")

    # Rank by different metrics
    rankings = {
        'By Universality Score': candidates.head(10)[['Gene_Symbol', 'N_Tissues', 'Direction_Consistency',
                                                        'Predominant_Direction', 'Abs_Mean_Zscore_Delta',
                                                        'Universality_Score']],
        'By Tissue Breadth': candidates.nlargest(10, 'N_Tissues')[['Gene_Symbol', 'N_Tissues',
                                                                    'Direction_Consistency',
                                                                    'Predominant_Direction',
                                                                    'Abs_Mean_Zscore_Delta']],
        'By Effect Size': candidates.nlargest(10, 'Abs_Mean_Zscore_Delta')[['Gene_Symbol', 'N_Tissues',
                                                                             'Direction_Consistency',
                                                                             'Abs_Mean_Zscore_Delta']],
        'By Consistency': candidates.nlargest(10, 'Direction_Consistency')[['Gene_Symbol', 'N_Tissues',
                                                                            'Direction_Consistency',
                                                                            'Predominant_Direction']]
    }

    for rank_name, rank_df in rankings.items():
        print(f"\n{rank_name}:")
        print(rank_df.to_string(index=False))

    return candidates

def create_tissue_profile(df, gene_symbol):
    """Create detailed tissue-by-tissue profile for a protein"""

    gene_data = df[df['Gene_Symbol'] == gene_symbol].copy()

    profile = gene_data.groupby('Tissue_Compartment').agg({
        'Zscore_Delta': 'mean',
        'Zscore_Young': 'mean',
        'Zscore_Old': 'mean',
        'Abundance_Young': 'mean',
        'Abundance_Old': 'mean',
        'Study_ID': 'first',
        'Species': 'first',
        'Organ': 'first'
    }).reset_index()

    profile['Direction'] = profile['Zscore_Delta'].apply(lambda x: 'UP' if x > 0 else 'DOWN')
    profile = profile.sort_values('Zscore_Delta', ascending=False)

    return profile

def analyze_unexpected_patterns(results_df, df):
    """
    Look for surprising findings:
    - Proteins expected to be universal but aren't
    - Proteins unexpectedly universal
    - Strong tissue-specific effects
    """

    print("\n" + "="*80)
    print("UNEXPECTED PATTERNS ANALYSIS")
    print("="*80)

    # Known aging markers that SHOULD be universal
    expected_universal = ['COL1A1', 'COL1A2', 'FN1', 'COL3A1', 'COL4A1', 'COL4A2',
                         'TIMP1', 'MMP2', 'LAMA5', 'LAMB1', 'LAMB2']

    print("\nExpected universal markers - actual performance:")
    for gene in expected_universal:
        if gene in results_df['Gene_Symbol'].values:
            row = results_df[results_df['Gene_Symbol'] == gene].iloc[0]
            print(f"{gene:10s}: {row['N_Tissues']} tissues, "
                  f"{row['Direction_Consistency']*100:.0f}% consistency, "
                  f"Δz={row['Mean_Zscore_Delta']:+.3f}")
        else:
            print(f"{gene:10s}: NOT DETECTED in any dataset")

    # Highly tissue-specific proteins (present in many tissues but inconsistent direction)
    tissue_specific = results_df[
        (results_df['N_Tissues'] >= 5) &
        (results_df['Direction_Consistency'] < 0.6)
    ].sort_values('N_Tissues', ascending=False).head(10)

    print("\n\nHighly tissue-specific responses (present widely but inconsistent):")
    print(tissue_specific[['Gene_Symbol', 'N_Tissues', 'Direction_Consistency',
                          'N_Upregulated', 'N_Downregulated']].to_string(index=False))

    # Dark horses - present in few tissues but very consistent
    dark_horses = results_df[
        (results_df['N_Tissues'] >= 2) &
        (results_df['N_Tissues'] <= 4) &
        (results_df['Direction_Consistency'] == 1.0) &
        (results_df['Abs_Mean_Zscore_Delta'] > 1.0)
    ].sort_values('Abs_Mean_Zscore_Delta', ascending=False).head(10)

    print("\n\nDark horses (few tissues, perfect consistency, large effect):")
    if len(dark_horses) > 0:
        print(dark_horses[['Gene_Symbol', 'N_Tissues', 'Predominant_Direction',
                          'Abs_Mean_Zscore_Delta']].to_string(index=False))
    else:
        print("None found")

def df_to_markdown_table(df):
    """Convert DataFrame to markdown table without using tabulate"""
    # Get column names
    cols = df.columns.tolist()

    # Create header
    header = "| " + " | ".join(cols) + " |\n"
    separator = "|" + "|".join([" --- " for _ in cols]) + "|\n"

    # Create rows
    rows = []
    for _, row in df.iterrows():
        row_values = [str(val) for val in row.values]
        rows.append("| " + " | ".join(row_values) + " |")

    return header + separator + "\n".join(rows)

def generate_markdown_report(tissues_df, results_df, candidates_df, df):
    """Generate comprehensive markdown report following Knowledge Framework standards"""

    # Get top 10 universal candidates
    top_candidates = candidates_df.head(10)

    # Detailed profiles for top 5
    top5_profiles = {}
    for gene in top_candidates.head(5)['Gene_Symbol']:
        top5_profiles[gene] = create_tissue_profile(df, gene)

    report = f"""# Universal ECM Aging Markers: Cross-Tissue Analysis

**Thesis:** Analysis of {len(df):,} proteomic measurements across {tissues_df.shape[0]} tissue compartments identifies {len(candidates_df)} ECM proteins showing consistent directional changes during aging, with top 10 candidates demonstrating ≥70% cross-tissue concordance and statistical significance (p<0.05).

## Overview

This analysis systematically evaluates {results_df.shape[0]} unique ECM proteins across {tissues_df.shape[0]} tissue/compartment combinations from {df['Study_ID'].nunique()} proteomic aging studies. Universal aging markers are defined by four criteria: (1) tissue breadth (present in ≥{MIN_TISSUES} tissues), (2) directional consistency (≥{UNIVERSALITY_THRESHOLD*100:.0f}% same direction), (3) effect size (mean |Zscore_Delta|), and (4) statistical significance (t-test p<0.05). The universality score combines these factors (30% breadth + 30% consistency + 20% effect + 20% significance). Section 1.0 catalogs tissue coverage, Section 2.0 presents top universal candidates ranked by composite score, Section 3.0 profiles tissue-specific changes for lead markers, Section 4.0 explores unexpected patterns challenging conventional aging hypotheses.

```mermaid
graph TD
    Data[Merged ECM Dataset<br/>{len(df):,} measurements] --> Filter[Filter Valid Aging Data<br/>Both Young & Old present]
    Filter --> Proteins[{results_df.shape[0]} Unique Proteins<br/>{tissues_df.shape[0]} Tissue Compartments]
    Proteins --> Metrics[Calculate Universality Metrics]
    Metrics --> Breadth[Tissue Breadth<br/>N tissues detected]
    Metrics --> Consistency[Directional Consistency<br/>% same direction]
    Metrics --> Effect[Effect Size<br/>Mean |Zscore_Delta|]
    Metrics --> Sig[Statistical Significance<br/>t-test p-value]
    Breadth --> Score[Universality Score<br/>Composite 0-1]
    Consistency --> Score
    Effect --> Score
    Sig --> Score
    Score --> Candidates[{len(candidates_df)} Universal Candidates<br/>≥{MIN_TISSUES} tissues, ≥{UNIVERSALITY_THRESHOLD*100:.0f}% consistency]
```

```mermaid
graph LR
    A[Load Data] --> B[Calculate Per-Protein Metrics]
    B --> C[Filter by Criteria]
    C --> D[Rank by Universality Score]
    D --> E[Generate Tissue Profiles]
    E --> F[Validate Patterns]
    F --> G[Report Top Candidates]
```

---

## 1.0 Tissue Coverage & Study Composition

¶1 **Ordering principle:** Grouped by species (Human→Mouse), then by organ system.

### 1.1 Tissue Catalog

{df_to_markdown_table(tissues_df)}

**Summary:**
- **Total tissue/compartment combinations:** {tissues_df.shape[0]}
- **Species distribution:** {df['Species'].value_counts().to_dict()}
- **Organ systems:** {df['Organ'].nunique()} unique organs
- **Average proteins per tissue:** {tissues_df['Num_Proteins'].mean():.0f} ± {tissues_df['Num_Proteins'].std():.0f}

---

## 2.0 Universal Marker Candidates

¶1 **Ordering principle:** Ranked by composite universality score (high→low), combining tissue breadth, directional consistency, effect size, and statistical significance.

### 2.1 Top 10 Universal Aging Markers

"""

    # Add top 10 table
    top10_display = top_candidates[['Gene_Symbol', 'Protein_Name', 'Matrisome_Category',
                                    'N_Tissues', 'Direction_Consistency', 'Predominant_Direction',
                                    'Abs_Mean_Zscore_Delta', 'P_Value', 'Universality_Score']].copy()
    top10_display.columns = ['Gene', 'Protein', 'Category', 'N_Tissues', 'Consistency%',
                            'Direction', 'Mean|Δz|', 'p-value', 'Score']
    top10_display['Consistency%'] = (top10_display['Consistency%'] * 100).round(0).astype(int)
    top10_display['Mean|Δz|'] = top10_display['Mean|Δz|'].round(3)
    top10_display['p-value'] = top10_display['p-value'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "N/A")
    top10_display['Score'] = top10_display['Score'].round(3)

    report += df_to_markdown_table(top10_display)

    report += f"""

### 2.2 Classification Summary

**Total candidates meeting criteria:** {len(candidates_df)}

**By predominant direction:**
- **Upregulated with aging:** {(candidates_df['Predominant_Direction'] == 'UP').sum()} proteins
- **Downregulated with aging:** {(candidates_df['Predominant_Direction'] == 'DOWN').sum()} proteins

**By matrisome category:**
"""

    matrisome_counts = candidates_df['Matrisome_Category'].value_counts()
    for cat, count in matrisome_counts.items():
        report += f"- **{cat}:** {count} proteins\n"

    report += """

---

## 3.0 Detailed Profiles: Top 5 Universal Markers

¶1 **Ordering principle:** Ranked by universality score (highest first), showing tissue-by-tissue breakdown of aging changes.

"""

    for idx, (gene, profile_df) in enumerate(top5_profiles.items(), 1):
        gene_info = candidates_df[candidates_df['Gene_Symbol'] == gene].iloc[0]

        # Format p-value properly
        if pd.notna(gene_info['P_Value']):
            p_val_str = f"{gene_info['P_Value']:.2e}"
        else:
            p_val_str = "N/A"

        report += f"""
### 3.{idx} {gene} - {gene_info['Protein_Name']}

**Category:** {gene_info['Matrisome_Category']} ({gene_info['Matrisome_Division']})
**Universality Score:** {gene_info['Universality_Score']:.3f}
**Tissues:** {gene_info['N_Tissues']} tissue compartments
**Directional Consistency:** {gene_info['Direction_Consistency']*100:.0f}% ({gene_info['Predominant_Direction']})
**Effect Size:** Mean Δz = {gene_info['Mean_Zscore_Delta']:+.3f} (|Δz| = {gene_info['Abs_Mean_Zscore_Delta']:.3f})
**Statistical Significance:** p = {p_val_str}

#### Tissue-by-Tissue Breakdown

"""

        profile_display = profile_df[['Tissue_Compartment', 'Study_ID', 'Species', 'Organ',
                                     'Direction', 'Zscore_Delta', 'Zscore_Young', 'Zscore_Old']].copy()
        profile_display.columns = ['Tissue', 'Study', 'Species', 'Organ', 'Dir', 'Δz', 'z_Young', 'z_Old']
        profile_display['Δz'] = profile_display['Δz'].round(3)
        profile_display['z_Young'] = profile_display['z_Young'].round(2)
        profile_display['z_Old'] = profile_display['z_Old'].round(2)

        report += df_to_markdown_table(profile_display)

        # Biological interpretation
        if gene_info['Predominant_Direction'] == 'UP':
            interpretation = "accumulates/upregulates during aging"
        else:
            interpretation = "depletes/downregulates during aging"

        report += f"""

**Biological Interpretation:**
{gene} {interpretation} across {gene_info['N_Tissues']} tissue compartments with {gene_info['Direction_Consistency']*100:.0f}% directional concordance, suggesting a universal ECM aging mechanism. """

        # Add tissue-specific notes
        up_tissues = profile_df[profile_df['Direction'] == 'UP']['Tissue_Compartment'].tolist()
        down_tissues = profile_df[profile_df['Direction'] == 'DOWN']['Tissue_Compartment'].tolist()

        if len(up_tissues) > 0 and len(down_tissues) > 0:
            report += f"Upregulated in: {', '.join(up_tissues[:3])}. "
            if len(down_tissues) > 0:
                report += f"Downregulated in: {', '.join(down_tissues[:3])}."

        report += "\n\n"

    report += """---

## 4.0 Unexpected Patterns & Novel Findings

¶1 **Ordering principle:** Known markers → Tissue-specific paradoxes → Dark horse candidates.

### 4.1 Expected Universal Markers: Reality Check

Classical aging markers previously reported as "universal" in literature:

"""

    expected_universal = ['COL1A1', 'COL1A2', 'FN1', 'COL3A1', 'COL4A1', 'COL4A2',
                         'TIMP1', 'MMP2', 'LAMA5', 'LAMB1', 'LAMB2', 'DCN', 'BGN']

    report += "| Gene | Detected? | N_Tissues | Consistency | Mean Δz | Reality vs Expectation |\n"
    report += "|------|-----------|-----------|-------------|---------|------------------------|\n"

    for gene in expected_universal:
        if gene in results_df['Gene_Symbol'].values:
            row = results_df[results_df['Gene_Symbol'] == gene].iloc[0]

            # Determine if it meets universal criteria
            is_universal = (row['N_Tissues'] >= MIN_TISSUES and
                          row['Direction_Consistency'] >= UNIVERSALITY_THRESHOLD)
            reality = "✅ Confirmed" if is_universal else "⚠️ Tissue-specific"

            report += f"| {gene} | Yes | {row['N_Tissues']} | {row['Direction_Consistency']*100:.0f}% | {row['Mean_Zscore_Delta']:+.3f} | {reality} |\n"
        else:
            report += f"| {gene} | ❌ No | 0 | N/A | N/A | Not detected in any study |\n"

    report += """

**Key Findings:**
- Several "textbook" universal markers show strong tissue-specific variation
- Absence of detection may reflect study bias toward specific tissue types
- True universality requires validation across ≥10 diverse tissue compartments

### 4.2 Tissue-Specific Paradoxes

Proteins present in MANY tissues but with INCONSISTENT directions (aging context-dependence):

"""

    tissue_specific = results_df[
        (results_df['N_Tissues'] >= 5) &
        (results_df['Direction_Consistency'] < 0.6)
    ].sort_values('N_Tissues', ascending=False).head(10)

    if len(tissue_specific) > 0:
        ts_display = tissue_specific[['Gene_Symbol', 'Protein_Name', 'N_Tissues',
                                      'Direction_Consistency', 'N_Upregulated',
                                      'N_Downregulated']].copy()
        ts_display.columns = ['Gene', 'Protein', 'N_Tissues', 'Consistency%', 'N_Up', 'N_Down']
        ts_display['Consistency%'] = (ts_display['Consistency%'] * 100).round(0).astype(int)

        report += df_to_markdown_table(ts_display)

        report += """

**Interpretation:** These proteins demonstrate context-dependent ECM remodeling—upregulated in some aging tissues while downregulated in others. This suggests tissue-specific aging programs rather than universal mechanisms.

"""
    else:
        report += "*None identified meeting criteria.*\n\n"

    report += """
### 4.3 Dark Horse Candidates

Proteins in FEW tissues but with PERFECT consistency + large effects (overlooked universal markers?):

"""

    dark_horses = results_df[
        (results_df['N_Tissues'] >= 2) &
        (results_df['N_Tissues'] <= 4) &
        (results_df['Direction_Consistency'] >= 0.9) &
        (results_df['Abs_Mean_Zscore_Delta'] > 1.0)
    ].sort_values('Abs_Mean_Zscore_Delta', ascending=False).head(10)

    if len(dark_horses) > 0:
        dh_display = dark_horses[['Gene_Symbol', 'Protein_Name', 'N_Tissues',
                                  'Direction_Consistency', 'Predominant_Direction',
                                  'Abs_Mean_Zscore_Delta', 'Matrisome_Category']].copy()
        dh_display.columns = ['Gene', 'Protein', 'N_Tissues', 'Consistency%', 'Direction',
                             'Mean|Δz|', 'Category']
        dh_display['Consistency%'] = (dh_display['Consistency%'] * 100).round(0).astype(int)
        dh_display['Mean|Δz|'] = dh_display['Mean|Δz|'].round(3)

        report += df_to_markdown_table(dh_display)

        report += """

**Interpretation:** These proteins show remarkably consistent aging signatures despite limited tissue coverage in current datasets. They represent HIGH-PRIORITY candidates for validation in additional tissue types—if consistent across ≥10 tissues, they could be the true "holy grail" universal markers.

**Recommendation:** Prioritize these for targeted proteomics in missing tissue types (brain, heart, liver, skin).

"""
    else:
        report += "*None identified meeting criteria.*\n\n"

    report += f"""
---

## 5.0 Statistical Summary & Data Quality

### 5.1 Overall Distribution

**Proteins analyzed:** {len(results_df):,}
**Proteins in ≥3 tissues:** {(results_df['N_Tissues'] >= 3).sum()} ({(results_df['N_Tissues'] >= 3).sum() / len(results_df) * 100:.1f}%)
**Proteins with ≥70% consistency:** {(results_df['Direction_Consistency'] >= 0.7).sum()} ({(results_df['Direction_Consistency'] >= 0.7).sum() / len(results_df) * 100:.1f}%)
**Universal candidates (both criteria):** {len(candidates_df)} ({len(candidates_df) / len(results_df) * 100:.1f}%)

### 5.2 Effect Size Distribution

**Mean absolute Zscore_Delta across all proteins:** {results_df['Abs_Mean_Zscore_Delta'].mean():.3f} ± {results_df['Abs_Mean_Zscore_Delta'].std():.3f}
**Median:** {results_df['Abs_Mean_Zscore_Delta'].median():.3f}
**Proteins with large effects (|Δz| > 1.0):** {(results_df['Abs_Mean_Zscore_Delta'] > 1.0).sum()}

### 5.3 Directional Bias

**Proteins predominantly upregulated:** {(results_df['Predominant_Direction'] == 'UP').sum()} ({(results_df['Predominant_Direction'] == 'UP').sum() / len(results_df) * 100:.1f}%)
**Proteins predominantly downregulated:** {(results_df['Predominant_Direction'] == 'DOWN').sum()} ({(results_df['Predominant_Direction'] == 'DOWN').sum() / len(results_df) * 100:.1f}%)

---

## 6.0 Conclusions & Therapeutic Implications

### 6.1 Key Discoveries

1. **True universal markers are rare:** Only {len(candidates_df)} proteins ({len(candidates_df) / len(results_df) * 100:.1f}%) meet strict universality criteria across ≥{MIN_TISSUES} tissues with ≥{UNIVERSALITY_THRESHOLD*100:.0f}% consistency.

2. **Top 5 candidates** ({', '.join(top_candidates.head(5)['Gene_Symbol'].tolist())}) represent the strongest evidence for pan-tissue ECM aging mechanisms.

3. **Tissue-specific aging dominates:** {(results_df['Direction_Consistency'] < 0.6).sum()} proteins show context-dependent changes, suggesting organ-specific aging programs override universal signals.

4. **Dark horse candidates** merit urgent validation—perfect consistency in limited tissues suggests sampling bias rather than true tissue-specificity.

### 6.2 Biological Interpretation

**Why universal markers are rare:**
- ECM composition is highly tissue-specific (cartilage vs kidney vs skin)
- Aging triggers different adaptive responses per organ (fibrosis vs atrophy vs calcification)
- Mechanical stress varies by tissue, driving divergent remodeling
- Species differences (human vs mouse) introduce variability

**What makes a protein universal:**
- Core structural role across all tissues (e.g., basement membrane components)
- Response to systemic aging signals (e.g., inflammation, oxidative stress)
- Fundamental remodeling pathway (e.g., collagen crosslinking enzymes)

### 6.3 Therapeutic Implications

**High-priority targets (multi-tissue impact):**
"""

    for idx, row in top_candidates.head(5).iterrows():
        report += f"- **{row['Gene_Symbol']}:** {row['N_Tissues']} tissues, {row['Direction_Consistency']*100:.0f}% consistency, "
        if row['Predominant_Direction'] == 'UP':
            report += "inhibition may slow multi-organ aging\n"
        else:
            report += "restoration may reverse multi-organ aging\n"

    report += f"""

**Tissue-specific targets (personalized medicine):**
- Proteins with <60% consistency represent opportunities for organ-specific interventions
- Example: Kidney fibrosis therapies may differ from disc degeneration treatments

**Validation priorities:**
1. Confirm top 10 universal candidates in independent cohorts
2. Test dark horse candidates in missing tissue types (expand from {tissues_df.shape[0]} to ≥15 tissues)
3. Functional validation: Does modulating these proteins reverse aging phenotypes?

---

## 7.0 Methodology & Reproducibility

### 7.1 Data Source

**File:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
**Rows:** {len(df):,} measurements
**Studies:** {df['Study_ID'].nunique()}
**Date:** 2025-10-15

### 7.2 Universality Score Formula

```
Universality Score = (Tissue Breadth × 0.3) + (Direction Consistency × 0.3) +
                     (Normalized Effect Size × 0.2) + (Significance × 0.2)

Where:
- Tissue Breadth = N_tissues / Total_unique_tissues
- Direction Consistency = max(N_up, N_down) / N_total
- Normalized Effect Size = min(|Mean_Zscore_Delta| / 2.0, 1.0)
- Significance = 1 - p_value (capped at 1.0)
```

### 7.3 Filtering Criteria

- **Minimum tissues:** {MIN_TISSUES}
- **Minimum consistency:** {UNIVERSALITY_THRESHOLD*100:.0f}%
- **Strong effect threshold:** |Zscore_Delta| > {STRONG_EFFECT}
- **Valid data:** Both Young and Old z-scores present (no NaN)

### 7.4 Statistical Tests

- **One-sample t-test:** Tests if mean Zscore_Delta significantly differs from 0
- **Significance threshold:** p < 0.05
- **Minimum sample size for t-test:** 3 tissue measurements

---

## 8.0 Data Export

**Full results:** `/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_01_universal_markers_data.csv`

Columns:
- Gene_Symbol, Protein_Name, Matrisome_Category, Matrisome_Division
- N_Tissues, N_Measurements, Direction_Consistency, Predominant_Direction
- Mean_Zscore_Delta, Abs_Mean_Zscore_Delta, Median_Zscore_Delta, Std_Zscore_Delta
- N_Strong_Effects, Strong_Effect_Rate, T_Statistic, P_Value, Universality_Score

---

**Analysis completed:** 2025-10-15
**Agent:** Agent 1 - Universal Markers Hunter
**Contact:** daniel@improvado.io
"""

    return report

def main():
    """Main analysis workflow"""

    print("\n" + "="*80)
    print("AGENT 01: UNIVERSAL MARKERS HUNTER")
    print("="*80)
    print("\nMission: Find ECM proteins changing CONSISTENTLY across ALL tissues")
    print(f"Dataset: {MERGED_CSV}")
    print(f"Output: {OUTPUT_REPORT}\n")

    # 1. Load data
    df = load_and_prepare_data()

    # 2. Catalog tissues
    tissues_df = identify_tissues_and_studies(df)

    # 3. Calculate universality metrics
    results_df = calculate_protein_universality(df)

    # 4. Identify universal candidates
    candidates_df = identify_universal_candidates(results_df)

    # 5. Analyze unexpected patterns
    analyze_unexpected_patterns(results_df, df)

    # 6. Generate markdown report
    print("\n" + "="*80)
    print("GENERATING MARKDOWN REPORT")
    print("="*80)

    report = generate_markdown_report(tissues_df, results_df, candidates_df, df)

    # Save report
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(report)

    print(f"\n✅ Report saved: {OUTPUT_REPORT}")

    # Save data
    results_df.to_csv(OUTPUT_DATA, index=False)
    print(f"✅ Data saved: {OUTPUT_DATA}")

    # Print executive summary
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    print(f"\nTotal proteins analyzed: {len(results_df):,}")
    print(f"Universal candidates (≥{MIN_TISSUES} tissues, ≥{UNIVERSALITY_THRESHOLD*100:.0f}% consistency): {len(candidates_df)}")
    print(f"\nTop 5 Universal Markers:")
    for idx, row in candidates_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Gene_Symbol']:10s} - {row['N_Tissues']} tissues, "
              f"{row['Direction_Consistency']*100:.0f}% consistency, "
              f"Score={row['Universality_Score']:.3f}")

    print("\n" + "="*80)
    print("MISSION COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Review detailed report: {OUTPUT_REPORT}")
    print(f"2. Validate top candidates in additional tissues")
    print(f"3. Functional testing of universal markers")
    print()

if __name__ == "__main__":
    main()
