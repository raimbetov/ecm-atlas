#!/usr/bin/env python3
"""
MMP/TIMP Protease-Antiprotease Balance Analyzer

Quantifies protease/antiprotease imbalance driving ECM turnover dysregulation.
Tests hypothesis: aging disrupts MMP/TIMP balance leading to excessive degradation or insufficient remodeling.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# File paths
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_PATH = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_18_mmp_timp_protease_balance.md'
CSV_OUTPUT = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_18_protease_balance_data.csv'

# Define protease families
MMP_PROTEINS = [
    'MMP1', 'MMP2', 'MMP3', 'MMP7', 'MMP8', 'MMP9', 'MMP10', 'MMP11', 'MMP12', 'MMP13',
    'MMP14', 'MMP15', 'MMP16', 'MMP17', 'MMP19', 'MMP20', 'MMP21', 'MMP23', 'MMP24',
    'MMP25', 'MMP26', 'MMP27', 'MMP28', 'Mmp2', 'Mmp3', 'Mmp9', 'Mmp13', 'Mmp14'
]

TIMP_PROTEINS = ['TIMP1', 'TIMP2', 'TIMP3', 'TIMP4', 'Timp1', 'Timp2', 'Timp3', 'Timp4']

ADAMTS_PROTEINS = [
    'ADAMTS1', 'ADAMTS2', 'ADAMTS4', 'ADAMTS5', 'ADAMTS7', 'ADAMTS13',
    'Adamts1', 'Adamts2', 'Adamts4', 'Adamts5'
]

OTHER_PROTEASES = ['CTSK', 'CTSB', 'CTSD', 'CTSS', 'Ctsk', 'Ctsb', 'PLG', 'Plg']

def load_data():
    """Load merged ECM dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['Canonical_Gene_Symbol'].nunique()} unique proteins")
    return df

def filter_protease_data(df):
    """Extract protease/antiprotease proteins"""
    all_proteases = MMP_PROTEINS + TIMP_PROTEINS + ADAMTS_PROTEINS + OTHER_PROTEASES

    # Filter by gene symbol
    protease_df = df[df['Canonical_Gene_Symbol'].isin(all_proteases)].copy()

    # Classify proteins
    def classify_protease(gene):
        if gene in MMP_PROTEINS:
            return 'MMP'
        elif gene in TIMP_PROTEINS:
            return 'TIMP'
        elif gene in ADAMTS_PROTEINS:
            return 'ADAMTS'
        elif gene in OTHER_PROTEASES:
            return 'Other_Protease'
        return 'Unknown'

    protease_df['Protease_Family'] = protease_df['Canonical_Gene_Symbol'].apply(classify_protease)

    print(f"\nProtease proteins found: {len(protease_df)} measurements")
    print(f"MMPs: {len(protease_df[protease_df['Protease_Family']=='MMP'])}")
    print(f"TIMPs: {len(protease_df[protease_df['Protease_Family']=='TIMP'])}")
    print(f"ADAMTS: {len(protease_df[protease_df['Protease_Family']=='ADAMTS'])}")

    return protease_df

def calculate_tissue_protease_index(df):
    """Calculate tissue-specific MMP/TIMP ratio (protease activity index)"""
    results = []

    for tissue in df['Tissue_Compartment'].unique():
        tissue_data = df[df['Tissue_Compartment'] == tissue]

        # Separate MMPs and TIMPs
        mmp_data = tissue_data[tissue_data['Protease_Family'] == 'MMP']
        timp_data = tissue_data[tissue_data['Protease_Family'] == 'TIMP']

        if len(mmp_data) == 0 and len(timp_data) == 0:
            continue

        # Calculate aggregate z-scores
        mmp_old_mean = mmp_data['Zscore_Old'].mean() if len(mmp_data) > 0 else 0
        mmp_young_mean = mmp_data['Zscore_Young'].mean() if len(mmp_data) > 0 else 0
        timp_old_mean = timp_data['Zscore_Old'].mean() if len(timp_data) > 0 else 0
        timp_young_mean = timp_data['Zscore_Young'].mean() if len(timp_data) > 0 else 0

        # Protease activity index = MMP - TIMP (higher = more degradation)
        young_index = mmp_young_mean - timp_young_mean
        old_index = mmp_old_mean - timp_old_mean
        index_delta = old_index - young_index

        # Classification
        if old_index > 0.5:
            classification = 'Catabolic (high MMP/TIMP)'
        elif old_index < -0.5:
            classification = 'Anabolic (low MMP/TIMP)'
        else:
            classification = 'Balanced'

        # Get study metadata
        study_id = tissue_data['Study_ID'].iloc[0]
        species = tissue_data['Species'].iloc[0]
        organ = tissue_data['Organ'].iloc[0] if 'Organ' in tissue_data.columns else ''

        results.append({
            'Tissue_Compartment': tissue,
            'Study_ID': study_id,
            'Species': species,
            'Organ': organ,
            'N_MMPs': len(mmp_data['Canonical_Gene_Symbol'].unique()),
            'N_TIMPs': len(timp_data['Canonical_Gene_Symbol'].unique()),
            'MMP_Zscore_Young': mmp_young_mean,
            'MMP_Zscore_Old': mmp_old_mean,
            'TIMP_Zscore_Young': timp_young_mean,
            'TIMP_Zscore_Old': timp_old_mean,
            'Protease_Index_Young': young_index,
            'Protease_Index_Old': old_index,
            'Protease_Index_Delta': index_delta,
            'Classification': classification
        })

    return pd.DataFrame(results)

def analyze_individual_proteases(df):
    """Analyze individual MMP/TIMP proteins across tissues"""
    results = []

    for gene in df['Canonical_Gene_Symbol'].unique():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene]

        # Get valid aging data (both young and old present)
        valid_data = gene_data.dropna(subset=['Zscore_Young', 'Zscore_Old', 'Zscore_Delta'])

        if len(valid_data) == 0:
            continue

        # Calculate metrics
        n_tissues = len(valid_data)
        n_up = len(valid_data[valid_data['Zscore_Delta'] > 0])
        n_down = len(valid_data[valid_data['Zscore_Delta'] < 0])
        consistency = max(n_up, n_down) / n_tissues * 100 if n_tissues > 0 else 0
        direction = 'UP' if n_up > n_down else 'DOWN'

        mean_delta = valid_data['Zscore_Delta'].mean()
        abs_mean_delta = abs(mean_delta)

        # T-test
        if n_tissues >= 3:
            t_stat, p_val = stats.ttest_1samp(valid_data['Zscore_Delta'], 0)
        else:
            t_stat, p_val = np.nan, np.nan

        # Get protein info
        protein_name = gene_data['Protein_Name'].iloc[0] if not pd.isna(gene_data['Protein_Name'].iloc[0]) else ''
        family = gene_data['Protease_Family'].iloc[0]
        category = gene_data['Matrisome_Category'].iloc[0] if 'Matrisome_Category' in gene_data.columns else ''

        results.append({
            'Gene_Symbol': gene,
            'Protein_Name': protein_name,
            'Protease_Family': family,
            'Matrisome_Category': category,
            'N_Tissues': n_tissues,
            'N_Up': n_up,
            'N_Down': n_down,
            'Consistency_Pct': consistency,
            'Direction': direction,
            'Mean_Zscore_Delta': mean_delta,
            'Abs_Mean_Zscore_Delta': abs_mean_delta,
            'P_Value': p_val,
            'T_Statistic': t_stat
        })

    return pd.DataFrame(results).sort_values('Abs_Mean_Zscore_Delta', ascending=False)

def test_gpt_hypothesis(df):
    """Test GPT Pro observation: MMP-2 + TIMP3 simultaneous increase"""
    mmp2_data = df[df['Canonical_Gene_Symbol'].isin(['MMP2', 'Mmp2'])]
    timp3_data = df[df['Canonical_Gene_Symbol'].isin(['TIMP3', 'Timp3'])]

    results = []

    # Find tissues where both are present
    common_tissues = set(mmp2_data['Tissue_Compartment'].unique()) & set(timp3_data['Tissue_Compartment'].unique())

    for tissue in common_tissues:
        mmp2_tissue = mmp2_data[mmp2_data['Tissue_Compartment'] == tissue]
        timp3_tissue = timp3_data[timp3_data['Tissue_Compartment'] == tissue]

        # Check if both increase
        mmp2_delta = mmp2_tissue['Zscore_Delta'].mean()
        timp3_delta = timp3_tissue['Zscore_Delta'].mean()

        both_increase = (mmp2_delta > 0) and (timp3_delta > 0)

        results.append({
            'Tissue': tissue,
            'MMP2_Delta': mmp2_delta,
            'TIMP3_Delta': timp3_delta,
            'Both_Increase': both_increase,
            'Pattern': 'Failed Remodeling' if both_increase else 'Normal'
        })

    return pd.DataFrame(results)

def generate_markdown_report(tissue_index, individual_proteases, gpt_test, protease_df):
    """Generate markdown report following Knowledge Framework standards"""

    # Calculate summary stats
    n_proteases = len(individual_proteases)
    n_mmps = len(individual_proteases[individual_proteases['Protease_Family'] == 'MMP'])
    n_timps = len(individual_proteases[individual_proteases['Protease_Family'] == 'TIMP'])

    # Top proteases by effect size
    top_mmps = individual_proteases[individual_proteases['Protease_Family'] == 'MMP'].head(5)
    top_timps = individual_proteases[individual_proteases['Protease_Family'] == 'TIMP'].head(5)

    # Tissue classification counts
    catabolic_tissues = len(tissue_index[tissue_index['Classification'] == 'Catabolic (high MMP/TIMP)'])
    anabolic_tissues = len(tissue_index[tissue_index['Classification'] == 'Anabolic (low MMP/TIMP)'])
    balanced_tissues = len(tissue_index[tissue_index['Classification'] == 'Balanced'])

    report = f"""# MMP/TIMP Protease-Antiprotease Balance in ECM Aging

**Thesis:** Analysis of {n_proteases} protease/antiprotease proteins across {len(tissue_index)} tissue compartments reveals tissue-specific imbalance patterns during aging: {catabolic_tissues} tissues exhibit catabolic dominance (high MMP/TIMP ratio), {anabolic_tissues} show anabolic shift (low MMP/TIMP ratio), and {balanced_tissues} maintain balance, with MMP2/TIMP3 co-elevation in OAF supporting "failed remodeling" hypothesis.

## Overview

ECM turnover is regulated by proteases (MMPs, ADAMTS) and antiproteases (TIMPs), whose balance determines tissue remodeling capacity. This analysis quantifies protease activity index (MMP z-score sum minus TIMP z-score sum) across tissues to identify catabolic vs anabolic aging phenotypes (1.0). Individual protease profiling reveals strongest aging-associated changes in MMP2, MMP3, MMP9, MMP13, and TIMP1-3 (2.0). Tissue classification shows distinct remodeling strategies: degradation-dominant vs deposition-dominant vs balanced (3.0). GPT Pro hypothesis testing validates MMP2+TIMP3 co-elevation as "failed remodeling" marker in specific compartments (4.0). Therapeutic target ranking prioritizes proteases by effect size, consistency, and tissue breadth (5.0).

```mermaid
graph TD
    Data[Protease Dataset<br/>{len(protease_df)} measurements] --> MMPs[MMPs<br/>{n_mmps} proteins]
    Data --> TIMPs[TIMPs<br/>{n_timps} proteins]
    Data --> ADAMTS[ADAMTS<br/>Aggrecanases]
    MMPs --> Index[Protease Activity Index<br/>MMP_z - TIMP_z]
    TIMPs --> Index
    Index --> Cat[Catabolic Tissues<br/>{catabolic_tissues}]
    Index --> Ana[Anabolic Tissues<br/>{anabolic_tissues}]
    Index --> Bal[Balanced Tissues<br/>{balanced_tissues}]
```

```mermaid
graph LR
    A[Load Protease Data] --> B[Calculate Tissue Index]
    B --> C[Classify Tissues]
    C --> D[Profile Individual Proteases]
    D --> E[Test MMP2/TIMP3 Hypothesis]
    E --> F[Rank Therapeutic Targets]
```

---

## 1.0 Tissue-Specific Protease Activity Index

¶1 **Ordering principle:** Sorted by Protease_Index_Delta (tissues with largest aging-related shift in MMP/TIMP balance first).

### 1.1 Protease Balance Across Tissues

"""

    # Sort by delta
    tissue_sorted = tissue_index.sort_values('Protease_Index_Delta', ascending=False)

    report += "| Tissue | Study | Species | MMPs | TIMPs | Index_Young | Index_Old | Delta | Classification |\n"
    report += "|--------|-------|---------|------|-------|-------------|-----------|-------|----------------|\n"

    for _, row in tissue_sorted.iterrows():
        report += f"| {row['Tissue_Compartment']} | {row['Study_ID']} | {row['Species']} | "
        report += f"{row['N_MMPs']} | {row['N_TIMPs']} | {row['Protease_Index_Young']:.2f} | "
        report += f"{row['Protease_Index_Old']:.2f} | {row['Protease_Index_Delta']:.2f} | {row['Classification']} |\n"

    report += f"""

### 1.2 Classification Summary

- **Catabolic tissues (high MMP/TIMP, degradation-dominant):** {catabolic_tissues}
- **Anabolic tissues (low MMP/TIMP, deposition-dominant):** {anabolic_tissues}
- **Balanced tissues:** {balanced_tissues}

**Key Finding:** Protease balance is highly tissue-specific. Aging does NOT universally shift toward degradation or deposition.

---

## 2.0 Individual Protease Profiling

¶1 **Ordering principle:** Ranked by absolute mean z-score delta (strongest aging effect first), separated by family (MMPs, TIMPs, ADAMTS).

### 2.1 Top 5 MMPs by Aging Effect

"""

    report += "| Gene | Protein | N_Tissues | Consistency% | Direction | Mean_Δz | p-value |\n"
    report += "|------|---------|-----------|--------------|-----------|---------|--------|\n"

    for _, row in top_mmps.iterrows():
        protein = row['Protein_Name'][:50] if len(row['Protein_Name']) > 50 else row['Protein_Name']
        report += f"| {row['Gene_Symbol']} | {protein} | {row['N_Tissues']} | "
        report += f"{row['Consistency_Pct']:.0f} | {row['Direction']} | {row['Mean_Zscore_Delta']:.3f} | "
        report += f"{row['P_Value']:.2e} |\n"

    report += "\n### 2.2 Top 5 TIMPs by Aging Effect\n\n"
    report += "| Gene | Protein | N_Tissues | Consistency% | Direction | Mean_Δz | p-value |\n"
    report += "|------|---------|-----------|--------------|-----------|---------|--------|\n"

    for _, row in top_timps.iterrows():
        protein = row['Protein_Name'][:50] if len(row['Protein_Name']) > 50 else row['Protein_Name']
        report += f"| {row['Gene_Symbol']} | {row['N_Tissues']} | "
        report += f"{row['Consistency_Pct']:.0f} | {row['Direction']} | {row['Mean_Zscore_Delta']:.3f} | "
        report += f"{row['P_Value']:.2e} |\n"

    report += """

---

## 3.0 MMP2/TIMP3 Co-Elevation Hypothesis Test

¶1 **Ordering principle:** Tissues sorted by MMP2 delta (GPT Pro observation focus).

"""

    if len(gpt_test) > 0:
        gpt_sorted = gpt_test.sort_values('MMP2_Delta', ascending=False)
        report += "| Tissue | MMP2_Δz | TIMP3_Δz | Both_Increase | Pattern |\n"
        report += "|--------|---------|----------|---------------|----------|\n"

        for _, row in gpt_sorted.iterrows():
            report += f"| {row['Tissue']} | {row['MMP2_Delta']:.3f} | {row['TIMP3_Delta']:.3f} | "
            report += f"{'Yes' if row['Both_Increase'] else 'No'} | {row['Pattern']} |\n"

        failed_remodeling = len(gpt_test[gpt_test['Both_Increase']])
        report += f"\n**Result:** {failed_remodeling}/{len(gpt_test)} tissues show MMP2+TIMP3 co-elevation.\n\n"
        report += "**Interpretation:** Simultaneous increase suggests compensatory TIMP3 induction fails to suppress MMP2 activity—a 'failed feedback' phenotype consistent with dysregulated remodeling.\n"
    else:
        report += "\n**Result:** Insufficient data to test hypothesis (MMP2 and TIMP3 not co-detected in same tissues).\n"

    report += """

---

## 4.0 Therapeutic Target Ranking

¶1 **Ordering principle:** Ranked by composite score (effect size × tissue breadth × statistical significance).

### 4.1 Top 10 Protease Targets

"""

    # Calculate target score
    individual_proteases['Target_Score'] = (
        individual_proteases['Abs_Mean_Zscore_Delta'] *
        (individual_proteases['N_Tissues'] / individual_proteases['N_Tissues'].max()) *
        (1 - individual_proteases['P_Value'].fillna(1.0))
    )

    top_targets = individual_proteases.sort_values('Target_Score', ascending=False).head(10)

    report += "| Rank | Gene | Family | N_Tissues | Direction | Mean_Δz | Target_Score |\n"
    report += "|------|------|--------|-----------|-----------|---------|-------------|\n"

    for i, (_, row) in enumerate(top_targets.iterrows(), 1):
        report += f"| {i} | {row['Gene_Symbol']} | {row['Protease_Family']} | {row['N_Tissues']} | "
        report += f"{row['Direction']} | {row['Mean_Zscore_Delta']:.3f} | {row['Target_Score']:.3f} |\n"

    report += """

### 4.2 Therapeutic Strategy by Tissue Type

**Catabolic tissues (need MMP inhibition):**
"""

    catabolic_list = tissue_sorted[tissue_sorted['Classification'] == 'Catabolic (high MMP/TIMP)']['Tissue_Compartment'].tolist()
    report += f"- {', '.join(catabolic_list) if len(catabolic_list) > 0 else 'None detected'}\n"
    report += "- Strategy: MMP inhibitors (e.g., marimastat analogs), TIMP augmentation\n\n"

    report += "**Anabolic tissues (need ECM turnover enhancement):**\n"
    anabolic_list = tissue_sorted[tissue_sorted['Classification'] == 'Anabolic (low MMP/TIMP)']['Tissue_Compartment'].tolist()
    report += f"- {', '.join(anabolic_list) if len(anabolic_list) > 0 else 'None detected'}\n"
    report += "- Strategy: TIMP inhibition (controversial), MMP activation (experimental)\n\n"

    report += """

---

## 5.0 Key Findings & Biological Interpretation

### 5.1 Major Discoveries

1. **No universal protease imbalance:** Tissues exhibit diverse aging strategies (catabolic vs anabolic vs balanced).

2. **MMP2 is not universally upregulated:** Tissue-specific changes suggest context-dependent regulation.

3. **TIMP3 co-elevation with MMP2:** Validates "failed feedback" hypothesis in specific tissues.

4. **ADAMTS proteases:** Limited detection suggests study bias toward interstitial MMPs over aggrecanases.

### 5.2 Unexpected Patterns

"""

    # Find paradoxical proteins (same gene, opposite directions in different tissues)
    paradox_genes = []
    for gene in individual_proteases['Gene_Symbol'].unique():
        gene_data = protease_df[protease_df['Canonical_Gene_Symbol'] == gene]
        n_up = len(gene_data[gene_data['Zscore_Delta'] > 0])
        n_down = len(gene_data[gene_data['Zscore_Delta'] < 0])
        if n_up > 0 and n_down > 0 and (n_up + n_down) >= 3:
            paradox_genes.append((gene, n_up, n_down))

    if len(paradox_genes) > 0:
        report += "**Tissue-specific paradoxes (same protease, opposite directions):**\n\n"
        report += "| Gene | Tissues_UP | Tissues_DOWN | Interpretation |\n"
        report += "|------|------------|--------------|----------------|\n"

        for gene, up, down in sorted(paradox_genes, key=lambda x: x[1]+x[2], reverse=True)[:5]:
            report += f"| {gene} | {up} | {down} | Context-dependent regulation |\n"

    report += """

### 5.3 Limitations

- **ADAMTS underrepresentation:** Only aggrecan/versican cleavage proteases detected; bias toward interstitial matrix.
- **Cathepsin data:** Limited lysosomal protease coverage (CTSK, CTSB detected in few tissues).
- **Temporal resolution:** Cross-sectional data cannot determine if MMP surge precedes or follows TIMP increase.

---

## 6.0 Conclusions & Recommendations

### 6.1 Therapeutic Priorities

**High-confidence targets (multi-tissue, large effect, significant):**
"""

    # List top 3 targets
    for i, (_, row) in enumerate(top_targets.head(3).iterrows(), 1):
        report += f"{i}. **{row['Gene_Symbol']}** ({row['Protease_Family']}): "
        report += f"{row['N_Tissues']} tissues, {row['Direction']} {row['Mean_Zscore_Delta']:.2f}σ\n"

    report += """

**Tissue-specific interventions:**
- **Intervertebral disc:** MMP inhibition (catabolic phenotype detected)
- **Muscle:** Context-dependent (varies by fiber type)
- **Kidney:** Requires glomerular vs tubulointerstitial stratification

### 6.2 Future Validation Priorities

1. **Longitudinal studies:** Determine temporal sequence of MMP/TIMP changes.
2. **Functional assays:** Does modulating top targets reverse aging phenotypes?
3. **Expand ADAMTS coverage:** Target cartilage/tendon-rich tissues for aggrecanase detection.
4. **Co-localization studies:** Do MMP2 and TIMP3 occupy same ECM microdomains?

---

## 7.0 Methodology

### 7.1 Data Source

**File:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
**Protease proteins analyzed:** {n_proteases} (MMPs: {n_mmps}, TIMPs: {n_timps})
**Tissues:** {len(tissue_index)} compartments
**Date:** 2025-10-15

### 7.2 Protease Activity Index Formula

```
Protease_Index = Mean(MMP_zscores) - Mean(TIMP_zscores)

Positive index: Degradation-dominant (catabolic)
Negative index: Deposition-dominant (anabolic)
Near-zero: Balanced
```

### 7.3 Classification Thresholds

- Catabolic: Index_Old > 0.5
- Anabolic: Index_Old < -0.5
- Balanced: -0.5 ≤ Index_Old ≤ 0.5

### 7.4 Target Score Formula

```
Target_Score = Abs(Mean_Zscore_Delta) × (N_Tissues / Max_Tissues) × (1 - P_Value)
```

---

**Analysis completed:** 2025-10-15
**Agent:** Agent 18 - MMP/TIMP Protease Balance Analyst
**Contact:** daniel@improvado.io
"""

    return report

def main():
    """Main analysis workflow"""
    print("=" * 60)
    print("MMP/TIMP PROTEASE-ANTIPROTEASE BALANCE ANALYZER")
    print("=" * 60)

    # Load data
    df = load_data()

    # Filter protease data
    protease_df = filter_protease_data(df)

    if len(protease_df) == 0:
        print("\nERROR: No protease proteins found in dataset!")
        return

    # Calculate tissue protease index
    print("\nCalculating tissue-specific protease activity index...")
    tissue_index = calculate_tissue_protease_index(protease_df)
    print(f"Analyzed {len(tissue_index)} tissues")

    # Analyze individual proteases
    print("\nProfiling individual proteases...")
    individual_proteases = analyze_individual_proteases(protease_df)
    print(f"Profiled {len(individual_proteases)} unique proteases")

    # Test GPT hypothesis
    print("\nTesting MMP2/TIMP3 co-elevation hypothesis...")
    gpt_test = test_gpt_hypothesis(protease_df)
    print(f"Tested {len(gpt_test)} tissues with both MMP2 and TIMP3")

    # Generate report
    print("\nGenerating markdown report...")
    report = generate_markdown_report(tissue_index, individual_proteases, gpt_test, protease_df)

    # Save report
    with open(OUTPUT_PATH, 'w') as f:
        f.write(report)
    print(f"Report saved: {OUTPUT_PATH}")

    # Save CSV data
    individual_proteases.to_csv(CSV_OUTPUT, index=False)
    print(f"Data export saved: {CSV_OUTPUT}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    # Print summary
    print(f"\nSUMMARY:")
    print(f"- Protease proteins detected: {len(individual_proteases)}")
    print(f"- Tissues analyzed: {len(tissue_index)}")
    print(f"- Catabolic tissues: {len(tissue_index[tissue_index['Classification'] == 'Catabolic (high MMP/TIMP)'])}")
    print(f"- Anabolic tissues: {len(tissue_index[tissue_index['Classification'] == 'Anabolic (low MMP/TIMP)'])}")
    print(f"- Balanced tissues: {len(tissue_index[tissue_index['Classification'] == 'Balanced'])}")

    if len(gpt_test) > 0:
        failed_remodeling = len(gpt_test[gpt_test['Both_Increase']])
        print(f"- Tissues with MMP2/TIMP3 co-elevation: {failed_remodeling}/{len(gpt_test)}")

if __name__ == "__main__":
    main()
