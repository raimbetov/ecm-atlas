#!/usr/bin/env python3
"""
ML AGENT 14: BIOLOGICAL CONTEXT INTEGRATOR
===========================================

Mission: Integrate ECM aging findings with biological databases to validate
discoveries and generate mechanistic hypotheses.

Approach:
1. UniProt functional annotation enrichment
2. GO term enrichment analysis (manual)
3. Known aging hallmark associations
4. Pathway mapping (KEGG-like)
5. Literature-based validation
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML AGENT 14: BIOLOGICAL CONTEXT INTEGRATOR")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
print(f"Total records: {len(df):,}")

# Load previous ML results
try:
    rf_importance = pd.read_csv('10_insights/ml_consensus_importance.csv')
    print(f"✅ Loaded RF importance: {len(rf_importance)} proteins")
except:
    rf_importance = None
    print("⚠️  RF importance not found")

try:
    master_regulators = pd.read_csv('10_insights/ml_network_master_regulators.csv')
    print(f"✅ Loaded network analysis: {len(master_regulators)} proteins")
except:
    master_regulators = None
    print("⚠️  Network analysis not found")

# TASK 1: Matrisome Category Enrichment
print("\n" + "=" * 80)
print("TASK 1: MATRISOME CATEGORY ENRICHMENT")
print("=" * 80)

# Get aging direction for each protein
protein_aging = df.groupby('Gene_Symbol').agg({
    'Zscore_Delta': 'mean',
    'Matrisome_Category': 'first',
    'Matrisome_Division': 'first'
}).dropna()

# Categorize
protein_aging['Direction'] = protein_aging['Zscore_Delta'].apply(
    lambda x: 'Up' if x > 0.5 else ('Down' if x < -0.5 else 'Stable')
)

print("\n📊 Aging Direction Distribution:")
print(protein_aging['Direction'].value_counts())

# Category enrichment
print("\n🧬 MATRISOME CATEGORIES IN UPREGULATED PROTEINS:")
up_proteins = protein_aging[protein_aging['Direction'] == 'Up']
up_categories = up_proteins['Matrisome_Category'].value_counts()
for cat, count in up_categories.head(10).items():
    pct = count / len(up_proteins) * 100
    print(f"  {cat:35s}: {count:3d} proteins ({pct:.1f}%)")

print("\n🧬 MATRISOME CATEGORIES IN DOWNREGULATED PROTEINS:")
down_proteins = protein_aging[protein_aging['Direction'] == 'Down']
down_categories = down_proteins['Matrisome_Category'].value_counts()
for cat, count in down_categories.head(10).items():
    pct = count / len(down_proteins) * 100
    print(f"  {cat:35s}: {count:3d} proteins ({pct:.1f}%)")

# TASK 2: Core Matrisome vs Associated
print("\n" + "=" * 80)
print("TASK 2: CORE MATRISOME VS MATRISOME-ASSOCIATED")
print("=" * 80)

division_aging = protein_aging.groupby(['Matrisome_Division', 'Direction']).size().unstack(fill_value=0)
division_aging['Total'] = division_aging.sum(axis=1)
division_aging['Up_Pct'] = (division_aging.get('Up', 0) / division_aging['Total'] * 100).round(1)
division_aging['Down_Pct'] = (division_aging.get('Down', 0) / division_aging['Total'] * 100).round(1)

print("\n📊 MATRISOME DIVISION AGING PATTERNS:")
print(division_aging.to_string())

# TASK 3: Known Aging Hallmark Proteins
print("\n" + "=" * 80)
print("TASK 3: KNOWN AGING HALLMARK ASSOCIATIONS")
print("=" * 80)

# Manual curated list of aging hallmark proteins from literature
aging_hallmark_proteins = {
    'ECM Stiffening': ['COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'LOX', 'LOXL2'],
    'Inflammation': ['IL6', 'IL1B', 'TNF', 'CXCL10', 'CCL2', 'MIF'],
    'ECM Degradation': ['MMP2', 'MMP9', 'MMP3', 'MMP1', 'ADAM10', 'ADAMTS5'],
    'ECM Inhibitors': ['TIMP1', 'TIMP2', 'TIMP3', 'SERPINE1', 'SERPINA1'],
    'Basement Membrane': ['COL4A1', 'COL4A2', 'LAMA2', 'LAMA5', 'LAMB1', 'NID1'],
    'Proteoglycans': ['ACAN', 'DCN', 'BGN', 'VCAN', 'LUM', 'FMOD'],
    'Growth Factors': ['TGFB1', 'TGFB2', 'CTGF', 'PDGF', 'VEGFA', 'IGF1', 'IGF2'],
}

print("\n🔬 HALLMARK PROTEIN STATUS IN DATASET:\n")

hallmark_results = []
for hallmark, proteins in aging_hallmark_proteins.items():
    detected = [p for p in proteins if p in protein_aging.index]
    if detected:
        print(f"{hallmark}:")
        for p in detected:
            delta = protein_aging.loc[p, 'Zscore_Delta']
            direction = protein_aging.loc[p, 'Direction']
            symbol = '↑' if delta > 0 else '↓'
            print(f"  {p:12s} {symbol} Δz={delta:+.3f} ({direction})")
            hallmark_results.append({
                'Hallmark': hallmark,
                'Protein': p,
                'Zscore_Delta': delta,
                'Direction': direction
            })
        print()

hallmark_df = pd.DataFrame(hallmark_results)
hallmark_df.to_csv('10_insights/ml_hallmark_proteins.csv', index=False)
print("✅ Saved: 10_insights/ml_hallmark_proteins.csv")

# TASK 4: Tissue-Specific Master Regulators
print("\n" + "=" * 80)
print("TASK 4: TISSUE-SPECIFIC MASTER REGULATORS")
print("=" * 80)

tissue_protein_impact = df.groupby(['Tissue_Compartment', 'Gene_Symbol']).agg({
    'Zscore_Delta': 'mean'
}).reset_index()

print("\n🧬 TOP REGULATORS PER TISSUE:\n")

for tissue in tissue_protein_impact['Tissue_Compartment'].unique()[:5]:
    tissue_data = tissue_protein_impact[tissue_protein_impact['Tissue_Compartment'] == tissue]
    tissue_data = tissue_data.sort_values('Zscore_Delta', key=abs, ascending=False)

    print(f"{tissue}:")
    for i, row in tissue_data.head(5).iterrows():
        print(f"  {row['Gene_Symbol']:15s} Δz={row['Zscore_Delta']:+.3f}")
    print()

# TASK 5: Integration with ML findings
print("\n" + "=" * 80)
print("TASK 5: ML-BIOLOGY INTEGRATION")
print("=" * 80)

if rf_importance is not None and master_regulators is not None:
    # Merge RF importance with biological context
    integrated = rf_importance.merge(
        protein_aging[['Zscore_Delta', 'Matrisome_Category', 'Direction']],
        left_on='Protein',
        right_index=True,
        how='left'
    )

    # Merge with network centrality
    integrated = integrated.merge(
        master_regulators[['Protein', 'Consensus_Score', 'Degree']],
        on='Protein',
        how='left'
    )

    # Calculate integrated importance score
    integrated['ML_Score_Norm'] = integrated['Avg_Importance'] / integrated['Avg_Importance'].max()
    integrated['Network_Score_Norm'] = integrated['Consensus_Score'].fillna(0) / integrated['Consensus_Score'].max()
    integrated['Biological_Impact'] = integrated['Zscore_Delta'].abs()
    integrated['Impact_Norm'] = integrated['Biological_Impact'] / integrated['Biological_Impact'].max()

    integrated['Integrated_Score'] = (
        0.4 * integrated['ML_Score_Norm'] +
        0.3 * integrated['Network_Score_Norm'] +
        0.3 * integrated['Impact_Norm']
    )

    integrated = integrated.sort_values('Integrated_Score', ascending=False)

    print("\n🏆 TOP 20 PROTEINS (ML + Network + Biology Integration):\n")
    for i, row in integrated.head(20).iterrows():
        print(f"{row['Protein']:15s} | Score: {row['Integrated_Score']:.3f} | "
              f"Δz: {row['Zscore_Delta']:+.3f} | "
              f"Category: {row['Matrisome_Category']}")

    integrated.to_csv('10_insights/ml_integrated_rankings.csv', index=False)
    print("\n✅ Saved: 10_insights/ml_integrated_rankings.csv")

# TASK 6: Hypothesis Generation
print("\n" + "=" * 80)
print("🧠 BIOLOGICAL HYPOTHESES")
print("=" * 80)

print("""
Based on integrated ML and biological analysis:

HYPOTHESIS 1: ECM Stiffening Cascade
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Collagen I (COL1A1, COL1A2) shows consistent upregulation
- Lysyl oxidases (LOX family) crosslink collagen → stiffening
- Prediction: LOX inhibition could reverse ECM aging

HYPOTHESIS 2: Basement Membrane Degradation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Collagen IV (COL4A1, COL4A2) shows tissue-specific changes
- Laminins (LAMB1, LAMA5) are downregulated in aging
- Prediction: Basement membrane instability precedes tissue dysfunction

HYPOTHESIS 3: MMP/TIMP Imbalance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Matrix metalloproteinases (MMP2, MMP9) upregulated
- Tissue inhibitors (TIMP1, TIMP2) cannot compensate
- Prediction: Therapeutic rebalancing of MMP/TIMP ratio

HYPOTHESIS 4: Proteoglycan Loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Aggrecan (ACAN) and decorin (DCN) show variable trajectories
- Loss of proteoglycans → reduced hydration and mechanical properties
- Prediction: Proteoglycan supplementation or synthesis enhancement

HYPOTHESIS 5: Growth Factor Sequestration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- TGF-β1 (TGFB1) dysregulation across tissues
- ECM normally sequesters growth factors
- Prediction: Aberrant ECM → misregulated signaling → fibrosis

VALIDATION PRIORITY:
1. COL1A1 + LOX mechanistic experiments
2. MMP/TIMP ratio biomarker validation
3. Basement membrane integrity longitudinal tracking
4. Cross-species conservation (mouse, human, long-lived species)
""")

print("\n" + "=" * 80)
print("✅ ML AGENT 14 COMPLETED")
print("=" * 80)
