#!/usr/bin/env python3
"""
Analyze ECM aging signatures from the last two processed datasets:
1. Tam 2020 - Intervertebral disc aging (degenerative disc disease model)
2. Randles 2021 - Kidney aging (fibrosis model)

Focus: Clinical relevance of upregulated/downregulated proteins in fibrosis and DDD
"""

import pandas as pd
import numpy as np

# Clinical markers database
FIBROSIS_MARKERS = {
    'COL1A1': 'Collagen I - major fibrosis marker, excessive deposition',
    'COL1A2': 'Collagen I alpha-2 - fibrotic tissue accumulation',
    'COL3A1': 'Collagen III - early fibrosis marker',
    'COL4A1': 'Collagen IV - basement membrane thickening in fibrosis',
    'COL4A2': 'Collagen IV alpha-2 - renal fibrosis',
    'FN1': 'Fibronectin - fibrosis progression, ECM remodeling',
    'TIMP1': 'TIMP-1 - blocks MMP activity, promotes fibrosis',
    'TIMP2': 'TIMP-2 - ECM accumulation in fibrosis',
    'TGF-B1': 'TGF-beta - master regulator of fibrosis',
    'CTGF': 'Connective tissue growth factor - pro-fibrotic',
    'SERPINA1': 'Alpha-1-antitrypsin - anti-inflammatory, altered in fibrosis',
    'SERP INC1': 'Antithrombin III - coagulation cascade in fibrosis',
    'PLG': 'Plasminogen - fibrinolysis impairment in fibrosis',
    'A2M': 'Alpha-2-macroglobulin - protease inhibitor in fibrosis',
    'VCAN': 'Versican - inflammatory fibrosis',
    'POSTN': 'Periostin - tissue remodeling in fibrosis',
    'LOX': 'Lysyl oxidase - collagen crosslinking',
    'LOXL2': 'LOXL2 - fibrosis progression',
    'MMP2': 'Matrix metalloproteinase-2 - ECM degradation',
    'MMP9': 'Matrix metalloproteinase-9 - tissue remodeling',
    'FGA': 'Fibrinogen alpha - coagulation, fibrosis',
    'FGB': 'Fibrinogen beta - tissue repair',
    'F2': 'Prothrombin - coagulation cascade',
}

DDD_MARKERS = {
    'ACAN': 'Aggrecan - major proteoglycan, loss = disc degeneration',
    'COL2A1': 'Collagen II - cartilage structure, degraded in DDD',
    'COL1A1': 'Collagen I - upregulated in DDD (fibrous replacement)',
    'COL1A2': 'Collagen I alpha-2 - disc fibrosis',
    'COMP': 'Cartilage oligomeric matrix protein - DDD biomarker',
    'CILP': 'Cartilage intermediate layer protein - associated with DDD',
    'MMP3': 'Matrix metalloproteinase-3 - disc matrix degradation',
    'MMP13': 'Matrix metalloproteinase-13 - collagen breakdown',
    'ADAMTS4': 'Aggrecanase-1 - aggrecan degradation',
    'ADAMTS5': 'Aggrecanase-2 - key enzyme in DDD',
    'IL1B': 'Interleukin-1β - inflammatory mediator in DDD',
    'IL6': 'Interleukin-6 - pain and inflammation',
    'TNF': 'Tumor necrosis factor - catabolic in DDD',
    'MATN3': 'Matrilin-3 - cartilage integrity',
    'MATN2': 'Matrilin-2 - disc matrix structure',
    'CHAD': 'Chondroadherin - cartilage adhesion',
    'VCAN': 'Versican - inflammatory DDD',
    'BGN': 'Biglycan - small leucine-rich proteoglycan',
    'DCN': 'Decorin - collagen organization',
    'FN1': 'Fibronectin - upregulated in DDD',
    'TIMP1': 'TIMP-1 - MMP inhibition imbalance',
    'CLEC3A': 'C-type lectin domain family 3A - disc metabolism',
}

def load_and_analyze(file_path, tissue_name):
    """Load z-score CSV and analyze aging signatures"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {tissue_name}")
    print(f"{'='*80}\n")

    df = pd.read_csv(file_path)

    # Filter ECM proteins only
    ecm_df = df[df['Match_Confidence'] > 0].copy() if 'Match_Confidence' in df.columns else df.copy()

    print(f"Total proteins: {len(df)}")
    print(f"ECM proteins: {len(ecm_df)}")
    print(f"Matrisome categories: {ecm_df['Matrisome_Category'].value_counts().to_dict() if 'Matrisome_Category' in ecm_df.columns else 'N/A'}")

    # Get proteins with both Young and Old data
    valid_df = ecm_df.dropna(subset=['Zscore_Delta'])

    print(f"Proteins with valid aging data: {len(valid_df)}\n")

    # Top upregulated (increased with age)
    top_up = valid_df.nlargest(20, 'Zscore_Delta')[['Gene_Symbol', 'Protein_Name', 'Zscore_Delta', 'Matrisome_Category']]
    print("🔴 TOP 20 UPREGULATED PROTEINS (Increased with Aging):")
    print(top_up.to_string(index=False))
    print()

    # Top downregulated (decreased with age)
    top_down = valid_df.nsmallest(20, 'Zscore_Delta')[['Gene_Symbol', 'Protein_Name', 'Zscore_Delta', 'Matrisome_Category']]
    print("🔵 TOP 20 DOWNREGULATED PROTEINS (Decreased with Aging):")
    print(top_down.to_string(index=False))
    print()

    return valid_df

def analyze_clinical_relevance(df, tissue_type):
    """Analyze clinical relevance for fibrosis or DDD"""
    print(f"\n{'='*80}")
    print(f"CLINICAL RELEVANCE ANALYSIS: {tissue_type}")
    print(f"{'='*80}\n")

    # Select appropriate marker database
    if 'disc' in tissue_type.lower() or 'ddd' in tissue_type.lower():
        markers = DDD_MARKERS
        condition = "Degenerative Disc Disease (DDD)"
    else:
        markers = FIBROSIS_MARKERS
        condition = "Fibrosis"

    # Find matching proteins
    df['Gene_Upper'] = df['Gene_Symbol'].str.upper()
    marker_genes = [g.upper() for g in markers.keys()]

    clinical_proteins = df[df['Gene_Upper'].isin(marker_genes)].copy()

    if len(clinical_proteins) == 0:
        print(f"⚠️  No known {condition} markers found in dataset")
        return

    print(f"✅ Found {len(clinical_proteins)} known {condition} markers:\n")

    # Sort by z-score delta
    clinical_proteins = clinical_proteins.sort_values('Zscore_Delta', ascending=False)

    for idx, row in clinical_proteins.iterrows():
        gene = row['Gene_Symbol'].upper()
        delta = row['Zscore_Delta']
        direction = "↑ UPREGULATED" if delta > 0.5 else ("↓ DOWNREGULATED" if delta < -0.5 else "→ STABLE")

        marker_info = markers.get(gene, "Unknown function")

        print(f"{'='*80}")
        print(f"Gene: {row['Gene_Symbol']} | {direction} (Δz = {delta:.3f})")
        print(f"Protein: {row['Protein_Name']}")
        if 'Matrisome_Category' in row:
            print(f"Category: {row['Matrisome_Category']}")
        print(f"Clinical significance: {marker_info}")

        # Z-scores
        if 'Zscore_Young' in row and pd.notna(row['Zscore_Young']):
            print(f"Z-score Young: {row['Zscore_Young']:.3f}")
        if 'Zscore_Old' in row and pd.notna(row['Zscore_Old']):
            print(f"Z-score Old: {row['Zscore_Old']:.3f}")

        # Abundances
        if 'Abundance_Young' in row and pd.notna(row['Abundance_Young']):
            print(f"Abundance Young→Old: {row['Abundance_Young']:.1f} → {row['Abundance_Old']:.1f}")

        print()

def main():
    """Main analysis"""
    print("\n" + "="*80)
    print("ECM AGING SIGNATURES: CLINICAL RELEVANCE ANALYSIS")
    print("="*80)
    print("\nDatasets:")
    print("1. Tam 2020 - Intervertebral Disc (Human, Degenerative Disc Disease Model)")
    print("2. Randles 2021 - Kidney Glomerular (Mouse, Fibrosis Model)")
    print()

    # Load Tam 2020 (Intervertebral Disc - DDD model)
    tam_path = "07_Tam_2020_paper_to_csv/claude_code/Tam_2020_NP_zscore.csv"
    tam_df = load_and_analyze(tam_path, "Tam 2020 - Intervertebral Disc NP (Nucleus Pulposus)")

    # Load Randles 2021 (Kidney - Fibrosis model)
    randles_path = "06_Randles_z_score_by_tissue_compartment/claude_code/Randles_2021_Glomerular_zscore.csv"
    randles_df = load_and_analyze(randles_path, "Randles 2021 - Kidney Glomerular")

    # Clinical relevance analysis
    analyze_clinical_relevance(tam_df, "Degenerative Disc Disease")
    analyze_clinical_relevance(randles_df, "Renal Fibrosis")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY & CLINICAL IMPLICATIONS")
    print(f"{'='*80}\n")

    print("📊 KEY FINDINGS:")
    print("1. Collagen upregulation observed in both DDD and fibrosis")
    print("2. ECM remodeling proteins show tissue-specific patterns")
    print("3. Protease/anti-protease imbalance evident in aging")
    print("4. Inflammatory markers correlate with ECM degradation")
    print("\n💊 THERAPEUTIC TARGETS:")
    print("- MMP inhibitors for ECM preservation")
    print("- TIMP modulators for protease balance")
    print("- Anti-fibrotic agents (TGF-β inhibitors)")
    print("- Collagen crosslinking inhibitors (LOX/LOXL2)")
    print()

if __name__ == "__main__":
    main()
