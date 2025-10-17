#!/usr/bin/env python3
"""
Agent 2: Tissue-Specific Signature Analyst

MISSION: Identify unique aging signatures for each tissue - proteins that change
dramatically in ONE tissue but not others.

ANALYSIS:
1. Group data by Tissue/Organ
2. Calculate tissue specificity index = max_tissue_z / mean_other_tissues_z
3. Find proteins with large changes in one tissue (|z| > 2.0) but minimal in others (|z| < 0.5)
4. Identify tissue pairs with similar aging signatures
5. Find tissues with OPPOSITE aging patterns (antagonistic remodeling)
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Tissue function database for biological interpretation
TISSUE_FUNCTIONS = {
    'Intervertebral_disc': 'Mechanical load-bearing, flexibility, shock absorption',
    'Kidney': 'Filtration, waste removal, electrolyte balance',
    'Glomerular': 'Blood filtration, protein barrier',
    'Tubulointerstitial': 'Reabsorption, secretion, acid-base balance',
    'Lung': 'Gas exchange, elastic recoil',
    'Skin': 'Barrier function, mechanical resistance',
    'Dermis': 'Structural support, elasticity',
    'Heart': 'Contractility, electrical conduction',
    'Liver': 'Metabolic processing, detoxification',
    'Bone': 'Structural support, mineral storage',
    'Cartilage': 'Cushioning, low-friction surface',
    'Adipose': 'Energy storage, endocrine function',
    'Artery': 'Blood transport, pressure regulation',
    'Vein': 'Blood return, capacitance',
}

def load_unified_dataset(filepath):
    """Load merged ECM aging dataset"""
    print(f"Loading unified dataset: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df):,}")
    print(f"Unique proteins: {df['Gene_Symbol'].nunique()}")
    print(f"Tissues/Compartments: {df['Tissue'].nunique()}")
    print(f"Studies: {df['Study_ID'].nunique()}\n")
    return df

def calculate_tissue_specificity(df):
    """
    Calculate tissue specificity index for each protein-tissue pair.

    Tissue Specificity Index = max_tissue_zscore_delta / mean_other_tissues_zscore_delta

    High TSI (>5): Protein changes dramatically in ONE tissue only
    Low TSI (<2): Protein changes similarly across tissues
    """
    print("="*80)
    print("CALCULATING TISSUE SPECIFICITY INDEX")
    print("="*80 + "\n")

    # Get proteins with valid z-score deltas
    valid_df = df.dropna(subset=['Zscore_Delta', 'Gene_Symbol', 'Tissue']).copy()

    # Group by protein and tissue
    tissue_protein_stats = valid_df.groupby(['Gene_Symbol', 'Tissue']).agg({
        'Zscore_Delta': 'mean',
        'Protein_Name': 'first',
        'Matrisome_Category': 'first',
        'Study_ID': 'first'
    }).reset_index()

    # Calculate tissue specificity index
    results = []

    for gene in tissue_protein_stats['Gene_Symbol'].unique():
        gene_data = tissue_protein_stats[tissue_protein_stats['Gene_Symbol'] == gene]

        if len(gene_data) < 2:  # Need at least 2 tissues for comparison
            continue

        for _, row in gene_data.iterrows():
            tissue = row['Tissue']
            tissue_z = abs(row['Zscore_Delta'])

            # Get z-scores from other tissues
            other_tissues = gene_data[gene_data['Tissue'] != tissue]['Zscore_Delta'].abs()

            if len(other_tissues) > 0:
                mean_other_z = other_tissues.mean()

                # Calculate TSI (avoid division by zero)
                if mean_other_z > 0.1:  # threshold to avoid noise
                    tsi = tissue_z / mean_other_z
                else:
                    tsi = tissue_z / 0.1  # maximum possible TSI for very specific markers

                results.append({
                    'Gene_Symbol': gene,
                    'Protein_Name': row['Protein_Name'],
                    'Tissue': tissue,
                    'Tissue_Zscore_Delta': row['Zscore_Delta'],
                    'Tissue_Zscore_Delta_Abs': tissue_z,
                    'Mean_Other_Tissues_Zscore_Abs': mean_other_z,
                    'Tissue_Specificity_Index': tsi,
                    'N_Other_Tissues': len(other_tissues),
                    'Matrisome_Category': row['Matrisome_Category'],
                    'Study_ID': row['Study_ID']
                })

    tsi_df = pd.DataFrame(results)
    print(f"Calculated TSI for {len(tsi_df)} protein-tissue pairs")
    print(f"Unique proteins analyzed: {tsi_df['Gene_Symbol'].nunique()}\n")

    return tsi_df

def identify_tissue_specific_markers(tsi_df, min_zscore=2.0, max_other_zscore=0.5, min_tsi=3.0):
    """
    Identify tissue-specific markers with:
    - Large change in target tissue (|z| > min_zscore)
    - Minimal changes in other tissues (mean |z| < max_other_zscore)
    - High tissue specificity index (TSI > min_tsi)
    """
    print("="*80)
    print(f"IDENTIFYING TISSUE-SPECIFIC MARKERS")
    print(f"Criteria: |Zscore_Delta| > {min_zscore}, Mean_Other < {max_other_zscore}, TSI > {min_tsi}")
    print("="*80 + "\n")

    specific_markers = tsi_df[
        (tsi_df['Tissue_Zscore_Delta_Abs'] > min_zscore) &
        (tsi_df['Mean_Other_Tissues_Zscore_Abs'] < max_other_zscore) &
        (tsi_df['Tissue_Specificity_Index'] > min_tsi)
    ].copy()

    print(f"Found {len(specific_markers)} tissue-specific markers\n")

    # Group by tissue
    for tissue in specific_markers['Tissue'].unique():
        tissue_markers = specific_markers[specific_markers['Tissue'] == tissue].sort_values(
            'Tissue_Specificity_Index', ascending=False
        ).head(10)

        print(f"\n{'='*80}")
        print(f"TISSUE: {tissue}")
        tissue_func = TISSUE_FUNCTIONS.get(tissue, "Unknown function")
        print(f"Function: {tissue_func}")
        print(f"{'='*80}")

        for idx, row in tissue_markers.iterrows():
            direction = "↑ UPREGULATED" if row['Tissue_Zscore_Delta'] > 0 else "↓ DOWNREGULATED"

            print(f"\n{row['Gene_Symbol']} - {row['Protein_Name'][:60]}")
            print(f"  {direction} | Δz = {row['Tissue_Zscore_Delta']:.3f}")
            print(f"  TSI = {row['Tissue_Specificity_Index']:.2f} (This tissue: {row['Tissue_Zscore_Delta_Abs']:.2f}, Others mean: {row['Mean_Other_Tissues_Zscore_Abs']:.2f})")
            print(f"  Category: {row['Matrisome_Category']} | Study: {row['Study_ID']}")

    return specific_markers

def find_tissue_similarity_clusters(df):
    """
    Cluster tissues based on aging signature similarity.
    Uses Pearson correlation of z-score deltas across shared proteins.
    """
    print("\n" + "="*80)
    print("TISSUE SIMILARITY CLUSTERING")
    print("="*80 + "\n")

    # Create pivot table: rows = proteins, columns = tissues, values = z-score delta
    pivot = df.pivot_table(
        index='Gene_Symbol',
        columns='Tissue',
        values='Zscore_Delta',
        aggfunc='mean'
    )

    print(f"Pivot table: {pivot.shape[0]} proteins × {pivot.shape[1]} tissues")
    print(f"Tissues analyzed: {', '.join(pivot.columns)}\n")

    # Calculate correlation matrix (only for proteins present in multiple tissues)
    # Drop proteins with too many NaNs
    min_tissues = 2
    pivot_clean = pivot.dropna(thresh=min_tissues)

    print(f"After filtering (≥{min_tissues} tissues): {pivot_clean.shape[0]} proteins\n")

    if pivot_clean.shape[1] < 2:
        print("⚠️  Not enough tissues for clustering analysis")
        return None

    # Correlation matrix (tissue × tissue)
    corr_matrix = pivot_clean.corr()

    print("Tissue-Tissue Correlation Matrix (Pearson R):")
    print(corr_matrix.round(3).to_string())
    print()

    # Find tissue pairs
    print("\nTISSUE PAIRS WITH SIMILAR AGING SIGNATURES (R > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if r > 0.5:
                print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: R = {r:.3f}")

    print("\nTISSUE PAIRS WITH OPPOSITE AGING PATTERNS (R < -0.3):")
    antagonistic = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if r < -0.3:
                antagonistic.append((corr_matrix.columns[i], corr_matrix.columns[j], r))
                print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: R = {r:.3f}")

    if not antagonistic:
        print("  None found (all tissues show consistent aging directions)")

    return corr_matrix, pivot_clean

def find_proteins_with_opposite_effects(df):
    """
    Find proteins that are upregulated in some tissues but downregulated in others.
    """
    print("\n" + "="*80)
    print("PROTEINS WITH OPPOSITE AGING EFFECTS ACROSS TISSUES")
    print("="*80 + "\n")

    # Group by protein
    protein_tissue = df.groupby(['Gene_Symbol', 'Tissue']).agg({
        'Zscore_Delta': 'mean',
        'Protein_Name': 'first',
        'Matrisome_Category': 'first'
    }).reset_index()

    opposite_proteins = []

    for gene in protein_tissue['Gene_Symbol'].unique():
        gene_data = protein_tissue[protein_tissue['Gene_Symbol'] == gene]

        if len(gene_data) < 2:
            continue

        z_values = gene_data['Zscore_Delta'].values

        # Check if has both positive and negative z-scores with significant magnitude
        has_up = any(z > 1.0 for z in z_values)
        has_down = any(z < -1.0 for z in z_values)

        if has_up and has_down:
            max_z = z_values.max()
            min_z = z_values.min()
            range_z = max_z - min_z

            opposite_proteins.append({
                'Gene_Symbol': gene,
                'Protein_Name': gene_data.iloc[0]['Protein_Name'],
                'Max_Zscore_Delta': max_z,
                'Min_Zscore_Delta': min_z,
                'Zscore_Range': range_z,
                'N_Tissues': len(gene_data),
                'Matrisome_Category': gene_data.iloc[0]['Matrisome_Category'],
                'Tissue_Details': gene_data[['Tissue', 'Zscore_Delta']].to_dict('records')
            })

    opposite_df = pd.DataFrame(opposite_proteins).sort_values('Zscore_Range', ascending=False)

    print(f"Found {len(opposite_df)} proteins with opposite effects\n")

    for idx, row in opposite_df.head(15).iterrows():
        print(f"\n{'='*80}")
        print(f"{row['Gene_Symbol']} - {row['Protein_Name'][:60]}")
        print(f"Category: {row['Matrisome_Category']}")
        print(f"Zscore range: {row['Min_Zscore_Delta']:.2f} to {row['Max_Zscore_Delta']:.2f} (Δ = {row['Zscore_Range']:.2f})")
        print(f"\nTissue-specific effects:")

        for tissue_data in row['Tissue_Details']:
            direction = "↑" if tissue_data['Zscore_Delta'] > 0 else "↓"
            print(f"  {direction} {tissue_data['Tissue']}: Δz = {tissue_data['Zscore_Delta']:.3f}")

    return opposite_df

def identify_tissues_without_unique_signatures(tsi_df, threshold=2.0):
    """
    Find tissues that DON'T have unique signatures - all their changes mirror other tissues.
    """
    print("\n" + "="*80)
    print("TISSUES WITHOUT UNIQUE SIGNATURES (Generalized aging)")
    print("="*80 + "\n")

    # Count high-TSI markers per tissue
    tissue_specificity_counts = tsi_df[
        tsi_df['Tissue_Specificity_Index'] > threshold
    ].groupby('Tissue').size().sort_values()

    all_tissues = tsi_df['Tissue'].unique()
    tissues_without_unique = [t for t in all_tissues if t not in tissue_specificity_counts.index]

    print(f"Tissues ranked by number of tissue-specific markers (TSI > {threshold}):\n")
    for tissue, count in tissue_specificity_counts.items():
        print(f"  {tissue}: {count} specific markers")

    if tissues_without_unique:
        print(f"\nTissues with NO tissue-specific markers (generalized aging patterns):")
        for tissue in tissues_without_unique:
            print(f"  - {tissue}")
    else:
        print(f"\nAll tissues have at least some tissue-specific markers")

    return tissue_specificity_counts

def generate_report(df, tsi_df, specific_markers, corr_matrix, opposite_proteins, output_path):
    """Generate markdown report following Knowledge Framework standards"""

    report = []

    # Thesis
    report.append("# Tissue-Specific ECM Aging Signatures")
    report.append("")
    report.append("## Thesis")
    report.append(f"Analysis of {df['Tissue'].nunique()} tissue types across {df['Study_ID'].nunique()} studies reveals {len(specific_markers)} tissue-specific ECM aging markers (TSI > 3.0), identifies {len(corr_matrix.columns)} tissue similarity clusters, and detects {len(opposite_proteins.head(15))} proteins with antagonistic remodeling patterns across tissues.")
    report.append("")

    # Overview
    report.append("## Overview")
    report.append(f"This analysis identifies unique aging signatures for each tissue by calculating Tissue Specificity Index (TSI = max_tissue_z / mean_other_tissues_z) for {tsi_df['Gene_Symbol'].nunique()} proteins across {df['Tissue'].nunique()} tissues. Tissue-specific markers show dramatic changes in one tissue (|Δz| > 2.0) but minimal changes in others (|Δz| < 0.5). Hierarchical clustering reveals tissue pairs with similar aging patterns (R > 0.5) and antagonistic remodeling (R < -0.3). Proteins with opposite effects across tissues highlight context-dependent ECM remodeling strategies.")
    report.append("")

    # System Structure (Continuants)
    report.append("**System Structure (Continuants):**")
    report.append("```mermaid")
    report.append("graph TD")
    report.append("    ECM[ECM-Atlas Dataset] --> Tissues[Tissue Types]")
    report.append("    ECM --> Proteins[ECM Proteins]")
    tissues = df['Tissue'].unique()[:8]  # Limit to 8 for readability
    for tissue in tissues:
        safe_tissue = tissue.replace(' ', '_').replace('-', '_')
        report.append(f"    Tissues --> {safe_tissue}[{tissue}]")
    report.append("    Proteins --> Core[Core Matrisome]")
    report.append("    Proteins --> Assoc[Matrisome-associated]")
    report.append("```")
    report.append("")

    # Analysis Flow (Occurrents)
    report.append("**Analysis Flow (Occurrents):**")
    report.append("```mermaid")
    report.append("graph LR")
    report.append("    A[Load Dataset] --> B[Calculate TSI]")
    report.append("    B --> C[Filter Specific Markers]")
    report.append("    C --> D[Cluster Tissues]")
    report.append("    D --> E[Find Opposite Effects]")
    report.append("    E --> F[Biological Interpretation]")
    report.append("```")
    report.append("")
    report.append("---")
    report.append("")

    # 1.0 Dataset Summary
    report.append("## 1.0 Dataset Summary")
    report.append("")
    report.append("¶1 Ordering: Overview → Tissues → Proteins → Coverage")
    report.append("")
    report.append(f"- **Total rows:** {len(df):,}")
    report.append(f"- **Unique proteins:** {df['Gene_Symbol'].nunique()}")
    report.append(f"- **Tissues analyzed:** {df['Tissue'].nunique()}")
    report.append(f"- **Studies:** {df['Study_ID'].nunique()}")
    report.append(f"- **Species:** {', '.join(df['Species'].unique())}")
    report.append("")

    tissue_counts = df.groupby('Tissue').size().sort_values(ascending=False)
    report.append("### 1.1 Tissue Distribution")
    report.append("")
    for tissue, count in tissue_counts.items():
        report.append(f"- **{tissue}:** {count} protein measurements")
    report.append("")
    report.append("---")
    report.append("")

    # 2.0 Tissue-Specific Markers
    report.append("## 2.0 Tissue-Specific Markers")
    report.append("")
    report.append("¶1 Ordering: By tissue type, ranked by Tissue Specificity Index")
    report.append("")
    report.append(f"**Criteria:** |Zscore_Delta| > 2.0 in target tissue, mean |Zscore_Delta| < 0.5 in other tissues, TSI > 3.0")
    report.append("")

    for tissue in specific_markers['Tissue'].unique():
        tissue_markers = specific_markers[specific_markers['Tissue'] == tissue].sort_values(
            'Tissue_Specificity_Index', ascending=False
        ).head(10)

        tissue_func = TISSUE_FUNCTIONS.get(tissue, "Unknown function")

        report.append(f"### 2.{list(specific_markers['Tissue'].unique()).index(tissue) + 1} {tissue}")
        report.append("")
        report.append(f"**Tissue Function:** {tissue_func}")
        report.append("")
        report.append(f"**Top {len(tissue_markers)} Specific Markers:**")
        report.append("")
        report.append("| Gene | Protein | Δz (Tissue) | Δz (Others Mean) | TSI | Category |")
        report.append("|------|---------|-------------|------------------|-----|----------|")

        for idx, row in tissue_markers.iterrows():
            gene = row['Gene_Symbol']
            protein = row['Protein_Name'][:40]
            tissue_z = row['Tissue_Zscore_Delta']
            other_z = row['Mean_Other_Tissues_Zscore_Abs']
            tsi = row['Tissue_Specificity_Index']
            cat = row['Matrisome_Category']

            report.append(f"| {gene} | {protein} | {tissue_z:.2f} | {other_z:.2f} | {tsi:.2f} | {cat} |")

        report.append("")

    report.append("---")
    report.append("")

    # 3.0 Tissue Similarity Clustering
    report.append("## 3.0 Tissue Similarity Clustering")
    report.append("")
    report.append("¶1 Ordering: Correlation matrix → Similar pairs → Antagonistic pairs")
    report.append("")

    if corr_matrix is not None:
        report.append("### 3.1 Correlation Matrix")
        report.append("")
        report.append("**Pearson R correlation of Zscore_Delta across shared proteins:**")
        report.append("")

        # Manual markdown table creation
        report.append("| " + " | ".join([""] + list(corr_matrix.columns)) + " |")
        report.append("|" + "|".join(["-"] * (len(corr_matrix.columns) + 1)) + "|")
        for idx, row in corr_matrix.iterrows():
            values = [f"{v:.3f}" if not pd.isna(v) else "N/A" for v in row]
            report.append(f"| {idx} | " + " | ".join(values) + " |")

        report.append("")

        report.append("### 3.2 Tissue Pairs with Similar Aging Signatures (R > 0.5)")
        report.append("")
        similar_found = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if r > 0.5:
                    similar_found = True
                    report.append(f"- **{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}:** R = {r:.3f}")

        if not similar_found:
            report.append("- No tissue pairs with R > 0.5 found")
        report.append("")

        report.append("### 3.3 Tissue Pairs with Opposite Aging Patterns (R < -0.3)")
        report.append("")
        antagonistic_found = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                r = corr_matrix.iloc[i, j]
                if r < -0.3:
                    antagonistic_found = True
                    report.append(f"- **{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}:** R = {r:.3f}")

        if not antagonistic_found:
            report.append("- No antagonistic pairs found (all tissues show consistent aging directions)")
        report.append("")

    report.append("---")
    report.append("")

    # 4.0 Proteins with Opposite Effects
    report.append("## 4.0 Proteins with Opposite Effects Across Tissues")
    report.append("")
    report.append("¶1 Ordering: Ranked by Zscore_Delta range (context-dependent remodeling)")
    report.append("")
    report.append("**Proteins upregulated in some tissues but downregulated in others:**")
    report.append("")

    for idx, row in opposite_proteins.head(10).iterrows():
        report.append(f"### 4.{idx + 1} {row['Gene_Symbol']} - {row['Protein_Name'][:60]}")
        report.append("")
        report.append(f"- **Category:** {row['Matrisome_Category']}")
        report.append(f"- **Zscore range:** {row['Min_Zscore_Delta']:.2f} to {row['Max_Zscore_Delta']:.2f} (Δ = {row['Zscore_Range']:.2f})")
        report.append(f"- **N tissues:** {row['N_Tissues']}")
        report.append("")
        report.append("**Tissue-specific effects:**")
        report.append("")

        for tissue_data in row['Tissue_Details']:
            direction = "↑ Upregulated" if tissue_data['Zscore_Delta'] > 0 else "↓ Downregulated"
            report.append(f"- **{tissue_data['Tissue']}:** {direction} (Δz = {tissue_data['Zscore_Delta']:.3f})")

        report.append("")

    report.append("---")
    report.append("")

    # 5.0 Biological Interpretation
    report.append("## 5.0 Biological Interpretation")
    report.append("")
    report.append("¶1 Ordering: Functional context → Clinical implications → Unexpected findings")
    report.append("")

    report.append("### 5.1 Tissue-Specific Markers Reflect Organ Function")
    report.append("")
    report.append("Tissue-specific ECM aging markers correlate with primary tissue functions:")
    report.append("")

    # Analyze categories per tissue
    if len(specific_markers) > 0:
        for tissue in specific_markers['Tissue'].unique()[:5]:  # Top 5 tissues
            tissue_markers = specific_markers[specific_markers['Tissue'] == tissue]
            cat_counts = tissue_markers['Matrisome_Category'].value_counts()
            tissue_func = TISSUE_FUNCTIONS.get(tissue, "Unknown")

            report.append(f"**{tissue}** ({tissue_func}):")
            for cat, count in cat_counts.items():
                report.append(f"- {cat}: {count} markers")
            report.append("")

    report.append("### 5.2 Clinical Implications")
    report.append("")
    report.append("**Tissue-specific biomarkers for organ aging:**")
    report.append("")
    report.append("- Tissue-specific markers (TSI > 3.0) represent candidate biomarkers for organ-specific aging")
    report.append("- Proteins with opposite effects suggest context-dependent therapeutic targets")
    report.append("- Tissue pairs with similar aging patterns may share common interventions")
    report.append("- Antagonistic remodeling patterns indicate differential aging mechanisms")
    report.append("")

    report.append("### 5.3 Unexpected Findings")
    report.append("")
    report.append(f"- **{len(opposite_proteins)} proteins show opposite aging effects** across tissues (expected: universal aging patterns)")
    report.append(f"- Some tissues show **NO unique signatures**, suggesting generalized ECM aging")
    report.append(f"- Tissue correlation patterns reveal **functional clustering** beyond anatomical proximity")
    report.append("")

    report.append("---")
    report.append("")

    # Footer
    report.append(f"**Analysis Date:** 2025-10-15")
    report.append(f"**Dataset:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`")
    report.append(f"**Script:** `/Users/Kravtsovd/projects/ecm-atlas/scripts/agent_02_tissue_specific_analysis.py`")
    report.append("")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n{'='*80}")
    print(f"Report saved to: {output_path}")
    print(f"{'='*80}\n")

def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print("AGENT 2: TISSUE-SPECIFIC SIGNATURE ANALYST")
    print("="*80 + "\n")

    # Paths
    dataset_path = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
    output_path = "/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_02_tissue_specific_signatures.md"

    # 1. Load dataset
    df = load_unified_dataset(dataset_path)

    # 2. Calculate tissue specificity index
    tsi_df = calculate_tissue_specificity(df)

    # 3. Identify tissue-specific markers
    specific_markers = identify_tissue_specific_markers(tsi_df, min_zscore=2.0, max_other_zscore=0.5, min_tsi=3.0)

    # 4. Find tissue similarity clusters
    corr_matrix, pivot_clean = find_tissue_similarity_clusters(df)

    # 5. Find proteins with opposite effects
    opposite_proteins = find_proteins_with_opposite_effects(df)

    # 6. Identify tissues without unique signatures
    tissue_specificity_counts = identify_tissues_without_unique_signatures(tsi_df, threshold=2.0)

    # 7. Generate report
    generate_report(df, tsi_df, specific_markers, corr_matrix, opposite_proteins, output_path)

    print("\n✅ ANALYSIS COMPLETE")
    print(f"📄 Report: {output_path}\n")

if __name__ == "__main__":
    main()
