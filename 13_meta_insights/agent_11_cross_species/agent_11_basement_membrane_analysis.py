#!/usr/bin/env python3
"""
Agent 11: Basement Membrane Degradation Specialist
Tests the "basement membrane collapse" hypothesis from GPT Pro analysis.

Key hypothesis: COL4A3 (collagen IV α3) loss is the primary driver of
glomerular aging, leading to GBM dysfunction.

Analysis steps:
1. Identify ALL basement membrane components
2. Calculate z-score deltas across ALL tissues
3. Test if COL4A3 loss is universal or kidney-specific
4. Correlate BM protein losses with fibrinogen infiltration
5. Rank BM proteins by aging predictive power
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define basement membrane protein families
BM_PROTEINS = {
    'Collagen IV': ['COL4A1', 'COL4A2', 'COL4A3', 'COL4A4', 'COL4A5', 'COL4A6'],
    'Laminins': ['LAMA1', 'LAMA2', 'LAMA3', 'LAMA4', 'LAMA5',
                 'LAMB1', 'LAMB2', 'LAMB3', 'LAMC1', 'LAMC2', 'LAMC3'],
    'Nidogens': ['NID1', 'NID2'],
    'Perlecan': ['HSPG2'],
    'Agrin': ['AGRN'],
    'Collagen XVIII': ['COL18A1'],
    'Other BM': ['FREM1', 'FREM2', 'FRAS1', 'SNED1']
}

# Plasma proteins (test for basement membrane breach)
PLASMA_PROTEINS = ['FGA', 'FGB', 'FGG', 'ALB', 'SERPINA1', 'A2M', 'HP', 'HPX', 'APOA1', 'APOB']

def load_data(filepath):
    """Load and prepare ECM aging dataset."""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    print(f"Total rows: {len(df)}")
    print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    print(f"Tissues: {df['Tissue'].nunique()}")
    return df

def identify_bm_proteins(df):
    """Find all basement membrane proteins in dataset."""
    all_bm_genes = []
    for family, genes in BM_PROTEINS.items():
        all_bm_genes.extend(genes)

    bm_data = df[df['Canonical_Gene_Symbol'].isin(all_bm_genes)].copy()

    # Add family annotation
    gene_to_family = {}
    for family, genes in BM_PROTEINS.items():
        for gene in genes:
            gene_to_family[gene] = family

    bm_data['BM_Family'] = bm_data['Canonical_Gene_Symbol'].map(gene_to_family)

    print(f"\n{'='*70}")
    print(f"BASEMENT MEMBRANE PROTEINS IDENTIFIED")
    print(f"{'='*70}")
    print(f"Total BM protein entries: {len(bm_data)}")
    print(f"Unique BM proteins found: {bm_data['Canonical_Gene_Symbol'].nunique()}")
    print(f"\nProteins by family:")
    for family in BM_PROTEINS.keys():
        family_proteins = bm_data[bm_data['BM_Family'] == family]['Canonical_Gene_Symbol'].unique()
        print(f"  {family}: {len(family_proteins)} ({', '.join(sorted(family_proteins))})")

    return bm_data

def analyze_zscore_deltas(bm_data):
    """Calculate and rank z-score deltas for BM proteins."""
    # Filter for proteins with valid z-score deltas
    valid_data = bm_data[bm_data['Zscore_Delta'].notna()].copy()

    # Calculate statistics per protein across all tissues
    protein_stats = valid_data.groupby('Canonical_Gene_Symbol').agg({
        'Zscore_Delta': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'Tissue': lambda x: list(x.unique()),
        'BM_Family': 'first'
    }).round(3)

    protein_stats.columns = ['Mean_Delta', 'Median_Delta', 'Std_Delta',
                              'Min_Delta', 'Max_Delta', 'N_Tissues', 'Tissues', 'BM_Family']
    protein_stats = protein_stats.sort_values('Mean_Delta')

    print(f"\n{'='*70}")
    print(f"Z-SCORE DELTA ANALYSIS (ALL TISSUES)")
    print(f"{'='*70}")
    print(f"Proteins with valid deltas: {len(protein_stats)}")
    print(f"\nTop 10 DECREASING BM proteins (aging loss):")
    print(protein_stats[['BM_Family', 'Mean_Delta', 'Median_Delta', 'N_Tissues']].head(10))

    print(f"\nTop 10 INCREASING BM proteins (aging gain):")
    print(protein_stats[['BM_Family', 'Mean_Delta', 'Median_Delta', 'N_Tissues']].tail(10))

    return protein_stats

def test_col4a3_hypothesis(bm_data, df):
    """Test if COL4A3 loss is universal or kidney-specific."""
    print(f"\n{'='*70}")
    print(f"COL4A3 HYPOTHESIS TEST")
    print(f"{'='*70}")

    col4a3_data = bm_data[bm_data['Canonical_Gene_Symbol'] == 'COL4A3']

    if len(col4a3_data) == 0:
        print("COL4A3 NOT FOUND in dataset!")
        return None

    print(f"\nCOL4A3 observations: {len(col4a3_data)}")
    print(f"Tissues with COL4A3:")

    tissue_analysis = col4a3_data.groupby('Tissue').agg({
        'Zscore_Delta': ['mean', 'count'],
        'Abundance_Young': 'mean',
        'Abundance_Old': 'mean'
    }).round(3)

    tissue_analysis.columns = ['Mean_Delta', 'N_obs', 'Young_Abundance', 'Old_Abundance']
    tissue_analysis['Abundance_Change'] = tissue_analysis['Old_Abundance'] - tissue_analysis['Young_Abundance']
    tissue_analysis = tissue_analysis.sort_values('Mean_Delta')

    print(tissue_analysis)

    # Test if kidney shows strongest COL4A3 loss
    if 'Kidney' in tissue_analysis.index or any('kidney' in str(t).lower() for t in tissue_analysis.index):
        print("\n✓ COL4A3 detected in kidney tissue")
        kidney_rows = [t for t in tissue_analysis.index if 'kidney' in str(t).lower()]
        if kidney_rows:
            kidney_delta = tissue_analysis.loc[kidney_rows[0], 'Mean_Delta']
            other_deltas = tissue_analysis.loc[~tissue_analysis.index.isin(kidney_rows), 'Mean_Delta']

            if not other_deltas.empty:
                print(f"  Kidney COL4A3 delta: {kidney_delta:.3f}")
                print(f"  Other tissues mean delta: {other_deltas.mean():.3f}")
                print(f"  Kidney shows {'STRONGEST' if kidney_delta < other_deltas.mean() else 'WEAKER'} loss")
    else:
        print("\n✗ COL4A3 NOT detected in kidney tissue!")

    return col4a3_data

def analyze_tissue_specificity(bm_data):
    """Analyze which tissues show strongest BM degradation."""
    tissue_bm = bm_data[bm_data['Zscore_Delta'].notna()].groupby('Tissue').agg({
        'Zscore_Delta': ['mean', 'median', 'count'],
        'Canonical_Gene_Symbol': lambda x: len(x.unique())
    }).round(3)

    tissue_bm.columns = ['Mean_Delta', 'Median_Delta', 'N_obs', 'N_unique_proteins']
    tissue_bm = tissue_bm.sort_values('Mean_Delta')

    print(f"\n{'='*70}")
    print(f"TISSUE-SPECIFIC BM DEGRADATION")
    print(f"{'='*70}")
    print(f"Tissues ranked by BM protein loss (most degraded first):")
    print(tissue_bm)

    return tissue_bm

def correlate_with_plasma_proteins(df, bm_data):
    """Test if BM protein loss correlates with plasma protein infiltration."""
    print(f"\n{'='*70}")
    print(f"BASEMENT MEMBRANE BREACH TEST")
    print(f"{'='*70}")
    print(f"Testing correlation: BM protein loss → Plasma protein infiltration")

    # Get plasma proteins
    plasma_data = df[df['Canonical_Gene_Symbol'].isin(PLASMA_PROTEINS)].copy()
    print(f"\nPlasma proteins found: {plasma_data['Canonical_Gene_Symbol'].nunique()}")
    print(f"  ({', '.join(sorted(plasma_data['Canonical_Gene_Symbol'].unique()))})")

    # Calculate per-tissue metrics
    tissue_correlations = []

    for tissue in df['Tissue'].unique():
        tissue_bm = bm_data[(bm_data['Tissue'] == tissue) &
                            (bm_data['Zscore_Delta'].notna())]
        tissue_plasma = plasma_data[(plasma_data['Tissue'] == tissue) &
                                    (plasma_data['Zscore_Delta'].notna())]

        if len(tissue_bm) >= 3 and len(tissue_plasma) >= 1:
            bm_loss = tissue_bm['Zscore_Delta'].mean()  # Negative = loss
            plasma_gain = tissue_plasma['Zscore_Delta'].mean()  # Positive = infiltration

            tissue_correlations.append({
                'Tissue': tissue,
                'BM_Mean_Delta': bm_loss,
                'Plasma_Mean_Delta': plasma_gain,
                'N_BM_proteins': len(tissue_bm),
                'N_Plasma_proteins': len(tissue_plasma)
            })

    if len(tissue_correlations) >= 3:
        corr_df = pd.DataFrame(tissue_correlations)

        # Test correlation: BM loss (negative) should correlate with plasma gain (positive)
        # So we expect negative correlation
        r, p = stats.pearsonr(corr_df['BM_Mean_Delta'], corr_df['Plasma_Mean_Delta'])

        print(f"\n{'Correlation Results':^70}")
        print(f"{'-'*70}")
        print(f"Pearson r: {r:.3f}")
        print(f"P-value: {p:.4f}")
        print(f"N tissues: {len(corr_df)}")

        if p < 0.05:
            if r < 0:
                print(f"\n✓ SIGNIFICANT negative correlation detected!")
                print(f"  → BM protein LOSS correlates with plasma protein INFILTRATION")
                print(f"  → Supports 'BM breach' hypothesis")
            else:
                print(f"\n✗ Correlation is positive (unexpected direction)")
        else:
            print(f"\n✗ No significant correlation (p={p:.4f})")

        print(f"\nPer-tissue breakdown:")
        corr_df = corr_df.sort_values('BM_Mean_Delta')
        print(corr_df.to_string(index=False))

        return corr_df
    else:
        print(f"\nInsufficient data for correlation analysis (only {len(tissue_correlations)} tissues)")
        return None

def rank_therapeutic_targets(protein_stats, bm_data):
    """Rank BM proteins by therapeutic potential."""
    print(f"\n{'='*70}")
    print(f"THERAPEUTIC TARGET RANKING")
    print(f"{'='*70}")

    # Criteria for ranking:
    # 1. Strong loss signal (low mean delta)
    # 2. Consistency across tissues (low std)
    # 3. Present in multiple tissues (high N_Tissues)
    # 4. Structural importance (prioritize Col IV, laminins)

    targets = protein_stats.copy()

    # Calculate composite score (lower = better target)
    # Normalize each metric to 0-1 range
    targets['Loss_Score'] = (targets['Mean_Delta'] - targets['Mean_Delta'].min()) / (targets['Mean_Delta'].max() - targets['Mean_Delta'].min())
    targets['Consistency_Score'] = 1 - ((targets['Std_Delta'] - targets['Std_Delta'].min()) / (targets['Std_Delta'].max() - targets['Std_Delta'].min()))
    targets['Coverage_Score'] = (targets['N_Tissues'] - targets['N_Tissues'].min()) / (targets['N_Tissues'].max() - targets['N_Tissues'].min())

    # Structural importance (manual weighting)
    importance_weights = {
        'Collagen IV': 1.0,
        'Laminins': 0.9,
        'Nidogens': 0.7,
        'Perlecan': 0.8,
        'Agrin': 0.6,
        'Collagen XVIII': 0.5,
        'Other BM': 0.4
    }
    targets['Structure_Score'] = targets['BM_Family'].map(importance_weights)

    # Composite score (invert Loss_Score so negative = better)
    targets['Therapeutic_Score'] = (
        (1 - targets['Loss_Score']) * 0.4 +  # 40% weight on loss magnitude
        targets['Consistency_Score'] * 0.2 +  # 20% weight on consistency
        targets['Coverage_Score'] * 0.2 +     # 20% weight on tissue coverage
        targets['Structure_Score'] * 0.2      # 20% weight on structural importance
    )

    targets = targets.sort_values('Therapeutic_Score', ascending=False)

    print(f"\nTop 15 Therapeutic Targets (Ranked by Composite Score):")
    print(f"{'-'*70}")
    display_cols = ['BM_Family', 'Mean_Delta', 'N_Tissues', 'Therapeutic_Score']
    print(targets[display_cols].head(15).to_string())

    # Highlight COL4A3 position
    if 'COL4A3' in targets.index:
        col4a3_rank = list(targets.index).index('COL4A3') + 1
        col4a3_score = targets.loc['COL4A3', 'Therapeutic_Score']
        print(f"\nCOL4A3 ranking: #{col4a3_rank} (score: {col4a3_score:.3f})")
    else:
        print(f"\nCOL4A3: Not found in dataset")

    return targets

def create_cross_tissue_heatmap_data(bm_data):
    """Prepare data for cross-tissue BM degradation heatmap."""
    # Create pivot table: proteins x tissues
    heatmap_data = bm_data[bm_data['Zscore_Delta'].notna()].pivot_table(
        index='Canonical_Gene_Symbol',
        columns='Tissue',
        values='Zscore_Delta',
        aggfunc='mean'
    )

    print(f"\n{'='*70}")
    print(f"CROSS-TISSUE HEATMAP DATA")
    print(f"{'='*70}")
    print(f"Matrix dimensions: {heatmap_data.shape[0]} proteins × {heatmap_data.shape[1]} tissues")
    print(f"Data coverage: {(~heatmap_data.isna()).sum().sum()} / {heatmap_data.size} cells ({100*(~heatmap_data.isna()).sum().sum()/heatmap_data.size:.1f}%)")

    # Save for visualization
    output_file = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/bm_tissue_heatmap.csv'
    heatmap_data.to_csv(output_file)
    print(f"\nSaved to: {output_file}")

    return heatmap_data

def main():
    """Execute full basement membrane collapse analysis."""
    print(f"\n{'#'*70}")
    print(f"{'AGENT 11: BASEMENT MEMBRANE DEGRADATION SPECIALIST':^70}")
    print(f"{'#'*70}\n")

    # Load data
    data_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
    df = load_data(data_path)

    # 1. Identify BM proteins
    bm_data = identify_bm_proteins(df)

    # 2. Analyze z-score deltas
    protein_stats = analyze_zscore_deltas(bm_data)

    # 3. Test COL4A3 hypothesis
    col4a3_data = test_col4a3_hypothesis(bm_data, df)

    # 4. Tissue specificity
    tissue_analysis = analyze_tissue_specificity(bm_data)

    # 5. Correlate with plasma proteins (BM breach test)
    breach_correlation = correlate_with_plasma_proteins(df, bm_data)

    # 6. Rank therapeutic targets
    therapeutic_targets = rank_therapeutic_targets(protein_stats, bm_data)

    # 7. Prepare heatmap data
    heatmap_data = create_cross_tissue_heatmap_data(bm_data)

    # Save results
    output_dir = Path('/Users/Kravtsovd/projects/ecm-atlas/10_insights')

    protein_stats.to_csv(output_dir / 'bm_protein_statistics.csv')
    therapeutic_targets.to_csv(output_dir / 'bm_therapeutic_targets.csv')

    if breach_correlation is not None:
        breach_correlation.to_csv(output_dir / 'bm_breach_correlation.csv', index=False)

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - bm_protein_statistics.csv")
    print(f"  - bm_therapeutic_targets.csv")
    print(f"  - bm_breach_correlation.csv")
    print(f"  - bm_tissue_heatmap.csv")

    return {
        'bm_data': bm_data,
        'protein_stats': protein_stats,
        'tissue_analysis': tissue_analysis,
        'breach_correlation': breach_correlation,
        'therapeutic_targets': therapeutic_targets,
        'heatmap_data': heatmap_data
    }

if __name__ == '__main__':
    results = main()
