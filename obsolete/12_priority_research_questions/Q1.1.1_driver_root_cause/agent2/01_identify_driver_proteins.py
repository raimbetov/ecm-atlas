#!/usr/bin/env python3
"""
Identify 4 driver proteins showing early decline (age 30-50) from ECM-Atlas database.
Focus on proteins with universal decline patterns across tissues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.1_driver_root_cause/agent2")

def load_ecm_data():
    """Load ECM-Atlas database"""
    print("Loading ECM-Atlas database...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records, {df['Canonical_Gene_Symbol'].nunique()} unique proteins")
    return df

def identify_universal_decliners(df, min_studies=3, consistency_threshold=0.8):
    """
    Identify proteins showing consistent decline across multiple studies

    Parameters:
    - min_studies: minimum number of studies protein must appear in
    - consistency_threshold: % of studies showing decline (negative Zscore_Delta)
    """
    print(f"\nAnalyzing for universal decliners (min {min_studies} studies, {consistency_threshold*100}% consistency)...")

    # Group by protein
    protein_stats = []

    for gene in df['Canonical_Gene_Symbol'].dropna().unique():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene].copy()

        # Skip if insufficient data
        if len(gene_data) < min_studies:
            continue

        # Count studies
        n_studies = gene_data['Study_ID'].nunique()
        if n_studies < min_studies:
            continue

        # Calculate per-study averages
        study_deltas = gene_data.groupby('Study_ID')['Zscore_Delta'].mean()

        # Count declining studies (negative delta)
        n_declining = (study_deltas < 0).sum()
        consistency = n_declining / n_studies if n_studies > 0 else 0

        # Average metrics
        avg_delta = study_deltas.mean()
        tissues = gene_data['Tissue'].nunique()

        protein_stats.append({
            'Gene': gene,
            'N_Studies': n_studies,
            'N_Declining': n_declining,
            'Consistency': consistency,
            'Avg_Zscore_Delta': avg_delta,
            'N_Tissues': tissues,
            'Matrisome_Category': gene_data['Matrisome_Category'].mode()[0] if len(gene_data) > 0 else 'Unknown'
        })

    stats_df = pd.DataFrame(protein_stats)

    # Filter for universal decliners
    decliners = stats_df[
        (stats_df['Consistency'] >= consistency_threshold) &
        (stats_df['Avg_Zscore_Delta'] < 0)
    ].sort_values('Avg_Zscore_Delta')

    print(f"Found {len(decliners)} universal declining proteins")

    return decliners, stats_df

def analyze_temporal_pattern(df, gene_symbol):
    """Analyze age-related decline pattern for specific protein"""
    gene_data = df[df['Canonical_Gene_Symbol'] == gene_symbol].copy()

    # We need to infer age from abundance patterns since age not directly in data
    # Use Zscore_Old and Zscore_Young as proxies

    return gene_data

def main():
    # Load data
    df = load_ecm_data()

    # Identify universal decliners
    decliners, all_stats = identify_universal_decliners(df, min_studies=4, consistency_threshold=0.8)

    print("\n" + "="*80)
    print("TOP 10 UNIVERSAL DECLINING PROTEINS")
    print("="*80)
    print(decliners.head(10).to_string(index=False))

    # Save results
    output_file = OUTPUT_DIR / "driver_proteins_analysis.csv"
    decliners.to_csv(output_file, index=False)
    print(f"\n✓ Saved analysis to {output_file}")

    # Focus on top 4 candidates
    top4 = decliners.head(4)

    print("\n" + "="*80)
    print("TOP 4 DRIVER PROTEIN CANDIDATES")
    print("="*80)
    for idx, row in top4.iterrows():
        print(f"\n{row['Gene']}:")
        print(f"  - Studies: {row['N_Studies']}")
        print(f"  - Consistency: {row['Consistency']*100:.0f}%")
        print(f"  - Avg Δz-score: {row['Avg_Zscore_Delta']:.3f}")
        print(f"  - Tissues: {row['N_Tissues']}")
        print(f"  - Category: {row['Matrisome_Category']}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Top 4 Driver Proteins: Universal Decline Analysis', fontsize=16, fontweight='bold')

    for idx, (_, protein) in enumerate(top4.iterrows()):
        ax = axes[idx // 2, idx % 2]
        gene = protein['Gene']

        # Get detailed data for this protein
        gene_data = df[df['Canonical_Gene_Symbol'] == gene].copy()

        # Plot z-score deltas by study
        study_deltas = gene_data.groupby('Study_ID')['Zscore_Delta'].mean().sort_values()

        ax.barh(range(len(study_deltas)), study_deltas.values,
                color=['red' if x < 0 else 'green' for x in study_deltas.values])
        ax.set_yticks(range(len(study_deltas)))
        ax.set_yticklabels([s[:20] for s in study_deltas.index], fontsize=8)
        ax.set_xlabel('Avg Δz-score (Old - Young)', fontsize=10)
        ax.set_title(f'{gene}\n({protein["N_Studies"]} studies, {protein["Consistency"]*100:.0f}% decline)',
                     fontsize=11, fontweight='bold')
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    viz_file = OUTPUT_DIR / "driver_proteins_visualization.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {viz_file}")

    return decliners, top4

if __name__ == "__main__":
    decliners, top4 = main()
    print("\n✓ Analysis complete!")
