#!/usr/bin/env python3
"""
Agent 11: Cross-Species Comparative Biologist

MISSION: Compare aging patterns across species (human, mouse, rat, etc.)
to find evolutionarily conserved vs species-specific aging mechanisms

Author: Claude Code
Date: 2025-10-15
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Lifespan data for evolutionary context
SPECIES_LIFESPAN = {
    'Homo sapiens': {'max_years': 122, 'average_years': 73, 'rank': 1},
    'Mus musculus': {'max_years': 4, 'average_years': 2, 'rank': 3},
    'Rattus norvegicus': {'max_years': 4, 'average_years': 3, 'rank': 3},
    'Bos taurus': {'max_years': 22, 'average_years': 18, 'rank': 2},
}

def load_dataset(path):
    """Load merged ECM dataset"""
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}\n")
    return df

def analyze_species_distribution(df):
    """Analyze species distribution in dataset"""
    print("="*80)
    print("1.0 SPECIES DISTRIBUTION")
    print("="*80)

    species_counts = df['Species'].value_counts()
    print(f"\nUnique species: {len(species_counts)}")
    print(f"\nProtein measurements per species:")
    for species, count in species_counts.items():
        lifespan = SPECIES_LIFESPAN.get(species, {}).get('max_years', 'Unknown')
        print(f"  {species}: {count} proteins (max lifespan: {lifespan} years)")

    # Proteins with valid aging data per species
    valid_aging = df.dropna(subset=['Zscore_Delta'])
    species_valid = valid_aging['Species'].value_counts()
    print(f"\nProteins with valid aging data (Zscore_Delta not NaN):")
    for species, count in species_valid.items():
        print(f"  {species}: {count} proteins")

    return species_counts, valid_aging

def identify_orthologous_proteins(df):
    """Find proteins present in multiple species (orthologous)"""
    print("\n" + "="*80)
    print("2.0 ORTHOLOGOUS PROTEIN IDENTIFICATION")
    print("="*80)

    # Group by Gene Symbol
    gene_species = df.groupby('Canonical_Gene_Symbol')['Species'].apply(set).to_dict()

    # Filter genes present in 2+ species
    multi_species_genes = {gene: species for gene, species in gene_species.items()
                           if len(species) > 1}

    print(f"\nTotal unique genes: {len(gene_species)}")
    print(f"Genes present in multiple species: {len(multi_species_genes)}")

    # Distribution of gene presence
    species_count_dist = defaultdict(int)
    for gene, species_set in multi_species_genes.items():
        species_count_dist[len(species_set)] += 1

    print(f"\nDistribution of orthologous genes:")
    for n_species, n_genes in sorted(species_count_dist.items()):
        print(f"  {n_species} species: {n_genes} genes")

    # Most widely distributed genes
    widely_distributed = sorted([(gene, species) for gene, species in multi_species_genes.items()],
                                 key=lambda x: len(x[1]), reverse=True)[:20]

    print(f"\nTop 20 most widely distributed genes:")
    for gene, species_set in widely_distributed:
        print(f"  {gene}: {len(species_set)} species - {', '.join(sorted(species_set))}")

    return multi_species_genes

def calculate_conservation_scores(df, multi_species_genes):
    """Calculate conservation score: correlation of Zscore_Delta between species"""
    print("\n" + "="*80)
    print("3.0 EVOLUTIONARY CONSERVATION ANALYSIS")
    print("="*80)

    conservation_results = []

    # Filter valid aging data
    valid_df = df.dropna(subset=['Zscore_Delta'])

    # Get all species pairs
    species_list = valid_df['Species'].unique()

    print(f"\nComparing aging trajectories across {len(species_list)} species...\n")

    for i, sp1 in enumerate(species_list):
        for sp2 in species_list[i+1:]:
            # Get shared genes between two species
            shared_genes = [gene for gene, species_set in multi_species_genes.items()
                           if sp1 in species_set and sp2 in species_set]

            if len(shared_genes) < 3:  # Need at least 3 points for correlation
                continue

            # Extract z-score deltas for shared genes
            sp1_data = valid_df[valid_df['Species'] == sp1].set_index('Canonical_Gene_Symbol')['Zscore_Delta']
            sp2_data = valid_df[valid_df['Species'] == sp2].set_index('Canonical_Gene_Symbol')['Zscore_Delta']

            # Get z-scores for shared genes
            shared_z1 = []
            shared_z2 = []
            for gene in shared_genes:
                if gene in sp1_data.index and gene in sp2_data.index:
                    z1_vals = sp1_data.loc[gene]
                    z2_vals = sp2_data.loc[gene]

                    # Handle multiple entries per gene (different tissues)
                    if isinstance(z1_vals, pd.Series):
                        z1_vals = z1_vals.values
                    else:
                        z1_vals = [z1_vals]

                    if isinstance(z2_vals, pd.Series):
                        z2_vals = z2_vals.values
                    else:
                        z2_vals = [z2_vals]

                    # Use mean if multiple values
                    shared_z1.append(np.nanmean(z1_vals))
                    shared_z2.append(np.nanmean(z2_vals))

            if len(shared_z1) < 3:
                continue

            # Calculate Pearson correlation
            r, p = pearsonr(shared_z1, shared_z2)

            conservation_results.append({
                'Species_1': sp1,
                'Species_2': sp2,
                'N_Shared_Genes': len(shared_z1),
                'Conservation_Score_R': r,
                'P_value': p,
                'Significance': 'Conserved' if r > 0.3 and p < 0.05 else
                               ('Divergent' if r < -0.3 and p < 0.05 else 'Neutral')
            })

    results_df = pd.DataFrame(conservation_results).sort_values('Conservation_Score_R', ascending=False)

    print("Species-pair conservation scores (Pearson R):")
    print(results_df.to_string(index=False))

    return results_df

def identify_conserved_aging_proteins(df, multi_species_genes):
    """Find proteins with consistent aging direction across species"""
    print("\n" + "="*80)
    print("4.0 UNIVERSALLY CONSERVED AGING PROTEINS")
    print("="*80)

    valid_df = df.dropna(subset=['Zscore_Delta'])

    conserved_proteins = []

    for gene, species_set in multi_species_genes.items():
        if len(species_set) < 2:
            continue

        gene_data = valid_df[valid_df['Canonical_Gene_Symbol'] == gene]

        if len(gene_data) < 2:
            continue

        # Get z-score deltas for all species
        deltas_by_species = {}
        for species in species_set:
            species_data = gene_data[gene_data['Species'] == species]['Zscore_Delta'].values
            if len(species_data) > 0:
                deltas_by_species[species] = np.nanmean(species_data)

        if len(deltas_by_species) < 2:
            continue

        deltas = list(deltas_by_species.values())

        # Check if all same direction
        all_positive = all(d > 0.5 for d in deltas)
        all_negative = all(d < -0.5 for d in deltas)

        if all_positive or all_negative:
            conserved_proteins.append({
                'Gene': gene,
                'N_Species': len(deltas_by_species),
                'Species': ', '.join(deltas_by_species.keys()),
                'Mean_Zscore_Delta': np.mean(deltas),
                'Direction': 'Upregulated' if all_positive else 'Downregulated',
                'Conservation_Type': 'Universal aging signature',
                'Species_Deltas': '; '.join([f"{sp}:{d:.2f}" for sp, d in deltas_by_species.items()])
            })

    conserved_df = pd.DataFrame(conserved_proteins).sort_values('N_Species', ascending=False)

    print(f"\nFound {len(conserved_df)} proteins with conserved aging direction across species")
    print(f"\nUpregulated: {len(conserved_df[conserved_df['Direction'] == 'Upregulated'])}")
    print(f"Downregulated: {len(conserved_df[conserved_df['Direction'] == 'Downregulated'])}")

    print(f"\nTop 20 most conserved aging proteins:")
    print(conserved_df.head(20).to_string(index=False))

    return conserved_df

def identify_species_specific_proteins(df):
    """Find human-specific aging proteins (potential therapeutic targets)"""
    print("\n" + "="*80)
    print("5.0 SPECIES-SPECIFIC AGING MARKERS")
    print("="*80)

    valid_df = df.dropna(subset=['Zscore_Delta'])

    for species in valid_df['Species'].unique():
        print(f"\n{species}-specific aging proteins:")

        # Get genes only present in this species
        all_genes_by_species = valid_df.groupby('Canonical_Gene_Symbol')['Species'].apply(set).to_dict()
        species_only_genes = [gene for gene, sp_set in all_genes_by_species.items()
                             if sp_set == {species}]

        species_data = valid_df[
            (valid_df['Species'] == species) &
            (valid_df['Canonical_Gene_Symbol'].isin(species_only_genes))
        ]

        # Strong aging signature
        strong_aging = species_data[abs(species_data['Zscore_Delta']) > 1.5]

        print(f"  Genes only measured in {species}: {len(species_only_genes)}")
        print(f"  Strong aging changes (|Δz| > 1.5): {len(strong_aging)}")

        if len(strong_aging) > 0:
            top_species_specific = strong_aging.nlargest(10, 'Zscore_Delta', keep='all')
            print(f"\n  Top species-specific upregulated proteins:")
            for _, row in top_species_specific.head(5).iterrows():
                print(f"    {row['Canonical_Gene_Symbol']}: Δz={row['Zscore_Delta']:.2f} ({row['Matrisome_Category']})")

def lifespan_correlation_analysis(df, multi_species_genes):
    """Test if long-lived species show different ECM aging patterns"""
    print("\n" + "="*80)
    print("6.0 LIFESPAN CORRELATION HYPOTHESIS")
    print("="*80)

    valid_df = df.dropna(subset=['Zscore_Delta'])

    print("\nHypothesis: Long-lived species show slower/different ECM aging")
    print("\nSpecies maximum lifespan rankings:")
    for species, data in sorted(SPECIES_LIFESPAN.items(), key=lambda x: x[1]['max_years'], reverse=True):
        print(f"  {species}: {data['max_years']} years")

    # Calculate mean absolute z-score delta per species
    species_aging_rates = {}
    for species in valid_df['Species'].unique():
        species_data = valid_df[valid_df['Species'] == species]
        mean_abs_delta = species_data['Zscore_Delta'].abs().mean()
        species_aging_rates[species] = mean_abs_delta

    print(f"\nMean absolute aging rate (|Δz|) per species:")
    for species in sorted(species_aging_rates.keys(), key=lambda x: species_aging_rates[x], reverse=True):
        lifespan = SPECIES_LIFESPAN.get(species, {}).get('max_years', 'Unknown')
        print(f"  {species}: |Δz|={species_aging_rates[species]:.3f} (lifespan: {lifespan} yrs)")

    # Test correlation
    lifespans = []
    aging_rates = []
    for species in species_aging_rates.keys():
        if species in SPECIES_LIFESPAN:
            lifespans.append(SPECIES_LIFESPAN[species]['max_years'])
            aging_rates.append(species_aging_rates[species])

    if len(lifespans) > 2:
        r, p = pearsonr(lifespans, aging_rates)
        print(f"\nLifespan vs Aging Rate Correlation: R={r:.3f}, p={p:.3f}")
        if p < 0.05:
            print(f"  SIGNIFICANT: {'Negative' if r < 0 else 'Positive'} correlation")
        else:
            print(f"  NOT SIGNIFICANT (p > 0.05)")

def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print("AGENT 11: CROSS-SPECIES COMPARATIVE ANALYSIS")
    print("="*80)
    print("\nEvolution reveals what matters most - conserved changes are universal truths!")
    print("\nDataset: merged_ecm_aging_zscore.csv")
    print("="*80 + "\n")

    # Load data
    df_path = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
    df = load_dataset(df_path)

    # 1. Species distribution
    species_counts, valid_aging = analyze_species_distribution(df)

    # 2. Identify orthologous proteins
    multi_species_genes = identify_orthologous_proteins(df)

    # 3. Conservation scores
    conservation_df = calculate_conservation_scores(df, multi_species_genes)

    # 4. Conserved aging proteins
    conserved_df = identify_conserved_aging_proteins(df, multi_species_genes)

    # 5. Species-specific markers
    identify_species_specific_proteins(df)

    # 6. Lifespan correlation
    lifespan_correlation_analysis(df, multi_species_genes)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    return {
        'species_distribution': species_counts,
        'orthologous_genes': multi_species_genes,
        'conservation_scores': conservation_df,
        'conserved_proteins': conserved_df
    }

if __name__ == "__main__":
    results = main()
