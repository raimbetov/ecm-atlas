#!/usr/bin/env python3
"""
Network Topology Mapper for ECM-Atlas
Agent 8: Protein Co-expression Network Analysis

Builds protein-protein correlation networks to identify modules that age together.
Uses correlation analysis, community detection, and centrality metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

def load_ecm_data(filepath):
    """Load and prepare ECM aging dataset for network analysis."""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"  Total records: {len(df):,}")
    print(f"  Unique proteins: {df['Gene_Symbol'].nunique()}")
    print(f"  Tissues: {df['Tissue'].nunique()}")
    print(f"  Studies: {df['Study_ID'].nunique()}")
    return df

def prepare_correlation_matrix(df):
    """
    Create protein x sample matrix for correlation analysis.
    Rows = proteins, Columns = tissue/study combinations
    Values = Zscore_Delta (aging signature)
    """
    print("\nPreparing correlation matrix...")

    # Create unique sample identifier
    df['Sample_ID'] = df['Tissue_Compartment'] + '_' + df['Study_ID']

    # Pivot to wide format: proteins x samples
    matrix = df.pivot_table(
        index='Gene_Symbol',
        columns='Sample_ID',
        values='Zscore_Delta',
        aggfunc='mean'  # Average if duplicates
    )

    print(f"  Matrix shape: {matrix.shape[0]} proteins x {matrix.shape[1]} samples")
    print(f"  Missing data: {matrix.isna().sum().sum() / matrix.size * 100:.1f}%")

    return matrix

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================

def calculate_protein_correlations(matrix, min_shared_samples=3, method='spearman'):
    """
    Calculate pairwise protein correlations.

    Args:
        matrix: Proteins x Samples DataFrame
        min_shared_samples: Minimum overlapping samples required
        method: 'pearson' or 'spearman'

    Returns:
        Correlation matrix, p-values matrix, valid pairs count
    """
    print(f"\nCalculating {method} correlations...")

    proteins = matrix.index.tolist()
    n_proteins = len(proteins)

    # Initialize matrices
    corr_matrix = pd.DataFrame(
        np.nan,
        index=proteins,
        columns=proteins
    )
    pval_matrix = pd.DataFrame(
        np.nan,
        index=proteins,
        columns=proteins
    )
    n_samples_matrix = pd.DataFrame(
        0,
        index=proteins,
        columns=proteins
    )

    # Calculate correlations
    print(f"  Computing {n_proteins * (n_proteins - 1) // 2:,} pairwise correlations...")

    for i, prot1 in enumerate(proteins):
        if i % 50 == 0:
            print(f"    Progress: {i}/{n_proteins} proteins processed")

        for j, prot2 in enumerate(proteins):
            if i >= j:  # Skip diagonal and duplicates
                continue

            # Get valid (non-NaN) shared samples
            data1 = matrix.loc[prot1]
            data2 = matrix.loc[prot2]

            valid_mask = ~(data1.isna() | data2.isna())
            n_shared = valid_mask.sum()

            if n_shared >= min_shared_samples:
                vals1 = data1[valid_mask].values
                vals2 = data2[valid_mask].values

                # Calculate correlation
                if method == 'spearman':
                    corr, pval = stats.spearmanr(vals1, vals2)
                else:
                    corr, pval = stats.pearsonr(vals1, vals2)

                # Store (symmetric)
                corr_matrix.loc[prot1, prot2] = corr
                corr_matrix.loc[prot2, prot1] = corr
                pval_matrix.loc[prot1, prot2] = pval
                pval_matrix.loc[prot2, prot1] = pval
                n_samples_matrix.loc[prot1, prot2] = n_shared
                n_samples_matrix.loc[prot2, prot1] = n_shared

    # Set diagonal to 1
    for prot in proteins:
        corr_matrix.loc[prot, prot] = 1.0
        pval_matrix.loc[prot, prot] = 0.0

    print(f"  Valid correlations computed: {(~corr_matrix.isna()).sum().sum() - n_proteins:,}")

    return corr_matrix, pval_matrix, n_samples_matrix

def identify_strong_edges(corr_matrix, pval_matrix, n_samples_matrix,
                          corr_threshold=0.7, pval_threshold=0.05):
    """
    Identify significant protein-protein edges for network.

    Returns: DataFrame of edges with correlation, p-value, etc.
    """
    print(f"\nIdentifying edges (|r| > {corr_threshold}, p < {pval_threshold})...")

    edges = []
    proteins = corr_matrix.index.tolist()

    for i, prot1 in enumerate(proteins):
        for j, prot2 in enumerate(proteins):
            if i >= j:  # Skip diagonal and duplicates
                continue

            corr = corr_matrix.loc[prot1, prot2]
            pval = pval_matrix.loc[prot1, prot2]
            n_samples = n_samples_matrix.loc[prot1, prot2]

            if pd.notna(corr) and abs(corr) >= corr_threshold and pval < pval_threshold:
                edges.append({
                    'Protein_A': prot1,
                    'Protein_B': prot2,
                    'Correlation': corr,
                    'P_value': pval,
                    'N_Samples': n_samples,
                    'Direction': 'Positive' if corr > 0 else 'Negative'
                })

    edges_df = pd.DataFrame(edges)

    if len(edges_df) > 0:
        print(f"  Total significant edges: {len(edges_df):,}")
        print(f"    Positive correlations: {(edges_df['Correlation'] > 0).sum():,}")
        print(f"    Negative correlations: {(edges_df['Correlation'] < 0).sum():,}")
        print(f"    Mean |r|: {edges_df['Correlation'].abs().mean():.3f}")
    else:
        print("  WARNING: No significant edges found!")

    return edges_df

# ============================================================================
# 3. NETWORK ANALYSIS
# ============================================================================

def calculate_node_statistics(edges_df):
    """Calculate degree centrality for each protein."""
    print("\nCalculating network statistics...")

    # Degree centrality (number of connections)
    from collections import Counter

    all_proteins = list(edges_df['Protein_A']) + list(edges_df['Protein_B'])
    degree_counts = Counter(all_proteins)

    node_stats = pd.DataFrame([
        {'Protein': prot, 'Degree': degree}
        for prot, degree in degree_counts.items()
    ]).sort_values('Degree', ascending=False)

    print(f"  Proteins in network: {len(node_stats)}")
    print(f"  Mean degree: {node_stats['Degree'].mean():.1f}")
    print(f"  Max degree: {node_stats['Degree'].max()}")

    return node_stats

def detect_communities_hierarchical(corr_matrix, edges_df, n_clusters=10):
    """
    Detect protein modules using hierarchical clustering.

    Uses proteins that have at least one significant edge.
    """
    print(f"\nDetecting communities (hierarchical clustering, k={n_clusters})...")

    # Get proteins in network
    network_proteins = set(edges_df['Protein_A']) | set(edges_df['Protein_B'])
    network_proteins = sorted(network_proteins)

    print(f"  Proteins in network: {len(network_proteins)}")

    # Extract submatrix
    sub_corr = corr_matrix.loc[network_proteins, network_proteins]

    # Convert correlation to distance (1 - |r|)
    distance_matrix = 1 - sub_corr.abs().fillna(0)

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix.values, checks=False)
    Z = linkage(condensed_dist, method='average')

    # Cut tree
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Create module assignments
    modules = pd.DataFrame({
        'Protein': network_proteins,
        'Module': cluster_labels
    })

    print(f"  Modules created: {modules['Module'].nunique()}")
    for mod_id in sorted(modules['Module'].unique()):
        mod_size = (modules['Module'] == mod_id).sum()
        print(f"    Module {mod_id}: {mod_size} proteins")

    return modules, Z

def annotate_modules_with_metadata(modules, original_df):
    """Add matrisome categories and other metadata to module assignments."""
    print("\nAnnotating modules with protein metadata...")

    # Get unique protein info
    protein_info = original_df.groupby('Gene_Symbol').agg({
        'Matrisome_Category': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
        'Matrisome_Division': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown',
        'Protein_Name': 'first'
    }).reset_index()

    # Merge
    modules_annotated = modules.merge(
        protein_info,
        left_on='Protein',
        right_on='Gene_Symbol',
        how='left'
    )

    return modules_annotated

def analyze_module_enrichment(modules_annotated):
    """Test for enrichment of matrisome categories in modules."""
    print("\nTesting module enrichment for matrisome categories...")

    enrichment_results = []

    for module_id in sorted(modules_annotated['Module'].unique()):
        module_proteins = modules_annotated[modules_annotated['Module'] == module_id]

        # Category distribution in module
        module_cats = module_proteins['Matrisome_Category'].value_counts()

        # Background distribution
        bg_cats = modules_annotated['Matrisome_Category'].value_counts()

        for cat in module_cats.index:
            n_module_cat = module_cats[cat]
            n_module_total = len(module_proteins)
            n_bg_cat = bg_cats[cat]
            n_bg_total = len(modules_annotated)

            # Fisher exact test
            from scipy.stats import fisher_exact

            contingency = [
                [n_module_cat, n_module_total - n_module_cat],
                [n_bg_cat, n_bg_total - n_bg_cat]
            ]

            oddsratio, pval = fisher_exact(contingency)

            enrichment_results.append({
                'Module': module_id,
                'Category': cat,
                'Count': n_module_cat,
                'Module_Size': n_module_total,
                'Fraction': n_module_cat / n_module_total,
                'Background_Fraction': n_bg_cat / n_bg_total,
                'Enrichment': (n_module_cat / n_module_total) / (n_bg_cat / n_bg_total),
                'P_value': pval
            })

    enrichment_df = pd.DataFrame(enrichment_results)

    # Multiple testing correction (Bonferroni)
    enrichment_df['P_value_adjusted'] = enrichment_df['P_value'] * len(enrichment_df)
    enrichment_df['P_value_adjusted'] = enrichment_df['P_value_adjusted'].clip(upper=1.0)

    # Filter significant
    sig_enrichment = enrichment_df[enrichment_df['P_value_adjusted'] < 0.05].sort_values('P_value')

    print(f"  Significant enrichments (p < 0.05): {len(sig_enrichment)}")

    return enrichment_df, sig_enrichment

# ============================================================================
# 4. HUB & BRIDGE ANALYSIS
# ============================================================================

def identify_hub_proteins(node_stats, edges_df, top_n=20):
    """Identify highly connected hub proteins."""
    print(f"\nIdentifying top {top_n} hub proteins...")

    hubs = node_stats.head(top_n).copy()

    # Add avg correlation strength
    hub_correlations = []
    for prot in hubs['Protein']:
        edges_involving = edges_df[
            (edges_df['Protein_A'] == prot) | (edges_df['Protein_B'] == prot)
        ]
        avg_corr = edges_involving['Correlation'].abs().mean()
        hub_correlations.append(avg_corr)

    hubs['Avg_Correlation_Strength'] = hub_correlations

    print("\nTop hub proteins:")
    for i, row in hubs.head(10).iterrows():
        print(f"  {row['Protein']}: degree={row['Degree']}, avg_r={row['Avg_Correlation_Strength']:.3f}")

    return hubs

def identify_bridge_proteins(modules_annotated, edges_df, min_external_edges=2):
    """
    Identify bridge proteins connecting different modules.

    Bridge = protein with edges to multiple modules.
    """
    print(f"\nIdentifying bridge proteins (min {min_external_edges} cross-module edges)...")

    # For each protein, count edges to other modules
    bridges = []

    for prot in modules_annotated['Protein']:
        prot_module = modules_annotated[modules_annotated['Protein'] == prot]['Module'].iloc[0]

        # Get all edges
        prot_edges = edges_df[
            (edges_df['Protein_A'] == prot) | (edges_df['Protein_B'] == prot)
        ].copy()

        # Get partners
        partners = []
        for _, edge in prot_edges.iterrows():
            partner = edge['Protein_B'] if edge['Protein_A'] == prot else edge['Protein_A']
            partners.append(partner)

        # Get partner modules
        partner_modules = modules_annotated[
            modules_annotated['Protein'].isin(partners)
        ]['Module'].values

        # Count external (different module) connections
        external_modules = [m for m in partner_modules if m != prot_module]
        n_external = len(external_modules)
        n_unique_external_modules = len(set(external_modules))

        if n_external >= min_external_edges:
            bridges.append({
                'Protein': prot,
                'Module': prot_module,
                'Total_Edges': len(prot_edges),
                'External_Edges': n_external,
                'External_Modules_Connected': n_unique_external_modules
            })

    bridges_df = pd.DataFrame(bridges).sort_values('External_Edges', ascending=False)

    print(f"  Bridge proteins found: {len(bridges_df)}")
    if len(bridges_df) > 0:
        print(f"  Top bridges:")
        for i, row in bridges_df.head(5).iterrows():
            print(f"    {row['Protein']}: {row['External_Edges']} edges to {row['External_Modules_Connected']} modules")

    return bridges_df

# ============================================================================
# 5. EXPORT & VISUALIZATION PREP
# ============================================================================

def export_network_for_gephi(edges_df, node_stats, modules_annotated, output_prefix):
    """Export network in Gephi-compatible format."""
    print(f"\nExporting network files (prefix: {output_prefix})...")

    # Nodes file
    nodes = modules_annotated.merge(
        node_stats[['Protein', 'Degree']],
        on='Protein',
        how='left'
    )
    nodes['Degree'] = nodes['Degree'].fillna(0)

    nodes_export = nodes[[
        'Protein', 'Module', 'Degree',
        'Matrisome_Category', 'Matrisome_Division', 'Protein_Name'
    ]].rename(columns={'Protein': 'Id', 'Protein_Name': 'Label'})

    nodes_file = f"{output_prefix}_nodes.csv"
    nodes_export.to_csv(nodes_file, index=False)
    print(f"  Nodes: {nodes_file} ({len(nodes_export)} nodes)")

    # Edges file
    edges_export = edges_df[[
        'Protein_A', 'Protein_B', 'Correlation', 'P_value', 'Direction'
    ]].rename(columns={
        'Protein_A': 'Source',
        'Protein_B': 'Target',
        'Correlation': 'Weight'
    })

    edges_file = f"{output_prefix}_edges.csv"
    edges_export.to_csv(edges_file, index=False)
    print(f"  Edges: {edges_file} ({len(edges_export)} edges)")

    return nodes_file, edges_file

def create_summary_statistics(df, corr_matrix, edges_df, node_stats,
                              modules_annotated, enrichment_df, hubs, bridges):
    """Create comprehensive summary statistics dictionary."""

    summary = {
        'dataset': {
            'total_records': len(df),
            'unique_proteins': df['Gene_Symbol'].nunique(),
            'tissues': df['Tissue'].nunique(),
            'studies': df['Study_ID'].nunique(),
        },
        'network': {
            'proteins_in_network': len(node_stats),
            'total_edges': len(edges_df),
            'positive_correlations': (edges_df['Correlation'] > 0).sum(),
            'negative_correlations': (edges_df['Correlation'] < 0).sum(),
            'mean_correlation': edges_df['Correlation'].abs().mean(),
            'mean_degree': node_stats['Degree'].mean(),
            'max_degree': node_stats['Degree'].max(),
        },
        'modules': {
            'n_modules': modules_annotated['Module'].nunique(),
            'module_sizes': modules_annotated.groupby('Module').size().to_dict(),
            'significant_enrichments': len(enrichment_df[enrichment_df['P_value_adjusted'] < 0.05]),
        },
        'hubs': {
            'n_hubs': len(hubs),
            'top_hub': hubs.iloc[0]['Protein'] if len(hubs) > 0 else None,
            'top_hub_degree': hubs.iloc[0]['Degree'] if len(hubs) > 0 else None,
        },
        'bridges': {
            'n_bridges': len(bridges),
            'top_bridge': bridges.iloc[0]['Protein'] if len(bridges) > 0 else None,
            'top_bridge_connections': bridges.iloc[0]['External_Modules_Connected'] if len(bridges) > 0 else None,
        }
    }

    return summary

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Run complete network topology analysis pipeline."""

    print("="*80)
    print("ECM-ATLAS NETWORK TOPOLOGY ANALYSIS")
    print("Agent 8: Protein Co-expression Network Mapper")
    print("="*80)

    # Configuration
    DATA_FILE = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
    OUTPUT_PREFIX = "/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_08_network"

    CORRELATION_METHOD = 'spearman'  # or 'pearson'
    CORRELATION_THRESHOLD = 0.7
    PVAL_THRESHOLD = 0.05
    MIN_SHARED_SAMPLES = 3
    N_CLUSTERS = 8

    # 1. Load data
    df = load_ecm_data(DATA_FILE)

    # 2. Prepare matrix
    matrix = prepare_correlation_matrix(df)

    # 3. Calculate correlations
    corr_matrix, pval_matrix, n_samples_matrix = calculate_protein_correlations(
        matrix,
        min_shared_samples=MIN_SHARED_SAMPLES,
        method=CORRELATION_METHOD
    )

    # 4. Identify edges
    edges_df = identify_strong_edges(
        corr_matrix, pval_matrix, n_samples_matrix,
        corr_threshold=CORRELATION_THRESHOLD,
        pval_threshold=PVAL_THRESHOLD
    )

    if len(edges_df) == 0:
        print("\nERROR: No significant edges found. Try lowering thresholds.")
        return

    # 5. Network statistics
    node_stats = calculate_node_statistics(edges_df)

    # 6. Community detection
    modules, linkage_matrix = detect_communities_hierarchical(
        corr_matrix, edges_df, n_clusters=N_CLUSTERS
    )
    modules_annotated = annotate_modules_with_metadata(modules, df)

    # 7. Module enrichment
    enrichment_df, sig_enrichment = analyze_module_enrichment(modules_annotated)

    # 8. Hub proteins
    hubs = identify_hub_proteins(node_stats, edges_df, top_n=20)

    # 9. Bridge proteins
    bridges = identify_bridge_proteins(modules_annotated, edges_df, min_external_edges=2)

    # 10. Export network
    nodes_file, edges_file = export_network_for_gephi(
        edges_df, node_stats, modules_annotated, OUTPUT_PREFIX
    )

    # 11. Summary statistics
    summary = create_summary_statistics(
        df, corr_matrix, edges_df, node_stats,
        modules_annotated, enrichment_df, hubs, bridges
    )

    # 12. Save all results
    print("\nSaving analysis results...")

    # Correlation matrix
    corr_matrix.to_csv(f"{OUTPUT_PREFIX}_correlation_matrix.csv")
    print(f"  Correlation matrix: {OUTPUT_PREFIX}_correlation_matrix.csv")

    # Module assignments
    modules_annotated.to_csv(f"{OUTPUT_PREFIX}_module_assignments.csv", index=False)
    print(f"  Module assignments: {OUTPUT_PREFIX}_module_assignments.csv")

    # Enrichment results
    enrichment_df.to_csv(f"{OUTPUT_PREFIX}_module_enrichment.csv", index=False)
    sig_enrichment.to_csv(f"{OUTPUT_PREFIX}_significant_enrichments.csv", index=False)
    print(f"  Enrichment: {OUTPUT_PREFIX}_module_enrichment.csv")

    # Hub proteins
    hubs.to_csv(f"{OUTPUT_PREFIX}_hub_proteins.csv", index=False)
    print(f"  Hubs: {OUTPUT_PREFIX}_hub_proteins.csv")

    # Bridge proteins
    bridges.to_csv(f"{OUTPUT_PREFIX}_bridge_proteins.csv", index=False)
    print(f"  Bridges: {OUTPUT_PREFIX}_bridge_proteins.csv")

    # Summary JSON
    import json
    with open(f"{OUTPUT_PREFIX}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {OUTPUT_PREFIX}_summary.json")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"  - Network proteins: {summary['network']['proteins_in_network']}")
    print(f"  - Network edges: {summary['network']['total_edges']}")
    print(f"  - Detected modules: {summary['modules']['n_modules']}")
    print(f"  - Hub proteins: {summary['hubs']['n_hubs']}")
    print(f"  - Bridge proteins: {summary['bridges']['n_bridges']}")
    print(f"\nTop hub: {summary['hubs']['top_hub']} (degree={summary['hubs']['top_hub_degree']})")
    if summary['bridges']['top_bridge']:
        print(f"Top bridge: {summary['bridges']['top_bridge']} (connects {summary['bridges']['top_bridge_connections']} modules)")

    return summary, modules_annotated, hubs, bridges, enrichment_df

if __name__ == "__main__":
    summary, modules, hubs, bridges, enrichment = main()
