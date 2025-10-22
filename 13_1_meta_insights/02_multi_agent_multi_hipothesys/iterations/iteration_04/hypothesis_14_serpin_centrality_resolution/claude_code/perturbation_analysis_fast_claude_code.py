#!/usr/bin/env python3
"""
FAST In Silico Knockout Analysis for H14 (Optimized for Large Networks)

Simulates serpin knockouts and correlates centrality metrics with network impact.
Skips expensive connectivity calculations for speed.

Agent: claude_code
Created: 2025-10-21
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code")

print("="*80)
print("H14: IN SILICO KNOCKOUT ANALYSIS (FAST VERSION)")
print("="*80)

# Load data
print("\n[6/9] Loading network and centrality data...")
edges_df = pd.read_csv(OUTPUT_DIR / "network_edges_claude_code.csv")
centrality_df = pd.read_csv(OUTPUT_DIR / "centrality_all_metrics_claude_code.csv")

# Reconstruct graph
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['Protein_A'], row['Protein_B'], weight=row['Abs_Correlation'])

print(f"  Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Serpins
SERPINS = [
    'SERPINA1', 'SERPINA3', 'SERPINA5', 'SERPINB1', 'SERPINB6', 'SERPINB8',
    'SERPINC1', 'SERPINE1', 'SERPINE2', 'SERPINF1', 'SERPINF2', 'SERPING1', 'SERPINH1'
]
serpins_in_network = [s for s in SERPINS if s in G.nodes()]
print(f"  Serpins in network: {len(serpins_in_network)}")

# Baseline metrics (fast)
print("\n  Computing baseline network metrics (fast mode)...")
baseline = {
    'num_nodes': G.number_of_nodes(),
    'num_edges': G.number_of_edges(),
    'density': nx.density(G),
    'num_components': nx.number_connected_components(G),
    'avg_clustering': nx.average_clustering(G, weight='weight'),
    'avg_degree': np.mean([d for n, d in G.degree()])
}

# Get largest component
components = list(nx.connected_components(G))
baseline['largest_component_size'] = len(max(components, key=len))

print(f"    Nodes: {baseline['num_nodes']}, Edges: {baseline['num_edges']}")
print(f"    Density: {baseline['density']:.4f}, Components: {baseline['num_components']}")

# Knockout simulations
print(f"\n  Performing {len(serpins_in_network)} knockout simulations...")

knockout_results = []

for i, serpin in enumerate(serpins_in_network, 1):
    print(f"    [{i}/{len(serpins_in_network)}] {serpin}...")

    # Create knockout graph
    G_ko = G.copy()
    neighbors = list(G_ko.neighbors(serpin))
    degree_before = len(neighbors)

    G_ko.remove_node(serpin)

    # Fast metrics only
    num_edges_ko = G_ko.number_of_edges()
    num_components_ko = nx.number_connected_components(G_ko)
    density_ko = nx.density(G_ko)
    avg_clustering_ko = nx.average_clustering(G_ko, weight='weight')
    avg_degree_ko = np.mean([d for n, d in G_ko.degree()])

    components_ko = list(nx.connected_components(G_ko))
    largest_comp_size_ko = len(max(components_ko, key=len))

    # Deltas
    delta_edges = baseline['num_edges'] - num_edges_ko
    delta_components = num_components_ko - baseline['num_components']
    delta_largest_comp = baseline['largest_component_size'] - largest_comp_size_ko
    delta_density = baseline['density'] - density_ko
    delta_clustering = baseline['avg_clustering'] - avg_clustering_ko
    delta_degree = baseline['avg_degree'] - avg_degree_ko

    # Impact score (weighted sum of important deltas)
    impact_score = (
        delta_edges * 0.4 +  # Lost edges (direct connections)
        abs(delta_components) * 50 +  # Network fragmentation
        delta_largest_comp * 1.0 +  # Main component shrinkage
        abs(delta_clustering) * 100 +  # Clustering change
        delta_degree * 10  # Average degree reduction
    )

    knockout_results.append({
        'Protein': serpin,
        'Original_Degree': degree_before,
        'Delta_Edges': delta_edges,
        'Delta_Num_Components': delta_components,
        'Delta_Largest_Component_Size': delta_largest_comp,
        'Delta_Density': delta_density,
        'Delta_Avg_Clustering': delta_clustering,
        'Delta_Avg_Degree': delta_degree,
        'Impact_Score': impact_score
    })

knockout_df = pd.DataFrame(knockout_results)
knockout_df = knockout_df.sort_values('Impact_Score', ascending=False).reset_index(drop=True)
knockout_df.to_csv(OUTPUT_DIR / "knockout_impact_claude_code.csv", index=False)

print("\n  Top 5 serpins by knockout impact:")
print(knockout_df[['Protein', 'Impact_Score', 'Delta_Edges', 'Original_Degree']].head().to_string(index=False))

# =============================================================================
# CORRELATE CENTRALITY WITH KNOCKOUT IMPACT
# =============================================================================

print(f"\n[7/9] Correlating centrality metrics with knockout impact...")

merged = knockout_df.merge(centrality_df, on='Protein', how='left')

metrics_to_test = ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']
correlation_results = []

print("\n  Centrality metric vs. knockout impact correlations:")
print("  " + "="*70)

for metric in metrics_to_test:
    rho_spearman, p_spearman = spearmanr(
        merged[metric].fillna(0),
        merged['Impact_Score']
    )
    rho_pearson, p_pearson = pearsonr(
        merged[metric].fillna(0),
        merged['Impact_Score']
    )

    correlation_results.append({
        'Metric': metric,
        'Spearman_rho': rho_spearman,
        'Spearman_p': p_spearman,
        'Pearson_r': rho_pearson,
        'Pearson_p': p_pearson
    })

    print(f"    {metric:15s}: Spearman ρ={rho_spearman:6.3f} (p={p_spearman:.4f}) | "
          f"Pearson r={rho_pearson:6.3f} (p={p_pearson:.4f})")

correlation_results_df = pd.DataFrame(correlation_results)
correlation_results_df.to_csv(OUTPUT_DIR / "centrality_knockout_correlation_claude_code.csv", index=False)

# Best metrics
best_spearman_idx = correlation_results_df['Spearman_rho'].idxmax()
best_pearson_idx = correlation_results_df['Pearson_r'].idxmax()

print("\n  WINNER:")
best_metric_spearman = correlation_results_df.loc[best_spearman_idx, 'Metric']
best_rho = correlation_results_df.loc[best_spearman_idx, 'Spearman_rho']
print(f"    Best metric (Spearman): {best_metric_spearman} (ρ={best_rho:.3f})")

best_metric_pearson = correlation_results_df.loc[best_pearson_idx, 'Metric']
best_r = correlation_results_df.loc[best_pearson_idx, 'Pearson_r']
print(f"    Best metric (Pearson):  {best_metric_pearson} (r={best_r:.3f})")

# H02 Resolution
print("\n  H02 RESOLUTION:")
betweenness_rho = correlation_results_df[correlation_results_df['Metric'] == 'Betweenness']['Spearman_rho'].values[0]
eigenvector_rho = correlation_results_df[correlation_results_df['Metric'] == 'Eigenvector']['Spearman_rho'].values[0]

print(f"    Claude (Betweenness): ρ={betweenness_rho:.3f} with knockout impact")
print(f"    Codex (Eigenvector):  ρ={eigenvector_rho:.3f} with knockout impact")

if betweenness_rho > eigenvector_rho:
    winner = "CLAUDE"
    print(f"    → {winner} WAS CORRECT: Betweenness better predicts network impact!")
elif eigenvector_rho > betweenness_rho:
    winner = "CODEX"
    print(f"    → {winner} WAS CORRECT: Eigenvector better predicts network impact!")
else:
    winner = "TIE"
    print(f"    → {winner}: Both metrics equally predictive")

# =============================================================================
# EXPERIMENTAL VALIDATION
# =============================================================================

print(f"\n  Cross-referencing with experimental literature...")

experimental_importance = {
    'SERPINE1': {'Severity': 'Beneficial', 'Score': -10, 'Evidence': 'Knockout +7yr lifespan, protective'},
    'SERPINC1': {'Severity': 'Moderate', 'Score': 5, 'Evidence': 'Null mutation 66% VTE-free survival'},
    'SERPINF1': {'Severity': 'Unknown', 'Score': 0, 'Evidence': 'Limited knockout data'},
    'SERPINF2': {'Severity': 'Unknown', 'Score': 0, 'Evidence': 'Limited knockout data'},
    'SERPING1': {'Severity': 'Moderate', 'Score': 5, 'Evidence': 'C1-inhibitor deficiency (angioedema)'},
    'SERPINH1': {'Severity': 'Mild', 'Score': 2, 'Evidence': 'Collagen chaperone, not lethal'},
}

experimental_df = pd.DataFrame([
    {'Protein': k, **v} for k, v in experimental_importance.items()
])
experimental_df.to_csv(OUTPUT_DIR / "experimental_validation_claude_code.csv", index=False)

print("  Experimental phenotype severity (literature-derived):")
for protein, data in experimental_importance.items():
    print(f"    {protein:12s}: {data['Severity']:12s} (Score: {data['Score']:+3d}) - {data['Evidence']}")

print("\n  Knockout analysis complete!")
print("="*80)
