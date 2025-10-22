#!/usr/bin/env python3
"""
In Silico Knockout/Perturbation Analysis for Serpin Centrality Validation (H14)

Simulates removal of each serpin protein from network,
measures network impact, correlates with centrality metrics
to identify which metric best predicts functional importance.

Agent: claude_code
Created: 2025-10-21
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code")
VIZ_DIR = OUTPUT_DIR / "visualizations_claude_code"

print("="*80)
print("H14: IN SILICO KNOCKOUT ANALYSIS")
print("="*80)

# Load network and centrality data
print("\n[6/9] Loading network and centrality data...")
edges_df = pd.read_csv(OUTPUT_DIR / "network_edges_claude_code.csv")
centrality_df = pd.read_csv(OUTPUT_DIR / "centrality_all_metrics_claude_code.csv")

# Reconstruct graph
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['Protein_A'], row['Protein_B'], weight=row['Abs_Correlation'])

# Get largest connected component
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
else:
    G_main = G.copy()

print(f"  Network loaded: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges")

# Identify serpins in network
SERPINS = [
    'SERPINA1', 'SERPINA3', 'SERPINA5', 'SERPINB1', 'SERPINB6', 'SERPINB8',
    'SERPINC1', 'SERPINE1', 'SERPINE2', 'SERPINF1', 'SERPINF2', 'SERPING1', 'SERPINH1'
]
serpins_in_network = [s for s in SERPINS if s in G_main.nodes()]
print(f"  Serpins in network: {len(serpins_in_network)}/{len(SERPINS)}")
print(f"    Present: {serpins_in_network}")

# =============================================================================
# BASELINE NETWORK METRICS
# =============================================================================

print("\n  Computing baseline network metrics...")

def compute_network_metrics(graph):
    """Compute comprehensive network topology metrics."""
    metrics = {}

    # Connectivity
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    metrics['density'] = nx.density(graph)

    # Components
    if nx.is_connected(graph):
        metrics['num_components'] = 1
        metrics['largest_component_size'] = graph.number_of_nodes()
        metrics['largest_component_fraction'] = 1.0
        # Connectivity for connected graph
        metrics['node_connectivity'] = nx.node_connectivity(graph) if graph.number_of_nodes() > 1 else 0
        metrics['edge_connectivity'] = nx.edge_connectivity(graph) if graph.number_of_nodes() > 1 else 0
    else:
        components = list(nx.connected_components(graph))
        metrics['num_components'] = len(components)
        metrics['largest_component_size'] = len(max(components, key=len))
        metrics['largest_component_fraction'] = metrics['largest_component_size'] / graph.number_of_nodes()
        metrics['node_connectivity'] = 0
        metrics['edge_connectivity'] = 0

    # Clustering
    metrics['avg_clustering'] = nx.average_clustering(graph, weight='weight')

    # Path length (only for connected graph)
    if nx.is_connected(graph):
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(graph, weight='weight')
    else:
        # Average within largest component
        largest_comp = graph.subgraph(max(nx.connected_components(graph), key=len))
        if largest_comp.number_of_nodes() > 1:
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(largest_comp, weight='weight')
        else:
            metrics['avg_shortest_path'] = 0

    # Degree statistics
    degrees = [d for n, d in graph.degree()]
    metrics['avg_degree'] = np.mean(degrees)
    metrics['max_degree'] = np.max(degrees) if degrees else 0

    return metrics

baseline_metrics = compute_network_metrics(G_main)
print("  Baseline network metrics:")
for key, value in baseline_metrics.items():
    print(f"    {key:30s}: {value:.4f}" if isinstance(value, float) else f"    {key:30s}: {value}")

# =============================================================================
# KNOCKOUT SIMULATIONS
# =============================================================================

print(f"\n  Performing knockout simulations for {len(serpins_in_network)} serpins...")

knockout_results = []

for i, serpin in enumerate(serpins_in_network, 1):
    print(f"  [{i}/{len(serpins_in_network)}] Simulating {serpin} knockout...")

    # Create knockout graph
    G_knockout = G_main.copy()
    neighbors = list(G_knockout.neighbors(serpin))
    num_neighbors = len(neighbors)

    G_knockout.remove_node(serpin)

    # Compute metrics after knockout
    ko_metrics = compute_network_metrics(G_knockout)

    # Calculate deltas
    result = {
        'Protein': serpin,
        'Original_Degree': num_neighbors,
        # Network impact metrics
        'Delta_Nodes': baseline_metrics['num_nodes'] - ko_metrics['num_nodes'],
        'Delta_Edges': baseline_metrics['num_edges'] - ko_metrics['num_edges'],
        'Delta_Density': baseline_metrics['density'] - ko_metrics['density'],
        'Delta_Num_Components': ko_metrics['num_components'] - baseline_metrics['num_components'],
        'Delta_Largest_Component_Size': baseline_metrics['largest_component_size'] - ko_metrics['largest_component_size'],
        'Delta_Largest_Component_Fraction': baseline_metrics['largest_component_fraction'] - ko_metrics['largest_component_fraction'],
        'Delta_Node_Connectivity': baseline_metrics['node_connectivity'] - ko_metrics['node_connectivity'],
        'Delta_Edge_Connectivity': baseline_metrics['edge_connectivity'] - ko_metrics['edge_connectivity'],
        'Delta_Avg_Clustering': baseline_metrics['avg_clustering'] - ko_metrics['avg_clustering'],
        'Delta_Avg_Shortest_Path': ko_metrics['avg_shortest_path'] - baseline_metrics['avg_shortest_path'],  # Increase is bad
        'Delta_Avg_Degree': baseline_metrics['avg_degree'] - ko_metrics['avg_degree'],
        'Delta_Max_Degree': baseline_metrics['max_degree'] - ko_metrics['max_degree'],
    }

    # Composite impact score (weighted sum of normalized deltas)
    # Higher score = more important protein
    result['Impact_Score'] = (
        result['Delta_Edges'] * 0.3 +  # Edge loss
        abs(result['Delta_Num_Components']) * 20 +  # Component fragmentation
        result['Delta_Largest_Component_Fraction'] * 100 +  # Main component reduction
        abs(result['Delta_Avg_Clustering']) * 50 +  # Clustering change
        abs(result['Delta_Node_Connectivity']) * 10  # Connectivity loss
    )

    knockout_results.append(result)

    print(f"      Impact score: {result['Impact_Score']:.3f} | "
          f"Lost edges: {result['Delta_Edges']} | "
          f"Δ components: {result['Delta_Num_Components']}")

knockout_df = pd.DataFrame(knockout_results)
knockout_df = knockout_df.sort_values('Impact_Score', ascending=False).reset_index(drop=True)
knockout_df.to_csv(OUTPUT_DIR / "knockout_impact_claude_code.csv", index=False)

print("\n  Top 5 serpins by knockout impact:")
print(knockout_df[['Protein', 'Impact_Score', 'Delta_Edges', 'Delta_Num_Components', 'Original_Degree']].head())

# =============================================================================
# CORRELATE CENTRALITY WITH KNOCKOUT IMPACT
# =============================================================================

print(f"\n[7/9] Correlating centrality metrics with knockout impact...")

# Merge centrality scores with knockout results
merged = knockout_df.merge(centrality_df, on='Protein', how='left')

# Compute correlations
metrics_to_test = ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']
correlation_results = []

print("\n  Centrality metric vs. knockout impact correlations:")
print("  " + "="*70)

for metric in metrics_to_test:
    # Spearman correlation (rank-based)
    rho_spearman, p_spearman = spearmanr(
        merged[metric].fillna(0),
        merged['Impact_Score']
    )

    # Pearson correlation (linear)
    rho_pearson, p_pearson = pearsonr(
        merged[metric].fillna(0),
        merged['Impact_Score']
    )

    correlation_results.append({
        'Metric': metric,
        'Spearman_rho': rho_spearman,
        'Spearman_p': p_spearman,
        'Pearson_r': rho_pearson,
        'Pearson_p': p_pearson,
        'Best_By_Spearman': False,
        'Best_By_Pearson': False
    })

    print(f"    {metric:15s}: Spearman ρ={rho_spearman:6.3f} (p={p_spearman:.4f}) | "
          f"Pearson r={rho_pearson:6.3f} (p={p_pearson:.4f})")

correlation_results_df = pd.DataFrame(correlation_results)

# Mark best metrics
best_spearman_idx = correlation_results_df['Spearman_rho'].idxmax()
best_pearson_idx = correlation_results_df['Pearson_r'].idxmax()
correlation_results_df.loc[best_spearman_idx, 'Best_By_Spearman'] = True
correlation_results_df.loc[best_pearson_idx, 'Best_By_Pearson'] = True

correlation_results_df.to_csv(OUTPUT_DIR / "centrality_knockout_correlation_claude_code.csv", index=False)

print("\n  WINNER:")
best_metric_spearman = correlation_results_df.loc[best_spearman_idx, 'Metric']
best_rho = correlation_results_df.loc[best_spearman_idx, 'Spearman_rho']
print(f"    Best metric (Spearman): {best_metric_spearman} (ρ={best_rho:.3f})")
best_metric_pearson = correlation_results_df.loc[best_pearson_idx, 'Metric']
best_r = correlation_results_df.loc[best_pearson_idx, 'Pearson_r']
print(f"    Best metric (Pearson):  {best_metric_pearson} (r={best_r:.3f})")

print("\n  H02 RESOLUTION:")
betweenness_rho = correlation_results_df[correlation_results_df['Metric'] == 'Betweenness']['Spearman_rho'].values[0]
eigenvector_rho = correlation_results_df[correlation_results_df['Metric'] == 'Eigenvector']['Spearman_rho'].values[0]
print(f"    Claude (Betweenness): ρ={betweenness_rho:.3f} with knockout impact")
print(f"    Codex (Eigenvector):  ρ={eigenvector_rho:.3f} with knockout impact")

if betweenness_rho > eigenvector_rho:
    print(f"    → CLAUDE WAS CORRECT: Betweenness better predicts network impact!")
elif eigenvector_rho > betweenness_rho:
    print(f"    → CODEX WAS CORRECT: Eigenvector better predicts network impact!")
else:
    print(f"    → TIE: Both metrics equally predictive")

# =============================================================================
# EXPERIMENTAL VALIDATION CROSS-REFERENCE
# =============================================================================

print(f"\n  Cross-referencing with experimental literature...")

# Manual curation from literature search
experimental_importance = {
    'SERPINE1': {'Severity': 'Beneficial', 'Score': -10, 'Evidence': 'Knockout +7yr lifespan, protective'},
    'SERPINC1': {'Severity': 'Moderate', 'Score': 5, 'Evidence': 'Null mutation 66% VTE-free survival'},
    'SERPINF1': {'Severity': 'Unknown', 'Score': 0, 'Evidence': 'Limited knockout data'},
    'SERPINF2': {'Severity': 'Unknown', 'Score': 0, 'Evidence': 'Limited knockout data'},
    'SERPING1': {'Severity': 'Moderate', 'Score': 5, 'Evidence': 'C1-inhibitor deficiency (angioedema)'},
    'SERPINH1': {'Severity': 'Mild', 'Score': 2, 'Evidence': 'Collagen chaperone, not lethal'},
}

# Add experimental scores to knockout dataframe
knockout_df['Experimental_Score'] = knockout_df['Protein'].map(
    lambda x: experimental_importance.get(x, {}).get('Score', 0)
)
knockout_df['Experimental_Evidence'] = knockout_df['Protein'].map(
    lambda x: experimental_importance.get(x, {}).get('Evidence', 'No data')
)

# Save experimental validation
experimental_df = pd.DataFrame([
    {'Protein': k, **v} for k, v in experimental_importance.items()
])
experimental_df.to_csv(OUTPUT_DIR / "experimental_validation_claude_code.csv", index=False)

print("  Experimental phenotype severity (literature-derived):")
for protein, data in experimental_importance.items():
    print(f"    {protein:12s}: {data['Severity']:12s} (Score: {data['Score']:+3d}) - {data['Evidence']}")

# Correlate experimental score with centrality metrics
print("\n  Centrality vs. experimental severity correlations:")
for metric in metrics_to_test:
    # Get centrality scores for proteins with experimental data
    exp_proteins = list(experimental_importance.keys())
    exp_scores = [experimental_importance[p]['Score'] for p in exp_proteins]
    centrality_scores = [
        centrality_df[centrality_df['Protein'] == p][metric].values[0]
        if p in centrality_df['Protein'].values else 0
        for p in exp_proteins
    ]

    if len(exp_scores) > 2:
        rho, p = spearmanr(centrality_scores, exp_scores)
        print(f"    {metric:15s}: ρ={rho:6.3f} (p={p:.4f})")

print("\n  Knockout analysis complete!")
