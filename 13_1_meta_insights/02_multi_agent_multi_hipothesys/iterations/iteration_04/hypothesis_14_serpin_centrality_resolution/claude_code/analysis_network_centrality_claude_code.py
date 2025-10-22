#!/usr/bin/env python3
"""
Comprehensive Network Centrality Analysis for Serpin Centrality Resolution (H14)

Computes 7 centrality metrics, validates with knockout simulations,
resolves H02 disagreement (Claude betweenness vs Codex eigenvector).

Agent: claude_code
Created: 2025-10-21
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code")
VIZ_DIR = OUTPUT_DIR / "visualizations_claude_code"
VIZ_DIR.mkdir(exist_ok=True, parents=True)

# Network construction parameters (matching H02/H05)
CORRELATION_THRESHOLD = 0.5
P_VALUE_THRESHOLD = 0.05

# Serpin family proteins
SERPINS = [
    'SERPINA1', 'SERPINA3', 'SERPINA5', 'SERPINB1', 'SERPINB6', 'SERPINB8',
    'SERPINC1', 'SERPINE1', 'SERPINE2', 'SERPINF1', 'SERPINF2', 'SERPING1', 'SERPINH1'
]

print("="*80)
print("H14: SERPIN NETWORK CENTRALITY RESOLUTION")
print("="*80)
print(f"Agent: claude_code")
print(f"Dataset: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("="*80)

# =============================================================================
# 2. LOAD AND PREPARE DATA
# =============================================================================

print("\n[1/9] Loading ECM aging dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows")
print(f"  Columns: {list(df.columns)}")

# Check for Z_score column (may vary by dataset version)
zscore_col = 'Z_score' if 'Z_score' in df.columns else 'Zscore_Delta'
if zscore_col not in df.columns:
    # Calculate z-scores if not present
    print("  Z-scores not found, calculating from Abundance...")
    df['Z_score'] = df.groupby(['Gene_Symbol', 'Tissue'])['Abundance'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    zscore_col = 'Z_score'

print(f"  Using z-score column: {zscore_col}")

# Create protein × tissue matrix for correlation
print("  Creating protein × tissue pivot matrix...")
pivot = df.pivot_table(
    values=zscore_col,
    index='Gene_Symbol',
    columns='Tissue',
    aggfunc='mean'  # Average if multiple studies per tissue
)

print(f"  Matrix shape: {pivot.shape[0]} proteins × {pivot.shape[1]} tissues")
print(f"  Missing values: {pivot.isna().sum().sum()} ({pivot.isna().sum().sum() / pivot.size * 100:.1f}%)")

# Fill NaN with 0 for correlation calculation
pivot_filled = pivot.fillna(0)

# =============================================================================
# 3. NETWORK RECONSTRUCTION (CORRELATION-BASED)
# =============================================================================

print(f"\n[2/9] Reconstructing protein correlation network...")
print(f"  Correlation threshold: |ρ| > {CORRELATION_THRESHOLD}")
print(f"  P-value threshold: p < {P_VALUE_THRESHOLD}")

# Compute pairwise Spearman correlations
proteins = pivot_filled.index.tolist()
edges = []
correlations_dict = {}

print(f"  Computing {len(proteins) * (len(proteins)-1) // 2:,} pairwise correlations...")
for i in range(len(proteins)):
    if i % 100 == 0:
        print(f"    Progress: {i}/{len(proteins)} proteins...")
    for j in range(i+1, len(proteins)):
        p1, p2 = proteins[i], proteins[j]
        vec1 = pivot_filled.loc[p1].values
        vec2 = pivot_filled.loc[p2].values

        # Spearman correlation
        rho, p = spearmanr(vec1, vec2)

        # Filter by threshold
        if abs(rho) > CORRELATION_THRESHOLD and p < P_VALUE_THRESHOLD:
            edges.append({
                'Protein_A': p1,
                'Protein_B': p2,
                'Correlation': rho,
                'P_value': p,
                'Abs_Correlation': abs(rho)
            })
            correlations_dict[(p1, p2)] = rho

print(f"  Total edges passing threshold: {len(edges):,}")

# Create NetworkX graph
G = nx.Graph()
for edge in edges:
    G.add_edge(
        edge['Protein_A'],
        edge['Protein_B'],
        weight=edge['Abs_Correlation'],
        correlation=edge['Correlation']
    )

# Save edges
edges_df = pd.DataFrame(edges)
edges_df.to_csv(OUTPUT_DIR / "network_edges_claude_code.csv", index=False)
print(f"  Saved edges to network_edges_claude_code.csv")

# Network statistics
print(f"\n  Network Statistics:")
print(f"    Nodes: {G.number_of_nodes()}")
print(f"    Edges: {G.number_of_edges()}")
print(f"    Density: {nx.density(G):.4f}")
print(f"    Connected components: {nx.number_connected_components(G)}")

# Get largest connected component
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    print(f"    Largest component: {G_main.number_of_nodes()} nodes ({G_main.number_of_nodes()/G.number_of_nodes()*100:.1f}%)")
else:
    G_main = G.copy()
    print(f"    Network is fully connected")

# Calculate average clustering for connected component
avg_clustering = nx.average_clustering(G_main, weight='weight')
print(f"    Average clustering coefficient: {avg_clustering:.4f}")

# Save network stats
network_stats = {
    'nodes': G.number_of_nodes(),
    'edges': G.number_of_edges(),
    'density': nx.density(G),
    'connected_components': nx.number_connected_components(G),
    'largest_component_nodes': G_main.number_of_nodes(),
    'average_clustering': avg_clustering,
    'correlation_threshold': CORRELATION_THRESHOLD,
    'p_value_threshold': P_VALUE_THRESHOLD
}

with open(OUTPUT_DIR / "network_stats_claude_code.json", 'w') as f:
    json.dump(network_stats, f, indent=2)

# =============================================================================
# 4. CENTRALITY METRICS COMPUTATION (7 METRICS)
# =============================================================================

print(f"\n[3/9] Computing 7 centrality metrics...")

centrality_results = {}

# Use largest connected component for centrality calculations
# (some metrics require connected graph)

# Metric 1: Degree Centrality
print("  [1/7] Degree centrality...")
centrality_results['Degree'] = nx.degree_centrality(G_main)

# Metric 2: Betweenness Centrality (Claude H02 method)
print("  [2/7] Betweenness centrality...")
centrality_results['Betweenness'] = nx.betweenness_centrality(G_main, weight='weight')

# Metric 3: Eigenvector Centrality (Codex H02 method)
print("  [3/7] Eigenvector centrality...")
try:
    centrality_results['Eigenvector'] = nx.eigenvector_centrality(
        G_main, weight='weight', max_iter=1000
    )
except:
    print("    Warning: Eigenvector failed to converge, using power iteration...")
    centrality_results['Eigenvector'] = nx.eigenvector_centrality_numpy(G_main, weight='weight')

# Metric 4: Closeness Centrality
print("  [4/7] Closeness centrality...")
centrality_results['Closeness'] = nx.closeness_centrality(G_main, distance='weight')

# Metric 5: PageRank
print("  [5/7] PageRank...")
centrality_results['PageRank'] = nx.pagerank(G_main, weight='weight')

# Metric 6: Katz Centrality
print("  [6/7] Katz centrality...")
try:
    # Calculate alpha as 1/(max eigenvalue) for convergence
    eigenvalues = np.linalg.eigvalsh(nx.adjacency_matrix(G_main).todense())
    alpha = 0.9 / max(abs(eigenvalues))
    centrality_results['Katz'] = nx.katz_centrality(
        G_main, alpha=alpha, weight='weight', max_iter=1000
    )
except:
    print("    Warning: Katz failed, using numpy version...")
    centrality_results['Katz'] = nx.katz_centrality_numpy(G_main, alpha=0.01, weight='weight')

# Metric 7: Subgraph Centrality
print("  [7/7] Subgraph centrality...")
centrality_results['Subgraph'] = nx.subgraph_centrality(G_main)

# Combine into DataFrame
print("\n  Creating centrality matrix...")
centrality_df = pd.DataFrame(centrality_results)
centrality_df.index.name = 'Protein'
centrality_df = centrality_df.reset_index()

print(f"  Centrality matrix shape: {centrality_df.shape}")
print(f"  Proteins with centrality scores: {len(centrality_df)}")

# Save centrality scores
centrality_df.to_csv(OUTPUT_DIR / "centrality_all_metrics_claude_code.csv", index=False)
print(f"  Saved to centrality_all_metrics_claude_code.csv")

# Display sample
print("\n  Sample centrality scores:")
print(centrality_df.head(10).to_string(index=False))

# =============================================================================
# 5. SERPIN FAMILY RANKING
# =============================================================================

print(f"\n[4/9] Ranking serpin family proteins...")
print(f"  Serpin proteins to analyze: {len(SERPINS)}")

serpin_rankings = []

for metric in ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']:
    # Rank all proteins by this metric
    centrality_df_sorted = centrality_df.sort_values(metric, ascending=False).reset_index(drop=True)
    centrality_df_sorted['Rank'] = centrality_df_sorted.index + 1

    # Extract serpin ranks
    serpins_present = [s for s in SERPINS if s in centrality_df_sorted['Protein'].values]

    for serpin in serpins_present:
        rank = centrality_df_sorted[centrality_df_sorted['Protein'] == serpin]['Rank'].values[0]
        score = centrality_df_sorted[centrality_df_sorted['Protein'] == serpin][metric].values[0]
        percentile = (rank / len(centrality_df_sorted)) * 100

        serpin_rankings.append({
            'Protein': serpin,
            'Metric': metric,
            'Rank': rank,
            'Total_Proteins': len(centrality_df_sorted),
            'Percentile': percentile,
            'Score': score,
            'Is_Central': percentile < 20  # Top quintile
        })

serpin_df = pd.DataFrame(serpin_rankings)
serpin_df.to_csv(OUTPUT_DIR / "serpin_rankings_claude_code.csv", index=False)

# Summary statistics per metric
print("\n  Serpin centrality summary by metric:")
summary = serpin_df.groupby('Metric').agg({
    'Percentile': ['mean', 'median', 'min', 'max'],
    'Is_Central': 'sum'
}).round(2)
print(summary)

# Top serpins per metric
print("\n  Top serpin per metric:")
for metric in ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']:
    top_serpin = serpin_df[serpin_df['Metric'] == metric].nsmallest(1, 'Rank')
    if len(top_serpin) > 0:
        protein = top_serpin['Protein'].values[0]
        rank = top_serpin['Rank'].values[0]
        percentile = top_serpin['Percentile'].values[0]
        print(f"    {metric:15s}: {protein:12s} (Rank {rank:3d}, {percentile:5.1f}th percentile)")

# Classification summary
print("\n  H02 DISAGREEMENT RESOLUTION:")
betw_serpins_central = serpin_df[
    (serpin_df['Metric'] == 'Betweenness') & (serpin_df['Is_Central'])
]['Protein'].tolist()
eigen_serpins_central = serpin_df[
    (serpin_df['Metric'] == 'Eigenvector') & (serpin_df['Is_Central'])
]['Protein'].tolist()

print(f"    Claude (Betweenness): {len(betw_serpins_central)} central serpins - {betw_serpins_central}")
print(f"    Codex (Eigenvector):  {len(eigen_serpins_central)} central serpins - {eigen_serpins_central}")

# =============================================================================
# 6. METRIC CORRELATION MATRIX
# =============================================================================

print(f"\n[5/9] Computing metric correlation matrix...")

# Calculate pairwise Spearman correlations between metrics
metric_cols = ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']
correlation_matrix = centrality_df[metric_cols].corr(method='spearman')

print("\n  Metric correlation matrix (Spearman ρ):")
print(correlation_matrix.round(3))

correlation_matrix.to_csv(OUTPUT_DIR / "metric_correlation_matrix_claude_code.csv")

# Key comparisons
print("\n  Key metric agreements:")
print(f"    Betweenness vs Eigenvector: ρ = {correlation_matrix.loc['Betweenness', 'Eigenvector']:.3f}")
print(f"    Betweenness vs Degree:      ρ = {correlation_matrix.loc['Betweenness', 'Degree']:.3f}")
print(f"    Eigenvector vs PageRank:    ρ = {correlation_matrix.loc['Eigenvector', 'PageRank']:.3f}")
print(f"    Eigenvector vs Degree:      ρ = {correlation_matrix.loc['Eigenvector', 'Degree']:.3f}")

print("\n  Analysis complete! Continuing to knockout simulations...")
