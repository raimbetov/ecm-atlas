#!/usr/bin/env python3
"""
Visualization and Consensus Centrality Analysis for H14

Generates all required visualizations and computes consensus centrality ensemble.

Agent: claude_code
Created: 2025-10-21
"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import community as community_louvain

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code")
VIZ_DIR = OUTPUT_DIR / "visualizations_claude_code"

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("H14: VISUALIZATION AND CONSENSUS CENTRALITY")
print("="*80)

# Load data
print("\n[8/9] Loading data for visualization...")
edges_df = pd.read_csv(OUTPUT_DIR / "network_edges_claude_code.csv")
centrality_df = pd.read_csv(OUTPUT_DIR / "centrality_all_metrics_claude_code.csv")
serpin_df = pd.read_csv(OUTPUT_DIR / "serpin_rankings_claude_code.csv")
knockout_df = pd.read_csv(OUTPUT_DIR / "knockout_impact_claude_code.csv")
correlation_matrix = pd.read_csv(OUTPUT_DIR / "metric_correlation_matrix_claude_code.csv", index_col=0)

# =============================================================================
# VISUALIZATION 1: METRIC CORRELATION HEATMAP
# =============================================================================

print("  Creating visualization 1: Metric correlation heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Spearman ρ'},
    ax=ax
)
ax.set_title('Network Centrality Metrics Correlation Matrix\n(Spearman Rank Correlation)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Centrality Metric', fontsize=12)
ax.set_ylabel('Centrality Metric', fontsize=12)
plt.tight_layout()
plt.savefig(VIZ_DIR / "centrality_heatmap_claude_code.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: centrality_heatmap_claude_code.png")

# =============================================================================
# VISUALIZATION 2: SERPIN RANKS COMPARISON (VIOLIN PLOT)
# =============================================================================

print("  Creating visualization 2: Serpin ranks comparison...")

fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=serpin_df,
    x='Metric',
    y='Percentile',
    inner='box',
    palette='Set2',
    ax=ax
)
ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Central threshold (20th percentile)')
ax.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='Median')
ax.set_ylim([0, 100])
ax.set_xlabel('Centrality Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Serpin Percentile Rank (%)', fontsize=12, fontweight='bold')
ax.set_title('Serpin Family Centrality Rankings Across Metrics\nLower = More Central',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(VIZ_DIR / "serpin_ranks_comparison_claude_code.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: serpin_ranks_comparison_claude_code.png")

# =============================================================================
# VISUALIZATION 3: BETWEENNESS VS EIGENVECTOR SCATTER
# =============================================================================

print("  Creating visualization 3: Betweenness vs Eigenvector scatter...")

# Identify serpins
SERPINS = [
    'SERPINA1', 'SERPINA3', 'SERPINA5', 'SERPINB1', 'SERPINB6', 'SERPINB8',
    'SERPINC1', 'SERPINE1', 'SERPINE2', 'SERPINF1', 'SERPINF2', 'SERPING1', 'SERPINH1'
]
centrality_df['Is_Serpin'] = centrality_df['Protein'].isin(SERPINS)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot non-serpins
non_serpins = centrality_df[~centrality_df['Is_Serpin']]
ax.scatter(
    non_serpins['Betweenness'],
    non_serpins['Eigenvector'],
    c='lightgray',
    s=30,
    alpha=0.5,
    label='Other proteins'
)

# Plot serpins
serpins_data = centrality_df[centrality_df['Is_Serpin']]
ax.scatter(
    serpins_data['Betweenness'],
    serpins_data['Eigenvector'],
    c='red',
    s=100,
    alpha=0.8,
    edgecolors='black',
    linewidths=1,
    label='Serpins'
)

# Annotate serpins
for _, row in serpins_data.iterrows():
    ax.annotate(
        row['Protein'],
        (row['Betweenness'], row['Eigenvector']),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8,
        fontweight='bold'
    )

ax.set_xlabel('Betweenness Centrality (Claude H02)', fontsize=12, fontweight='bold')
ax.set_ylabel('Eigenvector Centrality (Codex H02)', fontsize=12, fontweight='bold')
ax.set_title('H02 Disagreement: Betweenness vs Eigenvector Centrality\nSerpins Highlighted in Red',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / "betweenness_vs_eigenvector_claude_code.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: betweenness_vs_eigenvector_claude_code.png")

# =============================================================================
# VISUALIZATION 4: KNOCKOUT IMPACT SCATTER (CENTRALITY VS IMPACT)
# =============================================================================

print("  Creating visualization 4: Centrality vs knockout impact...")

# Merge knockout with centrality
knockout_with_centrality = knockout_df.merge(centrality_df, on='Protein', how='left')

# Create subplots for each metric
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

metrics = ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    x = knockout_with_centrality[metric].fillna(0)
    y = knockout_with_centrality['Impact_Score']

    # Scatter plot
    ax.scatter(x, y, s=80, alpha=0.7, c='steelblue', edgecolors='black', linewidths=0.5)

    # Add labels for each serpin
    for _, row in knockout_with_centrality.iterrows():
        ax.annotate(
            row['Protein'],
            (row[metric], row['Impact_Score']),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=7
        )

    # Fit line
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)

        # Compute correlation
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(x, y)
        ax.text(
            0.05, 0.95,
            f'ρ = {rho:.3f}\np = {p_val:.4f}',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    ax.set_xlabel(f'{metric} Centrality', fontsize=10)
    ax.set_ylabel('Knockout Impact Score', fontsize=10)
    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Remove extra subplot
axes[-1].axis('off')

fig.suptitle('Centrality Metrics vs. In Silico Knockout Impact\nHigher Impact = More Critical Protein',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(VIZ_DIR / "knockout_impact_scatter_claude_code.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: knockout_impact_scatter_claude_code.png")

# =============================================================================
# CONSENSUS CENTRALITY COMPUTATION
# =============================================================================

print("\n[9/9] Computing consensus centrality ensemble...")

# Z-score normalize each metric
metrics_list = ['Degree', 'Betweenness', 'Eigenvector', 'Closeness', 'PageRank', 'Katz', 'Subgraph']
centrality_normalized = centrality_df.copy()

for metric in metrics_list:
    centrality_normalized[f'{metric}_zscore'] = zscore(centrality_df[metric].fillna(0))

# Consensus: mean of all z-scores
centrality_normalized['Consensus'] = centrality_normalized[
    [f'{m}_zscore' for m in metrics_list]
].mean(axis=1)

# Rank by consensus
centrality_normalized = centrality_normalized.sort_values('Consensus', ascending=False).reset_index(drop=True)
centrality_normalized['Consensus_Rank'] = centrality_normalized.index + 1

# Save consensus
consensus_df = centrality_normalized[['Protein', 'Consensus', 'Consensus_Rank'] + metrics_list]
consensus_df.to_csv(OUTPUT_DIR / "consensus_centrality_claude_code.csv", index=False)

print("\n  Top 20 proteins by consensus centrality:")
print(consensus_df[['Protein', 'Consensus', 'Consensus_Rank']].head(20).to_string(index=False))

# Serpin consensus ranks
serpin_consensus = consensus_df[consensus_df['Protein'].isin(SERPINS)].copy()
serpin_consensus['Percentile'] = (serpin_consensus['Consensus_Rank'] / len(consensus_df)) * 100

print("\n  Serpin consensus centrality rankings:")
print(serpin_consensus[['Protein', 'Consensus', 'Consensus_Rank', 'Percentile']].to_string(index=False))

# =============================================================================
# VISUALIZATION 5: NETWORK GRAPH WITH SERPINS HIGHLIGHTED
# =============================================================================

print("\n  Creating visualization 5: Network graph with serpins...")

# Reconstruct network (subsample for visualization)
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['Protein_A'], row['Protein_B'], weight=row['Abs_Correlation'])

# Get largest component
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
else:
    G_main = G.copy()

# If too large, sample top degree nodes
if G_main.number_of_nodes() > 200:
    print(f"    Network large ({G_main.number_of_nodes()} nodes), sampling top 200 by degree...")
    degrees = dict(G_main.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:200]
    G_vis = G_main.subgraph(top_nodes).copy()
else:
    G_vis = G_main.copy()

# Node colors: serpins = red, others = lightblue
node_colors = ['red' if n in SERPINS else 'lightblue' for n in G_vis.nodes()]
node_sizes = [500 if n in SERPINS else 50 for n in G_vis.nodes()]

fig, ax = plt.subplots(figsize=(14, 14))
pos = nx.spring_layout(G_vis, k=0.3, iterations=50, seed=42)

# Draw network
nx.draw_networkx_edges(G_vis, pos, alpha=0.2, width=0.5, ax=ax)
nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)

# Label only serpins
serpin_labels = {n: n for n in G_vis.nodes() if n in SERPINS}
nx.draw_networkx_labels(G_vis, pos, labels=serpin_labels, font_size=10, font_weight='bold', ax=ax)

ax.set_title(f'ECM Aging Protein Network (Top {G_vis.number_of_nodes()} Nodes)\nSerpins Highlighted in Red',
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig(VIZ_DIR / "network_graph_serpins_claude_code.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: network_graph_serpins_claude_code.png")

# =============================================================================
# VISUALIZATION 6: COMMUNITY DETECTION
# =============================================================================

print("\n  Creating visualization 6: Community modules...")

# Detect communities (Louvain algorithm)
communities = community_louvain.best_partition(G_main, weight='weight')

# Serpin community assignments
serpin_communities = {s: communities[s] for s in SERPINS if s in communities}
num_serpin_communities = len(set(serpin_communities.values()))

print(f"\n  Community detection results:")
print(f"    Total communities: {len(set(communities.values()))}")
print(f"    Serpins span {num_serpin_communities} communities")
print(f"    Serpin community assignments:")
for serpin, comm in sorted(serpin_communities.items(), key=lambda x: x[1]):
    print(f"      {serpin:12s} → Community {comm}")

# Save community assignments
community_df = pd.DataFrame([
    {'Protein': protein, 'Community': comm}
    for protein, comm in communities.items()
])
community_df.to_csv(OUTPUT_DIR / "community_assignments_claude_code.csv", index=False)

# Visualize communities (sample if too large)
if G_main.number_of_nodes() > 300:
    top_nodes = sorted(dict(G_main.degree()).items(), key=lambda x: x[1], reverse=True)[:300]
    top_nodes = [n[0] for n in top_nodes]
    G_comm = G_main.subgraph(top_nodes).copy()
    communities_vis = {n: communities[n] for n in G_comm.nodes()}
else:
    G_comm = G_main.copy()
    communities_vis = communities

fig, ax = plt.subplots(figsize=(16, 16))
pos = nx.spring_layout(G_comm, k=0.5, iterations=50, seed=42)

# Color by community
import matplotlib.cm as cm
num_communities = len(set(communities_vis.values()))
cmap = cm.get_cmap('tab20', num_communities)
node_colors_comm = [cmap(communities_vis[n]) for n in G_comm.nodes()]
node_sizes_comm = [500 if n in SERPINS else 30 for n in G_comm.nodes()]

nx.draw_networkx_edges(G_comm, pos, alpha=0.1, width=0.3, ax=ax)
nx.draw_networkx_nodes(G_comm, pos, node_color=node_colors_comm, node_size=node_sizes_comm, alpha=0.8, ax=ax)

# Label serpins
serpin_labels = {n: n for n in G_comm.nodes() if n in SERPINS}
nx.draw_networkx_labels(G_comm, pos, labels=serpin_labels, font_size=9, font_weight='bold', ax=ax)

ax.set_title(f'Protein Network Community Structure (Louvain Algorithm)\n{num_communities} Communities, Serpins Labeled',
             fontsize=16, fontweight='bold', pad=20)
ax.axis('off')
plt.tight_layout()
plt.savefig(VIZ_DIR / "community_modules_claude_code.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: community_modules_claude_code.png")

print("\n" + "="*80)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*80)
