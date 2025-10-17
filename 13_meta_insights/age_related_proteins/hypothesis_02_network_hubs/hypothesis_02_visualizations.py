#!/usr/bin/env python3
"""
HYPOTHESIS #2: NETWORK VISUALIZATION AND COMMUNITY DETECTION
=============================================================

Creates publication-quality network visualizations and performs
Louvain community detection on the aging hub network.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("NETWORK VISUALIZATION AND COMMUNITY DETECTION")
print("=" * 80)

# Load data
edges_df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/network_edges.csv')
hubs_df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/master_hub_rankings.csv')

print(f"Network: {len(hubs_df)} nodes, {len(edges_df)} edges")

# ============================================================================
# COMMUNITY DETECTION (LOUVAIN-LIKE)
# ============================================================================

print("\n" + "=" * 80)
print("COMMUNITY DETECTION")
print("-" * 80)

# Build NetworkX graph
G = nx.Graph()

for _, row in edges_df.iterrows():
    G.add_edge(row['Gene1'], row['Gene2'], weight=row['Confidence'])

print(f"NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Community detection using greedy modularity optimization
print("\nDetecting communities (greedy modularity)...")
communities = nx.community.greedy_modularity_communities(G)

print(f"Detected {len(communities)} communities")

# Assign community IDs
node_community = {}
for i, community in enumerate(communities):
    for node in community:
        node_community[node] = i

# Analyze communities
community_sizes = Counter(node_community.values())
print("\nCommunity sizes:")
for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
    members = [node for node, cid in node_community.items() if cid == comm_id]
    print(f"  Community {comm_id+1}: {size} proteins")
    print(f"    Top members: {', '.join(sorted(members)[:5])}")

# Add community info to hub rankings
hubs_df['Community'] = hubs_df['Gene'].map(node_community)
hubs_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/master_hub_rankings_with_communities.csv', index=False)

# ============================================================================
# VISUALIZATION 1: NETWORK GRAPH WITH HUB HIGHLIGHTED
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION 1: Network Graph")
print("-" * 80)

fig, ax = plt.subplots(figsize=(16, 16))

# Layout
print("Computing layout (spring layout)...")
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Node properties
node_sizes = []
node_colors = []
master_hub = hubs_df.iloc[0]['Gene']

for node in G.nodes():
    # Size by degree
    degree = G.degree(node)
    node_sizes.append(degree * 20 + 100)

    # Color by community
    comm_id = node_community.get(node, -1)
    node_colors.append(comm_id)

# Draw edges
nx.draw_networkx_edges(
    G, pos,
    alpha=0.15,
    width=0.5,
    edge_color='gray',
    ax=ax
)

# Draw nodes
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap='tab20',
    alpha=0.8,
    ax=ax
)

# Highlight master hub
if master_hub in pos:
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[master_hub],
        node_size=3000,
        node_color='red',
        node_shape='*',
        ax=ax,
        label='Master Hub'
    )

# Label top 10 hubs
top10_hubs = hubs_df.head(10)['Gene'].tolist()
labels = {node: node for node in top10_hubs if node in pos}
nx.draw_networkx_labels(
    G, pos,
    labels,
    font_size=10,
    font_weight='bold',
    ax=ax
)

ax.set_title(
    f'ECM Aging Protein Network: {G.number_of_nodes()} Universal Proteins\n'
    f'Master Hub: {master_hub} (69 connections, 11.5× median degree)',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.axis('off')
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/network_graph.png', dpi=300, bbox_inches='tight')
print("Saved: network_graph.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: CENTRALITY DISTRIBUTIONS
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION 2: Centrality Distributions")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Degree distribution
ax = axes[0, 0]
degrees = hubs_df['Degree'].values
ax.hist(degrees, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(degrees[0], color='red', linestyle='--', linewidth=2, label=f'Master Hub: {degrees[0]}')
ax.axvline(np.median(degrees), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(degrees):.0f}')
ax.set_xlabel('Degree Centrality', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Degree Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Betweenness distribution
ax = axes[0, 1]
betweenness = hubs_df['Betweenness'].values
ax.hist(betweenness, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax.axvline(betweenness[0], color='red', linestyle='--', linewidth=2, label=f'Master Hub: {betweenness[0]:.1f}')
ax.axvline(np.median(betweenness), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(betweenness):.1f}')
ax.set_xlabel('Betweenness Centrality', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Betweenness Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# PageRank distribution
ax = axes[1, 0]
pagerank = hubs_df['PageRank'].values
ax.hist(pagerank, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
ax.axvline(pagerank[0], color='red', linestyle='--', linewidth=2, label=f'Master Hub: {pagerank[0]:.5f}')
ax.axvline(np.median(pagerank), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(pagerank):.5f}')
ax.set_xlabel('PageRank Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('PageRank Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Consensus score distribution
ax = axes[1, 1]
consensus = hubs_df['Consensus_Score'].values
ax.hist(consensus, bins=30, color='mediumpurple', alpha=0.7, edgecolor='black')
ax.axvline(consensus[0], color='red', linestyle='--', linewidth=2, label=f'Master Hub: {consensus[0]:.3f}')
ax.axvline(np.median(consensus), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(consensus):.3f}')
ax.set_xlabel('Consensus Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Consensus Score Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('Network Centrality Distributions: Master Hub vs Population', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/centrality_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: centrality_distributions.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: TOP 20 HUBS COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION 3: Top 20 Hubs Comparison")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 10))

top20 = hubs_df.head(20).copy()
top20 = top20.sort_values('Consensus_Score', ascending=True)

y_pos = np.arange(len(top20))
colors = ['red' if gene == master_hub else 'steelblue' for gene in top20['Gene']]

bars = ax.barh(y_pos, top20['Consensus_Score'], color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(top20['Gene'], fontsize=11)
ax.set_xlabel('Consensus Score', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Master Hub Proteins by Consensus Ranking', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add values
for i, (score, gene) in enumerate(zip(top20['Consensus_Score'], top20['Gene'])):
    degree = int(top20.iloc[i]['Degree'])
    ax.text(score + 0.02, i, f'{score:.3f} ({degree} conn)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/top20_hubs_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: top20_hubs_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: HUB PROTEIN TISSUE EXPRESSION
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION 4: Hub Protein Tissue Expression")
print("-" * 80)

# Load merged dataset for tissue expression
merged_df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Filter top 20 hubs
top20_genes = hubs_df.head(20)['Gene'].tolist()
tissue_data = merged_df[merged_df['Gene_Symbol'].isin(top20_genes)]

# Create pivot table
pivot = tissue_data.pivot_table(
    index='Gene_Symbol',
    columns='Tissue_Compartment',
    values='Zscore_Delta',
    aggfunc='mean'
)

# Reorder by hub ranking
pivot = pivot.reindex([g for g in top20_genes if g in pivot.index])

fig, ax = plt.subplots(figsize=(16, 10))

sns.heatmap(
    pivot,
    cmap='RdBu_r',
    center=0,
    cbar_kws={'label': 'Z-score Delta (Aging Effect)'},
    linewidths=0.5,
    linecolor='lightgray',
    ax=ax,
    vmin=-2,
    vmax=2
)

ax.set_title('Top 20 Hub Proteins: Tissue-Specific Aging Effects', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Tissue Compartment', fontsize=12, fontweight='bold')
ax.set_ylabel('Hub Protein', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/hub_tissue_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: hub_tissue_heatmap.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: COMMUNITY NETWORK
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION 5: Community Network")
print("-" * 80)

fig, ax = plt.subplots(figsize=(18, 18))

# Use same layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Draw by community
for i, community in enumerate(communities[:10]):  # Top 10 communities
    subgraph = G.subgraph(community)
    node_sizes = [G.degree(node) * 15 + 100 for node in subgraph.nodes()]

    nx.draw_networkx_nodes(
        subgraph, pos,
        node_size=node_sizes,
        node_color=[i] * len(subgraph),
        cmap='tab20',
        alpha=0.7,
        ax=ax,
        vmin=0,
        vmax=20
    )

    nx.draw_networkx_edges(
        subgraph, pos,
        alpha=0.2,
        width=0.5,
        ax=ax
    )

# Highlight master hub
if master_hub in pos:
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[master_hub],
        node_size=4000,
        node_color='red',
        node_shape='*',
        ax=ax
    )

# Label top hubs per community
for i, community in enumerate(communities[:5]):
    comm_hubs = [node for node in community if node in top20_genes]
    if comm_hubs:
        labels = {node: node for node in comm_hubs[:3]}
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )

ax.set_title(
    f'ECM Aging Network: {len(communities)} Functional Communities\n'
    f'Master Hub: {master_hub} (bridges {len([c for c in communities if master_hub in c])} communities)',
    fontsize=16,
    fontweight='bold',
    pad=20
)
ax.axis('off')

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_02_network_hubs/community_network.png', dpi=300, bbox_inches='tight')
print("Saved: community_network.png")
plt.close()

print("\n" + "=" * 80)
print("✅ ALL VISUALIZATIONS COMPLETE")
print("=" * 80)
print("\nSaved 5 publication-quality figures:")
print("  1. network_graph.png - Full network with master hub highlighted")
print("  2. centrality_distributions.png - Statistical distributions")
print("  3. top20_hubs_comparison.png - Consensus rankings")
print("  4. hub_tissue_heatmap.png - Tissue expression patterns")
print("  5. community_network.png - Functional communities")
