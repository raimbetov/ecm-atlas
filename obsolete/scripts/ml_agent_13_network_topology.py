#!/usr/bin/env python3
"""
ML AGENT 13: PROTEIN NETWORK TOPOLOGY ANALYZER
===============================================

Mission: Analyze ECM proteins as networks to find hub proteins, communities,
and central regulators using graph theory and network science.

Approach:
1. Build correlation networks
2. Identify hub proteins (high degree, betweenness)
3. Community detection (Louvain algorithm)
4. PageRank for importance ranking
5. Network motifs and modules
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML AGENT 13: PROTEIN NETWORK TOPOLOGY ANALYZER")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
print(f"Total records: {len(df):,}")

# Create protein-tissue matrix
pivot = df[df['Zscore_Delta'].notna()].pivot_table(
    index='Gene_Symbol',
    columns='Tissue_Compartment',
    values='Zscore_Delta',
    aggfunc='mean'
).fillna(0)

print(f"Network nodes: {pivot.shape[0]} proteins")

# TASK 1: Build correlation network
print("\n" + "=" * 80)
print("TASK 1: BUILDING CORRELATION NETWORK")
print("=" * 80)

print("\n🔗 Computing pairwise protein correlations...")
proteins = pivot.index.tolist()
edges = []
correlation_threshold = 0.7  # Strong correlation

for i, p1 in enumerate(proteins):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(proteins)} proteins...")

    for p2 in proteins[i+1:]:
        v1 = pivot.loc[p1].values
        v2 = pivot.loc[p2].values

        # Remove missing values
        mask = ~(np.isnan(v1) | np.isnan(v2))
        if mask.sum() < 3:  # Need at least 3 data points
            continue

        corr, pval = pearsonr(v1[mask], v2[mask])
        if abs(corr) >= correlation_threshold and pval < 0.05:
            edges.append({
                'Protein1': p1,
                'Protein2': p2,
                'Correlation': corr,
                'P_value': pval,
                'Type': 'Synergistic' if corr > 0 else 'Antagonistic'
            })

edges_df = pd.DataFrame(edges)
print(f"\n✅ Network edges: {len(edges_df):,}")
print(f"   Synergistic: {len(edges_df[edges_df['Type'] == 'Synergistic']):,}")
print(f"   Antagonistic: {len(edges_df[edges_df['Type'] == 'Antagonistic']):,}")

# TASK 2: Network centrality metrics
print("\n" + "=" * 80)
print("TASK 2: HUB PROTEIN IDENTIFICATION")
print("=" * 80)

# Degree centrality (number of connections)
degree_counter = Counter()
for _, edge in edges_df.iterrows():
    degree_counter[edge['Protein1']] += 1
    degree_counter[edge['Protein2']] += 1

hub_proteins = pd.DataFrame([
    {'Protein': protein, 'Degree': degree}
    for protein, degree in degree_counter.most_common()
]).sort_values('Degree', ascending=False)

print("\n🌟 TOP 20 HUB PROTEINS (Most Connected):")
for i, row in hub_proteins.head(20).iterrows():
    print(f"  {row['Protein']:15s} → {row['Degree']} connections")

# TASK 3: Betweenness centrality (simplified)
print("\n" + "=" * 80)
print("TASK 3: BRIDGE PROTEINS (High Betweenness)")
print("=" * 80)

# Build adjacency list
adj_list = defaultdict(set)
for _, edge in edges_df.iterrows():
    adj_list[edge['Protein1']].add(edge['Protein2'])
    adj_list[edge['Protein2']].add(edge['Protein1'])

# Estimate betweenness using neighborhood diversity
betweenness_scores = {}
for protein in proteins:
    if protein not in adj_list:
        betweenness_scores[protein] = 0
        continue

    neighbors = adj_list[protein]
    if len(neighbors) < 2:
        betweenness_scores[protein] = 0
        continue

    # Count connections between neighbors (clustering coefficient inverse)
    neighbor_connections = 0
    for n1 in neighbors:
        for n2 in neighbors:
            if n1 != n2 and n2 in adj_list.get(n1, set()):
                neighbor_connections += 1

    max_possible = len(neighbors) * (len(neighbors) - 1)
    clustering = neighbor_connections / max_possible if max_possible > 0 else 0
    betweenness_scores[protein] = (1 - clustering) * len(neighbors)  # High betweenness if low clustering

bridge_proteins = pd.DataFrame([
    {'Protein': protein, 'Betweenness_Score': score}
    for protein, score in betweenness_scores.items()
]).sort_values('Betweenness_Score', ascending=False)

print("\n🌉 TOP 20 BRIDGE PROTEINS (Connecting Different Modules):")
for i, row in bridge_proteins.head(20).iterrows():
    print(f"  {row['Protein']:15s} → Betweenness: {row['Betweenness_Score']:.2f}")

# TASK 4: Simplified PageRank
print("\n" + "=" * 80)
print("TASK 4: PAGERANK IMPORTANCE")
print("=" * 80)

print("\n🎯 Computing PageRank scores...")
# Simple iterative PageRank
damping = 0.85
n_iterations = 20
pagerank = {p: 1.0 / len(proteins) for p in proteins}

for iteration in range(n_iterations):
    new_pagerank = {}
    for protein in proteins:
        rank_sum = 0
        # Sum contributions from incoming links
        for neighbor in adj_list.get(protein, []):
            neighbor_degree = len(adj_list.get(neighbor, []))
            if neighbor_degree > 0:
                rank_sum += pagerank[neighbor] / neighbor_degree

        new_pagerank[protein] = (1 - damping) / len(proteins) + damping * rank_sum

    pagerank = new_pagerank

pagerank_df = pd.DataFrame([
    {'Protein': protein, 'PageRank': score}
    for protein, score in pagerank.items()
]).sort_values('PageRank', ascending=False)

print("\n🏅 TOP 20 PROTEINS BY PAGERANK:")
for i, row in pagerank_df.head(20).iterrows():
    print(f"  {row['Protein']:15s} → PageRank: {row['PageRank']:.6f}")

# TASK 5: Community detection (simplified Louvain-like)
print("\n" + "=" * 80)
print("TASK 5: PROTEIN COMMUNITIES")
print("=" * 80)

# Simplified community detection: connected components
visited = set()
communities = []

def dfs(node, community):
    visited.add(node)
    community.add(node)
    for neighbor in adj_list.get(node, []):
        if neighbor not in visited:
            dfs(neighbor, community)

for protein in proteins:
    if protein not in visited:
        community = set()
        dfs(protein, community)
        if len(community) >= 3:  # Only communities with 3+ members
            communities.append(community)

communities.sort(key=len, reverse=True)

print(f"\n🏘️  Detected {len(communities)} communities (size ≥ 3)")

print("\nTop 5 Largest Communities:")
for i, comm in enumerate(communities[:5]):
    print(f"\nCommunity {i+1} (n={len(comm)}):")
    comm_list = sorted(list(comm))[:10]
    print(f"  Members: {', '.join(comm_list)}")

# Save all results
hub_proteins.to_csv('10_insights/ml_network_hubs.csv', index=False)
bridge_proteins.to_csv('10_insights/ml_network_bridges.csv', index=False)
pagerank_df.to_csv('10_insights/ml_network_pagerank.csv', index=False)
edges_df.to_csv('10_insights/ml_network_edges.csv', index=False)

print("\n✅ Saved network analysis results")

# TASK 6: Consensus master regulators
print("\n" + "=" * 80)
print("🎯 CONSENSUS MASTER REGULATORS")
print("=" * 80)

# Combine all centrality metrics
consensus = hub_proteins.merge(bridge_proteins, on='Protein', how='outer').fillna(0)
consensus = consensus.merge(pagerank_df, on='Protein', how='outer').fillna(0)

# Normalize scores
consensus['Degree_Norm'] = consensus['Degree'] / consensus['Degree'].max()
consensus['Betweenness_Norm'] = consensus['Betweenness_Score'] / consensus['Betweenness_Score'].max()
consensus['PageRank_Norm'] = consensus['PageRank'] / consensus['PageRank'].max()

# Consensus score (average of normalized metrics)
consensus['Consensus_Score'] = (
    consensus['Degree_Norm'] +
    consensus['Betweenness_Norm'] +
    consensus['PageRank_Norm']
) / 3

consensus = consensus.sort_values('Consensus_Score', ascending=False)

print("\n👑 TOP 20 MASTER REGULATORS (Consensus Ranking):")
for i, row in consensus.head(20).iterrows():
    print(f"{row['Protein']:15s} → Score: {row['Consensus_Score']:.3f} (Degree: {row['Degree']}, PR: {row['PageRank']:.5f})")

consensus.to_csv('10_insights/ml_network_master_regulators.csv', index=False)
print("\n✅ Saved: 10_insights/ml_network_master_regulators.csv")

print("\n" + "=" * 80)
print("✅ ML AGENT 13 COMPLETED")
print("=" * 80)
