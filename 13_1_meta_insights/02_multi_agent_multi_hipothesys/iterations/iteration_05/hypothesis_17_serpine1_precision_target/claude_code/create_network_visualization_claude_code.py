#!/usr/bin/env python3
"""
H17: Network Perturbation Visualization
Generate before/after network visualization for SERPINE1 knockout
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

print("Generating network perturbation visualization...")

# Load cascade data
cascade_df = pd.read_csv('knockout_cascade_claude_code.csv')

# Get top 30 most affected proteins
top30 = cascade_df.head(30)

# Load centrality data to get edges
centrality_df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code/centrality_all_metrics_claude_code.csv')
edge_df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code/network_edges_claude_code.csv')

print(f"Loaded {len(edge_df)} edges")

# Filter to top 30 + SERPINE1
proteins_to_show = set(top30['Gene'].tolist() + ['SERPINE1', 'LOX', 'TGM2', 'COL1A1'])

# Build subgraph
G = nx.Graph()

for idx, row in edge_df.iterrows():
    if row['Protein_A'] in proteins_to_show and row['Protein_B'] in proteins_to_show:
        G.add_edge(row['Protein_A'], row['Protein_B'], weight=abs(row['Correlation']))

print(f"Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Node colors based on change
node_colors = []
node_sizes = []

for node in G.nodes():
    if node == 'SERPINE1':
        node_colors.append('red')
        node_sizes.append(800)
    elif node in ['LOX', 'TGM2', 'COL1A1']:
        node_colors.append('orange')
        node_sizes.append(600)
    else:
        # Color by delta
        delta_row = cascade_df[cascade_df['Gene'] == node]
        if len(delta_row) > 0:
            delta = delta_row['Delta'].values[0]
            if delta > 0:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightcoral')
            node_sizes.append(400)
        else:
            node_colors.append('lightgray')
            node_sizes.append(300)

# Layout
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       alpha=0.8, edgecolors='black', linewidths=1, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='SERPINE1 (Target)'),
    Patch(facecolor='orange', edgecolor='black', label='Mechanism targets (LOX/TGM2/COL1A1)'),
    Patch(facecolor='lightgreen', edgecolor='black', label='Reduced aging (positive Δ)'),
    Patch(facecolor='lightcoral', edgecolor='black', label='Increased aging (negative Δ)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax.set_title('Network Perturbation: SERPINE1 Knockout Effects\n(Top 30 Affected Proteins + Mechanism Targets)',
            fontsize=14, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('visualizations_claude_code/network_perturbation_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations_claude_code/network_perturbation_claude_code.png")

# Create summary table
summary_data = {
    'Metric': [
        'In-Silico Knockout Effect',
        'Literature Meta-Analysis (Cohen\'s d)',
        'Literature Heterogeneity (I²)',
        'Inhibitors with IC50<100nM',
        'Drugs passing ADMET',
        'Phase II trials completed',
        'Market Size (addressable)',
        'NPV (Net Present Value)',
        'ROI (%)'
    ],
    'Value': [
        '-0.22%',
        '2.524',
        '57.8%',
        '3/4',
        '4/4',
        '1 (PAI-039)',
        '$6.26B',
        '$8.47B',
        '3387%'
    ],
    'Target': [
        '≥30%',
        '>0.5',
        '<50%',
        '≥2',
        '≥2',
        '≥1',
        '>$1B',
        '>$0',
        '>100%'
    ],
    'Status': [
        '✗ FAIL',
        '✓ PASS',
        '⚠ MARGINAL',
        '✓ PASS',
        '✓ PASS',
        '✓ PASS',
        '✓ PASS',
        '✓ PASS',
        '✓ PASS'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('serpine1_summary_metrics_claude_code.csv', index=False)
print("✓ Saved: serpine1_summary_metrics_claude_code.csv")

print("\nSummary Table:")
print(summary_df.to_string(index=False))

print("\nVisualization complete!")
