#!/usr/bin/env python3
"""
Create visualizations for the multi-hypothesis synthesis report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
base_dir = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys")
raw_data = pd.read_csv(base_dir / "raw_extraction_data.csv")
ranking = pd.read_csv(base_dir / "synthesis_master_ranking_table.csv")

# Create output directory
viz_dir = base_dir / "synthesis_visualizations"
viz_dir.mkdir(exist_ok=True)

print("Creating visualizations...")
print("=" * 80)

# 1. Hypothesis Completion Rate per Iteration
print("1. Creating completion rate bar chart...")
iteration_data = []
for i in range(1, 6):
    iter_files = [f for f in raw_data['File'] if f.startswith(f'iteration_0{i}/')]
    claude_count = sum(1 for f in iter_files if 'claude' in f)
    codex_count = sum(1 for f in iter_files if 'codex' in f)

    iteration_data.append({
        'Iteration': f'Iteration {i:02d}',
        'Claude': claude_count,
        'Codex': codex_count,
        'Total': claude_count + codex_count
    })

iter_df = pd.DataFrame(iteration_data)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(iter_df))
width = 0.35

ax.bar(x - width/2, iter_df['Claude'], width, label='Claude Code', color='#4CAF50')
ax.bar(x + width/2, iter_df['Codex'], width, label='Codex', color='#2196F3')

ax.set_xlabel('Iteration', fontweight='bold')
ax.set_ylabel('Hypotheses Completed', fontweight='bold')
ax.set_title('Hypothesis Completion Rate Per Iteration', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(iter_df['Iteration'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / "completion_rate_per_iteration.png")
plt.close()
print(f"  Saved: completion_rate_per_iteration.png")

# 2. Claude vs Codex Agreement Matrix
print("2. Creating agreement matrix...")
# Get hypotheses where both agents completed
both = ranking[ranking['Both_Completed'] == True].copy()
if len(both) > 0:
    agreement_counts = both['Claude_Codex_Agreement'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'AGREE': '#4CAF50', 'PARTIAL': '#FFC107', 'DISAGREE': '#F44336'}
    bars = ax.bar(agreement_counts.index, agreement_counts.values,
                   color=[colors.get(x, '#999') for x in agreement_counts.index])

    ax.set_xlabel('Agreement Type', fontweight='bold')
    ax.set_ylabel('Number of Hypotheses', fontweight='bold')
    ax.set_title('Claude Code vs Codex Agreement Distribution', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(viz_dir / "agreement_distribution.png")
    plt.close()
    print(f"  Saved: agreement_distribution.png")

# 3. Metrics Distribution (Success vs Failure hypotheses)
print("3. Creating metrics distribution scatter...")
success_hyp = raw_data[raw_data['Status'] == 'SUCCESS']
failure_hyp = raw_data[raw_data['Status'] != 'SUCCESS']

fig, ax = plt.subplots(figsize=(10, 7))

# Plot scores vs hypothesis ID
if 'Score' in raw_data.columns:
    for agent in ['claude', 'codex']:
        agent_data = raw_data[raw_data['Agent'] == agent]
        marker = 'o' if agent == 'claude' else 's'
        color = '#4CAF50' if agent == 'claude' else '#2196F3'
        ax.scatter(agent_data['Hypothesis_ID'], agent_data['Score'],
                   label=agent.title(), marker=marker, s=100, alpha=0.7, color=color)

ax.set_xlabel('Hypothesis', fontweight='bold')
ax.set_ylabel('Self-Evaluation Score (out of 100)', fontweight='bold')
ax.set_title('Hypothesis Performance Scores', fontweight='bold', fontsize=14)
ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90+)')
ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good (70+)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(viz_dir / "score_distribution.png")
plt.close()
print(f"  Saved: score_distribution.png")

# 4. Top 10 Hypotheses Heatmap
print("4. Creating top 10 hypotheses summary heatmap...")

# Create heatmap data for top 10
top10 = ranking.head(10).copy()

# Prepare data for heatmap
heatmap_data = pd.DataFrame({
    'Rank': top10['Rank'],
    'Hypothesis': [f"H{str(x).split('H')[1][:2]}" for x in top10['Hypothesis_ID']],
    'Claude_Score': top10['Claude_Score'].fillna(0),
    'Codex_Score': top10['Codex_Score'].fillna(0),
    'Status': [1 if x == 'SUCCESS' else 0 for x in top10['Status']],
})

fig, ax = plt.subplots(figsize=(8, 10))

# Prepare matrix for heatmap (transpose for better layout)
matrix = heatmap_data[['Claude_Score', 'Codex_Score', 'Status']].values.T

im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks
ax.set_xticks(np.arange(len(heatmap_data)))
ax.set_yticks(np.arange(3))
ax.set_xticklabels(heatmap_data['Hypothesis'])
ax.set_yticklabels(['Claude Score', 'Codex Score', 'Success (0/1)'])

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score/Status', rotation=270, labelpad=15)

# Add text annotations
for i in range(3):
    for j in range(len(heatmap_data)):
        val = matrix[i, j]
        text = ax.text(j, i, f'{val:.0f}',
                       ha="center", va="center", color="black", fontsize=8)

ax.set_title('Top 10 Hypotheses: Agent Scores and Success Status', fontweight='bold', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(viz_dir / "top10_heatmap.png")
plt.close()
print(f"  Saved: top10_heatmap.png")

# 5. Status distribution pie chart
print("5. Creating status distribution pie chart...")
status_counts = raw_data['Status'].value_counts()

fig, ax = plt.subplots(figsize=(8, 8))
colors_status = {'SUCCESS': '#4CAF50', 'PARTIAL': '#FFC107', 'FAILURE': '#F44336', 'UNKNOWN': '#999'}
pie_colors = [colors_status.get(x, '#999') for x in status_counts.index]

wedges, texts, autotexts = ax.pie(status_counts.values, labels=status_counts.index,
                                    autopct='%1.1f%%', startangle=90,
                                    colors=pie_colors, textprops={'fontsize': 12, 'fontweight': 'bold'})

ax.set_title('Hypothesis Outcome Distribution\n(All Results)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(viz_dir / "status_distribution.png")
plt.close()
print(f"  Saved: status_distribution.png")

print("\n" + "=" * 80)
print(f"All visualizations saved to: {viz_dir}")
print("=" * 80)
