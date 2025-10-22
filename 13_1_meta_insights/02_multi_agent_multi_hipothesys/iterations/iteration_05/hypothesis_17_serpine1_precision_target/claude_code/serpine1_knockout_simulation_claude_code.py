#!/usr/bin/env python3
"""
H17: SERPINE1 Precision Target Validation - In-Silico Knockout Analysis
=========================================================================

Performs GNN-based perturbation analysis to validate SERPINE1 as a causal
drug target for ECM aging intervention.

Author: Claude Code
Date: 2025-10-21
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("H17: SERPINE1 In-Silico Knockout Simulation")
print("="*80)

# ============================================================================
# 1. LOAD DATA AND BUILD NETWORK
# ============================================================================

print("\n[1] Loading ECM aging dataset...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

print(f"   Dataset: {len(df)} rows, {df['Gene_Symbol'].nunique()} proteins")

# Load H14 centrality data
centrality_df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/claude_code/centrality_all_metrics_claude_code.csv')

print(f"   Centrality data: {len(centrality_df)} proteins")

# Verify SERPINE1 presence
serpine1_row = centrality_df[centrality_df['Protein'] == 'SERPINE1']
if len(serpine1_row) > 0:
    print(f"\n   ✓ SERPINE1 found:")
    print(f"     - Eigenvector centrality: {serpine1_row['Eigenvector'].values[0]:.6f}")
    print(f"     - Degree: {serpine1_row['Degree'].values[0]:.4f}")
    print(f"     - Betweenness: {serpine1_row['Betweenness'].values[0]:.6f}")
else:
    print("   ✗ SERPINE1 NOT FOUND in centrality data!")

# ============================================================================
# 2. BUILD PROTEIN-PROTEIN CORRELATION NETWORK
# ============================================================================

print("\n[2] Building protein-protein correlation network...")

# Pivot to protein × tissue matrix
pivot = df.pivot_table(
    values='Zscore_Delta',
    index='Gene_Symbol',
    columns='Tissue',
    aggfunc='mean'
)

# Fill NaN with 0 for correlation
pivot_filled = pivot.fillna(0)
proteins = pivot_filled.index.tolist()
n_proteins = len(proteins)

print(f"   Proteins: {n_proteins}")
print(f"   Tissues: {len(pivot_filled.columns)}")

# Compute Spearman correlations
print("   Computing pairwise correlations (this may take a few minutes)...")
edges = []
edge_weights = []

for i in range(n_proteins):
    if i % 100 == 0:
        print(f"     Progress: {i}/{n_proteins}")

    for j in range(i+1, n_proteins):
        rho, p = spearmanr(pivot_filled.iloc[i], pivot_filled.iloc[j])

        if abs(rho) > 0.5 and p < 0.05:  # Threshold from H14
            edges.append([i, j])
            edges.append([j, i])  # Undirected
            edge_weights.append(abs(rho))
            edge_weights.append(abs(rho))

print(f"\n   Network constructed:")
print(f"     - Edges: {len(edges)//2} (undirected)")
print(f"     - Density: {len(edges)/(n_proteins*(n_proteins-1)):.4f}")

# Get SERPINE1 index
try:
    serpine1_idx = proteins.index('SERPINE1')
    print(f"     - SERPINE1 index: {serpine1_idx}")
except ValueError:
    print("     - WARNING: SERPINE1 not in protein list!")
    serpine1_idx = None

# ============================================================================
# 3. BUILD GRAPH NEURAL NETWORK (GNN)
# ============================================================================

print("\n[3] Building Graph Neural Network...")

# Prepare node features (protein expression profiles)
X = torch.tensor(pivot_filled.values, dtype=torch.float32)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Prepare edge index
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Create target: mean aging score per protein (mean absolute z-score delta)
y = torch.tensor(pivot_filled.abs().mean(axis=1).values, dtype=torch.float32)

print(f"   Node features shape: {X_tensor.shape}")
print(f"   Edge index shape: {edge_index.shape}")
print(f"   Target shape: {y.shape}")

# Create PyTorch Geometric Data object
data = Data(x=X_tensor, edge_index=edge_index, y=y)


# Define GNN model
class ECM_AgingGNN(nn.Module):
    """Graph Neural Network for ECM aging prediction"""

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(ECM_AgingGNN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        # GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Prediction head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x.squeeze()


# Initialize model
input_dim = X_tensor.shape[1]
model = ECM_AgingGNN(input_dim=input_dim, hidden_dim=64)

print(f"\n   Model architecture:")
print(f"     - Input dim: {input_dim}")
print(f"     - Hidden dim: 64")
print(f"     - GCN layers: 3")
print(f"     - Total parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 4. TRAIN GNN MODEL
# ============================================================================

print("\n[4] Training GNN model...")

# Split data for validation
train_mask = torch.rand(n_proteins) < 0.8
val_mask = ~train_mask

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.MSELoss()

train_losses = []
val_losses = []

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])

    # Backward pass
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_out = model(data.x, data.edge_index)
        val_loss = criterion(val_out[val_mask], data.y[val_mask])

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"   Epoch {epoch+1}/{epochs}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

print(f"\n   Training complete!")
print(f"     - Final train loss: {train_losses[-1]:.4f}")
print(f"     - Final val loss: {val_losses[-1]:.4f}")

# Save model
torch.save(model.state_dict(), 'visualizations_claude_code/gnn_ecm_aging_claude_code.pth')
print(f"     - Model saved to: visualizations_claude_code/gnn_ecm_aging_claude_code.pth")

# ============================================================================
# 5. IN-SILICO SERPINE1 KNOCKOUT
# ============================================================================

print("\n[5] Performing in-silico SERPINE1 knockout...")

if serpine1_idx is None:
    print("   ERROR: SERPINE1 not found, cannot perform knockout!")
else:
    model.eval()

    # BASELINE: Normal prediction
    with torch.no_grad():
        y_baseline = model(data.x, data.edge_index).numpy()

    # KNOCKOUT: Set SERPINE1 features to zero
    X_knockout = data.x.clone()
    X_knockout[serpine1_idx, :] = 0.0  # Simulate complete knockout

    with torch.no_grad():
        y_knockout = model(X_knockout, data.edge_index).numpy()

    # Calculate effects
    delta = y_baseline - y_knockout
    percent_change = (delta / (y_baseline + 1e-6)) * 100

    # Overall aging score change
    mean_baseline = y_baseline.mean()
    mean_knockout = y_knockout.mean()
    overall_reduction = ((mean_baseline - mean_knockout) / mean_baseline) * 100

    print(f"\n   KNOCKOUT RESULTS:")
    print(f"     - Baseline aging score (mean): {mean_baseline:.4f}")
    print(f"     - Knockout aging score (mean): {mean_knockout:.4f}")
    print(f"     - Overall reduction: {overall_reduction:.2f}%")

    if overall_reduction >= 30:
        print(f"     ✓ SUCCESS: ≥30% reduction achieved!")
    else:
        print(f"     ⚠ MARGINAL: <30% reduction (target not met)")

    # ========================================================================
    # 6. ANALYZE CASCADE EFFECTS
    # ========================================================================

    print("\n[6] Analyzing cascade effects on downstream proteins...")

    # Create results dataframe
    cascade_df = pd.DataFrame({
        'Gene': proteins,
        'Baseline': y_baseline,
        'Knockout': y_knockout,
        'Delta': delta,
        'Percent_Change': percent_change
    })

    # Sort by absolute change
    cascade_df = cascade_df.sort_values('Delta', ascending=False)

    # Save full results
    cascade_df.to_csv('knockout_cascade_claude_code.csv', index=False)
    print(f"   ✓ Saved cascade analysis: knockout_cascade_claude_code.csv")

    # Top 20 affected proteins
    top20 = cascade_df.head(20)
    print(f"\n   Top 20 proteins affected by SERPINE1 knockout:")
    print(top20[['Gene', 'Delta', 'Percent_Change']].to_string(index=False))

    # Check for LOX, TGM2, COL1A1
    targets = ['LOX', 'TGM2', 'COL1A1']
    print(f"\n   Checking mechanism targets (LOX, TGM2, COL1A1):")

    for target in targets:
        target_row = cascade_df[cascade_df['Gene'] == target]
        if len(target_row) > 0:
            rank = list(cascade_df['Gene']).index(target) + 1
            delta_val = target_row['Delta'].values[0]
            pct_val = target_row['Percent_Change'].values[0]

            in_top20 = rank <= 20
            status = "✓ IN TOP 20" if in_top20 else "✗ NOT IN TOP 20"

            print(f"     {target}: Rank {rank}/{n_proteins}, Δ={delta_val:.4f}, {pct_val:.2f}% [{status}]")
        else:
            print(f"     {target}: NOT FOUND in dataset")

    # Count targets in top 20
    targets_in_top20 = sum([1 for t in targets if t in top20['Gene'].values])
    print(f"\n   Mechanism validation: {targets_in_top20}/3 targets in top 20")

    if targets_in_top20 >= 2:
        print(f"     ✓ MECHANISM CONFIRMED (≥2/3 targets affected)")
    else:
        print(f"     ⚠ MECHANISM UNCERTAIN (<2/3 targets affected)")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n[7] Generating visualizations...")

# Training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Val Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('GNN Training Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations_claude_code/gnn_training_curves_claude_code.png', dpi=300)
print("   ✓ Saved: visualizations_claude_code/gnn_training_curves_claude_code.png")

if serpine1_idx is not None:
    # Waterfall plot of top 50 affected proteins
    top50 = cascade_df.head(50)

    plt.figure(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in top50['Delta']]
    plt.barh(range(len(top50)), top50['Delta'], color=colors, alpha=0.7)
    plt.yticks(range(len(top50)), top50['Gene'], fontsize=8)
    plt.xlabel('Δ Aging Score (Baseline - Knockout)')
    plt.title('Top 50 Proteins Affected by SERPINE1 Knockout\n(Green = Aging reduced, Red = Aging increased)')
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('visualizations_claude_code/knockout_waterfall_claude_code.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: visualizations_claude_code/knockout_waterfall_claude_code.png")

    # Heatmap: Baseline vs Knockout for top 30 proteins
    top30 = cascade_df.head(30)

    heatmap_data = pd.DataFrame({
        'Baseline': top30['Baseline'].values,
        'Knockout': top30['Knockout'].values
    }, index=top30['Gene'])

    plt.figure(figsize=(6, 10))
    sns.heatmap(heatmap_data.T, cmap='RdYlGn_r', annot=False, cbar_kws={'label': 'Aging Score'})
    plt.title('Aging Scores: Baseline vs SERPINE1 Knockout (Top 30)')
    plt.ylabel('Condition')
    plt.xlabel('Protein')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig('visualizations_claude_code/knockout_heatmap_claude_code.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: visualizations_claude_code/knockout_heatmap_claude_code.png")

print("\n" + "="*80)
print("IN-SILICO KNOCKOUT SIMULATION COMPLETE")
print("="*80)
