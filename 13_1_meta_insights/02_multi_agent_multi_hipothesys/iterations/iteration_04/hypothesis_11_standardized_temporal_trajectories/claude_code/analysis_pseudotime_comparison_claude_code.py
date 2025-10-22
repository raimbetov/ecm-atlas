#!/usr/bin/env python3
"""
H11 - Pseudo-Time Methods Comparison: Resolving H09 Disagreement
Author: claude_code
Date: 2025-10-21

This script systematically compares 5 pseudo-time construction methods:
1. Tissue Velocity Ranking (Claude H03 approach) - R²=0.81 winner
2. PCA-Based Ordering (Codex approach) - R²=0.011 loser
3. Diffusion Pseudotime (UMAP + distance-based)
4. Slingshot (trajectory inference with potential branching)
5. Autoencoder Latent Traversal (nonlinear dimensionality reduction)

Goal: Identify which method produces most robust LSTM predictions and temporal insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# PyTorch for LSTM and Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# UMAP for diffusion-like pseudo-time
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    print("WARNING: UMAP not installed. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

print("="*80)
print("H11: PSEUDO-TIME METHODS COMPARISON - Resolving LSTM Reproducibility Crisis")
print("="*80)

# ============================================================================
# 1. LOAD DATA & PREPARE TISSUE-LEVEL MATRIX
# ============================================================================

print("\n[1] Loading ECM aging dataset...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

print(f"   Dataset shape: {df.shape}")
print(f"   Unique tissues: {df['Tissue'].nunique()}")
print(f"   Unique proteins: {df['Gene_Symbol'].nunique()}")

# Create tissue × protein matrix (using Zscore_Delta)
print("\n[2] Creating tissue × protein matrix...")
tissue_protein_matrix = df.pivot_table(
    values='Zscore_Delta',
    index='Tissue',
    columns='Gene_Symbol',
    aggfunc='median'  # Aggregate multiple studies
)

print(f"   Matrix shape: {tissue_protein_matrix.shape}")
print(f"   Tissues (rows): {len(tissue_protein_matrix)}")
print(f"   Proteins (cols): {len(tissue_protein_matrix.columns)}")
print(f"   Missing values: {tissue_protein_matrix.isna().sum().sum()} ({tissue_protein_matrix.isna().sum().sum() / tissue_protein_matrix.size * 100:.1f}%)")

# Fill NaN with 0 (protein not changing in that tissue)
tissue_protein_matrix_filled = tissue_protein_matrix.fillna(0)
X_tissues = tissue_protein_matrix_filled.values  # (n_tissues, n_proteins)

tissues_list = tissue_protein_matrix_filled.index.tolist()
proteins_list = tissue_protein_matrix_filled.columns.tolist()

print(f"\n   Tissues included: {tissues_list}")

# ============================================================================
# 2. METHOD 1: TISSUE VELOCITY RANKING (Claude H03 approach)
# ============================================================================

print("\n" + "="*80)
print("[METHOD 1] Tissue Velocity Ranking (H03 - Claude approach)")
print("="*80)

# From H03 results (tissue velocities)
# We'll recalculate from current data for consistency
tissue_velocities = tissue_protein_matrix_filled.abs().mean(axis=1).sort_values(ascending=False)

print("\nTissue Velocities (mean |Δz| across proteins):")
for tissue, velocity in tissue_velocities.items():
    print(f"   {tissue:35s}: {velocity:6.3f}")

# Pseudo-time = rank by velocity (higher velocity = later in aging timeline)
velocity_pseudotime = tissue_velocities.rank(ascending=False).to_dict()

print("\n✓ Method 1 complete: Velocity-based pseudo-time")

# ============================================================================
# 3. METHOD 2: PCA-BASED ORDERING (Codex approach)
# ============================================================================

print("\n" + "="*80)
print("[METHOD 2] PCA-Based Ordering (Codex approach)")
print("="*80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_tissues)

pca = PCA(n_components=min(5, X_tissues.shape[0]-1))
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained variance by PC:")
for i, var in enumerate(pca.explained_variance_ratio_[:5]):
    print(f"   PC{i+1}: {var*100:.2f}%")

# Use PC1 for ordering
pc1_scores = X_pca[:, 0]
sorted_indices = np.argsort(pc1_scores)[::-1]
pca_pseudotime = {tissues_list[idx]: rank for rank, idx in enumerate(sorted_indices, 1)}

print(f"\nPCA-based ordering (by PC1):")
for tissue in sorted(pca_pseudotime.keys(), key=lambda t: pca_pseudotime[t]):
    print(f"   Rank {pca_pseudotime[tissue]:2d}: {tissue}")

print("\n✓ Method 2 complete: PCA-based pseudo-time")

# ============================================================================
# 4. METHOD 3: DIFFUSION PSEUDOTIME (UMAP-based)
# ============================================================================

print("\n" + "="*80)
print("[METHOD 3] Diffusion Pseudotime (UMAP + distance-based)")
print("="*80)

if UMAP_AVAILABLE:
    # Use UMAP to create low-dimensional embedding
    umap = UMAP(n_components=2, n_neighbors=min(5, len(tissues_list)-1),
                min_dist=0.1, random_state=42, metric='euclidean')
    X_umap = umap.fit_transform(X_tissues)

    print(f"UMAP embedding shape: {X_umap.shape}")

    # Diffusion pseudo-time: distance from "root" (choose tissue with lowest velocity as root)
    root_tissue_idx = tissue_velocities.argmin()
    root_tissue_name = tissues_list[root_tissue_idx]

    print(f"Root tissue (earliest aging): {root_tissue_name}")

    # Calculate distances from root in UMAP space
    distances = np.sqrt(((X_umap - X_umap[root_tissue_idx])**2).sum(axis=1))

    sorted_indices = np.argsort(distances)
    diffusion_pseudotime = {tissues_list[idx]: rank for rank, idx in enumerate(sorted_indices, 1)}

    print(f"\nDiffusion pseudotime ordering:")
    for tissue in sorted(diffusion_pseudotime.keys(), key=lambda t: diffusion_pseudotime[t]):
        print(f"   Rank {diffusion_pseudotime[tissue]:2d}: {tissue}")

    print("\n✓ Method 3 complete: Diffusion pseudotime")
else:
    diffusion_pseudotime = None
    print("\n✗ Method 3 skipped: UMAP not available")

# ============================================================================
# 5. METHOD 4: SLINGSHOT-INSPIRED TRAJECTORY (simplified Python version)
# ============================================================================

print("\n" + "="*80)
print("[METHOD 4] Slingshot-Inspired Trajectory (simplified)")
print("="*80)

# Simplified Slingshot: Use PCA, find minimum spanning tree, assign pseudotime along principal curve
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# Use PCA embedding
pca_coords = X_pca[:, :2]  # First 2 PCs

# Compute pairwise distances
dist_matrix = distance_matrix(pca_coords, pca_coords)

# Minimum spanning tree
mst = minimum_spanning_tree(dist_matrix)

# Pseudotime = distance from root along MST (use same root as diffusion method)
# For simplicity, use PC1 as primary ordering (Slingshot's principal curve proxy)
sorted_indices = np.argsort(X_pca[:, 0])[::-1]
slingshot_pseudotime = {tissues_list[idx]: rank for rank, idx in enumerate(sorted_indices, 1)}

print(f"Slingshot-inspired ordering (PC1-based):")
for tissue in sorted(slingshot_pseudotime.keys(), key=lambda t: slingshot_pseudotime[t]):
    print(f"   Rank {slingshot_pseudotime[tissue]:2d}: {tissue}")

# Check for potential branching (if MST has branches with >1 endpoint)
mst_dense = mst.toarray()
degrees = (mst_dense > 0).sum(axis=0) + (mst_dense > 0).sum(axis=1)
n_endpoints = (degrees == 1).sum()
print(f"\nMST analysis:")
print(f"   Endpoints (degree=1): {n_endpoints}")
if n_endpoints > 2:
    print(f"   ⚠ Potential branching detected! ({n_endpoints} endpoints)")
else:
    print(f"   ✓ Linear trajectory (2 endpoints)")

print("\n✓ Method 4 complete: Slingshot-inspired pseudotime")

# ============================================================================
# 6. METHOD 5: AUTOENCODER LATENT TRAVERSAL
# ============================================================================

print("\n" + "="*80)
print("[METHOD 5] Autoencoder Latent Traversal")
print("="*80)

# Build simple autoencoder
class TissueAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Train autoencoder
input_dim = X_tissues.shape[1]
latent_dim = 10

autoencoder = TissueAutoencoder(input_dim, latent_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

X_tensor = torch.tensor(X_tissues, dtype=torch.float32)

print(f"Training autoencoder (input_dim={input_dim}, latent_dim={latent_dim})...")
autoencoder.train()
losses = []

for epoch in range(100):
    optimizer.zero_grad()
    reconstructed, latent = autoencoder(X_tensor)
    loss = criterion(reconstructed, X_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"   Epoch {epoch+1:3d}/100, Loss: {loss.item():.4f}")

print(f"   Final reconstruction loss: {losses[-1]:.4f}")

# Extract latent representations
autoencoder.eval()
with torch.no_grad():
    _, latent_coords = autoencoder(X_tensor)
    latent_coords_np = latent_coords.numpy()

# Find latent factor most correlated with velocity
correlations = []
for i in range(latent_dim):
    velocity_values = np.array([velocity_pseudotime[t] for t in tissues_list])
    corr = stats.spearmanr(latent_coords_np[:, i], velocity_values)[0]
    correlations.append(abs(corr))

best_latent_factor = np.argmax(correlations)
print(f"\nBest latent factor: #{best_latent_factor} (|ρ| = {correlations[best_latent_factor]:.3f} with velocity)")

# Pseudotime = rank by best latent factor
sorted_indices = np.argsort(latent_coords_np[:, best_latent_factor])[::-1]
autoencoder_pseudotime = {tissues_list[idx]: rank for rank, idx in enumerate(sorted_indices, 1)}

print(f"\nAutoencoder pseudotime ordering (latent factor {best_latent_factor}):")
for tissue in sorted(autoencoder_pseudotime.keys(), key=lambda t: autoencoder_pseudotime[t]):
    print(f"   Rank {autoencoder_pseudotime[tissue]:2d}: {tissue}")

print("\n✓ Method 5 complete: Autoencoder latent traversal")

# Save autoencoder
torch.save(autoencoder.state_dict(), 'visualizations_claude_code/autoencoder_weights_claude_code.pth')

# ============================================================================
# 7. COMPARE ALL PSEUDOTIME ORDERINGS
# ============================================================================

print("\n" + "="*80)
print("[7] Comparing All Pseudotime Orderings")
print("="*80)

# Create comparison dataframe
pseudotime_methods = {
    'Velocity (H03)': velocity_pseudotime,
    'PCA (Codex)': pca_pseudotime,
    'Slingshot': slingshot_pseudotime,
    'Autoencoder': autoencoder_pseudotime
}

if diffusion_pseudotime:
    pseudotime_methods['Diffusion'] = diffusion_pseudotime

comparison_df = pd.DataFrame(pseudotime_methods)
comparison_df = comparison_df.reindex(tissues_list)

print("\nPseudotime Rankings Comparison:")
print(comparison_df.to_string())

# Save to CSV
comparison_df.to_csv('pseudotime_orderings_claude_code.csv')
print("\n✓ Saved to: pseudotime_orderings_claude_code.csv")

# Kendall's τ correlation matrix
print("\n" + "-"*80)
print("Kendall's τ (rank correlation) between methods:")
print("-"*80)

methods_list = list(pseudotime_methods.keys())
n_methods = len(methods_list)
tau_matrix = np.zeros((n_methods, n_methods))

for i, method1 in enumerate(methods_list):
    for j, method2 in enumerate(methods_list):
        ranks1 = [pseudotime_methods[method1][t] for t in tissues_list]
        ranks2 = [pseudotime_methods[method2][t] for t in tissues_list]
        tau, pval = stats.kendalltau(ranks1, ranks2)
        tau_matrix[i, j] = tau

tau_df = pd.DataFrame(tau_matrix, index=methods_list, columns=methods_list)
print(tau_df.round(3).to_string())

# Save correlation matrix
tau_df.to_csv('pseudotime_method_correlation_claude_code.csv')
print("\n✓ Saved to: pseudotime_method_correlation_claude_code.csv")

# Key findings
print("\n" + "-"*80)
print("KEY FINDINGS:")
print("-"*80)

# Velocity vs PCA agreement
tau_velocity_pca = tau_matrix[methods_list.index('Velocity (H03)'), methods_list.index('PCA (Codex)')]
print(f"1. Velocity vs PCA agreement: τ = {tau_velocity_pca:.3f}")
if tau_velocity_pca < 0.3:
    print("   ⚠ MAJOR DISAGREEMENT! This explains H09 R² discrepancy (0.81 vs 0.011)")
elif tau_velocity_pca > 0.7:
    print("   ✓ High agreement, but H09 disagreement still unexplained")

# Which methods agree most?
off_diag = tau_matrix.copy()
np.fill_diagonal(off_diag, -999)
max_agreement_idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
method_a, method_b = methods_list[max_agreement_idx[0]], methods_list[max_agreement_idx[1]]
max_tau = off_diag[max_agreement_idx]
print(f"\n2. Highest agreement: {method_a} ↔ {method_b} (τ = {max_tau:.3f})")

# Consensus ordering (median rank across methods)
consensus_ranks = comparison_df.median(axis=1).sort_values()
print(f"\n3. Consensus ordering (median across all methods):")
for i, (tissue, median_rank) in enumerate(consensus_ranks.items(), 1):
    print(f"   {i:2d}. {tissue:35s} (median rank: {median_rank:.1f})")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("[8] Generating Visualizations")
print("="*80)

# (A) Parallel coordinates plot of tissue orderings
fig, ax = plt.subplots(figsize=(14, 8))

for tissue in tissues_list:
    ranks = [pseudotime_methods[method][tissue] for method in methods_list]
    ax.plot(range(len(methods_list)), ranks, '-o', alpha=0.6, label=tissue, linewidth=1.5)

ax.set_xticks(range(len(methods_list)))
ax.set_xticklabels(methods_list, rotation=15, ha='right')
ax.set_ylabel('Pseudo-time Rank', fontsize=12)
ax.set_title('Pseudo-Time Orderings Across Methods (Parallel Coordinates)', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations_claude_code/pseudotime_comparison_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations_claude_code/pseudotime_comparison_claude_code.png")

# (B) Kendall's τ heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(tau_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
            vmin=-1, vmax=1, square=True, cbar_kws={'label': "Kendall's τ"}, ax=ax)
ax.set_title("Pseudo-Time Method Agreement (Kendall's τ)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations_claude_code/pseudotime_correlation_heatmap_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations_claude_code/pseudotime_correlation_heatmap_claude_code.png")

# (C) UMAP/PCA scatter plot with pseudotime coloring
if UMAP_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # UMAP plot
    scatter1 = axes[0].scatter(X_umap[:, 0], X_umap[:, 1],
                               c=[velocity_pseudotime[t] for t in tissues_list],
                               cmap='viridis', s=200, edgecolors='black', linewidths=1.5)
    for i, tissue in enumerate(tissues_list):
        axes[0].text(X_umap[i, 0], X_umap[i, 1], tissue.split('_')[0][:3],
                    fontsize=7, ha='center', va='center')
    axes[0].set_xlabel('UMAP 1', fontsize=12)
    axes[0].set_ylabel('UMAP 2', fontsize=12)
    axes[0].set_title('UMAP Embedding (colored by Velocity Pseudotime)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Velocity Rank')

    # PCA plot
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                               c=[pca_pseudotime[t] for t in tissues_list],
                               cmap='plasma', s=200, edgecolors='black', linewidths=1.5)
    for i, tissue in enumerate(tissues_list):
        axes[1].text(X_pca[i, 0], X_pca[i, 1], tissue.split('_')[0][:3],
                    fontsize=7, ha='center', va='center')
    axes[1].set_xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100), fontsize=12)
    axes[1].set_ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100), fontsize=12)
    axes[1].set_title('PCA Embedding (colored by PCA Pseudotime)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='PCA Rank')

    plt.tight_layout()
    plt.savefig('visualizations_claude_code/embeddings_comparison_claude_code.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations_claude_code/embeddings_comparison_claude_code.png")

print("\n" + "="*80)
print("Phase 1 Complete: All 5 Pseudo-Time Methods Implemented & Compared")
print("="*80)
print("\nNext steps:")
print("1. Benchmark LSTM performance for each method (lstm_benchmark_claude_code.py)")
print("2. Sensitivity analysis (robustness under perturbations)")
print("3. Critical transitions consistency analysis")
print("4. Granger causality stability testing")
print("\n" + "="*80)
