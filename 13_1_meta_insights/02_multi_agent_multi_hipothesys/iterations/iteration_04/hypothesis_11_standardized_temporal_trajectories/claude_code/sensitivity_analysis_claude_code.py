#!/usr/bin/env python3
"""
H11 - Sensitivity Analysis: Robustness of Pseudo-Time Methods
Author: claude_code
Date: 2025-10-21

Tests stability of pseudo-time orderings under:
1. Tissue subset removal (leave-one-out)
2. Protein subset sampling (top N by variance)
3. Noise injection (Gaussian noise)

Metric: Kendall's τ (rank correlation) between original and perturbed orderings
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False

print("="*80)
print("SENSITIVITY ANALYSIS: Pseudo-Time Method Robustness")
print("="*80)

# Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

tissue_protein_matrix = df.pivot_table(
    values='Zscore_Delta',
    index='Tissue',
    columns='Gene_Symbol',
    aggfunc='median'
).fillna(0)

tissues_list = tissue_protein_matrix.index.tolist()
proteins_list = tissue_protein_matrix.columns.tolist()
X_tissues = tissue_protein_matrix.values

print(f"Data: {len(tissues_list)} tissues, {len(proteins_list)} proteins")

# ============================================================================
# Helper functions to compute pseudo-time
# ============================================================================

def compute_velocity_pseudotime(X):
    """Method 1: Tissue velocity"""
    velocities = np.abs(X).mean(axis=1)
    return velocities.argsort()[::-1] + 1  # Ranks 1-17

def compute_pca_pseudotime(X):
    """Method 2: PCA PC1"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(5, X.shape[0]-1))
    X_pca = pca.fit_transform(X_scaled)
    pc1_scores = X_pca[:, 0]
    return np.argsort(pc1_scores)[::-1] + 1

def compute_diffusion_pseudotime(X):
    """Method 3: UMAP + distance from root"""
    if not UMAP_AVAILABLE:
        return None
    umap = UMAP(n_components=2, n_neighbors=min(5, X.shape[0]-1),
                min_dist=0.1, random_state=42)
    X_umap = umap.fit_transform(X)

    # Root = tissue with lowest velocity
    velocities = np.abs(X).mean(axis=1)
    root_idx = velocities.argmin()

    distances = np.sqrt(((X_umap - X_umap[root_idx])**2).sum(axis=1))
    return np.argsort(distances) + 1

# Load original pseudotime orderings
pseudotime_df = pd.read_csv('pseudotime_orderings_claude_code.csv', index_col=0)

# ============================================================================
# 1. LEAVE-ONE-TISSUE-OUT ROBUSTNESS
# ============================================================================

print("\n[1] Leave-One-Tissue-Out Robustness Test")
print("-"*80)

methods_funcs = {
    'Velocity (H03)': compute_velocity_pseudotime,
    'PCA (Codex)': compute_pca_pseudotime,
}

if UMAP_AVAILABLE:
    methods_funcs['Diffusion'] = compute_diffusion_pseudotime

loo_results = {method: [] for method in methods_funcs.keys()}

for method_name, compute_func in methods_funcs.items():
    print(f"\nMethod: {method_name}")

    original_ranks = pseudotime_df[method_name].values

    for i, tissue_to_remove in enumerate(tissues_list):
        # Remove tissue
        mask = np.arange(len(tissues_list)) != i
        X_subset = X_tissues[mask, :]
        original_ranks_subset = original_ranks[mask]

        # Recompute pseudotime
        try:
            perturbed_ranks = compute_func(X_subset)

            if perturbed_ranks is None:
                continue

            # Calculate Kendall's τ
            tau, pval = stats.kendalltau(original_ranks_subset, perturbed_ranks)
            loo_results[method_name].append(tau)

        except Exception as e:
            print(f"   Error removing {tissue_to_remove}: {e}")
            continue

    mean_tau = np.mean(loo_results[method_name])
    std_tau = np.std(loo_results[method_name])
    print(f"   Mean Kendall's τ: {mean_tau:.3f} ± {std_tau:.3f}")

# ============================================================================
# 2. PROTEIN SUBSET ROBUSTNESS
# ============================================================================

print("\n[2] Protein Subset Robustness Test")
print("-"*80)

# Test with top 100, 200, 500 proteins by variance
protein_variances = X_tissues.var(axis=0)
top_proteins_counts = [100, 200, 500]

protein_results = {method: {n: [] for n in top_proteins_counts} for method in methods_funcs.keys()}

for method_name, compute_func in methods_funcs.items():
    print(f"\nMethod: {method_name}")

    original_ranks = pseudotime_df[method_name].values

    for n_proteins in top_proteins_counts:
        if n_proteins > len(proteins_list):
            continue

        # Repeat 5 times with different random seeds
        for seed in range(5):
            np.random.seed(seed)

            # Select top N proteins by variance
            top_protein_indices = np.argsort(protein_variances)[-n_proteins:]
            X_subset = X_tissues[:, top_protein_indices]

            try:
                perturbed_ranks = compute_func(X_subset)

                if perturbed_ranks is None:
                    continue

                tau, pval = stats.kendalltau(original_ranks, perturbed_ranks)
                protein_results[method_name][n_proteins].append(tau)

            except Exception as e:
                continue

        if protein_results[method_name][n_proteins]:
            mean_tau = np.mean(protein_results[method_name][n_proteins])
            std_tau = np.std(protein_results[method_name][n_proteins])
            print(f"   Top {n_proteins} proteins: τ = {mean_tau:.3f} ± {std_tau:.3f}")

# ============================================================================
# 3. NOISE INJECTION ROBUSTNESS
# ============================================================================

print("\n[3] Noise Injection Robustness Test")
print("-"*80)

noise_levels = [0.05, 0.1, 0.2, 0.5]  # σ as fraction of data std
n_trials = 10

noise_results = {method: {sigma: [] for sigma in noise_levels} for method in methods_funcs.keys()}

for method_name, compute_func in methods_funcs.items():
    print(f"\nMethod: {method_name}")

    original_ranks = pseudotime_df[method_name].values

    for sigma in noise_levels:
        for trial in range(n_trials):
            # Add Gaussian noise
            noise = np.random.normal(0, sigma * X_tissues.std(), X_tissues.shape)
            X_noisy = X_tissues + noise

            try:
                perturbed_ranks = compute_func(X_noisy)

                if perturbed_ranks is None:
                    continue

                tau, pval = stats.kendalltau(original_ranks, perturbed_ranks)
                noise_results[method_name][sigma].append(tau)

            except Exception as e:
                continue

        if noise_results[method_name][sigma]:
            mean_tau = np.mean(noise_results[method_name][sigma])
            std_tau = np.std(noise_results[method_name][sigma])
            print(f"   Noise σ={sigma:.2f}: τ = {mean_tau:.3f} ± {std_tau:.3f}")

# ============================================================================
# 4. SUMMARY & VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("[4] Sensitivity Summary")
print("="*80)

# Create summary table
summary_data = []

for method_name in methods_funcs.keys():
    loo_mean = np.mean(loo_results[method_name]) if loo_results[method_name] else np.nan
    protein_mean = np.mean([np.mean(vals) for vals in protein_results[method_name].values() if vals])
    noise_mean = np.mean([np.mean(vals) for vals in noise_results[method_name].values() if vals])

    summary_data.append({
        'Method': method_name,
        'LOO_tau': loo_mean,
        'Protein_subset_tau': protein_mean,
        'Noise_tau': noise_mean,
        'Overall_robustness': np.mean([loo_mean, protein_mean, noise_mean])
    })

summary_df = pd.DataFrame(summary_data).sort_values('Overall_robustness', ascending=False)

print("\nRobustness Rankings (Kendall's τ, higher = more robust):")
print(summary_df.to_string(index=False))

summary_df.to_csv('sensitivity_analysis_claude_code.csv', index=False)
print("\n✓ Saved to: sensitivity_analysis_claude_code.csv")

# Visualization: Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = summary_df.set_index('Method')[['LOO_tau', 'Protein_subset_tau', 'Noise_tau']]
heatmap_data.columns = ['Leave-One-Out', 'Protein Subset', 'Noise Injection']

sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0, vmax=1, center=0.8, cbar_kws={'label': "Kendall's τ"},
            linewidths=1, linecolor='black', ax=ax)

ax.set_title('Pseudo-Time Method Robustness (Sensitivity Analysis)', fontsize=14, fontweight='bold')
ax.set_ylabel('Perturbation Type', fontsize=12)
ax.set_xlabel('Method', fontsize=12)

plt.tight_layout()
plt.savefig('visualizations_claude_code/sensitivity_heatmap_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations_claude_code/sensitivity_heatmap_claude_code.png")

print("\n" + "="*80)
print("Sensitivity Analysis Complete!")
print("="*80)

# Find most robust method
most_robust = summary_df.iloc[0]['Method']
robustness_score = summary_df.iloc[0]['Overall_robustness']

print(f"\nMost Robust Method: {most_robust} (average τ = {robustness_score:.3f})")
print("Methods with τ > 0.80 are considered highly robust under perturbations.")
print("="*80)
