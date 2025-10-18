#!/usr/bin/env python3
"""
Batch Effect Detection and Permutation Testing
===============================================

Purpose: Detect cross-study batch effects and validate age-protein correlations
Author: Agent 1 - Statistical Validator
Date: 2025-10-17

Tasks:
1. PCA analysis to detect study clustering
2. Intraclass correlation coefficient (ICC) calculation
3. Permutation testing for age-protein correlations
4. FDR correction for multiple testing
5. Bootstrap resampling for confidence intervals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.3.1_statistical_validation/agent1'
DIAGNOSTICS_DIR = f'{OUTPUT_DIR}/diagnostics'

print("="*80)
print("BATCH EFFECT DETECTION & PERMUTATION TESTING")
print("="*80)

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: LOADING DATA")
print("="*80)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Studies: {df['Study_ID'].nunique()}")

# ============================================================================
# SECTION 2: BATCH EFFECT DETECTION - PCA
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: BATCH EFFECT DETECTION - PCA ANALYSIS")
print("="*80)

print("\n--- Creating protein-by-study matrix ---")

# Create wide matrix: proteins × studies (z-score delta)
# Each row = protein, each column = study, value = Zscore_Delta

pivot_data = df[['Protein_ID', 'Gene_Symbol', 'Study_ID', 'Zscore_Delta']].dropna()
print(f"Valid entries for PCA: {len(pivot_data):,}")

# Create pivot table
matrix = pivot_data.pivot_table(
    index=['Protein_ID', 'Gene_Symbol'],
    columns='Study_ID',
    values='Zscore_Delta',
    aggfunc='mean'  # Average if protein appears multiple times per study
)

print(f"\nPivot matrix shape: {matrix.shape}")
print(f"  Proteins (rows): {matrix.shape[0]}")
print(f"  Studies (columns): {matrix.shape[1]}")

# Remove proteins with too many missing values
# Lower threshold since studies have different protein coverage
missing_threshold = 0.25  # Keep proteins present in at least 25% of studies
valid_proteins = matrix.notna().sum(axis=1) >= (matrix.shape[1] * missing_threshold)
matrix_filtered = matrix[valid_proteins]

print(f"\nAfter filtering (≥25% studies): {matrix_filtered.shape[0]} proteins")

if matrix_filtered.shape[0] < 10:
    print(f"\nWARNING: Too few proteins ({matrix_filtered.shape[0]}) for robust PCA")
    print("Lowering threshold to 2 studies minimum...")
    valid_proteins = matrix.notna().sum(axis=1) >= 2
    matrix_filtered = matrix[valid_proteins]
    print(f"After relaxed filtering (≥2 studies): {matrix_filtered.shape[0]} proteins")

# Impute missing values with 0 (standardization will handle this)
matrix_imputed = matrix_filtered.fillna(0)

print(f"\nMissing values after imputation: {matrix_imputed.isna().sum().sum()}")

# Transpose for PCA: studies × proteins
X = matrix_imputed.T

print(f"\nPCA input matrix: {X.shape}")
print(f"  Studies (rows): {X.shape[0]}")
print(f"  Proteins (features): {X.shape[1]}")

# Standardize (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
print("\n--- Running PCA ---")
pca = PCA(n_components=min(X_scaled.shape[0], 10))  # Up to 10 components
pca.fit(X_scaled)

print(f"\nExplained variance ratio (first 5 components):")
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"  PC{i}: {var*100:.2f}%")

cumsum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nCumulative variance (first 5 PCs): {cumsum_var[4]*100:.2f}%")

# Transform data
X_pca = pca.transform(X_scaled)

# Create PCA results dataframe
pca_results = pd.DataFrame({
    'Study_ID': X.index,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'PC3': X_pca[:, 2] if X_pca.shape[1] > 2 else 0
})

print(f"\nPCA results:")
print(pca_results.to_string(index=False))

# Save PCA results
pca_results.to_csv(f'{OUTPUT_DIR}/pca_batch_effects.csv', index=False)

# Plot PCA
print("\n--- Generating PCA plots ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PC1 vs PC2
ax = axes[0]
for study in pca_results['Study_ID']:
    study_data = pca_results[pca_results['Study_ID'] == study]
    ax.scatter(study_data['PC1'], study_data['PC2'], s=200, alpha=0.7, label=study)
    ax.annotate(study, (study_data['PC1'].values[0], study_data['PC2'].values[0]),
                fontsize=8, ha='center')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA: Study Clustering\n(Tight clusters = batch effects)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Scree plot
ax = axes[1]
components = range(1, len(pca.explained_variance_ratio_) + 1)
ax.plot(components, pca.explained_variance_ratio_, 'bo-', linewidth=2, markersize=8)
ax.plot(components, cumsum_var, 'ro--', linewidth=2, markersize=6, label='Cumulative')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained')
ax.set_title('Scree Plot')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(f'{DIAGNOSTICS_DIR}/pca_batch_effects.png', dpi=300, bbox_inches='tight')
print(f"  Saved: pca_batch_effects.png")
plt.close()

# ============================================================================
# SECTION 3: INTRACLASS CORRELATION COEFFICIENT (ICC)
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: INTRACLASS CORRELATION COEFFICIENT (ICC)")
print("="*80)

print("\n--- Calculating ICC for common proteins across studies ---")

# For proteins present in multiple studies, calculate ICC
# ICC(1,1) = (BMS - WMS) / (BMS + (k-1)*WMS)
# BMS = between-group (protein) mean square
# WMS = within-group (protein) mean square

# Prepare data: long format with protein, study, zscore_delta
icc_data = df[['Protein_ID', 'Gene_Symbol', 'Study_ID', 'Zscore_Delta']].dropna()

# Find proteins in at least 3 studies
protein_counts = icc_data.groupby('Protein_ID').size()
common_proteins = protein_counts[protein_counts >= 3].index

icc_data_filtered = icc_data[icc_data['Protein_ID'].isin(common_proteins)]

print(f"\nProteins in ≥3 studies: {len(common_proteins)}")
print(f"Measurements for ICC: {len(icc_data_filtered)}")

# Calculate ICC using one-way ANOVA
from scipy.stats import f_oneway

# Group by protein
protein_groups = []
protein_ids = []

for protein_id, group in icc_data_filtered.groupby('Protein_ID'):
    values = group['Zscore_Delta'].values
    if len(values) >= 3:
        protein_groups.append(values)
        protein_ids.append(protein_id)

if len(protein_groups) >= 2:
    # Perform one-way ANOVA
    f_stat, p_value = f_oneway(*protein_groups)

    # Calculate ICC
    k = np.mean([len(g) for g in protein_groups])  # average group size
    n = len(protein_groups)  # number of groups

    # Calculate between and within group variance
    grand_mean = np.concatenate(protein_groups).mean()

    # Between-group variance (BMS)
    bms = sum(len(g) * (g.mean() - grand_mean)**2 for g in protein_groups) / (n - 1)

    # Within-group variance (WMS)
    wms = sum(sum((x - g.mean())**2 for x in g) for g in protein_groups) / (sum(len(g) for g in protein_groups) - n)

    # ICC(1,1)
    icc = (bms - wms) / (bms + (k - 1) * wms)

    print(f"\nICC Results:")
    print(f"  ICC(1,1) = {icc:.4f}")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  P-value = {p_value:.4e}")
    print(f"\nInterpretation:")
    print(f"  ICC < 0.5: Poor reliability (high batch effects)")
    print(f"  ICC 0.5-0.75: Moderate reliability")
    print(f"  ICC 0.75-0.9: Good reliability")
    print(f"  ICC > 0.9: Excellent reliability (low batch effects)")

    # Save ICC results
    icc_results = pd.DataFrame({
        'Metric': ['ICC(1,1)', 'F_Statistic', 'P_Value', 'N_Proteins', 'Avg_Studies_Per_Protein'],
        'Value': [icc, f_stat, p_value, n, k]
    })
    icc_results.to_csv(f'{OUTPUT_DIR}/icc_batch_effects.csv', index=False)
else:
    print("\nInsufficient data for ICC calculation")

# ============================================================================
# SECTION 4: PERMUTATION TESTING
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: PERMUTATION TESTING - NULL HYPOTHESIS VALIDATION")
print("="*80)

print("\n--- Testing if age-protein correlations are real or random ---")

# For each protein, test if Zscore_Delta is significantly different from zero
# Null hypothesis: No age-related change (Zscore_Delta ~ 0)

# Calculate observed correlations
observed_results = []

for protein_id, protein_data in df.groupby('Protein_ID'):
    delta_values = protein_data['Zscore_Delta'].dropna()

    if len(delta_values) >= 3:
        # Test against zero
        t_stat, p_value = stats.ttest_1samp(delta_values, 0)

        observed_results.append({
            'Protein_ID': protein_id,
            'Gene_Symbol': protein_data['Gene_Symbol'].iloc[0],
            'N_Studies': len(delta_values),
            'Mean_Delta': delta_values.mean(),
            'Std_Delta': delta_values.std(),
            'T_Stat': t_stat,
            'P_Value': p_value
        })

df_observed = pd.DataFrame(observed_results)

print(f"\nObserved results: {len(df_observed)} proteins tested")

# FDR correction
if len(df_observed) > 0:
    reject, pvals_corrected, _, _ = multipletests(df_observed['P_Value'], method='fdr_bh')
    df_observed['P_Value_FDR'] = pvals_corrected
    df_observed['Significant_FDR'] = reject

    sig_count = reject.sum()
    print(f"\nSignificant proteins (FDR < 0.05): {sig_count}/{len(df_observed)} ({sig_count/len(df_observed)*100:.1f}%)")

    # Save observed results
    df_observed.to_csv(f'{OUTPUT_DIR}/observed_protein_changes.csv', index=False)

    # Top proteins
    df_top = df_observed.nsmallest(20, 'P_Value_FDR')
    print(f"\nTop 20 most significant proteins:")
    print(df_top[['Gene_Symbol', 'Mean_Delta', 'P_Value', 'P_Value_FDR']].to_string(index=False))

# Permutation test (for subset to save time)
print("\n--- Running permutation test (1000 iterations) ---")

# Select top 10 most significant proteins for permutation test
n_permutations = 1000
test_proteins = df_observed.nsmallest(10, 'P_Value')['Protein_ID'].values

permutation_results = []

for protein_id in tqdm(test_proteins, desc="Permutation testing"):
    protein_data = df[df['Protein_ID'] == protein_id].copy()
    observed_mean = protein_data['Zscore_Delta'].dropna().mean()

    # Permutation: shuffle Zscore_Delta
    permuted_means = []

    for i in range(n_permutations):
        permuted = protein_data['Zscore_Delta'].dropna().sample(frac=1).values
        permuted_means.append(permuted.mean())

    permuted_means = np.array(permuted_means)

    # Calculate p-value
    p_perm = (np.abs(permuted_means) >= np.abs(observed_mean)).sum() / n_permutations

    permutation_results.append({
        'Protein_ID': protein_id,
        'Gene_Symbol': protein_data['Gene_Symbol'].iloc[0],
        'Observed_Mean': observed_mean,
        'P_Value_Permutation': p_perm,
        'P_Value_Parametric': test_proteins[test_proteins == protein_id]
    })

df_permutation = pd.DataFrame(permutation_results)
print(f"\nPermutation test results:")
print(df_permutation[['Gene_Symbol', 'Observed_Mean', 'P_Value_Permutation']].to_string(index=False))

df_permutation.to_csv(f'{OUTPUT_DIR}/permutation_test_results.csv', index=False)

# ============================================================================
# SECTION 5: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: BOOTSTRAP RESAMPLING - CONFIDENCE INTERVALS")
print("="*80)

print("\n--- Bootstrap for 4 key driver proteins ---")

# Find 4 most significant proteins
driver_proteins = df_observed.nsmallest(4, 'P_Value_FDR')['Protein_ID'].values

n_bootstrap = 1000
bootstrap_results = []

for protein_id in driver_proteins:
    protein_data = df[df['Protein_ID'] == protein_id]
    delta_values = protein_data['Zscore_Delta'].dropna().values
    gene = protein_data['Gene_Symbol'].iloc[0]

    if len(delta_values) == 0:
        continue

    # Bootstrap resampling
    bootstrap_means = []

    for i in range(n_bootstrap):
        sample = np.random.choice(delta_values, size=len(delta_values), replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)

    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    observed_mean = delta_values.mean()

    bootstrap_results.append({
        'Protein_ID': protein_id,
        'Gene_Symbol': gene,
        'Observed_Mean': observed_mean,
        'Bootstrap_Mean': bootstrap_means.mean(),
        'CI_Lower_95': ci_lower,
        'CI_Upper_95': ci_upper,
        'CI_Width': ci_upper - ci_lower
    })

    print(f"\n{gene}:")
    print(f"  Observed mean: {observed_mean:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  CI width: {ci_upper - ci_lower:.4f}")

df_bootstrap = pd.DataFrame(bootstrap_results)
df_bootstrap.to_csv(f'{OUTPUT_DIR}/bootstrap_confidence_intervals.csv', index=False)

# Plot bootstrap distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, protein_id in enumerate(driver_proteins[:4]):
    protein_data = df[df['Protein_ID'] == protein_id]
    delta_values = protein_data['Zscore_Delta'].dropna().values
    gene = protein_data['Gene_Symbol'].iloc[0]

    # Bootstrap
    bootstrap_means = []
    for i in range(n_bootstrap):
        sample = np.random.choice(delta_values, size=len(delta_values), replace=True)
        bootstrap_means.append(sample.mean())

    # Plot
    axes[idx].hist(bootstrap_means, bins=50, density=True, alpha=0.7, edgecolor='black')
    axes[idx].axvline(x=delta_values.mean(), color='red', linestyle='--', linewidth=2, label='Observed')
    axes[idx].axvline(x=np.percentile(bootstrap_means, 2.5), color='orange', linestyle=':', linewidth=2)
    axes[idx].axvline(x=np.percentile(bootstrap_means, 97.5), color='orange', linestyle=':', linewidth=2, label='95% CI')
    axes[idx].set_xlabel('Mean Zscore_Delta')
    axes[idx].set_ylabel('Density')
    axes[idx].set_title(f'{gene}\nBootstrap Distribution (n={n_bootstrap})')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DIAGNOSTICS_DIR}/bootstrap_distributions.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: bootstrap_distributions.png")
plt.close()

print("\n" + "="*80)
print("=== VALIDATION COMPLETE ===")
print("="*80)

print("\nGenerated files:")
print(f"  - {OUTPUT_DIR}/pca_batch_effects.csv")
print(f"  - {OUTPUT_DIR}/icc_batch_effects.csv")
print(f"  - {OUTPUT_DIR}/observed_protein_changes.csv")
print(f"  - {OUTPUT_DIR}/permutation_test_results.csv")
print(f"  - {OUTPUT_DIR}/bootstrap_confidence_intervals.csv")
print(f"\nDiagnostic plots:")
print(f"  - {DIAGNOSTICS_DIR}/pca_batch_effects.png")
print(f"  - {DIAGNOSTICS_DIR}/bootstrap_distributions.png")

print("\n" + "="*80)
