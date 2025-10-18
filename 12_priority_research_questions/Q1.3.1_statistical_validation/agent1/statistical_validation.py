#!/usr/bin/env python3
"""
Statistical Validation of Z-Score Normalization & Cross-Study Integration
==========================================================================

Purpose: Comprehensive audit of ECM-Atlas z-score methodology with skeptical rigor.
Author: Agent 1 - Statistical Validator
Date: 2025-10-17

Tasks:
1. Z-score methodology audit (assumptions, implementation)
2. Sensitivity analysis (alternative normalizations)
3. Batch effect detection (PCA, ICC)
4. Permutation testing (null hypothesis validation)
5. Diagnostic plots (Q-Q, residual, batch effects)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro, levene, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from statsmodels.stats.multitest import multipletests
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
print("STATISTICAL VALIDATION OF ECM-ATLAS Z-SCORE METHODOLOGY")
print("="*80)
print(f"\nData source: {DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Diagnostics directory: {DIAGNOSTICS_DIR}")

# ============================================================================
# SECTION 1: DATA LOADING & EXPLORATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: DATA LOADING & INITIAL EXPLORATION")
print("="*80)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nStudies present: {df['Study_ID'].nunique()}")
print(df['Study_ID'].value_counts().to_string())

print(f"\nColumns: {list(df.columns)}")

# Key statistics
print(f"\nKey dataset characteristics:")
print(f"  Unique proteins: {df['Protein_ID'].nunique():,}")
print(f"  Unique genes: {df['Gene_Symbol'].nunique():,}")
print(f"  Tissues/compartments: {df['Tissue'].nunique() if 'Tissue' in df.columns else 'N/A'}")

# Check z-score columns
zscore_cols = ['Zscore_Young', 'Zscore_Old', 'Zscore_Delta']
for col in zscore_cols:
    if col in df.columns:
        valid_count = df[col].notna().sum()
        pct = valid_count / len(df) * 100
        print(f"  {col}: {valid_count:,}/{len(df):,} ({pct:.1f}%) non-NaN")

# ============================================================================
# SECTION 2: Z-SCORE ASSUMPTION VALIDATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Z-SCORE ASSUMPTIONS AUDIT")
print("="*80)

print("\n--- ASSUMPTION 1: NORMALITY ---")
print("Testing if z-scores follow standard normal distribution N(0,1)")

# Test normality for each study separately
normality_results = []

for study in df['Study_ID'].unique():
    study_data = df[df['Study_ID'] == study]

    for col in ['Zscore_Young', 'Zscore_Old']:
        if col not in study_data.columns:
            continue

        values = study_data[col].dropna()

        if len(values) < 3:
            continue

        # Shapiro-Wilk test (if n < 5000)
        if len(values) < 5000:
            stat, p_value = shapiro(values)
        else:
            # For large samples, use Kolmogorov-Smirnov test
            stat, p_value = stats.kstest(values, 'norm', args=(0, 1))

        # Calculate moments
        mean_val = values.mean()
        std_val = values.std()
        skew_val = skew(values)
        kurt_val = kurtosis(values)

        normality_results.append({
            'Study': study,
            'Column': col,
            'N': len(values),
            'Mean': mean_val,
            'Std': std_val,
            'Skewness': skew_val,
            'Kurtosis': kurt_val,
            'Test_Stat': stat,
            'P_Value': p_value,
            'Normal': p_value > 0.05
        })

df_normality = pd.DataFrame(normality_results)
print(f"\nNormality test results (p > 0.05 = normal):")
print(df_normality.to_string(index=False))

# Summary
normal_count = df_normality['Normal'].sum()
total_tests = len(df_normality)
print(f"\nSummary: {normal_count}/{total_tests} ({normal_count/total_tests*100:.1f}%) passed normality test")

# Save results
df_normality.to_csv(f'{OUTPUT_DIR}/normality_test_results.csv', index=False)

print("\n--- ASSUMPTION 2: HOMOSCEDASTICITY (Equal Variance) ---")
print("Testing if variance is constant across studies")

# Levene's test for equal variances
homoscedasticity_results = []

for col in ['Zscore_Young', 'Zscore_Old']:
    if col not in df.columns:
        continue

    # Collect data by study
    study_groups = []
    study_names = []

    for study in df['Study_ID'].unique():
        values = df[df['Study_ID'] == study][col].dropna()
        if len(values) >= 3:
            study_groups.append(values)
            study_names.append(study)

    if len(study_groups) >= 2:
        stat, p_value = levene(*study_groups)

        # Calculate variance for each study
        variances = [g.var() for g in study_groups]

        homoscedasticity_results.append({
            'Column': col,
            'N_Studies': len(study_groups),
            'Levene_Stat': stat,
            'P_Value': p_value,
            'Homoscedastic': p_value > 0.05,
            'Min_Variance': min(variances),
            'Max_Variance': max(variances),
            'Variance_Ratio': max(variances) / min(variances)
        })

df_homoscedasticity = pd.DataFrame(homoscedasticity_results)
print(df_homoscedasticity.to_string(index=False))

df_homoscedasticity.to_csv(f'{OUTPUT_DIR}/homoscedasticity_test_results.csv', index=False)

print("\n--- ASSUMPTION 3: INDEPENDENCE ---")
print("Checking for within-study correlations that shouldn't exist")

# For each study, check if z-scores are correlated with protein abundance
independence_results = []

for study in df['Study_ID'].unique():
    study_data = df[df['Study_ID'] == study]

    # Check correlation between raw abundance and z-score (should be close to 0 if independent)
    for age in ['Young', 'Old']:
        abundance_col = f'Abundance_{age}'
        zscore_col = f'Zscore_{age}'

        if abundance_col in study_data.columns and zscore_col in study_data.columns:
            valid = study_data[[abundance_col, zscore_col]].dropna()

            if len(valid) >= 3:
                corr, p_value = pearsonr(valid[abundance_col], valid[zscore_col])

                independence_results.append({
                    'Study': study,
                    'Age_Group': age,
                    'N': len(valid),
                    'Abundance_Zscore_Corr': corr,
                    'P_Value': p_value,
                    'Independent': abs(corr) < 0.1  # Arbitrary threshold
                })

df_independence = pd.DataFrame(independence_results)
print(df_independence.to_string(index=False))

df_independence.to_csv(f'{OUTPUT_DIR}/independence_test_results.csv', index=False)

# ============================================================================
# SECTION 3: DIAGNOSTIC PLOTS
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: GENERATING DIAGNOSTIC PLOTS")
print("="*80)

# Plot 1: Q-Q plots for each study
print("\n--- Generating Q-Q plots ---")

studies = df['Study_ID'].unique()
n_studies = len(studies)
n_cols = 3
n_rows = int(np.ceil(n_studies / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for idx, study in enumerate(studies):
    study_data = df[df['Study_ID'] == study]

    # Combine Young and Old z-scores
    zscores = pd.concat([
        study_data['Zscore_Young'].dropna(),
        study_data['Zscore_Old'].dropna()
    ])

    if len(zscores) > 0:
        stats.probplot(zscores, dist="norm", plot=axes[idx])
        axes[idx].set_title(f'{study}\n(n={len(zscores)})', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(studies), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{DIAGNOSTICS_DIR}/qq_plots_by_study.png', dpi=300, bbox_inches='tight')
print(f"  Saved: qq_plots_by_study.png")
plt.close()

# Plot 2: Distribution of z-scores
print("\n--- Generating z-score distribution plots ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, col in enumerate(['Zscore_Young', 'Zscore_Old']):
    if col in df.columns:
        values = df[col].dropna()

        # Histogram with theoretical normal
        axes[idx].hist(values, bins=100, density=True, alpha=0.7, label='Observed')

        # Theoretical N(0,1)
        x = np.linspace(-5, 5, 100)
        axes[idx].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')

        axes[idx].set_xlabel('Z-Score')
        axes[idx].set_ylabel('Density')
        axes[idx].set_title(f'{col}\nMean={values.mean():.4f}, Std={values.std():.4f}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DIAGNOSTICS_DIR}/zscore_distributions.png', dpi=300, bbox_inches='tight')
print(f"  Saved: zscore_distributions.png")
plt.close()

# Plot 3: Variance by study
print("\n--- Generating variance comparison plot ---")

variance_data = []
for study in df['Study_ID'].unique():
    study_data = df[df['Study_ID'] == study]

    for col in ['Zscore_Young', 'Zscore_Old']:
        if col in study_data.columns:
            values = study_data[col].dropna()
            if len(values) > 0:
                variance_data.append({
                    'Study': study,
                    'Age': col.replace('Zscore_', ''),
                    'Variance': values.var(),
                    'Std': values.std()
                })

df_variance = pd.DataFrame(variance_data)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.barplot(data=df_variance, x='Study', y='Variance', hue='Age', ax=ax)
ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Expected (σ²=1)')
ax.set_ylabel('Variance')
ax.set_xlabel('Study')
ax.set_title('Z-Score Variance by Study (Should be ≈ 1.0)')
ax.legend()
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{DIAGNOSTICS_DIR}/variance_by_study.png', dpi=300, bbox_inches='tight')
print(f"  Saved: variance_by_study.png")
plt.close()

# ============================================================================
# SECTION 4: SENSITIVITY ANALYSIS - ALTERNATIVE NORMALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: SENSITIVITY ANALYSIS - ALTERNATIVE NORMALIZATIONS")
print("="*80)

print("\n--- Testing alternative normalization methods ---")

# Focus on one study for detailed comparison
test_study = df['Study_ID'].value_counts().index[0]
study_data = df[df['Study_ID'] == test_study].copy()

print(f"\nTest study: {test_study} (n={len(study_data)} proteins)")

# Get a tissue/compartment with enough data
if 'Tissue' in study_data.columns:
    tissue = study_data['Tissue'].value_counts().index[0]
    compartment_data = study_data[study_data['Tissue'] == tissue].copy()
    print(f"Test compartment: {tissue} (n={len(compartment_data)} proteins)")
else:
    compartment_data = study_data.copy()

# Get raw abundances - need matched pairs
valid_data = compartment_data[['Abundance_Young', 'Abundance_Old']].dropna()
young_abundance = valid_data['Abundance_Young']
old_abundance = valid_data['Abundance_Old']

print(f"\nYoung samples: {len(young_abundance)} proteins")
print(f"Old samples: {len(old_abundance)} proteins")

# Method 1: Standard z-score (current method)
zscore_standard_young = (young_abundance - young_abundance.mean()) / young_abundance.std()
zscore_standard_old = (old_abundance - old_abundance.mean()) / old_abundance.std()

# Method 2: Median-centered robust z-score
median_young = young_abundance.median()
mad_young = np.median(np.abs(young_abundance - median_young))
zscore_robust_young = (young_abundance - median_young) / (1.4826 * mad_young)

median_old = old_abundance.median()
mad_old = np.median(np.abs(old_abundance - median_old))
zscore_robust_old = (old_abundance - median_old) / (1.4826 * mad_old)

# Method 3: Quantile normalization
quantile_transformer = QuantileTransformer(output_distribution='normal')
zscore_quantile_young = quantile_transformer.fit_transform(young_abundance.values.reshape(-1, 1)).flatten()
zscore_quantile_old = quantile_transformer.fit_transform(old_abundance.values.reshape(-1, 1)).flatten()

# Method 4: Rank-based (percentile)
rank_young = stats.rankdata(young_abundance) / (len(young_abundance) + 1)  # Avoid 0 and 1
rank_young = np.clip(rank_young, 0.001, 0.999)  # Clip to avoid inf
zscore_rank_young = stats.norm.ppf(rank_young)

rank_old = stats.rankdata(old_abundance) / (len(old_abundance) + 1)
rank_old = np.clip(rank_old, 0.001, 0.999)
zscore_rank_old = stats.norm.ppf(rank_old)

# Compare methods
sensitivity_results = {
    'Method': ['Standard', 'Robust (MAD)', 'Quantile', 'Rank-based'],
    'Young_Mean': [
        zscore_standard_young.mean(),
        zscore_robust_young.mean(),
        zscore_quantile_young.mean(),
        zscore_rank_young.mean()
    ],
    'Young_Std': [
        zscore_standard_young.std(),
        zscore_robust_young.std(),
        zscore_quantile_young.std(),
        zscore_rank_young.std()
    ],
    'Old_Mean': [
        zscore_standard_old.mean(),
        zscore_robust_old.mean(),
        zscore_quantile_old.mean(),
        zscore_rank_old.mean()
    ],
    'Old_Std': [
        zscore_standard_old.std(),
        zscore_robust_old.std(),
        zscore_quantile_old.std(),
        zscore_rank_old.std()
    ]
}

df_sensitivity = pd.DataFrame(sensitivity_results)
print(f"\nNormalization method comparison:")
print(df_sensitivity.to_string(index=False))

df_sensitivity.to_csv(f'{OUTPUT_DIR}/sensitivity_analysis_methods.csv', index=False)

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

methods = [
    ('Standard Z-Score', zscore_standard_young, zscore_standard_old),
    ('Robust (MAD)', zscore_robust_young, zscore_robust_old),
    ('Quantile Transform', zscore_quantile_young, zscore_quantile_old),
    ('Rank-Based', zscore_rank_young, zscore_rank_old)
]

for idx, (method_name, young_vals, old_vals) in enumerate(methods):
    ax = axes.flatten()[idx]

    ax.scatter(young_vals, old_vals, alpha=0.3, s=20)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Young Z-Score')
    ax.set_ylabel('Old Z-Score')
    ax.set_title(f'{method_name}\nCorr={np.corrcoef(young_vals, old_vals)[0,1]:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{DIAGNOSTICS_DIR}/sensitivity_normalization_methods.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: sensitivity_normalization_methods.png")
plt.close()

print("\n=== VALIDATION SCRIPT COMPLETE (Part 1/2) ===")
print("\nNext: Run permutation testing and batch effect analysis...")
