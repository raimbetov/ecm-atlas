#!/usr/bin/env python3
"""
Alternative Normalization Methods Analysis
Compares 5 normalization approaches for ECM aging data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import quantile_transform
import warnings
warnings.filterwarnings('ignore')

# For mixed-effects models
try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available for mixed-effects models")

# For ComBat batch correction
try:
    from combat.pycombat import pycombat
    HAS_COMBAT = False  # pycombat often has issues, we'll implement our own
except ImportError:
    HAS_COMBAT = False

print("=" * 80)
print("ALTERNATIVE NORMALIZATION METHODS ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. LOADING DATA...")
data_path = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Studies: {df['Study_ID'].nunique()}")
print(f"Proteins: {df['Canonical_Gene_Symbol'].nunique()}")
print(f"Tissues: {df['Tissue'].nunique()}")

# Filter to rows with both old and young measurements
df_paired = df[(df['Abundance_Old'].notna()) & (df['Abundance_Young'].notna())].copy()
print(f"Paired old/young measurements: {len(df_paired):,}")

# ============================================================================
# 2. CURRENT METHOD (METHOD 0): WITHIN-STUDY Z-SCORES
# ============================================================================
print("\n2. METHOD 0: CURRENT WITHIN-STUDY Z-SCORES")
print("-" * 80)

# Extract current z-scores
df_paired['method0_zscore_delta'] = df_paired['Zscore_Delta']
df_paired['method0_available'] = df_paired['Zscore_Delta'].notna()

print(f"Available z-scores: {df_paired['method0_available'].sum():,}")

# Calculate aging effect for each protein across all studies
method0_results = df_paired.groupby('Canonical_Gene_Symbol').agg({
    'method0_zscore_delta': ['mean', 'std', 'count'],
    'Zscore_Old': 'mean',
    'Zscore_Young': 'mean'
}).reset_index()

method0_results.columns = ['Gene', 'Mean_ZDelta', 'Std_ZDelta', 'N_Obs', 'Mean_ZOld', 'Mean_ZYoung']
method0_results['SE'] = method0_results['Std_ZDelta'] / np.sqrt(method0_results['N_Obs'])
method0_results['T_Stat'] = method0_results['Mean_ZDelta'] / method0_results['SE']
method0_results['P_Value'] = 2 * (1 - stats.t.cdf(np.abs(method0_results['T_Stat']), method0_results['N_Obs'] - 1))
method0_results['Abs_Effect'] = np.abs(method0_results['Mean_ZDelta'])
method0_results['Method'] = 'Current_ZScore'

print(f"\nTop 10 proteins by absolute effect size:")
top10_method0 = method0_results.nlargest(10, 'Abs_Effect')[['Gene', 'Mean_ZDelta', 'P_Value', 'N_Obs']]
print(top10_method0.to_string(index=False))

# ============================================================================
# 3. METHOD 1: QUANTILE NORMALIZATION ACROSS STUDIES
# ============================================================================
print("\n\n3. METHOD 1: QUANTILE NORMALIZATION ACROSS STUDIES")
print("-" * 80)

def quantile_normalize_across_studies(df):
    """Apply quantile normalization to make distributions identical across studies"""

    # Create wide format: rows=proteins, cols=study_tissue combinations
    results = []

    for age_group in ['Old', 'Young']:
        col = f'Abundance_{age_group}'

        # Pivot to wide format
        pivot = df.pivot_table(
            values=col,
            index='Canonical_Gene_Symbol',
            columns=['Study_ID', 'Tissue'],
            aggfunc='mean'
        )

        # Apply quantile normalization
        # Strategy: replace values with mean of values at that quantile across all columns
        ranked = pivot.rank(method='average')
        sorted_df = pivot.apply(lambda x: np.sort(x.dropna()), axis=0, result_type='expand')
        mean_quantiles = sorted_df.mean(axis=1)

        # Map back to original positions
        normalized = ranked.copy()
        for col_name in normalized.columns:
            col_data = ranked[col_name].dropna()
            if len(col_data) > 0:
                quantile_values = mean_quantiles.iloc[:len(col_data)]
                rank_to_value = dict(zip(range(1, len(col_data) + 1), quantile_values))
                normalized[col_name] = col_data.map(rank_to_value)

        # Convert back to long format
        normalized_long = normalized.stack(level=[0, 1]).reset_index()
        normalized_long.columns = ['Canonical_Gene_Symbol', 'Study_ID', 'Tissue', f'Norm_{age_group}']

        results.append(normalized_long)

    # Merge old and young
    merged = results[0].merge(
        results[1],
        on=['Canonical_Gene_Symbol', 'Study_ID', 'Tissue'],
        how='outer'
    )

    # Calculate z-scores within each study-tissue
    merged['Norm_Delta'] = merged['Norm_Old'] - merged['Norm_Young']

    return merged

try:
    df_method1 = quantile_normalize_across_studies(df_paired)

    # Calculate overall statistics
    method1_results = df_method1.groupby('Canonical_Gene_Symbol').agg({
        'Norm_Delta': ['mean', 'std', 'count']
    }).reset_index()

    method1_results.columns = ['Gene', 'Mean_Delta', 'Std_Delta', 'N_Obs']
    method1_results['SE'] = method1_results['Std_Delta'] / np.sqrt(method1_results['N_Obs'])
    method1_results['T_Stat'] = method1_results['Mean_Delta'] / method1_results['SE']
    method1_results['P_Value'] = 2 * (1 - stats.t.cdf(np.abs(method1_results['T_Stat']), method1_results['N_Obs'] - 1))
    method1_results['Abs_Effect'] = np.abs(method1_results['Mean_Delta'])
    method1_results['Method'] = 'Quantile_Norm'

    print(f"\nTop 10 proteins by absolute effect size:")
    top10_method1 = method1_results.nlargest(10, 'Abs_Effect')[['Gene', 'Mean_Delta', 'P_Value', 'N_Obs']]
    print(top10_method1.to_string(index=False))

except Exception as e:
    print(f"Error in quantile normalization: {e}")
    method1_results = pd.DataFrame()

# ============================================================================
# 4. METHOD 2: RANK-BASED (SPEARMAN CORRELATION WITH AGE)
# ============================================================================
print("\n\n4. METHOD 2: RANK-BASED SPEARMAN CORRELATION")
print("-" * 80)

def rank_based_analysis(df):
    """Use ranks instead of raw values, calculate Spearman correlation with age"""

    results = []

    for gene in df['Canonical_Gene_Symbol'].unique():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene].copy()

        # Create age labels (0=young, 1=old) and get abundances
        old_vals = gene_data['Abundance_Old'].dropna().values
        young_vals = gene_data['Abundance_Young'].dropna().values

        if len(old_vals) == 0 and len(young_vals) == 0:
            continue

        # Combine into single array with age labels
        abundances = np.concatenate([old_vals, young_vals])
        ages = np.concatenate([np.ones(len(old_vals)), np.zeros(len(young_vals))])

        if len(abundances) < 3:
            continue

        # Calculate Spearman correlation
        try:
            rho, p_value = spearmanr(ages, abundances)

            results.append({
                'Gene': gene,
                'Spearman_Rho': rho,
                'P_Value': p_value,
                'N_Obs': len(abundances),
                'Median_Old': np.median(old_vals) if len(old_vals) > 0 else np.nan,
                'Median_Young': np.median(young_vals) if len(young_vals) > 0 else np.nan
            })
        except:
            continue

    return pd.DataFrame(results)

method2_results = rank_based_analysis(df_paired)
method2_results['Abs_Effect'] = np.abs(method2_results['Spearman_Rho'])
method2_results['Method'] = 'Rank_Spearman'

print(f"Analyzed {len(method2_results)} proteins")
print(f"\nTop 10 proteins by absolute Spearman correlation:")
top10_method2 = method2_results.nlargest(10, 'Abs_Effect')[['Gene', 'Spearman_Rho', 'P_Value', 'N_Obs']]
print(top10_method2.to_string(index=False))

# ============================================================================
# 5. METHOD 3: MIXED-EFFECTS MODEL (STUDY AS RANDOM EFFECT)
# ============================================================================
print("\n\n5. METHOD 3: MIXED-EFFECTS MODEL")
print("-" * 80)

def mixed_effects_analysis(df):
    """Fit mixed-effects model with age as fixed effect, study as random effect"""

    if not HAS_STATSMODELS:
        print("Statsmodels not available, using simplified approach")
        return pd.DataFrame()

    results = []

    # Prepare data: need long format with age as variable
    df_long = []
    for idx, row in df.iterrows():
        if pd.notna(row['Abundance_Old']):
            df_long.append({
                'Gene': row['Canonical_Gene_Symbol'],
                'Study': row['Study_ID'],
                'Tissue': row['Tissue'],
                'Age': 1,  # Old
                'Abundance': row['Abundance_Old']
            })
        if pd.notna(row['Abundance_Young']):
            df_long.append({
                'Gene': row['Canonical_Gene_Symbol'],
                'Study': row['Study_ID'],
                'Tissue': row['Tissue'],
                'Age': 0,  # Young
                'Abundance': row['Abundance_Young']
            })

    df_long = pd.DataFrame(df_long)

    # Analyze each protein separately
    genes_to_analyze = df_long['Gene'].value_counts()
    genes_to_analyze = genes_to_analyze[genes_to_analyze >= 10].index[:100]  # Top 100 for speed

    print(f"Analyzing {len(genes_to_analyze)} proteins with >=10 observations (sample)...")

    for i, gene in enumerate(genes_to_analyze):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(genes_to_analyze)}", end='\r')

        gene_data = df_long[df_long['Gene'] == gene].copy()

        if len(gene_data) < 10 or gene_data['Study'].nunique() < 2:
            continue

        try:
            # Fit mixed model: Abundance ~ Age + (1|Study)
            model = MixedLM(
                endog=gene_data['Abundance'],
                exog=sm.add_constant(gene_data['Age']),
                groups=gene_data['Study']
            )
            result = model.fit(method='nm', maxiter=100)

            # Extract age coefficient
            age_coef = result.params['Age']
            age_pval = result.pvalues['Age']

            results.append({
                'Gene': gene,
                'Age_Coefficient': age_coef,
                'P_Value': age_pval,
                'N_Obs': len(gene_data),
                'N_Studies': gene_data['Study'].nunique()
            })
        except Exception as e:
            continue

    print()
    return pd.DataFrame(results)

try:
    method3_results = mixed_effects_analysis(df_paired)
    if len(method3_results) > 0:
        method3_results['Abs_Effect'] = np.abs(method3_results['Age_Coefficient'])
        method3_results['Method'] = 'Mixed_Effects'

        print(f"Successfully analyzed {len(method3_results)} proteins")
        print(f"\nTop 10 proteins by absolute age coefficient:")
        top10_method3 = method3_results.nlargest(10, 'Abs_Effect')[['Gene', 'Age_Coefficient', 'P_Value', 'N_Studies']]
        print(top10_method3.to_string(index=False))
    else:
        print("No results from mixed-effects analysis")
except Exception as e:
    print(f"Error in mixed-effects analysis: {e}")
    method3_results = pd.DataFrame()

# ============================================================================
# 6. METHOD 4: COMBAT BATCH CORRECTION + Z-SCORES
# ============================================================================
print("\n\n6. METHOD 4: COMBAT-STYLE BATCH CORRECTION")
print("-" * 80)

def combat_style_correction(df):
    """
    Simplified ComBat-style batch correction:
    1. Standardize within each batch (study)
    2. Pool variance estimates
    3. Adjust for batch mean and variance
    """

    results = []

    # For each protein
    for gene in df['Canonical_Gene_Symbol'].unique():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene].copy()

        # Get all abundance values with study labels
        old_data = gene_data[['Study_ID', 'Abundance_Old']].dropna()
        young_data = gene_data[['Study_ID', 'Abundance_Young']].dropna()

        if len(old_data) == 0 and len(young_data) == 0:
            continue

        # Combine
        all_data = pd.concat([
            old_data.assign(Age='Old'),
            young_data.assign(Age='Young')
        ])
        all_data.columns = ['Study', 'Abundance', 'Age']

        # Calculate global mean and std
        global_mean = all_data['Abundance'].mean()
        global_std = all_data['Abundance'].std()

        if global_std == 0:
            continue

        # Adjust each study's data
        corrected = []
        for study in all_data['Study'].unique():
            study_data = all_data[all_data['Study'] == study].copy()

            # Study-specific mean and std
            study_mean = study_data['Abundance'].mean()
            study_std = study_data['Abundance'].std()

            if study_std == 0:
                study_std = global_std

            # ComBat correction: (x - study_mean) / study_std * global_std + global_mean
            study_data['Corrected'] = (
                (study_data['Abundance'] - study_mean) / study_std * global_std + global_mean
            )
            corrected.append(study_data)

        all_corrected = pd.concat(corrected)

        # Calculate effect of age on corrected values
        old_corrected = all_corrected[all_corrected['Age'] == 'Old']['Corrected']
        young_corrected = all_corrected[all_corrected['Age'] == 'Young']['Corrected']

        if len(old_corrected) > 0 and len(young_corrected) > 0:
            # T-test on corrected values
            t_stat, p_val = stats.ttest_ind(old_corrected, young_corrected)

            results.append({
                'Gene': gene,
                'Mean_Diff': old_corrected.mean() - young_corrected.mean(),
                'Std_Diff': np.sqrt(old_corrected.var() + young_corrected.var()),
                'T_Stat': t_stat,
                'P_Value': p_val,
                'N_Old': len(old_corrected),
                'N_Young': len(young_corrected)
            })

    return pd.DataFrame(results)

try:
    method4_results = combat_style_correction(df_paired)
    if len(method4_results) > 0:
        method4_results['Abs_Effect'] = np.abs(method4_results['Mean_Diff'] / method4_results['Std_Diff'])
        method4_results['Method'] = 'ComBat_Corrected'

        print(f"Analyzed {len(method4_results)} proteins")
        print(f"\nTop 10 proteins by standardized effect size:")
        top10_method4 = method4_results.nlargest(10, 'Abs_Effect')[['Gene', 'Mean_Diff', 'P_Value', 'N_Old', 'N_Young']]
        print(top10_method4.to_string(index=False))
    else:
        print("No results from ComBat correction")
except Exception as e:
    print(f"Error in ComBat correction: {e}")
    method4_results = pd.DataFrame()

# ============================================================================
# 7. CROSS-METHOD COMPARISON
# ============================================================================
print("\n\n7. CROSS-METHOD COMPARISON")
print("=" * 80)

# Get top 10 from each method
top10_sets = {
    'Current_ZScore': set(top10_method0['Gene'].tolist()),
    'Quantile_Norm': set(top10_method1['Gene'].tolist()) if len(method1_results) > 0 else set(),
    'Rank_Spearman': set(top10_method2['Gene'].tolist()),
    'Mixed_Effects': set(top10_method3['Gene'].tolist()) if len(method3_results) > 0 else set(),
    'ComBat_Corrected': set(top10_method4['Gene'].tolist()) if len(method4_results) > 0 else set()
}

print("\n7.1 TOP 10 PROTEINS PER METHOD:")
print("-" * 80)
for method, genes in top10_sets.items():
    if len(genes) > 0:
        print(f"\n{method}:")
        print(", ".join(sorted(genes)))

# Calculate overlap matrix
print("\n\n7.2 OVERLAP MATRIX (Number of shared proteins in top 10):")
print("-" * 80)
methods = [m for m in top10_sets.keys() if len(top10_sets[m]) > 0]
overlap_matrix = pd.DataFrame(index=methods, columns=methods, dtype=int)

for m1 in methods:
    for m2 in methods:
        overlap = len(top10_sets[m1] & top10_sets[m2])
        overlap_matrix.loc[m1, m2] = overlap

print(overlap_matrix.to_string())

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n\n8. SAVING RESULTS...")
print("-" * 80)

# Combine all results
all_results = []

if len(method0_results) > 0:
    method0_out = method0_results[['Gene', 'Mean_ZDelta', 'P_Value', 'N_Obs', 'Method']].copy()
    method0_out.columns = ['Gene', 'Effect_Size', 'P_Value', 'N_Obs', 'Method']
    all_results.append(method0_out)

if len(method1_results) > 0:
    method1_out = method1_results[['Gene', 'Mean_Delta', 'P_Value', 'N_Obs', 'Method']].copy()
    method1_out.columns = ['Gene', 'Effect_Size', 'P_Value', 'N_Obs', 'Method']
    all_results.append(method1_out)

if len(method2_results) > 0:
    method2_out = method2_results[['Gene', 'Spearman_Rho', 'P_Value', 'N_Obs', 'Method']].copy()
    method2_out.columns = ['Gene', 'Effect_Size', 'P_Value', 'N_Obs', 'Method']
    all_results.append(method2_out)

if len(method3_results) > 0:
    method3_out = method3_results[['Gene', 'Age_Coefficient', 'P_Value', 'N_Obs', 'Method']].copy()
    method3_out.columns = ['Gene', 'Effect_Size', 'P_Value', 'N_Obs', 'Method']
    all_results.append(method3_out)

if len(method4_results) > 0:
    method4_out = method4_results.copy()
    method4_out['Effect_Size'] = method4_out['Mean_Diff'] / method4_out['Std_Diff']
    method4_out['N_Obs'] = method4_out['N_Old'] + method4_out['N_Young']
    method4_out = method4_out[['Gene', 'Effect_Size', 'P_Value', 'N_Obs', 'Method']]
    all_results.append(method4_out)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv('normalization_methods_comparison.csv', index=False)
print(f"Saved comparison to: normalization_methods_comparison.csv")

# Save overlap matrix
overlap_matrix.to_csv('method_overlap_matrix.csv')
print(f"Saved overlap matrix to: method_overlap_matrix.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
