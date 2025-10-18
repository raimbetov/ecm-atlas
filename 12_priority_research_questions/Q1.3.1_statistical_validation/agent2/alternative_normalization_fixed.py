#!/usr/bin/env python3
"""
Alternative Normalization Methods Analysis - FIXED VERSION
Compares 5 normalization approaches for ECM aging data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# For mixed-effects models
try:
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

print("=" * 80)
print("ALTERNATIVE NORMALIZATION METHODS ANALYSIS (FIXED)")
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
# 2. METHOD 0: CURRENT WITHIN-STUDY Z-SCORES
# ============================================================================
print("\n2. METHOD 0: CURRENT WITHIN-STUDY Z-SCORES")
print("-" * 80)

df_paired['method0_zscore_delta'] = df_paired['Zscore_Delta']
df_paired['method0_available'] = df_paired['Zscore_Delta'].notna()

print(f"Available z-scores: {df_paired['method0_available'].sum():,}")

# Calculate aging effect for each protein across all studies
method0_results = df_paired[df_paired['Zscore_Delta'].notna()].groupby('Canonical_Gene_Symbol').agg({
    'Zscore_Delta': ['mean', 'std', 'count'],
    'Zscore_Old': 'mean',
    'Zscore_Young': 'mean'
}).reset_index()

method0_results.columns = ['Gene', 'Mean_ZDelta', 'Std_ZDelta', 'N_Obs', 'Mean_ZOld', 'Mean_ZYoung']

# Filter to proteins with at least 3 observations
method0_results = method0_results[method0_results['N_Obs'] >= 3].copy()

method0_results['SE'] = method0_results['Std_ZDelta'] / np.sqrt(method0_results['N_Obs'])
method0_results['T_Stat'] = method0_results['Mean_ZDelta'] / method0_results['SE']
method0_results['P_Value'] = 2 * (1 - stats.t.cdf(np.abs(method0_results['T_Stat']), method0_results['N_Obs'] - 1))
method0_results['Abs_Effect'] = np.abs(method0_results['Mean_ZDelta'])
method0_results['Method'] = 'Current_ZScore'

# Sort by absolute effect and filter by p-value < 0.05
method0_sig = method0_results[method0_results['P_Value'] < 0.05].copy()
method0_sig = method0_sig.sort_values('Abs_Effect', ascending=False)

print(f"Proteins with N >= 3: {len(method0_results)}")
print(f"Significant proteins (p < 0.05): {len(method0_sig)}")

print(f"\nTop 10 proteins by absolute effect size (p < 0.05):")
top10_method0 = method0_sig.head(10)[['Gene', 'Mean_ZDelta', 'P_Value', 'N_Obs']]
print(top10_method0.to_string(index=False))

# ============================================================================
# 3. METHOD 1: SIMPLE PERCENTILE NORMALIZATION
# ============================================================================
print("\n\n3. METHOD 1: PERCENTILE NORMALIZATION")
print("-" * 80)

def percentile_normalize(df):
    """Convert abundances to percentiles within each study, then compare"""

    results = []

    for gene in df['Canonical_Gene_Symbol'].unique():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene].copy()

        # For each study, convert to percentiles
        percentile_deltas = []

        for study in gene_data['Study_ID'].unique():
            study_gene = gene_data[gene_data['Study_ID'] == study]

            if len(study_gene) == 0:
                continue

            # Get all abundances for this study (all proteins) to calculate percentiles
            study_all = df[df['Study_ID'] == study]

            # Calculate percentiles for old and young
            old_val = study_gene['Abundance_Old'].iloc[0]
            young_val = study_gene['Abundance_Young'].iloc[0]

            if pd.notna(old_val) and pd.notna(young_val):
                # Get percentile within study
                all_old = study_all['Abundance_Old'].dropna()
                all_young = study_all['Abundance_Young'].dropna()

                if len(all_old) > 0 and len(all_young) > 0:
                    pct_old = stats.percentileofscore(all_old, old_val, kind='rank')
                    pct_young = stats.percentileofscore(all_young, young_val, kind='rank')

                    percentile_deltas.append(pct_old - pct_young)

        if len(percentile_deltas) >= 3:
            results.append({
                'Gene': gene,
                'Mean_Pct_Delta': np.mean(percentile_deltas),
                'Std_Pct_Delta': np.std(percentile_deltas, ddof=1),
                'N_Obs': len(percentile_deltas)
            })

    return pd.DataFrame(results)

try:
    method1_results = percentile_normalize(df_paired)

    if len(method1_results) > 0:
        method1_results['SE'] = method1_results['Std_Pct_Delta'] / np.sqrt(method1_results['N_Obs'])
        method1_results['T_Stat'] = method1_results['Mean_Pct_Delta'] / method1_results['SE']
        method1_results['P_Value'] = 2 * (1 - stats.t.cdf(np.abs(method1_results['T_Stat']), method1_results['N_Obs'] - 1))
        method1_results['Abs_Effect'] = np.abs(method1_results['Mean_Pct_Delta'])
        method1_results['Method'] = 'Percentile_Norm'

        method1_sig = method1_results[method1_results['P_Value'] < 0.05].copy()
        method1_sig = method1_sig.sort_values('Abs_Effect', ascending=False)

        print(f"Analyzed {len(method1_results)} proteins")
        print(f"Significant proteins (p < 0.05): {len(method1_sig)}")

        print(f"\nTop 10 proteins by absolute effect size (p < 0.05):")
        top10_method1 = method1_sig.head(10)[['Gene', 'Mean_Pct_Delta', 'P_Value', 'N_Obs']]
        print(top10_method1.to_string(index=False))
    else:
        print("No results from percentile normalization")
        method1_sig = pd.DataFrame()

except Exception as e:
    print(f"Error in percentile normalization: {e}")
    method1_results = pd.DataFrame()
    method1_sig = pd.DataFrame()

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

        if len(abundances) < 6:  # Need at least 6 points for meaningful correlation
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

method2_sig = method2_results[method2_results['P_Value'] < 0.05].copy()
method2_sig = method2_sig.sort_values('Abs_Effect', ascending=False)

print(f"Analyzed {len(method2_results)} proteins")
print(f"Significant proteins (p < 0.05): {len(method2_sig)}")

print(f"\nTop 10 proteins by absolute Spearman correlation (p < 0.05):")
top10_method2 = method2_sig.head(10)[['Gene', 'Spearman_Rho', 'P_Value', 'N_Obs']]
print(top10_method2.to_string(index=False))

# ============================================================================
# 5. METHOD 3: MIXED-EFFECTS MODEL (STUDY AS RANDOM EFFECT)
# ============================================================================
print("\n\n5. METHOD 3: MIXED-EFFECTS MODEL")
print("-" * 80)

def mixed_effects_analysis(df):
    """Fit mixed-effects model with age as fixed effect, study as random effect"""

    if not HAS_STATSMODELS:
        print("Statsmodels not available")
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

    # Analyze each protein separately - select genes with sufficient data
    gene_counts = df_long.groupby('Gene').agg({
        'Abundance': 'count',
        'Study': 'nunique'
    }).reset_index()
    gene_counts.columns = ['Gene', 'N', 'N_Studies']

    # Need at least 20 observations across at least 3 studies
    genes_to_analyze = gene_counts[(gene_counts['N'] >= 20) & (gene_counts['N_Studies'] >= 3)]['Gene'].tolist()

    print(f"Analyzing {len(genes_to_analyze)} proteins with >=20 obs and >=3 studies...")

    for i, gene in enumerate(genes_to_analyze):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(genes_to_analyze)}", end='\r')

        gene_data = df_long[df_long['Gene'] == gene].copy()

        try:
            # Fit mixed model: Abundance ~ Age + (1|Study)
            model = MixedLM(
                endog=gene_data['Abundance'],
                exog=sm.add_constant(gene_data['Age']),
                groups=gene_data['Study']
            )
            result = model.fit(method='nm', maxiter=200, disp=False)

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

        method3_sig = method3_results[method3_results['P_Value'] < 0.05].copy()
        method3_sig = method3_sig.sort_values('Abs_Effect', ascending=False)

        print(f"Successfully analyzed {len(method3_results)} proteins")
        print(f"Significant proteins (p < 0.05): {len(method3_sig)}")

        print(f"\nTop 10 proteins by absolute age coefficient (p < 0.05):")
        top10_method3 = method3_sig.head(10)[['Gene', 'Age_Coefficient', 'P_Value', 'N_Studies']]
        print(top10_method3.to_string(index=False))
    else:
        print("No results from mixed-effects analysis")
        method3_sig = pd.DataFrame()
except Exception as e:
    print(f"Error in mixed-effects analysis: {e}")
    import traceback
    traceback.print_exc()
    method3_results = pd.DataFrame()
    method3_sig = pd.DataFrame()

# ============================================================================
# 6. METHOD 4: SIMPLE STANDARDIZATION ACROSS STUDIES
# ============================================================================
print("\n\n6. METHOD 4: GLOBAL STANDARDIZATION")
print("-" * 80)

def global_standardization(df):
    """
    Standardize abundances globally (across all studies), then compare
    This removes study-specific scaling effects
    """

    results = []

    # For each protein, get all abundance values
    for gene in df['Canonical_Gene_Symbol'].unique():
        gene_data = df[df['Canonical_Gene_Symbol'] == gene].copy()

        # Get all old and young values
        old_vals = gene_data['Abundance_Old'].dropna()
        young_vals = gene_data['Abundance_Young'].dropna()

        if len(old_vals) < 3 or len(young_vals) < 3:
            continue

        # Calculate global mean and std for this protein
        all_vals = pd.concat([old_vals, young_vals])
        global_mean = all_vals.mean()
        global_std = all_vals.std()

        if global_std == 0:
            continue

        # Standardize
        old_standardized = (old_vals - global_mean) / global_std
        young_standardized = (young_vals - global_mean) / global_std

        # T-test on standardized values
        try:
            t_stat, p_val = stats.ttest_ind(old_standardized, young_standardized)

            results.append({
                'Gene': gene,
                'Mean_Std_Old': old_standardized.mean(),
                'Mean_Std_Young': young_standardized.mean(),
                'Mean_Diff': old_standardized.mean() - young_standardized.mean(),
                'T_Stat': t_stat,
                'P_Value': p_val,
                'N_Old': len(old_standardized),
                'N_Young': len(young_standardized)
            })
        except:
            continue

    return pd.DataFrame(results)

try:
    method4_results = global_standardization(df_paired)
    if len(method4_results) > 0:
        method4_results['Abs_Effect'] = np.abs(method4_results['Mean_Diff'])
        method4_results['Method'] = 'Global_Standard'

        method4_sig = method4_results[method4_results['P_Value'] < 0.05].copy()
        method4_sig = method4_sig.sort_values('Abs_Effect', ascending=False)

        print(f"Analyzed {len(method4_results)} proteins")
        print(f"Significant proteins (p < 0.05): {len(method4_sig)}")

        print(f"\nTop 10 proteins by standardized effect size (p < 0.05):")
        top10_method4 = method4_sig.head(10)[['Gene', 'Mean_Diff', 'P_Value', 'N_Old', 'N_Young']]
        print(top10_method4.to_string(index=False))
    else:
        print("No results from global standardization")
        method4_sig = pd.DataFrame()
except Exception as e:
    print(f"Error in global standardization: {e}")
    method4_results = pd.DataFrame()
    method4_sig = pd.DataFrame()

# ============================================================================
# 7. CROSS-METHOD COMPARISON
# ============================================================================
print("\n\n7. CROSS-METHOD COMPARISON")
print("=" * 80)

# Get top 10 from each method (significant only)
top10_sets = {
    'Current_ZScore': set(top10_method0['Gene'].tolist()),
    'Percentile_Norm': set(top10_method1['Gene'].tolist()) if len(method1_sig) > 0 else set(),
    'Rank_Spearman': set(top10_method2['Gene'].tolist()),
    'Mixed_Effects': set(top10_method3['Gene'].tolist()) if len(method3_sig) > 0 else set(),
    'Global_Standard': set(top10_method4['Gene'].tolist()) if len(method4_sig) > 0 else set()
}

print("\n7.1 TOP 10 PROTEINS PER METHOD (p < 0.05):")
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

# Find consensus proteins (appear in multiple methods)
print("\n\n7.3 CONSENSUS PROTEINS (appear in >=2 methods):")
print("-" * 80)
all_genes = set()
for genes in top10_sets.values():
    all_genes.update(genes)

consensus = []
for gene in all_genes:
    count = sum(1 for genes in top10_sets.values() if gene in genes)
    if count >= 2:
        methods_list = [m for m, genes in top10_sets.items() if gene in genes]
        consensus.append({
            'Gene': gene,
            'N_Methods': count,
            'Methods': ', '.join(methods_list)
        })

if len(consensus) > 0:
    consensus_df = pd.DataFrame(consensus).sort_values('N_Methods', ascending=False)
    print(consensus_df.to_string(index=False))
else:
    print("No consensus proteins found")

# ============================================================================
# 8. METHOD STATISTICS COMPARISON
# ============================================================================
print("\n\n8. METHOD STATISTICS:")
print("=" * 80)

stats_comparison = []

for method_name, method_df in [
    ('Current_ZScore', method0_results),
    ('Percentile_Norm', method1_results),
    ('Rank_Spearman', method2_results),
    ('Mixed_Effects', method3_results),
    ('Global_Standard', method4_results)
]:
    if len(method_df) > 0:
        sig_df = method_df[method_df['P_Value'] < 0.05]

        stats_comparison.append({
            'Method': method_name,
            'N_Proteins_Analyzed': len(method_df),
            'N_Significant_p005': len(sig_df),
            'Pct_Significant': f"{100 * len(sig_df) / len(method_df):.1f}%",
            'Most_Conservative': 'Yes' if len(sig_df) == min([len(m[m['P_Value'] < 0.05]) for m in [method0_results, method1_results, method2_results, method3_results, method4_results] if len(m) > 0]) else 'No',
            'Strongest_Signal': 'Yes' if len(sig_df) == max([len(m[m['P_Value'] < 0.05]) for m in [method0_results, method1_results, method2_results, method3_results, method4_results] if len(m) > 0]) else 'No'
        })

stats_comp_df = pd.DataFrame(stats_comparison)
print(stats_comp_df.to_string(index=False))

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n\n9. SAVING RESULTS...")
print("-" * 80)

# Save individual method results
if len(method0_results) > 0:
    method0_results.to_csv('method0_current_zscore.csv', index=False)
    print("Saved method0_current_zscore.csv")

if len(method1_results) > 0:
    method1_results.to_csv('method1_percentile_norm.csv', index=False)
    print("Saved method1_percentile_norm.csv")

if len(method2_results) > 0:
    method2_results.to_csv('method2_rank_spearman.csv', index=False)
    print("Saved method2_rank_spearman.csv")

if len(method3_results) > 0:
    method3_results.to_csv('method3_mixed_effects.csv', index=False)
    print("Saved method3_mixed_effects.csv")

if len(method4_results) > 0:
    method4_results.to_csv('method4_global_standard.csv', index=False)
    print("Saved method4_global_standard.csv")

# Save overlap matrix
overlap_matrix.to_csv('method_overlap_matrix.csv')
print("Saved method_overlap_matrix.csv")

# Save consensus
if len(consensus) > 0:
    consensus_df.to_csv('consensus_proteins.csv', index=False)
    print("Saved consensus_proteins.csv")

# Save statistics comparison
stats_comp_df.to_csv('method_statistics_comparison.csv', index=False)
print("Saved method_statistics_comparison.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
