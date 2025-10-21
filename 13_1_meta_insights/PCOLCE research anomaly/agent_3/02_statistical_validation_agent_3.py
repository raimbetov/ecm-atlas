#!/usr/bin/env python3
"""
PCOLCE Statistical Validation Script
Agent 3 - Research Anomaly Investigation

Comprehensive statistical analysis of PCOLCE measurements across:
- Study-level effects
- Tissue specificity
- Age stratification
- Quality metrics
- V1 vs V2 batch comparison
- Meta-analysis with effect sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas")
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/PCOLCE research anomaly/agent_3")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Data paths
V1_PATH = BASE_DIR / "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
V2_PATH = BASE_DIR / "obsolete/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv"

print("="*80)
print("PCOLCE STATISTICAL VALIDATION - AGENT 3")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/9] Loading datasets...")

try:
    df_v1 = pd.read_csv(V1_PATH)
    print(f"✓ V1 loaded: {len(df_v1):,} rows")
except Exception as e:
    print(f"✗ V1 load failed: {e}")
    df_v1 = None

try:
    df_v2 = pd.read_csv(V2_PATH)
    print(f"✓ V2 loaded: {len(df_v2):,} rows")
except Exception as e:
    print(f"✗ V2 load failed: {e}")
    df_v2 = None

# ============================================================================
# 2. EXTRACT PCOLCE DATA
# ============================================================================
print("\n[2/9] Extracting PCOLCE and PCOLCE2 data...")

def extract_pcolce(df, version_name):
    """Extract PCOLCE and PCOLCE2 entries"""
    if df is None:
        return None, None

    # Try different column name variations
    gene_col = None
    for col in ['Canonical_Gene_Symbol', 'Gene_Symbol', 'gene_symbol']:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        print(f"  ✗ {version_name}: No gene symbol column found")
        return None, None

    pcolce = df[df[gene_col].str.upper().isin(['PCOLCE', 'Pcolce'])].copy()
    pcolce2 = df[df[gene_col].str.upper().isin(['PCOLCE2', 'Pcolce2'])].copy()

    print(f"  {version_name}: PCOLCE={len(pcolce):,}, PCOLCE2={len(pcolce2):,}")

    return pcolce, pcolce2

pcolce_v1, pcolce2_v1 = extract_pcolce(df_v1, "V1")
pcolce_v2, pcolce2_v2 = extract_pcolce(df_v2, "V2")

# ============================================================================
# 3. STUDY-LEVEL ANALYSIS
# ============================================================================
print("\n[3/9] Analyzing PCOLCE per study...")

def study_level_analysis(df, version_name):
    """Calculate effect size per study"""
    if df is None or len(df) == 0:
        return None

    results = []

    for study in df['Study_ID'].unique():
        study_data = df[df['Study_ID'] == study]

        # Calculate metrics
        n_obs = len(study_data)
        mean_zscore_delta = study_data['Zscore_Delta'].mean()
        std_zscore_delta = study_data['Zscore_Delta'].std()
        sem = std_zscore_delta / np.sqrt(n_obs) if n_obs > 0 else np.nan

        # 95% CI
        ci_lower = mean_zscore_delta - 1.96 * sem
        ci_upper = mean_zscore_delta + 1.96 * sem

        # Direction (negative = decrease with age)
        direction = "DECREASE" if mean_zscore_delta < 0 else "INCREASE"

        # Tissue info
        tissues = study_data['Tissue'].unique()
        tissue_str = ", ".join(tissues)

        # Species
        species = study_data['Species'].unique()[0] if 'Species' in study_data.columns else "Unknown"

        results.append({
            'Study_ID': study,
            'Version': version_name,
            'N_observations': n_obs,
            'Mean_Zscore_Delta': mean_zscore_delta,
            'SD_Zscore_Delta': std_zscore_delta,
            'SEM': sem,
            'CI_95_lower': ci_lower,
            'CI_95_upper': ci_upper,
            'Direction': direction,
            'Tissues': tissue_str,
            'Species': species
        })

    return pd.DataFrame(results)

study_results_v1 = study_level_analysis(pcolce_v1, "V1")
study_results_v2 = study_level_analysis(pcolce_v2, "V2")

if study_results_v1 is not None:
    print(f"\n  V1 Study Breakdown:")
    print(study_results_v1[['Study_ID', 'Mean_Zscore_Delta', 'Direction', 'N_observations']].to_string(index=False))

if study_results_v2 is not None:
    print(f"\n  V2 Study Breakdown:")
    print(study_results_v2[['Study_ID', 'Mean_Zscore_Delta', 'Direction', 'N_observations']].to_string(index=False))

# Save
if study_results_v1 is not None and study_results_v2 is not None:
    combined_study = pd.concat([study_results_v1, study_results_v2])
    combined_study.to_csv(OUTPUT_DIR / "pcolce_study_breakdown.csv", index=False)
    print(f"\n✓ Saved: pcolce_study_breakdown.csv")

# ============================================================================
# 4. TISSUE-SPECIFIC ANALYSIS
# ============================================================================
print("\n[4/9] Tissue-specific analysis...")

def tissue_analysis(df, version_name):
    """Analyze PCOLCE by tissue/compartment"""
    if df is None or len(df) == 0:
        return None

    results = []

    # Group by Tissue_Compartment if available, else Tissue
    groupby_col = 'Tissue_Compartment' if 'Tissue_Compartment' in df.columns else 'Tissue'

    for tissue in df[groupby_col].unique():
        tissue_data = df[df[groupby_col] == tissue]

        n_obs = len(tissue_data)
        mean_delta = tissue_data['Zscore_Delta'].mean()
        std_delta = tissue_data['Zscore_Delta'].std()

        # T-test vs zero (is the change significant?)
        if n_obs > 2:
            t_stat, p_val = stats.ttest_1samp(tissue_data['Zscore_Delta'], 0)
        else:
            t_stat, p_val = np.nan, np.nan

        # Number of studies contributing
        n_studies = tissue_data['Study_ID'].nunique()

        results.append({
            'Tissue': tissue,
            'Version': version_name,
            'N_observations': n_obs,
            'N_studies': n_studies,
            'Mean_Zscore_Delta': mean_delta,
            'SD_Zscore_Delta': std_delta,
            'T_statistic': t_stat,
            'P_value': p_val,
            'Significant': "YES" if p_val < 0.05 else "NO"
        })

    return pd.DataFrame(results).sort_values('Mean_Zscore_Delta')

tissue_results_v1 = tissue_analysis(pcolce_v1, "V1")
tissue_results_v2 = tissue_analysis(pcolce_v2, "V2")

if tissue_results_v2 is not None:
    print(f"\n  V2 Tissue Breakdown (top 10 by magnitude):")
    top10 = tissue_results_v2.nlargest(10, 'Mean_Zscore_Delta', keep='all')
    print(top10[['Tissue', 'Mean_Zscore_Delta', 'N_studies', 'P_value', 'Significant']].to_string(index=False))

# Save
if tissue_results_v1 is not None and tissue_results_v2 is not None:
    combined_tissue = pd.concat([tissue_results_v1, tissue_results_v2])
    combined_tissue.to_csv(OUTPUT_DIR / "pcolce_tissue_analysis.csv", index=False)
    print(f"\n✓ Saved: pcolce_tissue_analysis.csv")

# ============================================================================
# 5. AGE STRATIFICATION
# ============================================================================
print("\n[5/9] Age stratification analysis...")

def age_stratification(df, version_name):
    """
    Stratify by age if age columns available
    Note: Our data uses Old/Young binary, not continuous age
    """
    if df is None or len(df) == 0:
        return None

    # Check if we have actual age data or just Old/Young
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    print(f"  {version_name} age-related columns: {age_cols}")

    # For now, analyze by Old vs Young z-scores
    results = {
        'Version': version_name,
        'Mean_Zscore_Old': df['Zscore_Old'].mean() if 'Zscore_Old' in df.columns else np.nan,
        'Mean_Zscore_Young': df['Zscore_Young'].mean() if 'Zscore_Young' in df.columns else np.nan,
        'Mean_Zscore_Delta': df['Zscore_Delta'].mean(),
        'N_observations': len(df)
    }

    return pd.DataFrame([results])

age_results_v1 = age_stratification(pcolce_v1, "V1")
age_results_v2 = age_stratification(pcolce_v2, "V2")

if age_results_v2 is not None:
    print(f"\n  V2 Age Analysis:")
    print(age_results_v2.to_string(index=False))

# Save
if age_results_v1 is not None and age_results_v2 is not None:
    combined_age = pd.concat([age_results_v1, age_results_v2])
    combined_age.to_csv(OUTPUT_DIR / "pcolce_age_stratified.csv", index=False)
    print(f"\n✓ Saved: pcolce_age_stratified.csv")

# ============================================================================
# 6. QUALITY METRICS
# ============================================================================
print("\n[6/9] Quality control metrics...")

def quality_metrics(df, version_name):
    """Calculate data quality metrics"""
    if df is None or len(df) == 0:
        return None

    # Missingness
    total_rows = len(df)
    missing_abundance_young = df['Abundance_Young'].isna().sum()
    missing_abundance_old = df['Abundance_Old'].isna().sum()
    missing_zscore = df['Zscore_Delta'].isna().sum()

    # Sample sizes
    n_profiles_young = df['N_Profiles_Young'].sum() if 'N_Profiles_Young' in df.columns else np.nan
    n_profiles_old = df['N_Profiles_Old'].sum() if 'N_Profiles_Old' in df.columns else np.nan

    # Studies and tissues
    n_studies = df['Study_ID'].nunique()
    n_tissues = df['Tissue'].nunique() if 'Tissue' in df.columns else np.nan

    # Z-score distribution
    zscore_mean = df['Zscore_Delta'].mean()
    zscore_median = df['Zscore_Delta'].median()
    zscore_std = df['Zscore_Delta'].std()
    zscore_min = df['Zscore_Delta'].min()
    zscore_max = df['Zscore_Delta'].max()

    # Directional consistency
    n_decrease = (df['Zscore_Delta'] < 0).sum()
    n_increase = (df['Zscore_Delta'] > 0).sum()
    consistency = max(n_decrease, n_increase) / (n_decrease + n_increase) if (n_decrease + n_increase) > 0 else np.nan

    metrics = {
        'Version': version_name,
        'Total_observations': total_rows,
        'Missing_Abundance_Young': missing_abundance_young,
        'Missing_Abundance_Old': missing_abundance_old,
        'Missing_Zscore_Delta': missing_zscore,
        'Pct_Missing_Young': (missing_abundance_young / total_rows) * 100,
        'Pct_Missing_Old': (missing_abundance_old / total_rows) * 100,
        'N_Studies': n_studies,
        'N_Tissues': n_tissues,
        'Total_Profiles_Young': n_profiles_young,
        'Total_Profiles_Old': n_profiles_old,
        'Zscore_Delta_Mean': zscore_mean,
        'Zscore_Delta_Median': zscore_median,
        'Zscore_Delta_SD': zscore_std,
        'Zscore_Delta_Min': zscore_min,
        'Zscore_Delta_Max': zscore_max,
        'N_Decrease': n_decrease,
        'N_Increase': n_increase,
        'Directional_Consistency': consistency
    }

    return pd.DataFrame([metrics])

quality_v1 = quality_metrics(pcolce_v1, "V1")
quality_v2 = quality_metrics(pcolce_v2, "V2")

if quality_v2 is not None:
    print(f"\n  V2 Quality Metrics:")
    print(f"    Total observations: {quality_v2['Total_observations'].values[0]}")
    print(f"    N studies: {quality_v2['N_Studies'].values[0]}")
    print(f"    N tissues: {quality_v2['N_Tissues'].values[0]}")
    print(f"    Missing Young: {quality_v2['Pct_Missing_Young'].values[0]:.1f}%")
    print(f"    Missing Old: {quality_v2['Pct_Missing_Old'].values[0]:.1f}%")
    print(f"    Zscore Delta: {quality_v2['Zscore_Delta_Mean'].values[0]:.3f} ± {quality_v2['Zscore_Delta_SD'].values[0]:.3f}")
    print(f"    Directional consistency: {quality_v2['Directional_Consistency'].values[0]:.2f}")
    print(f"    Decrease/Increase: {quality_v2['N_Decrease'].values[0]}/{quality_v2['N_Increase'].values[0]}")

# Save
if quality_v1 is not None and quality_v2 is not None:
    combined_quality = pd.concat([quality_v1, quality_v2])
    combined_quality.to_csv(OUTPUT_DIR / "pcolce_quality_metrics.csv", index=False)
    print(f"\n✓ Saved: pcolce_quality_metrics.csv")

# ============================================================================
# 7. V1 vs V2 COMPARISON
# ============================================================================
print("\n[7/9] V1 vs V2 batch correction comparison...")

def compare_versions(v1_df, v2_df):
    """Compare V1 and V2 distributions"""
    if v1_df is None or v2_df is None:
        print("  ✗ Cannot compare - missing data")
        return None

    # Match by Study_ID and Tissue
    merge_cols = ['Study_ID', 'Tissue']

    # Get mean Zscore_Delta per study-tissue in each version
    v1_agg = v1_df.groupby(merge_cols)['Zscore_Delta'].mean().reset_index()
    v1_agg.columns = merge_cols + ['V1_Zscore_Delta']

    v2_agg = v2_df.groupby(merge_cols)['Zscore_Delta'].mean().reset_index()
    v2_agg.columns = merge_cols + ['V2_Zscore_Delta']

    # Merge
    comparison = pd.merge(v1_agg, v2_agg, on=merge_cols, how='outer')
    comparison['Delta_Change'] = comparison['V2_Zscore_Delta'] - comparison['V1_Zscore_Delta']
    comparison['Pct_Change'] = (comparison['Delta_Change'] / comparison['V1_Zscore_Delta'].abs()) * 100

    # Stats
    paired_data = comparison.dropna(subset=['V1_Zscore_Delta', 'V2_Zscore_Delta'])

    if len(paired_data) > 0:
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(paired_data['V1_Zscore_Delta'], paired_data['V2_Zscore_Delta'])

        # Correlation
        corr, corr_p = stats.pearsonr(paired_data['V1_Zscore_Delta'], paired_data['V2_Zscore_Delta'])

        print(f"\n  Paired comparison (n={len(paired_data)}):")
        print(f"    V1 mean: {paired_data['V1_Zscore_Delta'].mean():.3f}")
        print(f"    V2 mean: {paired_data['V2_Zscore_Delta'].mean():.3f}")
        print(f"    Mean change: {paired_data['Delta_Change'].mean():.3f}")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    Correlation: r={corr:.3f}, p={corr_p:.4f}")

        # Direction flip check
        flips = ((paired_data['V1_Zscore_Delta'] > 0) & (paired_data['V2_Zscore_Delta'] < 0)) | \
                ((paired_data['V1_Zscore_Delta'] < 0) & (paired_data['V2_Zscore_Delta'] > 0))
        n_flips = flips.sum()
        print(f"    Direction flips: {n_flips}/{len(paired_data)} ({n_flips/len(paired_data)*100:.1f}%)")

    return comparison

comparison_df = compare_versions(pcolce_v1, pcolce_v2)

if comparison_df is not None:
    comparison_df.to_csv(OUTPUT_DIR / "pcolce_v1_v2_comparison.csv", index=False)
    print(f"\n✓ Saved: pcolce_v1_v2_comparison.csv")

# ============================================================================
# 8. META-ANALYSIS (Random Effects Model)
# ============================================================================
print("\n[8/9] Meta-analysis of study effect sizes...")

def meta_analysis(study_results):
    """
    Simple random-effects meta-analysis
    Using DerSimonian-Laird method
    """
    if study_results is None or len(study_results) == 0:
        return None

    # Filter out studies with NaN
    valid_studies = study_results.dropna(subset=['Mean_Zscore_Delta', 'SEM'])

    if len(valid_studies) < 2:
        print("  ✗ Insufficient studies for meta-analysis")
        return None

    # Calculate weights (inverse variance)
    valid_studies['Variance'] = valid_studies['SEM'] ** 2
    valid_studies['Weight'] = 1 / valid_studies['Variance']

    # Fixed-effect pooled estimate
    fe_pooled = np.sum(valid_studies['Mean_Zscore_Delta'] * valid_studies['Weight']) / np.sum(valid_studies['Weight'])
    fe_var = 1 / np.sum(valid_studies['Weight'])
    fe_se = np.sqrt(fe_var)

    # Heterogeneity (Q statistic)
    Q = np.sum(valid_studies['Weight'] * (valid_studies['Mean_Zscore_Delta'] - fe_pooled) ** 2)
    df = len(valid_studies) - 1
    p_heterogeneity = 1 - stats.chi2.cdf(Q, df)

    # I² statistic
    I2 = max(0, ((Q - df) / Q) * 100) if Q > 0 else 0

    # Between-study variance (τ²)
    C = np.sum(valid_studies['Weight']) - (np.sum(valid_studies['Weight']**2) / np.sum(valid_studies['Weight']))
    tau2 = max(0, (Q - df) / C) if C > 0 else 0

    # Random-effects weights
    valid_studies['RE_Weight'] = 1 / (valid_studies['Variance'] + tau2)

    # Random-effects pooled estimate
    re_pooled = np.sum(valid_studies['Mean_Zscore_Delta'] * valid_studies['RE_Weight']) / np.sum(valid_studies['RE_Weight'])
    re_var = 1 / np.sum(valid_studies['RE_Weight'])
    re_se = np.sqrt(re_var)

    # 95% CI
    re_ci_lower = re_pooled - 1.96 * re_se
    re_ci_upper = re_pooled + 1.96 * re_se

    # Z-test
    z_score = re_pooled / re_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    meta_results = {
        'N_studies': len(valid_studies),
        'Fixed_Effect_Pooled': fe_pooled,
        'Fixed_Effect_SE': fe_se,
        'Random_Effect_Pooled': re_pooled,
        'Random_Effect_SE': re_se,
        'RE_CI_95_lower': re_ci_lower,
        'RE_CI_95_upper': re_ci_upper,
        'Z_score': z_score,
        'P_value': p_value,
        'Q_statistic': Q,
        'Q_df': df,
        'Q_p_value': p_heterogeneity,
        'I2_percent': I2,
        'Tau2': tau2
    }

    return meta_results, valid_studies

if study_results_v2 is not None:
    meta_v2, meta_studies_v2 = meta_analysis(study_results_v2)

    if meta_v2:
        print(f"\n  V2 Meta-Analysis:")
        print(f"    N studies: {meta_v2['N_studies']}")
        print(f"    Random-effects pooled: {meta_v2['Random_Effect_Pooled']:.3f} (SE={meta_v2['Random_Effect_SE']:.3f})")
        print(f"    95% CI: [{meta_v2['RE_CI_95_lower']:.3f}, {meta_v2['RE_CI_95_upper']:.3f}]")
        print(f"    Z-score: {meta_v2['Z_score']:.3f}, p={meta_v2['P_value']:.4e}")
        print(f"    Heterogeneity: Q={meta_v2['Q_statistic']:.2f}, p={meta_v2['Q_p_value']:.4f}")
        print(f"    I²: {meta_v2['I2_percent']:.1f}%")
        print(f"    τ²: {meta_v2['Tau2']:.4f}")

        # Interpretation
        if meta_v2['I2_percent'] < 25:
            heterogeneity_interp = "LOW heterogeneity - studies are consistent"
        elif meta_v2['I2_percent'] < 50:
            heterogeneity_interp = "MODERATE heterogeneity"
        elif meta_v2['I2_percent'] < 75:
            heterogeneity_interp = "SUBSTANTIAL heterogeneity"
        else:
            heterogeneity_interp = "CONSIDERABLE heterogeneity - pooled estimate questionable"

        print(f"    Interpretation: {heterogeneity_interp}")

        # Save
        pd.DataFrame([meta_v2]).to_csv(OUTPUT_DIR / "pcolce_meta_analysis.csv", index=False)
        print(f"\n✓ Saved: pcolce_meta_analysis.csv")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n[9/9] Creating visualizations...")

# 9.1 Forest plot
if study_results_v2 is not None and len(study_results_v2) > 0:
    plt.figure(figsize=(10, 8))

    study_results_sorted = study_results_v2.sort_values('Mean_Zscore_Delta')

    y_pos = np.arange(len(study_results_sorted))

    plt.errorbar(
        study_results_sorted['Mean_Zscore_Delta'],
        y_pos,
        xerr=[
            study_results_sorted['Mean_Zscore_Delta'] - study_results_sorted['CI_95_lower'],
            study_results_sorted['CI_95_upper'] - study_results_sorted['Mean_Zscore_Delta']
        ],
        fmt='o',
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
        color='darkblue',
        alpha=0.7
    )

    plt.yticks(y_pos, study_results_sorted['Study_ID'])
    plt.xlabel('Mean PCOLCE Zscore Delta (Negative = Decrease with Age)', fontsize=12, fontweight='bold')
    plt.ylabel('Study ID', fontsize=12, fontweight='bold')
    plt.title('PCOLCE Effect Sizes by Study (V2 Batch-Corrected)\nForest Plot with 95% Confidence Intervals',
              fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No Effect')

    # Add pooled estimate if meta-analysis done
    if 'meta_v2' in locals() and meta_v2:
        plt.axvline(
            x=meta_v2['Random_Effect_Pooled'],
            color='green',
            linestyle='-',
            linewidth=3,
            alpha=0.8,
            label=f"Pooled RE: {meta_v2['Random_Effect_Pooled']:.2f}"
        )

    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pcolce_meta_analysis_forest_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pcolce_meta_analysis_forest_plot.png")

# 9.2 V1 vs V2 scatter
if comparison_df is not None:
    paired = comparison_df.dropna(subset=['V1_Zscore_Delta', 'V2_Zscore_Delta'])

    if len(paired) > 0:
        plt.figure(figsize=(8, 8))

        plt.scatter(paired['V1_Zscore_Delta'], paired['V2_Zscore_Delta'],
                   s=100, alpha=0.6, edgecolors='black', linewidth=1.5)

        # Unity line
        min_val = min(paired['V1_Zscore_Delta'].min(), paired['V2_Zscore_Delta'].min())
        max_val = max(paired['V1_Zscore_Delta'].max(), paired['V2_Zscore_Delta'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Unity (no change)')

        plt.xlabel('V1 Zscore Delta', fontsize=12, fontweight='bold')
        plt.ylabel('V2 Zscore Delta (Batch-Corrected)', fontsize=12, fontweight='bold')
        plt.title('PCOLCE: V1 vs V2 Batch Correction Comparison', fontsize=14, fontweight='bold')

        # Add correlation
        corr = paired[['V1_Zscore_Delta', 'V2_Zscore_Delta']].corr().iloc[0, 1]
        plt.text(0.05, 0.95, f'Correlation: r={corr:.3f}',
                transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pcolce_v1_v2_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: pcolce_v1_v2_comparison.png")

# 9.3 Tissue heatmap
if tissue_results_v2 is not None and len(tissue_results_v2) > 5:
    plt.figure(figsize=(6, 10))

    top_tissues = tissue_results_v2.nlargest(20, 'Mean_Zscore_Delta', keep='all')

    # Create heatmap data
    heatmap_data = top_tissues[['Tissue', 'Mean_Zscore_Delta']].set_index('Tissue')

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Mean Zscore Delta'}, linewidths=0.5)

    plt.title('PCOLCE Tissue-Specific Effect Sizes (V2)\nTop 20 Tissues', fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pcolce_tissue_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pcolce_tissue_heatmap.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL VALIDATION COMPLETE")
print("="*80)

print("\n📊 KEY FINDINGS:")

if quality_v2 is not None:
    print(f"\n1. DATA QUALITY (V2):")
    print(f"   - Observations: {quality_v2['Total_observations'].values[0]}")
    print(f"   - Studies: {quality_v2['N_Studies'].values[0]}")
    print(f"   - Tissues: {quality_v2['N_Tissues'].values[0]}")
    print(f"   - Directional consistency: {quality_v2['Directional_Consistency'].values[0]:.1%}")
    print(f"   - Mean effect: Δz = {quality_v2['Zscore_Delta_Mean'].values[0]:.3f}")

if 'meta_v2' in locals() and meta_v2:
    print(f"\n2. META-ANALYSIS (V2):")
    print(f"   - Pooled effect: {meta_v2['Random_Effect_Pooled']:.3f} [{meta_v2['RE_CI_95_lower']:.3f}, {meta_v2['RE_CI_95_upper']:.3f}]")
    print(f"   - Significance: p = {meta_v2['P_value']:.4e}")
    print(f"   - Heterogeneity: I² = {meta_v2['I2_percent']:.1f}%")

    if meta_v2['P_value'] < 0.05:
        print(f"   - INTERPRETATION: PCOLCE significantly DECREASES with aging (p < 0.05)")
    else:
        print(f"   - INTERPRETATION: No significant aging effect")

if comparison_df is not None:
    paired = comparison_df.dropna()
    if len(paired) > 0:
        print(f"\n3. BATCH CORRECTION IMPACT:")
        print(f"   - V1 mean: {paired['V1_Zscore_Delta'].mean():.3f}")
        print(f"   - V2 mean: {paired['V2_Zscore_Delta'].mean():.3f}")
        print(f"   - Change: {paired['Delta_Change'].mean():.3f} ({paired['Delta_Change'].mean() / abs(paired['V1_Zscore_Delta'].mean()) * 100:.1f}%)")

print("\n✓ All outputs saved to:")
print(f"   {OUTPUT_DIR}")

print("\n" + "="*80)
