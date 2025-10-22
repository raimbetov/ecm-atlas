#!/usr/bin/env python3
"""
H13 Independent Dataset Validation - Meta-Analysis Script
Agent: claude_code

Purpose: Combine our data + external data, test heterogeneity (I² statistic)
Success Criterion: I² < 50% for top aging proteins (low heterogeneity)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple

#============================================================================
# CONFIGURATION
#============================================================================

WORKSPACE = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code")

# Our merged dataset
OUR_DATA = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# Top proteins from previous hypotheses
TOP_PROTEINS = {
    "H06_biomarkers": ['F13B', 'SERPINF1', 'S100A9', 'FSTL1', 'GAS6', 'CTSA', 'COL1A1', 'BGN'],
    "H08_s100_family": ['S100A4', 'S100A6', 'S100A8', 'S100A9', 'S100A10', 'S100A11'],
    "H03_fast_aging": ['COL1A1', 'COL3A1', 'FN1', 'LOX', 'TGM2', 'TIMP1']
}

#============================================================================
# I² HETEROGENEITY CALCULATION
#============================================================================

def calculate_i_squared(effect_sizes: List[float], variances: List[float]) -> Tuple[float, float, float]:
    """
    Calculate I² heterogeneity statistic for meta-analysis

    I² = (Q - df) / Q * 100%
    where Q is Cochran's Q statistic

    Interpretation:
    - I² < 25%: Low heterogeneity (consistent across studies)
    - I² 25-50%: Moderate heterogeneity
    - I² > 50%: High heterogeneity (study-specific effects)

    Returns:
        (I2, Q_statistic, p_value)
    """
    n_studies = len(effect_sizes)
    if n_studies < 2:
        return np.nan, np.nan, np.nan

    # Weights (inverse variance)
    weights = [1/v if v > 0 else 0 for v in variances]
    total_weight = sum(weights)

    if total_weight == 0:
        return np.nan, np.nan, np.nan

    # Weighted mean effect
    weighted_mean = sum(w * e for w, e in zip(weights, effect_sizes)) / total_weight

    # Cochran's Q statistic
    Q = sum(w * (e - weighted_mean)**2 for w, e in zip(weights, effect_sizes))

    # Degrees of freedom
    df = n_studies - 1

    # I² statistic
    if Q > df:
        I2 = ((Q - df) / Q) * 100
    else:
        I2 = 0.0

    # P-value for Q statistic (chi-square test)
    p_value = 1 - stats.chi2.cdf(Q, df) if df > 0 else np.nan

    return I2, Q, p_value

def fixed_effect_meta_analysis(
    effect_sizes: List[float],
    standard_errors: List[float]
) -> Dict:
    """
    Fixed-effect meta-analysis combining multiple studies

    Returns:
        dict with combined effect size, SE, CI, I²
    """
    n = len(effect_sizes)
    if n < 2:
        return {
            "combined_effect": effect_sizes[0] if n == 1 else np.nan,
            "combined_se": standard_errors[0] if n == 1 else np.nan,
            "I2": np.nan,
            "Q": np.nan,
            "p_het": np.nan
        }

    # Variances
    variances = [se**2 for se in standard_errors]

    # Weights
    weights = [1/v if v > 0 else 0 for v in variances]
    total_weight = sum(weights)

    # Combined effect size
    combined_effect = sum(w * e for w, e in zip(weights, effect_sizes)) / total_weight

    # Combined SE
    combined_se = np.sqrt(1 / total_weight)

    # 95% CI
    ci_lower = combined_effect - 1.96 * combined_se
    ci_upper = combined_effect + 1.96 * combined_se

    # Heterogeneity
    I2, Q, p_het = calculate_i_squared(effect_sizes, variances)

    return {
        "n_studies": n,
        "combined_effect": combined_effect,
        "combined_se": combined_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "I2": I2,
        "Q": Q,
        "p_heterogeneity": p_het
    }

#============================================================================
# PROTEIN-LEVEL META-ANALYSIS
#============================================================================

def meta_analyze_protein(
    protein: str,
    our_data: pd.DataFrame,
    external_data: pd.DataFrame,
    tissue: str = None
) -> Dict:
    """
    Perform meta-analysis for a single protein across our + external datasets

    Computes:
    - Delta Z-score (old - young) for each dataset
    - Fixed-effect meta-analysis
    - I² heterogeneity
    """
    results = {
        "protein": protein,
        "tissue": tissue,
        "n_datasets": 0,
        "combined_delta_z": np.nan,
        "I2": np.nan,
        "classification": "UNKNOWN"
    }

    # Extract data for this protein
    our_protein = our_data[our_data['Gene_Symbol'] == protein].copy()
    ext_protein = external_data[external_data['Gene_Symbol'] == protein].copy()

    if len(our_protein) == 0 and len(ext_protein) == 0:
        return results

    # Calculate effect sizes (Delta Z-score) and SEs for each dataset
    effect_sizes = []
    standard_errors = []

    # Our dataset
    if len(our_protein) > 0:
        if tissue:
            our_protein = our_protein[our_protein['Tissue'] == tissue]

        if 'Age_Group' in our_protein.columns:
            # Compute Delta Z between old and young
            young = our_protein[our_protein['Age_Group'] == 'Young']['Z_score']
            old = our_protein[our_protein['Age_Group'] == 'Old']['Z_score']

            if len(young) > 0 and len(old) > 0:
                delta_our = old.mean() - young.mean()
                se_our = np.sqrt(old.std()**2/len(old) + young.std()**2/len(young))
                effect_sizes.append(delta_our)
                standard_errors.append(se_our)

    # External dataset
    if len(ext_protein) > 0:
        if tissue:
            ext_protein = ext_protein[ext_protein['Tissue'] == tissue]

        if 'Age_Group' in ext_protein.columns:
            young = ext_protein[ext_protein['Age_Group'] == 'Young']['Z_score']
            old = ext_protein[ext_protein['Age_Group'] == 'Old']['Z_score']

            if len(young) > 0 and len(old) > 0:
                delta_ext = old.mean() - young.mean()
                se_ext = np.sqrt(old.std()**2/len(old) + young.std()**2/len(young))
                effect_sizes.append(delta_ext)
                standard_errors.append(se_ext)

    # Perform meta-analysis
    if len(effect_sizes) >= 2:
        meta_result = fixed_effect_meta_analysis(effect_sizes, standard_errors)

        I2 = meta_result['I2']

        # Classify protein stability
        if I2 < 25:
            classification = "STABLE"
        elif I2 < 50:
            classification = "MODERATE"
        else:
            classification = "VARIABLE"

        results.update({
            "n_datasets": len(effect_sizes),
            "combined_delta_z": meta_result['combined_effect'],
            "combined_se": meta_result['combined_se'],
            "ci_lower": meta_result['ci_lower'],
            "ci_upper": meta_result['ci_upper'],
            "I2": I2,
            "Q": meta_result['Q'],
            "p_heterogeneity": meta_result['p_heterogeneity'],
            "classification": classification,
            "effect_sizes": effect_sizes,
            "standard_errors": standard_errors
        })

    return results

#============================================================================
# COMPREHENSIVE META-ANALYSIS
#============================================================================

def run_comprehensive_meta_analysis(
    our_data_file: Path,
    external_data_file: Path,
    output_dir: Path
) -> pd.DataFrame:
    """
    Run meta-analysis for all top proteins from H03, H06, H08
    """
    print("\n" + "="*80)
    print("H13 META-ANALYSIS: COMBINING OUR + EXTERNAL DATASETS")
    print("="*80)

    # Load data
    print("\nLoading datasets...")
    our_df = pd.read_csv(our_data_file)
    external_df = pd.read_csv(external_data_file)

    print(f"  Our data: {len(our_df)} rows, {our_df['Gene_Symbol'].nunique()} proteins")
    print(f"  External data: {len(external_df)} rows, {external_df['Gene_Symbol'].nunique()} proteins")

    # Compile list of proteins to analyze
    all_top_proteins = []
    for category, proteins in TOP_PROTEINS.items():
        all_top_proteins.extend(proteins)
    all_top_proteins = list(set(all_top_proteins))  # Remove duplicates

    print(f"\n  Analyzing {len(all_top_proteins)} top proteins")

    # Run meta-analysis for each protein
    print("\nRunning meta-analyses...")
    results = []

    for protein in all_top_proteins:
        print(f"  Processing {protein}...")
        result = meta_analyze_protein(protein, our_df, external_df)
        results.append(result)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by I² (ascending - most stable first)
    df_results = df_results.sort_values('I2')

    # Save results
    output_file = output_dir / "meta_analysis_results_claude_code.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved meta-analysis results to: {output_file}")

    # Summary statistics
    print("\n" + "="*80)
    print("META-ANALYSIS SUMMARY")
    print("="*80)

    stable = df_results[df_results['classification'] == 'STABLE']
    moderate = df_results[df_results['classification'] == 'MODERATE']
    variable = df_results[df_results['classification'] == 'VARIABLE']

    print(f"\nProtein Stability Classification:")
    print(f"  STABLE (I² < 25%):    {len(stable)} proteins")
    print(f"  MODERATE (I² 25-50%): {len(moderate)} proteins")
    print(f"  VARIABLE (I² > 50%):  {len(variable)} proteins")

    print(f"\nTop 5 Most Stable Proteins (lowest I²):")
    for _, row in df_results.head(5).iterrows():
        print(f"  {row['protein']}: I²={row['I2']:.1f}%, ΔZ={row['combined_delta_z']:.2f}")

    print(f"\nSuccess Criterion: ≥15/20 proteins with I² < 50%")
    success_count = len(df_results[df_results['I2'] < 50])
    print(f"Result: {success_count}/{len(all_top_proteins)} proteins")
    if success_count >= 15:
        print("  ✅ SUCCESS: Findings are robust across cohorts")
    else:
        print("  ⚠️  WARNING: High heterogeneity detected")

    return df_results

#============================================================================
# FOREST PLOT VISUALIZATION
#============================================================================

def create_forest_plot(df_results: pd.DataFrame, output_path: Path):
    """
    Create forest plot showing effect sizes and heterogeneity
    """
    # Filter to proteins with valid results
    df_plot = df_results[~df_results['combined_delta_z'].isna()].copy()
    df_plot = df_plot.head(20)  # Top 20

    fig, ax = plt.subplots(figsize=(10, 12))

    y_positions = range(len(df_plot))

    for i, (_, row) in enumerate(df_plot.iterrows()):
        # Effect size point
        ax.plot(row['combined_delta_z'], i, 'o', markersize=8, color='black')

        # Confidence interval
        ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 'k-', linewidth=2)

        # Color by stability
        if row['I2'] < 25:
            color = 'green'
        elif row['I2'] < 50:
            color = 'orange'
        else:
            color = 'red'

        ax.plot(row['combined_delta_z'], i, 'o', markersize=8, color=color)

    # Formatting
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['protein']} (I²={row['I2']:.0f}%)"
                         for _, row in df_plot.iterrows()])
    ax.set_xlabel('Combined Effect Size (ΔZ-score)')
    ax.set_title('Meta-Analysis: Aging Effect Sizes Across Studies')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved forest plot to: {output_path}")

#============================================================================
# MAIN EXECUTION
#============================================================================

def main():
    """Main execution for meta-analysis"""

    print("\n" + "="*80)
    print("H13: META-ANALYSIS OF AGING PROTEIN SIGNATURES")
    print("="*80)

    our_data = Path(OUR_DATA)
    external_data = WORKSPACE / "external_datasets" / "merged_external_zscore.csv"

    if not external_data.exists():
        print("\n⚠️  EXTERNAL DATA NOT YET AVAILABLE")
        print(f"   Expected: {external_data}")
        print("\n📋 PLACEHOLDER EXECUTION")
        print("   This script is ready to run once external data is processed")
        return

    # Run meta-analysis
    df_results = run_comprehensive_meta_analysis(
        our_data,
        external_data,
        WORKSPACE
    )

    # Create forest plot
    plot_path = WORKSPACE / "visualizations_claude_code" / "meta_forest_plot_claude_code.png"
    plot_path.parent.mkdir(exist_ok=True)
    create_forest_plot(df_results, plot_path)

if __name__ == "__main__":
    main()
