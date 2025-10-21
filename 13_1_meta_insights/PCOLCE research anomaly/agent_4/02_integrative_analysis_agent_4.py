#!/usr/bin/env python3
"""
Agent 4: PCOLCE Research Anomaly - Integrative Correlation Analysis

Purpose:
    Resolve paradox where PCOLCE decreases with aging (Δz=-1.41, 7 studies)
    despite literature showing PCOLCE promotes fibrosis and collagen accumulation.

Approach:
    1. Load V2 batch-corrected ECM dataset
    2. Extract PCOLCE trajectories across all studies
    3. Identify collagen protein family members
    4. Compute cross-study correlations (PCOLCE vs collagens)
    5. Screen for compensatory proteases (BMP-1, ADAMTS, etc.)
    6. Tissue-stratified analysis
    7. Generate publication-quality figures and summary tables

Output:
    - CSV: correlation matrices, compensatory protease scores
    - PNG: heatmaps, scatter plots, network diagrams
    - JSON: statistical summaries for downstream synthesis

Author: Agent 4 (Claude Code)
Date: 2025-10-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
V2_PATH = "/Users/Kravtsovd/projects/ecm-atlas/obsolete/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv"
V1_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# Output directory
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/PCOLCE research anomaly/agent_4/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target protein families
COLLAGEN_FIBRILLAR = ['COL1A1', 'COL1A2', 'COL2A1', 'COL3A1', 'COL5A1', 'COL5A2',
                      'COL11A1', 'COL11A2', 'COL24A1', 'COL27A1']

COLLAGEN_NETWORK = ['COL4A1', 'COL4A2', 'COL4A3', 'COL4A4', 'COL4A5', 'COL4A6',
                    'COL6A1', 'COL6A2', 'COL6A3', 'COL6A4', 'COL6A5', 'COL6A6']

COLLAGEN_FACIT = ['COL9A1', 'COL9A2', 'COL9A3', 'COL12A1', 'COL14A1', 'COL16A1',
                  'COL19A1', 'COL20A1', 'COL21A1', 'COL22A1']

PROCOLLAGEN_PROCESSORS = {
    'C-proteinases': ['BMP1', 'BMP-1', 'PCP'],  # BMP-1 and aliases
    'N-proteinases': ['ADAMTS2', 'ADAMTS3', 'ADAMTS14'],
    'Tolloid family': ['TLL1', 'TLL2', 'TOLLOID'],
    'General': ['FURIN', 'PCSK5', 'PCSK6']
}

COLLAGEN_MATURATION = {
    'Lysyl_hydroxylases': ['PLOD1', 'PLOD2', 'PLOD3'],
    'Prolyl_hydroxylases': ['P4HA1', 'P4HA2', 'P4HA3', 'P4HB'],
    'Lysyl_oxidases': ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4']
}

# Statistical thresholds
CORRELATION_THRESHOLD = 0.3  # Minimum |r| for significance
MIN_STUDIES = 2  # Minimum studies for meta-analysis

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_and_validate_data(path, expected_pcolce_studies=7):
    """
    Load ECM dataset and validate PCOLCE presence.

    Args:
        path: Path to CSV file
        expected_pcolce_studies: Expected number of studies with PCOLCE

    Returns:
        DataFrame with validated data
    """
    print(f"\n{'='*80}")
    print(f"Loading data from: {Path(path).name}")
    print(f"{'='*80}")

    df = pd.read_csv(path)

    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Unique proteins: {df['Gene_Symbol'].nunique():,}")
    print(f"  Unique studies: {df['Study_ID'].nunique()}")
    print(f"  Tissues: {df['Tissue'].nunique()}")

    # Validate PCOLCE
    pcolce_data = df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE'].copy()
    if len(pcolce_data) == 0:
        # Try alternative names
        pcolce_data = df[df['Gene_Symbol'].str.upper().str.contains('PCOLCE|PCPE', na=False)].copy()

    pcolce_studies = pcolce_data['Study_ID'].nunique()
    pcolce_mean_delta = pcolce_data['Zscore_Delta'].mean()

    print(f"\n  PCOLCE validation:")
    print(f"    Measurements: {len(pcolce_data)}")
    print(f"    Studies: {pcolce_studies} (expected: {expected_pcolce_studies})")
    print(f"    Mean Δz: {pcolce_mean_delta:.3f}")

    if pcolce_studies < expected_pcolce_studies:
        print(f"    ⚠️  WARNING: Found {pcolce_studies} studies, expected {expected_pcolce_studies}")
    else:
        print(f"    ✓ PCOLCE validation passed")

    return df

def extract_protein_family(df, gene_list, family_name):
    """
    Extract subset of data for protein family.

    Args:
        df: Full dataset
        gene_list: List of gene symbols to extract
        family_name: Descriptive name for logging

    Returns:
        Filtered DataFrame
    """
    # Try canonical gene symbol first
    mask = df['Canonical_Gene_Symbol'].str.upper().isin([g.upper() for g in gene_list])
    subset = df[mask].copy()

    # Fallback to Gene_Symbol if canonical is empty
    if len(subset) == 0 or subset['Canonical_Gene_Symbol'].isna().all():
        mask = df['Gene_Symbol'].str.upper().isin([g.upper() for g in gene_list])
        subset = df[mask].copy()

    if len(subset) > 0:
        n_genes = subset['Canonical_Gene_Symbol'].nunique() if not subset['Canonical_Gene_Symbol'].isna().all() else subset['Gene_Symbol'].nunique()
        print(f"  {family_name}: {n_genes} genes, {len(subset)} measurements")

    return subset

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_pcolce_correlations(df, protein_list, protein_family_name):
    """
    Compute per-study and meta-analytic correlations between PCOLCE and target proteins.

    Args:
        df: Full dataset
        protein_list: List of protein symbols to correlate with PCOLCE
        protein_family_name: Name for output labeling

    Returns:
        DataFrame with correlation results
    """
    print(f"\n{'='*80}")
    print(f"Computing correlations: PCOLCE × {protein_family_name}")
    print(f"{'='*80}")

    # Extract PCOLCE data
    pcolce_df = df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE'].copy()
    if len(pcolce_df) == 0:
        pcolce_df = df[df['Gene_Symbol'].str.upper().str.contains('PCOLCE', na=False)].copy()

    # Extract target protein family
    target_df = extract_protein_family(df, protein_list, protein_family_name)

    if len(target_df) == 0:
        print(f"  ⚠️  No data found for {protein_family_name}")
        return pd.DataFrame()

    results = []

    # Get unique target genes
    target_genes = target_df['Canonical_Gene_Symbol'].dropna().unique()
    if len(target_genes) == 0:
        target_genes = target_df['Gene_Symbol'].dropna().unique()

    for gene in target_genes:
        # Extract gene data
        gene_df = target_df[
            (target_df['Canonical_Gene_Symbol'] == gene) |
            (target_df['Gene_Symbol'] == gene)
        ].copy()

        # Merge with PCOLCE on Study_ID and Tissue_Compartment
        merged = pd.merge(
            pcolce_df[['Study_ID', 'Tissue_Compartment', 'Zscore_Delta']],
            gene_df[['Study_ID', 'Tissue_Compartment', 'Zscore_Delta']],
            on=['Study_ID', 'Tissue_Compartment'],
            suffixes=('_PCOLCE', '_Target')
        )

        if len(merged) < 3:
            continue  # Need at least 3 points for correlation

        # Compute Pearson and Spearman correlations
        pearson_r, pearson_p = stats.pearsonr(merged['Zscore_Delta_PCOLCE'],
                                                merged['Zscore_Delta_Target'])
        spearman_r, spearman_p = stats.spearmanr(merged['Zscore_Delta_PCOLCE'],
                                                   merged['Zscore_Delta_Target'])

        # Meta-statistics
        n_studies = merged['Study_ID'].nunique()
        n_tissues = merged['Tissue_Compartment'].nunique()
        mean_pcolce_delta = merged['Zscore_Delta_PCOLCE'].mean()
        mean_target_delta = merged['Zscore_Delta_Target'].mean()

        results.append({
            'Protein_Family': protein_family_name,
            'Gene': gene,
            'N_Measurements': len(merged),
            'N_Studies': n_studies,
            'N_Tissues': n_tissues,
            'Pearson_r': pearson_r,
            'Pearson_p': pearson_p,
            'Spearman_r': spearman_r,
            'Spearman_p': spearman_p,
            'Mean_PCOLCE_Delta': mean_pcolce_delta,
            'Mean_Target_Delta': mean_target_delta,
            'Significant': (abs(pearson_r) > CORRELATION_THRESHOLD) and (pearson_p < 0.05)
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print(f"\n  Analyzed {len(results_df)} genes:")
        print(f"    Significant correlations (|r|>{CORRELATION_THRESHOLD}, p<0.05): {results_df['Significant'].sum()}")
        print(f"    Positive correlations: {(results_df['Pearson_r'] > 0).sum()}")
        print(f"    Negative correlations: {(results_df['Pearson_r'] < 0).sum()}")

        # Show top correlations
        if results_df['Significant'].sum() > 0:
            top = results_df[results_df['Significant']].nlargest(5, 'Pearson_r', keep='all')
            print(f"\n  Top positive correlations:")
            for _, row in top.head(3).iterrows():
                print(f"    {row['Gene']}: r={row['Pearson_r']:.3f}, p={row['Pearson_p']:.4f}")

            bottom = results_df[results_df['Significant']].nsmallest(5, 'Pearson_r', keep='all')
            print(f"\n  Top negative correlations:")
            for _, row in bottom.head(3).iterrows():
                print(f"    {row['Gene']}: r={row['Pearson_r']:.3f}, p={row['Pearson_p']:.4f}")

    return results_df

def compute_compensatory_analysis(df):
    """
    Screen for compensatory proteases showing inverse trajectories to PCOLCE.

    Args:
        df: Full dataset

    Returns:
        DataFrame with protease analysis results
    """
    print(f"\n{'='*80}")
    print(f"Compensatory Protease Analysis")
    print(f"{'='*80}")

    # Extract PCOLCE mean delta
    pcolce_df = df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE'].copy()
    if len(pcolce_df) == 0:
        pcolce_df = df[df['Gene_Symbol'].str.upper().str.contains('PCOLCE', na=False)].copy()

    pcolce_mean_delta = pcolce_df['Zscore_Delta'].mean()
    print(f"\n  PCOLCE mean Δz: {pcolce_mean_delta:.3f}")

    results = []

    for category, proteases in PROCOLLAGEN_PROCESSORS.items():
        print(f"\n  {category}:")

        for protease in proteases:
            protease_df = df[
                df['Canonical_Gene_Symbol'].str.upper().str.contains(protease.upper(), na=False) |
                df['Gene_Symbol'].str.upper().str.contains(protease.upper(), na=False)
            ].copy()

            if len(protease_df) == 0:
                print(f"    {protease}: NOT FOUND")
                continue

            mean_delta = protease_df['Zscore_Delta'].mean()
            n_studies = protease_df['Study_ID'].nunique()
            n_measurements = len(protease_df)

            # Check if inverse (compensatory)
            is_inverse = (pcolce_mean_delta < 0 and mean_delta > 0) or \
                         (pcolce_mean_delta > 0 and mean_delta < 0)

            inverse_strength = abs(mean_delta + pcolce_mean_delta)  # Lower = more inverse

            print(f"    {protease}: Δz={mean_delta:.3f} ({n_studies} studies, {n_measurements} measurements) "
                  f"{'✓ INVERSE' if is_inverse else ''}")

            results.append({
                'Category': category,
                'Protease': protease,
                'Mean_Delta_Z': mean_delta,
                'N_Studies': n_studies,
                'N_Measurements': n_measurements,
                'Is_Inverse_to_PCOLCE': is_inverse,
                'Inverse_Strength': inverse_strength if is_inverse else np.nan
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=['Mean_Delta_Z'])

    if len(results_df) > 0:
        n_inverse = results_df['Is_Inverse_to_PCOLCE'].sum()
        print(f"\n  Summary:")
        print(f"    Total proteases found: {len(results_df)}")
        print(f"    Inverse to PCOLCE: {n_inverse}")
        print(f"    Same direction: {len(results_df) - n_inverse}")

    return results_df

def tissue_stratified_analysis(df):
    """
    Analyze PCOLCE-collagen correlations stratified by tissue type.

    Args:
        df: Full dataset

    Returns:
        DataFrame with tissue-stratified results
    """
    print(f"\n{'='*80}")
    print(f"Tissue-Stratified Analysis")
    print(f"{'='*80}")

    # Get PCOLCE data
    pcolce_df = df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE'].copy()
    if len(pcolce_df) == 0:
        pcolce_df = df[df['Gene_Symbol'].str.upper().str.contains('PCOLCE', na=False)].copy()

    # Get all fibrillar collagens
    all_collagens = COLLAGEN_FIBRILLAR + COLLAGEN_NETWORK + COLLAGEN_FACIT
    collagen_df = extract_protein_family(df, all_collagens, "All Collagens")

    results = []

    tissues = df['Tissue'].dropna().unique()

    for tissue in tissues:
        tissue_pcolce = pcolce_df[pcolce_df['Tissue'] == tissue]
        tissue_collagen = collagen_df[collagen_df['Tissue'] == tissue]

        if len(tissue_pcolce) == 0 or len(tissue_collagen) == 0:
            continue

        # Merge
        merged = pd.merge(
            tissue_pcolce[['Study_ID', 'Tissue_Compartment', 'Zscore_Delta']],
            tissue_collagen[['Study_ID', 'Tissue_Compartment', 'Gene_Symbol', 'Zscore_Delta']],
            on=['Study_ID', 'Tissue_Compartment'],
            suffixes=('_PCOLCE', '_Collagen')
        )

        if len(merged) < 3:
            continue

        # Compute correlation
        r, p = stats.pearsonr(merged['Zscore_Delta_PCOLCE'], merged['Zscore_Delta_Collagen'])

        mean_pcolce = tissue_pcolce['Zscore_Delta'].mean()
        mean_collagen = tissue_collagen['Zscore_Delta'].mean()

        n_collagens = merged['Gene_Symbol'].nunique()

        print(f"\n  {tissue}:")
        print(f"    PCOLCE: Δz={mean_pcolce:.3f} ({len(tissue_pcolce)} measurements)")
        print(f"    Collagens: Δz={mean_collagen:.3f} ({n_collagens} genes, {len(tissue_collagen)} measurements)")
        print(f"    Correlation: r={r:.3f}, p={p:.4f}")

        results.append({
            'Tissue': tissue,
            'N_PCOLCE_Measurements': len(tissue_pcolce),
            'N_Collagen_Genes': n_collagens,
            'N_Collagen_Measurements': len(tissue_collagen),
            'Mean_PCOLCE_Delta': mean_pcolce,
            'Mean_Collagen_Delta': mean_collagen,
            'Correlation_r': r,
            'Correlation_p': p,
            'Significant': (abs(r) > CORRELATION_THRESHOLD) and (p < 0.05)
        })

    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_correlation_heatmap(corr_df, output_path):
    """
    Create heatmap of PCOLCE-protein correlations.

    Args:
        corr_df: DataFrame with correlation results
        output_path: Path to save figure
    """
    if len(corr_df) == 0:
        print("  ⚠️  No data to plot")
        return

    # Pivot for heatmap
    pivot = corr_df.pivot_table(
        index='Gene',
        columns='Protein_Family',
        values='Pearson_r',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot)*0.4)))

    sns.heatmap(
        pivot,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Pearson r'},
        ax=ax
    )

    ax.set_title('PCOLCE × Collagen Family Correlations\n(Δz aging trajectories)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Protein Family', fontsize=12)
    ax.set_ylabel('Gene', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved heatmap: {output_path.name}")

def plot_pcolce_trajectory(df, output_path):
    """
    Plot PCOLCE z-score trajectories across studies.

    Args:
        df: Full dataset
        output_path: Path to save figure
    """
    pcolce_df = df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE'].copy()
    if len(pcolce_df) == 0:
        pcolce_df = df[df['Gene_Symbol'].str.upper().str.contains('PCOLCE', na=False)].copy()

    if len(pcolce_df) == 0:
        print("  ⚠️  No PCOLCE data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Δz by study
    study_summary = pcolce_df.groupby('Study_ID')['Zscore_Delta'].agg(['mean', 'std', 'count']).reset_index()
    study_summary = study_summary.sort_values('mean')

    axes[0].barh(study_summary['Study_ID'], study_summary['mean'],
                 xerr=study_summary['std'], capsize=5, color='steelblue', alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Mean Δz (Old - Young)', fontsize=12)
    axes[0].set_ylabel('Study', fontsize=12)
    axes[0].set_title('PCOLCE Δz by Study\n(7 studies, mean=-1.41)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Panel B: Tissue distribution
    if 'Tissue' in pcolce_df.columns:
        tissue_summary = pcolce_df.groupby('Tissue')['Zscore_Delta'].agg(['mean', 'std', 'count']).reset_index()
        tissue_summary = tissue_summary.sort_values('mean')

        axes[1].barh(tissue_summary['Tissue'], tissue_summary['mean'],
                     xerr=tissue_summary['std'], capsize=5, color='coral', alpha=0.7)
        axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Mean Δz (Old - Young)', fontsize=12)
        axes[1].set_ylabel('Tissue', fontsize=12)
        axes[1].set_title('PCOLCE Δz by Tissue', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved trajectory plot: {output_path.name}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline."""

    print("\n" + "="*80)
    print("AGENT 4: PCOLCE RESEARCH ANOMALY - INTEGRATIVE ANALYSIS")
    print("="*80)
    print("\nObjective: Resolve paradox of PCOLCE depletion in aging vs upregulation in fibrosis")
    print("Approach: Correlation analysis with collagen proteins and compensatory proteases")

    # Load data
    df = load_and_validate_data(V2_PATH, expected_pcolce_studies=7)

    # Create output subdirectory
    results_dir = OUTPUT_DIR
    results_dir.mkdir(exist_ok=True)

    # ========================================================================
    # ANALYSIS 1: PCOLCE × Fibrillar Collagens
    # ========================================================================

    fibrillar_corr = compute_pcolce_correlations(
        df,
        COLLAGEN_FIBRILLAR,
        "Fibrillar_Collagens"
    )

    if len(fibrillar_corr) > 0:
        fibrillar_corr.to_csv(results_dir / "pcolce_fibrillar_collagen_correlations.csv", index=False)
        print(f"\n  ✓ Saved: pcolce_fibrillar_collagen_correlations.csv")

    # ========================================================================
    # ANALYSIS 2: PCOLCE × Network Collagens
    # ========================================================================

    network_corr = compute_pcolce_correlations(
        df,
        COLLAGEN_NETWORK,
        "Network_Collagens"
    )

    if len(network_corr) > 0:
        network_corr.to_csv(results_dir / "pcolce_network_collagen_correlations.csv", index=False)
        print(f"\n  ✓ Saved: pcolce_network_collagen_correlations.csv")

    # ========================================================================
    # ANALYSIS 3: PCOLCE × FACIT Collagens
    # ========================================================================

    facit_corr = compute_pcolce_correlations(
        df,
        COLLAGEN_FACIT,
        "FACIT_Collagens"
    )

    if len(facit_corr) > 0:
        facit_corr.to_csv(results_dir / "pcolce_facit_collagen_correlations.csv", index=False)
        print(f"\n  ✓ Saved: pcolce_facit_collagen_correlations.csv")

    # ========================================================================
    # ANALYSIS 4: Compensatory Protease Screen
    # ========================================================================

    compensatory = compute_compensatory_analysis(df)

    if len(compensatory) > 0:
        compensatory.to_csv(results_dir / "compensatory_protease_analysis.csv", index=False)
        print(f"\n  ✓ Saved: compensatory_protease_analysis.csv")

    # ========================================================================
    # ANALYSIS 5: Tissue Stratification
    # ========================================================================

    tissue_strat = tissue_stratified_analysis(df)

    if len(tissue_strat) > 0:
        tissue_strat.to_csv(results_dir / "tissue_stratified_correlations.csv", index=False)
        print(f"\n  ✓ Saved: tissue_stratified_correlations.csv")

    # ========================================================================
    # ANALYSIS 6: Collagen Maturation Enzymes
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"Collagen Maturation Enzyme Analysis")
    print(f"{'='*80}")

    maturation_results = []

    for category, enzymes in COLLAGEN_MATURATION.items():
        print(f"\n  {category}:")
        for enzyme in enzymes:
            enzyme_df = df[
                df['Canonical_Gene_Symbol'].str.upper().str.contains(enzyme.upper(), na=False) |
                df['Gene_Symbol'].str.upper().str.contains(enzyme.upper(), na=False)
            ].copy()

            if len(enzyme_df) > 0:
                mean_delta = enzyme_df['Zscore_Delta'].mean()
                n_studies = enzyme_df['Study_ID'].nunique()
                print(f"    {enzyme}: Δz={mean_delta:.3f} ({n_studies} studies)")

                maturation_results.append({
                    'Category': category,
                    'Enzyme': enzyme,
                    'Mean_Delta_Z': mean_delta,
                    'N_Studies': n_studies,
                    'N_Measurements': len(enzyme_df)
                })

    if len(maturation_results) > 0:
        maturation_df = pd.DataFrame(maturation_results)
        maturation_df.to_csv(results_dir / "collagen_maturation_enzymes.csv", index=False)
        print(f"\n  ✓ Saved: collagen_maturation_enzymes.csv")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"Generating Visualizations")
    print(f"{'='*80}")

    # Combine all correlations for heatmap
    all_corr = pd.concat([fibrillar_corr, network_corr, facit_corr], ignore_index=True)
    if len(all_corr) > 0:
        plot_correlation_heatmap(all_corr, results_dir / "pcolce_collagen_correlation_heatmap.png")

    # PCOLCE trajectory plot
    plot_pcolce_trajectory(df, results_dir / "pcolce_trajectory_by_study_tissue.png")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")

    summary = {
        'Dataset': 'V2_Batch_Corrected',
        'Total_Rows': len(df),
        'Total_Proteins': df['Gene_Symbol'].nunique(),
        'Total_Studies': df['Study_ID'].nunique(),
        'PCOLCE_Mean_Delta': df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE']['Zscore_Delta'].mean(),
        'PCOLCE_Studies': df[df['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE']['Study_ID'].nunique(),
        'Fibrillar_Collagens_Analyzed': len(fibrillar_corr),
        'Network_Collagens_Analyzed': len(network_corr),
        'FACIT_Collagens_Analyzed': len(facit_corr),
        'Significant_Positive_Correlations': all_corr[all_corr['Significant'] & (all_corr['Pearson_r'] > 0)].shape[0] if len(all_corr) > 0 else 0,
        'Significant_Negative_Correlations': all_corr[all_corr['Significant'] & (all_corr['Pearson_r'] < 0)].shape[0] if len(all_corr) > 0 else 0,
        'Compensatory_Proteases_Found': len(compensatory),
        'Inverse_Proteases': compensatory['Is_Inverse_to_PCOLCE'].sum() if len(compensatory) > 0 else 0
    }

    with open(results_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n  Analysis Summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {results_dir}")
    print("\nKey findings:")
    print(f"  - PCOLCE mean Δz: {summary['PCOLCE_Mean_Delta']:.3f}")
    print(f"  - Collagens analyzed: {summary['Fibrillar_Collagens_Analyzed'] + summary['Network_Collagens_Analyzed'] + summary['FACIT_Collagens_Analyzed']}")
    print(f"  - Significant positive correlations: {summary['Significant_Positive_Correlations']}")
    print(f"  - Significant negative correlations: {summary['Significant_Negative_Correlations']}")
    print(f"  - Inverse compensatory proteases: {summary['Inverse_Proteases']}")

    print("\nNext steps:")
    print("  1. Review correlation results in CSV files")
    print("  2. Examine heatmap for patterns")
    print("  3. Interpret findings in biological context")
    print("  4. Draft unified model in 04_unified_model_agent_4.md")

if __name__ == "__main__":
    main()
