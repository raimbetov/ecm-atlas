#!/usr/bin/env python3
"""
H12 - Enhanced Phase-Specific Enrichment Analysis
Agent: claude_code

Uses ECM-specific metabolic proxies and mechanical markers from actual dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_12_metabolic_mechanical_transition/claude_code'
VIZ_DIR = f'{OUTPUT_DIR}/visualizations_claude_code'

VELOCITY_THRESHOLD = 1.65

# Tissue velocities from H09
TISSUE_VELOCITIES = {
    'Skeletal_muscle_Gastrocnemius': 1.02,
    'Brain_Hippocampus': 1.18,
    'Liver': 1.34,
    'Heart_Native': 1.58,
    'Ovary_Cortex': 1.53,
    'Heart': 1.82,
    'Kidney_Cortex': 1.91,
    'Skin_Dermis': 2.17,
    'Intervertebral_disc_IAF': 3.12,
    'Tubulointerstitial': 3.45,
    'Lung': 4.29,
}

# Enhanced marker definitions based on actual dataset
METABOLIC_PHASE_I_MARKERS = {
    # Oxygen-dependent collagen processing (metabolic regulation)
    'collagen_hydroxylases': ['PLOD1', 'PLOD2', 'PLOD3', 'P4HA1', 'P4HA2', 'P4HA3'],

    # Coagulation cascade (early inflammatory response, metabolic dysregulation)
    'coagulation': ['F2', 'F9', 'F10', 'F12', 'F13A1', 'F13B', 'PLG', 'FGG', 'FGB', 'FGA'],

    # Serpins (protease inhibitors, early protective response)
    'serpins': ['SERPINA1', 'SERPINA3', 'SERPINA5', 'SERPINA7', 'SERPINC1', 'SERPIND1',
                'SERPINE1', 'SERPINE2', 'SERPINF1', 'SERPING1', 'SERPINH1'],

    # Early inflammatory secreted factors
    'inflammatory_factors': ['S100A1', 'S100A4', 'S100A6', 'S100A8', 'S100A9', 'S100A11', 'S100B',
                            'TGFB1', 'TGFB2', 'TGFB3', 'IL11', 'CCL21', 'CXCL10', 'CXCL12', 'CXCL14'],

    # Cysteine protease inhibitors (metabolic regulation of degradation)
    'cystatin_inhibitors': ['CST3', 'CSTB', 'CSTA', 'CST6'],
}

MECHANICAL_PHASE_II_MARKERS = {
    # Crosslinking enzymes (irreversible mechanical remodeling)
    'crosslinking': ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4', 'TGM1', 'TGM2', 'TGM3'],

    # Fibrillar collagens (structural load-bearing)
    'fibrillar_collagens': ['COL1A1', 'COL1A2', 'COL2A1', 'COL3A1', 'COL5A1', 'COL5A2',
                           'COL11A1', 'COL11A2'],

    # Network-forming collagens
    'network_collagens': ['COL4A1', 'COL4A2', 'COL4A3', 'COL6A1', 'COL6A2', 'COL6A3',
                         'COL8A1', 'COL8A2'],

    # Mechanical ECM glycoproteins
    'mechanical_glycoproteins': ['FN1', 'TNC', 'THBS1', 'THBS2', 'THBS3', 'THBS4',
                                'POSTN', 'SPARC', 'FBLN1', 'FBLN2', 'FBLN5'],

    # Matricellular proteins (mechanotransduction)
    'matricellular': ['CTGF', 'NOV', 'WISP1', 'WISP2', 'CCN1', 'CCN2', 'CCN3'],
}

def load_and_prepare_data():
    """Load dataset and assign phases."""
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    df = pd.read_csv(DATA_PATH)
    print(f"Total rows: {len(df)}")
    print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    print(f"Tissues: {df['Tissue'].nunique()}")

    # Assign velocities
    df['Velocity'] = df['Tissue'].map(TISSUE_VELOCITIES)
    df = df[df['Velocity'].notna()]  # Keep only tissues with known velocities

    # Assign phase
    df['Phase'] = df['Velocity'].apply(lambda v: 'Phase_I' if v < VELOCITY_THRESHOLD else 'Phase_II')

    print(f"\nAfter filtering for known velocities: {len(df)} rows")
    print(f"Phase I tissues: {df[df['Phase']=='Phase_I']['Tissue'].nunique()}")
    print(f"Phase II tissues: {df[df['Phase']=='Phase_II']['Tissue'].nunique()}")

    return df

def calculate_protein_phase_profiles(df):
    """
    For each protein, calculate mean z-score in Phase I vs Phase II.
    Returns DataFrame with protein-level statistics.
    """
    print("\n" + "="*80)
    print("PROTEIN PHASE PROFILES")
    print("="*80)

    # Group by protein and phase
    phase_profiles = df.groupby(['Canonical_Gene_Symbol', 'Phase']).agg({
        'Zscore_Delta': ['mean', 'std', 'count']
    }).reset_index()

    phase_profiles.columns = ['Protein', 'Phase', 'Mean_Zscore', 'Std_Zscore', 'N_Samples']

    # Pivot to get Phase I and Phase II side by side
    pivot = phase_profiles.pivot(index='Protein', columns='Phase', values='Mean_Zscore').reset_index()
    pivot.columns.name = None

    # Calculate differential
    if 'Phase_I' in pivot.columns and 'Phase_II' in pivot.columns:
        pivot['Delta_Phase_II_vs_I'] = pivot['Phase_II'] - pivot['Phase_I']
        pivot['Upregulated_Phase_II'] = pivot['Delta_Phase_II_vs_I'] > 0
        pivot['Upregulated_Phase_I'] = pivot['Delta_Phase_II_vs_I'] < 0

    print(f"Total proteins with phase profiles: {len(pivot)}")

    return pivot

def test_marker_enrichment(protein_profiles, marker_dict, target_phase, category_name):
    """
    Test if markers are enriched in target phase using Fisher's exact test.
    """
    print(f"\n=== Testing {category_name} enrichment in {target_phase} ===")

    results = []

    for category, markers in marker_dict.items():
        # Find markers present in dataset
        present_markers = [m for m in markers if m in protein_profiles['Protein'].values]

        if len(present_markers) == 0:
            print(f"  {category}: No markers found in dataset")
            continue

        print(f"\n  {category}: {len(present_markers)}/{len(markers)} markers present")

        # Determine upregulation column
        if target_phase == 'Phase_II':
            upreg_col = 'Upregulated_Phase_II'
        else:
            upreg_col = 'Upregulated_Phase_I'

        # Contingency table
        marker_mask = protein_profiles['Protein'].isin(present_markers)

        a = sum(marker_mask & protein_profiles[upreg_col])  # Markers upregulated
        b = sum(marker_mask & ~protein_profiles[upreg_col])  # Markers not upregulated
        c = sum(~marker_mask & protein_profiles[upreg_col])  # Non-markers upregulated
        d = sum(~marker_mask & ~protein_profiles[upreg_col])  # Non-markers not upregulated

        # Fisher's exact test
        try:
            oddsratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')

            significant = (p_value < 0.05 and oddsratio > 2.0)

            results.append({
                'Category': category,
                'Markers_Present': len(present_markers),
                'Markers_Upregulated': a,
                'Percentage_Upregulated': 100 * a / len(present_markers) if len(present_markers) > 0 else 0,
                'Odds_Ratio': oddsratio,
                'P_Value': p_value,
                'Significant': significant,
                'Target_Phase': target_phase
            })

            status = "✓ SIGNIFICANT" if significant else "✗ Not significant"
            print(f"    OR={oddsratio:.2f}, p={p_value:.4f}, {a}/{len(present_markers)} upregulated - {status}")

        except Exception as e:
            print(f"    Error: {e}")

    return pd.DataFrame(results)

def compare_phase_marker_expression(df, marker_dict, phase_name):
    """
    Compare marker expression between Phase I and Phase II.
    Returns t-test results.
    """
    print(f"\n=== Comparing {phase_name} marker expression across phases ===")

    # Flatten markers
    all_markers = []
    for markers in marker_dict.values():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))

    # Filter for markers present
    marker_data = df[df['Canonical_Gene_Symbol'].isin(all_markers)]

    if len(marker_data) == 0:
        print("  No markers found in dataset")
        return None

    # Group by phase
    phase_i_scores = marker_data[marker_data['Phase'] == 'Phase_I']['Zscore_Delta']
    phase_ii_scores = marker_data[marker_data['Phase'] == 'Phase_II']['Zscore_Delta']

    print(f"  Phase I: n={len(phase_i_scores)}, mean={phase_i_scores.mean():.3f}±{phase_i_scores.std():.3f}")
    print(f"  Phase II: n={len(phase_ii_scores)}, mean={phase_ii_scores.mean():.3f}±{phase_ii_scores.std():.3f}")

    # T-test
    if len(phase_i_scores) > 0 and len(phase_ii_scores) > 0:
        t_stat, p_val = ttest_ind(phase_i_scores, phase_ii_scores)
        cohen_d = (phase_ii_scores.mean() - phase_i_scores.mean()) / np.sqrt((phase_i_scores.var() + phase_ii_scores.var()) / 2)

        print(f"  T-test: t={t_stat:.2f}, p={p_val:.4e}")
        print(f"  Cohen's d: {cohen_d:.2f}")

        return {
            'Phase_I_Mean': phase_i_scores.mean(),
            'Phase_II_Mean': phase_ii_scores.mean(),
            'T_Statistic': t_stat,
            'P_Value': p_val,
            'Cohens_D': cohen_d
        }

    return None

def visualize_phase_markers(protein_profiles, metabolic_enrichment, mechanical_enrichment):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Volcano plot: Phase II vs Phase I
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate -log10(p) for visualization (using absolute delta as proxy)
    protein_profiles['abs_delta'] = protein_profiles['Delta_Phase_II_vs_I'].abs()

    # Plot all proteins
    ax.scatter(protein_profiles['Delta_Phase_II_vs_I'],
              protein_profiles['abs_delta'],
              c='gray', alpha=0.3, s=20, label='Other proteins')

    # Highlight metabolic markers
    metabolic_all = []
    for markers in METABOLIC_PHASE_I_MARKERS.values():
        metabolic_all.extend(markers)
    metabolic_mask = protein_profiles['Protein'].isin(metabolic_all)

    ax.scatter(protein_profiles.loc[metabolic_mask, 'Delta_Phase_II_vs_I'],
              protein_profiles.loc[metabolic_mask, 'abs_delta'],
              c='blue', alpha=0.7, s=50, label='Metabolic markers', marker='^')

    # Highlight mechanical markers
    mechanical_all = []
    for markers in MECHANICAL_PHASE_II_MARKERS.values():
        mechanical_all.extend(markers)
    mechanical_mask = protein_profiles['Protein'].isin(mechanical_all)

    ax.scatter(protein_profiles.loc[mechanical_mask, 'Delta_Phase_II_vs_I'],
              protein_profiles.loc[mechanical_mask, 'abs_delta'],
              c='red', alpha=0.7, s=50, label='Mechanical markers', marker='s')

    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Δ Z-score (Phase II - Phase I)', fontsize=12)
    ax.set_ylabel('|Δ Z-score|', fontsize=12)
    ax.set_title('Protein Expression Change: Phase II vs Phase I', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/phase_volcano_plot_claude_code.png')
    print("  Saved: phase_volcano_plot_claude_code.png")
    plt.close()

    # 2. Enrichment comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Metabolic enrichment
    if not metabolic_enrichment.empty:
        met_sorted = metabolic_enrichment.sort_values('Odds_Ratio', ascending=True)
        colors = ['green' if sig else 'lightgray' for sig in met_sorted['Significant']]
        ax1.barh(met_sorted['Category'], met_sorted['Odds_Ratio'], color=colors)
        ax1.axvline(2.0, color='red', linestyle='--', linewidth=2, label='OR=2.0 threshold')
        ax1.axvline(1.0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Odds Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('Phase I: Metabolic/Regulatory Enrichment', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.set_xlim(left=0)

    # Mechanical enrichment
    if not mechanical_enrichment.empty:
        mech_sorted = mechanical_enrichment.sort_values('Odds_Ratio', ascending=True)
        colors = ['red' if sig else 'lightgray' for sig in mech_sorted['Significant']]
        ax2.barh(mech_sorted['Category'], mech_sorted['Odds_Ratio'], color=colors)
        ax2.axvline(2.0, color='red', linestyle='--', linewidth=2, label='OR=2.0 threshold')
        ax2.axvline(1.0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Odds Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Phase II: Mechanical/Structural Enrichment', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/enrichment_comparison_enhanced_claude_code.png')
    print("  Saved: enrichment_comparison_enhanced_claude_code.png")
    plt.close()

def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print("H12: ENHANCED PHASE-SPECIFIC ANALYSIS")
    print("Agent: claude_code")
    print("="*80 + "\n")

    # Load data
    df = load_and_prepare_data()

    # Calculate protein phase profiles
    protein_profiles = calculate_protein_phase_profiles(df)

    # Test enrichment
    print("\n" + "="*80)
    print("FISHER'S EXACT TEST: MARKER ENRICHMENT")
    print("="*80)

    metabolic_enrichment = test_marker_enrichment(
        protein_profiles, METABOLIC_PHASE_I_MARKERS, 'Phase_I', 'Metabolic/Regulatory'
    )

    mechanical_enrichment = test_marker_enrichment(
        protein_profiles, MECHANICAL_PHASE_II_MARKERS, 'Phase_II', 'Mechanical/Structural'
    )

    # Compare expression
    print("\n" + "="*80)
    print("EXPRESSION COMPARISON")
    print("="*80)

    metabolic_stats = compare_phase_marker_expression(df, METABOLIC_PHASE_I_MARKERS, 'Metabolic')
    mechanical_stats = compare_phase_marker_expression(df, MECHANICAL_PHASE_II_MARKERS, 'Mechanical')

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    protein_profiles.to_csv(f'{OUTPUT_DIR}/protein_phase_profiles_claude_code.csv', index=False)
    print("  Saved: protein_phase_profiles_claude_code.csv")

    if not metabolic_enrichment.empty:
        metabolic_enrichment.to_csv(f'{OUTPUT_DIR}/enrichment_metabolic_phase_i_claude_code.csv', index=False)
        print("  Saved: enrichment_metabolic_phase_i_claude_code.csv")

    if not mechanical_enrichment.empty:
        mechanical_enrichment.to_csv(f'{OUTPUT_DIR}/enrichment_mechanical_phase_ii_claude_code.csv', index=False)
        print("  Saved: enrichment_mechanical_phase_ii_claude_code.csv")

    # Visualizations
    visualize_phase_markers(protein_profiles, metabolic_enrichment, mechanical_enrichment)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not metabolic_enrichment.empty:
        sig_metabolic = metabolic_enrichment[metabolic_enrichment['Significant']]
        print(f"\n✓ Metabolic markers: {len(sig_metabolic)}/{len(metabolic_enrichment)} categories significant (OR>2.0, p<0.05)")
        for _, row in sig_metabolic.iterrows():
            print(f"    - {row['Category']}: OR={row['Odds_Ratio']:.2f}, p={row['P_Value']:.4f}")

    if not mechanical_enrichment.empty:
        sig_mechanical = mechanical_enrichment[mechanical_enrichment['Significant']]
        print(f"\n✓ Mechanical markers: {len(sig_mechanical)}/{len(mechanical_enrichment)} categories significant (OR>2.0, p<0.05)")
        for _, row in sig_mechanical.iterrows():
            print(f"    - {row['Category']}: OR={row['Odds_Ratio']:.2f}, p={row['P_Value']:.4f}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
