#!/usr/bin/env python3
"""
Compartment Antagonistic Mechanical Stress Adaptation Analysis
Agent: claude_code
Hypothesis: High-load compartments upregulate structural ECM proteins while low-load
compartments downregulate the same proteins during aging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_01/hypothesis_01_compartment_mechanical_stress/claude_code'
VIZ_DIR = f'{OUTPUT_DIR}/visualizations_claude_code'

# Mechanical stress classification (based on biomechanics literature)
MECHANICAL_STRESS_MAP = {
    # High-load compartments (continuous weight-bearing, high compression)
    'Skeletal_muscle_Soleus': 1.0,  # Postural slow-twitch muscle
    'NP': 1.0,  # Nucleus pulposus - hydraulic shock absorber
    'Cortex': 1.0,  # Bone cortex - primary load-bearing

    # Low-load compartments (intermittent/minimal mechanical stress)
    'Skeletal_muscle_TA': 0.0,  # Tibialis anterior - fast-twitch, intermittent
    'IAF': 0.0,  # Inner annulus fibrosus - tensile, less compression
    'OAF': 0.0,  # Outer annulus fibrosus - tensile, peripheral
    'Hippocampus': 0.0,  # Brain tissue - minimal mechanical load

    # Mixed/moderate-load compartments
    'Skeletal_muscle_Gastrocnemius': 0.5,  # Mixed fiber type
    'Skeletal_muscle_EDL': 0.5,  # Extensor digitorum longus - intermediate
    'Glomerular': 0.5,  # Fluid shear stress
    'Tubulointerstitial': 0.5,  # Moderate fluid dynamics
    'Lung': 0.5,  # Cyclic stretch
    'Skin dermis': 0.5,  # Variable tensile stress
    'Ovary': 0.5,  # Cyclic remodeling
}

# Protein categorization
STRUCTURAL_PROTEINS = [
    'COL1A1', 'COL1A2', 'COL2A1', 'COL3A1', 'COL4A1', 'COL4A2', 'COL5A1', 'COL5A2',
    'COL6A1', 'COL6A2', 'COL6A3', 'COL11A1', 'COL11A2', 'COL12A1', 'COL14A1', 'COL15A1',
    'FBN1', 'FBN2', 'FBLN1', 'FBLN2', 'FBLN5',
    'LAMA1', 'LAMA2', 'LAMA4', 'LAMA5', 'LAMB1', 'LAMB2', 'LAMC1',
    'ACAN', 'VCAN', 'DCN', 'BGN', 'LUM', 'FMOD', 'PRELP', 'OGN',
    'CILP', 'CILP2', 'CHAD', 'POSTN', 'TNC', 'TNR', 'TNXB'
]

REGULATORY_PROTEINS = [
    'MMP1', 'MMP2', 'MMP3', 'MMP7', 'MMP9', 'MMP10', 'MMP13', 'MMP14',
    'TIMP1', 'TIMP2', 'TIMP3', 'TIMP4',
    'SERPINA1', 'SERPINA3', 'SERPINC1', 'SERPINE1', 'SERPINF1', 'SERPINH1',
    'PLAU', 'PLAT', 'PLG', 'F2', 'F9', 'F10', 'F12', 'F13A1',
    'PLOD1', 'PLOD2', 'PLOD3', 'P4HA1', 'P4HA2', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4',
    'CTSK', 'CTSL', 'CTSB', 'CTSD', 'ADAMTS1', 'ADAMTS4', 'ADAMTS5'
]

# ============================================================================
# 2. DATA LOADING AND VALIDATION
# ============================================================================

def load_and_validate_data():
    """Load dataset and perform sanity checks."""
    print("=" * 80)
    print("LOADING AND VALIDATING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Sanity checks
    assert df.shape[0] >= 3000, f"Expected ≥3000 rows, got {df.shape[0]}"
    assert 'Tissue_Compartment' in df.columns, "Missing Tissue_Compartment column!"
    assert 'Zscore_Delta' in df.columns, "Missing Zscore_Delta column!"
    print("✓ Required columns present")

    # Check compartment coverage
    compartments = df['Tissue_Compartment'].unique()
    print(f"✓ Total compartments: {len(compartments)}")

    muscle_comps = [c for c in compartments if 'Soleus' in c or 'TA' in c]
    disc_comps = [c for c in compartments if c in ['NP', 'IAF', 'OAF']]

    assert len(muscle_comps) > 0, "Missing muscle compartments!"
    assert len(disc_comps) > 0, "Missing disc compartments!"
    print(f"✓ Muscle compartments: {muscle_comps}")
    print(f"✓ Disc compartments: {disc_comps}")

    return df

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """Filter and annotate data for analysis."""
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)

    # Filter out non-specific compartments
    exclude_compartments = ['Native_Tissue', 'Decellularized_Tissue']
    df_filtered = df[~df['Tissue_Compartment'].isin(exclude_compartments)].copy()
    print(f"✓ Excluded non-specific compartments: {len(df) - len(df_filtered)} rows removed")

    # Keep only valid Zscore_Delta
    df_filtered = df_filtered[df_filtered['Zscore_Delta'].notna()].copy()
    print(f"✓ Kept valid Zscore_Delta: {len(df_filtered)} rows")

    # Add mechanical stress annotation
    df_filtered['Mechanical_Stress'] = df_filtered['Tissue_Compartment'].map(MECHANICAL_STRESS_MAP)
    classified = df_filtered['Mechanical_Stress'].notna().sum()
    print(f"✓ Compartments classified by mechanical stress: {classified}/{len(df_filtered)}")

    # Add protein type annotation
    df_filtered['Protein_Type'] = 'Other'
    df_filtered.loc[df_filtered['Canonical_Gene_Symbol'].isin(STRUCTURAL_PROTEINS), 'Protein_Type'] = 'Structural'
    df_filtered.loc[df_filtered['Canonical_Gene_Symbol'].isin(REGULATORY_PROTEINS), 'Protein_Type'] = 'Regulatory'

    print(f"✓ Structural proteins: {(df_filtered['Protein_Type'] == 'Structural').sum()}")
    print(f"✓ Regulatory proteins: {(df_filtered['Protein_Type'] == 'Regulatory').sum()}")

    return df_filtered

# ============================================================================
# 4. ANTAGONISM DETECTION
# ============================================================================

def identify_antagonistic_pairs(df):
    """Identify protein-compartment pairs showing antagonistic aging patterns."""
    print("\n" + "=" * 80)
    print("IDENTIFYING ANTAGONISTIC PAIRS")
    print("=" * 80)

    antagonistic_pairs = []

    # Get unique proteins present in multiple compartments
    proteins = df.groupby('Canonical_Gene_Symbol')['Tissue_Compartment'].nunique()
    multi_compartment_proteins = proteins[proteins >= 2].index.tolist()
    print(f"✓ Proteins in ≥2 compartments: {len(multi_compartment_proteins)}")

    # For each protein, find antagonistic compartment pairs
    for protein in multi_compartment_proteins:
        protein_data = df[df['Canonical_Gene_Symbol'] == protein].copy()

        # Get all compartment pairs for this protein
        compartments = protein_data['Tissue_Compartment'].unique()

        for comp_a, comp_b in combinations(compartments, 2):
            data_a = protein_data[protein_data['Tissue_Compartment'] == comp_a]
            data_b = protein_data[protein_data['Tissue_Compartment'] == comp_b]

            if len(data_a) == 0 or len(data_b) == 0:
                continue

            # Take mean if multiple measurements
            zscore_a = data_a['Zscore_Delta'].mean()
            zscore_b = data_b['Zscore_Delta'].mean()

            # Check for antagonism (opposite signs)
            if np.sign(zscore_a) != np.sign(zscore_b):
                antagonism_mag = abs(zscore_a - zscore_b)

                # Minimum threshold: 1.0 SD difference
                if antagonism_mag >= 1.0:
                    antagonistic_pairs.append({
                        'Gene_Symbol': protein,
                        'Compartment_A': comp_a,
                        'Compartment_B': comp_b,
                        'Delta_z_A': zscore_a,
                        'Delta_z_B': zscore_b,
                        'Antagonism_Magnitude': antagonism_mag,
                        'Tissue': data_a['Tissue'].iloc[0],
                        'Matrisome_Category': data_a['Matrisome_Category_Simplified'].iloc[0],
                        'Mechanical_Stress_A': data_a['Mechanical_Stress'].iloc[0] if 'Mechanical_Stress' in data_a.columns else np.nan,
                        'Mechanical_Stress_B': data_b['Mechanical_Stress'].iloc[0] if 'Mechanical_Stress' in data_b.columns else np.nan,
                        'Protein_Type': data_a['Protein_Type'].iloc[0] if 'Protein_Type' in data_a.columns else 'Other'
                    })

    df_antagonistic = pd.DataFrame(antagonistic_pairs)
    df_antagonistic = df_antagonistic.sort_values('Antagonism_Magnitude', ascending=False)

    print(f"✓ Total antagonistic pairs identified: {len(df_antagonistic)}")
    print(f"✓ Top antagonism magnitude: {df_antagonistic['Antagonism_Magnitude'].max():.2f} SD")
    print(f"\nTop 10 antagonistic proteins:")
    print(df_antagonistic[['Gene_Symbol', 'Compartment_A', 'Compartment_B', 'Antagonism_Magnitude']].head(10))

    return df_antagonistic

# ============================================================================
# 5. STATISTICAL TESTING
# ============================================================================

def test_mechanical_stress_hypothesis(df, df_antagonistic):
    """Test if high-load compartments show higher Δz for structural proteins."""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING: HIGH-LOAD vs LOW-LOAD")
    print("=" * 80)

    # Filter for structural proteins with classified mechanical stress
    structural = df[(df['Protein_Type'] == 'Structural') &
                    (df['Mechanical_Stress'].notna())].copy()

    high_load = structural[structural['Mechanical_Stress'] == 1.0]['Zscore_Delta'].values
    low_load = structural[structural['Mechanical_Stress'] == 0.0]['Zscore_Delta'].values

    print(f"High-load structural proteins: n = {len(high_load)}")
    print(f"  Mean Δz: {np.mean(high_load):.3f} ± {np.std(high_load):.3f}")
    print(f"  Median Δz: {np.median(high_load):.3f}")

    print(f"\nLow-load structural proteins: n = {len(low_load)}")
    print(f"  Mean Δz: {np.mean(low_load):.3f} ± {np.std(low_load):.3f}")
    print(f"  Median Δz: {np.median(low_load):.3f}")

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(high_load, low_load, alternative='greater')

    # Effect size (rank-biserial correlation)
    n1, n2 = len(high_load), len(low_load)
    r = 1 - (2*statistic) / (n1 * n2)  # rank-biserial correlation

    print(f"\nMann-Whitney U test (high-load > low-load):")
    print(f"  U-statistic: {statistic:.1f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Effect size (r): {r:.3f}")
    print(f"  Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")

    return {
        'test': 'Mann-Whitney U',
        'comparison': 'High-load vs Low-load (structural proteins)',
        'n_high': len(high_load),
        'n_low': len(low_load),
        'mean_high': np.mean(high_load),
        'mean_low': np.mean(low_load),
        'median_high': np.median(high_load),
        'median_low': np.median(low_load),
        'U_statistic': statistic,
        'p_value': p_value,
        'effect_size_r': r,
        'significant': p_value < 0.05
    }

def test_mechanical_stress_correlation(df):
    """Test correlation between mechanical stress and Δz for different protein types."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS: MECHANICAL STRESS vs Δz")
    print("=" * 80)

    results = []

    for protein_type in ['Structural', 'Regulatory']:
        subset = df[(df['Protein_Type'] == protein_type) &
                    (df['Mechanical_Stress'].notna())].copy()

        if len(subset) < 10:
            print(f"\n{protein_type} proteins: insufficient data (n={len(subset)})")
            continue

        rho, p_value = stats.spearmanr(subset['Mechanical_Stress'], subset['Zscore_Delta'])

        print(f"\n{protein_type} proteins:")
        print(f"  n = {len(subset)}")
        print(f"  Spearman ρ: {rho:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant: {'✓ YES' if p_value < 0.05 else '✗ NO'}")

        results.append({
            'Protein_Type': protein_type,
            'n': len(subset),
            'Spearman_rho': rho,
            'p_value': p_value,
            'significant': p_value < 0.05
        })

    return pd.DataFrame(results)

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def create_antagonism_heatmap(df, df_antagonistic, output_path):
    """Create heatmap of top antagonistic proteins across compartments."""
    print("\n" + "=" * 80)
    print("CREATING ANTAGONISM HEATMAP")
    print("=" * 80)

    # Get top 20 antagonistic proteins
    top_proteins = df_antagonistic.nlargest(20, 'Antagonism_Magnitude')['Gene_Symbol'].unique()

    # Create pivot table
    heatmap_data = df[df['Canonical_Gene_Symbol'].isin(top_proteins)].copy()
    pivot = heatmap_data.pivot_table(
        values='Zscore_Delta',
        index='Canonical_Gene_Symbol',
        columns='Tissue_Compartment',
        aggfunc='mean'
    )

    # Sort by compartments with mechanical stress classification
    compartment_order = sorted(pivot.columns,
                               key=lambda x: MECHANICAL_STRESS_MAP.get(x, 0.25),
                               reverse=True)
    pivot = pivot[compartment_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                cbar_kws={'label': 'Zscore_Delta (aging trajectory)'},
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_xlabel('Tissue Compartment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene Symbol', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Antagonistic Proteins: Aging Trajectories Across Compartments\n(High-load → Low-load ordering)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Heatmap saved: {output_path}")

def create_scatter_plot(df, output_path):
    """Create scatter plot of Δz vs mechanical stress."""
    print("\n" + "=" * 80)
    print("CREATING SCATTER PLOT: Δz vs MECHANICAL STRESS")
    print("=" * 80)

    plot_data = df[(df['Mechanical_Stress'].notna()) &
                   (df['Protein_Type'].isin(['Structural', 'Regulatory']))].copy()

    # Add jitter to mechanical stress for visualization
    np.random.seed(42)
    plot_data['Mechanical_Stress_Jittered'] = plot_data['Mechanical_Stress'] + np.random.normal(0, 0.03, len(plot_data))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, protein_type in enumerate(['Structural', 'Regulatory']):
        subset = plot_data[plot_data['Protein_Type'] == protein_type]

        ax = axes[idx]

        # Scatter plot
        sns.scatterplot(data=subset, x='Mechanical_Stress_Jittered', y='Zscore_Delta',
                       hue='Matrisome_Category_Simplified', alpha=0.6, s=50, ax=ax)

        # Regression line
        from scipy.stats import linregress
        valid = subset[['Mechanical_Stress', 'Zscore_Delta']].dropna()
        if len(valid) > 5:
            slope, intercept, r_value, p_value, std_err = linregress(valid['Mechanical_Stress'], valid['Zscore_Delta'])
            x_line = np.array([0, 1])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f}, p={p_value:.4f})')

        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Mechanical Stress (0=low, 1=high)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Zscore_Delta (aging trajectory)', fontsize=12, fontweight='bold')
        ax.set_title(f'{protein_type} Proteins (n={len(subset)})', fontsize=13, fontweight='bold')
        ax.set_xlim(-0.15, 1.15)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Mechanical Stress vs Aging Trajectory by Protein Type',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Scatter plot saved: {output_path}")

def create_top_antagonistic_barplot(df_antagonistic, output_path):
    """Create bar plot of top 20 antagonistic proteins."""
    print("\n" + "=" * 80)
    print("CREATING TOP ANTAGONISTIC PROTEINS BAR PLOT")
    print("=" * 80)

    top20 = df_antagonistic.nlargest(20, 'Antagonism_Magnitude').copy()
    top20['Label'] = top20['Gene_Symbol'] + '\n(' + top20['Compartment_A'] + ' vs ' + top20['Compartment_B'] + ')'

    fig, ax = plt.subplots(figsize=(12, 10))

    bars = ax.barh(range(len(top20)), top20['Antagonism_Magnitude'],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top20))))

    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['Label'].tolist(), fontsize=9)
    ax.set_xlabel('Antagonism Magnitude (|Δz_A - Δz_B|)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Antagonistic Protein-Compartment Pairs\n(Ranked by Magnitude of Opposite Aging Trajectories)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top20['Antagonism_Magnitude'])):
        ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Bar plot saved: {output_path}")

def create_violin_plot(df, output_path):
    """Create violin plot of Δz distribution by compartment load type."""
    print("\n" + "=" * 80)
    print("CREATING VIOLIN PLOT: Load-Dependent Distribution")
    print("=" * 80)

    plot_data = df[(df['Mechanical_Stress'].notna()) &
                   (df['Protein_Type'].isin(['Structural', 'Regulatory']))].copy()

    # Categorize mechanical stress
    plot_data['Load_Category'] = plot_data['Mechanical_Stress'].map({
        0.0: 'Low-Load',
        0.5: 'Mixed-Load',
        1.0: 'High-Load'
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.violinplot(data=plot_data, x='Load_Category', y='Zscore_Delta',
                   hue='Protein_Type', split=True, inner='box', palette='Set2', ax=ax)

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Compartment Load Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Zscore_Delta (aging trajectory)', fontsize=12, fontweight='bold')
    ax.set_title('Aging Trajectory Distribution by Mechanical Load and Protein Type\n(Split violins: Structural vs Regulatory)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Protein Type', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Violin plot saved: {output_path}")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("COMPARTMENT ANTAGONISTIC MECHANICAL STRESS ANALYSIS")
    print("Agent: claude_code")
    print("=" * 80)

    # Load and preprocess data
    df = load_and_validate_data()
    df = preprocess_data(df)

    # Identify antagonistic pairs
    df_antagonistic = identify_antagonistic_pairs(df)

    # Save antagonistic pairs CSV
    antagonistic_csv = f'{OUTPUT_DIR}/antagonistic_pairs_claude_code.csv'
    df_antagonistic.to_csv(antagonistic_csv, index=False)
    print(f"\n✓ Antagonistic pairs saved: {antagonistic_csv}")

    # Statistical testing
    mann_whitney_results = test_mechanical_stress_hypothesis(df, df_antagonistic)
    correlation_results = test_mechanical_stress_correlation(df)

    # Combine statistical results
    stats_summary = pd.DataFrame([mann_whitney_results])
    stats_combined = pd.concat([
        stats_summary.assign(Analysis='Mann-Whitney U Test'),
        correlation_results.assign(Analysis='Spearman Correlation')
    ], ignore_index=True)

    # Save correlation statistics CSV
    correlation_csv = f'{OUTPUT_DIR}/mechanical_stress_correlation_claude_code.csv'
    stats_combined.to_csv(correlation_csv, index=False)
    print(f"\n✓ Correlation statistics saved: {correlation_csv}")

    # Generate visualizations
    create_antagonism_heatmap(df, df_antagonistic, f'{VIZ_DIR}/heatmap_claude_code.png')
    create_scatter_plot(df, f'{VIZ_DIR}/scatter_mechanical_stress_claude_code.png')
    create_top_antagonistic_barplot(df_antagonistic, f'{VIZ_DIR}/top20_antagonistic_claude_code.png')
    create_violin_plot(df, f'{VIZ_DIR}/violin_load_distribution_claude_code.png')

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output files:")
    print(f"  - {antagonistic_csv}")
    print(f"  - {correlation_csv}")
    print(f"  - {VIZ_DIR}/")
    print("=" * 80 + "\n")

    return df, df_antagonistic, stats_combined

if __name__ == '__main__':
    df, df_antagonistic, stats = main()
