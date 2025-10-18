#!/usr/bin/env python3
"""
Claude Code Agent 02: Entropy Analysis V2 (Batch-Corrected Dataset)
Mission: Re-analyze entropy metrics post-batch correction and compare with V1
Framework: DEATh theorem validation
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/01_entropy_multi_agent_after_batch_corection/claude_code_agent_02/'

# ============================================================================
# ENTROPY CALCULATION FUNCTIONS (identical to agent_09)
# ============================================================================

def calculate_shannon_entropy(abundances):
    """
    Calculate Shannon entropy: H(X) = -Σ p(x) * log2(p(x))
    Higher entropy = more disorder/unpredictability
    """
    values = abundances[~np.isnan(abundances)]
    if len(values) == 0:
        return np.nan

    # Normalize to probabilities
    values = values - values.min() + 1  # Shift to positive
    probabilities = values / values.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return entropy


def calculate_variance_entropy_ratio(abundances):
    """
    Variance entropy proxy: CV = std/mean
    High CV = high disorder
    """
    values = abundances[~np.isnan(abundances)]
    if len(values) < 2:
        return np.nan

    mean_val = values.mean()
    if mean_val == 0:
        return np.nan

    cv = values.std() / abs(mean_val)
    return cv


def calculate_predictability_score(z_scores):
    """
    Predictability: aging direction consistency
    Returns: (consistency score 0-1, direction)
    """
    valid_zscores = z_scores[~np.isnan(z_scores)]
    if len(valid_zscores) < 2:
        return np.nan, 'insufficient_data'

    positive_count = (valid_zscores > 0).sum()
    negative_count = (valid_zscores < 0).sum()
    total = len(valid_zscores)

    consistency = max(positive_count, negative_count) / total

    if positive_count > negative_count:
        direction = 'increase'
    elif negative_count > positive_count:
        direction = 'decrease'
    else:
        direction = 'mixed'

    return consistency, direction


def calculate_entropy_transition_score(df_protein):
    """
    Detect young→old variability changes
    High transition = regime shift
    """
    old_data = df_protein[df_protein['Abundance_Old'].notna()]
    young_data = df_protein[df_protein['Abundance_Young'].notna()]

    if len(old_data) < 2 or len(young_data) < 2:
        return np.nan

    cv_old = calculate_variance_entropy_ratio(old_data['Abundance_Old'].values)
    cv_young = calculate_variance_entropy_ratio(young_data['Abundance_Young'].values)

    if np.isnan(cv_old) or np.isnan(cv_young):
        return np.nan

    transition = abs(cv_old - cv_young)
    return transition


# ============================================================================
# DATA LOADING & VALIDATION
# ============================================================================

def load_and_validate_data(filepath):
    """Load V2 dataset and perform quality checks"""
    print("="*80)
    print("LOADING & VALIDATING V2 DATASET (BATCH-CORRECTED)")
    print("="*80)

    df = pd.read_csv(filepath)

    print(f"\n📊 Dataset Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    print(f"  Unique studies: {df['Study_ID'].nunique()}")
    print(f"  Unique tissues: {df['Tissue'].nunique()}")

    # Data quality validation
    print(f"\n🔍 Data Quality Checks:")
    print(f"  NaN in Zscore_Delta: {df['Zscore_Delta'].isna().sum()} ({df['Zscore_Delta'].isna().sum()/len(df)*100:.1f}%)")
    print(f"  NaN in Abundance_Old: {df['Abundance_Old'].isna().sum()} ({df['Abundance_Old'].isna().sum()/len(df)*100:.1f}%)")
    print(f"  NaN in Abundance_Young: {df['Abundance_Young'].isna().sum()} ({df['Abundance_Young'].isna().sum()/len(df)*100:.1f}%)")

    # Z-score distribution validation
    valid_zscores = df['Zscore_Delta'].dropna()
    print(f"\n📈 Z-score Distribution:")
    print(f"  Mean: {valid_zscores.mean():.3f}")
    print(f"  Std: {valid_zscores.std():.3f}")
    print(f"  Median: {valid_zscores.median():.3f}")
    print(f"  Range: [{valid_zscores.min():.2f}, {valid_zscores.max():.2f}]")

    return df


def calculate_protein_entropy_metrics(df):
    """Calculate all 4 entropy metrics for each protein in V2"""
    print("\n"+"="*80)
    print("CALCULATING ENTROPY METRICS V2")
    print("="*80)

    results = []
    proteins = df['Canonical_Gene_Symbol'].unique()

    print(f"\nProcessing {len(proteins)} proteins...")

    for i, protein in enumerate(proteins):
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(proteins)}")

        df_prot = df[df['Canonical_Gene_Symbol'] == protein].copy()

        # Basic info
        n_studies = df_prot['Study_ID'].nunique()
        n_tissues = df_prot['Tissue'].nunique()
        matrisome_category = df_prot['Matrisome_Category'].mode()[0] if len(df_prot) > 0 else 'Unknown'
        matrisome_division = df_prot['Matrisome_Division'].mode()[0] if len(df_prot) > 0 else 'Unknown'

        # Collect all abundance values
        all_abundances = pd.concat([
            df_prot['Abundance_Old'].dropna(),
            df_prot['Abundance_Young'].dropna()
        ])

        # Entropy metrics
        shannon_entropy = calculate_shannon_entropy(all_abundances.values)
        variance_entropy = calculate_variance_entropy_ratio(all_abundances.values)

        # Predictability
        z_delta = df_prot['Zscore_Delta'].dropna().values
        predictability, direction = calculate_predictability_score(z_delta)

        # Transition score
        transition = calculate_entropy_transition_score(df_prot)

        # Mean z-score delta
        mean_z_delta = z_delta.mean() if len(z_delta) > 0 else np.nan

        results.append({
            'Protein': protein,
            'N_Studies': n_studies,
            'N_Tissues': n_tissues,
            'Matrisome_Category': matrisome_category,
            'Matrisome_Division': matrisome_division,
            'Shannon_Entropy': shannon_entropy,
            'Variance_Entropy_CV': variance_entropy,
            'Predictability_Score': predictability,
            'Aging_Direction': direction,
            'Entropy_Transition': transition,
            'Mean_Zscore_Delta': mean_z_delta,
            'N_Observations': len(df_prot)
        })

    df_entropy = pd.DataFrame(results)

    # Filter for multi-study proteins
    df_entropy = df_entropy[df_entropy['N_Studies'] >= 2].copy()
    df_entropy = df_entropy[~df_entropy['Shannon_Entropy'].isna()].copy()

    print(f"\n✅ Proteins with sufficient data (≥2 studies): {len(df_entropy)}")

    return df_entropy


# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

def perform_entropy_clustering(df_entropy, n_clusters=4):
    """Hierarchical clustering on 4 entropy features"""
    print("\n"+"="*80)
    print(f"HIERARCHICAL CLUSTERING (n={n_clusters})")
    print("="*80)

    features = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']

    # Prepare data
    df_cluster = df_entropy[features].copy()
    df_cluster = df_cluster.fillna(df_cluster.median())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_entropy['Entropy_Cluster'] = clusters

    # Cluster characteristics
    print("\n📊 Cluster Characteristics:")
    for cluster_id in range(1, n_clusters + 1):
        cluster_data = df_entropy[df_entropy['Entropy_Cluster'] == cluster_id]
        print(f"\n  Cluster {cluster_id}: n={len(cluster_data)}")
        print(f"    Shannon H: {cluster_data['Shannon_Entropy'].mean():.3f} ± {cluster_data['Shannon_Entropy'].std():.3f}")
        print(f"    Variance CV: {cluster_data['Variance_Entropy_CV'].mean():.3f} ± {cluster_data['Variance_Entropy_CV'].std():.3f}")
        print(f"    Predictability: {cluster_data['Predictability_Score'].mean():.3f} ± {cluster_data['Predictability_Score'].std():.3f}")
        print(f"    Transition: {cluster_data['Entropy_Transition'].mean():.3f} ± {cluster_data['Entropy_Transition'].std():.3f}")

    return df_entropy, linkage_matrix, X_scaled


# ============================================================================
# V1-V2 COMPARISON ANALYSIS
# ============================================================================

def compare_v1_v2_entropy(df_entropy_v2, v1_metrics_path):
    """Compare entropy metrics before and after batch correction"""
    print("\n"+"="*80)
    print("V1-V2 COMPARISON ANALYSIS")
    print("="*80)

    # Load V1 results
    df_entropy_v1 = pd.read_csv(v1_metrics_path)
    print(f"\n📂 Loaded V1 results: {len(df_entropy_v1)} proteins")

    # Merge on protein name
    df_merged = df_entropy_v1.merge(
        df_entropy_v2,
        on='Protein',
        how='inner',
        suffixes=('_V1', '_V2')
    )

    print(f"✅ Overlapping proteins: {len(df_merged)}")

    # Calculate correlations
    print("\n📈 Entropy Ranking Correlations (Spearman ρ):")

    metrics_to_compare = [
        ('Shannon_Entropy', 'Shannon Entropy'),
        ('Variance_Entropy_CV', 'Variance CV'),
        ('Predictability_Score', 'Predictability'),
        ('Entropy_Transition', 'Entropy Transition')
    ]

    correlations = {}
    for metric_col, metric_name in metrics_to_compare:
        v1_col = f"{metric_col}_V1"
        v2_col = f"{metric_col}_V2"

        valid_mask = (~df_merged[v1_col].isna()) & (~df_merged[v2_col].isna())
        if valid_mask.sum() > 10:
            rho, pval = stats.spearmanr(
                df_merged.loc[valid_mask, v1_col],
                df_merged.loc[valid_mask, v2_col]
            )
            correlations[metric_name] = (rho, pval)
            print(f"  {metric_name}: ρ={rho:.3f}, p={pval:.2e}")
        else:
            correlations[metric_name] = (np.nan, np.nan)
            print(f"  {metric_name}: insufficient data")

    # Cluster stability (Adjusted Rand Index)
    if 'Entropy_Cluster_V1' in df_merged.columns and 'Entropy_Cluster_V2' in df_merged.columns:
        ari = adjusted_rand_score(df_merged['Entropy_Cluster_V1'], df_merged['Entropy_Cluster_V2'])
        print(f"\n🎯 Cluster Stability (Adjusted Rand Index): {ari:.3f}")
        print("   Interpretation: 1.0=perfect agreement, 0.0=random, <0=worse than random")

    # Identify proteins with largest entropy changes
    df_merged['Delta_Shannon'] = df_merged['Shannon_Entropy_V2'] - df_merged['Shannon_Entropy_V1']
    df_merged['Delta_Predictability'] = df_merged['Predictability_Score_V2'] - df_merged['Predictability_Score_V1']

    print("\n🔍 Top 10 proteins with LARGEST Shannon entropy DECREASE (V2 < V1):")
    print("   These are likely ARTIFACTS removed by batch correction:")
    largest_decrease = df_merged.nsmallest(10, 'Delta_Shannon')[['Protein', 'Shannon_Entropy_V1', 'Shannon_Entropy_V2', 'Delta_Shannon']]
    for idx, row in largest_decrease.iterrows():
        print(f"     {row['Protein']}: {row['Shannon_Entropy_V1']:.3f} → {row['Shannon_Entropy_V2']:.3f} (Δ={row['Delta_Shannon']:.3f})")

    print("\n🔍 Top 10 proteins with LARGEST Shannon entropy INCREASE (V2 > V1):")
    largest_increase = df_merged.nlargest(10, 'Delta_Shannon')[['Protein', 'Shannon_Entropy_V1', 'Shannon_Entropy_V2', 'Delta_Shannon']]
    for idx, row in largest_increase.iterrows():
        print(f"     {row['Protein']}: {row['Shannon_Entropy_V1']:.3f} → {row['Shannon_Entropy_V2']:.3f} (Δ={row['Delta_Shannon']:.3f})")

    return df_merged, correlations


# ============================================================================
# DEATh THEOREM ANALYSIS
# ============================================================================

def death_theorem_analysis(df_entropy):
    """Test DEATh theorem predictions on V2 data"""
    print("\n"+"="*80)
    print("DEATh THEOREM VALIDATION (V2)")
    print("="*80)

    # Lemma 2: Structural vs Regulatory
    structural = df_entropy[df_entropy['Matrisome_Division'] == 'Core matrisome'].copy()
    regulatory = df_entropy[df_entropy['Matrisome_Division'] == 'Matrisome-associated'].copy()

    print(f"\n🧬 Lemma 2: Structural vs Regulatory Entropy")
    print(f"  Core matrisome (structural): n={len(structural)}")
    print(f"    Shannon H: {structural['Shannon_Entropy'].mean():.3f} ± {structural['Shannon_Entropy'].std():.3f}")
    print(f"    Predictability: {structural['Predictability_Score'].mean():.3f} ± {structural['Predictability_Score'].std():.3f}")

    print(f"\n  Matrisome-associated (regulatory): n={len(regulatory)}")
    print(f"    Shannon H: {regulatory['Shannon_Entropy'].mean():.3f} ± {regulatory['Shannon_Entropy'].std():.3f}")
    print(f"    Predictability: {regulatory['Predictability_Score'].mean():.3f} ± {regulatory['Predictability_Score'].std():.3f}")

    # Statistical tests
    if len(structural) > 0 and len(regulatory) > 0:
        entropy_stat, entropy_pval = stats.mannwhitneyu(
            structural['Shannon_Entropy'].dropna(),
            regulatory['Shannon_Entropy'].dropna(),
            alternative='two-sided'
        )

        predict_stat, predict_pval = stats.mannwhitneyu(
            structural['Predictability_Score'].dropna(),
            regulatory['Predictability_Score'].dropna(),
            alternative='two-sided'
        )

        print(f"\n  📊 Statistical Tests (Mann-Whitney U):")
        print(f"    Entropy difference: p={entropy_pval:.4f} {'***' if entropy_pval < 0.001 else '**' if entropy_pval < 0.01 else '*' if entropy_pval < 0.05 else 'NS'}")
        print(f"    Predictability difference: p={predict_pval:.4f} {'***' if predict_pval < 0.001 else '**' if predict_pval < 0.01 else '*' if predict_pval < 0.05 else 'NS'}")

    # Collagen analysis
    collagens = df_entropy[df_entropy['Protein'].str.startswith('COL')].copy()
    print(f"\n🧱 Collagen Analysis:")
    print(f"  n={len(collagens)} collagens")

    if len(collagens) > 5:
        col_predictability = collagens['Predictability_Score'].mean()
        all_predictability = df_entropy['Predictability_Score'].mean()

        print(f"  Mean predictability: {col_predictability:.3f}")
        print(f"  Overall mean: {all_predictability:.3f}")
        print(f"  Difference: {((col_predictability/all_predictability - 1)*100):.1f}% {'higher' if col_predictability > all_predictability else 'lower'}")

        print(f"\n  Aging direction distribution:")
        for direction, count in collagens['Aging_Direction'].value_counts().items():
            print(f"    {direction}: {count} ({count/len(collagens)*100:.1f}%)")

        if col_predictability > all_predictability:
            print(f"\n  ✅ SUPPORTS DEATh: Collagens are deterministic!")
        else:
            print(f"\n  ⚠️  WEAKENS DEATh: Collagens not more predictable")

    # Top transition proteins
    high_transition = df_entropy.nlargest(10, 'Entropy_Transition')
    print(f"\n🔄 Top 10 Entropy Transition Proteins (V2):")
    for idx, row in high_transition.iterrows():
        print(f"  {row['Protein']}: transition={row['Entropy_Transition']:.3f}, category={row['Matrisome_Category']}")

    return structural, regulatory, collagens, entropy_pval, predict_pval


# ============================================================================
# VISUALIZATION SUITE
# ============================================================================

def create_visualizations(df_entropy_v2, df_merged, linkage_matrix, X_scaled):
    """Generate all required publication-quality visualizations"""
    print("\n"+"="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Figure 1: Entropy distributions V2 with V1 overlay
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('Shannon_Entropy', 'Shannon Entropy', axes[0, 0]),
        ('Variance_Entropy_CV', 'Variance CV', axes[0, 1]),
        ('Predictability_Score', 'Predictability Score', axes[1, 0]),
        ('Entropy_Transition', 'Entropy Transition', axes[1, 1])
    ]

    for metric_col, metric_name, ax in metrics:
        # V2 histogram
        ax.hist(df_entropy_v2[metric_col].dropna(), bins=30, alpha=0.7,
                label='V2 (batch-corrected)', edgecolor='black', color='steelblue')

        # V1 overlay (dashed line)
        if f"{metric_col}_V1" in df_merged.columns:
            v1_vals = df_merged[f"{metric_col}_V1"].dropna()
            ax.hist(v1_vals, bins=30, alpha=0.3, histtype='step',
                    linewidth=2, linestyle='--', label='V1 (original)', color='red')

        ax.axvline(df_entropy_v2[metric_col].median(), color='green',
                   linestyle='-', linewidth=2, label='V2 Median')
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}entropy_distributions_v2.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: entropy_distributions_v2.png")
    plt.close()

    # Figure 2: Clustering dendrogram + heatmap
    fig = plt.figure(figsize=(16, 10))

    # Dendrogram
    ax1 = plt.subplot(2, 1, 1)
    dendrogram(linkage_matrix, ax=ax1, no_labels=True)
    ax1.set_xlabel('Protein Index', fontsize=11)
    ax1.set_ylabel('Distance', fontsize=11)
    ax1.set_title('Hierarchical Clustering Dendrogram (V2)', fontsize=12, fontweight='bold')

    # Cluster heatmap
    ax2 = plt.subplot(2, 1, 2)
    cluster_means = df_entropy_v2.groupby('Entropy_Cluster')[
        ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']
    ].mean()

    sns.heatmap(cluster_means.T, annot=True, fmt='.3f', cmap='RdYlGn',
                ax=ax2, cbar_kws={'label': 'Mean Value'})
    ax2.set_xlabel('Cluster ID', fontsize=11)
    ax2.set_ylabel('Entropy Metric', fontsize=11)
    ax2.set_title('Cluster Profiles (V2)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}entropy_clustering_v2.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: entropy_clustering_v2.png")
    plt.close()

    # Figure 3: Entropy-Predictability space
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        df_entropy_v2['Shannon_Entropy'],
        df_entropy_v2['Predictability_Score'],
        c=df_entropy_v2['Entropy_Cluster'],
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    ax.set_xlabel('Shannon Entropy (Disorder)', fontsize=12)
    ax.set_ylabel('Predictability Score (Determinism)', fontsize=12)
    ax.set_title('Entropy-Predictability Space V2 (Batch-Corrected)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Quadrant lines
    x_mid = df_entropy_v2['Shannon_Entropy'].median()
    y_mid = df_entropy_v2['Predictability_Score'].median()
    ax.axvline(x_mid, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y_mid, color='red', linestyle='--', alpha=0.5)

    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.savefig(f'{OUTPUT_DIR}entropy_predictability_space_v2.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: entropy_predictability_space_v2.png")
    plt.close()

    # Figure 4: V1-V2 Comparison (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Shannon entropy correlation
    valid_mask = (~df_merged['Shannon_Entropy_V1'].isna()) & (~df_merged['Shannon_Entropy_V2'].isna())
    if valid_mask.sum() > 10:
        axes[0].scatter(df_merged.loc[valid_mask, 'Shannon_Entropy_V1'],
                       df_merged.loc[valid_mask, 'Shannon_Entropy_V2'],
                       alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

        # Identity line
        min_val = min(df_merged.loc[valid_mask, 'Shannon_Entropy_V1'].min(),
                     df_merged.loc[valid_mask, 'Shannon_Entropy_V2'].min())
        max_val = max(df_merged.loc[valid_mask, 'Shannon_Entropy_V1'].max(),
                     df_merged.loc[valid_mask, 'Shannon_Entropy_V2'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Identity')

        rho, pval = stats.spearmanr(df_merged.loc[valid_mask, 'Shannon_Entropy_V1'],
                                     df_merged.loc[valid_mask, 'Shannon_Entropy_V2'])
        axes[0].text(0.05, 0.95, f'ρ={rho:.3f}\np={pval:.2e}',
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        axes[0].set_xlabel('Shannon Entropy V1', fontsize=11)
        axes[0].set_ylabel('Shannon Entropy V2', fontsize=11)
        axes[0].set_title('Shannon Entropy Correlation', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Panel B: Predictability correlation
    valid_mask = (~df_merged['Predictability_Score_V1'].isna()) & (~df_merged['Predictability_Score_V2'].isna())
    if valid_mask.sum() > 10:
        axes[1].scatter(df_merged.loc[valid_mask, 'Predictability_Score_V1'],
                       df_merged.loc[valid_mask, 'Predictability_Score_V2'],
                       alpha=0.5, s=30, edgecolors='black', linewidth=0.5, color='green')

        min_val = 0.3
        max_val = 1.0
        axes[1].plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Identity')

        rho, pval = stats.spearmanr(df_merged.loc[valid_mask, 'Predictability_Score_V1'],
                                     df_merged.loc[valid_mask, 'Predictability_Score_V2'])
        axes[1].text(0.05, 0.95, f'ρ={rho:.3f}\np={pval:.2e}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        axes[1].set_xlabel('Predictability V1', fontsize=11)
        axes[1].set_ylabel('Predictability V2', fontsize=11)
        axes[1].set_title('Predictability Correlation', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Panel C: Cluster stability matrix
    if 'Entropy_Cluster_V1' in df_merged.columns and 'Entropy_Cluster_V2' in df_merged.columns:
        overlap_matrix = pd.crosstab(df_merged['Entropy_Cluster_V1'],
                                      df_merged['Entropy_Cluster_V2'])

        sns.heatmap(overlap_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                   cbar_kws={'label': 'Protein Count'})
        axes[2].set_xlabel('Cluster ID V2', fontsize=11)
        axes[2].set_ylabel('Cluster ID V1', fontsize=11)
        axes[2].set_title('Cluster Stability Matrix', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}entropy_comparison_v1_v2.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: entropy_comparison_v1_v2.png")
    plt.close()

    # Figure 5: DEATh theorem tests
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Core vs Associated entropy
    structural = df_entropy_v2[df_entropy_v2['Matrisome_Division'] == 'Core matrisome']
    regulatory = df_entropy_v2[df_entropy_v2['Matrisome_Division'] == 'Matrisome-associated']

    data_to_plot = [structural['Shannon_Entropy'].dropna(),
                    regulatory['Shannon_Entropy'].dropna()]
    bp = axes[0].boxplot(data_to_plot, labels=['Core\nmatrisome', 'Matrisome-\nassociated'],
                         patch_artist=True, widths=0.6)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    axes[0].set_ylabel('Shannon Entropy', fontsize=11)
    axes[0].set_title('Structural vs Regulatory\nEntropy (V2)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel B: Collagen predictability
    collagens = df_entropy_v2[df_entropy_v2['Protein'].str.startswith('COL')]
    others = df_entropy_v2[~df_entropy_v2['Protein'].str.startswith('COL')]

    col_mean = collagens['Predictability_Score'].mean()
    col_std = collagens['Predictability_Score'].std()
    other_mean = others['Predictability_Score'].mean()
    other_std = others['Predictability_Score'].std()

    axes[1].bar(['Collagens', 'Other ECM'], [col_mean, other_mean],
               yerr=[col_std, other_std], capsize=10,
               color=['salmon', 'steelblue'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Predictability Score', fontsize=11)
    axes[1].set_title('Collagen Predictability\n(V2)', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].axhline(y=other_mean, color='red', linestyle='--', alpha=0.5, label='Overall mean')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Panel C: Top transition proteins
    high_transition = df_entropy_v2.nlargest(10, 'Entropy_Transition').sort_values('Entropy_Transition')

    axes[2].barh(range(len(high_transition)), high_transition['Entropy_Transition'].values,
                color='purple', alpha=0.7, edgecolor='black')
    axes[2].set_yticks(range(len(high_transition)))
    axes[2].set_yticklabels(high_transition['Protein'].values, fontsize=9)
    axes[2].set_xlabel('Entropy Transition Score', fontsize=11)
    axes[2].set_title('Top 10 Entropy\nTransition Proteins (V2)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}death_theorem_v2.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: death_theorem_v2.png")
    plt.close()

    print("\n✅ All visualizations complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete entropy analysis V2"""
    print("\n")
    print("="*80)
    print(" "*20 + "CLAUDE CODE AGENT 02")
    print(" "*15 + "ENTROPY ANALYSIS V2 (BATCH-CORRECTED)")
    print("="*80)
    print("\n")

    # 1. Load and validate V2 data
    v2_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
    df_v2 = load_and_validate_data(v2_path)

    # 2. Calculate entropy metrics
    df_entropy_v2 = calculate_protein_entropy_metrics(df_v2)

    # 3. Save V2 metrics
    v2_output_path = f'{OUTPUT_DIR}entropy_metrics_v2.csv'
    df_entropy_v2.to_csv(v2_output_path, index=False)
    print(f"\n💾 Saved V2 entropy metrics: {v2_output_path}")

    # 4. Perform clustering
    df_entropy_v2, linkage_matrix, X_scaled = perform_entropy_clustering(df_entropy_v2, n_clusters=4)

    # 5. Compare with V1
    v1_path = '/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_09_entropy/entropy_metrics.csv'
    df_merged, correlations = compare_v1_v2_entropy(df_entropy_v2, v1_path)

    # 6. DEATh theorem analysis
    structural, regulatory, collagens, entropy_pval, predict_pval = death_theorem_analysis(df_entropy_v2)

    # 7. Create visualizations
    create_visualizations(df_entropy_v2, df_merged, linkage_matrix, X_scaled)

    # 8. Summary statistics
    print("\n"+"="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\n📊 V2 Dataset:")
    print(f"  Total proteins analyzed: {len(df_entropy_v2)}")
    print(f"  Mean Shannon Entropy: {df_entropy_v2['Shannon_Entropy'].mean():.3f} ± {df_entropy_v2['Shannon_Entropy'].std():.3f}")
    print(f"  Mean Predictability: {df_entropy_v2['Predictability_Score'].mean():.3f} ± {df_entropy_v2['Predictability_Score'].std():.3f}")
    print(f"  Mean Variance CV: {df_entropy_v2['Variance_Entropy_CV'].mean():.3f} ± {df_entropy_v2['Variance_Entropy_CV'].std():.3f}")
    print(f"  Mean Transition: {df_entropy_v2['Entropy_Transition'].mean():.3f} ± {df_entropy_v2['Entropy_Transition'].std():.3f}")

    print("\n"+"="*80)
    print("ANALYSIS COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*80)

    return df_entropy_v2, df_merged


if __name__ == '__main__':
    df_entropy_v2, df_merged = main()
