#!/usr/bin/env python3
"""
Claude Code Agent 01: Entropy Analysis V2 (Batch-Corrected Dataset)
Mission: Validate entropy-based aging theory on batch-corrected data
Compare with original agent_09 results to identify artifacts vs biology
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
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Output directory
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/01_entropy_multi_agent_after_batch_corection/claude_code_agent_01/'

# ============================================================================
# LOGGING SETUP
# ============================================================================

class Logger:
    """Simple logger to both console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(f'{OUTPUT_DIR}/execution.log')

print("="*80)
print("CLAUDE CODE AGENT 01: ENTROPY ANALYSIS V2")
print("Batch-Corrected Dataset Analysis")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# ENTROPY CALCULATION FUNCTIONS
# ============================================================================

def calculate_shannon_entropy(abundances):
    """
    Calculate Shannon entropy: H(X) = -Σ p(x) * log2(p(x))
    Higher entropy = more disorder/unpredictability
    """
    values = abundances[~np.isnan(abundances)]

    if len(values) == 0:
        return np.nan

    # Normalize to probabilities (shift to positive)
    values = values - values.min() + 1
    probabilities = values / values.sum()

    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return entropy


def calculate_variance_entropy_ratio(abundances):
    """
    Variance as entropy proxy: CV = std/mean
    Higher CV = higher disorder
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

    Returns:
    - consistency score (0-1): 1 = perfect consistency, 0 = random
    - direction: 'increase', 'decrease', or 'mixed'
    """
    valid_zscores = z_scores[~np.isnan(z_scores)]

    if len(valid_zscores) < 2:
        return np.nan, 'insufficient_data'

    # Direction consistency
    positive_count = (valid_zscores > 0).sum()
    negative_count = (valid_zscores < 0).sum()
    total = len(valid_zscores)

    # Consistency = max(pos, neg) / total
    consistency = max(positive_count, negative_count) / total

    # Direction
    if positive_count > negative_count:
        direction = 'increase'
    elif negative_count > positive_count:
        direction = 'decrease'
    else:
        direction = 'mixed'

    return consistency, direction


def calculate_entropy_transition_score(df_protein):
    """
    Detect proteins switching from ordered → disordered
    Compare CV in young vs old
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
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_validate_data(filepath):
    """Load batch-corrected dataset and validate quality"""
    print("\n" + "="*80)
    print("1. LOADING BATCH-CORRECTED DATASET")
    print("="*80)

    print(f"\nLoading: {filepath}")
    df = pd.read_csv(filepath)

    print(f"\n✅ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    print(f"  Unique studies: {df['Study_ID'].nunique()}")
    print(f"  Unique tissues: {df['Tissue'].nunique()}")

    # Data quality checks
    print(f"\n📊 Data Quality Checks:")
    print(f"  Missing values in Abundance_Old: {df['Abundance_Old'].isna().sum()} ({df['Abundance_Old'].isna().mean()*100:.1f}%)")
    print(f"  Missing values in Abundance_Young: {df['Abundance_Young'].isna().sum()} ({df['Abundance_Young'].isna().mean()*100:.1f}%)")
    print(f"  Missing values in Zscore_Delta: {df['Zscore_Delta'].isna().sum()} ({df['Zscore_Delta'].isna().mean()*100:.1f}%)")

    # Z-score distribution
    zscore_delta = df['Zscore_Delta'].dropna()
    print(f"\n  Zscore_Delta distribution:")
    print(f"    Mean: {zscore_delta.mean():.3f}")
    print(f"    Std: {zscore_delta.std():.3f}")
    print(f"    Median: {zscore_delta.median():.3f}")
    print(f"    Range: [{zscore_delta.min():.3f}, {zscore_delta.max():.3f}]")

    return df


# ============================================================================
# ENTROPY METRICS CALCULATION
# ============================================================================

def calculate_protein_entropy_metrics(df):
    """Calculate all 4 entropy metrics for each protein"""
    print("\n" + "="*80)
    print("2. CALCULATING ENTROPY METRICS")
    print("="*80)

    results = []
    proteins = df['Canonical_Gene_Symbol'].unique()

    print(f"\nProcessing {len(proteins)} unique proteins...")

    for i, protein in enumerate(proteins):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(proteins)} proteins processed")

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

        # Predictability from z-scores
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

    # Filter for proteins with sufficient data
    print(f"\n📊 Filtering proteins with ≥2 studies and valid entropy...")
    df_entropy = df_entropy[df_entropy['N_Studies'] >= 2].copy()
    df_entropy = df_entropy[~df_entropy['Shannon_Entropy'].isna()].copy()

    print(f"  ✅ {len(df_entropy)} proteins pass quality filter")

    # Summary statistics
    print(f"\n📈 Entropy Metrics Summary:")
    print(f"  Shannon Entropy: {df_entropy['Shannon_Entropy'].mean():.3f} ± {df_entropy['Shannon_Entropy'].std():.3f}")
    print(f"  Variance CV: {df_entropy['Variance_Entropy_CV'].mean():.3f} ± {df_entropy['Variance_Entropy_CV'].std():.3f}")
    print(f"  Predictability: {df_entropy['Predictability_Score'].mean():.3f} ± {df_entropy['Predictability_Score'].std():.3f}")
    print(f"  Entropy Transition: {df_entropy['Entropy_Transition'].mean():.3f} ± {df_entropy['Entropy_Transition'].std():.3f}")

    return df_entropy


# ============================================================================
# CLUSTERING ANALYSIS
# ============================================================================

def perform_entropy_clustering(df_entropy, n_clusters=4):
    """Hierarchical clustering on entropy profiles"""
    print("\n" + "="*80)
    print("3. HIERARCHICAL CLUSTERING ANALYSIS")
    print("="*80)

    print(f"\nPerforming clustering into {n_clusters} groups...")

    # Select features
    features = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']

    # Prepare data
    df_cluster = df_entropy[features].copy()
    df_cluster = df_cluster.fillna(df_cluster.median())

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='ward')

    # Assign clusters
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_entropy['Entropy_Cluster'] = clusters

    # Cluster characteristics
    print(f"\n✅ Clustering complete. Cluster profiles:")
    print("-" * 80)

    for cluster_id in range(1, n_clusters + 1):
        cluster_data = df_entropy[df_entropy['Entropy_Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id}: n={len(cluster_data)}")
        print(f"  Shannon Entropy: {cluster_data['Shannon_Entropy'].mean():.3f} ± {cluster_data['Shannon_Entropy'].std():.3f}")
        print(f"  Variance CV: {cluster_data['Variance_Entropy_CV'].mean():.3f} ± {cluster_data['Variance_Entropy_CV'].std():.3f}")
        print(f"  Predictability: {cluster_data['Predictability_Score'].mean():.3f} ± {cluster_data['Predictability_Score'].std():.3f}")
        print(f"  Entropy Transition: {cluster_data['Entropy_Transition'].mean():.3f} ± {cluster_data['Entropy_Transition'].std():.3f}")

    return df_entropy, linkage_matrix, X_scaled


# ============================================================================
# COMPARISON WITH ORIGINAL ANALYSIS
# ============================================================================

def compare_with_original(df_entropy_v2):
    """Compare V2 results with original agent_09 results"""
    print("\n" + "="*80)
    print("4. COMPARISON WITH ORIGINAL ANALYSIS (V1 vs V2)")
    print("="*80)

    # Load original results
    original_path = '/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_09_entropy/entropy_metrics.csv'
    print(f"\nLoading original results: {original_path}")

    try:
        df_entropy_v1 = pd.read_csv(original_path)
        print(f"  ✅ Loaded: {len(df_entropy_v1)} proteins from V1")
    except Exception as e:
        print(f"  ⚠️ Could not load original results: {e}")
        print(f"  Skipping comparison analysis.")
        return None

    # Check which columns are available in V1
    v1_cols = ['Protein', 'Shannon_Entropy', 'Variance_Entropy_CV',
               'Predictability_Score', 'Entropy_Transition']
    if 'Entropy_Cluster' in df_entropy_v1.columns:
        v1_cols.append('Entropy_Cluster')

    v2_cols = ['Protein', 'Shannon_Entropy', 'Variance_Entropy_CV',
               'Predictability_Score', 'Entropy_Transition', 'Entropy_Cluster']

    # Merge datasets on protein name
    df_merged = pd.merge(
        df_entropy_v1[v1_cols],
        df_entropy_v2[v2_cols],
        on='Protein',
        suffixes=('_V1', '_V2'),
        how='inner'
    )

    print(f"\n📊 Overlap: {len(df_merged)} proteins present in both V1 and V2")

    # Ranking correlations
    print(f"\n📈 Ranking Correlations (Spearman):")
    print("-" * 80)

    metrics = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']
    correlations = {}

    for metric in metrics:
        v1_col = f'{metric}_V1'
        v2_col = f'{metric}_V2'

        # Remove NaN pairs
        valid_mask = ~(df_merged[v1_col].isna() | df_merged[v2_col].isna())
        valid_data = df_merged[valid_mask]

        if len(valid_data) > 10:
            corr, pval = stats.spearmanr(valid_data[v1_col], valid_data[v2_col])
            correlations[metric] = corr
            print(f"  {metric}: r = {corr:.3f} (p < {pval:.2e}, n = {len(valid_data)})")
        else:
            correlations[metric] = np.nan
            print(f"  {metric}: insufficient data")

    # Cluster stability (only if V1 has cluster assignments)
    print(f"\n🔄 Cluster Stability:")
    print("-" * 80)

    if 'Entropy_Cluster_V1' in df_merged.columns and 'Entropy_Cluster_V2' in df_merged.columns:
        valid_clusters = ~(df_merged['Entropy_Cluster_V1'].isna() | df_merged['Entropy_Cluster_V2'].isna())
        if valid_clusters.sum() > 10:
            ari = adjusted_rand_score(
                df_merged.loc[valid_clusters, 'Entropy_Cluster_V1'],
                df_merged.loc[valid_clusters, 'Entropy_Cluster_V2']
            )
            print(f"  Adjusted Rand Index: {ari:.3f}")

            if ari > 0.7:
                print(f"  ✅ High stability (ARI > 0.7): batch correction preserves clusters")
            elif ari > 0.4:
                print(f"  ⚠️ Moderate stability (0.4 < ARI < 0.7): some cluster changes")
            else:
                print(f"  ❌ Low stability (ARI < 0.4): major cluster reorganization")
        else:
            print(f"  ⚠️ Insufficient overlap for cluster stability analysis")
    else:
        print(f"  ⚠️ V1 dataset does not have cluster assignments - skipping cluster stability test")

    # Identify proteins with large changes
    print(f"\n🔍 Proteins with Large Entropy Changes (potential artifacts):")
    print("-" * 80)

    df_merged['Shannon_Delta'] = df_merged['Shannon_Entropy_V2'] - df_merged['Shannon_Entropy_V1']
    df_merged['Predictability_Delta'] = df_merged['Predictability_Score_V2'] - df_merged['Predictability_Score_V1']

    large_entropy_change = df_merged.nlargest(10, 'Shannon_Delta', keep='all')
    print(f"\n  Top 10 proteins with INCREASED Shannon entropy:")
    for idx, row in large_entropy_change.head(10).iterrows():
        print(f"    {row['Protein']}: {row['Shannon_Entropy_V1']:.3f} → {row['Shannon_Entropy_V2']:.3f} (Δ = +{row['Shannon_Delta']:.3f})")

    large_entropy_decrease = df_merged.nsmallest(10, 'Shannon_Delta', keep='all')
    print(f"\n  Top 10 proteins with DECREASED Shannon entropy:")
    for idx, row in large_entropy_decrease.head(10).iterrows():
        print(f"    {row['Protein']}: {row['Shannon_Entropy_V1']:.3f} → {row['Shannon_Entropy_V2']:.3f} (Δ = {row['Shannon_Delta']:.3f})")

    return df_merged, correlations


# ============================================================================
# DEATH THEOREM TESTING
# ============================================================================

def death_theorem_analysis(df_entropy):
    """Test DEATh theorem predictions"""
    print("\n" + "="*80)
    print("5. DEATh THEOREM TESTING")
    print("="*80)

    # Test 1: Core vs Associated
    print("\n📋 Test 1: Structural vs Regulatory Proteins")
    print("-" * 80)

    structural = df_entropy[df_entropy['Matrisome_Division'] == 'Core matrisome'].copy()
    regulatory = df_entropy[df_entropy['Matrisome_Division'] == 'Matrisome-associated'].copy()

    if len(structural) > 0 and len(regulatory) > 0:
        print(f"\n  Core matrisome (structural): n = {len(structural)}")
        print(f"    Shannon Entropy: {structural['Shannon_Entropy'].mean():.3f} ± {structural['Shannon_Entropy'].std():.3f}")
        print(f"    Predictability: {structural['Predictability_Score'].mean():.3f} ± {structural['Predictability_Score'].std():.3f}")

        print(f"\n  Matrisome-associated (regulatory): n = {regulatory}")
        print(f"    Shannon Entropy: {regulatory['Shannon_Entropy'].mean():.3f} ± {regulatory['Shannon_Entropy'].std():.3f}")
        print(f"    Predictability: {regulatory['Predictability_Score'].mean():.3f} ± {regulatory['Predictability_Score'].std():.3f}")

        # Statistical tests
        entropy_pval = stats.mannwhitneyu(
            structural['Shannon_Entropy'].dropna(),
            regulatory['Shannon_Entropy'].dropna(),
            alternative='two-sided'
        ).pvalue

        pred_pval = stats.mannwhitneyu(
            structural['Predictability_Score'].dropna(),
            regulatory['Predictability_Score'].dropna(),
            alternative='two-sided'
        ).pvalue

        print(f"\n  📊 Statistical Tests (Mann-Whitney U):")
        print(f"    Shannon Entropy p-value: {entropy_pval:.4f} {'✅ SIGNIFICANT' if entropy_pval < 0.05 else '❌ NOT SIGNIFICANT'}")
        print(f"    Predictability p-value: {pred_pval:.4f} {'✅ SIGNIFICANT' if pred_pval < 0.05 else '❌ NOT SIGNIFICANT'}")

    # Test 2: Collagen predictability
    print("\n📋 Test 2: Collagen Determinism")
    print("-" * 80)

    collagens = df_entropy[df_entropy['Protein'].str.startswith('COL')].copy()

    if len(collagens) > 5:
        mean_pred_all = df_entropy['Predictability_Score'].mean()
        mean_pred_col = collagens['Predictability_Score'].mean()

        print(f"\n  Collagens: n = {len(collagens)}")
        print(f"    Mean Predictability: {mean_pred_col:.3f}")
        print(f"    Overall Mean Predictability: {mean_pred_all:.3f}")
        print(f"    Difference: {((mean_pred_col - mean_pred_all) / mean_pred_all * 100):.1f}% {'HIGHER' if mean_pred_col > mean_pred_all else 'LOWER'}")

        print(f"\n  Aging Direction Distribution:")
        direction_counts = collagens['Aging_Direction'].value_counts()
        for direction, count in direction_counts.items():
            print(f"    {direction}: {count} ({count/len(collagens)*100:.1f}%)")

        if mean_pred_col > mean_pred_all:
            print(f"\n  ✅ Collagens show HIGHER predictability (supports DEATh theorem)")
            print(f"     → Collagen aging is deterministic (crosslinking)")
        else:
            print(f"\n  ❌ Collagens do NOT show higher predictability (contradicts DEATh)")

    # Test 3: High transition proteins
    print("\n📋 Test 3: Entropy Transition Proteins")
    print("-" * 80)

    high_transition = df_entropy.nlargest(10, 'Entropy_Transition')
    print(f"\n  Top 10 proteins with highest entropy transitions:")
    for idx, row in high_transition.iterrows():
        print(f"    {row['Protein']}: transition = {row['Entropy_Transition']:.3f} ({row['Matrisome_Category']})")

    return structural, regulatory, collagens


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df_entropy, linkage_matrix, X_scaled, df_comparison=None):
    """Generate all required visualizations"""
    print("\n" + "="*80)
    print("6. GENERATING VISUALIZATIONS")
    print("="*80)

    # Figure 1: Entropy distributions
    print("\n📊 Creating entropy distributions plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Shannon entropy
    axes[0, 0].hist(df_entropy['Shannon_Entropy'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(df_entropy['Shannon_Entropy'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    axes[0, 0].set_xlabel('Shannon Entropy', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Shannon Entropy Distribution (V2)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Variance CV
    axes[0, 1].hist(df_entropy['Variance_Entropy_CV'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='coral')
    axes[0, 1].axvline(df_entropy['Variance_Entropy_CV'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    axes[0, 1].set_xlabel('Coefficient of Variation', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Variance Entropy (CV) Distribution (V2)', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Predictability
    axes[1, 0].hist(df_entropy['Predictability_Score'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='mediumseagreen')
    axes[1, 0].axvline(df_entropy['Predictability_Score'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    axes[1, 0].set_xlabel('Predictability Score (0-1)', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Aging Predictability Distribution (V2)', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Entropy transition
    axes[1, 1].hist(df_entropy['Entropy_Transition'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='mediumpurple')
    axes[1, 1].axvline(df_entropy['Entropy_Transition'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    axes[1, 1].set_xlabel('Entropy Transition Score', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('Entropy Transition (Young→Old) Distribution (V2)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/entropy_distributions_v2.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: entropy_distributions_v2.png")
    plt.close()

    # Figure 2: Clustering dendrogram + heatmap
    print("\n📊 Creating clustering visualization...")
    fig = plt.figure(figsize=(16, 10))

    # Dendrogram
    ax1 = plt.subplot(2, 1, 1)
    dendrogram(linkage_matrix, ax=ax1, no_labels=True, color_threshold=0)
    ax1.set_xlabel('Protein Index', fontsize=11)
    ax1.set_ylabel('Distance', fontsize=11)
    ax1.set_title('Hierarchical Clustering Dendrogram (V2)', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Heatmap of cluster profiles
    ax2 = plt.subplot(2, 1, 2)
    n_clusters = df_entropy['Entropy_Cluster'].nunique()
    cluster_profiles = []
    cluster_labels = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_data = df_entropy[df_entropy['Entropy_Cluster'] == cluster_id]
        cluster_profiles.append([
            cluster_data['Shannon_Entropy'].mean(),
            cluster_data['Variance_Entropy_CV'].mean(),
            cluster_data['Predictability_Score'].mean(),
            cluster_data['Entropy_Transition'].mean()
        ])
        cluster_labels.append(f'Cluster {cluster_id}\n(n={len(cluster_data)})')

    cluster_profiles = np.array(cluster_profiles)
    # Normalize for heatmap
    cluster_profiles_norm = (cluster_profiles - cluster_profiles.min(axis=0)) / (cluster_profiles.max(axis=0) - cluster_profiles.min(axis=0) + 1e-10)

    sns.heatmap(cluster_profiles_norm.T, annot=cluster_profiles, fmt='.3f',
                xticklabels=cluster_labels,
                yticklabels=['Shannon Entropy', 'Variance CV', 'Predictability', 'Entropy Transition'],
                cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Normalized Score'})
    ax2.set_title('Cluster Profiles Heatmap (V2)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/entropy_clustering_v2.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: entropy_clustering_v2.png")
    plt.close()

    # Figure 3: Entropy-Predictability space
    print("\n📊 Creating entropy-predictability space plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        df_entropy['Shannon_Entropy'],
        df_entropy['Predictability_Score'],
        c=df_entropy['Entropy_Cluster'],
        cmap='viridis',
        s=60,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    ax.set_xlabel('Shannon Entropy (Disorder)', fontsize=13)
    ax.set_ylabel('Predictability Score (Determinism)', fontsize=13)
    ax.set_title('Entropy-Predictability Space: DEATh Framework (V2)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Quadrant lines
    x_mid = df_entropy['Shannon_Entropy'].median()
    y_mid = df_entropy['Predictability_Score'].median()

    ax.axvline(x_mid, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.axhline(y_mid, color='red', linestyle='--', alpha=0.5, linewidth=2)

    # Quadrant labels
    x_range = df_entropy['Shannon_Entropy'].max() - df_entropy['Shannon_Entropy'].min()
    y_range = df_entropy['Predictability_Score'].max() - df_entropy['Predictability_Score'].min()

    ax.text(x_mid - x_range*0.2, y_mid + y_range*0.15, 'Deterministic\nStructural',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.text(x_mid + x_range*0.2, y_mid + y_range*0.15, 'Regulated\nChaos',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(x_mid - x_range*0.2, y_mid - y_range*0.15, 'Context\nDependent',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(x_mid + x_range*0.2, y_mid - y_range*0.15, 'Dysregulated',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))

    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.savefig(f'{OUTPUT_DIR}/entropy_predictability_space_v2.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: entropy_predictability_space_v2.png")
    plt.close()

    # Figure 4: V1 vs V2 comparison (if available)
    if df_comparison is not None:
        print("\n📊 Creating V1 vs V2 comparison plots...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        metrics = [
            ('Shannon_Entropy', 'Shannon Entropy'),
            ('Variance_Entropy_CV', 'Variance CV'),
            ('Predictability_Score', 'Predictability'),
            ('Entropy_Transition', 'Entropy Transition')
        ]

        for idx, (metric, label) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            v1_col = f'{metric}_V1'
            v2_col = f'{metric}_V2'

            # Remove NaN
            valid_mask = ~(df_comparison[v1_col].isna() | df_comparison[v2_col].isna())
            valid_data = df_comparison[valid_mask]

            if len(valid_data) > 10:
                ax.scatter(valid_data[v1_col], valid_data[v2_col], alpha=0.5, s=30, edgecolors='black', linewidth=0.3)

                # Add diagonal line
                min_val = min(valid_data[v1_col].min(), valid_data[v2_col].min())
                max_val = max(valid_data[v1_col].max(), valid_data[v2_col].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='y=x')

                # Calculate correlation
                corr, pval = stats.spearmanr(valid_data[v1_col], valid_data[v2_col])

                ax.set_xlabel(f'{label} (V1 - Original)', fontsize=11)
                ax.set_ylabel(f'{label} (V2 - Batch-Corrected)', fontsize=11)
                ax.set_title(f'{label} Comparison\nr = {corr:.3f}, p < {pval:.2e}', fontsize=12, fontweight='bold')
                ax.grid(alpha=0.3)
                ax.legend()

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/entropy_comparison_v1_v2.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: entropy_comparison_v1_v2.png")
        plt.close()

    # Figure 5: DEATh theorem tests
    print("\n📊 Creating DEATh theorem visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Core vs Associated entropy
    structural = df_entropy[df_entropy['Matrisome_Division'] == 'Core matrisome']
    regulatory = df_entropy[df_entropy['Matrisome_Division'] == 'Matrisome-associated']

    if len(structural) > 0 and len(regulatory) > 0:
        data_entropy = [structural['Shannon_Entropy'].dropna(), regulatory['Shannon_Entropy'].dropna()]
        bp1 = axes[0].boxplot(data_entropy, labels=['Core\nMatrisome', 'Matrisome\nAssociated'],
                               patch_artist=True, widths=0.6)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_ylabel('Shannon Entropy', fontsize=11)
        axes[0].set_title('Structural vs Regulatory Entropy', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')

        # Add p-value
        pval = stats.mannwhitneyu(data_entropy[0], data_entropy[1], alternative='two-sided').pvalue
        axes[0].text(0.5, 0.95, f'p = {pval:.4f}', transform=axes[0].transAxes,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='yellow' if pval < 0.05 else 'white', alpha=0.7))

    # Core vs Associated predictability
    if len(structural) > 0 and len(regulatory) > 0:
        data_pred = [structural['Predictability_Score'].dropna(), regulatory['Predictability_Score'].dropna()]
        bp2 = axes[1].boxplot(data_pred, labels=['Core\nMatrisome', 'Matrisome\nAssociated'],
                               patch_artist=True, widths=0.6)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        axes[1].set_ylabel('Predictability Score', fontsize=11)
        axes[1].set_title('Structural vs Regulatory Predictability', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')

        # Add p-value
        pval = stats.mannwhitneyu(data_pred[0], data_pred[1], alternative='two-sided').pvalue
        axes[1].text(0.5, 0.95, f'p = {pval:.4f}', transform=axes[1].transAxes,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='yellow' if pval < 0.05 else 'white', alpha=0.7))

    # Collagen predictability
    collagens = df_entropy[df_entropy['Protein'].str.startswith('COL')]
    non_collagens = df_entropy[~df_entropy['Protein'].str.startswith('COL')]

    if len(collagens) > 5:
        data_col = [collagens['Predictability_Score'].dropna(), non_collagens['Predictability_Score'].dropna()]
        bp3 = axes[2].boxplot(data_col, labels=['Collagens', 'Non-Collagens'],
                               patch_artist=True, widths=0.6)
        bp3['boxes'][0].set_facecolor('gold')
        bp3['boxes'][1].set_facecolor('lightgray')
        axes[2].set_ylabel('Predictability Score', fontsize=11)
        axes[2].set_title('Collagen Determinism Test', fontsize=12, fontweight='bold')
        axes[2].grid(alpha=0.3, axis='y')

        # Add mean difference
        mean_col = data_col[0].mean()
        mean_non = data_col[1].mean()
        diff_pct = ((mean_col - mean_non) / mean_non * 100)
        axes[2].text(0.5, 0.95, f'{diff_pct:+.1f}% difference', transform=axes[2].transAxes,
                     ha='center', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='yellow' if diff_pct > 0 else 'white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/death_theorem_v2.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: death_theorem_v2.png")
    plt.close()

    print("\n✅ All visualizations generated successfully!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline"""

    # Load batch-corrected data
    filepath = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
    df = load_and_validate_data(filepath)

    # Calculate entropy metrics
    df_entropy = calculate_protein_entropy_metrics(df)

    # Save entropy metrics
    output_file = f'{OUTPUT_DIR}/entropy_metrics_v2.csv'
    df_entropy.to_csv(output_file, index=False)
    print(f"\n💾 Saved entropy metrics: {output_file}")

    # Perform clustering
    df_entropy, linkage_matrix, X_scaled = perform_entropy_clustering(df_entropy, n_clusters=4)

    # Update CSV with cluster assignments
    df_entropy.to_csv(output_file, index=False)
    print(f"💾 Updated with cluster assignments: {output_file}")

    # Compare with original
    df_comparison, correlations = compare_with_original(df_entropy)

    # DEATh theorem analysis
    structural, regulatory, collagens = death_theorem_analysis(df_entropy)

    # Create visualizations
    create_visualizations(df_entropy, linkage_matrix, X_scaled, df_comparison)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n✅ Total proteins analyzed: {len(df_entropy)}")
    print(f"✅ Files generated:")
    print(f"   - entropy_metrics_v2.csv")
    print(f"   - entropy_distributions_v2.png")
    print(f"   - entropy_clustering_v2.png")
    print(f"   - entropy_predictability_space_v2.png")
    if df_comparison is not None:
        print(f"   - entropy_comparison_v1_v2.png")
    print(f"   - death_theorem_v2.png")
    print(f"   - execution.log")

    print(f"\n📊 Key Findings Preview:")
    print(f"   - Mean Shannon Entropy: {df_entropy['Shannon_Entropy'].mean():.3f}")
    print(f"   - Mean Predictability: {df_entropy['Predictability_Score'].mean():.3f}")
    print(f"   - Collagen Predictability: {collagens['Predictability_Score'].mean():.3f}" if len(collagens) > 0 else "   - Collagen Predictability: N/A")

    if df_comparison is not None and correlations:
        print(f"\n🔄 V1 vs V2 Correlations:")
        for metric, corr in correlations.items():
            if not np.isnan(corr):
                print(f"   - {metric}: r = {corr:.3f}")

    print(f"\n⏱️ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
