#!/usr/bin/env python3
"""
Agent 09: Entropy-Based Clustering Expert
Mission: Analyze ECM aging through information theory and entropy lens
Framework: DEATh theorem - aging as entropy management problem
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# ENTROPY CALCULATION FUNCTIONS
# ============================================================================

def calculate_shannon_entropy(abundances):
    """
    Calculate Shannon entropy of abundance distribution
    H(X) = -Σ p(x) * log2(p(x))

    Higher entropy = more disorder/unpredictability
    Lower entropy = more ordered/deterministic
    """
    # Remove NaN values
    values = abundances[~np.isnan(abundances)]

    if len(values) == 0:
        return np.nan

    # Normalize to probabilities (ensure positive)
    values = values - values.min() + 1  # Shift to positive
    probabilities = values / values.sum()

    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return entropy


def calculate_variance_entropy_ratio(abundances):
    """
    Variance as entropy proxy: high variance = high disorder
    Returns coefficient of variation (CV = std/mean)
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
    Predictability: how consistent is aging direction across studies?

    High predictability (low entropy): protein always goes up or down
    Low predictability (high entropy): protein changes randomly

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
    Detect proteins that switch from ordered → disordered regulation

    Compare entropy in young vs old contexts
    High transition = switches behavior
    """
    # Split by age if possible
    old_data = df_protein[df_protein['Abundance_Old'].notna()]
    young_data = df_protein[df_protein['Abundance_Young'].notna()]

    if len(old_data) < 2 or len(young_data) < 2:
        return np.nan

    # Calculate CV for each
    cv_old = calculate_variance_entropy_ratio(old_data['Abundance_Old'].values)
    cv_young = calculate_variance_entropy_ratio(young_data['Abundance_Young'].values)

    if np.isnan(cv_old) or np.isnan(cv_young):
        return np.nan

    # Transition = change in variability
    transition = abs(cv_old - cv_young)

    return transition


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def load_and_prepare_data(filepath):
    """Load dataset and prepare for analysis"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")
    print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    print(f"Unique studies: {df['Study_ID'].nunique()}")
    print(f"Unique tissues: {df['Tissue'].nunique()}")

    return df


def calculate_protein_entropy_metrics(df):
    """Calculate all entropy metrics for each protein"""
    print("\nCalculating entropy metrics for each protein...")

    results = []

    for protein in df['Canonical_Gene_Symbol'].unique():
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

        # Mean z-score delta (aging effect size)
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
    df_entropy = df_entropy[df_entropy['N_Studies'] >= 2].copy()
    df_entropy = df_entropy[~df_entropy['Shannon_Entropy'].isna()].copy()

    print(f"Proteins with sufficient data: {len(df_entropy)}")

    return df_entropy


def perform_entropy_clustering(df_entropy, n_clusters=4):
    """Cluster proteins by entropy profiles"""
    print(f"\nPerforming hierarchical clustering into {n_clusters} groups...")

    # Select features for clustering
    features = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']

    # Prepare data
    df_cluster = df_entropy[features].copy()
    df_cluster = df_cluster.fillna(df_cluster.median())  # Impute missing values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method='ward')

    # Assign clusters
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_entropy['Entropy_Cluster'] = clusters

    # Calculate cluster characteristics
    print("\nCluster characteristics:")
    for cluster_id in range(1, n_clusters + 1):
        cluster_data = df_entropy[df_entropy['Entropy_Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_data)} proteins")
        print(f"  Mean Shannon Entropy: {cluster_data['Shannon_Entropy'].mean():.3f}")
        print(f"  Mean Variance CV: {cluster_data['Variance_Entropy_CV'].mean():.3f}")
        print(f"  Mean Predictability: {cluster_data['Predictability_Score'].mean():.3f}")
        print(f"  Mean Transition: {cluster_data['Entropy_Transition'].mean():.3f}")

    return df_entropy, linkage_matrix, X_scaled


def death_theorem_analysis(df_entropy):
    """Connect entropy patterns to DEATh theorem predictions"""
    print("\n" + "="*80)
    print("DEATh THEOREM ANALYSIS")
    print("="*80)

    # Lemma 2: Structural proteins (matrix stiffening, E↓) vs regulatory (chaos, C↑)
    structural_proteins = df_entropy[
        df_entropy['Matrisome_Division'] == 'Core matrisome'
    ].copy()

    regulatory_proteins = df_entropy[
        df_entropy['Matrisome_Division'] == 'Matrisome-associated'
    ].copy()

    print("\nLemma 2 Test: Structural (E↓) vs Regulatory (C↑)")
    print("-" * 60)

    if len(structural_proteins) > 0 and len(regulatory_proteins) > 0:
        print(f"\nStructural proteins (Core matrisome): n={len(structural_proteins)}")
        print(f"  Mean Shannon Entropy: {structural_proteins['Shannon_Entropy'].mean():.3f}")
        print(f"  Mean Variance CV: {structural_proteins['Variance_Entropy_CV'].mean():.3f}")
        print(f"  Mean Predictability: {structural_proteins['Predictability_Score'].mean():.3f}")

        print(f"\nRegulatory proteins (Matrisome-associated): n={len(regulatory_proteins)}")
        print(f"  Mean Shannon Entropy: {regulatory_proteins['Shannon_Entropy'].mean():.3f}")
        print(f"  Mean Variance CV: {regulatory_proteins['Variance_Entropy_CV'].mean():.3f}")
        print(f"  Mean Predictability: {regulatory_proteins['Predictability_Score'].mean():.3f}")

        # Statistical test
        entropy_pval = stats.mannwhitneyu(
            structural_proteins['Shannon_Entropy'].dropna(),
            regulatory_proteins['Shannon_Entropy'].dropna(),
            alternative='two-sided'
        ).pvalue

        predictability_pval = stats.mannwhitneyu(
            structural_proteins['Predictability_Score'].dropna(),
            regulatory_proteins['Predictability_Score'].dropna(),
            alternative='two-sided'
        ).pvalue

        print(f"\nStatistical tests:")
        print(f"  Entropy difference p-value: {entropy_pval:.4f}")
        print(f"  Predictability difference p-value: {predictability_pval:.4f}")

        if entropy_pval < 0.05:
            print("  → Significant entropy difference between structural and regulatory!")
        if predictability_pval < 0.05:
            print("  → Significant predictability difference!")

    # Test: Do collagens show DECREASED entropy (crosslinking)?
    collagens = df_entropy[df_entropy['Protein'].str.startswith('COL')].copy()
    if len(collagens) > 5:
        print(f"\nCollagens analysis: n={len(collagens)}")
        print(f"  Mean Predictability: {collagens['Predictability_Score'].mean():.3f}")
        print(f"  Aging Direction distribution:")
        print(collagens['Aging_Direction'].value_counts())

        # Are collagens more predictable than average?
        all_predictability = df_entropy['Predictability_Score'].mean()
        col_predictability = collagens['Predictability_Score'].mean()

        if col_predictability > all_predictability:
            print(f"  → Collagens are MORE predictable ({col_predictability:.3f} vs {all_predictability:.3f})")
            print("  → Supports DEATh: collagen aging is deterministic (crosslinking)")

    return structural_proteins, regulatory_proteins, collagens


def create_visualizations(df_entropy, linkage_matrix, X_scaled):
    """Create comprehensive visualization suite"""
    print("\nGenerating visualizations...")

    output_dir = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/'

    # Figure 1: Entropy distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Shannon entropy distribution
    axes[0, 0].hist(df_entropy['Shannon_Entropy'].dropna(), bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df_entropy['Shannon_Entropy'].median(), color='red', linestyle='--', label='Median')
    axes[0, 0].set_xlabel('Shannon Entropy')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Shannon Entropy Distribution')
    axes[0, 0].legend()

    # Variance CV distribution
    axes[0, 1].hist(df_entropy['Variance_Entropy_CV'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].axvline(df_entropy['Variance_Entropy_CV'].median(), color='red', linestyle='--', label='Median')
    axes[0, 1].set_xlabel('Coefficient of Variation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Variance Entropy (CV) Distribution')
    axes[0, 1].legend()

    # Predictability distribution
    axes[1, 0].hist(df_entropy['Predictability_Score'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].axvline(df_entropy['Predictability_Score'].median(), color='red', linestyle='--', label='Median')
    axes[1, 0].set_xlabel('Predictability Score (0-1)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Aging Predictability Distribution')
    axes[1, 0].legend()

    # Entropy transition
    axes[1, 1].hist(df_entropy['Entropy_Transition'].dropna(), bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].axvline(df_entropy['Entropy_Transition'].median(), color='red', linestyle='--', label='Median')
    axes[1, 1].set_xlabel('Entropy Transition Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Entropy Transition (Young→Old)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: entropy_distributions.png")
    plt.close()

    # Figure 2: Clustering dendrogram
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(linkage_matrix, ax=ax, no_labels=True)
    ax.set_xlabel('Protein Index')
    ax.set_ylabel('Distance')
    ax.set_title('Hierarchical Clustering Dendrogram (Entropy Profiles)')
    plt.savefig(f'{output_dir}/entropy_dendrogram.png', dpi=300, bbox_inches='tight')
    print(f"Saved: entropy_dendrogram.png")
    plt.close()

    # Figure 3: Cluster profiles
    n_clusters = df_entropy['Entropy_Cluster'].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    features = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']
    titles = ['Shannon Entropy', 'Variance CV', 'Predictability', 'Entropy Transition']

    for idx, (feature, title) in enumerate(zip(features, titles)):
        ax = axes[idx // 2, idx % 2]

        for cluster_id in range(1, n_clusters + 1):
            cluster_data = df_entropy[df_entropy['Entropy_Cluster'] == cluster_id]
            ax.hist(cluster_data[feature].dropna(), alpha=0.5, label=f'Cluster {cluster_id}', bins=20)

        ax.set_xlabel(title)
        ax.set_ylabel('Count')
        ax.set_title(f'{title} by Cluster')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_clusters_profiles.png', dpi=300, bbox_inches='tight')
    print(f"Saved: entropy_clusters_profiles.png")
    plt.close()

    # Figure 4: Entropy vs Predictability scatter
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        df_entropy['Shannon_Entropy'],
        df_entropy['Predictability_Score'],
        c=df_entropy['Entropy_Cluster'],
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    ax.set_xlabel('Shannon Entropy (Disorder)', fontsize=12)
    ax.set_ylabel('Predictability Score (Determinism)', fontsize=12)
    ax.set_title('Entropy-Predictability Space: DEATh Theorem Framework', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    x_mid = df_entropy['Shannon_Entropy'].median()
    y_mid = df_entropy['Predictability_Score'].median()

    ax.axvline(x_mid, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y_mid, color='red', linestyle='--', alpha=0.5)

    ax.text(x_mid * 0.5, 0.95, 'LOW ENTROPY\nHIGH PREDICTABILITY\n(Structural, Deterministic)',
            transform=ax.transData, ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(x_mid * 1.5, 0.95, 'HIGH ENTROPY\nHIGH PREDICTABILITY\n(Regulated Chaos)',
            transform=ax.transData, ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(x_mid * 0.5, 0.3, 'LOW ENTROPY\nLOW PREDICTABILITY\n(Context-Dependent)',
            transform=ax.transData, ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(x_mid * 1.5, 0.3, 'HIGH ENTROPY\nLOW PREDICTABILITY\n(Dysregulated)',
            transform=ax.transData, ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))

    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.savefig(f'{output_dir}/entropy_predictability_space.png', dpi=300, bbox_inches='tight')
    print(f"Saved: entropy_predictability_space.png")
    plt.close()

    # Figure 5: DEATh theorem comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Core vs Associated
    divisions = ['Core matrisome', 'Matrisome-associated']
    metrics = ['Shannon_Entropy', 'Predictability_Score']

    for idx, metric in enumerate(metrics):
        data_to_plot = []
        labels = []

        for division in divisions:
            subset = df_entropy[df_entropy['Matrisome_Division'] == division]
            data_to_plot.append(subset[metric].dropna())
            labels.append(division)

        bp = axes[idx].boxplot(data_to_plot, labels=labels, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        axes[idx].set_ylabel(metric.replace('_', ' '))
        axes[idx].set_title(f'{metric.replace("_", " ")} by Division')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/death_theorem_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: death_theorem_comparison.png")
    plt.close()


def main():
    """Main execution"""
    print("="*80)
    print("AGENT 09: ENTROPY-BASED CLUSTERING EXPERT")
    print("Mission: Analyze ECM aging through information theory lens")
    print("="*80)

    # Load data
    filepath = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
    df = load_and_prepare_data(filepath)

    # Calculate entropy metrics
    df_entropy = calculate_protein_entropy_metrics(df)

    # Save entropy metrics
    output_file = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/entropy_metrics.csv'
    df_entropy.to_csv(output_file, index=False)
    print(f"\nSaved entropy metrics to: {output_file}")

    # Perform clustering
    df_entropy, linkage_matrix, X_scaled = perform_entropy_clustering(df_entropy, n_clusters=4)

    # DEATh theorem analysis
    structural, regulatory, collagens = death_theorem_analysis(df_entropy)

    # Create visualizations
    create_visualizations(df_entropy, linkage_matrix, X_scaled)

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal proteins analyzed: {len(df_entropy)}")
    print(f"Mean Shannon Entropy: {df_entropy['Shannon_Entropy'].mean():.3f} ± {df_entropy['Shannon_Entropy'].std():.3f}")
    print(f"Mean Predictability: {df_entropy['Predictability_Score'].mean():.3f} ± {df_entropy['Predictability_Score'].std():.3f}")
    print(f"Mean Variance CV: {df_entropy['Variance_Entropy_CV'].mean():.3f}")

    # High-entropy proteins (chaotic)
    high_entropy = df_entropy.nlargest(10, 'Shannon_Entropy')
    print("\nTop 10 HIGH-ENTROPY proteins (chaotic, context-dependent):")
    for idx, row in high_entropy.iterrows():
        print(f"  {row['Protein']}: entropy={row['Shannon_Entropy']:.3f}, predictability={row['Predictability_Score']:.3f}")

    # Low-entropy proteins (deterministic)
    low_entropy = df_entropy.nsmallest(10, 'Shannon_Entropy')
    print("\nTop 10 LOW-ENTROPY proteins (deterministic, predictable):")
    for idx, row in low_entropy.iterrows():
        print(f"  {row['Protein']}: entropy={row['Shannon_Entropy']:.3f}, predictability={row['Predictability_Score']:.3f}")

    # Entropy transition proteins
    high_transition = df_entropy.nlargest(10, 'Entropy_Transition')
    print("\nTop 10 ENTROPY TRANSITION proteins (ordered→disordered):")
    for idx, row in high_transition.iterrows():
        print(f"  {row['Protein']}: transition={row['Entropy_Transition']:.3f}")

    print("\n" + "="*80)
    print("Analysis complete! Check /10_insights/ for outputs.")
    print("="*80)


if __name__ == '__main__':
    main()
