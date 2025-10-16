#!/usr/bin/env python3
"""
Outlier Protein Investigator - Agent 6
Identifies proteins with weird, unexpected, or contradictory aging patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/'

def load_data():
    """Load and prepare dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['Canonical_Gene_Symbol'].nunique()} unique proteins")
    return df

def calculate_protein_statistics(df):
    """Calculate per-protein statistics for outlier detection"""

    # Group by protein and calculate comprehensive statistics
    protein_stats = df.groupby('Canonical_Gene_Symbol').agg({
        'Zscore_Delta': ['mean', 'std', 'count', 'min', 'max'],
        'Zscore_Old': ['mean', 'std'],
        'Zscore_Young': ['mean', 'std'],
        'Abundance_Old': 'mean',
        'Abundance_Young': 'mean',
        'Tissue': lambda x: x.nunique(),  # tissue diversity
        'Study_ID': lambda x: x.nunique(),  # cross-study replication
        'Matrisome_Category': lambda x: x.mode().iloc[0] if len(x) > 0 else None,
        'Matrisome_Division': lambda x: x.mode().iloc[0] if len(x) > 0 else None,
    }).reset_index()

    # Flatten column names
    protein_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                             for col in protein_stats.columns.values]

    # Calculate additional metrics
    protein_stats['zscore_delta_abs_mean'] = protein_stats['Zscore_Delta_mean'].abs()
    protein_stats['zscore_delta_range'] = protein_stats['Zscore_Delta_max'] - protein_stats['Zscore_Delta_min']
    protein_stats['coefficient_of_variation'] = (protein_stats['Zscore_Delta_std'] /
                                                  (protein_stats['Zscore_Delta_mean'].abs() + 0.001))

    # Direction consistency: check if all measurements go same direction
    protein_direction = df.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].apply(
        lambda x: (x > 0).sum() / len(x)
    ).reset_index()
    protein_direction.columns = ['Canonical_Gene_Symbol', 'pct_increasing']

    protein_stats = protein_stats.merge(protein_direction, on='Canonical_Gene_Symbol')
    protein_stats['direction_consistency'] = (protein_stats['pct_increasing'] - 0.5).abs() * 2

    return protein_stats

def detect_statistical_outliers(protein_stats):
    """Identify proteins with extreme statistical properties"""

    outliers = {}

    # 1. Extreme Z-score delta (|z| > 3.0)
    extreme_zscore = protein_stats[
        protein_stats['zscore_delta_abs_mean'] > 3.0
    ].sort_values('zscore_delta_abs_mean', ascending=False)
    outliers['extreme_zscore'] = extreme_zscore

    # 2. High variance (unstable across studies)
    high_variance = protein_stats[
        protein_stats['Zscore_Delta_std'] > protein_stats['Zscore_Delta_std'].quantile(0.95)
    ].sort_values('Zscore_Delta_std', ascending=False)
    outliers['high_variance'] = high_variance

    # 3. Contradictory directions (inconsistent)
    contradictory = protein_stats[
        (protein_stats['direction_consistency'] < 0.6) &
        (protein_stats['Zscore_Delta_count'] >= 3)  # need multiple measurements
    ].sort_values('direction_consistency')
    outliers['contradictory'] = contradictory

    # 4. Extreme range (large swings)
    extreme_range = protein_stats[
        protein_stats['zscore_delta_range'] > protein_stats['zscore_delta_range'].quantile(0.95)
    ].sort_values('zscore_delta_range', ascending=False)
    outliers['extreme_range'] = extreme_range

    # 5. High coefficient of variation (noisy)
    noisy = protein_stats[
        (protein_stats['coefficient_of_variation'] > protein_stats['coefficient_of_variation'].quantile(0.95)) &
        (protein_stats['Zscore_Delta_count'] >= 3)
    ].sort_values('coefficient_of_variation', ascending=False)
    outliers['noisy'] = noisy

    return outliers

def find_paradoxical_proteins(df, protein_stats):
    """Identify proteins with biologically unexpected behavior"""

    paradoxical = {}

    # Get biological categories
    fibrotic_keywords = ['COLL', 'COL', 'FN1', 'ACTA2', 'CTGF', 'TIMP', 'SERPINE1', 'PLOD']
    inflammatory_keywords = ['IL', 'TNF', 'CCL', 'CXCL', 'MMP']
    anti_inflammatory = ['IL10', 'TGFB', 'SOCS', 'IL1RN']
    structural = ['LAMA', 'LAMB', 'LAMC', 'COL', 'FBN', 'EMILIN', 'FBLN']

    def matches_keywords(gene, keywords):
        return any(kw in str(gene).upper() for kw in keywords)

    # 1. Pro-fibrotic proteins that DECREASE
    fibrotic_proteins = protein_stats[
        protein_stats['Canonical_Gene_Symbol'].apply(lambda x: matches_keywords(x, fibrotic_keywords))
    ]
    decreasing_fibrotic = fibrotic_proteins[
        fibrotic_proteins['Zscore_Delta_mean'] < -0.5
    ].sort_values('Zscore_Delta_mean')
    paradoxical['decreasing_fibrotic'] = decreasing_fibrotic

    # 2. Pro-inflammatory that INCREASE (unexpected in healthy aging)
    inflammatory_proteins = protein_stats[
        protein_stats['Canonical_Gene_Symbol'].apply(lambda x: matches_keywords(x, inflammatory_keywords))
    ]
    increasing_inflammatory = inflammatory_proteins[
        inflammatory_proteins['Zscore_Delta_mean'] > 0.5
    ].sort_values('Zscore_Delta_mean', ascending=False)
    paradoxical['increasing_inflammatory'] = increasing_inflammatory

    # 3. Structural proteins with high variability across tissues
    structural_proteins = protein_stats[
        protein_stats['Canonical_Gene_Symbol'].apply(lambda x: matches_keywords(x, structural))
    ]
    variable_structural = structural_proteins[
        (structural_proteins['Tissue_<lambda>'] >= 3) &
        (structural_proteins['zscore_delta_range'] > 2.0)
    ].sort_values('zscore_delta_range', ascending=False)
    paradoxical['variable_structural'] = variable_structural

    return paradoxical

def find_sleeping_giants(protein_stats, df):
    """Identify proteins with small changes but high biological importance"""

    # Criteria: small z-score change BUT high consistency + multiple studies
    sleeping_giants = protein_stats[
        (protein_stats['zscore_delta_abs_mean'] < 0.5) &  # small change
        (protein_stats['zscore_delta_abs_mean'] > 0.1) &  # but not zero
        (protein_stats['direction_consistency'] > 0.8) &  # very consistent
        (protein_stats['Study_ID_<lambda>'] >= 3) &  # replicated
        (protein_stats['Tissue_<lambda>'] >= 2)  # multiple tissues
    ].sort_values('direction_consistency', ascending=False)

    # Add biological importance markers
    important_categories = ['Core matrisome', 'ECM Glycoproteins', 'ECM Regulators']
    sleeping_giants = sleeping_giants[
        sleeping_giants['Matrisome_Division_<lambda>'].isin(important_categories)
    ]

    return sleeping_giants

def multivariate_outlier_detection(protein_stats):
    """Use PCA to find outliers in multivariate space"""

    # Select numerical features for PCA
    features = [
        'Zscore_Delta_mean', 'Zscore_Delta_std', 'zscore_delta_range',
        'direction_consistency', 'Tissue_<lambda>', 'Study_ID_<lambda>'
    ]

    # Remove rows with NaN
    pca_data = protein_stats[features].dropna()
    protein_names = protein_stats.loc[pca_data.index, 'Canonical_Gene_Symbol']

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)

    # PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_data)

    # Calculate Mahalanobis distance (multivariate outlier score)
    mean = pca_coords.mean(axis=0)
    cov = np.cov(pca_coords.T)
    inv_cov = np.linalg.inv(cov)

    mahal_distances = []
    for point in pca_coords:
        diff = point - mean
        mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
        mahal_distances.append(mahal_dist)

    # Identify multivariate outliers (top 5%)
    mahal_threshold = np.percentile(mahal_distances, 95)
    outlier_indices = np.where(np.array(mahal_distances) > mahal_threshold)[0]

    pca_outliers = protein_stats.iloc[pca_data.index[outlier_indices]].copy()
    pca_outliers['mahalanobis_distance'] = np.array(mahal_distances)[outlier_indices]
    pca_outliers = pca_outliers.sort_values('mahalanobis_distance', ascending=False)

    return pca_outliers, pca_coords, protein_names, mahal_distances, pca

def create_visualizations(protein_stats, pca_coords, protein_names, mahal_distances, outliers, pca):
    """Generate comprehensive visualization plots"""

    fig = plt.figure(figsize=(20, 12))

    # 1. Z-score delta distribution with outliers highlighted
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(protein_stats['Zscore_Delta_mean'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=3.0, color='red', linestyle='--', label='|z| = 3.0 threshold')
    plt.axvline(x=-3.0, color='red', linestyle='--')
    plt.xlabel('Mean Z-score Delta')
    plt.ylabel('Frequency')
    plt.title('Distribution of Z-score Changes')
    plt.legend()

    # 2. Variance vs Mean (noise detection)
    ax2 = plt.subplot(2, 3, 2)
    plt.scatter(protein_stats['zscore_delta_abs_mean'],
                protein_stats['Zscore_Delta_std'],
                alpha=0.5, s=30)
    high_var = outliers['high_variance'].head(10)
    plt.scatter(high_var['zscore_delta_abs_mean'],
                high_var['Zscore_Delta_std'],
                color='red', s=100, marker='*', label='High variance outliers')
    plt.xlabel('|Mean Z-score Delta|')
    plt.ylabel('Z-score Std Dev')
    plt.title('Variance vs Magnitude (Noise Detection)')
    plt.legend()

    # 3. Direction consistency
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(protein_stats['direction_consistency'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Direction Consistency (0=contradictory, 1=consistent)')
    plt.ylabel('Frequency')
    plt.title('Directional Consistency Distribution')
    plt.axvline(x=0.6, color='red', linestyle='--', label='Inconsistency threshold')
    plt.legend()

    # 4. PCA outlier plot
    ax4 = plt.subplot(2, 3, 4)
    scatter = plt.scatter(pca_coords[:, 0], pca_coords[:, 1],
                         c=mahal_distances, cmap='YlOrRd', alpha=0.6, s=30)
    plt.colorbar(scatter, label='Mahalanobis Distance')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Multivariate Outlier Detection (PCA Space)')

    # Annotate top outliers
    mahal_array = np.array(mahal_distances)
    top_outlier_idx = np.argsort(mahal_array)[-5:]
    for idx in top_outlier_idx:
        plt.annotate(protein_names.iloc[idx],
                    (pca_coords[idx, 0], pca_coords[idx, 1]),
                    fontsize=8, alpha=0.7)

    # 5. Study replication vs effect size
    ax5 = plt.subplot(2, 3, 5)
    plt.scatter(protein_stats['Study_ID_<lambda>'],
                protein_stats['zscore_delta_abs_mean'],
                alpha=0.5, s=30)
    plt.xlabel('Number of Studies')
    plt.ylabel('|Mean Z-score Delta|')
    plt.title('Effect Size vs Cross-Study Replication')

    # 6. Tissue diversity vs variability
    ax6 = plt.subplot(2, 3, 6)
    plt.scatter(protein_stats['Tissue_<lambda>'],
                protein_stats['zscore_delta_range'],
                alpha=0.5, s=30)
    plt.xlabel('Number of Tissues')
    plt.ylabel('Z-score Delta Range')
    plt.title('Tissue Diversity vs Variability')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}agent_06_outlier_visualizations.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {OUTPUT_DIR}agent_06_outlier_visualizations.png")
    plt.close()

def generate_protein_profiles(df, top_proteins):
    """Generate detailed profiles for top outlier proteins"""

    profiles = []

    for protein in top_proteins:
        protein_data = df[df['Canonical_Gene_Symbol'] == protein]

        if len(protein_data) == 0:
            continue

        profile = {
            'Protein': protein,
            'N_measurements': len(protein_data),
            'N_studies': protein_data['Study_ID'].nunique(),
            'N_tissues': protein_data['Tissue'].nunique(),
            'Tissues': ', '.join(protein_data['Tissue'].unique()),
            'Studies': ', '.join(protein_data['Study_ID'].unique()),
            'Category': protein_data['Matrisome_Category'].mode().iloc[0] if len(protein_data) > 0 else 'Unknown',
            'Division': protein_data['Matrisome_Division'].mode().iloc[0] if len(protein_data) > 0 else 'Unknown',
            'Mean_Zscore_Delta': protein_data['Zscore_Delta'].mean(),
            'Std_Zscore_Delta': protein_data['Zscore_Delta'].std(),
            'Min_Zscore_Delta': protein_data['Zscore_Delta'].min(),
            'Max_Zscore_Delta': protein_data['Zscore_Delta'].max(),
            'Pct_Increasing': (protein_data['Zscore_Delta'] > 0).sum() / len(protein_data) * 100,
            'Mean_Old_Abundance': protein_data['Abundance_Old'].mean(),
            'Mean_Young_Abundance': protein_data['Abundance_Young'].mean(),
        }

        profiles.append(profile)

    return pd.DataFrame(profiles)

def calculate_discovery_potential(protein_stats, outliers, df):
    """Rank proteins by discovery potential (likelihood of breakthrough findings)"""

    # Combine all outlier types
    all_outliers = set()
    for category, outlier_df in outliers.items():
        all_outliers.update(outlier_df['Canonical_Gene_Symbol'].values)

    discovery_scores = []

    for protein in all_outliers:
        protein_info = protein_stats[protein_stats['Canonical_Gene_Symbol'] == protein]
        if len(protein_info) == 0:
            continue

        protein_info = protein_info.iloc[0]

        # Scoring criteria (higher = more interesting)
        score = 0

        # 1. Extreme effect size (max 30 points)
        effect_size = min(abs(protein_info['Zscore_Delta_mean']) / 5.0, 1.0) * 30
        score += effect_size

        # 2. Consistency (max 20 points) - consistent findings are more trustworthy
        consistency_score = protein_info['direction_consistency'] * 20
        score += consistency_score

        # 3. Cross-study replication (max 20 points)
        replication_score = min(protein_info['Study_ID_<lambda>'] / 5.0, 1.0) * 20
        score += replication_score

        # 4. Multi-tissue presence (max 15 points) - broader relevance
        tissue_score = min(protein_info['Tissue_<lambda>'] / 4.0, 1.0) * 15
        score += tissue_score

        # 5. Biological importance (max 15 points)
        if protein_info['Matrisome_Division_<lambda>'] == 'Core matrisome':
            score += 15
        elif protein_info['Matrisome_Division_<lambda>'] == 'Matrisome-associated':
            score += 10

        # Penalty for very high noise (subtract up to 10 points)
        if protein_info['coefficient_of_variation'] > 2.0:
            noise_penalty = min(protein_info['coefficient_of_variation'] / 5.0, 1.0) * 10
            score -= noise_penalty

        discovery_scores.append({
            'Protein': protein,
            'Discovery_Score': score,
            'Effect_Size_Points': effect_size,
            'Consistency_Points': consistency_score,
            'Replication_Points': replication_score,
            'Tissue_Points': tissue_score,
            'Category': protein_info['Matrisome_Category_<lambda>'],
            'Division': protein_info['Matrisome_Division_<lambda>'],
        })

    discovery_df = pd.DataFrame(discovery_scores).sort_values('Discovery_Score', ascending=False)
    return discovery_df

def main():
    """Main analysis pipeline"""

    print("=" * 80)
    print("AGENT 6: OUTLIER PROTEIN INVESTIGATOR")
    print("=" * 80)

    # Load data
    print("\n[1/8] Loading dataset...")
    df = load_data()

    # Calculate statistics
    print("\n[2/8] Calculating protein statistics...")
    protein_stats = calculate_protein_statistics(df)
    print(f"Analyzed {len(protein_stats)} unique proteins")

    # Detect statistical outliers
    print("\n[3/8] Detecting statistical outliers...")
    outliers = detect_statistical_outliers(protein_stats)
    print(f"  - Extreme Z-score (|z| > 3.0): {len(outliers['extreme_zscore'])} proteins")
    print(f"  - High variance: {len(outliers['high_variance'])} proteins")
    print(f"  - Contradictory directions: {len(outliers['contradictory'])} proteins")
    print(f"  - Extreme range: {len(outliers['extreme_range'])} proteins")
    print(f"  - Noisy measurements: {len(outliers['noisy'])} proteins")

    # Find paradoxical proteins
    print("\n[4/8] Finding paradoxical proteins...")
    paradoxical = find_paradoxical_proteins(df, protein_stats)
    print(f"  - Decreasing fibrotic: {len(paradoxical['decreasing_fibrotic'])} proteins")
    print(f"  - Increasing inflammatory: {len(paradoxical['increasing_inflammatory'])} proteins")
    print(f"  - Variable structural: {len(paradoxical['variable_structural'])} proteins")

    # Find sleeping giants
    print("\n[5/8] Identifying sleeping giants...")
    sleeping_giants = find_sleeping_giants(protein_stats, df)
    print(f"  - Found {len(sleeping_giants)} sleeping giants")

    # Multivariate outlier detection
    print("\n[6/8] Performing multivariate outlier detection...")
    pca_outliers, pca_coords, protein_names, mahal_distances, pca = multivariate_outlier_detection(protein_stats)
    print(f"  - Identified {len(pca_outliers)} multivariate outliers")

    # Create visualizations
    print("\n[7/8] Generating visualizations...")
    create_visualizations(protein_stats, pca_coords, protein_names, mahal_distances, outliers, pca)

    # Calculate discovery potential
    print("\n[8/8] Ranking by discovery potential...")
    discovery_ranking = calculate_discovery_potential(protein_stats, outliers, df)
    print(f"  - Ranked {len(discovery_ranking)} outlier proteins")

    # Compile top 20 for detailed profiling
    top_20 = discovery_ranking.head(20)['Protein'].tolist()
    top_20_profiles = generate_protein_profiles(df, top_20)

    print("\n" + "=" * 80)
    print("TOP 20 OUTLIER PROTEINS BY DISCOVERY POTENTIAL:")
    print("=" * 80)
    for idx, row in discovery_ranking.head(20).iterrows():
        print(f"{row['Protein']:15} | Score: {row['Discovery_Score']:.1f} | {row['Division']}")

    # Save all results
    print("\n" + "=" * 80)
    print("SAVING RESULTS...")
    print("=" * 80)

    # Save comprehensive results
    results = {
        'protein_stats': protein_stats,
        'statistical_outliers': outliers,
        'paradoxical': paradoxical,
        'sleeping_giants': sleeping_giants,
        'pca_outliers': pca_outliers,
        'discovery_ranking': discovery_ranking,
        'top_20_profiles': top_20_profiles
    }

    # Export to CSV
    protein_stats.to_csv(f'{OUTPUT_DIR}agent_06_protein_statistics.csv', index=False)
    discovery_ranking.to_csv(f'{OUTPUT_DIR}agent_06_discovery_ranking.csv', index=False)
    top_20_profiles.to_csv(f'{OUTPUT_DIR}agent_06_top20_profiles.csv', index=False)
    outliers['extreme_zscore'].to_csv(f'{OUTPUT_DIR}agent_06_extreme_zscore.csv', index=False)
    paradoxical['decreasing_fibrotic'].to_csv(f'{OUTPUT_DIR}agent_06_paradoxical_fibrotic.csv', index=False)
    sleeping_giants.to_csv(f'{OUTPUT_DIR}agent_06_sleeping_giants.csv', index=False)

    print(f"✓ Saved protein_statistics.csv ({len(protein_stats)} proteins)")
    print(f"✓ Saved discovery_ranking.csv ({len(discovery_ranking)} proteins)")
    print(f"✓ Saved top20_profiles.csv (20 proteins)")
    print(f"✓ Saved outlier category files")
    print(f"✓ Saved visualization plot")

    return results

if __name__ == '__main__':
    results = main()
    print("\n✓ ANALYSIS COMPLETE")
