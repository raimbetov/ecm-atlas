#!/usr/bin/env python3
"""
Cross-Tissue ECM Aging Correlation Analysis
Agent 1 - Q1.1.3: Universal vs. Personalized ECM Aging Trajectories

This script analyzes cross-tissue correlations of ECM aging signatures
to test whether aging patterns are universal or tissue-specific.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("CROSS-TISSUE ECM AGING CORRELATION ANALYSIS")
print("Agent 1 - Q1.1.3: Universal vs. Personalized Trajectories")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/7] Loading ECM-Atlas database...")

db_path = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
df = pd.read_csv(db_path)

print(f"Loaded {len(df):,} rows")
print(f"Columns: {', '.join(df.columns[:10])}...")
print(f"\nUnique tissues: {df['Tissue'].nunique()}")
print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")

# Filter for high-quality data with Zscore_Delta
df_clean = df[df['Zscore_Delta'].notna()].copy()
print(f"\nRows with Zscore_Delta: {len(df_clean):,}")

# Get tissue distribution
tissue_counts = df_clean['Tissue'].value_counts()
print(f"\nTissue distribution:")
for tissue, count in tissue_counts.head(10).items():
    print(f"  {tissue}: {count}")

# ============================================================================
# 2. IDENTIFY TOP AGING PROTEINS (UNIVERSAL CANDIDATES)
# ============================================================================
print("\n[2/7] Identifying top aging proteins across all tissues...")

# Calculate mean absolute z-score delta and frequency for each protein
protein_stats = df_clean.groupby('Canonical_Gene_Symbol').agg({
    'Zscore_Delta': ['mean', 'std', 'count'],
    'Tissue': 'nunique'
}).round(3)

protein_stats.columns = ['Mean_Zscore_Delta', 'Std_Zscore_Delta', 'N_Measurements', 'N_Tissues']
protein_stats['Abs_Mean_Delta'] = protein_stats['Mean_Zscore_Delta'].abs()

# Filter proteins present in 3+ tissues with strong signals
protein_stats_filtered = protein_stats[
    (protein_stats['N_Tissues'] >= 3) &
    (protein_stats['N_Measurements'] >= 5)
].copy()

protein_stats_filtered = protein_stats_filtered.sort_values('Abs_Mean_Delta', ascending=False)

print(f"\nProteins in 3+ tissues with 5+ measurements: {len(protein_stats_filtered)}")
print("\nTop 20 proteins by absolute mean z-score delta:")
print(protein_stats_filtered.head(20))

# Save protein statistics
protein_stats_filtered.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/protein_statistics.csv')

# Select top proteins for detailed analysis
top_proteins = protein_stats_filtered.head(30).index.tolist()
print(f"\nSelected {len(top_proteins)} proteins for cross-tissue analysis")

# ============================================================================
# 3. CREATE TISSUE × PROTEIN MATRIX
# ============================================================================
print("\n[3/7] Creating tissue × protein z-score delta matrix...")

# Filter for top proteins
df_top = df_clean[df_clean['Canonical_Gene_Symbol'].isin(top_proteins)].copy()

# Pivot to create tissue × protein matrix
# Average z-scores within tissue-protein combinations
tissue_protein_matrix = df_top.pivot_table(
    index='Tissue',
    columns='Canonical_Gene_Symbol',
    values='Zscore_Delta',
    aggfunc='mean'
)

print(f"\nMatrix shape: {tissue_protein_matrix.shape}")
print(f"Tissues: {tissue_protein_matrix.shape[0]}")
print(f"Proteins: {tissue_protein_matrix.shape[1]}")

# Calculate missingness
missingness = tissue_protein_matrix.isna().sum() / len(tissue_protein_matrix) * 100
print(f"\nProtein missingness across tissues:")
print(missingness.sort_values().head(10))

# Keep only proteins with data in at least 3 tissues (more permissive)
min_tissues = 3
proteins_to_keep = missingness[missingness < (100 * (1 - min_tissues/len(tissue_protein_matrix)))].index
tissue_protein_matrix_filtered = tissue_protein_matrix[proteins_to_keep]

print(f"\nProteins with data in ≥{min_tissues} tissues: {len(proteins_to_keep)}")
print(f"Final matrix: {tissue_protein_matrix_filtered.shape}")

# Save matrix
tissue_protein_matrix_filtered.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/tissue_protein_matrix.csv')

# ============================================================================
# 4. CROSS-TISSUE CORRELATION ANALYSIS
# ============================================================================
print("\n[4/7] Calculating pairwise tissue correlations...")

# Calculate pairwise correlations between tissues
# Transpose so tissues are columns for correlation
tissue_corr = tissue_protein_matrix_filtered.T.corr(method='pearson', min_periods=5)

print(f"\nTissue correlation matrix shape: {tissue_corr.shape}")

# Calculate mean correlation for each tissue pair
upper_triangle = tissue_corr.where(np.triu(np.ones(tissue_corr.shape), k=1).astype(bool))
correlations_flat = upper_triangle.stack()

print(f"\nCross-tissue correlation statistics:")
print(f"  Mean correlation: {correlations_flat.mean():.3f}")
print(f"  Median correlation: {correlations_flat.median():.3f}")
print(f"  Std correlation: {correlations_flat.std():.3f}")
print(f"  Range: [{correlations_flat.min():.3f}, {correlations_flat.max():.3f}]")

# Find most and least correlated tissue pairs
print(f"\nMost correlated tissue pairs:")
top_corr = correlations_flat.nlargest(10)
for (t1, t2), corr in top_corr.items():
    print(f"  {t1} <-> {t2}: r = {corr:.3f}")

print(f"\nLeast correlated tissue pairs:")
bottom_corr = correlations_flat.nsmallest(10)
for (t1, t2), corr in bottom_corr.items():
    print(f"  {t1} <-> {t2}: r = {corr:.3f}")

# Save correlation matrix
tissue_corr.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/tissue_correlation_matrix.csv')

# ============================================================================
# 5. HIERARCHICAL CLUSTERING OF TISSUES
# ============================================================================
print("\n[5/7] Performing hierarchical clustering of tissues...")

# Use correlation distance for clustering
# Make sure the matrix is symmetric
tissue_corr_symmetric = (tissue_corr + tissue_corr.T) / 2
corr_distance = 1 - tissue_corr_symmetric.abs()

# Convert to condensed distance matrix for linkage
from scipy.spatial.distance import pdist
condensed_dist = pdist(tissue_protein_matrix_filtered.fillna(0), metric='correlation')
linkage_matrix = hierarchy.linkage(condensed_dist, method='average')

# Get cluster assignments at different heights
from scipy.cluster.hierarchy import fcluster
clusters_2 = fcluster(linkage_matrix, 2, criterion='maxclust')
clusters_3 = fcluster(linkage_matrix, 3, criterion='maxclust')
clusters_4 = fcluster(linkage_matrix, 4, criterion='maxclust')

cluster_df = pd.DataFrame({
    'Tissue': tissue_corr.index,
    'Cluster_2': clusters_2,
    'Cluster_3': clusters_3,
    'Cluster_4': clusters_4
})

print("\nClustering results (k=3):")
for cluster_id in sorted(cluster_df['Cluster_3'].unique()):
    tissues_in_cluster = cluster_df[cluster_df['Cluster_3'] == cluster_id]['Tissue'].tolist()
    print(f"  Cluster {cluster_id}: {', '.join(tissues_in_cluster)}")

cluster_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/tissue_clusters.csv', index=False)

# ============================================================================
# 6. PRINCIPAL COMPONENT ANALYSIS
# ============================================================================
print("\n[6/7] Performing PCA on tissue aging patterns...")

# Prepare data for PCA (impute missing values with column means)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
tissue_protein_imputed = pd.DataFrame(
    imputer.fit_transform(tissue_protein_matrix_filtered),
    index=tissue_protein_matrix_filtered.index,
    columns=tissue_protein_matrix_filtered.columns
)

# Standardize
scaler = StandardScaler()
tissue_protein_scaled = pd.DataFrame(
    scaler.fit_transform(tissue_protein_imputed),
    index=tissue_protein_imputed.index,
    columns=tissue_protein_imputed.columns
)

# Perform PCA
pca = PCA()
pca_scores = pca.fit_transform(tissue_protein_scaled)

# Create PCA results dataframe
pca_df = pd.DataFrame(
    pca_scores[:, :10],
    index=tissue_protein_scaled.index,
    columns=[f'PC{i+1}' for i in range(10)]
)

# Variance explained
variance_explained = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_explained)

print(f"\nPCA variance explained:")
for i in range(min(10, len(variance_explained))):
    print(f"  PC{i+1}: {variance_explained[i]*100:.2f}% (cumulative: {cumulative_variance[i]*100:.2f}%)")

# PC loadings
loadings_df = pd.DataFrame(
    pca.components_[:10, :].T,
    index=tissue_protein_scaled.columns,
    columns=[f'PC{i+1}' for i in range(10)]
)

print(f"\nTop 10 proteins driving PC1 (universal aging component?):")
pc1_loadings = loadings_df['PC1'].abs().sort_values(ascending=False)
for protein, loading in pc1_loadings.head(10).items():
    direction = "↑" if loadings_df.loc[protein, 'PC1'] > 0 else "↓"
    print(f"  {protein}: {loading:.3f} {direction}")

# Save PCA results
pca_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/pca_scores.csv')
loadings_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/pca_loadings.csv')

variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(variance_explained))],
    'Variance_Explained': variance_explained,
    'Cumulative_Variance': cumulative_variance
})
variance_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/pca_variance.csv', index=False)

# ============================================================================
# 7. VARIANCE PARTITIONING: UNIVERSAL vs TISSUE-SPECIFIC
# ============================================================================
print("\n[7/7] Partitioning variance: universal vs. tissue-specific...")

# For each protein, calculate variance explained by:
# 1. Universal component (mean across all tissues)
# 2. Tissue-specific component (deviation from mean)

variance_partition = []

for protein in tissue_protein_matrix_filtered.columns:
    protein_data = tissue_protein_matrix_filtered[protein].dropna()

    if len(protein_data) < 3:
        continue

    # Universal component: grand mean
    grand_mean = protein_data.mean()

    # Total variance
    total_var = protein_data.var()

    # Variance explained by universal component (mean)
    universal_var = (grand_mean ** 2) / total_var if total_var > 0 else 0

    # Tissue-specific variance (residual)
    tissue_specific_var = 1 - universal_var if universal_var < 1 else 0

    variance_partition.append({
        'Protein': protein,
        'Grand_Mean_Zscore': grand_mean,
        'Total_Variance': total_var,
        'Universal_Component': universal_var,
        'Tissue_Specific_Component': tissue_specific_var,
        'N_Tissues': len(protein_data)
    })

variance_partition_df = pd.DataFrame(variance_partition)
variance_partition_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/variance_partition.csv', index=False)

print(f"\nVariance partitioning summary:")
print(f"  Mean universal component: {variance_partition_df['Universal_Component'].mean():.3f}")
print(f"  Mean tissue-specific component: {variance_partition_df['Tissue_Specific_Component'].mean():.3f}")

print("\nProteins with strongest universal signals (high grand mean, low tissue variance):")
universal_candidates = variance_partition_df.sort_values('Grand_Mean_Zscore', key=abs, ascending=False).head(15)
print(universal_candidates[['Protein', 'Grand_Mean_Zscore', 'Total_Variance', 'N_Tissues']])

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("Output files saved to: /Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent1/")
print("="*80)
