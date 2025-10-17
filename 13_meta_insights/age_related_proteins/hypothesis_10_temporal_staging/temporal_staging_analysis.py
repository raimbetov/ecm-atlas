#!/usr/bin/env python3
"""
HYPOTHESIS 10: TEMPORAL STAGING - Aging Phases Discovery
Cluster 405 universal proteins to discover STAGES of aging (early vs late changers)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading universal markers data...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_01_universal_markers/agent_01_universal_markers_data.csv')
print(f"Total proteins: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# FEATURE ENGINEERING FOR TEMPORAL PATTERNS
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING: Temporal Pattern Features")
print("="*80)

# Features for clustering:
# 1. Effect size (magnitude of change)
# 2. Consistency (reliability of direction)
# 3. Strong_Effect_Rate (proportion of strong effects - early vs late indicator)
# 4. N_Tissues (breadth of change)

feature_cols = [
    'Abs_Mean_Zscore_Delta',    # Magnitude
    'Direction_Consistency',     # Reliability
    'Strong_Effect_Rate',        # Early vs late?
    'N_Tissues'                  # Breadth
]

# Create feature matrix
X = df[feature_cols].values
protein_names = df['Gene_Symbol'].values

# Check for missing values
print("\nMissing values per feature:")
for col in feature_cols:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing} ({100*missing/len(df):.1f}%)")

# Remove rows with missing values
valid_mask = ~np.isnan(X).any(axis=1)
X_clean = X[valid_mask]
protein_names_clean = protein_names[valid_mask]
df_clean = df[valid_mask].copy()

print(f"\nProteins with complete features: {len(X_clean)}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

print("\nFeature statistics (standardized):")
print(pd.DataFrame(X_scaled, columns=feature_cols).describe())

# ============================================================================
# DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
print("\n" + "="*80)
print("OPTIMAL CLUSTER DETERMINATION")
print("="*80)

k_range = range(2, 8)
silhouette_scores = []
davies_bouldin_scores = []
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    inertia = kmeans.inertia_

    silhouette_scores.append(sil)
    davies_bouldin_scores.append(db)
    inertias.append(inertia)

    print(f"k={k}: Silhouette={sil:.3f}, Davies-Bouldin={db:.3f}, Inertia={inertia:.1f}")

# Plot elbow curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster SS)', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score (higher=better)', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

axes[2].plot(k_range, davies_bouldin_scores, 'o-', linewidth=2, markersize=8, color='red')
axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[2].set_ylabel('Davies-Bouldin Index (lower=better)', fontsize=12)
axes[2].set_title('Davies-Bouldin Analysis', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/cluster_optimization.png', dpi=300, bbox_inches='tight')
print("\nSaved: cluster_optimization.png")

# Select optimal k (highest silhouette score)
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k based on Silhouette score: {optimal_k}")

# ============================================================================
# K-MEANS CLUSTERING WITH OPTIMAL K
# ============================================================================
print("\n" + "="*80)
print(f"K-MEANS CLUSTERING: k={optimal_k}")
print("="*80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=50)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df_clean['Cluster'] = cluster_labels

# Calculate cluster statistics
print("\nCluster Statistics:")
print("="*80)

cluster_stats = []
for cluster_id in range(optimal_k):
    cluster_mask = cluster_labels == cluster_id
    cluster_data = df_clean[cluster_mask]

    stats = {
        'Cluster': cluster_id,
        'N_Proteins': cluster_mask.sum(),
        'Mean_Effect_Size': cluster_data['Abs_Mean_Zscore_Delta'].mean(),
        'Mean_Consistency': cluster_data['Direction_Consistency'].mean(),
        'Mean_Strong_Rate': cluster_data['Strong_Effect_Rate'].mean(),
        'Mean_N_Tissues': cluster_data['N_Tissues'].mean(),
        'Predominant_Direction': cluster_data['Predominant_Direction'].mode()[0] if len(cluster_data) > 0 else 'N/A'
    }

    cluster_stats.append(stats)

    print(f"\nCluster {cluster_id}: {stats['N_Proteins']} proteins")
    print(f"  Effect Size: {stats['Mean_Effect_Size']:.3f}")
    print(f"  Consistency: {stats['Mean_Consistency']:.3f}")
    print(f"  Strong Effect Rate: {stats['Mean_Strong_Rate']:.3f}")
    print(f"  Mean Tissues: {stats['Mean_N_Tissues']:.1f}")
    print(f"  Direction: {stats['Predominant_Direction']}")

cluster_stats_df = pd.DataFrame(cluster_stats)
cluster_stats_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/cluster_statistics.csv', index=False)
print("\nSaved: cluster_statistics.csv")

# ============================================================================
# INTERPRET CLUSTERS AS AGING STAGES
# ============================================================================
print("\n" + "="*80)
print("BIOLOGICAL INTERPRETATION: Aging Stages")
print("="*80)

# Sort clusters by Strong_Effect_Rate (early changers have higher rate)
cluster_stats_df = cluster_stats_df.sort_values('Mean_Strong_Rate', ascending=False)

stage_names = []
for idx, row in cluster_stats_df.iterrows():
    cluster_id = row['Cluster']

    # Interpret based on features
    if row['Mean_Strong_Rate'] > 0.35 and row['Mean_Effect_Size'] > 0.4:
        stage = "PHASE 1: Early Dramatic Changes (20-40 years)"
        interpretation = "High strong effect rate + high effect size = proteins that change dramatically early in aging"
    elif row['Mean_Strong_Rate'] < 0.2 and row['Mean_Effect_Size'] > 0.3:
        stage = "PHASE 3: Late Accumulation (60+ years)"
        interpretation = "Low strong effect rate + high effect size = proteins that accumulate slowly but reach high levels late"
    elif row['Mean_Consistency'] > 0.65:
        stage = "PHASE 2: Consistent Shifters (40-60 years)"
        interpretation = "High consistency = proteins that change reliably across tissues during middle age"
    else:
        stage = f"CLUSTER {cluster_id}: Gradual/Variable Changes"
        interpretation = "Moderate patterns that don't fit clear early/late profile"

    stage_names.append(stage)

    print(f"\n{stage}")
    print(f"  {interpretation}")
    print(f"  N={row['N_Proteins']} proteins")
    print(f"  Key metrics: Effect={row['Mean_Effect_Size']:.2f}, Consistency={row['Mean_Consistency']:.2f}, Strong Rate={row['Mean_Strong_Rate']:.2f}")

# Add stage names to cluster stats
cluster_stats_df['Stage_Name'] = stage_names

# ============================================================================
# HIERARCHICAL CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("HIERARCHICAL CLUSTERING")
print("="*80)

# Perform hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram
fig, ax = plt.subplots(figsize=(20, 8))
dendrogram(linkage_matrix,
          labels=protein_names_clean,
          leaf_font_size=6,
          color_threshold=0.7*max(linkage_matrix[:,2]))
ax.set_xlabel('Protein', fontsize=14)
ax.set_ylabel('Distance (Ward)', fontsize=14)
ax.set_title('Hierarchical Clustering Dendrogram: Temporal Aging Patterns', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/dendrogram.png', dpi=300, bbox_inches='tight')
print("Saved: dendrogram.png")

# ============================================================================
# PCA VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("PCA VISUALIZATION")
print("="*80)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
print(f"PC2 variance explained: {pca.explained_variance_ratio_[1]:.1%}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# Create PCA plot colored by cluster
fig, ax = plt.subplots(figsize=(12, 10))

colors = plt.cm.Set1(np.linspace(0, 1, optimal_k))
for cluster_id in range(optimal_k):
    cluster_mask = cluster_labels == cluster_id
    stage_name = cluster_stats_df[cluster_stats_df['Cluster']==cluster_id]['Stage_Name'].values[0]

    ax.scatter(X_pca[cluster_mask, 0],
              X_pca[cluster_mask, 1],
              c=[colors[cluster_id]],
              label=f"C{cluster_id}: {stage_name.split(':')[0]}",
              alpha=0.6,
              s=100,
              edgecolors='black',
              linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('PCA: Temporal Aging Clusters', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/pca_clusters.png', dpi=300, bbox_inches='tight')
print("Saved: pca_clusters.png")

# ============================================================================
# RADAR PLOT: CLUSTER PROFILES
# ============================================================================
print("\n" + "="*80)
print("RADAR PLOT: Cluster Profiles")
print("="*80)

from math import pi

# Normalize cluster statistics for radar plot (0-1 scale)
radar_features = ['Mean_Effect_Size', 'Mean_Consistency', 'Mean_Strong_Rate', 'Mean_N_Tissues']
cluster_stats_norm = cluster_stats_df.copy()
for feat in radar_features:
    max_val = cluster_stats_norm[feat].max()
    min_val = cluster_stats_norm[feat].min()
    if max_val > min_val:
        cluster_stats_norm[feat] = (cluster_stats_norm[feat] - min_val) / (max_val - min_val)

# Setup radar plot
categories = ['Effect\nSize', 'Consistency', 'Strong\nEffect Rate', 'Tissue\nBreadth']
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

colors_radar = plt.cm.Set1(np.linspace(0, 1, optimal_k))
for idx, row in cluster_stats_norm.iterrows():
    values = row[radar_features].tolist()
    values += values[:1]

    cluster_id = row['Cluster']
    stage_name = row['Stage_Name'].split(':')[0]

    ax.plot(angles, values, 'o-', linewidth=2,
           label=f"C{cluster_id}: {stage_name}",
           color=colors_radar[cluster_id])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[cluster_id])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
ax.set_title('Temporal Aging Cluster Profiles', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/cluster_radar.png', dpi=300, bbox_inches='tight')
print("Saved: cluster_radar.png")

# ============================================================================
# TOP PROTEINS PER CLUSTER
# ============================================================================
print("\n" + "="*80)
print("TOP PROTEINS PER CLUSTER")
print("="*80)

top_proteins_per_cluster = []

for cluster_id in range(optimal_k):
    cluster_data = df_clean[df_clean['Cluster'] == cluster_id].copy()

    # Sort by effect size within cluster
    cluster_data = cluster_data.sort_values('Abs_Mean_Zscore_Delta', ascending=False)

    stage_name = cluster_stats_df[cluster_stats_df['Cluster']==cluster_id]['Stage_Name'].values[0]

    print(f"\n{stage_name}")
    print("="*80)
    print(f"Top 20 proteins:")

    for idx, row in cluster_data.head(20).iterrows():
        protein_info = {
            'Cluster': cluster_id,
            'Stage': stage_name,
            'Gene_Symbol': row['Gene_Symbol'],
            'Protein_Name': row['Protein_Name'],
            'Matrisome_Category': row['Matrisome_Category'],
            'N_Tissues': row['N_Tissues'],
            'Direction': row['Predominant_Direction'],
            'Effect_Size': row['Abs_Mean_Zscore_Delta'],
            'Consistency': row['Direction_Consistency'],
            'Strong_Effect_Rate': row['Strong_Effect_Rate'],
            'Universality_Score': row['Universality_Score']
        }
        top_proteins_per_cluster.append(protein_info)

        print(f"  {row['Gene_Symbol']:12s} | {row['Matrisome_Category']:20s} | "
              f"Effect={row['Abs_Mean_Zscore_Delta']:.2f} | "
              f"N_Tissues={row['N_Tissues']:2d} | "
              f"{row['Predominant_Direction']:4s}")

# Save top proteins
top_proteins_df = pd.DataFrame(top_proteins_per_cluster)
top_proteins_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/top_proteins_by_cluster.csv', index=False)
print("\nSaved: top_proteins_by_cluster.csv")

# ============================================================================
# CLUSTER HEATMAP
# ============================================================================
print("\n" + "="*80)
print("CLUSTER HEATMAP")
print("="*80)

# Create matrix of normalized features per cluster
heatmap_data = cluster_stats_norm[radar_features].values
heatmap_labels = [f"C{row['Cluster']}: {row['Stage_Name'].split(':')[0]}"
                 for _, row in cluster_stats_norm.iterrows()]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data,
           annot=True,
           fmt='.2f',
           xticklabels=['Effect Size', 'Consistency', 'Strong Rate', 'N Tissues'],
           yticklabels=heatmap_labels,
           cmap='YlOrRd',
           cbar_kws={'label': 'Normalized Value'},
           ax=ax,
           linewidths=1,
           linecolor='white')

ax.set_title('Temporal Cluster Profiles Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/cluster_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: cluster_heatmap.png")

# ============================================================================
# TIMELINE DIAGRAM
# ============================================================================
print("\n" + "="*80)
print("AGING TIMELINE DIAGRAM")
print("="*80)

# Sort clusters by Strong_Effect_Rate (proxy for temporal order)
timeline_clusters = cluster_stats_df.sort_values('Mean_Strong_Rate', ascending=False)

fig, ax = plt.subplots(figsize=(16, 8))

# Define age ranges
age_ranges = {
    'PHASE 1': (20, 40),
    'PHASE 2': (40, 60),
    'PHASE 3': (60, 80)
}

y_positions = np.linspace(0.8, 0.2, optimal_k)

for idx, (_, row) in enumerate(timeline_clusters.iterrows()):
    stage = row['Stage_Name']
    cluster_id = row['Cluster']
    n_proteins = row['N_Proteins']

    # Determine position on timeline
    if 'PHASE 1' in stage or 'Early' in stage:
        x_start, x_end = 20, 40
        color = 'red'
    elif 'PHASE 3' in stage or 'Late' in stage:
        x_start, x_end = 60, 80
        color = 'purple'
    elif 'PHASE 2' in stage or 'Consistent' in stage:
        x_start, x_end = 40, 60
        color = 'blue'
    else:
        x_start, x_end = 30, 70
        color = 'gray'

    y_pos = y_positions[idx]

    # Draw bar
    ax.barh(y_pos, x_end - x_start, left=x_start, height=0.08,
           color=color, alpha=0.6, edgecolor='black', linewidth=2)

    # Add label
    ax.text(x_start - 5, y_pos, f"{stage.split(':')[0]}\n({n_proteins} proteins)",
           ha='right', va='center', fontsize=10, fontweight='bold')

# Format
ax.set_xlim(15, 85)
ax.set_ylim(0, 1)
ax.set_xlabel('Age (years)', fontsize=14, fontweight='bold')
ax.set_title('Temporal Staging: Aging Phases with Protein Signatures',
            fontsize=16, fontweight='bold')
ax.set_yticks([])
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add age markers
for age in [20, 30, 40, 50, 60, 70, 80]:
    ax.axvline(age, color='gray', alpha=0.3, linestyle=':')
    ax.text(age, 0.95, str(age), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/aging_timeline.png', dpi=300, bbox_inches='tight')
print("Saved: aging_timeline.png")

# ============================================================================
# SAVE ALL CLUSTERED PROTEINS
# ============================================================================
print("\n" + "="*80)
print("SAVING COMPLETE RESULTS")
print("="*80)

# Add stage names to full dataset
stage_mapping = dict(zip(cluster_stats_df['Cluster'], cluster_stats_df['Stage_Name']))
df_clean['Stage_Name'] = df_clean['Cluster'].map(stage_mapping)

# Save complete clustered dataset
output_file = '/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_10_temporal_staging/clustered_proteins_complete.csv'
df_clean.to_csv(output_file, index=False)
print(f"Saved: clustered_proteins_complete.csv")
print(f"Total proteins with cluster assignments: {len(df_clean)}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal proteins analyzed: {len(df_clean)}")
print(f"Number of clusters: {optimal_k}")
print(f"Optimal clustering quality (Silhouette): {silhouette_scores[optimal_k-2]:.3f}")

print("\nCluster Size Distribution:")
for _, row in cluster_stats_df.iterrows():
    pct = 100 * row['N_Proteins'] / len(df_clean)
    print(f"  {row['Stage_Name']}: {row['N_Proteins']} proteins ({pct:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. cluster_optimization.png - Elbow curve and cluster metrics")
print("  2. dendrogram.png - Hierarchical clustering tree")
print("  3. pca_clusters.png - PCA visualization of clusters")
print("  4. cluster_radar.png - Radar plot of cluster profiles")
print("  5. cluster_heatmap.png - Heatmap of cluster features")
print("  6. aging_timeline.png - Timeline diagram of aging phases")
print("  7. cluster_statistics.csv - Summary statistics per cluster")
print("  8. top_proteins_by_cluster.csv - Top 20 proteins per cluster")
print("  9. clustered_proteins_complete.csv - All proteins with cluster assignments")
