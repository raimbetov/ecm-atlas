#!/usr/bin/env python3
"""
ML AGENT 12: DIMENSIONALITY REDUCTION & CLUSTERING
===================================================

Mission: Find hidden structure in ECM aging data using PCA, t-SNE, UMAP
and cluster analysis to reveal protein subpopulations and meta-patterns.

Approach:
1. PCA for variance explanation
2. t-SNE for visualization
3. UMAP for topology-preserving embedding
4. K-means and hierarchical clustering
5. Identify archetypal aging patterns
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML AGENT 12: DIMENSIONALITY REDUCTION & PATTERN DISCOVERY")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
print(f"Total records: {len(df):,}")

# Create protein-tissue matrix for dimensionality reduction
print("\n" + "=" * 80)
print("PREPARING DATA MATRIX")
print("=" * 80)

# Pivot: rows = proteins, columns = tissue_compartments, values = z-score deltas
pivot = df[df['Zscore_Delta'].notna()].pivot_table(
    index='Gene_Symbol',
    columns='Tissue_Compartment',
    values='Zscore_Delta',
    aggfunc='mean'
)

print(f"Matrix shape: {pivot.shape[0]} proteins × {pivot.shape[1]} tissue compartments")
print(f"Missing data: {pivot.isna().sum().sum()} / {pivot.size} ({pivot.isna().sum().sum() / pivot.size * 100:.1f}%)")

# Fill missing with 0 (no change)
pivot_filled = pivot.fillna(0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot_filled)

print("\n✅ Data prepared and scaled")

# TASK 1: PCA Analysis
print("\n" + "=" * 80)
print("TASK 1: PCA - PRINCIPAL COMPONENT ANALYSIS")
print("=" * 80)

pca = PCA(n_components=min(10, X_scaled.shape[1]))
X_pca = pca.fit_transform(X_scaled)

print(f"\n📊 Explained Variance by Component:")
for i, var in enumerate(pca.explained_variance_ratio_[:10]):
    cumsum = pca.explained_variance_ratio_[:i+1].sum()
    print(f"  PC{i+1}: {var:.3f} ({var*100:.1f}%) | Cumulative: {cumsum:.3f} ({cumsum*100:.1f}%)")

# Find proteins with highest loadings on PC1 and PC2
pc1_loadings = pd.DataFrame({
    'Protein': pivot.index,
    'PC1_Loading': pca.components_[0],
    'PC2_Loading': pca.components_[1] if pca.n_components_ > 1 else 0
}).sort_values('PC1_Loading', key=abs, ascending=False)

print("\n🏆 TOP 15 PROTEINS DRIVING PC1 (Main Aging Direction):")
for i, row in pc1_loadings.head(15).iterrows():
    direction = "↑" if row['PC1_Loading'] > 0 else "↓"
    print(f"  {row['Protein']:15s} {direction} Loading: {row['PC1_Loading']:+.4f}")

# TASK 2: K-means Clustering
print("\n" + "=" * 80)
print("TASK 2: K-MEANS CLUSTERING")
print("=" * 80)

# Find optimal k using silhouette score
silhouette_scores = []
K_range = range(2, 11)

print("\n🔍 Testing cluster numbers k=2 to k=10...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"  k={k}: silhouette={score:.3f}")

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✅ Optimal k: {optimal_k} (silhouette={max(silhouette_scores):.3f})")

# Apply optimal clustering
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans_optimal.fit_predict(X_scaled)

# Analyze clusters
cluster_analysis = pd.DataFrame({
    'Protein': pivot.index,
    'Cluster': cluster_labels
})

# Add average z-score delta per protein
avg_delta = df.groupby('Gene_Symbol')['Zscore_Delta'].mean()
cluster_analysis['Avg_Zscore_Delta'] = cluster_analysis['Protein'].map(avg_delta)

# Add matrisome category
protein_to_category = df.groupby('Gene_Symbol')['Matrisome_Category'].first()
cluster_analysis['Category'] = cluster_analysis['Protein'].map(protein_to_category)

print(f"\n📊 CLUSTER CHARACTERISTICS:")
for cluster_id in range(optimal_k):
    cluster_proteins = cluster_analysis[cluster_analysis['Cluster'] == cluster_id]
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id} (n={len(cluster_proteins)})")
    print(f"{'='*60}")
    print(f"Mean z-score delta: {cluster_proteins['Avg_Zscore_Delta'].mean():+.3f}")
    print(f"Std: {cluster_proteins['Avg_Zscore_Delta'].std():.3f}")

    # Top categories
    top_cats = cluster_proteins['Category'].value_counts().head(3)
    print(f"\nTop categories:")
    for cat, count in top_cats.items():
        print(f"  - {cat}: {count} proteins")

    # Sample proteins
    samples = cluster_proteins.nlargest(5, 'Avg_Zscore_Delta', keep='first')
    print(f"\nExample proteins (highest Δz):")
    for _, p in samples.iterrows():
        print(f"  - {p['Protein']:15s} (Δz={p['Avg_Zscore_Delta']:+.2f}, {p['Category']})")

cluster_analysis.to_csv('10_insights/ml_protein_clusters.csv', index=False)
print("\n✅ Saved: 10_insights/ml_protein_clusters.csv")

# TASK 3: Hierarchical Clustering to find archetypal patterns
print("\n" + "=" * 80)
print("TASK 3: HIERARCHICAL CLUSTERING - AGING ARCHETYPES")
print("=" * 80)

hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
hier_labels = hierarchical.fit_predict(X_scaled)

hier_analysis = pd.DataFrame({
    'Protein': pivot.index,
    'Archetype': hier_labels
})
hier_analysis['Avg_Zscore_Delta'] = hier_analysis['Protein'].map(avg_delta)

print("\n🎭 AGING ARCHETYPES:")
archetype_names = {
    0: "Strong Upregulation",
    1: "Moderate Upregulation",
    2: "Stable",
    3: "Moderate Downregulation",
    4: "Strong Downregulation"
}

# Sort clusters by mean delta
archetype_means = hier_analysis.groupby('Archetype')['Avg_Zscore_Delta'].mean().sort_values(ascending=False)
archetype_order = archetype_means.index.tolist()

for rank, archetype_id in enumerate(archetype_order):
    arch_proteins = hier_analysis[hier_analysis['Archetype'] == archetype_id]
    mean_delta = arch_proteins['Avg_Zscore_Delta'].mean()
    print(f"\nArchetype {archetype_id} (n={len(arch_proteins)}): Mean Δz = {mean_delta:+.3f}")

    # Top proteins
    top5 = arch_proteins.nlargest(5, 'Avg_Zscore_Delta', keep='first')
    for _, p in top5.iterrows():
        print(f"  - {p['Protein']}")

hier_analysis.to_csv('10_insights/ml_aging_archetypes.csv', index=False)
print("\n✅ Saved: 10_insights/ml_aging_archetypes.csv")

# TASK 4: DBSCAN for outlier detection
print("\n" + "=" * 80)
print("TASK 4: DBSCAN OUTLIER DETECTION")
print("=" * 80)

dbscan = DBSCAN(eps=2.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers = list(dbscan_labels).count(-1)

print(f"\n🔍 DBSCAN Results:")
print(f"  Clusters found: {n_clusters}")
print(f"  Outliers: {n_outliers} proteins")

if n_outliers > 0:
    outlier_proteins = pivot.index[dbscan_labels == -1]
    print(f"\n⚠️  OUTLIER PROTEINS (unique aging patterns):")
    for protein in outlier_proteins[:20]:
        delta = avg_delta.get(protein, 0)
        print(f"  - {protein:15s} (Δz={delta:+.3f})")

# TASK 5: Summary statistics
print("\n" + "=" * 80)
print("🎯 KEY DIMENSIONALITY REDUCTION INSIGHTS")
print("=" * 80)

print(f"""
1. PCA: First 3 components explain {pca.explained_variance_ratio_[:3].sum():.1%} of variance
2. Top PC1 driver: {pc1_loadings.iloc[0]['Protein']} (loading={pc1_loadings.iloc[0]['PC1_Loading']:.3f})
3. Optimal clusters: k={optimal_k} (silhouette={max(silhouette_scores):.3f})
4. DBSCAN outliers: {n_outliers} proteins with unique patterns
5. Aging archetypes: 5 distinct trajectories identified

📊 Data Compression:
   - Original: {X_scaled.shape[1]} dimensions
   - PCA (90% var): {np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0] + 1} components
   - Information preserved: {pca.explained_variance_ratio_[:5].sum():.1%}
""")

print("\n✅ ML AGENT 12 COMPLETED")
print("=" * 80)
