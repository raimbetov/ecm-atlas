#!/usr/bin/env python3
"""UMAP + HDBSCAN clustering on multi-omics features."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from hdbscan import HDBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_19_metabolomics_phase1/codex')
ANALYSES_DIR = ROOT / 'analyses_codex'
VIS_DIR = ROOT / 'visualizations_codex'

PROTEIN_GENES = ['COL1A1', 'COL3A1', 'COL5A1', 'FN1', 'ELN', 'LOX', 'TGM2', 'MMP2', 'MMP9', 'PLOD1', 'COL4A1', 'LAMC1']
METABOLITES = ['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate', 'Lactate/Pyruvate']


def main() -> None:
    np.random.seed(42)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ANALYSES_DIR / 'multiomics_samples_codex.csv')
    df = df[df['is_control'] == False].reset_index(drop=True)
    features = PROTEIN_GENES + METABOLITES

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(0.0))

    reducer = umap.UMAP(random_state=42, n_neighbors=10, min_dist=0.2)
    embedding = reducer.fit_transform(X)

    clusterer = HDBSCAN(min_cluster_size=4, min_samples=1)
    cluster_labels = clusterer.fit_predict(embedding)

    result = df[['dataset_id', 'sample_id', 'tissue', 'phase', 'velocity']].copy()
    result['umap_x'] = embedding[:, 0]
    result['umap_y'] = embedding[:, 1]
    result['cluster'] = cluster_labels
    result_path = ANALYSES_DIR / 'umap_hdbscan_clusters_codex.csv'
    result.to_csv(result_path, index=False)

    plt.figure(figsize=(6, 5))
    unique_clusters = np.unique(cluster_labels)
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        label = f'Cluster {cluster}' if cluster != -1 else 'Noise'
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            label=label,
            alpha=0.7,
            edgecolor='k',
            s=60
        )
    plt.title('UMAP + HDBSCAN Clusters')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'umap_hdbscan_clusters_codex.png', dpi=300)
    plt.close()

    summary = result.groupby('cluster').size().reset_index(name='count')
    summary_path = ANALYSES_DIR / 'umap_hdbscan_cluster_sizes_codex.csv'
    summary.to_csv(summary_path, index=False)


if __name__ == '__main__':
    main()
