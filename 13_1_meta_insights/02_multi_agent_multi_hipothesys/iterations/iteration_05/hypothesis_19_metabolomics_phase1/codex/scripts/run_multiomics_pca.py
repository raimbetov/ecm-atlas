#!/usr/bin/env python3
"""Run joint PCA for proteomics vs multi-omics datasets."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_19_metabolomics_phase1/codex')
ANALYSES_DIR = ROOT / 'analyses_codex'
VIS_DIR = ROOT / 'visualizations_codex'

PROTEIN_GENES = ['COL1A1', 'COL3A1', 'COL5A1', 'FN1', 'ELN', 'LOX', 'TGM2', 'MMP2', 'MMP9', 'PLOD1', 'COL4A1', 'LAMC1']
METABOLITES = ['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate', 'Lactate/Pyruvate']


def standardize(matrix: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(matrix.fillna(0.0)), scaler


def pca_analysis(data: pd.DataFrame, feature_cols: list[str], label: str) -> tuple[PCA, np.ndarray]:
    X, scaler = standardize(data[feature_cols])
    pca = PCA()
    components = pca.fit_transform(X)

    variance_path = ANALYSES_DIR / f'{label}_pca_variance_codex.csv'
    variance_df = pd.DataFrame({
        'component': np.arange(1, len(pca.explained_variance_ratio_) + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
    })
    variance_df.to_csv(variance_path, index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=[f'PC{i}' for i in range(1, len(feature_cols) + 1)]
    )
    loadings_path = ANALYSES_DIR / f'{label}_pc_loadings_codex.csv'
    loadings.to_csv(loadings_path)

    return pca, components


def plot_biplot(components: np.ndarray, data: pd.DataFrame, label: str, feature_cols: list[str], pca: PCA) -> None:
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        components[:, 0],
        components[:, 1],
        c=data['velocity'],
        cmap='viridis',
        edgecolor='k'
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'{label} PCA (PC1 vs PC2)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocity')

    # Plot top loading vectors for interpretability
    vectors = pca.components_[:2, :]
    for idx, feature in enumerate(feature_cols):
        loading = vectors[:, idx]
        plt.arrow(0, 0, loading[0] * 2, loading[1] * 2, color='r', alpha=0.4)
        plt.text(loading[0] * 2.2, loading[1] * 2.2, feature, color='r', fontsize=8)

    plt.tight_layout()
    out_path = VIS_DIR / f'{label}_pca_biplot_codex.png'
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ANALYSES_DIR / 'multiomics_samples_codex.csv')
    data = data[data['is_control'] == False].reset_index(drop=True)

    # Proteomics-only PCA
    proteomics_label = 'proteomics_only'
    proteomics_pca, proteomics_components = pca_analysis(data, PROTEIN_GENES, proteomics_label)
    plot_biplot(proteomics_components, data, proteomics_label, PROTEIN_GENES, proteomics_pca)

    # Multi-omics PCA
    multi_cols = PROTEIN_GENES + METABOLITES
    multi_label = 'multi_omics'
    multi_pca, multi_components = pca_analysis(data, multi_cols, multi_label)
    plot_biplot(multi_components, data, multi_label, multi_cols, multi_pca)

    # Variance comparison table
    proteomics_var = pd.read_csv(ANALYSES_DIR / f'{proteomics_label}_pca_variance_codex.csv')
    multi_var = pd.read_csv(ANALYSES_DIR / f'{multi_label}_pca_variance_codex.csv')
    comparison = pd.DataFrame({
        'model': ['proteomics_only', 'multi_omics'],
        'variance_pc1': [proteomics_var['explained_variance_ratio'].iloc[0], multi_var['explained_variance_ratio'].iloc[0]],
        'variance_pc1_pc2': [proteomics_var['cumulative_variance'].iloc[1], multi_var['cumulative_variance'].iloc[1]],
        'variance_cumulative_95pc': [
            proteomics_var[proteomics_var['cumulative_variance'] >= 0.95]['component'].min(),
            multi_var[multi_var['cumulative_variance'] >= 0.95]['component'].min(),
        ]
    })
    comparison_path = ANALYSES_DIR / 'multi_omics_pca_variance_codex.csv'
    comparison.to_csv(comparison_path, index=False)

    # Variance bar plot
    plt.figure(figsize=(6, 4))
    width = 0.35
    x = np.arange(2)
    plt.bar(x - width/2, comparison['variance_pc1_pc2'], width=width, label='PC1+PC2 cumulative')
    plt.bar(x + width/2, [0.95, 0.95], width=width, color='gray', alpha=0.3, label='95% target')
    plt.xticks(x, comparison['model'])
    plt.ylabel('Variance explained')
    plt.ylim(0, 1.05)
    plt.title('Variance Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'variance_comparison_codex.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
