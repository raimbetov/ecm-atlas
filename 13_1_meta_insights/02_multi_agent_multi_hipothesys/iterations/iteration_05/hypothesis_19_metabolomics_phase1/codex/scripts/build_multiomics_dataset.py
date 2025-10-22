#!/usr/bin/env python3
"""Construct multi-omics feature matrix for metabolomics + proteomics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas')
ITER_ROOT = ROOT / '13_1_meta_insights' / '02_multi_agent_multi_hipothesys' / 'iterations' / 'iteration_05' / 'hypothesis_19_metabolomics_phase1' / 'codex'
ANALYSES_DIR = ITER_ROOT / 'analyses_codex'
DATA_DIR = ROOT / '08_merged_ecm_dataset'

PROTEIN_GENES: List[str] = [
    'COL1A1', 'COL3A1', 'COL5A1', 'FN1', 'ELN', 'LOX', 'TGM2',
    'MMP2', 'MMP9', 'PLOD1', 'COL4A1', 'LAMC1'
]


def load_metabolomics() -> pd.DataFrame:
    df = pd.read_csv(ANALYSES_DIR / 'metabolomics_combined_codex.csv')
    return df


def build_metabolite_features(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index=['dataset_id', 'sample_id', 'tissue', 'phase', 'velocity', 'is_control'],
            columns='canonical_metabolite',
            values='percent_change',
            aggfunc='mean'
        )
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def load_proteomics() -> pd.DataFrame:
    proteomics = pd.read_csv(DATA_DIR / 'merged_ecm_aging_zscore.csv')
    proteomics = proteomics[proteomics['Gene_Symbol'].isin(PROTEIN_GENES)]
    agg = (
        proteomics.groupby(['Tissue', 'Gene_Symbol'])['Zscore_Delta']
        .mean()
        .unstack(fill_value=0.0)
        .reset_index()
    )
    return agg


def main() -> None:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    metab = load_metabolomics()
    metab_features = build_metabolite_features(metab)

    proteomics = load_proteomics()
    feature_df = metab_features.merge(
        proteomics,
        left_on='tissue',
        right_on='Tissue',
        how='left'
    )
    feature_df = feature_df.drop(columns=['Tissue'])

    # Impute missing proteomics values with 0 (no delta)
    for gene in PROTEIN_GENES:
        if gene not in feature_df.columns:
            feature_df[gene] = 0.0
    gene_cols = PROTEIN_GENES
    feature_df[gene_cols] = feature_df[gene_cols].fillna(0.0)

    # Save per-sample features
    sample_path = ANALYSES_DIR / 'multiomics_samples_codex.csv'
    feature_df.to_csv(sample_path, index=False)

    # Aggregate by tissue-phase for downstream PCA summaries
    agg = (
        feature_df.groupby(['tissue', 'phase', 'velocity', 'is_control'])
        .mean(numeric_only=True)
        .reset_index()
    )
    agg_path = ANALYSES_DIR / 'multiomics_tissue_phase_codex.csv'
    agg.to_csv(agg_path, index=False)


if __name__ == '__main__':
    main()
