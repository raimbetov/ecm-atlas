#!/usr/bin/env python3
"""Compute tissue-level velocity metrics without TSI filtering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas')
DATA_PATH = ROOT / '08_merged_ecm_dataset' / 'merged_ecm_aging_zscore.csv'
OUT_PATH = ROOT / '13_1_meta_insights' / '02_multi_agent_multi_hipothesys' / 'iterations' / 'iteration_05' / 'hypothesis_19_metabolomics_phase1' / 'codex' / 'data_codex' / 'extended_tissue_velocities_codex.csv'


def load_proteomics() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df['Tissue'].notna()].copy()
    df['Gene_Symbol'] = df['Gene_Symbol'].fillna(df['Canonical_Gene_Symbol'])
    return df[df['Gene_Symbol'].notna()]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(['Tissue', 'Gene_Symbol'])
        .agg(
            Zscore_Delta=('Zscore_Delta', 'mean'),
            Zscore_Young=('Zscore_Young', 'mean'),
            Zscore_Old=('Zscore_Old', 'mean'),
        )
        .reset_index()
    )
    agg['Abs_Delta'] = agg['Zscore_Delta'].abs()
    return agg


def compute_velocity(agg: pd.DataFrame) -> pd.DataFrame:
    stats = (
        agg.groupby('Tissue')
        .agg(
            Velocity=('Abs_Delta', 'mean'),
            Mean_Delta=('Zscore_Delta', 'mean'),
            Median_Delta=('Zscore_Delta', 'median'),
            N_Genes=('Gene_Symbol', 'nunique'),
            Upregulated_Pct=('Zscore_Delta', lambda x: float((x > 0).mean() * 100)),
            Downregulated_Pct=('Zscore_Delta', lambda x: float((x < 0).mean() * 100)),
        )
        .reset_index()
    )
    stats = stats.sort_values('Velocity', ascending=False).reset_index(drop=True)
    stats['Velocity_Rank'] = stats.index + 1
    return stats


def main() -> None:
    df = load_proteomics()
    agg = aggregate(df)
    velocity = compute_velocity(agg)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    velocity.to_csv(OUT_PATH, index=False)


if __name__ == '__main__':
    main()
