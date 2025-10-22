#!/usr/bin/env python3
"""Metabolomics Phase I validation pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_19_metabolomics_phase1/codex')
DATA_DIR = ROOT / 'data_codex'
ANALYSIS_DIR = ROOT / 'analyses_codex'
VIS_DIR = ROOT / 'visualizations_codex'

CANONICAL_SYNONYMS: Dict[str, List[str]] = {
    'ATP': ['ATP', 'ADENOSINE TRIPHOSPHATE'],
    'NAD+': ['NAD+', 'NAD', 'BETA-NAD+', 'NAD (OX)', 'NICOTINAMIDE ADENINE DINUCLEOTIDE'],
    'NADH': ['NADH', 'NAD (RED)'],
    'Lactate': ['Lactate', 'LACTIC ACID', 'L-LACTATE'],
    'Pyruvate': ['Pyruvate', 'PYRUVIC ACID'],
}

RATIO_METABOLITE = 'Lactate/Pyruvate'
TARGET_METABOLITES = list(CANONICAL_SYNONYMS.keys()) + [RATIO_METABOLITE]

PHASE_LABELS = {
    'Phase I': (0.0, 1.65),
    'Transition': (1.65, 2.17),
    'Phase II': (2.17, float('inf')),
}


def infer_phase(velocity: float) -> str:
    for phase, (low, high) in PHASE_LABELS.items():
        if low <= velocity < high:
            return phase
    return 'Unknown'


def canonical_name(raw: str) -> Optional[str]:
    name_upper = raw.upper()
    for canonical, synonyms in CANONICAL_SYNONYMS.items():
        if any(name_upper == syn.upper() for syn in synonyms):
            return canonical
    # loose matching
    for canonical, synonyms in CANONICAL_SYNONYMS.items():
        if any(syn.upper() in name_upper for syn in synonyms):
            return canonical
    return None


@dataclass
class DatasetConfig:
    dataset_id: str
    tissue: str
    velocity: float
    sample_filter: Optional[str] = None  # pandas query string
    tissue_column: Optional[str] = None
    additional_velocity_map: Optional[Dict[str, float]] = None
    control_filter: Optional[str] = None

    def phase(self, subgroup: Optional[str] = None) -> str:
        if self.additional_velocity_map and subgroup:
            vel = self.additional_velocity_map.get(subgroup, self.velocity)
            return infer_phase(vel)
        return infer_phase(self.velocity)


DATASETS: Dict[str, DatasetConfig] = {
    'ST001329': DatasetConfig(
        dataset_id='ST001329',
        tissue='Heart_Native_Tissue',
        velocity=1.6346478411541698,
        sample_filter="Treatment == 'None'",
        control_filter="`Age_(Months)` == '5-6'",
    ),
    'ST000828': DatasetConfig(
        dataset_id='ST000828',
        tissue='Lung',
        velocity=4.162418661183056,
        sample_filter="`Species` == 'Homo sapiens'",
        control_filter="`bleomycin/IPF` == 'N'",
    ),
    'ST002058': DatasetConfig(
        dataset_id='ST002058',
        tissue='Skeletal_muscle_TA',
        velocity=1.9552397727644706,
        sample_filter="Tissue == 'Tibialis_anterior'",
        control_filter="Treatment == 'Control'",
    ),
}


def load_dataset(config: DatasetConfig) -> pd.DataFrame:
    long_path = DATA_DIR / f"{config.dataset_id}_metabolites_long.csv"
    factors_path = DATA_DIR / f"{config.dataset_id}_sample_factors.csv"

    df = pd.read_csv(long_path)
    factors = pd.read_csv(factors_path)

    # Standardize column names and normalize treatment labels
    factors.columns = [c.replace(' ', '_') for c in factors.columns]
    if 'Treatment' in factors.columns:
        factors['Treatment'] = factors['Treatment'].fillna('None')

    factors['is_control'] = False
    if config.control_filter:
        control_mask = factors.eval(config.control_filter)
        factors.loc[control_mask, 'is_control'] = True

    if config.sample_filter:
        factors = factors.query(config.sample_filter)

    df = df[df['sample_id'].isin(factors['sample_id'])].copy()
    df['canonical_metabolite'] = df['metabolite'].map(canonical_name)
    df = df[df['canonical_metabolite'].isin(TARGET_METABOLITES)].copy()

    df = df.merge(factors, on='sample_id', how='left', suffixes=('', '_factor'))
    df['dataset_id'] = config.dataset_id

    # Assign velocity and phase
    if config.additional_velocity_map and config.tissue_column:
        df['tissue_subgroup'] = df[config.tissue_column]
        df['velocity'] = df['tissue_subgroup'].map(config.additional_velocity_map).fillna(config.velocity)
    else:
        df['velocity'] = config.velocity
    df['phase'] = df['velocity'].apply(infer_phase)
    df['tissue'] = config.tissue
    return df


def add_normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['log_value'] = np.log1p(df['value'])

    control_log = (
        df[df['is_control']]
        .groupby(['dataset_id', 'canonical_metabolite'])['log_value']
        .mean()
    )
    control_raw = (
        df[df['is_control']]
        .groupby(['dataset_id', 'canonical_metabolite'])['value']
        .mean()
    )

    def lookup(series, key):
        return series.get(key, np.nan)

    df['control_log'] = df.apply(
        lambda row: lookup(control_log, (row['dataset_id'], row['canonical_metabolite'])), axis=1
    )
    df['control_raw'] = df.apply(
        lambda row: lookup(control_raw, (row['dataset_id'], row['canonical_metabolite'])), axis=1
    )

    df['delta_log'] = df['log_value'] - df['control_log']
    df.loc[df['is_control'], 'delta_log'] = 0.0

    df['percent_change'] = np.where(
        df['control_raw'] > 0,
        (df['value'] - df['control_raw']) / df['control_raw'] * 100,
        np.nan,
    )
    df.loc[df['is_control'], 'percent_change'] = 0.0

    df['zscore_delta'] = (
        df.groupby(['dataset_id', 'canonical_metabolite'])['delta_log']
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0.0)
    )
    return df


def append_ratio_feature(df: pd.DataFrame) -> pd.DataFrame:
    ratio_rows = []
    required = {'Lactate', 'Pyruvate'}
    grouping_cols = ['dataset_id', 'sample_id', 'tissue', 'phase', 'velocity', 'is_control']
    for key, group in df.groupby(grouping_cols):
        metabolites = set(group['canonical_metabolite'])
        if not required.issubset(metabolites):
            continue
        lac = group[group['canonical_metabolite'] == 'Lactate'].iloc[0]
        pyr = group[group['canonical_metabolite'] == 'Pyruvate'].iloc[0]
        if pyr['value'] == 0 or pyr['control_raw'] == 0:
            continue
        ratio_value = lac['value'] / pyr['value']
        ratio_control = lac['control_raw'] / pyr['control_raw']
        ratio_log = math.log1p(ratio_value)
        ratio_control_log = math.log1p(ratio_control)
        delta_log = ratio_log - ratio_control_log
        percent_change = ((ratio_value - ratio_control) / ratio_control * 100) if ratio_control != 0 else np.nan
        ratio_rows.append({
            'dataset_id': key[0],
            'sample_id': key[1],
            'tissue': key[2],
            'phase': key[3],
            'velocity': key[4],
            'is_control': key[5],
            'canonical_metabolite': RATIO_METABOLITE,
            'metabolite': RATIO_METABOLITE,
            'value': ratio_value,
            'control_raw': ratio_control,
            'log_value': ratio_log,
            'control_log': ratio_control_log,
            'delta_log': delta_log,
            'percent_change': percent_change,
            'zscore_delta': np.nan,
        })
    if not ratio_rows:
        return df
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df['zscore_delta'] = (
        ratio_df.groupby('dataset_id')['delta_log']
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else 0.0)
    )
    return pd.concat([df, ratio_df], ignore_index=True)


def summarise_phase(df: pd.DataFrame, value_col: str = 'delta_log') -> pd.DataFrame:
    summary = (
        df.groupby(['tissue', 'phase', 'canonical_metabolite'])[value_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': f'{value_col}_mean', 'std': f'{value_col}_std', 'count': 'n'})
    )
    return summary


def phase_comparison(summary: pd.DataFrame, value_col: str = 'delta_log_mean') -> pd.DataFrame:
    piv = summary.pivot_table(
        index='canonical_metabolite', columns='phase', values=value_col
    )
    result_rows = []
    if 'Phase I' in piv and 'Phase II' in piv:
        for metabolite, row in piv.iterrows():
            phase1 = row.get('Phase I')
            phase2 = row.get('Phase II')
            if pd.notna(phase1) and pd.notna(phase2):
                delta = phase1 - phase2
                pct_change = np.nan
                if phase2 != 0:
                    pct_change = (phase1 - phase2) / abs(phase2) * 100
                result_rows.append({
                    'metabolite': metabolite,
                    'phase1_z': phase1,
                    'phase2_z': phase2,
                    'delta_metric': delta,
                    'pct_change_vs_phase2': pct_change,
                })
    return pd.DataFrame(result_rows)


def compute_correlations(df: pd.DataFrame, value_col: str = 'delta_log') -> pd.DataFrame:
    records = []
    analysis_df = df[~df['is_control']]
    for metabolite in TARGET_METABOLITES:
        sub = analysis_df[analysis_df['canonical_metabolite'] == metabolite]
        if sub['velocity'].nunique() < 2:
            continue
        corr, pval = spearmanr(sub[value_col], sub['velocity'])
        records.append({
            'metabolite': metabolite,
            'spearman_rho': corr,
            'p_value': pval,
            'n_samples': len(sub),
        })
    return pd.DataFrame(records)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    frames = [load_dataset(cfg) for cfg in DATASETS.values()]
    df = pd.concat(frames, ignore_index=True)
    df = add_normalized_metrics(df)
    df = append_ratio_feature(df)

    combined_path = ANALYSIS_DIR / 'metabolomics_combined_codex.csv'
    df.to_csv(combined_path, index=False)

    summary_delta = summarise_phase(df, value_col='delta_log')
    summary_pct = summarise_phase(df, value_col='percent_change')
    summary = summary_delta.merge(
        summary_pct,
        on=['tissue', 'phase', 'canonical_metabolite', 'n'],
        how='outer',
        suffixes=('_delta', '_pct')
    )
    summary_path = ANALYSIS_DIR / 'metabolite_phase_summary_codex.csv'
    summary.to_csv(summary_path, index=False)

    comparison_delta = phase_comparison(summary_delta, value_col='delta_log_mean')
    comparison_pct = phase_comparison(summary_pct, value_col='percent_change_mean')
    comparison = comparison_delta.merge(
        comparison_pct,
        on='metabolite',
        how='outer',
        suffixes=('_delta', '_pct')
    )
    comparison_path = ANALYSIS_DIR / 'phase1_vs_phase2_metabolites_codex.csv'
    comparison.to_csv(comparison_path, index=False)

    correlations = compute_correlations(df, value_col='percent_change')
    correlation_path = ANALYSIS_DIR / 'metabolite_velocity_correlations_codex.csv'
    correlations.to_csv(correlation_path, index=False)

    print('Saved combined metabolomics dataset to', combined_path)
    print('Saved phase summary to', summary_path)
    print('Saved phase comparison to', comparison_path)
    print('Saved correlations to', correlation_path)


if __name__ == '__main__':
    main()
