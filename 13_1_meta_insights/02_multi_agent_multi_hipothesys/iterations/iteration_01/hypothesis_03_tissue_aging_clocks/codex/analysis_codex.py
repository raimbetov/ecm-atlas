#!/usr/bin/env python3
"""Tissue-specific ECM aging velocity analysis for hypothesis H03."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas')
DATA_PATH = ROOT / '08_merged_ecm_dataset' / 'merged_ecm_aging_zscore.csv'
OUTPUT_DIR = ROOT / '13_1_meta_insights' / '02_multi_agent_multi_hipothesys' / 'iterations' / 'iteration_01' / 'hypothesis_03_tissue_aging_clocks' / 'codex'
VIS_DIR = OUTPUT_DIR / 'visualizations_codex'

RNG = np.random.default_rng(42)

STRUCTURAL = {"Collagens", "ECM Glycoproteins", "Proteoglycans"}
REGULATORY = {"ECM Regulators"}
SIGNALING = {"Secreted Factors", "ECM-affiliated Proteins"}

PATHWAY_KEYWORDS = {
    "Inflammation": ("IL", "CXCL", "CCL", "TNF", "SAA", "S100", "NFKB", "IFN", "ICAM", "VCAM"),
    "Oxidative_Stress": ("SOD", "GPX", "PRDX", "CAT", "TXN", "NQO"),
    "Coagulation": ("FGA", "FGB", "FGG", "VWF", "SERP", "PLG"),
    "Matrix_Remodeling": ("MMP", "ADAM", "ADAMTS", "LOX", "PLOD", "TIMP"),
}

INFLAMMATORY_PATTERN = r"^(?:IL|CXCL|CCL|TNF|SAA|S100|IFN|NFKB|ICAM|VCAM)"


def load_data() -> pd.DataFrame:
    """Load ECM aging dataset and retain core columns."""
    df = pd.read_csv(DATA_PATH)
    df = df[df['Tissue'].notna()].copy()
    df['Gene_Symbol'] = df['Gene_Symbol'].fillna(df['Canonical_Gene_Symbol'])
    df = df[df['Gene_Symbol'].notna()]
    return df


def aggregate_tissue_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate records by tissue and gene."""
    def mode_first(series: pd.Series) -> str:
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else series.iloc[0]

    agg = (
        df.groupby(['Tissue', 'Gene_Symbol'])
        .agg(
            Zscore_Delta=('Zscore_Delta', 'mean'),
            Zscore_Young=('Zscore_Young', 'mean'),
            Zscore_Old=('Zscore_Old', 'mean'),
            Matrisome_Category_Simplified=('Matrisome_Category_Simplified', mode_first),
        )
        .reset_index()
    )
    return agg


def classify_function(category: str) -> str:
    if category in STRUCTURAL:
        return 'Structural'
    if category in REGULATORY:
        return 'Regulatory'
    if category in SIGNALING:
        return 'Signaling'
    return 'Unknown'


def compute_tsi(agg: pd.DataFrame) -> pd.DataFrame:
    """Compute tissue specificity index using |Zscore_Delta| across tissues."""
    pivot = agg.pivot(index='Gene_Symbol', columns='Tissue', values='Zscore_Delta')
    records: List[Dict[str, object]] = []

    for gene, row in pivot.iterrows():
        row_non_na = row.dropna()
        if row_non_na.empty:
            continue
        abs_row = row_non_na.abs()
        top_tissue = abs_row.idxmax()
        max_val = abs_row.loc[top_tissue]
        others = abs_row.drop(index=top_tissue)
        if others.empty:
            tsi = np.nan
        else:
            denom = others.std(ddof=0)
            if np.isnan(denom) or denom == 0:
                denom = 1.0
            tsi = (max_val - others.mean()) / denom

        match = agg[(agg['Gene_Symbol'] == gene) & (agg['Tissue'] == top_tissue)]
        if match.empty:
            continue
        row_match = match.iloc[0]
        records.append(
            {
                'Gene_Symbol': gene,
                'Tissue': top_tissue,
                'TSI': tsi,
                'Zscore_Delta': row_match['Zscore_Delta'],
                'Zscore_Young': row_match['Zscore_Young'],
                'Zscore_Old': row_match['Zscore_Old'],
                'Matrisome_Category_Simplified': row_match['Matrisome_Category_Simplified'],
                'Function_Category': classify_function(row_match['Matrisome_Category_Simplified']),
                'Direction': 'Upregulated' if row_match['Zscore_Delta'] > 0 else 'Downregulated' if row_match['Zscore_Delta'] < 0 else 'Neutral',
            }
        )

    tsi_df = pd.DataFrame.from_records(records)
    tsi_df = tsi_df.dropna(subset=['TSI'])
    tsi_df['Rank_within_Tissue'] = tsi_df.groupby('Tissue')['TSI'].rank(ascending=False, method='dense').astype(int)
    tsi_df = tsi_df.sort_values(['Tissue', 'Rank_within_Tissue'])
    return tsi_df


def bootstrap_ci(data: np.ndarray, ci: float = 95, n_bootstrap: int = 2000) -> Tuple[float, float]:
    if data.size == 0:
        return (np.nan, np.nan)
    samples = RNG.choice(data, size=(n_bootstrap, data.size), replace=True)
    means = samples.mean(axis=1)
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return float(lower), float(upper)


def compute_velocity(markers: pd.DataFrame) -> pd.DataFrame:
    """Compute tissue aging velocity metrics from tissue-specific markers."""
    markers = markers.copy()
    markers['Abs_Delta'] = markers['Zscore_Delta'].abs()

    stats = (
        markers.groupby('Tissue')
        .agg(
            Mean_Delta=('Zscore_Delta', 'mean'),
            Velocity=('Abs_Delta', 'mean'),
            N_Markers=('Gene_Symbol', 'count'),
            Upregulated_Pct=('Zscore_Delta', lambda x: float((x > 0).mean() * 100)),
            Downregulated_Pct=('Zscore_Delta', lambda x: float((x < 0).mean() * 100)),
        )
        .reset_index()
    )

    cis = markers.groupby('Tissue')['Abs_Delta'].apply(lambda x: bootstrap_ci(x.to_numpy()))
    stats['Bootstrap_CI'] = stats['Tissue'].map(lambda t: cis.get(t, (np.nan, np.nan)))
    stats['Bootstrap_CI'] = stats['Bootstrap_CI'].apply(lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]" if not any(math.isnan(x) for x in ci) else 'NA')
    stats = stats.sort_values('Velocity', ascending=False).reset_index(drop=True)
    stats['Velocity_Rank'] = stats.index + 1
    return stats


def identify_fast_tissues(velocity_df: pd.DataFrame) -> List[str]:
    n_tissues = len(velocity_df)
    n_fast = max(1, math.ceil(n_tissues * 0.33))
    return velocity_df.sort_values('Velocity', ascending=False).head(n_fast)['Tissue'].tolist()


def annotate_pathways(gene: str) -> str:
    matches: List[str] = []
    for pathway, prefixes in PATHWAY_KEYWORDS.items():
        for prefix in prefixes:
            if gene.upper().startswith(prefix):
                matches.append(pathway)
                break
    return ';'.join(sorted(set(matches))) if matches else 'Other'


def shared_fast_proteins(agg: pd.DataFrame, fast_tissues: Iterable[str]) -> pd.DataFrame:
    fast_df = agg[agg['Tissue'].isin(fast_tissues)].copy()
    counts = fast_df.groupby('Gene_Symbol')['Tissue'].nunique()
    shared_genes = counts[counts >= 2].index
    shared = fast_df[fast_df['Gene_Symbol'].isin(shared_genes)]

    if shared.empty:
        return pd.DataFrame(columns=['Gene_Symbol', 'Shared_Tissue_Count', 'Fast_Tissues', 'Mean_Zscore_Delta', 'Direction', 'Function_Category', 'Pathway_Annotation'])

    records = []
    for gene, group in shared.groupby('Gene_Symbol'):
        tissues = sorted(group['Tissue'].unique())
        mean_delta = group['Zscore_Delta'].mean()
        direction = 'Upregulated' if mean_delta > 0 else 'Downregulated'
        categories = group['Matrisome_Category_Simplified'].mode()
        category = categories.iloc[0] if not categories.empty else group['Matrisome_Category_Simplified'].iloc[0]
        records.append(
            {
                'Gene_Symbol': gene,
                'Shared_Tissue_Count': len(tissues),
                'Fast_Tissues': ', '.join(tissues),
                'Mean_Zscore_Delta': mean_delta,
                'Direction': direction,
                'Function_Category': classify_function(category),
                'Pathway_Annotation': annotate_pathways(gene),
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary = summary.sort_values(['Shared_Tissue_Count', 'Mean_Zscore_Delta'], ascending=[False, False])
    return summary


def inflammation_signature(df: pd.DataFrame, fast_tissues: Iterable[str]) -> Dict[str, object]:
    mask_inflam = df['Gene_Symbol'].str.contains(INFLAMMATORY_PATTERN, case=False, na=False)
    fast_mask = df['Tissue'].isin(fast_tissues)

    fast_values = df.loc[mask_inflam & fast_mask, 'Zscore_Delta'].dropna()
    slow_values = df.loc[mask_inflam & ~fast_mask, 'Zscore_Delta'].dropna()

    if fast_values.empty or slow_values.empty:
        stat, pvalue = (np.nan, np.nan)
    else:
        stat, pvalue = mannwhitneyu(fast_values, slow_values, alternative='two-sided')

    return {
        'fast_mean': fast_values.mean() if not fast_values.empty else np.nan,
        'slow_mean': slow_values.mean() if not slow_values.empty else np.nan,
        'statistic': stat,
        'pvalue': pvalue,
        'n_fast': len(fast_values),
        'n_slow': len(slow_values),
        'fast_values': fast_values,
        'slow_values': slow_values,
    }


def save_velocity_plot(velocity_df: pd.DataFrame) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, max(6, 0.4 * len(velocity_df))))
    sns.barplot(data=velocity_df, x='Velocity', y='Tissue', hue='Tissue', dodge=False, palette='viridis', legend=False)
    plt.title('Tissue Aging Velocity (Mean |Δz|)')
    plt.xlabel('Velocity (Mean |Δz|)')
    plt.ylabel('Tissue')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'codex_velocity_bar.png', dpi=300)
    plt.close()


def save_marker_heatmap(markers: pd.DataFrame) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    top_markers = markers.sort_values('TSI', ascending=False).groupby('Tissue').head(10)
    if top_markers.empty:
        return
    pivot = top_markers.pivot(index='Gene_Symbol', columns='Tissue', values='Zscore_Delta').fillna(0)
    plt.figure(figsize=(1.2 * len(pivot.columns) + 4, 0.4 * len(pivot) + 4))
    sns.heatmap(pivot, cmap='coolwarm', center=0, annot=False)
    plt.title('Δz Heatmap for Top Tissue-Specific Markers')
    plt.xlabel('Tissue')
    plt.ylabel('Gene Symbol')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'codex_marker_heatmap.png', dpi=300)
    plt.close()


def save_inflammation_boxplot(summary: Dict[str, object]) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fast_values = summary['fast_values']
    slow_values = summary['slow_values']
    if fast_values.empty or slow_values.empty:
        return
    plot_df = pd.DataFrame(
        {
            'Δz': pd.concat([fast_values, slow_values], ignore_index=True),
            'Group': ['Fast'] * len(fast_values) + ['Slow'] * len(slow_values),
        }
    )
    plt.figure(figsize=(6, 6))
    palette = {'Fast': '#d73027', 'Slow': '#4575b4'}
    sns.boxplot(data=plot_df, x='Group', y='Δz', hue='Group', palette=palette, legend=False)
    sns.stripplot(data=plot_df, x='Group', y='Δz', color='black', alpha=0.4)
    plt.title('Inflammatory Marker Δz by Tissue Velocity Class')
    plt.ylabel('Zscore Δ (Old - Young)')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'codex_inflammation_boxplot.png', dpi=300)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    agg = aggregate_tissue_gene(df)
    tsi_df = compute_tsi(agg)

    markers = tsi_df[tsi_df['TSI'] > 3.0].copy()
    velocity_df = compute_velocity(markers)
    fast_tissues = identify_fast_tissues(velocity_df)
    shared_df = shared_fast_proteins(agg, fast_tissues)
    inflammation = inflammation_signature(df, fast_tissues)

    # Save artifacts
    tsi_df.to_csv(OUTPUT_DIR / 'tissue_specific_markers_codex.csv', index=False)
    velocity_df.to_csv(OUTPUT_DIR / 'tissue_aging_velocity_codex.csv', index=False)
    shared_df.to_csv(OUTPUT_DIR / 'fast_aging_mechanisms_codex.csv', index=False)

    save_velocity_plot(velocity_df)
    save_marker_heatmap(markers)
    save_inflammation_boxplot(inflammation)

    # Console summary
    print('Velocity ranking (top 5):')
    print(velocity_df[['Velocity_Rank', 'Tissue', 'Velocity', 'Bootstrap_CI']].head())
    print('\nFast-aging tissues:', fast_tissues)
    if not shared_df.empty:
        print('\nShared fast-aging proteins:')
        print(shared_df.head())
    else:
        print('\nNo shared fast-aging proteins detected (|Δz| threshold unmet).')

    print('\nInflammation Mann-Whitney U test:')
    print({k: v for k, v in inflammation.items() if k not in {'fast_values', 'slow_values'}})


if __name__ == '__main__':
    main()
