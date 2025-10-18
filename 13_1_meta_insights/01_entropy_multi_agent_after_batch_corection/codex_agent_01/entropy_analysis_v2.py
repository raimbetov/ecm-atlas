#!/usr/bin/env python3
"""Codex Agent 01: Entropy analysis after batch correction (V2)."""

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BASE_DIR = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/01_entropy_multi_agent_after_batch_corection/codex_agent_01')
BASE_DIR.mkdir(parents=True, exist_ok=True)

NEW_DATA_PATH = Path('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
OLD_METRICS_PATH = Path('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_09_entropy/entropy_metrics.csv')
OLD_DATA_BACKUP = Path('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore_OLD_BACKUP.csv')

METRICS_CSV_PATH = BASE_DIR / 'entropy_metrics_v2.csv'
EXECUTION_LOG_PATH = BASE_DIR / 'execution.log'

PLOT_DISTRIBUTIONS = BASE_DIR / 'entropy_distributions_v2.png'
PLOT_CLUSTERING = BASE_DIR / 'entropy_clustering_v2.png'
PLOT_SCATTER = BASE_DIR / 'entropy_predictability_space_v2.png'
PLOT_COMPARISON = BASE_DIR / 'entropy_comparison_v1_v2.png'
PLOT_TRANSITIONS = BASE_DIR / 'entropy_transition_highlights_v2.png'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('deep')

# ----------------------------------------------------------------------------
# Entropy helper functions
# ----------------------------------------------------------------------------

def _safe_array(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    return arr[~np.isnan(arr)]


def calculate_shannon_entropy(abundances: pd.Series) -> float:
    values = _safe_array(abundances)
    if values.size == 0:
        return np.nan

    shifted = values - values.min()
    shifted += 1.0  # ensure strictly positive
    probs = shifted / shifted.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


def calculate_variance_cv(abundances: pd.Series) -> float:
    values = _safe_array(abundances)
    if values.size < 2:
        return np.nan
    mean_val = np.mean(values)
    if math.isclose(mean_val, 0.0, abs_tol=1e-9):
        return np.nan
    cv = np.std(values, ddof=1) / abs(mean_val)
    return float(cv)


def calculate_predictability_score(z_scores: pd.Series) -> Tuple[float, str]:
    values = _safe_array(z_scores)
    if values.size < 2:
        return np.nan, 'insufficient_data'
    positive = np.sum(values > 0)
    negative = np.sum(values < 0)
    total = values.size
    consistency = max(positive, negative) / total
    if positive > negative:
        direction = 'increase'
    elif negative > positive:
        direction = 'decrease'
    else:
        direction = 'mixed'
    return float(consistency), direction


def calculate_transition_score(df_protein: pd.DataFrame) -> float:
    old_vals = _safe_array(df_protein['Abundance_Old'])
    young_vals = _safe_array(df_protein['Abundance_Young'])
    if old_vals.size < 2 or young_vals.size < 2:
        return np.nan
    cv_old = calculate_variance_cv(pd.Series(old_vals))
    cv_young = calculate_variance_cv(pd.Series(young_vals))
    if np.isnan(cv_old) or np.isnan(cv_young):
        return np.nan
    transition = abs(cv_old - cv_young)
    return float(transition)


# ----------------------------------------------------------------------------
# Core analysis functions
# ----------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_entropy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = df.groupby('Canonical_Gene_Symbol', sort=False)

    for protein, group in grouped:
        n_studies = group['Study_ID'].nunique()
        if n_studies < 2:
            continue

        n_tissues = group['Tissue'].nunique()
        matrisome_category = group['Matrisome_Category'].mode(dropna=True)
        matrisome_division = group['Matrisome_Division'].mode(dropna=True)

        shannon = calculate_shannon_entropy(
            pd.concat([
                group['Abundance_Old'].dropna(),
                group['Abundance_Young'].dropna()
            ])
        )
        variance_cv = calculate_variance_cv(
            pd.concat([
                group['Abundance_Old'].dropna(),
                group['Abundance_Young'].dropna()
            ])
        )
        predictability, direction = calculate_predictability_score(group['Zscore_Delta'])
        transition = calculate_transition_score(group)
        mean_z = group['Zscore_Delta'].dropna().mean() if group['Zscore_Delta'].notna().any() else np.nan

        records.append({
            'Protein': protein,
            'N_Studies': n_studies,
            'N_Tissues': n_tissues,
            'Matrisome_Category': matrisome_category.iloc[0] if not matrisome_category.empty else 'Unknown',
            'Matrisome_Division': matrisome_division.iloc[0] if not matrisome_division.empty else 'Unknown',
            'Shannon_Entropy': shannon,
            'Variance_Entropy_CV': variance_cv,
            'Predictability_Score': predictability,
            'Aging_Direction': direction,
            'Entropy_Transition': transition,
            'Mean_Zscore_Delta': mean_z,
            'N_Observations': len(group)
        })

    df_metrics = pd.DataFrame.from_records(records)
    df_metrics = df_metrics.dropna(subset=['Shannon_Entropy'])
    return df_metrics.reset_index(drop=True)


def perform_clustering(df_metrics: pd.DataFrame, n_clusters: int = 5) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']
    df_cluster = df_metrics[features].copy()
    df_cluster = df_cluster.apply(lambda col: col.fillna(col.median()))
    scaler = StandardScaler()
    X = scaler.fit_transform(df_cluster)
    linkage_matrix = linkage(X, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_metrics = df_metrics.copy()
    df_metrics['Entropy_Cluster'] = clusters
    return df_metrics, linkage_matrix, X


def evaluate_death_theorem(df_metrics: pd.DataFrame) -> Dict[str, float]:
    structural = df_metrics[df_metrics['Matrisome_Division'] == 'Core matrisome']
    regulatory = df_metrics[df_metrics['Matrisome_Division'] == 'Matrisome-associated']

    results: Dict[str, float] = {}
    for label, subset in [('structural', structural), ('regulatory', regulatory)]:
        results[f'{label}_count'] = len(subset)
        results[f'{label}_entropy_mean'] = subset['Shannon_Entropy'].mean()
        results[f'{label}_predictability_mean'] = subset['Predictability_Score'].mean()

    if len(structural) > 0 and len(regulatory) > 0:
        entropy_p = stats.mannwhitneyu(
            structural['Shannon_Entropy'].dropna(),
            regulatory['Shannon_Entropy'].dropna(),
            alternative='two-sided'
        ).pvalue
        predictability_p = stats.mannwhitneyu(
            structural['Predictability_Score'].dropna(),
            regulatory['Predictability_Score'].dropna(),
            alternative='two-sided'
        ).pvalue
    else:
        entropy_p = np.nan
        predictability_p = np.nan

    collagens = df_metrics[df_metrics['Protein'].str.startswith('COL')]
    collagen_predictability = collagens['Predictability_Score'].mean()

    results.update({
        'entropy_pvalue': entropy_p,
        'predictability_pvalue': predictability_p,
        'collagen_predictability_mean': collagen_predictability,
        'collagen_count': len(collagens),
        'all_predictability_mean': df_metrics['Predictability_Score'].mean(),
    })

    return results


def compare_with_legacy(new_metrics: pd.DataFrame, old_metrics: pd.DataFrame) -> Dict[str, float]:
    new = new_metrics.rename(columns={'Entropy_Cluster': 'Entropy_Cluster_new'})
    old = old_metrics.rename(columns={'Entropy_Cluster': 'Entropy_Cluster_old'})

    overlaps = new.merge(
        old,
        on='Protein',
        suffixes=('_new', '_old'),
        how='inner'
    )

    comparison: Dict[str, float] = {'overlap_count': len(overlaps)}
    if len(overlaps) == 0:
        return comparison

    for metric in ['Shannon_Entropy', 'Predictability_Score', 'Variance_Entropy_CV', 'Entropy_Transition']:
        new_col = f'{metric}_new'
        old_col = f'{metric}_old'
        valid = overlaps[[new_col, old_col]].dropna()
        if len(valid) >= 3:
            rho, pval = stats.spearmanr(valid[new_col], valid[old_col])
        else:
            rho, pval = np.nan, np.nan
        comparison[f'{metric.lower()}_spearman'] = rho
        comparison[f'{metric.lower()}_pvalue'] = pval

    if 'Entropy_Cluster_new' in overlaps.columns and 'Entropy_Cluster_old' in overlaps.columns:
        ari = adjusted_rand_score(overlaps['Entropy_Cluster_new'], overlaps['Entropy_Cluster_old'])
        nmi = normalized_mutual_info_score(overlaps['Entropy_Cluster_new'], overlaps['Entropy_Cluster_old'])
    else:
        ari = np.nan
        nmi = np.nan

    comparison['cluster_adjusted_rand'] = ari
    comparison['cluster_nmi'] = nmi

    overlaps['entropy_shift'] = overlaps['Shannon_Entropy_new'] - overlaps['Shannon_Entropy_old']
    comparison['top_entropy_gainers'] = overlaps.nlargest(5, 'entropy_shift')[['Protein', 'entropy_shift']].to_dict('records')
    comparison['top_entropy_losers'] = overlaps.nsmallest(5, 'entropy_shift')[['Protein', 'entropy_shift']].to_dict('records')

    return comparison


# ----------------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------------

def plot_distributions(df_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        ('Shannon_Entropy', 'Shannon Entropy'),
        ('Variance_Entropy_CV', 'Variance CV'),
        ('Predictability_Score', 'Predictability Score (0-1)'),
        ('Entropy_Transition', 'Entropy Transition (|CVΔ|)')
    ]

    for ax, (col, title) in zip(axes.flatten(), panels):
        series = df_metrics[col].dropna()
        ax.hist(series, bins=30, color='#1f77b4', alpha=0.75, edgecolor='black')
        ax.axvline(series.median(), color='red', linestyle='--', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(PLOT_DISTRIBUTIONS, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_clustering(df_metrics: pd.DataFrame, linkage_matrix: np.ndarray) -> None:
    features = ['Shannon_Entropy', 'Variance_Entropy_CV', 'Predictability_Score', 'Entropy_Transition']
    df_features = df_metrics.set_index('Protein')[features]
    df_features = df_features.apply(lambda col: col.fillna(col.median()))
    cg = sns.clustermap(
        df_features,
        method='ward',
        cmap='viridis',
        figsize=(10, 12),
        metric='euclidean',
        row_cluster=True,
        col_cluster=False,
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.02, 0.15)
    )
    cg.fig.suptitle('Entropy Profile Clustering (Ward linkage)', fontsize=14)
    cg.savefig(PLOT_CLUSTERING, dpi=300, bbox_inches='tight')
    plt.close(cg.fig)


def plot_entropy_space(df_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_metrics['Shannon_Entropy'],
        df_metrics['Predictability_Score'],
        c=df_metrics['Entropy_Cluster'],
        cmap='tab10',
        s=45,
        alpha=0.75,
        edgecolor='black',
        linewidth=0.3
    )
    ax.set_xlabel('Shannon Entropy (Disorder)')
    ax.set_ylabel('Predictability Score (Determinism)')
    ax.set_title('Entropy-Predictability Phase Space')
    plt.colorbar(scatter, label='Cluster ID', ax=ax)
    x_med = df_metrics['Shannon_Entropy'].median()
    y_med = df_metrics['Predictability_Score'].median()
    ax.axvline(x_med, color='grey', linestyle='--', linewidth=1)
    ax.axhline(y_med, color='grey', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison(new_metrics: pd.DataFrame, old_metrics: pd.DataFrame, comparison: Dict[str, float]) -> None:
    overlaps = new_metrics.merge(
        old_metrics,
        on='Protein',
        suffixes=('_new', '_old'),
        how='inner'
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if len(overlaps) > 0:
        axes[0].scatter(
            overlaps['Shannon_Entropy_old'],
            overlaps['Shannon_Entropy_new'],
            alpha=0.6,
            edgecolor='black',
            linewidth=0.3
        )
        min_val = min(overlaps['Shannon_Entropy_old'].min(), overlaps['Shannon_Entropy_new'].min())
        max_val = max(overlaps['Shannon_Entropy_old'].max(), overlaps['Shannon_Entropy_new'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        axes[0].set_title('Shannon Entropy: Legacy vs Batch-Corrected')
        axes[0].set_xlabel('Legacy Shannon Entropy')
        axes[0].set_ylabel('Batch-Corrected Shannon Entropy')
        rho = comparison.get('shannon_entropy_spearman', np.nan)
        axes[0].text(0.05, 0.9, f"ρ = {rho:.2f}", transform=axes[0].transAxes)

        axes[1].scatter(
            overlaps['Predictability_Score_old'],
            overlaps['Predictability_Score_new'],
            alpha=0.6,
            color='#2ca02c',
            edgecolor='black',
            linewidth=0.3
        )
        min_val = min(overlaps['Predictability_Score_old'].min(), overlaps['Predictability_Score_new'].min())
        max_val = max(overlaps['Predictability_Score_old'].max(), overlaps['Predictability_Score_new'].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        axes[1].set_title('Predictability: Legacy vs Batch-Corrected')
        axes[1].set_xlabel('Legacy Predictability')
        axes[1].set_ylabel('Batch-Corrected Predictability')
        rho_pred = comparison.get('predictability_score_spearman', np.nan)
        axes[1].text(0.05, 0.9, f"ρ = {rho_pred:.2f}", transform=axes[1].transAxes)
    else:
        for ax in axes:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No overlapping proteins found', ha='center', va='center')

    fig.suptitle('Entropy Metrics Before vs After Batch Correction')
    plt.tight_layout()
    plt.savefig(PLOT_COMPARISON, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_transition_highlights(df_metrics: pd.DataFrame) -> None:
    top_transitions = df_metrics.nlargest(15, 'Entropy_Transition')[['Protein', 'Entropy_Transition']]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_transitions,
        x='Entropy_Transition',
        y='Protein',
        ax=ax,
        palette='rocket'
    )
    ax.set_title('Top 15 Entropy Transition Proteins (|CV old - CV young|)')
    ax.set_xlabel('Transition Score')
    ax.set_ylabel('Protein')
    plt.tight_layout()
    plt.savefig(PLOT_TRANSITIONS, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------
# Execution logging helper
# ----------------------------------------------------------------------------

def append_log(message: str) -> None:
    with EXECUTION_LOG_PATH.open('a', encoding='utf-8') as handle:
        handle.write(message + '\n')


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def main() -> None:
    EXECUTION_LOG_PATH.write_text('Codex Agent 01 entropy analysis V2 log\n', encoding='utf-8')
    append_log('Step 1: Loading datasets')

    df_new = load_dataset(NEW_DATA_PATH)
    df_old_backup = load_dataset(OLD_DATA_BACKUP)
    old_metrics = pd.read_csv(OLD_METRICS_PATH)

    append_log(f'New dataset shape: {df_new.shape}')
    append_log(f'Old backup shape: {df_old_backup.shape}')
    append_log(f'Legacy metrics count: {len(old_metrics)}')

    append_log('Step 2: Computing entropy metrics for batch-corrected data')
    new_metrics = compute_entropy_metrics(df_new)
    append_log(f'Proteins meeting criteria: {len(new_metrics)}')
    if len(new_metrics) < 400:
        raise ValueError('Insufficient protein count after filtering (<400).')

    append_log('Step 3: Clustering new metrics (n_clusters=5)')
    new_metrics_clustered, linkage_matrix_new, X_new = perform_clustering(new_metrics, n_clusters=5)
    new_metrics_clustered.to_csv(METRICS_CSV_PATH, index=False)
    append_log(f'Saved metrics with clusters to {METRICS_CSV_PATH}')

    append_log('Step 4: Clustering legacy metrics for stability comparison')
    old_metrics_filtered = old_metrics.dropna(subset=['Shannon_Entropy']).copy()
    old_metrics_clustered, linkage_matrix_old, X_old = perform_clustering(old_metrics_filtered, n_clusters=5)
    old_metrics_clustered = old_metrics_clustered.rename(columns={'Entropy_Cluster': 'Entropy_Cluster_old'})

    append_log('Step 5: DEATh theorem evaluation')
    death_results = evaluate_death_theorem(new_metrics_clustered)
    append_log(json.dumps(death_results, indent=2))

    append_log('Step 6: Legacy comparison analytics')
    comparison = compare_with_legacy(new_metrics_clustered, old_metrics_clustered)
    append_log(json.dumps(comparison, indent=2))

    append_log('Step 7: Visualization suite generation')
    plot_distributions(new_metrics_clustered)
    plot_clustering(new_metrics_clustered, linkage_matrix_new)
    plot_entropy_space(new_metrics_clustered)
    plot_comparison(new_metrics_clustered, old_metrics_clustered, comparison)
    plot_transition_highlights(new_metrics_clustered)
    append_log('Visualizations saved')

    append_log('Step 8: Summary statistics')
    summary = {
        'mean_shannon': new_metrics_clustered['Shannon_Entropy'].mean(),
        'std_shannon': new_metrics_clustered['Shannon_Entropy'].std(),
        'mean_predictability': new_metrics_clustered['Predictability_Score'].mean(),
        'std_predictability': new_metrics_clustered['Predictability_Score'].std(),
        'mean_transition': new_metrics_clustered['Entropy_Transition'].mean(),
    }
    append_log(json.dumps(summary, indent=2))

    append_log('Step 9: Top lists for reporting')
    top_high_entropy = new_metrics_clustered.nlargest(10, 'Shannon_Entropy')[['Protein', 'Shannon_Entropy']]
    top_low_entropy = new_metrics_clustered.nsmallest(10, 'Shannon_Entropy')[['Protein', 'Shannon_Entropy']]
    top_transitions = new_metrics_clustered.nlargest(10, 'Entropy_Transition')[['Protein', 'Entropy_Transition']]
    append_log('Top high entropy:\n' + top_high_entropy.to_string(index=False))
    append_log('Top low entropy:\n' + top_low_entropy.to_string(index=False))
    append_log('Top transitions:\n' + top_transitions.to_string(index=False))

    append_log('Analysis complete')


if __name__ == '__main__':
    main()
