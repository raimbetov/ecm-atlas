#!/usr/bin/env python3
"""Codex Agent 02: Batch-corrected ECM entropy analysis (v2).
Reproduces and compares entropy metrics against agent_09 baseline, generating
figures, metrics CSV, and execution log within agent workspace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------
WORKSPACE = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/"
    "01_entropy_multi_agent_after_batch_corection/codex_agent_02"
)
WORKSPACE.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/"
    "merged_ecm_aging_zscore.csv"
)
LEGACY_METRICS_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_09_entropy/"
    "entropy_metrics.csv"
)

METRICS_CSV = WORKSPACE / "entropy_metrics_v2.csv"
EXECUTION_LOG = WORKSPACE / "execution.log"

FIG_DISTRIBUTIONS = WORKSPACE / "entropy_distributions_v2.png"
FIG_CLUSTERING = WORKSPACE / "entropy_clustering_v2.png"
FIG_SCATTER = WORKSPACE / "entropy_predictability_space_v2.png"
FIG_COMPARISON = WORKSPACE / "entropy_comparison_v1_v2.png"
FIG_TRANSITIONS = WORKSPACE / "entropy_transitions_v2.png"
FIG_DEATH = WORKSPACE / "death_theorem_comparison_v2.png"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 300, "axes.titleweight": "bold"})

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logger = logging.getLogger("codex_agent_02")
logger.setLevel(logging.INFO)

# Ensure we reset handlers to avoid duplication if script reruns
logger.handlers.clear()
file_handler = logging.FileHandler(EXECUTION_LOG, mode="w", encoding="utf-8")
console_handler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -----------------------------------------------------------------------------
# METRIC CALCULATIONS
# -----------------------------------------------------------------------------

def calculate_shannon_entropy(values: np.ndarray) -> float:
    """Compute Shannon entropy after shifting to positive domain."""
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    shifted = values - np.nanmin(values) + 1.0
    probabilities = shifted / np.nansum(shifted)
    entropy = -np.nansum(probabilities * np.log2(probabilities + 1e-12))
    return float(entropy)


def calculate_variance_entropy_ratio(values: np.ndarray) -> float:
    """Coefficient of variation as variance-based entropy proxy."""
    values = values[~np.isnan(values)]
    if values.size < 2:
        return np.nan
    mean_val = np.nanmean(values)
    if mean_val == 0:
        return np.nan
    cv = np.nanstd(values, ddof=1) / abs(mean_val)
    return float(cv)


def calculate_predictability_score(z_scores: np.ndarray) -> Tuple[float, str]:
    """Direction consistency as predictability metric."""
    valid = z_scores[~np.isnan(z_scores)]
    if valid.size < 2:
        return np.nan, "insufficient_data"
    pos = np.count_nonzero(valid > 0)
    neg = np.count_nonzero(valid < 0)
    total = valid.size
    consistency = max(pos, neg) / total
    if pos > neg:
        direction = "increase"
    elif neg > pos:
        direction = "decrease"
    else:
        direction = "mixed"
    return float(consistency), direction


def calculate_entropy_transition_score(df_protein: pd.DataFrame) -> float:
    """Absolute change in variability (CV) between young and old contexts."""
    old_vals = df_protein["Abundance_Old"].to_numpy(dtype=float)
    young_vals = df_protein["Abundance_Young"].to_numpy(dtype=float)
    old_vals = old_vals[~np.isnan(old_vals)]
    young_vals = young_vals[~np.isnan(young_vals)]
    if old_vals.size < 2 or young_vals.size < 2:
        return np.nan
    cv_old = calculate_variance_entropy_ratio(old_vals)
    cv_young = calculate_variance_entropy_ratio(young_vals)
    if np.isnan(cv_old) or np.isnan(cv_young):
        return np.nan
    return float(abs(cv_old - cv_young))


def calculate_contextual_entropies(df_protein: pd.DataFrame) -> Dict[str, float]:
    """Compute entropy within young vs old partitions for additional insight."""
    young_vals = df_protein["Abundance_Young"].to_numpy(dtype=float)
    old_vals = df_protein["Abundance_Old"].to_numpy(dtype=float)
    young_entropy = calculate_shannon_entropy(young_vals)
    old_entropy = calculate_shannon_entropy(old_vals)
    if np.isnan(young_entropy) or np.isnan(old_entropy):
        conditional = np.nan
    else:
        conditional = float(abs(old_entropy - young_entropy))
    return {
        "Shannon_Entropy_Young": young_entropy,
        "Shannon_Entropy_Old": old_entropy,
        "Conditional_Entropy_Delta": conditional,
    }


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    logger.info("Loading batch-corrected dataset from %s", path)
    df = pd.read_csv(path)
    numeric_cols = [
        "Abundance_Old",
        "Abundance_Old_transformed",
        "Abundance_Young",
        "Abundance_Young_transformed",
        "Zscore_Delta",
        "Zscore_Old",
        "Zscore_Young",
        "N_Profiles_Old",
        "N_Profiles_Young",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(
        "Dataset stats | rows=%d | proteins=%d | studies=%d | tissues=%d",
        len(df),
        df["Canonical_Gene_Symbol"].nunique(),
        df["Study_ID"].nunique(),
        df["Tissue"].nunique(),
    )
    return df


def compute_entropy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing entropy metrics per protein")
    records = []
    grouped = df.groupby("Canonical_Gene_Symbol", sort=False)

    for protein, df_prot in grouped:
        n_studies = df_prot["Study_ID"].nunique()
        n_tissues = df_prot["Tissue"].nunique()
        matrisome_category = df_prot["Matrisome_Category"].mode(dropna=True)
        matrisome_division = df_prot["Matrisome_Division"].mode(dropna=True)
        category = matrisome_category.iloc[0] if not matrisome_category.empty else "Unknown"
        division = matrisome_division.iloc[0] if not matrisome_division.empty else "Unknown"

        all_abundances = pd.concat(
            [
                df_prot["Abundance_Old"].dropna(),
                df_prot["Abundance_Young"].dropna(),
            ]
        ).to_numpy(dtype=float)

        shannon_entropy = calculate_shannon_entropy(all_abundances)
        variance_entropy = calculate_variance_entropy_ratio(all_abundances)
        z_delta = df_prot["Zscore_Delta"].to_numpy(dtype=float)
        predictability, direction = calculate_predictability_score(z_delta)
        transition = calculate_entropy_transition_score(df_prot)
        mean_z_delta = float(np.nanmean(z_delta)) if np.isfinite(np.nanmean(z_delta)) else np.nan

        contextual = calculate_contextual_entropies(df_prot)

        records.append(
            {
                "Protein": protein,
                "N_Studies": int(n_studies),
                "N_Tissues": int(n_tissues),
                "Matrisome_Category": category,
                "Matrisome_Division": division,
                "Shannon_Entropy": shannon_entropy,
                "Variance_Entropy_CV": variance_entropy,
                "Predictability_Score": predictability,
                "Aging_Direction": direction,
                "Entropy_Transition": transition,
                "Mean_Zscore_Delta": mean_z_delta,
                "N_Observations": int(len(df_prot)),
                **contextual,
            }
        )

    df_metrics = pd.DataFrame.from_records(records)
    before_filter = len(df_metrics)
    df_metrics = df_metrics[df_metrics["N_Studies"] >= 2]
    df_metrics = df_metrics[~df_metrics["Shannon_Entropy"].isna()]
    logger.info(
        "Filtered proteins | before=%d | after=%d | criterion=N_Studies>=2 and finite entropy",
        before_filter,
        len(df_metrics),
    )
    return df_metrics.reset_index(drop=True)


# -----------------------------------------------------------------------------
# CLUSTERING AND ANALYSIS
# -----------------------------------------------------------------------------

def perform_clustering(df_metrics: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, np.ndarray]:
    logger.info("Performing hierarchical clustering with %d clusters", n_clusters)
    features = [
        "Shannon_Entropy",
        "Variance_Entropy_CV",
        "Predictability_Score",
        "Entropy_Transition",
    ]
    df_features = df_metrics[features].copy()
    df_features = df_features.fillna(df_features.median())
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)
    linkage_matrix = linkage(X, method="ward")
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    df_metrics["Entropy_Cluster"] = cluster_labels
    logger.info("Cluster sizes: %s", df_metrics["Entropy_Cluster"].value_counts().to_dict())
    return df_metrics, linkage_matrix


def run_death_theorem_tests(df_metrics: pd.DataFrame) -> Dict[str, float]:
    logger.info("Evaluating DEATh theorem contrasts")
    structural = df_metrics[df_metrics["Matrisome_Division"] == "Core matrisome"]
    regulatory = df_metrics[df_metrics["Matrisome_Division"] == "Matrisome-associated"]

    results: Dict[str, float] = {}
    if not structural.empty and not regulatory.empty:
        entropy_p = stats.mannwhitneyu(
            structural["Shannon_Entropy"].dropna(),
            regulatory["Shannon_Entropy"].dropna(),
            alternative="two-sided",
        ).pvalue
        predictability_p = stats.mannwhitneyu(
            structural["Predictability_Score"].dropna(),
            regulatory["Predictability_Score"].dropna(),
            alternative="two-sided",
        ).pvalue
        results["struct_vs_reg_entropy_p"] = float(entropy_p)
        results["struct_vs_reg_predictability_p"] = float(predictability_p)
        results["structural_entropy_mean"] = float(structural["Shannon_Entropy"].mean())
        results["regulatory_entropy_mean"] = float(regulatory["Shannon_Entropy"].mean())
        results["structural_predictability_mean"] = float(structural["Predictability_Score"].mean())
        results["regulatory_predictability_mean"] = float(regulatory["Predictability_Score"].mean())
        logger.info(
            "DEATh test | entropy_p=%.4g | predictability_p=%.4g",
            entropy_p,
            predictability_p,
        )

    collagens = df_metrics[df_metrics["Protein"].str.startswith("COL")]
    if not collagens.empty:
        results["collagen_predictability_mean"] = float(collagens["Predictability_Score"].mean())
        results["collagen_count"] = int(len(collagens))
        logger.info(
            "Collagen predictability mean=%.3f across %d proteins",
            results["collagen_predictability_mean"],
            results["collagen_count"],
        )
    return results


def compare_with_legacy(
    df_metrics: pd.DataFrame,
    legacy_path: Path,
    n_clusters: int = 4,
) -> Dict[str, float]:
    logger.info("Comparing against legacy metrics at %s", legacy_path)
    legacy = pd.read_csv(legacy_path)
    legacy = legacy.rename(columns={"Shannon_entropy": "Shannon_Entropy"})
    # Legacy clustering for stability assessment
    common_cols = [
        "Shannon_Entropy",
        "Variance_Entropy_CV",
        "Predictability_Score",
        "Entropy_Transition",
    ]
    legacy_features = legacy[common_cols].copy()
    legacy_features = legacy_features.fillna(legacy_features.median())
    scaler = StandardScaler()
    X_legacy = scaler.fit_transform(legacy_features)
    legacy_linkage = linkage(X_legacy, method="ward")
    legacy_clusters = fcluster(legacy_linkage, n_clusters, criterion="maxclust")
    legacy["Entropy_Cluster"] = legacy_clusters

    merged = pd.merge(
        df_metrics,
        legacy,
        on="Protein",
        suffixes=("_v2", "_v1"),
    )
    logger.info("Proteins in common with legacy: %d", len(merged))
    comparison: Dict[str, float] = {"n_common_proteins": float(len(merged))}
    if merged.empty:
        logger.warning("No overlapping proteins found for comparison")
        return comparison

    for metric in ["Shannon_Entropy", "Predictability_Score", "Entropy_Transition"]:
        pairs = merged[[f"{metric}_v2", f"{metric}_v1"]].dropna()
        if len(pairs) < 2 or pairs.nunique().min() < 2:
            rho, pval = float('nan'), float('nan')
        else:
            rho, pval = stats.spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1])
        comparison[f"spearman_{metric.lower()}"] = float(rho)
        comparison[f"spearman_{metric.lower()}_p"] = float(pval)
        logger.info("Spearman %s | rho=%s | p=%s", metric, rho, pval)

    # Cluster stability: adjusted overlap proportion
    contingency = pd.crosstab(merged["Entropy_Cluster_v2"], merged["Entropy_Cluster_v1"])
    agreement = contingency.max(axis=1).sum() / contingency.values.sum()
    comparison["cluster_assignment_agreement"] = float(agreement)
    logger.info("Cluster assignment agreement=%.3f", agreement)

    legacy_struct = legacy[legacy["Matrisome_Division"] == "Core matrisome"]
    legacy_reg = legacy[legacy["Matrisome_Division"] == "Matrisome-associated"]
    if not legacy_struct.empty and not legacy_reg.empty:
        legacy_entropy_p = stats.mannwhitneyu(
            legacy_struct["Shannon_Entropy"].dropna(),
            legacy_reg["Shannon_Entropy"].dropna(),
            alternative="two-sided",
        ).pvalue
        legacy_predictability_p = stats.mannwhitneyu(
            legacy_struct["Predictability_Score"].dropna(),
            legacy_reg["Predictability_Score"].dropna(),
            alternative="two-sided",
        ).pvalue
        comparison["legacy_struct_vs_reg_entropy_p"] = float(legacy_entropy_p)
        comparison["legacy_struct_vs_reg_predictability_p"] = float(legacy_predictability_p)
        logger.info(
            "Legacy DEATh p-values | entropy=%.4g | predictability=%.4g",
            legacy_entropy_p,
            legacy_predictability_p,
        )
    return comparison


# -----------------------------------------------------------------------------
# VISUALIZATIONS
# -----------------------------------------------------------------------------

def plot_entropy_distributions(df_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    hist_args = {"bins": 30, "edgecolor": "black", "alpha": 0.75}

    metrics = [
        ("Shannon_Entropy", "Shannon Entropy"),
        ("Variance_Entropy_CV", "Variance CV"),
        ("Predictability_Score", "Predictability (0-1)"),
        ("Entropy_Transition", "Entropy Transition"),
    ]
    for ax, (column, label) in zip(axes.flat, metrics):
        data = df_metrics[column].dropna()
        ax.hist(data, color="#4C72B0", **hist_args)
        ax.axvline(data.median(), color="red", linestyle="--", label="Median")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DISTRIBUTIONS, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_heatmap(df_metrics: pd.DataFrame, linkage_matrix: np.ndarray) -> None:
    features = [
        "Shannon_Entropy",
        "Variance_Entropy_CV",
        "Predictability_Score",
        "Entropy_Transition",
    ]
    df_heatmap = df_metrics.set_index("Protein")[features]
    df_heatmap = df_heatmap.fillna(df_heatmap.median())
    scaler = StandardScaler()
    X = scaler.fit_transform(df_heatmap)
    df_scaled = pd.DataFrame(X, index=df_heatmap.index, columns=features)
    order = dendrogram(linkage_matrix, orientation="left", no_plot=True)["leaves"]
    df_scaled = df_scaled.iloc[order[::-1]]

    # Create figure with dendrogram and heatmap
    fig = plt.figure(figsize=(14, 9))
    dend_ax = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    dendrogram(linkage_matrix, orientation="left", ax=dend_ax, no_labels=True)
    dend_ax.set_title("Clustering Dendrogram")
    dend_ax.set_ylabel("Proteins")
    dend_ax.set_xlabel("Distance")

    heatmap_ax = fig.add_axes([0.32, 0.1, 0.63, 0.8])
    sns.heatmap(
        df_scaled,
        cmap="vlag",
        ax=heatmap_ax,
        cbar_kws={"label": "Z-score"},
        yticklabels=False,
    )
    heatmap_ax.set_title("Entropy Feature Heatmap (Scaled)")
    fig.savefig(FIG_CLUSTERING, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_predictability_scatter(df_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df_metrics["Shannon_Entropy"],
        df_metrics["Predictability_Score"],
        c=df_metrics["Entropy_Cluster"],
        cmap="viridis",
        s=60,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Shannon Entropy (Disorder)")
    ax.set_ylabel("Predictability Score (Determinism)")
    ax.set_title("Entropy-Predictability Landscape (Batch-Corrected)")
    x_mid = df_metrics["Shannon_Entropy"].median()
    y_mid = df_metrics["Predictability_Score"].median()
    ax.axvline(x_mid, color="red", linestyle="--", alpha=0.6)
    ax.axhline(y_mid, color="red", linestyle="--", alpha=0.6)
    ax.text(x_mid * 0.6, y_mid + 0.2, "Deterministic Core", ha="center")
    ax.text(x_mid * 1.4, y_mid + 0.2, "Regulated Chaos", ha="center")
    ax.text(x_mid * 0.6, y_mid - 0.2, "Contextual Stabilizers", ha="center")
    ax.text(x_mid * 1.4, y_mid - 0.2, "Chaotic Drift", ha="center")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")
    fig.tight_layout()
    fig.savefig(FIG_SCATTER, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_comparison(df_metrics: pd.DataFrame, legacy_path: Path) -> None:
    legacy = pd.read_csv(legacy_path)
    merged = pd.merge(
        df_metrics,
        legacy,
        on="Protein",
        suffixes=("_v2", "_v1"),
    )
    if merged.empty:
        logger.warning("Skipping before/after comparison plot due to no overlap")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, label in zip(
        axes,
        ["Shannon_Entropy", "Predictability_Score"],
        ["Shannon Entropy", "Predictability"],
    ):
        x = merged[f"{metric}_v1"]
        y = merged[f"{metric}_v2"]
        ax.scatter(x, y, alpha=0.7, edgecolors="black", linewidth=0.3)
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        rho, _ = stats.spearmanr(x, y)
        ax.set_xlabel(f"Legacy {label}")
        ax.set_ylabel(f"Batch-corrected {label}")
        ax.set_title(f"{label} Before vs After\nSpearman ρ={rho:.2f}")
    fig.tight_layout()
    fig.savefig(FIG_COMPARISON, bbox_inches="tight")
    plt.close(fig)


def plot_transition_bar(df_metrics: pd.DataFrame) -> None:
    top_transitions = df_metrics.nlargest(15, "Entropy_Transition")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=top_transitions,
        x="Entropy_Transition",
        y="Protein",
        hue="Aging_Direction",
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("Entropy Transition Score")
    ax.set_ylabel("Protein")
    ax.set_title("Top Entropy Transition Proteins")
    ax.legend(title="Aging Direction", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_TRANSITIONS, bbox_inches="tight")
    plt.close(fig)


def plot_death_comparison(df_metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    divisions = ["Core matrisome", "Matrisome-associated"]
    metrics = ["Shannon_Entropy", "Predictability_Score"]
    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            data=df_metrics[df_metrics["Matrisome_Division"].isin(divisions)],
            x="Matrisome_Division",
            y=metric,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric.replace('_', ' ')} by Division")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DEATH, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting Codex Agent 02 entropy analysis")
    df = load_dataset(DATA_PATH)
    df_metrics = compute_entropy_metrics(df)
    assert len(df_metrics) >= 400, "Insufficient protein coverage (<400)."

    df_metrics, linkage_matrix = perform_clustering(df_metrics, n_clusters=4)
    df_metrics.to_csv(METRICS_CSV, index=False)
    logger.info("Saved metrics (with clusters) to %s", METRICS_CSV)
    death_results = run_death_theorem_tests(df_metrics)
    comparison_results = compare_with_legacy(df_metrics, LEGACY_METRICS_PATH, n_clusters=4)

    plot_entropy_distributions(df_metrics)
    plot_clustering_heatmap(df_metrics, linkage_matrix)
    plot_entropy_predictability_scatter(df_metrics)
    plot_before_after_comparison(df_metrics, LEGACY_METRICS_PATH)
    plot_transition_bar(df_metrics)
    plot_death_comparison(df_metrics)

    # Summary logging
    high_entropy = df_metrics.nlargest(10, "Shannon_Entropy")["Protein"].tolist()
    low_entropy = df_metrics.nsmallest(10, "Shannon_Entropy")["Protein"].tolist()
    high_transition = df_metrics.nlargest(10, "Entropy_Transition")["Protein"].tolist()
    logger.info("Top high-entropy proteins: %s", high_entropy)
    logger.info("Top low-entropy proteins: %s", low_entropy)
    logger.info("Top transition proteins: %s", high_transition)
    logger.info("DEATh stats: %s", death_results)
    logger.info("Legacy comparison summary: %s", comparison_results)
    logger.info("Workflow complete")


if __name__ == "__main__":
    main()
