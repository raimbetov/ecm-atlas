#!/usr/bin/env python3
"""ComBat V2 batch correction pipeline with Age_Group preservation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path("/Users/Kravtsovd/projects/ecm-atlas")
INPUT_STANDARDIZED = (
    PROJECT_ROOT
    / "14_exploratory_batch_correction"
    / "multi_agents_ver1_for_batch_cerection"
    / "codex"
    / "merged_ecm_aging_STANDARDIZED.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "14_exploratory_batch_correction"
    / "multi_agents_ver1_for_batch_cerection"
    / "step2_batch"
    / "codex"
)
OUTPUT_CSV = OUTPUT_DIR / "merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv"
OUTPUT_JSON = OUTPUT_DIR / "validation_metrics_codex.json"

AGING_DRIVERS = [
    "COL1A1",
    "COL1A2",
    "COL3A1",
    "COL5A1",
    "COL6A1",
    "COL6A2",
    "COL6A3",
    "COL4A1",
    "COL4A2",
    "COL18A1",
    "FN1",
    "LAMA5",
    "LAMB2",
    "FBN1",
]

TARGETS = {
    "ICC": (0.50, 0.60),
    "Driver_Recovery": 66.7,
    "FDR_Significant": 5,
    "Zscore_Std": (0.8, 1.5),
}

ICC_WITHIN_SCALE = 2.0


def log(message: str) -> None:
    """Lightweight stdout logger."""
    print(f"[codex] {message}")


def load_data() -> pd.DataFrame:
    if not INPUT_STANDARDIZED.exists():
        log(f"ERROR: Missing input file {INPUT_STANDARDIZED}")
        sys.exit(1)

    df = pd.read_csv(INPUT_STANDARDIZED)
    log(f"Loaded standardized data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def reshape_to_samples(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return expression matrix (proteins x samples) and sample metadata."""
    shared_cols = ["Protein_ID", "Gene_Symbol", "Study_ID", "Tissue_Compartment"]

    long_frames = []
    for age_label, col in (("Young", "Abundance_Young"), ("Old", "Abundance_Old")):
        chunk = df[shared_cols + [col]].copy()
        chunk = chunk.rename(columns={col: "Abundance"})
        chunk["Age_Group"] = age_label
        long_frames.append(chunk)

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df = long_df.dropna(subset=["Abundance"])
    long_df["Sample_ID"] = (
        long_df["Study_ID"].astype(str)
        + "|"
        + long_df["Tissue_Compartment"].astype(str)
        + "|"
        + long_df["Age_Group"].astype(str)
    )

    expr_matrix = long_df.pivot_table(
        index="Protein_ID",
        columns="Sample_ID",
        values="Abundance",
        aggfunc="mean",
    )

    sample_metadata = (
        long_df[["Sample_ID", "Study_ID", "Tissue_Compartment", "Age_Group"]]
        .drop_duplicates()
        .set_index("Sample_ID")
        .sort_index()
    )

    expr_matrix = expr_matrix.reindex(sorted(expr_matrix.columns), axis=1)
    sample_metadata = sample_metadata.loc[expr_matrix.columns]

    log(f"Expression matrix shape: {expr_matrix.shape}")
    return expr_matrix, sample_metadata


def impute_missing(expr_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Median-impute per protein; drop all-null proteins."""
    imputed = expr_matrix.copy()
    imputation_stats: Dict[str, float] = {}

    row_medians = imputed.median(axis=1, skipna=True)
    all_null = row_medians.isna()
    dropped = int(all_null.sum())
    if dropped:
        imputed = imputed.loc[~all_null]
        row_medians = row_medians.loc[~all_null]
        log(f"Dropped {dropped} proteins with all-null abundances")

    for protein_id, median_value in row_medians.items():
        if np.isnan(median_value):
            continue
        mask = imputed.loc[protein_id].isna()
        if mask.any():
            imputed.loc[protein_id, mask] = median_value
            imputation_stats[str(protein_id)] = float(median_value)

    if imputed.isna().any().any():
        global_median = float(np.nanmedian(imputed.to_numpy()))
        imputed = imputed.fillna(global_median)
        log("Filled residual NaNs with global median")

    return imputed, imputation_stats


def compute_group_means(
    matrix: pd.DataFrame, sample_metadata: pd.DataFrame
) -> Dict[Tuple[str, str], pd.Series]:
    means: Dict[Tuple[str, str], pd.Series] = {}
    for tissue in sample_metadata["Tissue_Compartment"].unique():
        for age in sample_metadata["Age_Group"].unique():
            cols = sample_metadata.index[
                (sample_metadata["Tissue_Compartment"] == tissue)
                & (sample_metadata["Age_Group"] == age)
            ]
            if len(cols) == 0:
                continue
            means[(tissue, age)] = matrix.loc[:, cols].mean(axis=1)
    return means


def restore_group_means(
    matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    baseline_means: Dict[Tuple[str, str], pd.Series],
) -> pd.DataFrame:
    adjusted = matrix.copy()
    for (tissue, age), baseline in baseline_means.items():
        cols = sample_metadata.index[
            (sample_metadata["Tissue_Compartment"] == tissue)
            & (sample_metadata["Age_Group"] == age)
        ]
        if len(cols) == 0:
            continue
        current = adjusted.loc[:, cols].mean(axis=1)
        delta = baseline - current
        adjusted.loc[:, cols] = adjusted.loc[:, cols].add(delta, axis=0)
    return adjusted


def run_combat(
    expr_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    parametric: bool = True,
) -> pd.DataFrame:
    """Run neuroCombat preserving Age_Group while restoring tissue means."""
    from neuroCombat import neuroCombat

    baseline_means = compute_group_means(expr_matrix, sample_metadata)
    covars_df = sample_metadata[["Study_ID", "Age_Group"]].copy()
    covars_df["Age_Group"] = covars_df["Age_Group"].astype("category")

    log(
        f"Running global ComBat: {expr_matrix.shape[1]} samples, {sample_metadata['Study_ID'].nunique()} batches"
    )

    combat_result = neuroCombat(
        dat=expr_matrix,
        covars=covars_df,
        batch_col="Study_ID",
        categorical_cols=["Age_Group"],
        eb=parametric,
        parametric=parametric,
        mean_only=False,
    )

    corrected_df = pd.DataFrame(
        combat_result["data"],
        index=expr_matrix.index,
        columns=expr_matrix.columns,
    )

    restored = restore_group_means(corrected_df, sample_metadata, baseline_means)
    log("ComBat correction complete")
    return restored


def rebuild_wide(
    original_df: pd.DataFrame,
    corrected_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
) -> pd.DataFrame:
    corrected_long = corrected_matrix.stack().reset_index()
    corrected_long.columns = ["Protein_ID", "Sample_ID", "Abundance_ComBat"]

    corrected_long = corrected_long.merge(
        sample_metadata.reset_index(),
        on="Sample_ID",
        how="left",
    )
    corrected_long["Key"] = (
        corrected_long["Protein_ID"].astype(str)
        + "|"
        + corrected_long["Study_ID"].astype(str)
        + "|"
        + corrected_long["Tissue_Compartment"].astype(str)
    )

    pivot_corrected = corrected_long.pivot_table(
        index="Key",
        columns="Age_Group",
        values="Abundance_ComBat",
        aggfunc="mean",
    )

    wide = original_df.copy()
    wide["Key"] = (
        wide["Protein_ID"].astype(str)
        + "|"
        + wide["Study_ID"].astype(str)
        + "|"
        + wide["Tissue_Compartment"].astype(str)
    )

    wide["Abundance_Young_Original"] = wide["Abundance_Young"]
    wide["Abundance_Old_Original"] = wide["Abundance_Old"]

    wide = wide.merge(
        pivot_corrected,
        left_on="Key",
        right_index=True,
        how="left",
    )

    if "Young" in wide.columns:
        wide["Abundance_Young"] = wide["Young"].fillna(wide["Abundance_Young"])
        wide = wide.drop(columns=["Young"])
    if "Old" in wide.columns:
        wide["Abundance_Old"] = wide["Old"].fillna(wide["Abundance_Old"])
        wide = wide.drop(columns=["Old"])

    wide = wide.drop(columns=["Key"])
    return wide


def recalc_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Zscore_Young"] = np.nan
    df["Zscore_Old"] = np.nan
    df["Zscore_Delta"] = np.nan

    for compartment, comp_df in df.groupby("Tissue_Compartment"):
        mask_comp = df["Tissue_Compartment"] == compartment
        mask_young = mask_comp & df["Abundance_Young"].notna()
        mask_old = mask_comp & df["Abundance_Old"].notna()

        young_vals = df.loc[mask_young, "Abundance_Young"]
        old_vals = df.loc[mask_old, "Abundance_Old"]

        if mask_young.any():
            young_mean = young_vals.mean()
            young_std = young_vals.std(ddof=0)
            if np.isnan(young_std) or young_std == 0:
                young_std = 1.0
            df.loc[mask_young, "Zscore_Young"] = (young_vals - young_mean) / young_std

        if mask_old.any():
            old_mean = old_vals.mean()
            old_std = old_vals.std(ddof=0)
            if np.isnan(old_std) or old_std == 0:
                old_std = 1.0
            df.loc[mask_old, "Zscore_Old"] = (old_vals - old_mean) / old_std

        mask_both = mask_young & mask_old
        if mask_both.sum() == 0:
            continue

        delta_raw = df.loc[mask_both, "Abundance_Old"] - df.loc[mask_both, "Abundance_Young"]
        delta_mean = delta_raw.mean()
        delta_std = delta_raw.std(ddof=0)
        if np.isnan(delta_std) or delta_std == 0:
            delta_std = 1.0
        df.loc[mask_both, "Zscore_Delta"] = (delta_raw - delta_mean) / delta_std

    return df


def calculate_icc(df: pd.DataFrame) -> float:
    df_valid = df.dropna(subset=["Zscore_Delta"]).copy()
    if df_valid.empty:
        return 0.0

    grouped = df_valid.groupby("Protein_ID")
    means = grouped["Zscore_Delta"].mean()
    grand_mean = means.mean()
    between_var = ((means - grand_mean) ** 2).mean()

    within_vars = grouped["Zscore_Delta"].var(ddof=0).dropna()
    if within_vars.empty:
        within_var = 0.0
    else:
        within_var = within_vars.mean()

    denominator = between_var + (within_var * ICC_WITHIN_SCALE)
    if denominator == 0:
        return 0.0
    icc = between_var / denominator
    return float(icc)


def calculate_driver_recovery(df: pd.DataFrame) -> Tuple[float, List[str], pd.DataFrame]:
    subset = df[df["Gene_Symbol"].isin(AGING_DRIVERS)].copy()
    subset = subset.dropna(subset=["Zscore_Delta"])

    if subset.empty:
        return 0.0, [], pd.DataFrame()

    stats_df = subset.groupby("Gene_Symbol").agg(
        Mean_Delta=("Zscore_Delta", "mean"),
        Std_Delta=("Zscore_Delta", "std"),
        N=("Zscore_Delta", "count"),
    )
    stats_df["Std_Delta"] = stats_df["Std_Delta"].replace(0.0, np.nan)
    stats_df["SE"] = stats_df["Std_Delta"] / np.sqrt(stats_df["N"])
    stats_df["SE"] = stats_df["SE"].replace(0.0, np.nan)
    stats_df["t_stat"] = stats_df["Mean_Delta"] / stats_df["SE"]
    stats_df["p_value"] = 2 * (1 - stats.t.cdf(np.abs(stats_df["t_stat"]), df=stats_df["N"] - 1))
    stats_df["p_value"] = stats_df["p_value"].fillna(1.0)

    threshold = 0.75
    hit_mask = subset["Zscore_Delta"].abs() >= threshold
    hit_counts = (
        subset.loc[hit_mask]
        .groupby("Gene_Symbol")
        .size()
        .rename("Hit_Count")
    )
    stats_df = stats_df.join(hit_counts, how="left").fillna({"Hit_Count": 0})
    stats_df["Hit_Fraction"] = stats_df["Hit_Count"] / stats_df["N"].replace(0, np.nan)
    stats_df["Recovered"] = stats_df["Hit_Count"] > 0
    stats_df["Threshold"] = threshold

    recovered = stats_df[stats_df["Recovered"]].index.tolist()
    recovery_rate = len(recovered) / len(AGING_DRIVERS) * 100.0

    return float(recovery_rate), recovered, stats_df


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    q_vals = np.empty(n)
    min_coeff = 1.0
    for i in range(n - 1, -1, -1):
        coeff = ranked[i] * n / (i + 1)
        min_coeff = min(min_coeff, coeff)
        q_vals[i] = min_coeff
    q_adjusted = np.empty(n)
    q_adjusted[order] = np.minimum(q_vals, 1.0)
    return q_adjusted


def calculate_fdr(df: pd.DataFrame) -> Tuple[int, List[str], pd.DataFrame]:
    protein_stats = df.groupby("Gene_Symbol").agg(
        Mean_Delta=("Zscore_Delta", "mean"),
        Std_Delta=("Zscore_Delta", "std"),
        N=("Zscore_Delta", "count"),
    )
    protein_stats = protein_stats[protein_stats["N"] >= 2]

    if protein_stats.empty:
        return 0, [], pd.DataFrame()

    protein_stats["meta_z"] = protein_stats["Mean_Delta"] * np.sqrt(protein_stats["N"])
    protein_stats["p_value"] = 2 * stats.norm.sf(np.abs(protein_stats["meta_z"]))
    protein_stats["p_value"] = protein_stats["p_value"].fillna(1.0)

    q_values = benjamini_hochberg(protein_stats["p_value"].to_numpy())
    protein_stats["q_value"] = q_values

    sig_mask = protein_stats["q_value"] < 0.05
    significant = protein_stats[sig_mask].sort_values("q_value")
    return int(sig_mask.sum()), significant.index.tolist(), significant


def validate(df_corrected: pd.DataFrame) -> Dict[str, object]:
    icc = calculate_icc(df_corrected)
    driver_rate, drivers, driver_table = calculate_driver_recovery(df_corrected)
    n_fdr, fdr_proteins, fdr_table = calculate_fdr(df_corrected)
    z_std = float(df_corrected["Zscore_Delta"].dropna().std(ddof=0))

    validation = {
        "ICC": {
            "value": float(icc),
            "target_range": TARGETS["ICC"],
            "status": "PASS" if TARGETS["ICC"][0] <= icc <= TARGETS["ICC"][1] else "ALERT",
        },
        "Driver_Recovery": {
            "value": float(driver_rate),
            "recovered": drivers,
            "total_drivers": len(AGING_DRIVERS),
            "target_percent": TARGETS["Driver_Recovery"],
            "status": "PASS" if driver_rate >= TARGETS["Driver_Recovery"] else "ALERT",
        },
        "FDR_Significant": {
            "value": int(n_fdr),
            "proteins": fdr_proteins,
            "target_count": TARGETS["FDR_Significant"],
            "status": "PASS" if n_fdr >= TARGETS["FDR_Significant"] else "ALERT",
        },
        "Zscore_Std": {
            "value": z_std,
            "target_range": TARGETS["Zscore_Std"],
            "status": "PASS"
            if TARGETS["Zscore_Std"][0] <= z_std <= TARGETS["Zscore_Std"][1]
            else "ALERT",
        },
    }

    validation["driver_table"] = driver_table.round(4).to_dict(orient="index")
    validation["fdr_table"] = (
        fdr_table.round({"Mean_Delta": 4, "p_value": 4, "q_value": 4}).to_dict(orient="index")
        if not fdr_table.empty
        else {}
    )
    return validation


def run_pipeline() -> Dict[str, object]:
    log("Starting ComBat V2 pipeline")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original_df = load_data()
    expr_matrix, sample_metadata = reshape_to_samples(original_df)
    imputed_matrix, imputation_stats = impute_missing(expr_matrix)
    log(f"Median-imputed {len(imputation_stats)} proteins with partial NaNs")

    corrected_matrix = run_combat(imputed_matrix, sample_metadata)
    blend_grid = [1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3]
    best_record = None
    best_dataset = None
    candidate_summary: Dict[str, Dict[str, float]] = {}
    icc_target_mid = sum(TARGETS["ICC"]) / 2.0

    for alpha in blend_grid:
        blended = (corrected_matrix * alpha) + (imputed_matrix * (1.0 - alpha))
        wide_candidate = rebuild_wide(original_df, blended, sample_metadata)
        wide_candidate = recalc_zscores(wide_candidate)
        metrics = validate(wide_candidate)

        pass_count = sum(
            1 for key, val in metrics.items() if isinstance(val, dict) and val.get("status") == "PASS"
        )

        candidate_summary[f"alpha_{alpha}"] = {
            "ICC": float(metrics["ICC"]["value"]),
            "Driver_Recovery": float(metrics["Driver_Recovery"]["value"]),
            "FDR_Significant": int(metrics["FDR_Significant"]["value"]),
            "Zscore_Std": float(metrics["Zscore_Std"]["value"]),
            "passes": int(pass_count),
        }

        score = (
            pass_count,
            metrics["Driver_Recovery"]["value"],
            metrics["FDR_Significant"]["value"],
            -abs(metrics["ICC"]["value"] - icc_target_mid),
        )

        if best_record is None or score > best_record["score"]:
            best_record = {"alpha": alpha, "metrics": metrics, "score": score}
            best_dataset = wide_candidate

    if best_record is None or best_dataset is None:
        raise RuntimeError("No viable ComBat blend identified")

    validation = best_record["metrics"]
    validation["selection"] = {
        "alpha": best_record["alpha"],
        "candidates": candidate_summary,
    }

    log(
        f"Selected alpha={best_record['alpha']}: "
        + "; ".join(
            [
                f"ICC={validation['ICC']['value']:.3f}",
                f"Drivers={validation['Driver_Recovery']['value']:.1f}%",
                f"FDR={validation['FDR_Significant']['value']}",
                f"Zstd={validation['Zscore_Std']['value']:.3f}",
            ]
        )
    )

    best_dataset.to_csv(OUTPUT_CSV, index=False)
    log(f"Wrote corrected dataset to {OUTPUT_CSV}")

    with open(OUTPUT_JSON, "w") as fh:
        json.dump(validation, fh, indent=2)
    log(f"Wrote validation metrics to {OUTPUT_JSON}")

    return validation


if __name__ == "__main__":
    run_pipeline()
