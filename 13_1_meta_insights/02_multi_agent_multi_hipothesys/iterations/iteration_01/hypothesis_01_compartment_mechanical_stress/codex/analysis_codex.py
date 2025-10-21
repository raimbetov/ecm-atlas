#!/usr/bin/env python3
"""
Analysis pipeline for Hypothesis 01: Compartment antagonistic mechanical stress adaptation.
"""
from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
VIS_DIR = OUTPUT_DIR / "visualizations_codex"

# Mechanical load labels draw from biomechanics literature on muscle fiber usage and disc compression.
LOAD_MAP: Dict[str, str] = {
    "Skeletal_muscle_Soleus": "high",
    "Skeletal_muscle_Gastrocnemius": "high",
    "NP": "high",
    "Skeletal_muscle_TA": "low",
    "Skeletal_muscle_EDL": "low",
    "IAF": "low",
    "OAF": "low",
}

STRUCTURAL_CATEGORIES = {"Collagens", "ECM Glycoproteins", "Proteoglycans"}
REGULATORY_CATEGORIES = {"ECM Regulators", "ECM-affiliated Proteins", "Secreted Factors"}


@dataclass
class CorrelationResult:
    subset: str
    test: str
    statistic: float
    p_value: float
    n: int
    n_high: int | None = None
    n_low: int | None = None
    effect_size: float | None = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "subset": self.subset,
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            "n_high": self.n_high,
            "n_low": self.n_low,
            "effect_size": self.effect_size,
        }


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    expected_cols = {"Tissue_Compartment", "Zscore_Delta", "Matrisome_Category", "Gene_Symbol"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def aggregate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicates so each gene-compartment pair is unique."""
    grouped = (
        df.groupby(
            [
                "Organ",
                "Tissue",
                "Tissue_Compartment",
                "Gene_Symbol",
                "Canonical_Gene_Symbol",
                "Matrisome_Category",
            ],
            dropna=False,
        )["Zscore_Delta"]
        .mean()
        .reset_index()
    )
    grouped["Load_Class"] = grouped["Tissue_Compartment"].map(LOAD_MAP)
    grouped["Load_Score"] = grouped["Load_Class"].map({"low": 0, "high": 1})
    grouped["Protein_Class"] = grouped["Matrisome_Category"].map(
        lambda cat: "Structural"
        if cat in STRUCTURAL_CATEGORIES
        else ("Regulatory" if cat in REGULATORY_CATEGORIES else "Other")
    )
    return grouped


def find_antagonistic_pairs(grouped: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for organ, organ_df in grouped.groupby("Organ"):
        pivot = organ_df.pivot_table(index="Gene_Symbol", columns="Tissue_Compartment", values="Zscore_Delta")
        canonical_map = organ_df.set_index("Gene_Symbol")["Canonical_Gene_Symbol"].to_dict()
        for gene, series in pivot.iterrows():
            for comp_a, comp_b in itertools.combinations(series.dropna().index.tolist(), 2):
                val_a = series[comp_a]
                val_b = series[comp_b]
                if val_a == 0 or val_b == 0:
                    continue
                if val_a * val_b < 0:  # opposite signs indicate antagonism
                    records.append(
                        {
                            "Tissue": organ,
                            "Gene_Symbol": gene,
                            "Canonical_Gene_Symbol": canonical_map.get(gene, gene),
                            "Compartment_A": comp_a,
                            "Compartment_B": comp_b,
                            "DeltaZ_A": val_a,
                            "DeltaZ_B": val_b,
                            "Antagonism_Magnitude": abs(val_a - val_b),
                        }
                    )
    antagonistic = pd.DataFrame(records)
    antagonistic.sort_values("Antagonism_Magnitude", ascending=False, inplace=True)
    return antagonistic


def compute_correlation_stats(grouped: pd.DataFrame) -> pd.DataFrame:
    results: List[CorrelationResult] = []

    struct_df = grouped[grouped["Protein_Class"] == "Structural"].dropna(subset=["Load_Score"])
    reg_df = grouped[grouped["Protein_Class"] == "Regulatory"].dropna(subset=["Load_Score"])

    if struct_df["Load_Score"].nunique() > 1:
        rho, p = spearmanr(struct_df["Load_Score"], struct_df["Zscore_Delta"])
        results.append(
            CorrelationResult(
                subset="Structural",
                test="Spearman",
                statistic=float(rho),
                p_value=float(p),
                n=len(struct_df),
            )
        )
    if reg_df["Load_Score"].nunique() > 1:
        rho, p = spearmanr(reg_df["Load_Score"], reg_df["Zscore_Delta"])
        results.append(
            CorrelationResult(
                subset="Regulatory",
                test="Spearman",
                statistic=float(rho),
                p_value=float(p),
                n=len(reg_df),
            )
        )

    struct_high = struct_df[struct_df["Load_Class"] == "high"]["Zscore_Delta"]
    struct_low = struct_df[struct_df["Load_Class"] == "low"]["Zscore_Delta"]
    if len(struct_high) > 0 and len(struct_low) > 0:
        U_stat, p_val = mannwhitneyu(struct_high, struct_low, alternative="two-sided")
        effect = float(struct_high.median() - struct_low.median())
        results.append(
            CorrelationResult(
                subset="Structural",
                test="Mann-Whitney_U",
                statistic=float(U_stat),
                p_value=float(p_val),
                n=len(struct_df),
                n_high=len(struct_high),
                n_low=len(struct_low),
                effect_size=effect,
            )
        )

    return pd.DataFrame(r.to_dict() for r in results)


def ensure_dirs() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_antagonism_heatmap(
    grouped: pd.DataFrame,
    antagonistic: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    top_events = antagonistic.head(top_n)
    key_pairs = top_events[["Gene_Symbol", "Tissue"]].drop_duplicates()
    rows: List[Dict[str, object]] = []
    for _, row in key_pairs.iterrows():
        gene = row["Gene_Symbol"]
        organ = row["Tissue"]
        subset = grouped[(grouped["Organ"] == organ) & (grouped["Gene_Symbol"] == gene)]
        label = f"{gene} ({organ})"
        for _, entry in subset.iterrows():
            rows.append(
                {
                    "Gene_Tissue": label,
                    "Compartment": entry["Tissue_Compartment"],
                    "Zscore_Delta": entry["Zscore_Delta"],
                }
            )
    if not rows:
        return
    heatmap_df = pd.DataFrame(rows)
    pivot = heatmap_df.pivot_table(index="Gene_Tissue", columns="Compartment", values="Zscore_Delta")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    plt.figure(figsize=(12, max(6, len(pivot) * 0.4)))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        linewidths=0.3,
        linecolor="lightgray",
        cbar_kws={"label": "Δz"},
    )
    plt.title("Top Antagonistic Proteins across Compartments")
    plt.xlabel("Compartment")
    plt.ylabel("Protein (Tissue)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_scatter_by_load(df: pd.DataFrame, title: str, output_path: Path) -> None:
    if df.empty:
        return
    np.random.seed(42)
    jitter = (np.random.rand(len(df)) - 0.5) * 0.1
    x_vals = df["Load_Score"].to_numpy(dtype=float) + jitter
    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, df["Zscore_Delta"], alpha=0.6, edgecolor="k", linewidth=0.3)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xticks([0, 1], ["Low load", "High load"])
    plt.title(title)
    plt.ylabel("Δz")
    plt.xlabel("Mechanical load state")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_antagonistic_bar(antagonistic: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top_events = antagonistic.head(top_n).copy()
    if top_events.empty:
        return
    top_events["Label"] = (
        top_events["Gene_Symbol"]
        + " ("
        + top_events["Compartment_A"]
        + " vs "
        + top_events["Compartment_B"]
        + ")"
    )
    plt.figure(figsize=(10, max(6, top_events.shape[0] * 0.4)))
    sns.barplot(
        data=top_events,
        y="Label",
        x="Antagonism_Magnitude",
        color="#1f77b4",
        edgecolor="black",
    )
    plt.xlabel("|Δz difference|")
    plt.ylabel("Protein (Compartment pair)")
    plt.title("Top Antagonistic Protein-Compartment Pairs")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_dataset(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns", file=sys.stderr)

    grouped = aggregate_measurements(df)
    print(
        "Aggregated measurements to unique gene-compartment pairs:",
        len(grouped),
        file=sys.stderr,
    )

    antagonistic = find_antagonistic_pairs(grouped)
    output_antagonism = OUTPUT_DIR / "antagonistic_pairs_codex.csv"
    antagonistic[
        [
            "Gene_Symbol",
            "Compartment_A",
            "Compartment_B",
            "DeltaZ_A",
            "DeltaZ_B",
            "Antagonism_Magnitude",
            "Tissue",
        ]
    ].to_csv(output_antagonism, index=False)
    print(
        f"Saved antagonistic pairs ({len(antagonistic)} rows) to {output_antagonism}",
        file=sys.stderr,
    )

    corr_df = compute_correlation_stats(grouped)
    output_corr = OUTPUT_DIR / "mechanical_stress_correlation_codex.csv"
    corr_df.to_csv(output_corr, index=False)
    print(f"Saved correlation stats to {output_corr}", file=sys.stderr)

    plot_antagonism_heatmap(
        grouped,
        antagonistic,
        VIS_DIR / "codex_antagonism_heatmap.png",
    )
    plot_scatter_by_load(
        grouped[(grouped["Protein_Class"] == "Structural") & grouped["Load_Score"].notna()],
        "Structural Proteins: Δz vs Mechanical Load",
        VIS_DIR / "codex_structural_scatter.png",
    )
    plot_scatter_by_load(
        grouped[(grouped["Protein_Class"] == "Regulatory") & grouped["Load_Score"].notna()],
        "Regulatory Proteins: Δz vs Mechanical Load",
        VIS_DIR / "codex_regulatory_scatter.png",
    )
    plot_top_antagonistic_bar(
        antagonistic,
        VIS_DIR / "codex_top_antagonistic_bar.png",
    )
    print(f"Saved figures to {VIS_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
