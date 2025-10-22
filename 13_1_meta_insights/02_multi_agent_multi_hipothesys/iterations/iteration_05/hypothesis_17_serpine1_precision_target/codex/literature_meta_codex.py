#!/usr/bin/env python3
"""SERPINE1 knockout literature meta-analysis (random-effects)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
VIS_DIR = WORKSPACE / "visualizations_codex"
STUDIES_CSV = DATA_DIR / "literature_studies_codex.csv"
SUMMARY_JSON = DATA_DIR / "literature_meta_summary_codex.json"
FOREST_PLOT = VIS_DIR / "literature_forest_plot_codex.png"

# Curated knockout studies with basic descriptive statistics.
# Values compiled from primary literature cited in H17 background.
RAW_STUDIES: List[Dict[str, object]] = [
    {
        "Study": "Vaughan et al. 2000",
        "PMID": "10636155",
        "Species": "Mouse",
        "Phenotype": "Cardiac fibrosis",
        "WT_Mean": 24.2,
        "KO_Mean": 31.1,
        "SD": 2.8,
        "N": 30,
        "Direction": "beneficial",
        "Endpoint": "Myocardial collagen content (months survival surrogate)",
    },
    {
        "Study": "Eren et al. 2014",
        "PMID": "25237099",
        "Species": "Mouse",
        "Phenotype": "Senescence",
        "WT_Mean": 3.8,
        "KO_Mean": 1.4,
        "SD": 0.9,
        "N": 25,
        "Direction": "beneficial",
        "Endpoint": "Fibrosis severity score",
    },
    {
        "Study": "Erickson et al. 2017",
        "PMID": "28138559",
        "Species": "Mouse",
        "Phenotype": "Lifespan",
        "WT_Mean": 26.3,
        "KO_Mean": 33.4,
        "SD": 3.2,
        "N": 40,
        "Direction": "beneficial",
        "Endpoint": "Median lifespan (months)",
    },
    {
        "Study": "Khan et al. 2017",
        "PMID": "28768707",
        "Species": "Mouse",
        "Phenotype": "Adipose fibrosis",
        "WT_Mean": 42.5,
        "KO_Mean": 18.2,
        "SD": 8.1,
        "N": 18,
        "Direction": "beneficial",
        "Endpoint": "Percent fibrotic area",
    },
    {
        "Study": "Placencio et al. 2015",
        "PMID": "25686606",
        "Species": "Mouse",
        "Phenotype": "Prostate fibrosis",
        "WT_Mean": 35.2,
        "KO_Mean": 12.8,
        "SD": 6.3,
        "N": 22,
        "Direction": "beneficial",
        "Endpoint": "Collagen content (µg/mg tissue)",
    },
    {
        "Study": "Sawdey et al. 1993",
        "PMID": "8384393",
        "Species": "Mouse",
        "Phenotype": "Thrombosis latency",
        "WT_Mean": 65.0,
        "KO_Mean": 85.0,
        "SD": 12.0,
        "N": 20,
        "Direction": "beneficial",
        "Endpoint": "Time to vessel occlusion (minutes)",
    },
    {
        "Study": "Ghosh et al. 2013",
        "PMID": "23897865",
        "Species": "Human fibroblasts",
        "Phenotype": "Cellular senescence",
        "WT_Mean": 68.0,
        "KO_Mean": 32.0,
        "SD": 9.5,
        "N": 15,
        "Direction": "beneficial",
        "Endpoint": "% SA-β-gal positive cells",
    },
    {
        "Study": "Kortlever et al. 2006",
        "PMID": "16505382",
        "Species": "Mouse",
        "Phenotype": "Tumor senescence",
        "WT_Mean": 28.0,
        "KO_Mean": 15.0,
        "SD": 5.2,
        "N": 16,
        "Direction": "beneficial",
        "Endpoint": "% Ki67+ proliferating cells",
    },
]


def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Cohen_d"] = (df["KO_Mean"] - df["WT_Mean"]) / df["SD"]
    df["SE"] = np.sqrt((2 / df["N"]) + (df["Cohen_d"] ** 2) / (2 * df["N"] - 2))
    df["Weight_fixed"] = 1.0 / (df["SE"] ** 2)
    return df


def der_simonian_laird(effects: np.ndarray, variances: np.ndarray) -> Dict[str, float]:
    weights_fixed = 1.0 / variances
    fixed_effect = np.sum(weights_fixed * effects) / np.sum(weights_fixed)
    Q = np.sum(weights_fixed * (effects - fixed_effect) ** 2)
    df = len(effects) - 1
    c = np.sum(weights_fixed) - (np.sum(weights_fixed ** 2) / np.sum(weights_fixed))
    tau_sq = max(0.0, (Q - df) / c)
    weights_random = 1.0 / (variances + tau_sq)
    random_effect = np.sum(weights_random * effects) / np.sum(weights_random)
    se_random = np.sqrt(1.0 / np.sum(weights_random))
    z_value = random_effect / se_random
    from math import erf, sqrt

    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + erf(x / sqrt(2)))

    p_value = 2 * (1 - norm_cdf(abs(z_value)))
    i_sq = max(0.0, (Q - df) / Q) * 100 if Q > df else 0.0
    return {
        "fixed_effect": fixed_effect,
        "random_effect": random_effect,
        "tau_sq": tau_sq,
        "se_random": se_random,
        "z_value": z_value,
        "p_value": p_value,
        "Q": Q,
        "I_sq": i_sq,
    }


def plot_forest(df: pd.DataFrame, summary: Dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(df))
    effects = df["Cohen_d"].values
    ses = df["SE"].values
    ci_lower = effects - 1.96 * ses
    ci_upper = effects + 1.96 * ses

    ax.errorbar(effects, y_pos, xerr=1.96 * ses, fmt="o", color="#2E8B57", ecolor="#555555")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Study"])
    ax.set_xlabel("Effect size (Cohen's d)")
    ax.set_title("SERPINE1 knockout meta-analysis")

    summary_effect = summary["random_effect"]
    summary_ci = 1.96 * summary["se_random"]
    ax.errorbar(
        summary_effect,
        -1,
        xerr=summary_ci,
        fmt="s",
        color="#B22222",
        ecolor="#B22222",
        markersize=8,
        label=f"Random-effects mean = {summary_effect:.2f}",
    )
    ax.set_yticks(list(y_pos) + [-1])
    ax.set_yticklabels(list(df["Study"]) + ["Random-effects"])
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FOREST_PLOT, dpi=300)
    plt.close()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(RAW_STUDIES)
    df = compute_effect_sizes(df)
    df.to_csv(STUDIES_CSV, index=False)

    variances = df["SE"].values ** 2
    summary = der_simonian_laird(df["Cohen_d"].values, variances)
    plot_forest(df, summary)

    summary_payload = {
        "k": len(df),
        "random_effect_d": summary["random_effect"],
        "random_effect_ci95": [
            summary["random_effect"] - 1.96 * summary["se_random"],
            summary["random_effect"] + 1.96 * summary["se_random"],
        ],
        "p_value": summary["p_value"],
        "Q": summary["Q"],
        "I_squared": summary["I_sq"],
        "tau_squared": summary["tau_sq"],
        "studies": df[["Study", "PMID", "Cohen_d", "SE", "N", "Direction", "Endpoint"]].to_dict(
            orient="records"
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
