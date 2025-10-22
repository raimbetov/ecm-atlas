"""Pseudo-time comparison pipeline for H11 (agent: codex).

This script prepares the ECM aging dataset, constructs pseudo-time orderings
using multiple trajectory inference methods, and consolidates the results into
summary tables required for downstream benchmarking.

Outputs:
- pseudotime_orderings_codex.csv
- intermediate/tissue_gene_matrix.csv
- intermediate/pseudotime_diffusion_codex.csv
- intermediate/pseudotime_slingshot_codex.csv

The script computes three methods directly in Python (velocity, PCA, diffusion
maps) and attempts to delegate Slingshot to an R helper script. If the R
execution fails because the environment lacks the required Bioconductor
packages, a principled Slingshot-inspired fallback implemented in Python is
used instead (principal graph over cluster MST with geodesic pseudotime).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
)
VELOCITY_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_01/hypothesis_03_tissue_aging_clocks/claude_code/tissue_aging_velocity_claude_code.csv"
)
AGENT = "codex"
INTERMEDIATE_DIR = BASE_DIR / "intermediate"
OUTPUT_CSV = BASE_DIR / f"pseudotime_orderings_{AGENT}.csv"
DIFFUSION_OUTPUT = INTERMEDIATE_DIR / f"pseudotime_diffusion_{AGENT}.csv"
SLINGSHOT_OUTPUT = INTERMEDIATE_DIR / f"pseudotime_slingshot_{AGENT}.csv"
TISSUE_MATRIX_OUTPUT = INTERMEDIATE_DIR / "tissue_gene_matrix.csv"


def ensure_directories() -> None:
    INTERMEDIATE_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"Tissue", "Gene_Symbol", "Zscore_Delta"}
    if missing := required.difference(df.columns):
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def build_tissue_gene_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(values="Zscore_Delta", index="Tissue", columns="Gene_Symbol", aggfunc="mean")
    pivot = pivot.fillna(0.0)
    return pivot


def normalize_ranks(scores: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(scores), dtype=float)
    ranks = arr.argsort().argsort().astype(float)
    if len(arr) > 1:
        ranks /= (len(arr) - 1)
    else:
        ranks[:] = 0.0
    return ranks


def compute_velocity_pseudotime(matrix: pd.DataFrame, velocity_path: Path) -> pd.DataFrame:
    vel_df = pd.read_csv(velocity_path)
    if "Tissue" not in vel_df.columns or "Velocity" not in vel_df.columns:
        raise ValueError("Velocity file missing required columns 'Tissue' and 'Velocity'.")
    vel_map: Dict[str, float] = dict(zip(vel_df["Tissue"], vel_df["Velocity"]))
    alias_map = {
        'Brain_Cortex': 'Cortex',
        'Brain_Hippocampus': 'Hippocampus',
        'Heart_Native_Tissue': 'Native_Tissue',
        'Heart_Decellularized_Tissue': 'Decellularized_Tissue',
        'Kidney_Glomerular': 'Glomerular',
        'Kidney_Tubulointerstitial': 'Tubulointerstitial',
        'Intervertebral_disc_OAF': 'OAF',
        'Intervertebral_disc_IAF': 'IAF',
        'Intervertebral_disc_NP': 'NP',
        'Ovary_Cortex': 'Ovary',
    }
    tissues = matrix.index.tolist()
    scores = []
    missing = []
    for tissue in tissues:
        key = tissue
        if key not in vel_map and key in alias_map:
            key = alias_map[key]
        if key not in vel_map:
            missing.append(tissue)
            continue
        scores.append(vel_map[key])
    if missing:
        raise KeyError(f"Velocity scores missing for tissues: {missing}")
    norm = normalize_ranks(scores)
    return pd.DataFrame(
        {
            "method": "velocity",
            "Tissue": tissues,
            "pseudo_time_score": scores,
            "normalized_pseudo_time": norm,
            "rank": np.argsort(np.argsort(scores)) + 1,
        }
    )


def compute_pca_pseudotime(matrix: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(matrix.values)
    flat = scores[:, 0]
    norm = normalize_ranks(flat)
    return pd.DataFrame(
        {
            "method": "pca",
            "Tissue": matrix.index.tolist(),
            "pseudo_time_score": flat,
            "normalized_pseudo_time": norm,
            "rank": np.argsort(np.argsort(flat)) + 1,
        }
    )


def compute_diffusion_pseudotime(matrix: pd.DataFrame) -> pd.DataFrame:
    try:
        from pydiffmap import diffusion_map as dm
        from pydiffmap import kernel as dm_kernel
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pydiffmap is required for diffusion pseudotime. Install via 'pip install pydiffmap'."
        ) from exc
    k_neighbors = min(8, max(2, matrix.shape[0] - 1))
    ker = dm_kernel.Kernel(kernel_type='gaussian', k=k_neighbors, epsilon='bgh')
    ker.fit(matrix.values)
    mapper = dm.DiffusionMap(ker, alpha=0.5, n_evecs=3)
    embedding = mapper.fit_transform(matrix.values)
    scores = embedding[:, 0]
    norm = normalize_ranks(scores)
    df = pd.DataFrame(
        {
            "method": "diffusion",
            "Tissue": matrix.index.tolist(),
            "pseudo_time_score": scores,
            "normalized_pseudo_time": norm,
            "rank": np.argsort(np.argsort(scores)) + 1,
        }
    )
    df.to_csv(DIFFUSION_OUTPUT, index=False)
    return df


def run_r_script(script_name: str, input_path: Path, output_path: Path) -> None:
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required R script not found: {script_path}")
    cmd = ["Rscript", str(script_path), str(input_path), str(output_path)]
    subprocess.run(cmd, check=True)


def slingshot_fallback(matrix: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix.values)
    pca = PCA(n_components=min(3, scaled.shape[1]), random_state=42)
    coords = pca.fit_transform(scaled)
    dist_matrix = squareform(pdist(coords))
    graph = nx.Graph()
    for i in range(dist_matrix.shape[0]):
        for j in range(i + 1, dist_matrix.shape[0]):
            graph.add_edge(i, j, weight=float(dist_matrix[i, j]))
    mst = nx.minimum_spanning_tree(graph)
    start_idx = int(np.argmin(coords[:, 0]))
    lengths = nx.single_source_dijkstra_path_length(mst, start_idx)
    scores = np.array([lengths[i] for i in range(len(matrix))])
    norm = normalize_ranks(scores)
    df = pd.DataFrame(
        {
            "method": "slingshot",
            "Tissue": matrix.index.tolist(),
            "pseudo_time_score": scores,
            "normalized_pseudo_time": norm,
            "rank": np.argsort(np.argsort(scores)) + 1,
            "fallback_used": True,
        }
    )
    df.to_csv(SLINGSHOT_OUTPUT, index=False)
    return df


def compute_slingshot_pseudotime(matrix: pd.DataFrame) -> pd.DataFrame:
    try:
        run_r_script("slingshot_trajectory_codex.R", TISSUE_MATRIX_OUTPUT, SLINGSHOT_OUTPUT)
    except subprocess.CalledProcessError as exc:
        print("Warning: Slingshot R script failed, falling back to Python approximation.")
        print(exc)
        return slingshot_fallback(matrix)
    if not SLINGSHOT_OUTPUT.exists():
        print("Warning: Slingshot output missing, using fallback implementation.")
        return slingshot_fallback(matrix)
    df = pd.read_csv(SLINGSHOT_OUTPUT)
    required_cols = {"Tissue", "pseudo_time_score"}
    if not required_cols.issubset(df.columns):
        print("Warning: Slingshot output missing required columns, using fallback implementation.")
        return slingshot_fallback(matrix)
    df = df.sort_values("pseudo_time_score").reset_index(drop=True)
    df["method"] = "slingshot"
    df["normalized_pseudo_time"] = normalize_ranks(df["pseudo_time_score"].to_numpy())
    df["rank"] = np.arange(1, len(df) + 1)
    df["fallback_used"] = False
    return df


def save_matrix(matrix: pd.DataFrame) -> None:
    matrix.to_csv(TISSUE_MATRIX_OUTPUT)


def aggregate_results(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    if "fallback_used" not in combined.columns:
        combined["fallback_used"] = False
    else:
        combined["fallback_used"] = combined["fallback_used"].fillna(False)
    combined = combined.sort_values(["method", "rank"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_CSV, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct pseudo-time orderings for H11.")
    parser.add_argument("--skip-slingshot", action="store_true", help="Skip running the Slingshot R script (for debugging).")
    args = parser.parse_args()

    ensure_directories()
    df = load_dataset(DATA_PATH)
    tissue_matrix = build_tissue_gene_matrix(df)
    save_matrix(tissue_matrix)

    frames = [
        compute_velocity_pseudotime(tissue_matrix, VELOCITY_PATH),
        compute_pca_pseudotime(tissue_matrix),
        compute_diffusion_pseudotime(tissue_matrix),
    ]
    if args.skip_slingshot:
        print("Skipping Slingshot per CLI flag.")
    else:
        frames.append(compute_slingshot_pseudotime(tissue_matrix))

    aggregate_results(frames)
    print(f"Saved consolidated pseudo-time orderings to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
