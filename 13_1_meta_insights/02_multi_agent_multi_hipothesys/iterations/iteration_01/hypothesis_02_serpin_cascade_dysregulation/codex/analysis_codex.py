#!/usr/bin/env python3
"""Serpin cascade dysregulation analysis for codex agent."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import patches
from scipy import stats

BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas")
DATA_PATH = BASE_DIR / "08_merged_ecm_dataset" / "merged_ecm_aging_zscore.csv"
OUTPUT_DIR = BASE_DIR / "13_1_meta_insights" / "02_multi_agent_multi_hipothesys" / "iterations" / "iteration_01" / "hypothesis_02_serpin_cascade_dysregulation" / "codex"
VIS_DIR = OUTPUT_DIR / "visualizations_codex"
SERPIN_PROFILE_PATH = OUTPUT_DIR / "serpin_comprehensive_profile_codex.csv"
CENTRALITY_PATH = OUTPUT_DIR / "network_centrality_codex.csv"
PATHWAY_PATH = OUTPUT_DIR / "pathway_dysregulation_codex.csv"

SERPIN_KEYWORDS = ("SERPIN", "A2M", "PZP")

FUNCTION_MAP: Dict[str, str] = {
    "A2M": "Protease sink",
    "A2ML1": "Protease sink",
    "PZP": "Protease sink",
    "SERPINA1": "Anti-protease",
    "SERPINA3": "Inflammation control",
    "SERPINA4": "Kallikrein inhibitor",
    "SERPINA5": "Anticoagulant",
    "SERPINA6": "Hormone transport",
    "SERPINA7": "Angiotensin regulation",
    "SERPINA10": "Coagulation regulator",
    "SERPINA12": "Metabolic modulator",
    "SERPINB1": "Inflammation control",
    "SERPINB2": "Fibrinolysis inhibitor",
    "SERPINB3": "Barrier protease shield",
    "SERPINB4": "Barrier protease shield",
    "SERPINB5": "Tumor suppressor",
    "SERPINB6": "Cysteine protease inhibitor",
    "SERPINC1": "Antithrombin",
    "SERPIND1": "Heparin cofactor",
    "SERPINF1": "ECM modulator",
    "SERPINF2": "Plasmin inhibitor",
    "SERPINE1": "Fibrinolysis inhibitor",
    "SERPINE2": "Protease inhibitor",
    "SERPINE3": "Steroid binding",
    "SERPING1": "Complement regulator",
    "SERPINH1": "Collagen chaperone",
}

PATHWAY_MAP: Dict[str, List[str]] = {
    "SERPINC1": ["Coagulation"],
    "SERPIND1": ["Coagulation"],
    "SERPINF1": ["ECM Assembly"],
    "SERPINF2": ["Coagulation", "Fibrinolysis"],
    "SERPINE1": ["Fibrinolysis", "Inflammation"],
    "SERPINE2": ["Fibrinolysis"],
    "SERPINE3": ["Hormone Transport"],
    "SERPINB1": ["Inflammation"],
    "SERPINB2": ["Fibrinolysis", "Inflammation"],
    "SERPINB3": ["Inflammation", "ECM Assembly"],
    "SERPINB4": ["Inflammation", "ECM Assembly"],
    "SERPINB5": ["ECM Assembly"],
    "SERPINB6": ["Inflammation"],
    "SERPINA1": ["Inflammation", "Protease Control"],
    "SERPINA3": ["Inflammation", "ECM Assembly"],
    "SERPINA4": ["Coagulation"],
    "SERPINA5": ["Coagulation", "Fibrinolysis"],
    "SERPINA6": ["Hormone Transport"],
    "SERPINA7": ["Protease Control"],
    "SERPINA10": ["Coagulation"],
    "SERPINA12": ["Metabolic"],
    "SERPING1": ["Complement"],
    "SERPINH1": ["ECM Assembly"],
    "A2M": ["Protease Control", "Inflammation"],
    "A2ML1": ["Protease Control"],
    "PZP": ["Protease Control", "Inflammation"],
}

PRIMARY_PATHWAYS = [
    "Coagulation",
    "Inflammation",
    "ECM Assembly",
    "Fibrinolysis",
    "Complement",
    "Protease Control",
    "Hormone Transport",
    "Metabolic",
]
VENN_PATHWAYS = ["Coagulation", "Inflammation", "ECM Assembly"]


@dataclass
class CorrelationEdge:
    source: str
    target: str
    rho: float
    p_value: float


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def identify_serpins(df: pd.DataFrame) -> pd.Series:
    def row_is_serpin(row: pd.Series) -> bool:
        names: List[str] = []
        for field in ("Canonical_Gene_Symbol", "Gene_Symbol"):
            value = row.get(field)
            if isinstance(value, str):
                parts = [part.strip().upper() for part in value.split(";") if part]
                names.extend(parts)
        combined = ";".join(names)
        return any(keyword in combined for keyword in SERPIN_KEYWORDS)

    return df.apply(row_is_serpin, axis=1)


def compute_directional_consistency(group: pd.DataFrame) -> float:
    study_means = group.groupby("Study_ID")[["Zscore_Delta"]].mean().dropna()
    if study_means.empty:
        return float("nan")
    study_means = study_means[study_means["Zscore_Delta"] != 0]
    if study_means.empty:
        return float("nan")
    overall = study_means["Zscore_Delta"].mean()
    dominant_sign = math.copysign(1, overall) if overall != 0 else 0
    if dominant_sign == 0:
        summary = study_means["Zscore_Delta"].sum()
        dominant_sign = math.copysign(1, summary) if summary != 0 else 0
    if dominant_sign == 0:
        return float("nan")
    matches = (np.sign(study_means["Zscore_Delta"]) == dominant_sign).sum()
    return matches / len(study_means)


def lookup_mapping(symbol: str, mapping: Dict[str, Sequence[str] | str]) -> Sequence[str] | str | None:
    if symbol in mapping:
        return mapping[symbol]
    for key in mapping:
        if symbol.startswith(key):
            return mapping[key]
    return None


def build_serpin_profile(df: pd.DataFrame, serpin_mask: pd.Series) -> pd.DataFrame:
    serpin_df = df.loc[serpin_mask].copy()
    serpin_df["Canonical_Normalized"] = serpin_df["Canonical_Gene_Symbol"].str.upper()
    records: List[Dict[str, object]] = []

    for canon, group in serpin_df.groupby("Canonical_Normalized"):
        mean_delta = group["Zscore_Delta"].mean()
        mean_abs_delta = group["Zscore_Delta"].abs().mean()
        std_delta = group["Zscore_Delta"].std()
        tissues = group["Tissue_Compartment"].dropna().nunique()
        studies = group["Study_ID"].dropna().nunique()
        consistency = compute_directional_consistency(group)
        matrisome = (
            group["Matrisome_Category_Simplified"].dropna().mode().iat[0]
            if not group["Matrisome_Category_Simplified"].dropna().empty
            else None
        )
        function_entry = lookup_mapping(canon, FUNCTION_MAP)
        pathways_entry = lookup_mapping(canon, PATHWAY_MAP)
        function_class = function_entry if isinstance(function_entry, str) else "Other or Unknown"
        pathways: List[str]
        if isinstance(pathways_entry, list):
            pathways = list(pathways_entry)
        else:
            pathways = []
        records.append({
            "canonical_symbol": canon,
            "mean_delta_z": mean_delta,
            "mean_abs_delta_z": mean_abs_delta,
            "std_delta_z": std_delta,
            "tissue_breadth": tissues,
            "study_count": studies,
            "directional_consistency": consistency,
            "matrisome_category": matrisome,
            "function_category": function_class,
            "pathways": ";".join(pathways) if pathways else "",
            "entropy_metric": np.nan,
        })

    profile_df = pd.DataFrame.from_records(records)
    profile_df.sort_values(by="mean_abs_delta_z", ascending=False, inplace=True)
    return profile_df


def mann_whitney_serpin_vs_others(df: pd.DataFrame, serpin_mask: pd.Series) -> Tuple[float, float]:
    serpin_values = df.loc[serpin_mask, "Zscore_Delta"].abs().dropna()
    other_values = df.loc[~serpin_mask, "Zscore_Delta"].abs().dropna()
    if serpin_values.empty or other_values.empty:
        return float("nan"), float("nan")
    stat, p_value = stats.mannwhitneyu(serpin_values, other_values, alternative="two-sided")
    effect = serpin_values.mean() - other_values.mean()
    return p_value, effect


def make_expression_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Canonical_Normalized"] = df["Canonical_Gene_Symbol"].str.upper()
    df["Sample_Key"] = df["Study_ID"].fillna("NA") + "|" + df["Tissue_Compartment"].fillna("NA")
    matrix = df.pivot_table(
        index="Canonical_Normalized",
        columns="Sample_Key",
        values="Zscore_Delta",
        aggfunc="mean",
    ).sort_index()
    return matrix


def compute_pairwise_spearman(matrix: pd.DataFrame, threshold: float = 0.5, alpha: float = 0.05) -> List[CorrelationEdge]:
    edges: List[CorrelationEdge] = []
    genes = matrix.index.tolist()
    for i, j in itertools.combinations(range(len(genes)), 2):
        row_i = matrix.iloc[i, :]
        row_j = matrix.iloc[j, :]
        overlap = (~row_i.isna()) & (~row_j.isna())
        if overlap.sum() < 4:
            continue
        rho, p_value = stats.spearmanr(row_i[overlap], row_j[overlap])
        if np.isnan(rho) or np.isnan(p_value):
            continue
        if abs(rho) >= threshold and p_value < alpha:
            edges.append(CorrelationEdge(genes[i], genes[j], float(rho), float(p_value)))
    return edges


def build_network(edges: Iterable[CorrelationEdge]) -> nx.Graph:
    graph = nx.Graph()
    for edge in edges:
        weight = abs(edge.rho)
        graph.add_edge(edge.source, edge.target, weight=weight, rho=edge.rho, p_value=edge.p_value)
    return graph


def compute_centrality(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["canonical_symbol", "degree", "betweenness", "eigenvector"])
    degree_dict = dict(graph.degree(weight=None))
    betweenness_dict = nx.betweenness_centrality(graph, weight=None, normalized=True)
    eigenvector_dict = nx.eigenvector_centrality(graph, weight=None, max_iter=500)
    centrality_df = pd.DataFrame({
        "canonical_symbol": list(degree_dict.keys()),
        "degree": list(degree_dict.values()),
        "betweenness": [betweenness_dict.get(node, 0.0) for node in degree_dict.keys()],
        "eigenvector": [eigenvector_dict.get(node, 0.0) for node in degree_dict.keys()],
    })
    return centrality_df


def annotate_centrality(centrality_df: pd.DataFrame, serpin_profile: pd.DataFrame) -> pd.DataFrame:
    serpin_set = set(serpin_profile["canonical_symbol"])
    annotated = centrality_df.copy()
    annotated["is_serpin"] = annotated["canonical_symbol"].isin(serpin_set)
    profile_lookup = serpin_profile.set_index("canonical_symbol")
    annotated["function_category"] = annotated["canonical_symbol"].map(profile_lookup["function_category"])  # type: ignore[index]
    annotated["pathways"] = annotated["canonical_symbol"].map(profile_lookup["pathways"])  # type: ignore[index]
    return annotated


def summarize_pathways(serpin_profile: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    serpin_lookup = serpin_profile.set_index("canonical_symbol")
    for pathway in PRIMARY_PATHWAYS:
        members = [
            symbol
            for symbol, entry in serpin_lookup["pathways"].items()
            if entry and pathway in entry.split(";")
        ]
        if not members:
            continue
        subset = df[df["Canonical_Gene_Symbol"].str.upper().isin(members)]
        if subset.empty:
            continue
        mean_abs = subset["Zscore_Delta"].abs().mean()
        median_abs = subset["Zscore_Delta"].abs().median()
        rows.append({
            "pathway": pathway,
            "serpin_count": len(set(members)),
            "mean_abs_delta_z": mean_abs,
            "median_abs_delta_z": median_abs,
        })
    result = pd.DataFrame(rows).sort_values(by="mean_abs_delta_z", ascending=False)
    return result


def prepare_pathway_matrix(serpin_profile: pd.DataFrame) -> pd.DataFrame:
    data: Dict[str, List[int]] = {pathway: [] for pathway in PRIMARY_PATHWAYS}
    for _, row in serpin_profile.iterrows():
        pathways = row["pathways"].split(";") if row["pathways"] else []
        for pathway in PRIMARY_PATHWAYS:
            data[pathway].append(1 if pathway in pathways else 0)
    matrix = pd.DataFrame(data)
    matrix.insert(0, "canonical_symbol", serpin_profile["canonical_symbol"].tolist())
    return matrix


def create_network_visual(graph: nx.Graph, serpin_profile: pd.DataFrame) -> None:
    if graph.number_of_nodes() == 0:
        return
    serpin_set = set(serpin_profile["canonical_symbol"])
    eigenvector = nx.eigenvector_centrality(graph, weight=None, max_iter=500)
    sizes = [300 + (eigenvector.get(node, 0) * 2000) for node in graph.nodes]
    colors = ["red" if node in serpin_set else "lightgray" for node in graph.nodes]
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    serpin_labels = {node: node for node in graph.nodes if node in serpin_set}
    nx.draw_networkx_labels(graph, pos, labels=serpin_labels, font_size=8, font_color="black")
    plt.title("Protein Correlation Network Highlighting Serpins")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "serpin_network_codex.png", dpi=300)
    plt.close()


def create_pathway_bar(pathway_stats: pd.DataFrame) -> None:
    if pathway_stats.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(pathway_stats["pathway"], pathway_stats["mean_abs_delta_z"], color="#2a9d8f")
    plt.ylabel("Mean |Δz|")
    plt.xticks(rotation=45, ha="right")
    plt.title("Serpin Pathway Dysregulation Severity")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "pathway_dysregulation_codex.png", dpi=300)
    plt.close()


def create_venn_diagram(serpin_profile: pd.DataFrame) -> None:
    pathway_sets = {path: set() for path in VENN_PATHWAYS}
    for _, row in serpin_profile.iterrows():
        pathways = row["pathways"].split(";") if row["pathways"] else []
        for path in VENN_PATHWAYS:
            if path in pathways:
                pathway_sets[path].add(row["canonical_symbol"])
    a, b, c = (pathway_sets[path] for path in VENN_PATHWAYS)
    only_a = len(a - b - c)
    only_b = len(b - a - c)
    only_c = len(c - a - b)
    ab = len((a & b) - c)
    ac = len((a & c) - b)
    bc = len((b & c) - a)
    abc = len(a & b & c)

    fig, ax = plt.subplots(figsize=(6, 6))
    circle_a = patches.Circle((0.4, 0.62), 0.35, color="#f4a261", alpha=0.4)
    circle_b = patches.Circle((0.6, 0.62), 0.35, color="#e76f51", alpha=0.4)
    circle_c = patches.Circle((0.5, 0.35), 0.35, color="#264653", alpha=0.4)
    for circle in (circle_a, circle_b, circle_c):
        ax.add_patch(circle)
    ax.text(0.22, 0.78, str(only_a), ha="center", va="center")
    ax.text(0.78, 0.78, str(only_b), ha="center", va="center", color="white")
    ax.text(0.5, 0.08, str(only_c), ha="center", va="center", color="white")
    ax.text(0.5, 0.78, str(ab), ha="center", va="center")
    ax.text(0.35, 0.52, str(ac), ha="center", va="center", color="white")
    ax.text(0.65, 0.52, str(bc), ha="center", va="center", color="white")
    ax.text(0.5, 0.5, str(abc), ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    ax.text(0.3, 0.95, VENN_PATHWAYS[0], ha="center", fontsize=10)
    ax.text(0.7, 0.95, VENN_PATHWAYS[1], ha="center", fontsize=10)
    ax.text(0.5, 0.02, VENN_PATHWAYS[2], ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.title("Serpin Pathway Overlap")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "serpin_pathway_overlap_codex.png", dpi=300)
    plt.close()


def create_temporal_plot(df: pd.DataFrame, serpin_mask: pd.Series) -> None:
    serpin_values = df.loc[serpin_mask].groupby(df.loc[serpin_mask, "Canonical_Gene_Symbol"].str.upper())["Zscore_Delta"].apply(lambda x: x.abs().mean())
    target_values = df.loc[~serpin_mask].groupby(df.loc[~serpin_mask, "Canonical_Gene_Symbol"].str.upper())["Zscore_Delta"].apply(lambda x: x.abs().mean())
    serpin_mean = serpin_values.mean() if not serpin_values.empty else 0
    target_mean = target_values.mean() if not target_values.empty else 0
    plt.figure(figsize=(5, 5))
    plt.bar(["Serpins", "Non-serpin targets"], [serpin_mean, target_mean], color=["#d62828", "#003049"])
    plt.ylabel("Mean |Δz|")
    plt.title("Temporal Proxy: Serpin vs Target Dysregulation")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "temporal_proxy_codex.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_directories()
    df = pd.read_csv(DATA_PATH)

    serpin_mask = identify_serpins(df)
    profile_df = build_serpin_profile(df, serpin_mask)
    profile_df.to_csv(SERPIN_PROFILE_PATH, index=False)

    mw_p, mw_effect = mann_whitney_serpin_vs_others(df, serpin_mask)
    stats_path = OUTPUT_DIR / "codex_serpin_vs_nonserpin_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as handle:
        handle.write(f"Mann-Whitney p-value: {mw_p}\n")
        handle.write(f"Mean |Δz| difference (serpin - others): {mw_effect}\n")

    matrix = make_expression_matrix(df)
    edges = compute_pairwise_spearman(matrix)
    graph = build_network(edges)
    centrality_df = compute_centrality(graph)
    annotated_centrality = annotate_centrality(centrality_df, profile_df)
    annotated_centrality.to_csv(CENTRALITY_PATH, index=False)

    pathway_stats = summarize_pathways(profile_df, df)
    pathway_stats.to_csv(PATHWAY_PATH, index=False)

    pathway_matrix = prepare_pathway_matrix(profile_df)
    pathway_matrix.to_csv(OUTPUT_DIR / "serpin_pathway_matrix_codex.csv", index=False)

    create_network_visual(graph, profile_df)
    create_pathway_bar(pathway_stats)
    create_venn_diagram(profile_df)
    create_temporal_plot(df, serpin_mask)

    summary_path = OUTPUT_DIR / "codex_analysis_log.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Total serpins: {len(profile_df)}\n")
        handle.write(f"Total proteins: {matrix.shape[0]}\n")
        handle.write(f"Network nodes: {graph.number_of_nodes()}\n")
        handle.write(f"Network edges: {graph.number_of_edges()}\n")
        handle.write(f"Mann-Whitney p-value: {mw_p}\n")
        handle.write(f"Serpin vs others mean |Δz| diff: {mw_effect}\n")


if __name__ == "__main__":
    main()
