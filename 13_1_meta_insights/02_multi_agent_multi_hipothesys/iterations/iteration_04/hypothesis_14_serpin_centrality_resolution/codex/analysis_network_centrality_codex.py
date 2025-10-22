#!/usr/bin/env python3
"""Build ECM protein correlation network, compute centrality metrics, and export summaries."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

AGENT = "codex"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
OUTPUT_DIR = BASE_DIR
VIS_DIR = BASE_DIR / f"visualizations_{AGENT}"

EDGE_PATH = OUTPUT_DIR / f"network_edges_{AGENT}.csv"
STATS_PATH = OUTPUT_DIR / f"network_stats_{AGENT}.json"
CENTRALITY_PATH = OUTPUT_DIR / f"centrality_all_metrics_{AGENT}.csv"
SERPIN_RANKS_PATH = OUTPUT_DIR / f"serpin_rankings_{AGENT}.csv"
METRIC_CORR_PATH = OUTPUT_DIR / f"metric_correlation_matrix_{AGENT}.csv"
CONSENSUS_PATH = OUTPUT_DIR / f"consensus_centrality_{AGENT}.csv"

@dataclass
class NetworkArtifacts:
    graph: nx.Graph
    pivot: pd.DataFrame
    centrality: pd.DataFrame
    stats: Dict[str, float]


def build_expression_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sample_id"] = (
        df["Dataset_Name"].astype(str)
        + "|" + df["Organ"].astype(str)
        + "|" + df["Compartment"].astype(str)
    )
    pivot = df.pivot_table(
        index="sample_id",
        columns="Canonical_Gene_Symbol",
        values="Zscore_Delta",
        aggfunc="mean",
    )
    valid_cols = pivot.columns[pivot.std(axis=0, skipna=True) > 0]
    pivot = pivot.loc[:, valid_cols]
    pivot = pivot.fillna(0.0)
    return pivot


def build_network(matrix: pd.DataFrame, rho_threshold: float = 0.5, p_threshold: float = 0.05) -> nx.Graph:
    data = matrix.to_numpy()
    corr, pvals = spearmanr(data, axis=0)
    corr_df = pd.DataFrame(corr, index=matrix.columns, columns=matrix.columns)
    pval_df = pd.DataFrame(pvals, index=matrix.columns, columns=matrix.columns)
    graph = nx.Graph()
    for i, gene_i in enumerate(matrix.columns):
        graph.add_node(gene_i)
        for j in range(i + 1, len(matrix.columns)):
            gene_j = matrix.columns[j]
            rho = corr_df.iat[i, j]
            pval = pval_df.iat[i, j]
            if math.isnan(rho) or math.isnan(pval):
                continue
            if abs(rho) >= rho_threshold and pval < p_threshold:
                weight = abs(float(rho))
                sign = 1 if rho >= 0 else -1
                graph.add_edge(
                    gene_i,
                    gene_j,
                    weight=weight,
                    rho=float(rho),
                    sign=sign,
                    p_value=float(pval),
                )
    for u, v, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        data["distance"] = 1.0 / w if w else float("inf")
    return graph


def export_edges(graph: nx.Graph) -> None:
    rows = []
    for u, v, data in graph.edges(data=True):
        rows.append(
            {
                "protein_a": u,
                "protein_b": v,
                "spearman_rho": data.get("rho"),
                "abs_rho": data.get("weight"),
                "sign": data.get("sign"),
                "p_value": data.get("p_value"),
            }
        )
    edge_df = pd.DataFrame(rows)
    edge_df.to_csv(EDGE_PATH, index=False)


def compute_network_stats(graph: nx.Graph, matrix: pd.DataFrame) -> Dict[str, float]:
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "average_clustering": nx.average_clustering(graph, weight="weight") if graph.number_of_edges() else 0.0,
        "number_connected_components": nx.number_connected_components(graph),
        "matrix_samples": matrix.shape[0],
        "matrix_genes": matrix.shape[1],
    }
    degrees = [deg for _, deg in graph.degree(weight=None)]
    strengths = [deg for _, deg in graph.degree(weight="weight")]
    stats["average_degree"] = float(np.mean(degrees)) if degrees else 0.0
    stats["average_strength"] = float(np.mean(strengths)) if strengths else 0.0
    if graph.number_of_nodes() > 1:
        stats["global_efficiency"] = nx.global_efficiency(graph)
    else:
        stats["global_efficiency"] = 0.0
    try:
        stats["degree_assortativity"] = nx.degree_pearson_correlation_coefficient(graph)
    except Exception:
        stats["degree_assortativity"] = float("nan")
    components = list(nx.connected_components(graph))
    largest_component = max(components, key=len) if components else set()
    stats["largest_component_nodes"] = len(largest_component)
    stats["largest_component_fraction"] = (
        len(largest_component) / graph.number_of_nodes() if graph.number_of_nodes() else 0.0
    )
    return stats


def compute_centralities(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()
    metrics: Dict[str, Dict[str, float]] = {}
    metrics["degree_centrality"] = nx.degree_centrality(graph)
    metrics["strength"] = {node: data for node, data in graph.degree(weight="weight")}
    metrics["betweenness_centrality"] = nx.betweenness_centrality(graph, weight="distance")
    metrics["closeness_centrality"] = nx.closeness_centrality(graph, distance="distance")
    metrics["harmonic_centrality"] = nx.harmonic_centrality(graph, distance="distance")
    metrics["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(graph, weight="weight")
    metrics["pagerank"] = nx.pagerank(graph, weight="weight")
    metrics["clustering_coefficient"] = nx.clustering(graph, weight="weight")
    metrics["core_number"] = nx.core_number(graph)
    centrality_df = pd.DataFrame(metrics)
    centrality_df.index.name = "gene"
    centrality_df = centrality_df.sort_index()
    return centrality_df


def export_metric_tables(centrality_df: pd.DataFrame) -> None:
    centrality_df.to_csv(CENTRALITY_PATH)
    metric_corr = centrality_df.corr(method="spearman")
    metric_corr.to_csv(METRIC_CORR_PATH)
    normalized = centrality_df.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)
    consensus = normalized.mean(axis=1).to_frame(name="consensus_z")
    consensus["consensus_rank"] = consensus["consensus_z"].rank(ascending=False, method="min")
    consensus.sort_values("consensus_z", ascending=False, inplace=True)
    consensus.to_csv(CONSENSUS_PATH)


def export_serpin_rankings(centrality_df: pd.DataFrame) -> None:
    serpins = [g for g in centrality_df.index if g.upper().startswith("SERPIN")]
    if not serpins:
        empty_df = pd.DataFrame(columns=["gene"] + list(centrality_df.columns))
        empty_df.to_csv(SERPIN_RANKS_PATH, index=False)
        return
    ranks = centrality_df.rank(ascending=False, method="min")
    percentiles = ranks / len(centrality_df) * 100.0
    serpin_df = centrality_df.loc[serpins].copy()
    for col in centrality_df.columns:
        serpin_df[f"rank_{col}"] = ranks.loc[serpins, col]
        serpin_df[f"percentile_{col}"] = percentiles.loc[serpins, col]
    serpin_df.reset_index(inplace=True)
    serpin_df.rename(columns={"gene": "canonical_gene"}, inplace=True)
    serpin_df.to_csv(SERPIN_RANKS_PATH, index=False)


def ensure_directories() -> None:
    VIS_DIR.mkdir(exist_ok=True)


def plot_metric_heatmap(centrality_df: pd.DataFrame) -> None:
    corr = centrality_df.corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Centrality Metric Correlations")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"centrality_heatmap_{AGENT}.png", dpi=300)
    plt.close()


def plot_serpin_ranks(centrality_df: pd.DataFrame) -> None:
    serpins = [g for g in centrality_df.index if g.upper().startswith("SERPIN")]
    if not serpins:
        return
    ranks = centrality_df.rank(ascending=False, method="min")
    serpin_ranks = ranks.loc[serpins]
    melted = serpin_ranks.reset_index().melt(id_vars="gene", var_name="metric", value_name="rank")
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=melted, x="metric", y="rank", inner="point", scale="width")
    sns.stripplot(data=melted, x="metric", y="rank", hue="gene", dodge=True, linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rank (lower is more central)")
    plt.title("Serpin Rank Distribution Across Metrics")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"serpin_ranks_comparison_{AGENT}.png", dpi=300)
    plt.close()


def plot_betweenness_vs_eigenvector(centrality_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    serpins = centrality_df.index.str.upper().str.startswith("SERPIN")
    plt.scatter(
        centrality_df.loc[~serpins, "betweenness_centrality"],
        centrality_df.loc[~serpins, "eigenvector_centrality"],
        s=20,
        alpha=0.5,
        label="Other proteins",
    )
    plt.scatter(
        centrality_df.loc[serpins, "betweenness_centrality"],
        centrality_df.loc[serpins, "eigenvector_centrality"],
        s=60,
        color="red",
        edgecolor="black",
        label="SERPIN family",
    )
    plt.xlabel("Betweenness Centrality")
    plt.ylabel("Eigenvector Centrality")
    plt.title("Betweenness vs Eigenvector Centrality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"betweenness_vs_eigenvector_{AGENT}.png", dpi=300)
    plt.close()


def plot_network_overview(graph: nx.Graph) -> None:
    if graph.number_of_nodes() == 0 or graph.number_of_nodes() > 500:
        return
    serpins = {node for node in graph if node.upper().startswith("SERPIN")}
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(graph, pos, alpha=0.1, width=0.5)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=[n for n in graph if n not in serpins],
        node_size=20,
        node_color="#9ecae1",
        alpha=0.7,
    )
    if serpins:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=list(serpins),
            node_size=80,
            node_color="#de2d26",
            edgecolors="black",
        )
    plt.axis("off")
    plt.title("ECM Aging Network (Serpins Highlighted)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"network_graph_serpins_{AGENT}.png", dpi=300)
    plt.close()


def run_pipeline() -> NetworkArtifacts:
    ensure_directories()
    df = pd.read_csv(DATA_PATH)
    matrix = build_expression_matrix(df)
    graph = build_network(matrix)
    export_edges(graph)
    stats = compute_network_stats(graph, matrix)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    centrality_df = compute_centralities(graph)
    export_metric_tables(centrality_df)
    export_serpin_rankings(centrality_df)
    if not centrality_df.empty:
        plot_metric_heatmap(centrality_df)
        plot_serpin_ranks(centrality_df)
        plot_betweenness_vs_eigenvector(centrality_df)
    plot_network_overview(graph)
    return NetworkArtifacts(graph=graph, pivot=matrix, centrality=centrality_df, stats=stats)


if __name__ == "__main__":
    run_pipeline()
