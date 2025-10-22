#!/usr/bin/env python3
"""Simulate serpin knockouts and quantify impact on ECM network connectivity."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

random.seed(42)

AGENT = "codex"
BASE_DIR = Path(__file__).resolve().parent
EDGE_PATH = BASE_DIR / f"network_edges_{AGENT}.csv"
CENTRALITY_PATH = BASE_DIR / f"centrality_all_metrics_{AGENT}.csv"
KNOCKOUT_PATH = BASE_DIR / f"knockout_impact_{AGENT}.csv"
CORRELATION_PATH = BASE_DIR / f"centrality_knockout_correlation_{AGENT}.csv"
VIS_DIR = BASE_DIR / f"visualizations_{AGENT}"

SAMPLES = 40


def load_graph() -> nx.Graph:
    edges_df = pd.read_csv(EDGE_PATH)
    graph = nx.Graph()
    for row in edges_df.itertuples(index=False):
        graph.add_edge(
            row.protein_a,
            row.protein_b,
            weight=float(row.abs_rho),
        )
    for _, _, data in graph.edges(data=True):
        w = data.get("weight", 1.0)
        data["distance"] = 1.0 / w if w else float("inf")
    return graph


def largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    components = list(nx.connected_components(graph))
    largest_nodes = max(components, key=len)
    return graph.subgraph(largest_nodes).copy()


def approx_average_shortest_path(graph: nx.Graph, samples: int = SAMPLES) -> float:
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return float("nan")
    k = min(samples, len(nodes))
    sample_nodes = random.sample(nodes, k)
    distances_sum = 0.0
    count = 0
    for node in sample_nodes:
        lengths = nx.single_source_dijkstra_path_length(graph, node, weight="distance")
        if node in lengths:
            lengths.pop(node)
        distances_sum += sum(lengths.values())
        count += len(lengths)
    return distances_sum / count if count else float("nan")


def global_metrics(graph: nx.Graph) -> Dict[str, float]:
    largest = largest_component_subgraph(graph)
    metrics = {
        "edge_count": graph.number_of_edges(),
        "node_count": graph.number_of_nodes(),
        "component_count": nx.number_connected_components(graph),
        "largest_component_nodes": largest.number_of_nodes(),
        "largest_component_edges": largest.number_of_edges(),
        "approx_avg_shortest_path": approx_average_shortest_path(largest),
    }
    metrics["approx_global_efficiency"] = (
        1.0 / metrics["approx_avg_shortest_path"] if metrics["approx_avg_shortest_path"] not in (0, float("nan")) else float("nan")
    )
    return metrics


def knockout_analysis() -> pd.DataFrame:
    graph = load_graph()
    baseline = global_metrics(graph)
    centrality_df = pd.read_csv(CENTRALITY_PATH).set_index("gene")
    serpins = [node for node in graph if node.upper().startswith("SERPIN")]
    records: List[Dict[str, float]] = []
    for gene in serpins:
        knocked = graph.copy()
        if knocked.has_node(gene):
            knocked.remove_node(gene)
        metrics = global_metrics(knocked)
        record = {"gene": gene}
        for key, value in metrics.items():
            record[f"ko_{key}"] = value
            base_val = baseline.get(key)
            if base_val in (None, 0, float("nan")):
                record[f"delta_{key}"] = np.nan
            else:
                record[f"delta_{key}"] = base_val - value
        records.append(record)
    ko_df = pd.DataFrame(records)
    ko_df = ko_df.join(centrality_df, on="gene")
    ko_df.to_csv(KNOCKOUT_PATH, index=False)
    return ko_df


def centrality_vs_knockout(ko_df: pd.DataFrame, impact_metric: str = "delta_approx_global_efficiency") -> pd.DataFrame:
    centrality_cols = [
        col
        for col in [
            "degree_centrality",
            "strength",
            "betweenness_centrality",
            "closeness_centrality",
            "harmonic_centrality",
            "eigenvector_centrality",
            "pagerank",
            "clustering_coefficient",
            "core_number",
        ]
        if col in ko_df.columns
    ]
    valid = ko_df.dropna(subset=[impact_metric])
    results = []
    for metric in centrality_cols:
        subset = valid.dropna(subset=[metric])
        if subset.empty:
            continue
        rho, pval = spearmanr(subset[metric], subset[impact_metric])
        results.append({"metric": metric, "impact_metric": impact_metric, "spearman_rho": rho, "p_value": pval})
    corr_df = pd.DataFrame(results)
    corr_df.sort_values("spearman_rho", ascending=False, inplace=True)
    corr_df.to_csv(CORRELATION_PATH, index=False)
    return corr_df


def plot_knockout_scatter(ko_df: pd.DataFrame, metric: str, impact_metric: str) -> None:
    if metric not in ko_df.columns or impact_metric not in ko_df.columns:
        return
    VIS_DIR.mkdir(exist_ok=True)
    import seaborn as sns

    subset = ko_df.dropna(subset=[metric, impact_metric])
    if subset.empty:
        return
    plt.figure(figsize=(7, 5))
    sns.regplot(x=metric, y=impact_metric, data=subset, scatter_kws={"s": 60}, line_kws={"color": "black"})
    for _, row in subset.iterrows():
        plt.annotate(row["gene"], (row[metric], row[impact_metric]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel(impact_metric.replace("_", " ").title())
    plt.title(f"Knockout Impact vs {metric}")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"knockout_impact_scatter_{AGENT}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    ko_df = knockout_analysis()
    corr_df = centrality_vs_knockout(ko_df)
    if not corr_df.empty:
        top_metric = corr_df.iloc[0]["metric"]
        plot_knockout_scatter(ko_df, top_metric, "delta_approx_global_efficiency")
