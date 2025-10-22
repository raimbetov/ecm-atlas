#!/usr/bin/env python3
"""Run Louvain community detection on the ECM network and export module assignments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from community import community_louvain

AGENT = "codex"
BASE_DIR = Path(__file__).resolve().parent
EDGE_PATH = BASE_DIR / f"network_edges_{AGENT}.csv"
COMMUNITY_PATH = BASE_DIR / f"community_assignments_{AGENT}.csv"
VIS_DIR = BASE_DIR / f"visualizations_{AGENT}"


def load_graph() -> nx.Graph:
    edges_df = pd.read_csv(EDGE_PATH)
    graph = nx.Graph()
    for row in edges_df.itertuples(index=False):
        graph.add_edge(row.protein_a, row.protein_b, weight=float(row.abs_rho))
    return graph


def detect_communities(graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["gene", "community"])
    partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
    df = pd.DataFrame(list(partition.items()), columns=["gene", "community"])
    df.sort_values("community", inplace=True)
    df.to_csv(COMMUNITY_PATH, index=False)
    return df


def plot_communities(graph: nx.Graph, partition_df: pd.DataFrame) -> None:
    if graph.number_of_nodes() == 0 or partition_df.empty or graph.number_of_nodes() > 400:
        return
    VIS_DIR.mkdir(exist_ok=True)
    communities = partition_df.set_index("gene")["community"].to_dict()
    pos = nx.spring_layout(graph, seed=123)
    plt.figure(figsize=(10, 10))
    unique_comms = sorted(partition_df["community"].unique())
    cmap = plt.get_cmap("tab20")
    for idx, comm in enumerate(unique_comms):
        nodes = [node for node, c in communities.items() if c == comm]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_size=30,
            node_color=[cmap(idx % 20)],
            alpha=0.8,
        )
    nx.draw_networkx_edges(graph, pos, alpha=0.05)
    plt.axis("off")
    plt.title("Louvain Communities in ECM Network")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"community_modules_{AGENT}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    g = load_graph()
    df = detect_communities(g)
    plot_communities(g, df)
