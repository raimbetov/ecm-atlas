#!/usr/bin/env python3
"""SERPINE1 knockout perturbation analysis using the pre-trained GAT model from H05."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

# Reuse the trained GAT and data preparation pipeline from hypothesis H05
H05_DIR = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_02/hypothesis_05_gnn_aging_networks/codex"
)
import sys

sys.path.append(str(H05_DIR))
from analysis_gnn_codex import ProteinGAT, prepare_graph  # type: ignore

WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
VIS_DIR = WORKSPACE / "visualizations_codex"
WEIGHTS_PATH = H05_DIR / "gnn_weights_codex.pth"
OUTPUT_CSV = DATA_DIR / "knockout_cascade_codex.csv"
SUMMARY_JSON = DATA_DIR / "knockout_summary_codex.json"
WATERFALL_PNG = VIS_DIR / "knockout_waterfall_codex.png"
NETWORK_PNG = VIS_DIR / "network_perturbation_codex.png"
TARGET_GENE = "SERPINE1"
DOWNSTREAM_MARKERS = ["LOX", "TGM2", "COL1A1"]


def compute_expected_scores(logits: torch.Tensor) -> torch.Tensor:
    """Convert class logits into an expected aging score in [0, 2]."""
    weights = torch.tensor([0.0, 1.0, 2.0], dtype=logits.dtype, device=logits.device)
    probs = torch.softmax(logits, dim=1)
    return (probs * weights).sum(dim=1)


def knockout_perturbation() -> Dict[str, float]:
    graph_data = prepare_graph()
    num_features = graph_data.features.size(1)
    num_classes = int(graph_data.labels.max().item()) + 1

    model = ProteinGAT(num_features=num_features, num_classes=num_classes)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        baseline_logits, baseline_embeddings, _ = model(
            graph_data.features, graph_data.edge_index
        )
        baseline_scores = compute_expected_scores(baseline_logits)

    try:
        target_idx = graph_data.proteins.index(TARGET_GENE)
    except ValueError as exc:
        raise SystemExit(f"Target gene {TARGET_GENE} not found in protein list") from exc

    knockout_features = graph_data.features.clone()
    knockout_features[target_idx] = 0.0

    with torch.no_grad():
        knockout_logits, knockout_embeddings, _ = model(
            knockout_features, graph_data.edge_index
        )
        knockout_scores = compute_expected_scores(knockout_logits)

    score_delta = baseline_scores - knockout_scores
    percent_change = (score_delta / (baseline_scores.abs() + 1e-6)) * 100.0
    embedding_shift = torch.linalg.norm(
        baseline_embeddings - knockout_embeddings, dim=1
    )

    df = pd.DataFrame(
        {
            "Gene": graph_data.proteins,
            "Baseline_Score": baseline_scores.numpy(),
            "Knockout_Score": knockout_scores.numpy(),
            "Delta_Score": score_delta.numpy(),
            "Percent_Change": percent_change.numpy(),
            "Embedding_Shift": embedding_shift.numpy(),
        }
    )

    df = df.sort_values("Delta_Score", ascending=False).reset_index(drop=True)
    df_top = df.head(100)
    df_top.to_csv(OUTPUT_CSV, index=False)

    global_reduction = float(score_delta.mean().item())
    global_percent = float(
        (score_delta.mean() / (baseline_scores.mean() + 1e-6)).item() * 100.0
    )

    marker_stats = {
        marker: df.set_index("Gene").loc[marker].to_dict()
        for marker in DOWNSTREAM_MARKERS
        if marker in set(df["Gene"])
    }

    summary = {
        "target_gene": TARGET_GENE,
        "global_score_reduction": global_reduction,
        "global_percent_reduction": global_percent,
        "top_20_genes": df.head(20)["Gene"].tolist(),
        "downstream_markers": marker_stats,
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    plot_waterfall(df_top)
    plot_network(graph_data.graph, df)
    return summary


def plot_waterfall(df: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    x = np.arange(len(df))
    colors = ["#2E8B57" if val > 0 else "#B22222" for val in df["Percent_Change"]]
    plt.bar(x, df["Percent_Change"], color=colors)
    plt.xticks(x, df["Gene"], rotation=75, ha="right", fontsize=9)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Percent change in aging score")
    plt.title("Top 100 proteins impacted by SERPINE1 knockout (GNN perturbation)")
    plt.tight_layout()
    plt.savefig(WATERFALL_PNG, dpi=300)
    plt.close()


def plot_network(graph: nx.Graph, df: pd.DataFrame, top_n: int = 30) -> None:
    highlighted = df.head(top_n)
    impact = dict(zip(highlighted["Gene"], highlighted["Percent_Change"]))

    sub_nodes = set(impact.keys())
    neighbors = set()
    for node in sub_nodes:
        neighbors.update(graph.neighbors(node))
    subgraph = graph.subgraph(sub_nodes.union(neighbors)).copy()

    pos = nx.spring_layout(subgraph, seed=42)
    values = np.array([impact.get(node, 0.0) for node in subgraph.nodes()])
    vmax = max(values.max(), abs(values.min()))
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    node_colors = [cmap(norm(val)) for val in values]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_size=[200 + 30 * abs(impact.get(node, 0.0)) for node in subgraph.nodes()],
        node_color=node_colors,
        alpha=0.9,
    )
    nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
    nx.draw_networkx_labels(
        subgraph,
        pos,
        font_size=8,
        font_color="black",
    )
    plt.title("SERPINE1 knockout cascade network (top 30 impacted nodes)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, shrink=0.7, label="Percent change in aging score")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(NETWORK_PNG, dpi=300)
    plt.close()


if __name__ == "__main__":
    summary = knockout_perturbation()
    print(json.dumps(summary, indent=2))
