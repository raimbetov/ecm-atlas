#!/usr/bin/env python3
"""End-to-end pipeline for ECM master regulator discovery using a custom Graph Attention Network.

This script builds a protein interaction network from ECM aging z-scores, trains a multi-layer GAT
for node classification, extracts embeddings, derives multiple importance scores, compares
communities, and exports all required artifacts for hypothesis H05 (agent: codex).
"""
from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import hdbscan
import community as community_louvain  # python-louvain

RANDOM_SEED = 42
EDGE_THRESHOLD = 0.5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
MAX_EPOCHS = 350
PATIENCE = 40
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
NUM_HEADS_LAYER1 = 4
NUM_HEADS_LAYER2 = 4
GRAD_THRESHOLD = 0.1
KNN_PAGERANK = 8
PERTURBATION_THRESHOLD_QUANTILE = 0.85

DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent
VIS_DIR = OUTPUT_DIR / "visualizations_codex"

warnings.filterwarnings("ignore", category=UserWarning)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class GraphData:
    proteins: List[str]
    features: torch.Tensor
    labels: torch.Tensor
    edge_index: torch.Tensor  # shape [num_edges, 2]
    edge_weight_directed: torch.Tensor  # shape [num_edges]
    undirected_edges: np.ndarray  # shape [num_edges, 2] without duplication
    edge_weights: np.ndarray
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    feature_names: List[str]
    metadata: pd.DataFrame
    graph: nx.Graph


class MultiHeadGATLayer(nn.Module):
    """Lightweight multi-head graph attention layer without PyG dependency."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.weight = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: [N, F], edge_index: [E, 2]
        num_nodes = x.size(0)
        src, dst = edge_index[:, 0], edge_index[:, 1]
        outputs = []
        alphas = []

        for head in range(self.num_heads):
            W = self.weight[head]  # [F_in, F_out]
            Wh = x @ W  # [N, F_out]
            Wh_src = Wh[src]
            Wh_dst = Wh[dst]
            # Compute attention logits
            cat = torch.cat([Wh_src, Wh_dst], dim=1)
            e = self.leaky_relu(torch.sum(cat * self.attn[head], dim=1))

            # Numerical stability using log-sum-exp trick per destination node
            max_per_dst = torch.full((num_nodes,), -float("inf"), device=x.device)
            max_per_dst.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
            max_per_dst = torch.where(
                torch.isfinite(max_per_dst), max_per_dst, torch.tensor(0.0, device=x.device)
            )
            e_normalized = e - max_per_dst[dst]
            exp_e = torch.exp(e_normalized)
            sum_per_dst = torch.zeros(num_nodes, device=x.device)
            sum_per_dst.scatter_add_(0, dst, exp_e)
            attn_coeff = exp_e / (sum_per_dst[dst] + 1e-9)

            attn_coeff = self.dropout(attn_coeff)
            out = torch.zeros(num_nodes, self.out_dim, device=x.device)
            out.index_add_(0, dst, attn_coeff.unsqueeze(1) * Wh_src)
            outputs.append(out)
            alphas.append(attn_coeff.detach())

        if self.concat:
            h = torch.cat(outputs, dim=1)
        else:
            h = torch.stack(outputs, dim=0).mean(dim=0)

        alpha_tensor = torch.stack(alphas, dim=1)  # [E, num_heads]
        return h, {"alpha": alpha_tensor}


class ProteinGAT(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.gat1 = MultiHeadGATLayer(
            in_dim=num_features,
            out_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS_LAYER1,
            dropout=0.25,
            concat=True,
        )
        self.gat2 = MultiHeadGATLayer(
            in_dim=HIDDEN_DIM * NUM_HEADS_LAYER1,
            out_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS_LAYER2,
            dropout=0.25,
            concat=False,
        )
        self.lin = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h1, attn1 = self.gat1(x, edge_index)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)
        h2, attn2 = self.gat2(h1, edge_index)
        h2 = F.elu(h2)
        logits = self.lin(h2)
        return logits, h2, {"layer1": attn1, "layer2": attn2}


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Gene_Symbol"].notna()].copy()
    df["Gene_Symbol"] = df["Gene_Symbol"].str.strip()
    df = df[df["Gene_Symbol"] != ""]
    df["Zscore_Delta"] = df["Zscore_Delta"].fillna(0.0)
    return df


def build_feature_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    grouping_cols = ["Gene_Symbol"]
    basic_stats = df.groupby(grouping_cols).agg(
        delta_mean=("Zscore_Delta", "mean"),
        delta_std=("Zscore_Delta", "std"),
        delta_median=("Zscore_Delta", "median"),
        delta_abs_mean=("Zscore_Delta", lambda x: np.mean(np.abs(x))),
        tissue_count=("Tissue", pd.Series.nunique),
        study_count=("Study_ID", pd.Series.nunique),
        n_observations=("Zscore_Delta", "count"),
    )
    basic_stats["delta_std"] = basic_stats["delta_std"].fillna(0.0)

    # Matrisome one-hot encodings (simplified categories)
    matrisome = (
        df[["Gene_Symbol", "Matrisome_Category_Simplified"]]
        .dropna()
        .drop_duplicates()
        .assign(value=1)
        .pivot_table(
            index="Gene_Symbol",
            columns="Matrisome_Category_Simplified",
            values="value",
            fill_value=0,
        )
    )
    matrisome.columns = [f"matrisome_{c.lower().replace(' ', '_')}" for c in matrisome.columns]

    feature_table = basic_stats.join(matrisome, how="left").fillna(0.0)
    feature_table = feature_table.sort_index()

    labels = feature_table["delta_mean"].apply(classify_direction)

    metadata = feature_table[["delta_mean", "tissue_count"]].copy()
    metadata["matrisome_category"] = (
        matrisome.idxmax(axis=1).reindex(metadata.index).fillna("Unknown")
    )

    return feature_table, labels.astype(int), metadata


def classify_direction(delta_mean: float, up_thresh: float = 0.5, down_thresh: float = -0.5) -> int:
    if delta_mean >= up_thresh:
        return 2  # upregulated
    if delta_mean <= down_thresh:
        return 0  # downregulated
    return 1  # stable


def build_correlation_network(
    df: pd.DataFrame, proteins: List[str], threshold: float = EDGE_THRESHOLD
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    pivot = (
        df.pivot_table(values="Zscore_Delta", index="Gene_Symbol", columns="Tissue", aggfunc="mean")
        .reindex(proteins)
    )
    pivot = pivot.fillna(0.0)
    corr = pivot.T.corr(method="spearman").fillna(0.0)

    edges = []
    weights = []
    for i, u in enumerate(proteins):
        for j in range(i + 1, len(proteins)):
            w = corr.iat[i, j]
            if abs(w) >= threshold:
                edges.append((u, proteins[j]))
                weights.append(w)

    G = nx.Graph()
    G.add_nodes_from(proteins)
    for (u, v), w in zip(edges, weights):
        G.add_edge(u, v, weight=float(w))

    # Ensure every node has at least one edge by connecting top correlations
    for node in proteins:
        if G.degree(node) == 0:
            correlations = corr.loc[node].copy()
            correlations = correlations.drop(node, errors="ignore")
            top_partners = correlations.abs().sort_values(ascending=False).head(3)
            for partner, w in top_partners.items():
                if partner not in G:
                    continue
                G.add_edge(node, partner, weight=float(corr.loc[node, partner]))

    edge_array = np.array([[proteins.index(u), proteins.index(v)] for u, v in G.edges()])
    weight_array = np.array([G[u][v]["weight"] for u, v in G.edges()])
    return G, edge_array, weight_array


def create_edge_index(
    edge_array: np.ndarray, edge_weights: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    if edge_array.size == 0:
        raise ValueError("No edges available to build edge_index.")
    reversed_edge = edge_array[:, [1, 0]]
    directed_edges = np.vstack([edge_array, reversed_edge])
    directed_weights = np.concatenate([edge_weights, edge_weights])
    return (
        torch.tensor(directed_edges, dtype=torch.long),
        torch.tensor(directed_weights, dtype=torch.float32),
    )


def standardize_features(feature_table: pd.DataFrame) -> Tuple[torch.Tensor, List[str]]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_table.values)
    features = torch.tensor(scaled, dtype=torch.float32)
    feature_names = list(feature_table.columns)
    return features, feature_names


def stratified_masks(labels: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = labels.values
    idx = np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_SEED)
    train_idx, tmp_idx = next(sss.split(idx, y))

    remaining_labels = y[tmp_idx]
    val_size = VAL_RATIO / (1 - TRAIN_RATIO)
    sss_val = StratifiedShuffleSplit(n_splits=1, train_size=val_size, random_state=RANDOM_SEED)
    val_idx_rel, test_idx_rel = next(sss_val.split(tmp_idx, remaining_labels))
    val_idx = tmp_idx[val_idx_rel]
    test_idx = tmp_idx[test_idx_rel]

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True
    return train_mask, val_mask, test_mask


def prepare_graph() -> GraphData:
    df = load_dataset(DATA_PATH)
    feature_table, labels, metadata = build_feature_table(df)
    proteins = list(feature_table.index)

    graph, undirected_edges, edge_weights = build_correlation_network(df, proteins)
    features, feature_names = standardize_features(feature_table)
    edge_index, edge_weight_directed = create_edge_index(undirected_edges, edge_weights)
    train_mask, val_mask, test_mask = stratified_masks(labels)

    return GraphData(
        proteins=proteins,
        features=features,
        labels=torch.tensor(labels.values, dtype=torch.long),
        edge_index=edge_index,
        edge_weight_directed=edge_weight_directed,
        undirected_edges=undirected_edges,
        edge_weights=edge_weights,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        feature_names=feature_names,
        metadata=metadata,
        graph=graph,
    )


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    classes, counts = labels.unique(return_counts=True)
    total = labels.size(0)
    weights = total / (len(classes) * counts.float())
    weight_tensor = torch.zeros(int(classes.max()) + 1)
    weight_tensor[classes] = weights
    return weight_tensor


def logits_to_metrics(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> Dict[str, float]:
    if mask.sum() == 0:
        return {"loss": math.nan, "accuracy": math.nan, "f1_macro": math.nan}
    preds = logits[mask].argmax(dim=1).cpu().numpy()
    true = labels[mask].cpu().numpy()
    return {
        "accuracy": accuracy_score(true, preds),
        "f1_macro": f1_score(true, preds, average="macro"),
    }


def train_model(data: GraphData) -> Tuple[ProteinGAT, List[Dict[str, float]], Dict[str, torch.Tensor]]:
    device = torch.device("cpu")
    model = ProteinGAT(num_features=data.features.size(1), num_classes=3).to(device)
    features = data.features.to(device)
    labels = data.labels.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    class_weights = compute_class_weights(labels)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    metrics_log: List[Dict[str, float]] = []
    best_val_f1 = -float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, _, _ = model(features, edge_index)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval, embeddings, attn_info = model(features, edge_index)
            train_metrics = logits_to_metrics(logits_eval, labels, train_mask)
            val_metrics = logits_to_metrics(logits_eval, labels, val_mask)
            val_loss = criterion(logits_eval[val_mask], labels[val_mask]).item()

        metrics_log.append(
            {
                "epoch": epoch,
                "split": "train",
                "loss": loss.item(),
                "accuracy": train_metrics["accuracy"],
                "f1_macro": train_metrics["f1_macro"],
            }
        )
        metrics_log.append(
            {
                "epoch": epoch,
                "split": "val",
                "loss": val_loss,
                "accuracy": val_metrics["accuracy"],
                "f1_macro": val_metrics["f1_macro"],
            }
        )

        if val_metrics["f1_macro"] > best_val_f1 + 1e-4:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "embeddings": embeddings.detach().cpu(),
                "logits": logits_eval.detach().cpu(),
                "attention": {
                    layer: info["alpha"].detach().cpu() if isinstance(info, dict) else info.detach().cpu()
                    for layer, info in attn_info.items()
                },
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state recorded during training.")

    model.load_state_dict(best_state["state_dict"])
    return model, metrics_log, best_state


def attention_importance(
    attn: torch.Tensor, edge_index: torch.Tensor, edge_weights: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Summed incoming attention per node across heads."""
    src = edge_index[:, 0].cpu()
    dst = edge_index[:, 1].cpu()
    alpha = attn  # [E, heads]
    weight = edge_weights.cpu().abs()
    incoming = torch.zeros((num_nodes, alpha.size(1)))
    for head in range(alpha.size(1)):
        incoming[:, head].index_add_(0, dst, alpha[:, head] * weight)
    return incoming.mean(dim=1)


def gradient_importance(
    model: ProteinGAT, features: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    model.eval()
    feat_clone = features.clone().detach().requires_grad_(True)
    logits, _, _ = model(feat_clone, edge_index)
    probs = F.softmax(logits, dim=1)
    top = probs.argmax(dim=1)
    selected = probs[torch.arange(probs.size(0)), top]
    selected.sum().backward()
    grads = feat_clone.grad.detach().abs().mean(dim=1)
    return grads.cpu()


def pagerank_importance(embeddings: np.ndarray, k: int = KNN_PAGERANK) -> np.ndarray:
    knn_graph = kneighbors_graph(embeddings, n_neighbors=min(k, embeddings.shape[0] - 1), mode="distance")
    knn_graph = 0.5 * (knn_graph + knn_graph.T)
    weights = np.exp(-knn_graph.toarray())
    np.fill_diagonal(weights, 0.0)
    G = nx.from_numpy_array(weights)
    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    pr_vector = np.array([pr[i] for i in range(len(pr))])
    return pr_vector


def summarize_master_regulators(
    proteins: List[str],
    attention_scores: torch.Tensor,
    grad_scores: torch.Tensor,
    pagerank_scores: np.ndarray,
    metadata: pd.DataFrame,
    graph: nx.Graph,
    top_n: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Protein": proteins,
            "attention_importance": attention_scores.numpy(),
            "gradient_importance": grad_scores.numpy(),
            "pagerank_embeddings": pagerank_scores,
            "degree_centrality": pd.Series(nx.degree_centrality(graph)),
            "betweenness_centrality": pd.Series(nx.betweenness_centrality(graph, normalized=True)),
            "delta_mean": metadata["delta_mean"].values,
            "tissue_count": metadata["tissue_count"].values,
            "matrisome_category": metadata["matrisome_category"].values,
        }
    ).set_index("Protein")

    df["composite_score"] = (
        df[["attention_importance", "gradient_importance", "pagerank_embeddings"]]
        .apply(lambda col: (col - col.min()) / (col.max() - col.min() + 1e-9))
        .mean(axis=1)
    )

    return df.sort_values("composite_score", ascending=False).head(top_n)


def perturbation_analysis(
    model: ProteinGAT,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    baseline_embeddings: torch.Tensor,
    candidates: List[int],
    proteins: List[str],
    quantile: float = PERTURBATION_THRESHOLD_QUANTILE,
) -> pd.DataFrame:
    model.eval()
    device = features.device
    baseline = baseline_embeddings.detach().cpu().numpy()
    results = []
    for idx in candidates:
        perturbed = features.clone().detach()
        perturbed[idx] = 0.0
        logits, emb, _ = model(perturbed, edge_index)
        emb = emb.detach().cpu().numpy()
        diff = np.linalg.norm(emb - baseline, axis=1)
        threshold = np.quantile(diff, quantile)
        cascade_size = int((diff > threshold).sum())
        mean_shift = float(diff.mean())
        max_shift = float(diff.max())
        results.append(
            {
                "master_regulator": proteins[idx],
                "cascade_size": cascade_size,
                "mean_embedding_shift": mean_shift,
                "max_embedding_shift": max_shift,
                "threshold_used": float(threshold),
            }
        )
    return pd.DataFrame(results)


def export_embeddings(
    embeddings: torch.Tensor,
    proteins: List[str],
    path: Path,
) -> None:
    df = pd.DataFrame(embeddings.numpy(), index=proteins)
    df.index.name = "Protein"
    df.to_csv(path)


def export_metrics(metrics_log: List[Dict[str, float]], path: Path) -> None:
    pd.DataFrame(metrics_log).to_csv(path, index=False)


def export_model_weights(model: ProteinGAT, path: Path) -> None:
    torch.save(model.state_dict(), path)


def export_graphml(graph: nx.Graph, path: Path) -> None:
    nx.write_graphml(graph, path)


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, proteins: List[str], path: Path) -> None:
    reducer = UMAP(random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.3)
    umap_coords = reducer.fit_transform(embeddings)
    label_names = {0: "Down", 1: "Stable", 2: "Up"}
    palette = ListedColormap(["#d73027", "#fee090", "#1a9850"])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=[labels[p] for p in range(len(labels))],
        cmap=palette,
        s=18,
        alpha=0.9,
    )
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=label_names[i],
                          markerfacecolor=palette(i), markersize=8) for i in label_names]
    plt.legend(handles=handles, title="Aging class", frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("UMAP of GNN embeddings (colored by aging direction)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_attention_heatmap(
    attention_scores: torch.Tensor,
    edge_index: torch.Tensor,
    proteins: List[str],
    path: Path,
    top_k: int = 30,
) -> None:
    attention = attention_scores.mean(dim=1).numpy()
    edges = edge_index.cpu().numpy()
    df = pd.DataFrame(
        {
            "source": [proteins[s] for s in edges[:, 0]],
            "target": [proteins[t] for t in edges[:, 1]],
            "attention": attention,
        }
    )
    df = df.sort_values("attention", ascending=False).head(top_k)
    pivot = df.pivot(index="source", columns="target", values="attention").fillna(0.0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="viridis")
    plt.title("Top attention-weighted edges")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def community_comparison(
    graph: nx.Graph,
    embeddings: np.ndarray,
    proteins: List[str],
    metadata: pd.DataFrame,
    path: Path,
) -> Dict[str, float]:
    graph_for_louvain = graph.copy()
    for u, v, data in graph_for_louvain.edges(data=True):
        data["weight"] = abs(float(data.get("weight", 0.0)))

    louvaine_partitions = community_louvain.best_partition(
        graph_for_louvain, weight="weight", random_state=RANDOM_SEED
    )
    louvaine_labels = np.array([louvaine_partitions[p] for p in proteins])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean")
    embedding_labels = clusterer.fit_predict(embeddings)

    if len(np.unique(embedding_labels[embedding_labels >= 0])) < 2:
        n_clusters = min(5, embeddings.shape[0] - 1)
        if n_clusters < 2:
            n_clusters = 2
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init="auto")
        embedding_labels = km.fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, embedding_labels)
        labels_for_stats = embedding_labels
    else:
        clustered_mask = embedding_labels >= 0
        if clustered_mask.sum() >= 2 and len(np.unique(embedding_labels[clustered_mask])) >= 2:
            silhouette = silhouette_score(
                embeddings[clustered_mask], embedding_labels[clustered_mask]
            )
        else:
            silhouette = float("nan")
        labels_for_stats = embedding_labels.copy()
        if (embedding_labels == -1).any():
            offset = labels_for_stats.max() + 1
            labels_for_stats[embedding_labels == -1] = (
                offset + np.arange((embedding_labels == -1).sum())
            )

    ari = adjusted_rand_score(louvaine_labels, labels_for_stats)

    matrisome_purity = (
        metadata.assign(cluster=labels_for_stats)
        .groupby("cluster")
        ["matrisome_category"]
        .apply(lambda s: s.value_counts(normalize=True).max())
        .mean()
    )

    # Visualization: scatter plot with both community assignments
    reducer = UMAP(random_state=RANDOM_SEED, n_neighbors=15, min_dist=0.3)
    umap_coords = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1], c=louvaine_labels, cmap="tab20", s=15)
    axes[0].set_title("Louvain communities")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1], c=embedding_labels, cmap="tab20", s=15)
    axes[1].set_title("Embedding HDBSCAN clusters")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return {
        "ARI": float(ari),
        "Silhouette": float(silhouette),
        "MatrisomePurity": float(matrisome_purity),
        "n_louvain": int(len(np.unique(louvaine_labels))),
        "n_embedding_clusters": int(len(np.unique(embedding_labels))),
    }


def plot_network_snapshot(
    graph: nx.Graph,
    metadata: pd.DataFrame,
    master_regulators: Iterable[str],
    path: Path,
    max_edges: int = 400,
) -> None:
    ranked_edges = sorted(
        graph.edges(data=True), key=lambda item: abs(item[2].get("weight", 0.0)), reverse=True
    )
    sub_edges = ranked_edges[: max_edges]
    nodes = {u for u, v, _ in sub_edges} | {v for u, v, _ in sub_edges}
    subgraph = graph.subgraph(nodes).copy()

    node_list = list(subgraph.nodes())
    delta = metadata.loc[node_list]["delta_mean"].astype(float)
    norm = plt.Normalize(vmin=delta.min(), vmax=delta.max())
    cmap = plt.colormaps["RdYlGn"]
    node_colors = [cmap(norm(delta[node])) for node in subgraph.nodes()]
    node_sizes = 80 + 5 * metadata.loc[node_list]["tissue_count"].astype(float)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(subgraph, weight="weight", seed=RANDOM_SEED, k=0.25)
    edge_colors = [abs(subgraph[u][v].get("weight", 0.0)) for u, v in subgraph.edges()]
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edge_color=edge_colors,
        edge_cmap=plt.colormaps["Blues"],
        alpha=0.4,
        width=1.0,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=[2.5 if node in master_regulators else 0.5 for node in subgraph.nodes()],
        edgecolors=["black" if node in master_regulators else "#555555" for node in subgraph.nodes()],
        ax=ax,
    )
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels={node: node for node in master_regulators if node in subgraph},
        font_size=8,
        font_color="black",
        ax=ax,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Δz mean")
    ax.set_title("Protein interaction subnetwork with highlighted master regulators")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def export_summary(
    master_regulators: pd.DataFrame,
    community_stats: Dict[str, float],
    metrics_log: List[Dict[str, float]],
    metadata: pd.DataFrame,
    path: Path,
) -> None:
    best_val = (
        pd.DataFrame(metrics_log)
        .query("split == 'val'")
        .sort_values("f1_macro", ascending=False)
        .head(1)
        .squeeze()
    )
    with path.open("w") as fh:
        fh.write("# H05 – Master Regulator Discovery (codex)\n\n")
        fh.write("## Training Performance\n")
        fh.write(
            f"- Best epoch: {int(best_val['epoch'])}\n"
            f"- Validation F1 (macro): {best_val['f1_macro']:.3f}\n"
            f"- Validation accuracy: {best_val['accuracy']:.3f}\n"
        )
        fh.write("\n## Master Regulators (Top 10)\n")
        fh.write(master_regulators.to_markdown())
        fh.write("\n\n## Community Comparison\n")
        fh.write(
            json.dumps(community_stats, indent=2)
        )
        fh.write(
            "\n\n## Notes\n"
            "- Aging direction labels use Δz thresholds (up ≥ 0.5, down ≤ -0.5).\n"
            "- Attention, gradient, and PageRank scores were z-normalized before ranking.\n"
            "- Perturbation cascade size counts proteins whose embeddings shift above the 85th percentile.\n"
        )


def main() -> None:
    set_seeds(RANDOM_SEED)
    VIS_DIR.mkdir(exist_ok=True)

    graph_data = prepare_graph()
    model, metrics_log, best_state = train_model(graph_data)

    export_metrics(metrics_log, OUTPUT_DIR / "gnn_training_metrics_codex.csv")
    export_model_weights(model, OUTPUT_DIR / "gnn_weights_codex.pth")
    export_graphml(graph_data.graph, OUTPUT_DIR / "network_graph_codex.graphml")

    embeddings = best_state["embeddings"]
    export_embeddings(embeddings, graph_data.proteins, OUTPUT_DIR / "protein_embeddings_gnn_codex.csv")

    attention_layer2 = best_state["attention"]["layer2"]
    attention_scores = attention_importance(
        attention_layer2,
        graph_data.edge_index,
        graph_data.edge_weight_directed,
        len(graph_data.proteins),
    )
    gradient_scores = gradient_importance(
        model, graph_data.features, graph_data.edge_index
    )
    pagerank_scores = pagerank_importance(embeddings.numpy())

    master_regulators_df = summarize_master_regulators(
        graph_data.proteins,
        attention_scores,
        gradient_scores,
        pagerank_scores,
        graph_data.metadata,
        graph_data.graph,
        top_n=10,
    )
    master_regulators_df.to_csv(OUTPUT_DIR / "master_regulators_codex.csv")

    top_indices = [graph_data.proteins.index(p) for p in master_regulators_df.index]
    perturb_df = perturbation_analysis(
        model,
        graph_data.features,
        graph_data.edge_index,
        embeddings,
        top_indices,
        graph_data.proteins,
    )
    perturb_df.to_csv(OUTPUT_DIR / "perturbation_analysis_codex.csv", index=False)

    plot_umap(
        embeddings.numpy(),
        graph_data.labels.numpy(),
        graph_data.proteins,
        VIS_DIR / "gnn_umap_embeddings_codex.png",
    )
    plot_attention_heatmap(
        attention_layer2,
        graph_data.edge_index,
        graph_data.proteins,
        VIS_DIR / "attention_heatmap_codex.png",
    )
    plot_network_snapshot(
        graph_data.graph,
        graph_data.metadata,
        master_regulators_df.index,
        VIS_DIR / "network_graph_codex.png",
    )

    community_stats = community_comparison(
        graph_data.graph,
        embeddings.numpy(),
        graph_data.proteins,
        graph_data.metadata,
        VIS_DIR / "community_comparison_codex.png",
    )

    export_summary(
        master_regulators_df,
        community_stats,
        metrics_log,
        graph_data.metadata,
        OUTPUT_DIR / "90_results_codex.md",
    )


if __name__ == "__main__":
    main()
