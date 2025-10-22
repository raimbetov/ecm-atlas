#!/usr/bin/env python3
"""Graph neural network on ECM proteomic correlation network."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas')
ITER_ROOT = ROOT / '13_1_meta_insights' / '02_multi_agent_multi_hipothesys' / 'iterations' / 'iteration_05' / 'hypothesis_19_metabolomics_phase1' / 'codex'
ANALYSES_DIR = ITER_ROOT / 'analyses_codex'

VELOCITY_PATH = ROOT / '13_1_meta_insights' / '02_multi_agent_multi_hipothesys' / 'iterations' / 'iteration_01' / 'hypothesis_03_tissue_aging_clocks' / 'codex' / 'tissue_aging_velocity_codex.csv'
PROTEOMICS_PATH = ROOT / '08_merged_ecm_dataset' / 'merged_ecm_aging_zscore.csv'

HIGH_VELOCITY_THRESHOLD = 2.17
EDGE_THRESHOLD = 0.6
HIDDEN_DIM = 16
EPOCHS = 400
LEARNING_RATE = 5e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    proteomics = pd.read_csv(PROTEOMICS_PATH)
    velocity_df = pd.read_csv(VELOCITY_PATH)
    high_velocity_tissues = velocity_df[velocity_df['Velocity'] >= HIGH_VELOCITY_THRESHOLD]['Tissue'].tolist()

    pivot = proteomics.pivot_table(index='Gene_Symbol', columns='Tissue', values='Zscore_Delta', aggfunc='mean').fillna(0.0)
    pivot = pivot.loc[:, pivot.columns.intersection(velocity_df['Tissue'])]

    high_mean = pivot[high_velocity_tissues].mean(axis=1)
    labels = (high_mean > 0).astype(int)
    return pivot, labels


def build_graph(features: pd.DataFrame, labels: pd.Series) -> Data:
    X = torch.tensor(features.values, dtype=torch.float32)
    y = torch.tensor(labels.values, dtype=torch.long)

    corr = np.corrcoef(features.values)
    edge_list = []
    num_nodes = features.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.isnan(corr[i, j]):
                continue
            if abs(corr[i, j]) >= EDGE_THRESHOLD:
                edge_list.append((i, j))
                edge_list.append((j, i))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Training / test masks (80/20 split)
    num_train = int(0.8 * num_nodes)
    indices = np.random.RandomState(42).permutation(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:num_train]] = True
    test_mask[indices[num_train:]] = True

    data = Data(x=X, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


class GCNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


def train_model(data: Data) -> Tuple[GCNClassifier, list[float]]:
    model = GCNClassifier(input_dim=data.num_features, hidden_dim=HIDDEN_DIM).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return model, history


def evaluate(model: GCNClassifier, data: Data) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        train_acc = (preds[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        test_acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return {'train_accuracy': train_acc, 'test_accuracy': test_acc}


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    features, labels = load_data()
    data = build_graph(features, labels)
    model, history = train_model(data)
    metrics = evaluate(model, data.to(DEVICE))

    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ITER_ROOT / 'models_codex' / 'proteomic_gcn_codex.pt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    history_path = ANALYSES_DIR / 'gcn_loss_history_codex.csv'
    pd.DataFrame({'epoch': np.arange(1, EPOCHS + 1), 'loss': history}).to_csv(history_path, index=False)

    metrics_path = ANALYSES_DIR / 'gcn_metrics_codex.json'
    metrics.update({'num_nodes': int(features.shape[0]), 'num_edges': int(data.edge_index.size(1) / 2)})
    metrics_path.write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
