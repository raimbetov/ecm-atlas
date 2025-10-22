"""LSTM benchmarking across pseudo-time methods for H11 (agent: codex)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
)
PSEUDOTIME_PATH = BASE_DIR / "pseudotime_orderings_codex.csv"
OUTPUT_METRICS = BASE_DIR / "lstm_performance_codex.csv"
OUTPUT_SAMPLES = BASE_DIR / "intermediate/lstm_samples_codex.json"
SEED = 42
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SequenceConfig:
    input_window: int = 4
    output_window: int = 2
    batch_size: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.2
    num_epochs: int = 200
    patience: int = 30
    learning_rate: float = 1e-3


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, windows: List[Tuple[int, int]], input_window: int, output_window: int):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.windows = windows
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        protein_idx, start = self.windows[idx]
        seq = self.data[protein_idx]
        x = seq[start : start + self.input_window]
        y = seq[start + self.input_window : start + self.input_window + self.output_window]
        return x.unsqueeze(-1), y.unsqueeze(-1), protein_idx, start


def collate_batch(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    proteins = torch.tensor([item[2] for item in batch], dtype=torch.long)
    starts = torch.tensor([item[3] for item in batch], dtype=torch.long)
    return inputs, targets, proteins, starts


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2, output_window: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_window = output_window
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(src)
        decoder_input = src[:, -1:, :]
        outputs = []
        steps = self.output_window if tgt is None else tgt.size(1)
        for step in range(steps):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            prediction = self.fc(out)
            outputs.append(prediction)
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, step : step + 1, :]
            else:
                decoder_input = prediction
        return torch.cat(outputs, dim=1)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = {"Tissue", "Gene_Symbol", "Zscore_Delta"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def load_pseudotime_orders() -> Dict[str, List[str]]:
    df = pd.read_csv(PSEUDOTIME_PATH)
    orders: Dict[str, List[str]] = {}
    for method, subset in df.groupby("method"):
        subset_sorted = subset.sort_values("rank")
        orders[method] = subset_sorted["Tissue"].tolist()
    return orders


def prepare_matrix(df: pd.DataFrame, ordered_tissues: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    pivot = df.pivot_table(values="Zscore_Delta", index="Gene_Symbol", columns="Tissue")
    pivot = pivot.reindex(columns=ordered_tissues)
    pivot = pivot.fillna(pivot.median(axis=1))
    pivot = pivot.fillna(0.0)
    return pivot, pivot.values


def build_windows(num_proteins: int, num_timepoints: int, config: SequenceConfig):
    windows: Dict[str, List[Tuple[int, int]]] = {"train": [], "val": [], "test": []}
    train_end = int(num_timepoints * 0.6)
    val_end = int(num_timepoints * 0.8)
    for protein_idx in range(num_proteins):
        for start in range(num_timepoints - (config.input_window + config.output_window) + 1):
            target_end = start + config.input_window + config.output_window
            if target_end <= train_end:
                split = "train"
            elif target_end <= val_end:
                split = "val"
            else:
                split = "test"
            windows[split].append((protein_idx, start))
    return windows


def make_loader(matrix: np.ndarray, windows: List[Tuple[int, int]], config: SequenceConfig, shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(matrix, windows, config.input_window, config.output_window)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_batch)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, config: SequenceConfig) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(config.num_epochs):
        model.train()
        train_losses = []
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            teacher_ratio = max(0.1, 1 - epoch / (config.num_epochs / 1.5))
            preds = model(inputs, targets, teacher_forcing_ratio=teacher_ratio)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())
        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets, _, _ in val_loader:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(inputs, targets, teacher_forcing_ratio=0.0)
                val_losses.append(criterion(preds, targets).item())
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_mean)
        history["val_loss"].append(val_mean)
        if val_mean < best_val - 1e-4:
            best_val = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def evaluate_model(model: nn.Module, data_loader: DataLoader, config: SequenceConfig, method: str, proteins: List[str]) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model.eval()
    records = []
    samples: List[Dict[str, float]] = []
    with torch.no_grad():
        for inputs, targets, protein_idx, starts in data_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(inputs, None, teacher_forcing_ratio=0.0)
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for i in range(preds_np.shape[0]):
                record = {
                    "method": method,
                    "protein": proteins[int(protein_idx[i])],
                    "start_index": int(starts[i]),
                }
                errors = []
                for h in range(config.output_window):
                    pred_val = float(preds_np[i, h, 0])
                    true_val = float(targets_np[i, h, 0])
                    record[f"pred_h{h+1}"] = pred_val
                    record[f"true_h{h+1}"] = true_val
                    errors.append((pred_val - true_val) ** 2)
                record["mse_window"] = float(np.mean(errors))
                samples.append(record)
    if not samples:
        return pd.DataFrame(), []
    metrics = []
    for h in range(config.output_window):
        y_true = [row[f"true_h{h+1}"] for row in samples]
        y_pred = [row[f"pred_h{h+1}"] for row in samples]
        metrics.append(
            {
                "method": method,
                "horizon": h + 1,
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(metrics), samples


def run_benchmark():
    set_seed()
    df = load_dataset()
    orders = load_pseudotime_orders()
    config = SequenceConfig()

    all_metrics = []
    all_samples: List[Dict[str, float]] = []

    for method, ordered_tissues in orders.items():
        matrix_df, matrix_np = prepare_matrix(df, ordered_tissues)
        num_timepoints = matrix_np.shape[1]
        windows = build_windows(matrix_np.shape[0], num_timepoints, config)
        train_loader = make_loader(matrix_np, windows["train"], config, shuffle=True)
        val_loader = make_loader(matrix_np, windows["val"], config, shuffle=False)
        test_loader = make_loader(matrix_np, windows["test"], config, shuffle=False)

        model = LSTMSeq2Seq(hidden_dim=config.lstm_hidden, num_layers=config.lstm_layers, output_window=config.output_window, dropout=config.dropout)
        model.to(DEVICE)
        train_model(model, train_loader, val_loader, config)
        metrics_df, samples = evaluate_model(model, test_loader, config, method, matrix_df.index.tolist())
        if not metrics_df.empty:
            all_metrics.append(metrics_df)
        all_samples.extend(samples)

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(OUTPUT_METRICS, index=False)
    with open(OUTPUT_SAMPLES, "w", encoding="utf-8") as fp:
        json.dump(all_samples, fp, indent=2)


if __name__ == "__main__":
    run_benchmark()
