"""Temporal ECM trajectory analysis for hypothesis H09 (agent: codex).

This script loads the merged ECM aging dataset, constructs a pseudo-temporal
ordering of tissues, trains both LSTM and Transformer sequence-to-sequence
models to forecast future protein states, ranks early vs late changing proteins,
performs regression and Granger causality analyses, and exports the required
artifacts (models, CSVs, and visualizations).
"""

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

DATA_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
AGENT = "codex"
OUTPUT_DIR = "."
VIS_DIR = os.path.join(OUTPUT_DIR, f"visualizations_{AGENT}")
SEED = 42
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SequenceConfig:
    input_window: int = 4
    output_window: int = 3
    train_end: int = 10
    val_end: int = 13
    max_lag: int = 3
    batch_size: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    transformer_d_model: int = 96
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.2
    num_epochs_lstm: int = 250
    num_epochs_transformer: int = 300
    patience: int = 30
    learning_rate: float = 1e-3


class SequenceDataset(torch.utils.data.Dataset):
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
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2, output_window: int = 3, dropout: float = 0.2):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length, :]


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn_output, attn_weights = self.self_attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x, attn_weights


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 96, nhead: int = 4, num_layers: int = 2, output_window: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        self.output_head = nn.Linear(d_model, input_dim)
        self.output_window = output_window

    def forward(self, src: torch.Tensor, tgt: torch.Tensor | None = None, teacher_forcing_ratio: float = 0.5, need_weights: bool = False):
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        attn_store = []
        for layer in self.layers:
            x, attn_w = layer(x, need_weights=need_weights)
            if need_weights:
                attn_store.append(attn_w)
        decoder_input = x[:, -1:, :]
        outputs = []
        steps = self.output_window if tgt is None else tgt.size(1)
        if tgt is not None:
            tgt_proj = self.pos_encoder(self.input_proj(tgt))
        hidden = torch.zeros(1, src.size(0), x.size(-1), device=src.device)
        cell = torch.zeros_like(hidden)
        for step in range(steps):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.output_head(out)
            outputs.append(pred)
            if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt_proj[:, step : step + 1, :]
            else:
                decoder_input = self.pos_encoder(self.input_proj(pred))
        preds = torch.cat(outputs, dim=1)
        if need_weights:
            return preds, attn_store
        return preds


def compute_pseudo_time(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    pivot = df.pivot_table(values="Zscore_Delta", index="Tissue", columns="Gene_Symbol")
    pivot_filled = pivot.fillna(0)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=SEED)
    components = pca.fit_transform(pivot_filled.values)
    order = np.argsort(components[:, 0])
    ordered_tissues = pivot_filled.index[order].tolist()
    pseudo_df = pd.DataFrame(
        {
            "Tissue": pivot_filled.index.tolist(),
            "pseudo_time_score": components[:, 0],
        }
    ).sort_values("pseudo_time_score")
    pseudo_df["pseudo_time_rank"] = stats.rankdata(pseudo_df["pseudo_time_score"], method="ordinal")
    pseudo_df["normalized_pseudo_time"] = (pseudo_df["pseudo_time_rank"] - 1) / (len(pseudo_df) - 1)
    return ordered_tissues, pseudo_df


def prepare_matrix(df: pd.DataFrame, ordered_tissues: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    pivot = df.pivot_table(values="Zscore_Delta", index="Gene_Symbol", columns="Tissue")
    pivot = pivot.reindex(columns=ordered_tissues)
    col_medians = pivot.median(axis=0, skipna=True)
    pivot = pivot.fillna(col_medians)
    pivot = pivot.fillna(0)
    return pivot, pivot.values


def build_windows(num_proteins: int, num_timepoints: int, config: SequenceConfig):
    windows: Dict[str, List[Tuple[int, int]]] = {"train": [], "val": [], "test": []}
    for protein_idx in range(num_proteins):
        for start in range(num_timepoints - config.input_window - config.output_window + 1):
            target_end = start + config.input_window + config.output_window - 1
            if target_end <= config.train_end:
                split = "train"
            elif target_end <= config.val_end:
                split = "val"
            else:
                split = "test"
            windows[split].append((protein_idx, start))
    return windows


def train_model(model: nn.Module, train_loader, val_loader, config: SequenceConfig):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    total_epochs = config.num_epochs_lstm if isinstance(model, LSTMSeq2Seq) else config.num_epochs_transformer
    for epoch in range(total_epochs):
        model.train()
        train_losses = []
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            teacher_ratio = max(0.1, 1 - epoch / (total_epochs / 1.5))
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


def evaluate_model(model: nn.Module, data_loader, output_window: int):
    model.eval()
    samples = []
    with torch.no_grad():
        for inputs, targets, proteins, starts in data_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(inputs, None, teacher_forcing_ratio=0.0)
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for i in range(preds_np.shape[0]):
                record = {"protein_idx": int(proteins[i]), "start": int(starts[i])}
                for h in range(output_window):
                    record[f"pred_h{h+1}"] = float(preds_np[i, h, 0])
                    record[f"true_h{h+1}"] = float(targets_np[i, h, 0])
                samples.append(record)
    if not samples:
        return pd.DataFrame(), pd.DataFrame()
    sample_df = pd.DataFrame(samples)
    metric_rows = []
    for h in range(output_window):
        y_true = sample_df[f"true_h{h+1}"]
        y_pred = sample_df[f"pred_h{h+1}"]
        metric_rows.append(
            {
                "horizon": h + 1,
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
            }
        )
    metrics_df = pd.DataFrame(metric_rows)
    return metrics_df, sample_df


def compute_top_predictable(sample_df: pd.DataFrame, proteins: List[str], fraction: float = 0.1):
    mse_cols = [col for col in sample_df.columns if col.startswith("pred_h")]
    mse_values = []
    for _, row in sample_df.iterrows():
        errors = []
        for idx, pred_col in enumerate(mse_cols, start=1):
            true_col = pred_col.replace("pred", "true")
            errors.append((row[pred_col] - row[true_col]) ** 2)
        mse_values.append(np.mean(errors))
    sample_df = sample_df.copy()
    sample_df["mean_mse"] = mse_values
    protein_mse = sample_df.groupby("protein_idx")["mean_mse"].mean().reset_index()
    protein_mse["Gene_Symbol"] = protein_mse["protein_idx"].apply(lambda idx: proteins[idx])
    protein_mse = protein_mse.sort_values("mean_mse")
    top_n = max(1, int(len(protein_mse) * fraction))
    return protein_mse.head(top_n)


def compute_gradients(matrix: np.ndarray):
    gradients = np.gradient(matrix, axis=1)
    abs_grad = np.abs(gradients)
    magnitude = abs_grad.mean(axis=1)
    time_idx = np.arange(matrix.shape[1])
    centers = (abs_grad @ time_idx) / (abs_grad.sum(axis=1) + 1e-8)
    normalized = centers / (matrix.shape[1] - 1)
    return magnitude, normalized


def classify_early_late(proteins: List[str], matrix: np.ndarray):
    mag, centers = compute_gradients(matrix)
    df = pd.DataFrame(
        {
            "Gene_Symbol": proteins,
            "gradient_magnitude": mag,
            "transition_index": centers,
        }
    )
    q1 = df["transition_index"].quantile(0.25)
    q3 = df["transition_index"].quantile(0.75)
    early = df[df["transition_index"] <= q1].copy()
    late = df[df["transition_index"] >= q3].copy()
    df["class"] = "mid"
    df.loc[early.index, "class"] = "early"
    df.loc[late.index, "class"] = "late"
    return df, early, late


def regression_early_late(pivot: pd.DataFrame, early: pd.DataFrame, late: pd.DataFrame, tissues: List[str], max_lag: int = 3):
    early_expr = pivot.loc[early["Gene_Symbol"]].reindex(columns=tissues).values
    late_expr = pivot.loc[late["Gene_Symbol"]].reindex(columns=tissues).values
    early_mean = early_expr.mean(axis=0)
    late_mean = late_expr.mean(axis=0)
    scaler = StandardScaler()
    early_scaled = scaler.fit_transform(early_mean.reshape(-1, 1)).flatten()
    rows = []
    for lag in range(1, max_lag + 1):
        X = early_scaled[:-lag].reshape(-1, 1)
        y = late_mean[lag:]
        split = int(len(X) * 0.6)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        _, _, r_value, p_value, _ = stats.linregress(X.flatten(), y)
        rows.append(
            {
                "lag": lag,
                "r2_train": r2_score(y_train, y_train_pred),
                "r2_test": r2_score(y_test, y_test_pred),
                "coef": model.coef_[0],
                "intercept": model.intercept_,
                "p_value": p_value,
                "pearson_r": r_value,
            }
        )
    return pd.DataFrame(rows)


def enrichment_tests(early: pd.DataFrame, late: pd.DataFrame, universe: pd.DataFrame):
    coagulation_genes = {"FGA", "FGB", "FGG", "F2", "F3", "F5", "F7", "F8", "F9", "F10", "F11", "F12", "VWF", "SERPINC1", "SERPINA1", "PROS1", "KLKB1", "PLG", "KNG1"}
    collagen_genes = {gene for gene in universe["Gene_Symbol"] if gene.startswith("COL")}

    def fisher_exact(target: set, subset: pd.DataFrame, label: str):
        subset_set = set(subset["Gene_Symbol"])
        universe_set = set(universe["Gene_Symbol"])
        overlap = len(target & subset_set)
        subset_size = len(subset_set)
        target_in_universe = len(target & universe_set)
        table = np.array(
            [
                [overlap, subset_size - overlap],
                [target_in_universe - overlap, len(universe_set) - subset_size - (target_in_universe - overlap)],
            ]
        )
        odds, p_value = stats.fisher_exact(table)
        return {
            "test": label,
            "odds_ratio": odds,
            "p_value": p_value,
            "overlap": overlap,
            "subset_size": subset_size,
            "target_in_universe": target_in_universe,
        }

    records = [
        fisher_exact(coagulation_genes, early, "Coagulation_vs_early"),
        fisher_exact(collagen_genes, late, "Collagen_vs_late"),
    ]
    return pd.DataFrame(records)


def run_granger(pivot: pd.DataFrame, early: pd.DataFrame, late: pd.DataFrame, max_lag: int = 3):
    early_list = early.sort_values("transition_index").head(10)["Gene_Symbol"].tolist()
    late_list = late.sort_values("transition_index", ascending=False).head(10)["Gene_Symbol"].tolist()
    records = []
    for e in early_list:
        for l in late_list:
            e_series = pivot.loc[e].values
            l_series = pivot.loc[l].values
            if np.std(e_series) < 1e-6 or np.std(l_series) < 1e-6:
                continue
            data = np.column_stack([l_series, e_series])
            try:
                results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            except Exception:
                continue
            for lag, output in results.items():
                stats_dict = output[0]
                f_stat, p_value, df_denom, df_num = stats_dict["ssr_ftest"]
                records.append(
                    {
                        "early_protein": e,
                        "late_protein": l,
                        "lag": lag,
                        "f_stat": float(f_stat),
                        "p_value": float(p_value),
                        "df_denom": float(df_denom),
                        "df_num": float(df_num),
                    }
                )
    return pd.DataFrame(records)


def aggregate_attention(model: TemporalTransformer, loader, windows: List[Tuple[int, int]], num_timepoints: int):
    model.eval()
    attention_sum = np.zeros(num_timepoints)
    counts = np.zeros(num_timepoints)
    idx = 0
    with torch.no_grad():
        for inputs, _, proteins, starts in loader:
            inputs = inputs.to(DEVICE)
            preds, attn_list = model(inputs, None, teacher_forcing_ratio=0.0, need_weights=True)
            attn_tensor = torch.stack(attn_list).mean(dim=0).mean(dim=1)  # (batch, seq, seq)
            for b in range(attn_tensor.size(0)):
                weights = attn_tensor[b].cpu().numpy()
                seq_len = weights.shape[0]
                start = int(starts[b])
                col_mean = weights.mean(axis=0)
                for t in range(seq_len):
                    global_idx = start + t
                    if global_idx < num_timepoints:
                        attention_sum[global_idx] += col_mean[t]
                        counts[global_idx] += 1
                idx += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        attn_mean = np.divide(attention_sum, counts, out=np.zeros_like(attention_sum), where=counts > 0)
    return attn_mean


def paired_gradient_test(matrix: np.ndarray, indices: List[int]):
    gradients = np.gradient(matrix, axis=1)
    before_vals = []
    after_vals = []
    for idx in indices:
        if idx <= 0 or idx >= matrix.shape[1] - 1:
            continue
        before_vals.append(gradients[:, idx - 1])
        after_vals.append(gradients[:, idx + 1])
    if not before_vals:
        return {"t_stat": float("nan"), "p_value": float("nan")}
    before = np.stack(before_vals, axis=1).mean(axis=1)
    after = np.stack(after_vals, axis=1).mean(axis=1)
    t_stat, p_value = stats.ttest_rel(before, after, nan_policy="omit")
    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def plot_prediction_performance(perf_df: pd.DataFrame, path: str):
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=perf_df, x="horizon", y="mse", hue="model", style="split", markers=True, dashes=False)
    plt.title("Forecast MSE by Horizon")
    plt.xlabel("Forecast horizon")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_trajectories(pivot: pd.DataFrame, early: pd.DataFrame, late: pd.DataFrame, tissues: List[str], path: str):
    plt.figure(figsize=(8, 4))
    time = np.arange(len(tissues))
    early_mean = pivot.loc[early["Gene_Symbol"]].reindex(columns=tissues).mean(axis=0)
    late_mean = pivot.loc[late["Gene_Symbol"]].reindex(columns=tissues).mean(axis=0)
    plt.plot(time, early_mean, label="Early", marker="o")
    plt.plot(time, late_mean, label="Late", marker="s")
    plt.xlabel("Pseudo-time index")
    plt.ylabel("ΔZ-score")
    plt.title("Average protein trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_attention(attn_series: pd.Series, path: str):
    plt.figure(figsize=(10, 2.5))
    sns.heatmap(attn_series.values[np.newaxis, :], cmap="viridis", cbar=True)
    plt.xticks(np.arange(len(attn_series)) + 0.5, attn_series.index, rotation=90)
    plt.yticks([])
    plt.title("Transformer attention across pseudo-time")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_causal_network(edges: pd.DataFrame, path: str):
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["early_protein"], row["late_protein"], weight=row["f_stat"])
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=SEED)
    weights = [max(0.5, min(4.0, w)) for w in nx.get_edge_attributes(G, "weight").values()]
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=550)
    nx.draw_networkx_edges(G, pos, edge_color="grey", arrows=True, width=weights)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Early → Late protein causal network")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run_pipeline(args):
    set_seed(SEED)
    os.makedirs(VIS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    ordered_tissues, pseudo_df = compute_pseudo_time(df)
    pivot, matrix = prepare_matrix(df, ordered_tissues)
    config = SequenceConfig()
    windows = build_windows(matrix.shape[0], matrix.shape[1], config)
    datasets = {
        split: SequenceDataset(matrix, windows[split], config.input_window, config.output_window)
        for split in windows
    }
    loaders = {
        split: torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=(split == "train"), drop_last=False, collate_fn=collate_batch)
        for split, dataset in datasets.items()
    }

    lstm = LSTMSeq2Seq(hidden_dim=config.lstm_hidden, num_layers=config.lstm_layers, output_window=config.output_window, dropout=config.dropout).to(DEVICE)
    lstm_history = train_model(lstm, loaders["train"], loaders["val"], config)

    transformer = TemporalTransformer(d_model=config.transformer_d_model, nhead=config.transformer_heads, num_layers=config.transformer_layers, output_window=config.output_window, dropout=config.dropout).to(DEVICE)
    transformer_history = train_model(transformer, loaders["train"], loaders["val"], config)

    perf_rows = []
    sample_store: Dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        lstm_metrics, lstm_samples = evaluate_model(lstm, loaders[split], config.output_window)
        transformer_metrics, transformer_samples = evaluate_model(transformer, loaders[split], config.output_window)
        lstm_metrics["model"] = "LSTM"
        lstm_metrics["split"] = split
        transformer_metrics["model"] = "Transformer"
        transformer_metrics["split"] = split
        perf_rows.extend([lstm_metrics, transformer_metrics])
        if split == "test":
            sample_store["LSTM"] = lstm_samples
            sample_store["Transformer"] = transformer_samples
    perf_df = pd.concat(perf_rows, ignore_index=True)
    perf_df.to_csv(f"prediction_performance_{AGENT}.csv", index=False)
    plot_prediction_performance(perf_df, os.path.join(VIS_DIR, f"prediction_performance_{AGENT}.png"))

    top_predictable = compute_top_predictable(sample_store["LSTM"], pivot.index.tolist())
    top_predictable.to_csv(f"top_predictable_proteins_{AGENT}.csv", index=False)

    classification_df, early_df, late_df = classify_early_late(pivot.index.tolist(), matrix)
    early_df.to_csv(f"early_change_proteins_{AGENT}.csv", index=False)
    late_df.to_csv(f"late_change_proteins_{AGENT}.csv", index=False)

    regression_df = regression_early_late(pivot, early_df, late_df, ordered_tissues, max_lag=config.max_lag)
    regression_df.to_csv(f"early_late_regression_{AGENT}.csv", index=False)

    enrichment_df = enrichment_tests(early_df, late_df, classification_df)
    enrichment_df.to_csv(f"enrichment_analysis_{AGENT}.csv", index=False)

    granger_df = run_granger(pivot.reindex(columns=ordered_tissues), early_df, late_df, max_lag=config.max_lag)
    granger_df.to_csv(f"granger_causality_{AGENT}.csv", index=False)
    significant_edges = granger_df[granger_df["p_value"] < 0.05]
    if not significant_edges.empty:
        plot_causal_network(significant_edges.drop_duplicates(subset=["early_protein", "late_protein"]), os.path.join(VIS_DIR, f"causal_network_{AGENT}.png"))

    attn_mean = aggregate_attention(transformer, loaders["test"], windows["test"], matrix.shape[1])
    attn_series = pd.Series(attn_mean, index=ordered_tissues)
    attn_df = attn_series.reset_index()
    attn_df.columns = ["Tissue", "attention_weight"]
    attn_df.to_csv(f"attention_weights_per_timestep_{AGENT}.csv", index=False)

    cutoff = np.percentile(attn_series.values, 90)
    critical = attn_series[attn_series >= cutoff]
    critical_df = critical.reset_index()
    critical_df.columns = ["Tissue", "attention_weight"]
    critical_df.to_csv(f"critical_transitions_{AGENT}.csv", index=False)

    grad_stats = paired_gradient_test(matrix, [ordered_tissues.index(t) for t in critical.index])
    with open(f"critical_transition_stats_{AGENT}.json", "w", encoding="utf-8") as f:
        json.dump(grad_stats, f, indent=2)

    plot_attention(attn_series, os.path.join(VIS_DIR, f"attention_heatmap_{AGENT}.png"))
    plot_trajectories(pivot, early_df, late_df, ordered_tissues, os.path.join(VIS_DIR, f"trajectory_plot_{AGENT}.png"))

    torch.save(lstm.state_dict(), f"lstm_seq2seq_model_{AGENT}.pth")
    torch.save(transformer.state_dict(), f"transformer_model_{AGENT}.pth")

    summary = {
        "pseudo_time_order": ordered_tissues,
        "pseudo_time_scores": pseudo_df.set_index("Tissue")["pseudo_time_score"].to_dict(),
        "lstm_history": lstm_history,
        "transformer_history": transformer_history,
        "regression": regression_df.to_dict(orient="records"),
        "attention_threshold": float(cutoff),
        "gradient_test": grad_stats,
    }
    with open(f"analysis_metadata_{AGENT}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal ECM trajectory analysis")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to merged ECM dataset")
    args = parser.parse_args()
    if args.data != DATA_PATH:
        globals()['DATA_PATH'] = args.data
    run_pipeline(args)
