import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import shap
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).parent.resolve()
DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
)
VIS_DIR = BASE_DIR / "visualizations_codex"
DATA_CACHE_DIR = BASE_DIR / "data_cache"
MODEL_PATH = BASE_DIR / "coagulation_nn_model_codex.pth"
CLASSIFIER_MODEL_PATH = BASE_DIR / "coagulation_nn_classifier_codex.pth"
MODEL_PERF_PATH = BASE_DIR / "model_performance_codex.csv"
PREDICTIONS_PATH = BASE_DIR / "aging_velocity_predictions_codex.csv"
CLASSIFICATION_PRED_PATH = BASE_DIR / "aging_velocity_classification_codex.csv"
SHAP_IMPORTANCE_PATH = BASE_DIR / "shap_importance_codex.csv"
COAG_STATES_PATH = BASE_DIR / "coagulation_states_codex.csv"
EARLY_CHANGE_PATH = BASE_DIR / "early_change_proteins_codex.csv"
NETWORK_MODULES_PATH = BASE_DIR / "network_modules_codex.csv"
CENTRALITY_COMPARISON_PATH = BASE_DIR / "centrality_comparison_codex.csv"
LSTM_MODEL_PATH = BASE_DIR / "lstm_model_codex.pth"

SEQUENCE_CACHE = DATA_CACHE_DIR / "protein_sequences_codex.json"
EMBEDDING_CACHE = DATA_CACHE_DIR / "protein_embeddings_codex.npz"

COAG_PROTEINS: List[str] = [
    "F2",
    "F3",
    "F5",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13A1",
    "F13B",
    "FGA",
    "FGB",
    "FGG",
    "GAS6",
    "PROC",
    "PROS1",
    "THBD",
    "SERPINC1",
    "SERPINE1",
    "SERPINF2",
    "PLAU",
    "PLAUR",
    "VWF",
    "TFPI",
]

TRANSFER_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def ensure_directories() -> None:
    """Ensure output directories exist."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_uniprot_sequence(gene: str) -> str:
    """Fetch a human protein sequence for a gene symbol from UniProt."""
    params = {
        "query": f"gene_exact:{gene}+AND+organism_id:9606",
        "format": "fasta",
        "size": "1",
    }
    url = "https://rest.uniprot.org/uniprotkb/search"
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200 or not resp.text.strip():
        # Fallback to generic poly-alanine sequence to preserve model flow
        return "M" + "A" * 100
    fasta = resp.text.strip().splitlines()
    sequence = "".join(line.strip() for line in fasta if not line.startswith(">"))
    if not sequence:
        sequence = "M" + "A" * 100
    return sequence


def load_sequence_cache() -> Dict[str, str]:
    if SEQUENCE_CACHE.exists():
        with open(SEQUENCE_CACHE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_sequence_cache(cache: Dict[str, str]) -> None:
    with open(SEQUENCE_CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


def get_protein_sequences(genes: List[str]) -> Dict[str, str]:
    cache = load_sequence_cache()
    updated = False
    for gene in genes:
        if gene not in cache:
            cache[gene] = fetch_uniprot_sequence(gene)
            updated = True
    if updated:
        save_sequence_cache(cache)
    return {gene: cache[gene] for gene in genes}


def load_protein_embeddings(genes: List[str]) -> np.ndarray:
    """Load or compute protein embeddings using a pre-trained transformer."""
    if EMBEDDING_CACHE.exists():
        cached = np.load(EMBEDDING_CACHE, allow_pickle=True)
        cached_genes = cached["genes"].tolist()
        if cached_genes == genes:
            return cached["embeddings"]
    sequences = get_protein_sequences(genes)
    tokenizer = AutoTokenizer.from_pretrained(TRANSFER_MODEL_NAME)
    model = AutoModel.from_pretrained(TRANSFER_MODEL_NAME)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for gene in genes:
            seq = sequences[gene]
            tokens = tokenizer(
                seq,
                return_tensors="pt",
                truncation=True,
                max_length=1022,
            )
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state.squeeze(0)
            emb = hidden.mean(dim=0).cpu().numpy()
            embeddings.append(emb)
    embeddings_array = np.vstack(embeddings)
    np.savez(EMBEDDING_CACHE, genes=genes, embeddings=embeddings_array)
    return embeddings_array


class TransferEnhancedCoagNN(nn.Module):
    def __init__(self, input_dim: int, embedding_matrix: torch.Tensor):
        super().__init__()
        self.embedding_matrix = nn.Parameter(embedding_matrix.clone())
        combined_dim = input_dim + embedding_matrix.shape[1]
        self.network = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = torch.matmul(x, self.embedding_matrix)
        combined = torch.cat([x, context], dim=1)
        return self.network(combined)


def build_panels(df: pd.DataFrame) -> Dict[str, List[str]]:
    genes = df["Gene_Symbol"].dropna().unique()
    serpins = sorted([g for g in genes if g.startswith("SERPIN")])
    collagens = sorted([g for g in genes if g.startswith("COL")])
    mmps = sorted([g for g in genes if g.startswith("MMP")])
    return {
        "coag": [g for g in COAG_PROTEINS if g in genes],
        "serpin": serpins,
        "collagen": collagens,
        "mmp": mmps,
    }


def compute_aging_velocity(df: pd.DataFrame) -> pd.Series:
    pivot = df.pivot_table(
        values="Zscore_Delta",
        index="Tissue",
        columns="Gene_Symbol",
    )
    velocity = pivot.abs().mean(axis=1)
    return velocity


def prepare_feature_matrix(df: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    pivot = df[df["Gene_Symbol"].isin(genes)].pivot_table(
        values="Zscore_Delta", index="Tissue", columns="Gene_Symbol"
    )
    pivot = pivot.reindex(sorted(pivot.index))
    pivot = pivot.reindex(columns=genes)
    pivot = pivot.fillna(0.0)
    return pivot


def train_deep_nn(
    features: pd.DataFrame,
    target: pd.Series,
    embedding_matrix: np.ndarray,
    n_splits: int = 5,
    epochs: int = 500,
    lr: float = 1e-3,
) -> Tuple[pd.DataFrame, pd.DataFrame, TransferEnhancedCoagNN]:
    tissues = features.index.tolist()
    X = features.values.astype(np.float32)
    y = target.loc[tissues].values.astype(np.float32)

    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    metrics_records = []
    predictions_records = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    best_model = None
    best_r2 = -np.inf
    best_scaler_state = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])

        model = TransferEnhancedCoagNN(X.shape[1], embedding_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).squeeze(1).numpy()
        true_vals = y[test_idx]
        fold_r2 = r2_score(true_vals, preds)
        mae = mean_absolute_error(true_vals, preds)
        rmse = math.sqrt(mean_squared_error(true_vals, preds))
        if fold_r2 > best_r2:
            best_r2 = fold_r2
            best_model = model
            best_scaler_state = {
                "mean": scaler.mean_.copy(),
                "scale": scaler.scale_.copy(),
            }
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                    "embedding_matrix": embedding_matrix,
                    "coag_genes": features.columns.tolist(),
                },
                MODEL_PATH,
            )

        metrics_records.append(
            {
                "task": "regression",
                "fold": fold + 1,
                "r2": fold_r2,
                "mae": mae,
                "rmse": rmse,
            }
        )
        for idx, pred, true in zip(test_idx, preds, true_vals):
            predictions_records.append(
                {
                    "task": "regression",
                    "fold": fold + 1,
                    "tissue": tissues[idx],
                    "predicted_velocity": pred,
                    "true_velocity": true,
                }
            )

    metrics_df = pd.DataFrame(metrics_records)
    predictions_df = pd.DataFrame(predictions_records)
    return metrics_df, predictions_df, best_model, best_scaler_state


def prepare_fast_slow_dataset(
    features: pd.DataFrame,
    aging_velocity: pd.Series,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> Tuple[pd.DataFrame, pd.Series]:
    aligned_velocity = aging_velocity.loc[features.index]
    q_low = aligned_velocity.quantile(low_quantile)
    q_high = aligned_velocity.quantile(high_quantile)
    slow_idx = aligned_velocity[aligned_velocity <= q_low].index
    fast_idx = aligned_velocity[aligned_velocity >= q_high].index
    if len(slow_idx) < 5 or len(fast_idx) < 5:
        raise ValueError("Not enough tissues per class for stratified 5-fold CV.")
    selected_idx = slow_idx.union(fast_idx)
    selected_features = features.loc[selected_idx]
    labels = pd.Series(0, index=selected_idx, dtype=int)
    labels.loc[fast_idx] = 1
    return selected_features, labels


def train_deep_nn_classifier(
    features: pd.DataFrame,
    labels: pd.Series,
    embedding_matrix: np.ndarray,
    n_splits: int = 5,
    epochs: int = 400,
    lr: float = 5e-4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tissues = features.index.tolist()
    X = features.values.astype(np.float32)
    y = labels.loc[tissues].values.astype(np.float32)

    metrics_records = []
    predictions_records = []
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    best_auc = -np.inf
    best_model_state = None
    best_scaler_state = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])

        model = TransferEnhancedCoagNN(X.shape[1], embedding_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test = torch.tensor(y[test_idx], dtype=torch.float32).unsqueeze(1)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test).squeeze(1)
            probs = torch.sigmoid(logits).numpy()
        preds_binary = (probs >= 0.5).astype(int)
        true_vals = y[test_idx]

        auc = roc_auc_score(true_vals, probs)
        acc = accuracy_score(true_vals, preds_binary)
        precision = precision_score(true_vals, preds_binary, zero_division=0)
        recall = recall_score(true_vals, preds_binary, zero_division=0)
        f1 = f1_score(true_vals, preds_binary, zero_division=0)

        if auc > best_auc:
            best_auc = auc
            best_model_state = {
                "state_dict": model.state_dict(),
                "scaler_mean": scaler.mean_.copy(),
                "scaler_scale": scaler.scale_.copy(),
                "embedding_matrix": embedding_matrix,
                "coag_genes": features.columns.tolist(),
            }
            best_scaler_state = {
                "mean": scaler.mean_.copy(),
                "scale": scaler.scale_.copy(),
            }

        metrics_records.append(
            {
                "task": "classification",
                "fold": fold + 1,
                "auc": auc,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        for idx, prob, pred, true in zip(test_idx, probs, preds_binary, true_vals):
            predictions_records.append(
                {
                    "task": "classification",
                    "fold": fold + 1,
                    "tissue": tissues[idx],
                    "prob_fast": prob,
                    "predicted_label": int(pred),
                    "true_label": int(true),
                }
            )

    if best_model_state is not None:
        torch.save(best_model_state, CLASSIFIER_MODEL_PATH)

    metrics_df = pd.DataFrame(metrics_records)
    predictions_df = pd.DataFrame(predictions_records)
    predictions_df.to_csv(CLASSIFICATION_PRED_PATH, index=False)
    return metrics_df, predictions_df
def compute_shap_summary(
    model: TransferEnhancedCoagNN,
    features: pd.DataFrame,
    scaler_state: Dict[str, np.ndarray],
    background_size: int = 30,
) -> None:
    model.eval()
    X = features.values.astype(np.float32)
    mean = scaler_state["mean"]
    scale = scaler_state["scale"]
    scale_safe = np.where(scale == 0, 1.0, scale)
    X_scaled = (X - mean) / scale_safe

    background_size = min(background_size, X_scaled.shape[0])
    background = torch.tensor(X_scaled[:background_size], dtype=torch.float32)
    target_data = torch.tensor(X_scaled, dtype=torch.float32)

    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(target_data)
    except Exception:
        explainer = shap.KernelExplainer(
            model=lambda data: model(torch.tensor(data, dtype=torch.float32)).detach().numpy(),
            data=background.numpy(),
        )
        shap_values = explainer.shap_values(target_data.numpy())

    if isinstance(shap_values, list):
        shap_array = shap_values[0]
    else:
        shap_array = shap_values

    shap_matrix = np.array(shap_array)
    if shap_matrix.ndim == 1:
        shap_matrix = shap_matrix.reshape(-1, 1)
    elif shap_matrix.ndim == 3:
        shap_matrix = shap_matrix.reshape(shap_matrix.shape[0], shap_matrix.shape[1])
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": features.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    shap.summary_plot(
        shap_array,
        X_scaled,
        feature_names=features.columns,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(VIS_DIR / "feature_importance_shap_codex.png", dpi=300)
    plt.close()


class LSTMTemporalModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        final_hidden = output[:, -1, :]
        return self.regressor(final_hidden)


def build_temporal_sequences(
    coag_matrix: pd.DataFrame,
    ecm_matrix: pd.DataFrame,
    window: int = 5,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    tissues = coag_matrix.index.tolist()
    X_sequences = []
    y_targets = []
    target_tissues = []
    coag_values = coag_matrix.values.astype(np.float32)
    ecm_values = ecm_matrix.values.astype(np.float32)
    target_vector = ecm_values.mean(axis=1)
    for start in range(0, len(tissues) - window - horizon + 1):
        end = start + window
        target_idx = end + horizon - 1
        seq = coag_values[start:end]
        target_value = target_vector[target_idx]
        X_sequences.append(seq)
        y_targets.append(target_value)
        target_tissues.append(tissues[target_idx])
    return np.stack(X_sequences), np.array(y_targets), target_tissues


def train_lstm_temporal_model(
    sequences: np.ndarray,
    targets: np.ndarray,
    epochs: int = 400,
    lr: float = 1e-3,
) -> Tuple[pd.DataFrame, LSTMTemporalModel]:
    input_dim = sequences.shape[2]
    model = LSTMTemporalModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(sequences, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(sequences, dtype=torch.float32)).squeeze(1).numpy()
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = math.sqrt(mean_squared_error(targets, predictions))
    metrics = pd.DataFrame(
        {
            "metric": ["r2", "mae", "rmse"],
            "value": [r2, mae, rmse],
        }
    )
    torch.save(model.state_dict(), LSTM_MODEL_PATH)
    return metrics, model


def identify_early_change_proteins(
    aggregated_matrix: pd.DataFrame,
    module_map: Dict[str, str],
) -> pd.DataFrame:
    timelines = []
    num_steps = aggregated_matrix.shape[0]
    threshold_index = math.ceil(0.25 * num_steps)
    time_axis = np.arange(num_steps)

    for gene in aggregated_matrix.columns:
        series = aggregated_matrix[gene].values
        diffs = np.abs(np.diff(series, prepend=series[0]))
        cumulative = np.cumsum(diffs)
        total = cumulative[-1] if cumulative[-1] != 0 else 1e-6
        normalized = cumulative / total
        lead_index = int(np.argmax(normalized >= 0.25))
        timelines.append(
            {
                "protein": gene,
                "module": module_map.get(gene, "other"),
                "lead_index": lead_index,
                "early_change": lead_index <= threshold_index,
            }
        )
    df = pd.DataFrame(timelines).sort_values("lead_index")
    df.to_csv(EARLY_CHANGE_PATH, index=False)
    return df


def classify_coagulation_states(
    coag_features: pd.DataFrame,
    aging_velocity: pd.Series,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    data = coag_features.copy()
    required = [
        "F2",
        "SERPINC1",
        "FGA",
        "FGB",
        "FGG",
        "PLAU",
        "PLAUR",
        "SERPINE1",
        "F13B",
        "PROS1",
        "THBD",
    ]
    for gene in required:
        if gene not in data.columns:
            data[gene] = 0.0
    coag_score = (
        data["F2"]
        + data["FGA"]
        + data["FGB"]
        + data["FGG"]
        - data["SERPINC1"]
        - data["PROS1"]
        - data["THBD"]
    )
    fibrinolysis_score = (
        data["PLAU"]
        + data["PLAUR"]
        - data["SERPINE1"]
        - data["F13B"]
    )
    hypercoag_threshold = coag_score.quantile(0.7)
    hypocoag_threshold = coag_score.quantile(0.3)
    hyperfibr_threshold = fibrinolysis_score.quantile(0.7)

    def determine_state(row):
        s = row["coagulation_score"]
        f = row["fibrinolysis_score"]
        if s >= hypercoag_threshold:
            return "hypercoagulable"
        if s <= hypocoag_threshold and f >= hyperfibr_threshold:
            return "hyperfibrinolytic"
        return "balanced"

    state_df = pd.DataFrame(
        {
            "coagulation_score": coag_score,
            "fibrinolysis_score": fibrinolysis_score,
        }
    )
    states = state_df.apply(determine_state, axis=1)
    merged = pd.DataFrame(
        {
            "tissue": data.index,
            "coagulation_score": coag_score.values,
            "fibrinolysis_score": fibrinolysis_score.values,
            "aging_velocity": aging_velocity.loc[data.index].values,
            "state": states.values,
        }
    )
    if not predictions.empty:
        merged = merged.merge(
            predictions[["tissue", "predicted_velocity"]], on="tissue", how="left"
        )
    rho, pval = spearmanr(merged["coagulation_score"], merged["aging_velocity"])
    merged["spearman_rho"] = rho
    merged["spearman_pvalue"] = pval
    merged.to_csv(COAG_STATES_PATH, index=False)
    return merged


def plot_model_performance(predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=predictions,
        x="true_velocity",
        y="predicted_velocity",
        hue="fold",
        palette="viridis",
        s=60,
    )
    lims = [
        min(predictions["true_velocity"].min(), predictions["predicted_velocity"].min()) - 0.1,
        max(predictions["true_velocity"].max(), predictions["predicted_velocity"].max()) + 0.1,
    ]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("True aging velocity")
    plt.ylabel("Predicted aging velocity")
    plt.title("Coagulation NN performance")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "model_performance_codex.png", dpi=300)
    plt.close()


def plot_coagulation_state_scatter(states: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=states,
        x="coagulation_score",
        y="aging_velocity",
        hue="state",
        style="state",
        s=80,
        palette={
            "hypercoagulable": "#d62728",
            "hyperfibrinolytic": "#1f77b4",
            "balanced": "#2ca02c",
        },
    )
    plt.xlabel("Coagulation composite z-score")
    plt.ylabel("Aging velocity")
    plt.title("Coagulation state vs aging velocity")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "coagulation_state_scatter_codex.png", dpi=300)
    plt.close()


def plot_temporal_trajectory(
    aggregated: pd.DataFrame,
    module_assignments: Dict[str, str],
) -> None:
    plt.figure(figsize=(8, 5))
    time_axis = np.arange(aggregated.shape[0])
    module_groups = {}
    for gene, module in module_assignments.items():
        if gene not in aggregated.columns:
            continue
        module_groups.setdefault(module, []).append(gene)
    for module, genes in module_groups.items():
        series = aggregated[genes].mean(axis=1)
        sns.lineplot(x=time_axis, y=series, label=module)
    plt.xlabel("Pseudo-time (slow → fast aging tissues)")
    plt.ylabel("Mean z-score delta")
    plt.title("Temporal trajectories by protein module")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "temporal_trajectory_plot_codex.png", dpi=300)
    plt.close()


def run_network_analysis(
    df: pd.DataFrame,
    modules: Dict[str, List[str]],
    threshold: float = 0.55,
) -> None:
    genes = set(modules["coag"]) | set(modules["serpin"]) | set(modules["collagen"])
    sub_df = df[df["Gene_Symbol"].isin(genes)]
    pivot = sub_df.pivot_table(
        values="Zscore_Delta", index="Tissue", columns="Gene_Symbol"
    )
    pivot = pivot.fillna(0)
    corr = pivot.corr()
    G = nx.Graph()
    for gene in corr.columns:
        module = (
            "coag"
            if gene in modules["coag"]
            else "serpin"
            if gene in modules["serpin"]
            else "collagen"
        )
        G.add_node(gene, module=module)
    for i, gene_i in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            gene_j = corr.columns[j]
            value = corr.iat[i, j]
            if abs(value) >= threshold:
                G.add_edge(gene_i, gene_j, weight=value)
    betweenness = nx.betweenness_centrality(G, weight="weight")
    module_map = {}
    for gene in G.nodes:
        module_map[gene] = G.nodes[gene]["module"]
    centrality_df = pd.DataFrame(
        {
            "protein": list(betweenness.keys()),
            "betweenness": list(betweenness.values()),
            "module": [module_map[g] for g in betweenness.keys()],
        }
    )
    centrality_df.to_csv(NETWORK_MODULES_PATH, index=False)

    summary = (
        centrality_df.groupby("module")["betweenness"].agg(["mean", "median", "max"]).reset_index()
    )
    summary.to_csv(CENTRALITY_COMPARISON_PATH, index=False)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=RANDOM_STATE)
    colors = {"coag": "#d62728", "serpin": "#9467bd", "collagen": "#1f77b4"}
    node_colors = [colors.get(module_map[n], "gray") for n in G.nodes]
    sizes = [300 + 4000 * betweenness[n] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, alpha=0.85)
    edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=[2 * w for w in edge_weights], alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=7)
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=mod, markerfacecolor=col, markersize=10)
               for mod, col in colors.items()]
    plt.legend(handles=handles, title="Module", loc="best")
    plt.title("Coagulation vs Serpin vs Collagen network centrality")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "network_visualization_codex.png", dpi=300)
    plt.close()


def compute_full_model_baseline(df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
    pivot = df.pivot_table(
        values="Zscore_Delta", index="Tissue", columns="Gene_Symbol"
    ).fillna(0)
    common_tissues = target.index.intersection(pivot.index)
    X = pivot.loc[common_tissues].values
    y = target.loc[common_tissues].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preds = np.zeros_like(y)
    for train_idx, test_idx in kf.split(X):
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            max_depth=30,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = math.sqrt(mean_squared_error(y, preds))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def main() -> None:
    ensure_directories()
    df = pd.read_csv(DATA_PATH)
    panels = build_panels(df)
    aging_velocity = compute_aging_velocity(df)

    coag_features = prepare_feature_matrix(df, panels["coag"])
    embedding_matrix = load_protein_embeddings(panels["coag"])

    metrics_df, predictions_df, best_model, best_scaler_state = train_deep_nn(
        coag_features,
        aging_velocity,
        embedding_matrix,
    )
    if best_scaler_state is not None and best_model is not None:
        compute_shap_summary(best_model, coag_features, best_scaler_state)
    plot_model_performance(predictions_df)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    combined_metrics = metrics_df.copy()
    try:
        cls_features, cls_labels = prepare_fast_slow_dataset(coag_features, aging_velocity)
        cls_metrics_df, _ = train_deep_nn_classifier(
            cls_features,
            cls_labels,
            embedding_matrix,
        )
        combined_metrics = pd.concat([combined_metrics, cls_metrics_df], ignore_index=True)
    except ValueError as exc:
        print(f"[WARN] Classification dataset insufficient: {exc}")

    combined_metrics.to_csv(MODEL_PERF_PATH, index=False)

    full_baseline = compute_full_model_baseline(df, aging_velocity)
    baseline_df = pd.DataFrame([full_baseline])
    baseline_df.to_csv(BASE_DIR / "full_model_baseline_codex.csv", index=False)

    states_df = classify_coagulation_states(coag_features, aging_velocity, predictions_df)
    plot_coagulation_state_scatter(states_df)

    ordered_tissues = predictions_df.sort_values("true_velocity")["tissue"].unique()
    coag_ordered = coag_features.loc[ordered_tissues]
    ecm_genes = panels["collagen"] + panels["mmp"]
    ecm_features = prepare_feature_matrix(df, ecm_genes)
    ecm_features = ecm_features.reindex(coag_ordered.index).fillna(0)

    sequences, temporal_targets, target_tissues = build_temporal_sequences(
        coag_ordered,
        ecm_features,
    )
    lstm_metrics, lstm_model = train_lstm_temporal_model(sequences, temporal_targets)
    lstm_metrics.to_csv(BASE_DIR / "temporal_model_metrics_codex.csv", index=False)

    module_assignments = {
        **{g: "coag" for g in panels["coag"]},
        **{g: "serpin" for g in panels["serpin"]},
        **{g: "collagen" for g in panels["collagen"]},
        **{g: "mmp" for g in panels["mmp"]},
    }
    aggregated_matrix = pd.concat(
        [coag_ordered, ecm_features.reindex(columns=panels["collagen"] + panels["mmp"])],
        axis=1,
    ).fillna(0)
    early_change_df = identify_early_change_proteins(aggregated_matrix, module_assignments)
    plot_temporal_trajectory(aggregated_matrix, module_assignments)

    run_network_analysis(df, panels)

    # Save summary notebook of key metrics for manual reporting
    summary = {
        "deep_nn_mean_r2": metrics_df["r2"].mean(),
        "deep_nn_mean_mae": metrics_df["mae"].mean(),
        "deep_nn_mean_rmse": metrics_df["rmse"].mean(),
        "baseline_r2": full_baseline["r2"],
        "baseline_mae": full_baseline["mae"],
        "baseline_rmse": full_baseline["rmse"],
        "temporal_r2": lstm_metrics[lstm_metrics["metric"] == "r2"]["value"].iloc[0],
    }
    cls_metrics_subset = combined_metrics[combined_metrics["task"] == "classification"]
    if not cls_metrics_subset.empty:
        summary["classification_mean_auc"] = cls_metrics_subset["auc"].mean()
        summary["classification_mean_accuracy"] = cls_metrics_subset["accuracy"].mean()
    with open(BASE_DIR / "analysis_summary_codex.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
