"""Calcium signaling cascade analysis: mediation, ML models, and network synthesis."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
WORKSPACE = Path(__file__).resolve().parent
VIS_DIR = WORKSPACE / "visualizations_codex"
RESULTS_DIR = WORKSPACE
AGENT = "codex"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

S100_GENES = ["S100A8", "S100A9", "S100A10", "S100B", "S100A12", "S100A11", "S100A16", "S100A4"]
CALM_GENES = ["CALM1", "CALM2", "CALM3"]
CAMK_GENES = ["CAMK1", "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G"]
TARGET_GENES = ["LOX", "LOXL2", "LOXL3", "LOXL4", "TGM1", "TGM2", "TGM3"]
COLLAGEN_GENES = ["COL1A1", "COL3A1"]

GRAPH_GENES = S100_GENES + CALM_GENES + CAMK_GENES + ["LOX", "LOXL4", "TGM2"]
BOOTSTRAP_SAMPLES = 10000


class CalciumMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpressionAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


class SimpleGCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("ij,bjk->bik", adj, x)
        return support @ self.weight


class CalciumGCN(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layer1 = SimpleGCNLayer(feature_dim, hidden_dim)
        self.layer2 = SimpleGCNLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.layer1(x, adj))
        h2 = torch.relu(self.layer2(h1, adj))
        pooled = h2.mean(dim=1)
        return self.out(pooled)


@dataclass
class MediationResult:
    effect_type: str
    s100: str
    mediator1: str
    mediator2: Optional[str]
    target: str
    total_effect: float
    direct_effect: float
    indirect_effect: float
    mediated_pct: float
    p_value: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "effect_type": self.effect_type,
            "S100": self.s100,
            "CALM": self.mediator1,
            "CAMK": self.mediator2 or "",
            "Target": self.target,
            "total_effect": self.total_effect,
            "direct_effect": self.direct_effect,
            "indirect_effect": self.indirect_effect,
            "mediated_pct": self.mediated_pct,
            "p_value": self.p_value,
        }


def ensure_directories() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Gene_Symbol"] = df["Gene_Symbol"].str.upper()
    df["Tissue"] = df["Tissue"].astype(str)
    return df


def pivot_matrix(df: pd.DataFrame, genes: Sequence[str]) -> pd.DataFrame:
    sub = df[df["Gene_Symbol"].isin(genes)]
    piv = sub.pivot_table(values="Zscore_Delta", index="Tissue", columns="Gene_Symbol", aggfunc="mean")
    return piv


def build_feature_panel(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    panels = {
        "s100": pivot_matrix(df, S100_GENES),
        "calm": pivot_matrix(df, CALM_GENES),
        "camk": pivot_matrix(df, CAMK_GENES),
        "targets": pivot_matrix(df, TARGET_GENES),
        "collagen": pivot_matrix(df, COLLAGEN_GENES),
    }
    return panels


def compute_stiffness(panels: Dict[str, pd.DataFrame]) -> pd.Series:
    cross = panels["targets"]
    collagen = panels["collagen"]
    idx = cross.index.union(collagen.index)
    lox = cross.get("LOX", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    tgm2 = cross.get("TGM2", pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)
    col1 = collagen.get("COL1A1", pd.Series(np.nan, index=idx)).reindex(idx)
    col3 = collagen.get("COL3A1", pd.Series(np.nan, index=idx)).reindex(idx)
    ratio = (col1 / col3.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    stiffness = 0.5 * lox + 0.3 * tgm2 + 0.2 * ratio
    stiffness.name = "stiffness"
    return stiffness.dropna()


def align_features(index: pd.Index, matrix: pd.DataFrame) -> pd.DataFrame:
    aligned = matrix.reindex(index).fillna(0.0)
    return aligned


def load_external_expression() -> Optional[pd.DataFrame]:
    path = WORKSPACE / "external_datasets" / "GSE11475_expression_codex.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def fit_imputation_models(external: pd.DataFrame, features: List[str], targets: List[str]) -> Tuple[Dict[str, Ridge], List[Dict[str, float]]]:
    models: Dict[str, Ridge] = {}
    stats_records: List[Dict[str, float]] = []
    X = external[features]
    for gene in targets:
        if gene not in external.columns:
            continue
        y = external[gene]
        mask = (~X.isna().any(axis=1)) & (~y.isna())
        if mask.sum() < 5:
            continue
        model = Ridge(alpha=0.5, random_state=SEED)
        model.fit(X[mask], y[mask])
        score = model.score(X[mask], y[mask])
        models[gene] = model
        stats_records.append({
            "gene": gene,
            "r2_train": float(score),
            "n_samples": int(mask.sum()),
        })
    return models, stats_records


def impute_with_external(
    s100_matrix: pd.DataFrame,
    external: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, float]]]:
    shared_features = [g for g in s100_matrix.columns if g in external.columns]
    if not shared_features:
        return pd.DataFrame(index=s100_matrix.index), pd.DataFrame(index=s100_matrix.index), []
    calm_models, calm_stats = fit_imputation_models(external, shared_features, CALM_GENES)
    camk_models, camk_stats = fit_imputation_models(external, shared_features, CAMK_GENES)
    calm_pred = pd.DataFrame(index=s100_matrix.index)
    for gene, model in calm_models.items():
        calm_pred[gene] = model.predict(s100_matrix[shared_features])
    camk_pred = pd.DataFrame(index=s100_matrix.index)
    for gene, model in camk_models.items():
        camk_pred[gene] = model.predict(s100_matrix[shared_features])
    stats_records = calm_stats + camk_stats
    return calm_pred, camk_pred, stats_records


def spearman_matrix(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=a.columns, columns=b.columns, dtype=float)
    for col_a in a.columns:
        for col_b in b.columns:
            rho, pval = stats.spearmanr(a[col_a], b[col_b], nan_policy="omit")
            results.loc[col_a, col_b] = rho
    return results


def correlation_workflow(panels: Dict[str, pd.DataFrame], tissues: pd.Index) -> pd.DataFrame:
    s100 = align_features(tissues, panels["s100"])
    calm = align_features(tissues, panels["calm"])
    camk = align_features(tissues, panels["camk"])
    targets = align_features(tissues, panels["targets"])

    blocks = []
    blocks.append(spearman_matrix(s100, calm))
    blocks.append(spearman_matrix(calm, camk))
    blocks.append(spearman_matrix(camk, targets))

    heatmap_matrix = pd.concat(blocks, axis=1)
    heatmap_path = VIS_DIR / f"calcium_correlation_heatmap_{AGENT}.png"
    plt.figure(figsize=(18, 8))
    sns.heatmap(heatmap_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Spearman correlations: S100→CALM→CAMK→Targets")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

    network_rows = []
    for block, (left, right) in zip(blocks, [(s100, calm), (calm, camk), (camk, targets)]):
        for src in block.index:
            for dst in block.columns:
                rho = block.loc[src, dst]
                network_rows.append({
                    "source": src,
                    "target": dst,
                    "rho": rho,
                    "abs_rho": abs(rho),
                })
    network_df = pd.DataFrame(network_rows)
    network_df.to_csv(RESULTS_DIR / f"correlation_network_calcium_{AGENT}.csv", index=False)

    G = nx.DiGraph()
    for _, row in network_df.iterrows():
        if abs(row["rho"]) >= 0.5:
            G.add_edge(row["source"], row["target"], weight=row["rho"])
    pos = nx.spring_layout(G, seed=SEED)
    plt.figure(figsize=(12, 10))
    edges = G.edges(data=True)
    colors = ["#1f77b4" if data["weight"] > 0 else "#d62728" for _, _, data in edges]
    widths = [abs(data["weight"]) * 3 for _, _, data in edges]
    nx.draw_networkx(G, pos=pos, with_labels=True, edge_color=colors, width=widths, node_size=1200, font_size=9)
    plt.title("Calcium cascade correlation network (|ρ|≥0.5)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"calcium_network_{AGENT}.png", dpi=300)
    plt.close()
    return network_df


def ols_coefficients(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X_design = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return beta


def single_mediation(x: np.ndarray, m: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES) -> Tuple[float, float, float, float]:
    total = ols_coefficients(y, x.reshape(-1, 1))[1]
    a = ols_coefficients(m, x.reshape(-1, 1))[1]
    coefs_y = ols_coefficients(y, np.column_stack([x, m]))
    direct = coefs_y[1]
    b = coefs_y[2]
    indirect = a * b

    idx = np.random.randint(0, len(x), size=(n_boot, len(x)))
    indirect_samples = []
    for sample_idx in idx:
        xs = x[sample_idx]
        ms = m[sample_idx]
        ys = y[sample_idx]
        a_s = ols_coefficients(ms, xs.reshape(-1, 1))[1]
        coefs_y_s = ols_coefficients(ys, np.column_stack([xs, ms]))
        b_s = coefs_y_s[2]
        indirect_samples.append(a_s * b_s)
    indirect_samples = np.array(indirect_samples)
    p_value = 2 * min(
        np.mean(indirect_samples >= 0),
        np.mean(indirect_samples <= 0),
    )
    return total, direct, indirect, p_value


def sequential_mediation(x: np.ndarray, m1: np.ndarray, m2: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_SAMPLES) -> Tuple[float, float, float, float]:
    total = ols_coefficients(y, x.reshape(-1, 1))[1]
    a1 = ols_coefficients(m1, x.reshape(-1, 1))[1]
    coefs_m2 = ols_coefficients(m2, np.column_stack([x, m1]))
    a2 = coefs_m2[2]
    coefs_y = ols_coefficients(y, np.column_stack([x, m1, m2]))
    direct = coefs_y[1]
    b = coefs_y[3]
    indirect = a1 * a2 * b

    idx = np.random.randint(0, len(x), size=(n_boot, len(x)))
    indirect_samples = []
    for sample_idx in idx:
        xs = x[sample_idx]
        m1s = m1[sample_idx]
        m2s = m2[sample_idx]
        ys = y[sample_idx]
        a1_s = ols_coefficients(m1s, xs.reshape(-1, 1))[1]
        coefs_m2_s = ols_coefficients(m2s, np.column_stack([xs, m1s]))
        a2_s = coefs_m2_s[2]
        coefs_y_s = ols_coefficients(ys, np.column_stack([xs, m1s, m2s]))
        b_s = coefs_y_s[3]
        indirect_samples.append(a1_s * a2_s * b_s)
    indirect_samples = np.array(indirect_samples)
    p_value = 2 * min(np.mean(indirect_samples >= 0), np.mean(indirect_samples <= 0))
    return total, direct, indirect, p_value


def mediation_workflow(panels: Dict[str, pd.DataFrame], tissues: pd.Index, stiffness: pd.Series) -> pd.DataFrame:
    s100 = align_features(tissues, panels["s100"])
    calm = align_features(tissues, panels["calm"])
    camk = align_features(tissues, panels["camk"])
    targets = align_features(tissues, panels["targets"])

    results: List[MediationResult] = []
    for s_gene in S100_GENES:
        if s_gene not in s100:
            continue
        x = s100[s_gene].values
        for calm_gene in CALM_GENES:
            if calm_gene not in calm:
                continue
            m = calm[calm_gene].values
            for target_gene in TARGET_GENES:
                if target_gene not in targets:
                    continue
                y = targets[target_gene].values
                total, direct, indirect, p_val = single_mediation(x, m, y)
                mediated_pct = float(np.clip((indirect / total) * 100 if total != 0 else 0, -500, 500))
                results.append(MediationResult(
                    effect_type="S100→CALM→Target",
                    s100=s_gene,
                    mediator1=calm_gene,
                    mediator2=None,
                    target=target_gene,
                    total_effect=total,
                    direct_effect=direct,
                    indirect_effect=indirect,
                    mediated_pct=mediated_pct,
                    p_value=p_val,
                ))
                for camk_gene in CAMK_GENES:
                    if camk_gene not in camk:
                        continue
                    m2 = camk[camk_gene].values
                    total2, direct2, indirect2, p_val2 = sequential_mediation(x, m, m2, y)
                    mediated_pct2 = float(np.clip((indirect2 / total2) * 100 if total2 != 0 else 0, -500, 500))
                    results.append(MediationResult(
                        effect_type="S100→CALM→CAMK→Target",
                        s100=s_gene,
                        mediator1=calm_gene,
                        mediator2=camk_gene,
                        target=target_gene,
                        total_effect=total2,
                        direct_effect=direct2,
                        indirect_effect=indirect2,
                        mediated_pct=mediated_pct2,
                        p_value=p_val2,
                    ))
    df = pd.DataFrame([r.to_dict() for r in results])
    if df.empty:
        return df
    df.to_csv(RESULTS_DIR / f"mediation_results_{AGENT}.csv", index=False)

    filtered = df[(df["effect_type"] == "S100→CALM→CAMK→Target")]
    if not filtered.empty:
        pivot = filtered.pivot_table(
            values="mediated_pct",
            index=["S100", "CALM"],
            columns=["CAMK", "Target"],
            aggfunc="mean",
        )
        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot, cmap="viridis", annot=False)
        plt.title("% mediated for sequential path")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"mediation_heatmap_{AGENT}.png", dpi=300)
        plt.close()

    # Extract best path for diagram
    strong = filtered[(filtered["mediated_pct"].abs() >= 30) & (filtered["p_value"] <= 0.05)]
    if strong.empty:
        strong = filtered.nsmallest(1, "p_value")
    if not strong.empty:
        top = strong.iloc[0]
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.axis("off")
        nodes = ["S100", "CALM", "CAMK", "Target"]
        positions = {"S100": (0.1, 0.5), "CALM": (0.35, 0.5), "CAMK": (0.6, 0.5), "Target": (0.85, 0.5)}
        for node in nodes:
            ax.annotate(node, positions[node], ha="center", va="center", bbox=dict(boxstyle="round", fc="#e0f3ff"))
        ax.annotate(f"{top['S100']}", (0.1, 0.65), ha="center", fontsize=9)
        ax.annotate(f"{top['CALM']}", (0.35, 0.65), ha="center", fontsize=9)
        ax.annotate(f"{top['CAMK']}", (0.6, 0.65), ha="center", fontsize=9)
        ax.annotate(f"{top['Target']}", (0.85, 0.65), ha="center", fontsize=9)
        ax.annotate(f"total={top['total_effect']:.2f}", (0.5, 0.2), ha="center", fontsize=9)
        ax.annotate(f"direct={top['direct_effect']:.2f}", (0.5, 0.1), ha="center", fontsize=9)
        ax.annotate(f"indirect={top['indirect_effect']:.2f}\n%mediated={top['mediated_pct']:.1f}", (0.5, 0.0), ha="center", fontsize=9)
        for i in range(len(nodes) - 1):
            ax.annotate(
                "",
                xy=positions[nodes[i + 1]],
                xytext=positions[nodes[i]],
                arrowprops=dict(arrowstyle="->", lw=2),
            )
        plt.savefig(VIS_DIR / f"mediation_diagram_{AGENT}.png", dpi=300, bbox_inches="tight")
        plt.close()
    return df


def prepare_features(panels: Dict[str, pd.DataFrame], tissues: pd.Index) -> Dict[str, pd.DataFrame]:
    features = {
        "model_a": align_features(tissues, panels["s100"]),
        "model_b": pd.concat([align_features(tissues, panels["s100"]), align_features(tissues, panels["calm"])], axis=1),
        "model_c": pd.concat([
            align_features(tissues, panels["s100"]),
            align_features(tissues, panels["calm"]),
            align_features(tissues, panels["camk"]),
        ], axis=1),
    }
    features = {k: v.loc[:, sorted(v.columns)] for k, v in features.items()}
    return features


def train_mlp(features: pd.DataFrame, target: pd.Series, model_path: Optional[Path] = None) -> Tuple[CalciumMLP, Dict[str, float], pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    y = target.loc[features.index].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features.index.values, test_size=0.2, random_state=SEED
    )

    model = CalciumMLP(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    best_state = None
    best_loss = float("inf")
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_tensor)
        loss = criterion(preds, y_train_tensor)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor).squeeze().numpy()
        test_pred = model(X_test_tensor).squeeze().numpy()

    metrics = {
        "r2_train": r2_score(y_train, train_pred),
        "r2_test": r2_score(y_test, test_pred),
        "mae_test": mean_absolute_error(y_test, test_pred),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, test_pred))),
    }

    predictions = pd.DataFrame({
        "tissue": np.concatenate([idx_train, idx_test]),
        "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
        "actual": np.concatenate([y_train, y_test]),
        "predicted": np.concatenate([train_pred, test_pred]),
    })

    if model_path:
        torch.save(model.state_dict(), model_path)

    return model, metrics, predictions, scaler


def autoencoder_workflow(features: pd.DataFrame) -> Tuple[np.ndarray, float]:
    model = ExpressionAutoencoder(features.shape[1], latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    X = torch.tensor(features.values, dtype=torch.float32)
    best_state = None
    best_loss = float("inf")
    for epoch in range(2000):
        optimizer.zero_grad()
        recon, latent = model(X)
        loss = criterion(recon, X)
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    with torch.no_grad():
        _, latent = model(X)
    latent_np = latent.numpy()
    torch.save(model.state_dict(), WORKSPACE / f"autoencoder_model_{AGENT}.pth")
    latent_df = pd.DataFrame(latent_np, columns=[f"latent_{i+1}" for i in range(latent_np.shape[1])])
    latent_df.to_csv(WORKSPACE / f"autoencoder_latent_{AGENT}.csv", index=False)
    return latent_np, best_loss


def gcn_workflow(features: pd.DataFrame, target: pd.Series, adjacency: np.ndarray) -> Dict[str, float]:
    tissues = features.index.values
    X = torch.tensor(features.values[:, :, None], dtype=torch.float32)
    adj = torch.tensor(adjacency, dtype=torch.float32)
    y = torch.tensor(target.loc[tissues].values, dtype=torch.float32)

    model = CalciumGCN(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    idx_train, idx_test = train_test_split(np.arange(len(tissues)), test_size=0.2, random_state=SEED)
    idx_train_tensor = torch.tensor(idx_train)
    idx_test_tensor = torch.tensor(idx_test)

    best_state = None
    best_loss = float("inf")
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        preds = model(X[idx_train_tensor], adj)
        loss = criterion(preds.squeeze(), y[idx_train_tensor])
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_preds = model(X[idx_train_tensor], adj).squeeze().numpy()
        test_preds = model(X[idx_test_tensor], adj).squeeze().numpy()
    metrics = {
        "r2_train": r2_score(y[idx_train_tensor].numpy(), train_preds),
        "r2_test": r2_score(y[idx_test_tensor].numpy(), test_preds),
        "mae_test": mean_absolute_error(y[idx_test_tensor].numpy(), test_preds),
        "rmse_test": float(np.sqrt(mean_squared_error(y[idx_test_tensor].numpy(), test_preds))),
    }
    torch.save(model.state_dict(), WORKSPACE / f"gcn_model_{AGENT}.pth")
    return metrics


def build_adjacency(df: pd.DataFrame) -> np.ndarray:
    corr = df.corr(method="spearman").fillna(0.0)
    corr_values = np.abs(corr.values)
    np.fill_diagonal(corr_values, 0.0)
    threshold = 0.3
    mask = np.abs(corr_values) >= threshold
    adjacency = np.where(mask, corr_values, 0.0)
    adjacency = (adjacency + adjacency.T) / 2
    np.fill_diagonal(adjacency, 1.0)
    degree = adjacency.sum(axis=1)
    degree[degree <= 0] = 1e-6
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    normalized = d_inv_sqrt @ adjacency @ d_inv_sqrt
    return normalized


def clustering_workflow(features: pd.DataFrame, tissues: pd.Index) -> Dict[str, object]:
    gmm = GaussianMixture(n_components=3, random_state=SEED)
    gmm.fit(features.values)
    labels = gmm.predict(features.values)
    sil = silhouette_score(features.values, labels) if len(set(labels)) > 1 else np.nan
    cluster_df = pd.DataFrame({"tissue": tissues, "cluster": labels})
    cluster_df.to_csv(WORKSPACE / f"clustering_assignments_{AGENT}.csv", index=False)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=features.iloc[:, 0], y=features.iloc[:, 1], hue=labels, palette="deep")
    plt.title("Gaussian mixture clustering (first two genes)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"clustering_scatter_{AGENT}.png", dpi=300)
    plt.close()
    return {"silhouette": float(sil)}


def visualization_latent(latent: np.ndarray, tissues: pd.Index) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(latent[:, 0], latent[:, 1], c=np.arange(len(tissues)), cmap="Spectral")
    for idx, tissue in enumerate(tissues):
        plt.text(latent[idx, 0], latent[idx, 1], tissue, fontsize=8)
    plt.title("Autoencoder latent space (first two dimensions)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"autoencoder_latent_{AGENT}.png", dpi=300)
    plt.close()


def main() -> None:
    ensure_directories()
    df = load_dataset()
    panels = build_feature_panel(df)
    stiffness = compute_stiffness(panels)
    tissues = pd.Index(sorted(panels["s100"].index.intersection(stiffness.index)))
    external_df = load_external_expression()
    imputation_stats: List[Dict[str, float]] = []
    if external_df is not None and not tissues.empty:
        s100_for_impute = align_features(tissues, panels["s100"])
        calm_pred, camk_pred, imputation_stats = impute_with_external(s100_for_impute, external_df)
        if panels["calm"].empty:
            calm_combined = pd.DataFrame(index=tissues)
        else:
            calm_combined = panels["calm"].reindex(tissues)
        for gene in calm_pred.columns:
            existing = calm_combined.get(gene)
            if existing is None:
                calm_combined[gene] = calm_pred[gene]
            else:
                calm_combined[gene] = existing.fillna(calm_pred[gene])
        panels["calm"] = calm_combined.fillna(0.0)

        if panels["camk"].empty:
            camk_combined = pd.DataFrame(index=tissues)
        else:
            camk_combined = panels["camk"].reindex(tissues)
        for gene in camk_pred.columns:
            existing = camk_combined.get(gene)
            if existing is None:
                camk_combined[gene] = camk_pred[gene]
            else:
                camk_combined[gene] = existing.fillna(camk_pred[gene])
        panels["camk"] = camk_combined.fillna(0.0)

        imputed_df = pd.concat([calm_pred.add_suffix("_imputed"), camk_pred.add_suffix("_imputed")], axis=1)
        imputed_df.to_csv(WORKSPACE / f"imputed_calm_camk_{AGENT}.csv")
    network_df = correlation_workflow(panels, tissues)

    mediation_df = mediation_workflow(panels, tissues, stiffness)

    features = prepare_features(panels, tissues)
    metrics_records = []
    predictions_frames = []

    for label, feat in features.items():
        model_path = WORKSPACE / f"calcium_model_{label}_{AGENT}.pth"
        model, metrics, predictions, scaler = train_mlp(feat, stiffness, model_path=model_path if label == "model_c" else None)
        metrics_records.append({"model": label, **metrics, "n_features": feat.shape[1]})
        predictions["model"] = label
        predictions_frames.append(predictions)
        if label == "model_c":
            torch.save(model.state_dict(), WORKSPACE / f"calcium_signaling_model_{AGENT}.pth")
            joblib.dump(scaler, WORKSPACE / f"calcium_signaling_scaler_{AGENT}.joblib")

    predictions_df = pd.concat(predictions_frames, ignore_index=True)
    predictions_df.to_csv(WORKSPACE / f"model_predictions_{AGENT}.csv", index=False)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df["delta_r2_vs_model_a"] = metrics_df["r2_test"] - float(metrics_df.loc[metrics_df["model"] == "model_a", "r2_test"].iloc[0])
    metrics_df.to_csv(WORKSPACE / f"model_comparison_{AGENT}.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="model", y="r2_test", palette="viridis")
    plt.title("Model performance (R² test)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"model_performance_{AGENT}.png", dpi=300)
    plt.close()

    latent, recon_loss = autoencoder_workflow(features["model_c"])
    visualization_latent(latent, tissues)

    graph_features = features["model_c"].reindex(columns=GRAPH_GENES, fill_value=0.0)
    adjacency = build_adjacency(graph_features)
    gcn_metrics = gcn_workflow(graph_features, stiffness, adjacency)

    clustering_metrics = clustering_workflow(features["model_c"], tissues)

    summary = {
        "mlp_metrics": metrics_records,
        "gcn_metrics": gcn_metrics,
        "clustering": clustering_metrics,
        "reconstruction_loss": recon_loss,
        "top_mediation": [] if mediation_df.empty else mediation_df.nsmallest(5, "p_value").to_dict(orient="records"),
        "network_edges_strong": network_df[network_df["abs_rho"] >= 0.5].to_dict(orient="records"),
        "imputation_stats": imputation_stats,
    }
    with open(WORKSPACE / f"analysis_summary_{AGENT}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
