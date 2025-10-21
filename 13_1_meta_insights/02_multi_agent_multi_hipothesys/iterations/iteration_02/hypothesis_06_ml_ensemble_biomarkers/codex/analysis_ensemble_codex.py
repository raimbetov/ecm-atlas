"""
analysis_ensemble_codex.py

End-to-end pipeline for biomarker discovery using ensemble ML models (Random Forest, XGBoost, Neural Network),
stacking ensemble, SHAP-based interpretability, and reduced biomarker panel validation.

Outputs: model artifacts, metrics, SHAP values, visualizations, and supporting documentation files.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, precision_recall_fscore_support,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

plt.switch_backend("Agg")

DATA_PATH = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
)
OUTPUT_DIR = Path(
    "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_02/hypothesis_06_ml_ensemble_biomarkers/codex"
)
AGENT_TAG = "codex"
SEED = 42


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> Dict[str, Path]:
    viz_dir = OUTPUT_DIR / "visualizations_codex"
    shap_dep_dir = viz_dir / "shap_dependence_plots_codex"
    viz_dir.mkdir(exist_ok=True)
    shap_dep_dir.mkdir(parents=True, exist_ok=True)
    return {"viz": viz_dir, "shap_dep": shap_dep_dir}


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    tissue_index_train: List[str]
    tissue_index_test: List[str]
    scaler: Optional[StandardScaler]
    pivot_df: pd.DataFrame
    velocity: pd.Series


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def prepare_dataset() -> DatasetBundle:
    df = pd.read_csv(DATA_PATH)
    # unify tissue identifier using Organ + Compartment, fallback to Organ only
    df["Organ"] = df["Organ"].fillna("UnknownOrgan")
    df["Compartment"] = df["Compartment"].fillna("UnknownCompartment")
    df["Tissue_ID"] = df["Organ"].astype(str).str.strip() + "__" + df["Compartment"].astype(str).str.strip()

    # pivot to tissue (rows) × protein (columns)
    pivot_df = (
        df.pivot_table(
            index="Tissue_ID",
            columns="Gene_Symbol",
            values="Zscore_Delta",
            aggfunc="mean",
        )
        .sort_index()
    )

    feature_names = pivot_df.columns.tolist()

    # Replace missing z-scores with 0 (neutral change)
    pivot_filled = pivot_df.fillna(0.0)

    # Compute aging velocity as mean absolute delta per tissue
    velocity = pivot_filled.abs().mean(axis=1)

    # Binary label via median split: fast-aging (1) vs slow-aging (0)
    median_velocity = float(velocity.median())
    labels = (velocity > median_velocity).astype(int)

    X = pivot_filled.values.astype(np.float32)
    y = labels.values.astype(int)

    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X,
        y,
        pivot_filled.index.tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    # Standard scaler fitted on training data for models that need it (NN, Logistic)
    scaler = StandardScaler()
    scaler.fit(X_train)

    return DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        tissue_index_train=train_index,
        tissue_index_test=test_index,
        scaler=scaler,
        pivot_df=pivot_filled,
        velocity=velocity,
    )


# -----------------------------------------------------------------------------
# Torch MLP wrapper compatible with scikit-learn
# -----------------------------------------------------------------------------


class MLPNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim: Optional[int] = None,
        epochs: int = 150,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        val_fraction: float = 0.2,
        random_state: int = SEED,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[MLPNet] = None
        self.best_state_dict: Optional[dict] = None
        self._fitted = False
        self.history: Dict[str, List[float]] = {}

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "val_fraction": self.val_fraction,
            "random_state": self.random_state,
            "device": self.device,
        }

    def set_params(self, **params) -> "TorchNNClassifier":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _build_model(self, input_dim: int) -> None:
        self.model = MLPNet(input_dim)
        self.model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchNNClassifier":
        set_global_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        n_samples, n_features = X.shape
        if self.input_dim is None:
            self.input_dim = n_features
        elif self.input_dim != n_features:
            raise ValueError("Input dimension mismatch between provided data and estimator configuration.")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._build_model(self.input_dim)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y,
            test_size=self.val_fraction,
            random_state=self.random_state,
            stratify=y,
        )

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).unsqueeze(1).float(),
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).unsqueeze(1).float(),
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = math.inf
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            epoch_train_loss = running_loss / len(train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds = self.model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            epoch_val_loss = val_loss / len(val_loader.dataset)

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)

            if epoch_val_loss < best_val_loss - 1e-4:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Load best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.history = history
        self._fitted = True
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit before predict.")
        return self.scaler.transform(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError("TorchNNClassifier instance is not fitted yet.")
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._transform(X)
        tensor = torch.from_numpy(X_scaled).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            probs = self.model(tensor).cpu().numpy().reshape(-1, 1)
        probs = np.hstack([1 - probs, probs])
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)


# -----------------------------------------------------------------------------
# Metrics helpers
# -----------------------------------------------------------------------------

def compute_classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = (proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_true, proba)),
    }
    return metrics


# -----------------------------------------------------------------------------
# Base model training
# -----------------------------------------------------------------------------

def train_random_forest(bundle: DatasetBundle) -> Tuple[RandomForestClassifier, Dict[str, float], Dict[str, float]]:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        oob_score=True,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    rf.fit(bundle.X_train, bundle.y_train)
    proba_test = rf.predict_proba(bundle.X_test)[:, 1]
    metrics = compute_classification_metrics(bundle.y_test, proba_test)
    extra = {"oob_score": float(rf.oob_score_)}
    return rf, metrics, extra


def train_xgboost(bundle: DatasetBundle) -> Tuple[xgb.XGBClassifier, Dict[str, float], Dict[str, float]]:
    eval_size = 0.2
    X_tr, X_val, y_tr, y_val = train_test_split(
        bundle.X_train,
        bundle.y_train,
        test_size=eval_size,
        random_state=SEED,
        stratify=bundle.y_train,
    )
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        reg_lambda=1.0,
        random_state=SEED,
        eval_metric="auc",
        use_label_encoder=False,
    )
    xgb_model.set_params(early_stopping_rounds=20)
    xgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    proba_test = xgb_model.predict_proba(bundle.X_test)[:, 1]
    metrics = compute_classification_metrics(bundle.y_test, proba_test)
    best_iteration = xgb_model.best_iteration
    if best_iteration is None:
        best_iteration = xgb_model.n_estimators
    extra = {"best_iteration": int(best_iteration)}
    return xgb_model, metrics, extra


def train_neural_network(bundle: DatasetBundle) -> Tuple[TorchNNClassifier, Dict[str, float]]:
    nn_clf = TorchNNClassifier(input_dim=bundle.X_train.shape[1])
    nn_clf.fit(bundle.X_train, bundle.y_train)
    proba_test = nn_clf.predict_proba(bundle.X_test)[:, 1]
    metrics = compute_classification_metrics(bundle.y_test, proba_test)
    return nn_clf, metrics


# -----------------------------------------------------------------------------
# Cross-validated stacking ensemble
# -----------------------------------------------------------------------------


def stacking_ensemble(
    bundle: DatasetBundle,
    rf_model: RandomForestClassifier,
    xgb_model: xgb.XGBClassifier,
    nn_model: TorchNNClassifier,
    n_splits: int = 5,
) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    base_models = {
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=SEED,
            class_weight="balanced_subsample",
        ),
        "xgb": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            reg_lambda=1.0,
            random_state=SEED,
            eval_metric="auc",
            use_label_encoder=False,
        ),
        "nn": TorchNNClassifier(input_dim=bundle.X_train.shape[1]),
    }

    oof_preds = {name: np.zeros(len(bundle.y_train), dtype=float) for name in base_models}
    fold_metrics: Dict[str, List[float]] = {"auc": [], "accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(bundle.X_train, bundle.y_train)):
        X_tr, X_val = bundle.X_train[train_idx], bundle.X_train[val_idx]
        y_tr, y_val = bundle.y_train[train_idx], bundle.y_train[val_idx]

        # Random Forest fold training
        fold_rf = base_models["rf"]
        fold_rf.fit(X_tr, y_tr)
        oof_preds["rf"][val_idx] = fold_rf.predict_proba(X_val)[:, 1]

        # XGBoost fold training with early stopping on inner validation
        inner_train, inner_val, inner_y_tr, inner_y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr
        )
        fold_xgb = base_models["xgb"]
        fold_xgb.set_params(early_stopping_rounds=20)
        fold_xgb.fit(
            inner_train,
            inner_y_tr,
            eval_set=[(inner_val, inner_y_val)],
            verbose=False,
        )
        oof_preds["xgb"][val_idx] = fold_xgb.predict_proba(X_val)[:, 1]

        # Neural network fold training
        fold_nn = base_models["nn"]
        fold_nn.fit(X_tr, y_tr)
        oof_preds["nn"][val_idx] = fold_nn.predict_proba(X_val)[:, 1]

    # Concatenate OOF predictions for meta-model training
    meta_features = np.vstack([oof_preds["rf"], oof_preds["xgb"], oof_preds["nn"]]).T
    meta_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    meta_model.fit(meta_features, bundle.y_train)

    # Evaluate ensemble on training via OOF predictions
    oof_meta_pred = meta_model.predict_proba(meta_features)[:, 1]
    oof_metrics = compute_classification_metrics(bundle.y_train, oof_meta_pred)

    # Prepare final base model predictions on test data using fully trained models
    test_meta_features = np.vstack(
        [
            rf_model.predict_proba(bundle.X_test)[:, 1],
            xgb_model.predict_proba(bundle.X_test)[:, 1],
            nn_model.predict_proba(bundle.X_test)[:, 1],
        ]
    ).T
    ensemble_test_proba = meta_model.predict_proba(test_meta_features)[:, 1]
    test_metrics = compute_classification_metrics(bundle.y_test, ensemble_test_proba)

    # Compute ROC curve data for plotting
    roc_data = {}
    fpr, tpr, _ = roc_curve(bundle.y_test, ensemble_test_proba)
    roc_data["ensemble"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": test_metrics["auc"]}

    return {
        "meta_model": meta_model,
        "oof_meta_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "test_proba": ensemble_test_proba,
        "meta_features_train": meta_features,
        "test_meta_features": test_meta_features,
        "roc_data": roc_data,
    }


# -----------------------------------------------------------------------------
# SHAP analysis
# -----------------------------------------------------------------------------

def compute_tree_shap(model, X_background: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class for binary
    shap_values = np.asarray(shap_values)
    n_samples = X_eval.shape[0]
    if shap_values.ndim == 3:
        # Handle shapes like (samples, features, classes)
        if shap_values.shape[0] == n_samples:
            shap_values = shap_values[:, :, -1]
        elif shap_values.shape[1] == n_samples:
            shap_values = shap_values[-1, :, :]
        elif shap_values.shape[2] == n_samples:
            shap_values = shap_values[:, :, -1]
        else:
            shap_values = shap_values.reshape(n_samples, -1)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(n_samples, -1)
    elif shap_values.ndim != 2:
        shap_values = shap_values.reshape(n_samples, -1)
    return shap_values


def compute_nn_shap(nn_model: TorchNNClassifier, X_background: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    # Ensure CPU tensors for SHAP compatibility
    device = torch.device(nn_model.device)
    model = MLPNet(nn_model.input_dim)
    model.load_state_dict(nn_model.model.state_dict())
    model.eval()
    model.to(device)

    background = X_background
    if background.shape[0] > 200:
        background = background[:200]

    background_tensor = torch.from_numpy(nn_model._transform(background)).float().to(device)
    eval_tensor = torch.from_numpy(nn_model._transform(X_eval)).float().to(device)

    explainer = shap.DeepExplainer(model, background_tensor)
    shap_values = explainer.shap_values(eval_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()
    else:
        shap_values = np.asarray(shap_values)
    shap_values = np.squeeze(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(X_eval.shape[0], -1)
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    elif shap_values.ndim != 2:
        shap_values = shap_values.reshape(X_eval.shape[0], -1)
    return shap_values


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------

def save_model_comparison_bar(metrics_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ["Random Forest", "XGBoost", "Neural Network", "Stacked Ensemble"]
    subset = metrics_df.set_index("model").loc[order]
    subset["auc"].plot(kind="bar", color=["#6baed6", "#fd8d3c", "#74c476", "#9c9ede"], ax=ax)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model AUC Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_roc_curve(curves: Dict[str, Dict[str, List[float]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in curves.items():
        fpr = np.asarray(data["fpr"])
        tpr = np.asarray(data["tpr"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Reduced panel evaluation
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main() -> None:
    set_global_seed(SEED)
    dirs = ensure_dirs()

    bundle = prepare_dataset()

    # Train base models
    rf_model, rf_metrics, rf_extra = train_random_forest(bundle)
    xgb_model, xgb_metrics, xgb_extra = train_xgboost(bundle)
    nn_model, nn_metrics = train_neural_network(bundle)

    # Save base model artifacts
    joblib.dump(rf_model, OUTPUT_DIR / f"rf_model_{AGENT_TAG}.pkl")
    joblib.dump(xgb_model, OUTPUT_DIR / f"xgboost_model_{AGENT_TAG}.pkl")
    torch.save(nn_model.model.state_dict(), OUTPUT_DIR / f"nn_model_{AGENT_TAG}.pth")

    # Ensemble stacking
    ensemble_info = stacking_ensemble(bundle, rf_model, xgb_model, nn_model)
    ensemble_metrics = ensemble_info["test_metrics"]

    # Collect ROC data for base models
    roc_curves = ensemble_info["roc_data"]
    for name, model, proba in [
        ("Random Forest", rf_model, rf_model.predict_proba(bundle.X_test)[:, 1]),
        ("XGBoost", xgb_model, xgb_model.predict_proba(bundle.X_test)[:, 1]),
        ("Neural Network", nn_model, nn_model.predict_proba(bundle.X_test)[:, 1]),
    ]:
        fpr, tpr, _ = roc_curve(bundle.y_test, proba)
        roc_curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc_score(bundle.y_test, proba))}

    # Save ensemble model artifact (including meta-model and scaler for NN)
    ensemble_artifact = {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "nn_state_dict": nn_model.model.state_dict(),
        "nn_params": nn_model.get_params(),
        "nn_scaler": nn_model.scaler,
        "meta_model": ensemble_info["meta_model"],
        "feature_names": bundle.feature_names,
    }
    joblib.dump(ensemble_artifact, OUTPUT_DIR / f"ensemble_model_{AGENT_TAG}.pkl")

    # Metrics DataFrame
    metrics_records = [
        {"model": "Random Forest", **rf_metrics, **rf_extra},
        {"model": "XGBoost", **xgb_metrics, **xgb_extra},
        {"model": "Neural Network", **nn_metrics},
        {"model": "Stacked Ensemble", **ensemble_metrics},
    ]
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(OUTPUT_DIR / f"model_comparison_{AGENT_TAG}.csv", index=False)

    # Save visualizations
    save_model_comparison_bar(metrics_df, dirs["viz"] / f"model_comparison_bar_{AGENT_TAG}.png")
    save_roc_curve(roc_curves, OUTPUT_DIR / f"ensemble_roc_curve_{AGENT_TAG}.png")

    # SHAP computations on test data
    background_idx = np.random.choice(bundle.X_train.shape[0], size=min(200, bundle.X_train.shape[0]), replace=False)
    X_background = bundle.X_train[background_idx]
    X_test = bundle.X_test

    shap_rf = compute_tree_shap(rf_model, X_background, X_test)
    shap_xgb = compute_tree_shap(xgb_model, X_background, X_test)
    shap_nn = compute_nn_shap(nn_model, X_background, X_test)

    np.save(OUTPUT_DIR / f"shap_values_rf_{AGENT_TAG}.npy", shap_rf)
    np.save(OUTPUT_DIR / f"shap_values_xgboost_{AGENT_TAG}.npy", shap_xgb)

    # SHAP aggregation
    shap_nn = shap_nn  # already numpy
    shap_agg = (np.abs(shap_rf) + np.abs(shap_xgb) + np.abs(shap_nn)) / 3
    mean_abs_shap = shap_agg.mean(axis=0)

    feature_importance_df = pd.DataFrame(
        {
            "Protein": bundle.feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    # Feature selection via RFE
    scaler_for_rfe = StandardScaler()
    X_train_scaled = scaler_for_rfe.fit_transform(bundle.X_train)
    logistic_rfe = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=SEED,
    )
    # Select top 8 features (within required 5-10)
    rfe = RFE(logistic_rfe, n_features_to_select=8)
    rfe.fit(X_train_scaled, bundle.y_train)
    rfe_support = rfe.support_

    feature_importance_df["rfe_selected"] = rfe_support
    feature_importance_df["rfe_rank"] = rfe.ranking_

    consensus = feature_importance_df[feature_importance_df["rfe_selected"]].head(10)
    biometric_panel = consensus.head(8)

    biomarker_panel_df = biometric_panel.copy()
    biomarker_panel_df["rank"] = np.arange(1, len(biomarker_panel_df) + 1)
    biomarker_panel_df.to_csv(OUTPUT_DIR / f"biomarker_panel_{AGENT_TAG}.csv", index=False)

    selected_feature_indices = [bundle.feature_names.index(protein) for protein in biomarker_panel_df["Protein"]]

    # SHAP summary plot using aggregated shap values
    shap.summary_plot(
        shap_agg,
        features=bundle.X_test,
        feature_names=bundle.feature_names,
        show=False,
        plot_type="bar",
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(dirs["viz"] / f"shap_summary_plot_{AGENT_TAG}.png", dpi=300)
    plt.close()

    # Dependence plots for top features
    top_features = biomarker_panel_df["Protein"].tolist()
    X_test_df = pd.DataFrame(bundle.X_test, columns=bundle.feature_names)
    for protein in top_features:
        feature_idx = bundle.feature_names.index(protein)
        shap.dependence_plot(
            protein,
            shap_agg,
            X_test_df,
            show=False,
            interaction_index=None,
        )
        plt.tight_layout()
        plt.savefig(dirs["shap_dep"] / f"{protein}_dependence_{AGENT_TAG}.png", dpi=300)
        plt.close()

    # Reduced panel evaluation
    # Reduced feature bundle
    reduced_feature_names = [bundle.feature_names[idx] for idx in selected_feature_indices]
    reduced_bundle = DatasetBundle(
        X_train=bundle.X_train[:, selected_feature_indices],
        X_test=bundle.X_test[:, selected_feature_indices],
        y_train=bundle.y_train,
        y_test=bundle.y_test,
        feature_names=reduced_feature_names,
        tissue_index_train=bundle.tissue_index_train,
        tissue_index_test=bundle.tissue_index_test,
        scaler=StandardScaler().fit(bundle.X_train[:, selected_feature_indices]),
        pivot_df=bundle.pivot_df.iloc[:, selected_feature_indices],
        velocity=bundle.velocity,
    )

    rf_reduced, rf_reduced_metrics, _ = train_random_forest(reduced_bundle)
    xgb_reduced, xgb_reduced_metrics, _ = train_xgboost(reduced_bundle)
    nn_reduced, nn_reduced_metrics = train_neural_network(reduced_bundle)
    ensemble_reduced_info = stacking_ensemble(reduced_bundle, rf_reduced, xgb_reduced, nn_reduced)
    reduced_ensemble_metrics = ensemble_reduced_info["test_metrics"]

    reduced_df = pd.DataFrame(
        [
            {"model": "Random Forest", **rf_reduced_metrics},
            {"model": "XGBoost", **xgb_reduced_metrics},
            {"model": "Neural Network", **nn_reduced_metrics},
            {"model": "Stacked Ensemble", **reduced_ensemble_metrics},
        ]
    )
    reduced_df.to_csv(OUTPUT_DIR / f"reduced_panel_performance_{AGENT_TAG}.csv", index=False)
    reduced_metrics = {
        "rf": rf_reduced_metrics,
        "xgb": xgb_reduced_metrics,
        "nn": nn_reduced_metrics,
        "ensemble": reduced_ensemble_metrics,
    }

    # Ensemble CV/Test metrics
    ensemble_perf_df = pd.DataFrame(
        {
            "metric": list(ensemble_info["oof_meta_metrics"].keys()),
            "train_oof": list(ensemble_info["oof_meta_metrics"].values()),
            "test": [ensemble_info["test_metrics"][k] for k in ensemble_info["oof_meta_metrics"].keys()],
        }
    )
    ensemble_perf_df.to_csv(OUTPUT_DIR / f"ensemble_performance_{AGENT_TAG}.csv", index=False)

    # Persist Deep SHAP values
    np.save(OUTPUT_DIR / f"shap_values_nn_{AGENT_TAG}.npy", shap_nn)

    # Save summary JSON for downstream documentation
    summary_payload = {
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "nn_metrics": nn_metrics,
        "ensemble_metrics": ensemble_metrics,
        "reduced_metrics": reduced_metrics,
        "biomarker_panel": biomarker_panel_df.to_dict(orient="records"),
        "n_features": len(bundle.feature_names),
        "n_tissues": len(bundle.pivot_df),
        "median_velocity": float(bundle.velocity.median()),
    }
    with open(OUTPUT_DIR / f"analysis_summary_{AGENT_TAG}.json", "w") as f:
        json.dump(summary_payload, f, indent=2)


if __name__ == "__main__":
    main()
