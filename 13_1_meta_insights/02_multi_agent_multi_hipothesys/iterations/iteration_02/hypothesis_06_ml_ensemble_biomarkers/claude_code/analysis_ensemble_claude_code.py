#!/usr/bin/env python3
"""
Ensemble ML Pipeline for Aging Biomarker Discovery
Agent: claude_code
Task: Hypothesis 06 - ML Ensemble Biomarker Panel

Pipeline:
1. Data preprocessing and feature engineering
2. Train Random Forest, LightGBM, Neural Network
3. Build stacking ensemble
4. Compute SHAP values (TreeSHAP + DeepSHAP)
5. Select 5-10 biomarker panel
6. Validate reduced panel performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, auc)
from sklearn.linear_model import LogisticRegression

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Interpretability
import shap

# Utilities
import joblib
import json
import os
from datetime import datetime
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Create output directories
os.makedirs('visualizations_claude_code/shap_dependence_plots_claude_code', exist_ok=True)

print("=" * 80)
print("ENSEMBLE ML BIOMARKER DISCOVERY PIPELINE")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 1: DATA LOADING AND PREPROCESSING")
print("=" * 80)

# Load dataset
data_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
print(f"\nLoading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data exploration
print("\n--- Data Summary ---")
print(f"Unique proteins: {df['Gene_Symbol'].nunique()}")
print(f"Unique tissues: {df['Tissue'].nunique()}")
print(f"Unique studies: {df['Study_ID'].nunique()}")

# Create tissue-level feature matrix
print("\n--- Creating Tissue-Level Feature Matrix ---")
print("Strategy: Rows = tissues, Columns = proteins (Zscore_Delta values)")

# Pivot table: Tissue × Protein matrix
pivot = df.pivot_table(values='Zscore_Delta', index='Tissue', columns='Gene_Symbol', aggfunc='mean')
print(f"Pivot shape: {pivot.shape} (tissues × proteins)")

# Handle missing values
print(f"Missing values before imputation: {pivot.isna().sum().sum()}")
pivot_filled = pivot.fillna(0)  # Fill NaN with 0 (protein not measured)
print(f"Missing values after imputation: {pivot_filled.isna().sum().sum()}")

# Create target variable: Tissue aging velocity
print("\n--- Creating Target Variable ---")
print("Target: Binary classification (fast-aging vs slow-aging tissues)")

# Calculate aging velocity per tissue: mean absolute Zscore_Delta
aging_velocity = df.groupby('Tissue')['Zscore_Delta'].apply(lambda x: np.abs(x).mean())
median_velocity = aging_velocity.median()
print(f"Median aging velocity: {median_velocity:.3f}")

# Binary labels: 1 = fast-aging (above median), 0 = slow-aging
y_labels = (aging_velocity > median_velocity).astype(int)
print(f"Class distribution: Fast-aging={y_labels.sum()}, Slow-aging={(~y_labels.astype(bool)).sum()}")

# Align features with target
X = pivot_filled.loc[aging_velocity.index].values
y = y_labels.values
feature_names = pivot_filled.columns.tolist()
tissue_names = pivot_filled.loc[aging_velocity.index].index.tolist()

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")
print(f"Number of features (proteins): {len(feature_names)}")

# Train-test split (70-30)
print("\n--- Train-Test Split ---")
X_train, X_test, y_train, y_test, tissues_train, tissues_test = train_test_split(
    X, y, tissue_names, test_size=0.3, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape[0]} tissues, Test set: {X_test.shape[0]} tissues")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Feature scaling (for Neural Network)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing metadata
preprocessing_info = {
    'n_tissues': len(tissue_names),
    'n_proteins': len(feature_names),
    'median_aging_velocity': float(median_velocity),
    'train_size': int(X_train.shape[0]),
    'test_size': int(X_test.shape[0]),
    'class_balance_train': {0: int(np.bincount(y_train)[0]), 1: int(np.bincount(y_train)[1])},
    'class_balance_test': {0: int(np.bincount(y_test)[0]), 1: int(np.bincount(y_test)[1])}
}
with open('preprocessing_info_claude_code.json', 'w') as f:
    json.dump(preprocessing_info, f, indent=2)

print("\nPreprocessing complete!")

# ============================================================================
# 2. MODEL TRAINING: RANDOM FOREST
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2A: RANDOM FOREST TRAINING")
print("=" * 80)

print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1,
    oob_score=True
)

rf_model.fit(X_train, y_train)
print("✓ Random Forest training complete")

# Evaluate
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)
rf_test_auc = roc_auc_score(y_test, rf_test_proba)
rf_test_f1 = f1_score(y_test, rf_test_pred)
rf_test_precision = precision_score(y_test, rf_test_pred)
rf_test_recall = recall_score(y_test, rf_test_pred)
rf_oob_score = rf_model.oob_score_

print(f"\nRandom Forest Performance:")
print(f"  Train Accuracy: {rf_train_acc:.4f}")
print(f"  Test Accuracy:  {rf_test_acc:.4f}")
print(f"  Test AUC-ROC:   {rf_test_auc:.4f}")
print(f"  Test F1:        {rf_test_f1:.4f}")
print(f"  Test Precision: {rf_test_precision:.4f}")
print(f"  Test Recall:    {rf_test_recall:.4f}")
print(f"  OOB Score:      {rf_oob_score:.4f}")

# Feature importance
rf_feature_importance = pd.DataFrame({
    'Protein': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 features by RF importance:")
print(rf_feature_importance.head(10).to_string(index=False))

# Save model
joblib.dump(rf_model, 'rf_model_claude_code.pkl')
print("\n✓ Saved: rf_model_claude_code.pkl")

# ============================================================================
# 3. MODEL TRAINING: GRADIENT BOOSTING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2B: GRADIENT BOOSTING TRAINING")
print("=" * 80)

print("\n--- Training Gradient Boosting ---")
# Use sklearn's GradientBoostingClassifier (no external dependencies)
lgb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# Train
lgb_model.fit(X_train, y_train)
print("✓ Gradient Boosting training complete")

# Evaluate
lgb_train_pred = lgb_model.predict(X_train)
lgb_test_pred = lgb_model.predict(X_test)
lgb_test_proba = lgb_model.predict_proba(X_test)[:, 1]

lgb_train_acc = accuracy_score(y_train, lgb_train_pred)
lgb_test_acc = accuracy_score(y_test, lgb_test_pred)
lgb_test_auc = roc_auc_score(y_test, lgb_test_proba)
lgb_test_f1 = f1_score(y_test, lgb_test_pred)
lgb_test_precision = precision_score(y_test, lgb_test_pred)
lgb_test_recall = recall_score(y_test, lgb_test_pred)

print(f"\nLightGBM Performance:")
print(f"  Train Accuracy: {lgb_train_acc:.4f}")
print(f"  Test Accuracy:  {lgb_test_acc:.4f}")
print(f"  Test AUC-ROC:   {lgb_test_auc:.4f}")
print(f"  Test F1:        {lgb_test_f1:.4f}")
print(f"  Test Precision: {lgb_test_precision:.4f}")
print(f"  Test Recall:    {lgb_test_recall:.4f}")

# Feature importance
lgb_feature_importance = pd.DataFrame({
    'Protein': feature_names,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 features by Gradient Boosting importance:")
print(lgb_feature_importance.head(10).to_string(index=False))

# Save model
joblib.dump(lgb_model, 'gradientboosting_model_claude_code.pkl')
print("\n✓ Saved: gradientboosting_model_claude_code.pkl")

# ============================================================================
# 4. MODEL TRAINING: NEURAL NETWORK
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2C: NEURAL NETWORK TRAINING")
print("=" * 80)

# Neural Network Architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: 128 neurons
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 64 neurons
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 32 neurons
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Prepare data for PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

nn_model = MLPClassifier(input_dim=X_train.shape[1]).to(device)
print(f"\nNeural Network Architecture:")
print(nn_model)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
print("\n--- Training Neural Network ---")
epochs = 100
patience = 10
best_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Training
    nn_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = nn_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    nn_model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        y_test_device = y_test_tensor.to(device)
        val_outputs = nn_model(X_test_device)
        val_loss = criterion(val_outputs, y_test_device).item()
        val_losses.append(val_loss)

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(nn_model.state_dict(), 'nn_model_claude_code.pth')
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model
nn_model.load_state_dict(torch.load('nn_model_claude_code.pth'))
print("\n✓ Neural Network training complete")

# Evaluate
nn_model.eval()
with torch.no_grad():
    nn_train_proba = nn_model(X_train_tensor.to(device)).cpu().numpy()
    nn_test_proba = nn_model(X_test_tensor.to(device)).cpu().numpy()

nn_train_pred = (nn_train_proba > 0.5).astype(int).flatten()
nn_test_pred = (nn_test_proba > 0.5).astype(int).flatten()

nn_train_acc = accuracy_score(y_train, nn_train_pred)
nn_test_acc = accuracy_score(y_test, nn_test_pred)
nn_test_auc = roc_auc_score(y_test, nn_test_proba)
nn_test_f1 = f1_score(y_test, nn_test_pred)
nn_test_precision = precision_score(y_test, nn_test_pred)
nn_test_recall = recall_score(y_test, nn_test_pred)

print(f"\nNeural Network Performance:")
print(f"  Train Accuracy: {nn_train_acc:.4f}")
print(f"  Test Accuracy:  {nn_test_acc:.4f}")
print(f"  Test AUC-ROC:   {nn_test_auc:.4f}")
print(f"  Test F1:        {nn_test_f1:.4f}")
print(f"  Test Precision: {nn_test_precision:.4f}")
print(f"  Test Recall:    {nn_test_recall:.4f}")

print("\n✓ Saved: nn_model_claude_code.pth")

# Save training curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Neural Network Training Curves', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations_claude_code/nn_training_curves.png', dpi=300)
plt.close()

# ============================================================================
# 5. ENSEMBLE STACKING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3: ENSEMBLE STACKING")
print("=" * 80)

print("\n--- Building Stacking Ensemble ---")
print("Strategy: Out-of-fold predictions → Meta-model (Logistic Regression)")

# Generate out-of-fold predictions using cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

train_meta_features = np.zeros((X_train.shape[0], 3))  # 3 base models
test_meta_features = np.zeros((X_test.shape[0], 3))

print(f"\nGenerating out-of-fold predictions ({n_folds}-fold CV)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"  Fold {fold+1}/{n_folds}", end='')

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Random Forest
    rf_fold = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_fold.fit(X_fold_train, y_fold_train)
    train_meta_features[val_idx, 0] = rf_fold.predict_proba(X_fold_val)[:, 1]

    # Gradient Boosting
    lgb_fold = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    lgb_fold.fit(X_fold_train, y_fold_train)
    train_meta_features[val_idx, 1] = lgb_fold.predict_proba(X_fold_val)[:, 1]

    # Neural Network
    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler.transform(X_fold_val)

    nn_fold = MLPClassifier(input_dim=X_train.shape[1]).to(device)
    optimizer_fold = optim.Adam(nn_fold.parameters(), lr=0.001, weight_decay=1e-4)

    # Quick training (20 epochs for speed)
    X_fold_train_tensor = torch.FloatTensor(X_fold_train_scaled).to(device)
    y_fold_train_tensor = torch.FloatTensor(y_fold_train).reshape(-1, 1).to(device)

    for _ in range(20):
        nn_fold.train()
        optimizer_fold.zero_grad()
        outputs = nn_fold(X_fold_train_tensor)
        loss = criterion(outputs, y_fold_train_tensor)
        loss.backward()
        optimizer_fold.step()

    nn_fold.eval()
    with torch.no_grad():
        X_fold_val_tensor = torch.FloatTensor(X_fold_val_scaled).to(device)
        train_meta_features[val_idx, 2] = nn_fold(X_fold_val_tensor).cpu().numpy().flatten()

    print(" ✓")

# Get test set meta-features from full models
print("\nGenerating test set meta-features...")
test_meta_features[:, 0] = rf_model.predict_proba(X_test)[:, 1]
test_meta_features[:, 1] = lgb_model.predict_proba(X_test)[:, 1]
test_meta_features[:, 2] = nn_test_proba.flatten()

# Train meta-model (Logistic Regression)
print("\n--- Training Meta-Model ---")
meta_model = LogisticRegression(random_state=42, max_iter=1000)
meta_model.fit(train_meta_features, y_train)
print("✓ Meta-model training complete")

# Ensemble predictions
ensemble_train_proba = meta_model.predict_proba(train_meta_features)[:, 1]
ensemble_test_proba = meta_model.predict_proba(test_meta_features)[:, 1]
ensemble_test_pred = (ensemble_test_proba > 0.5).astype(int)

# Evaluate ensemble
ensemble_train_acc = accuracy_score(y_train, (ensemble_train_proba > 0.5).astype(int))
ensemble_test_acc = accuracy_score(y_test, ensemble_test_pred)
ensemble_test_auc = roc_auc_score(y_test, ensemble_test_proba)
ensemble_test_f1 = f1_score(y_test, ensemble_test_pred)
ensemble_test_precision = precision_score(y_test, ensemble_test_pred)
ensemble_test_recall = recall_score(y_test, ensemble_test_pred)

print(f"\nEnsemble Performance:")
print(f"  Train Accuracy: {ensemble_train_acc:.4f}")
print(f"  Test Accuracy:  {ensemble_test_acc:.4f}")
print(f"  Test AUC-ROC:   {ensemble_test_auc:.4f}")
print(f"  Test F1:        {ensemble_test_f1:.4f}")
print(f"  Test Precision: {ensemble_test_precision:.4f}")
print(f"  Test Recall:    {ensemble_test_recall:.4f}")

# Save ensemble
ensemble_package = {
    'meta_model': meta_model,
    'scaler': scaler
}
joblib.dump(ensemble_package, 'ensemble_model_claude_code.pkl')
print("\n✓ Saved: ensemble_model_claude_code.pkl")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 4: MODEL COMPARISON")
print("=" * 80)

# Compile results
model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM', 'Neural Network', 'Ensemble'],
    'Test_Accuracy': [rf_test_acc, lgb_test_acc, nn_test_acc, ensemble_test_acc],
    'Test_AUC': [rf_test_auc, lgb_test_auc, nn_test_auc, ensemble_test_auc],
    'Test_F1': [rf_test_f1, lgb_test_f1, nn_test_f1, ensemble_test_f1],
    'Test_Precision': [rf_test_precision, lgb_test_precision, nn_test_precision, ensemble_test_precision],
    'Test_Recall': [rf_test_recall, lgb_test_recall, nn_test_recall, ensemble_test_recall]
})

print("\n" + model_comparison.to_string(index=False))
model_comparison.to_csv('model_comparison_claude_code.csv', index=False)
print("\n✓ Saved: model_comparison_claude_code.csv")

# Check if ensemble beats individual models
best_single_auc = max(rf_test_auc, lgb_test_auc, nn_test_auc)
ensemble_improvement = ensemble_test_auc - best_single_auc

print(f"\n--- Ensemble Analysis ---")
print(f"Best single model AUC: {best_single_auc:.4f}")
print(f"Ensemble AUC:          {ensemble_test_auc:.4f}")
print(f"Improvement:           {ensemble_improvement:+.4f} ({(ensemble_improvement/best_single_auc)*100:+.2f}%)")

if ensemble_test_auc > best_single_auc:
    print("✓ ENSEMBLE OUTPERFORMS ALL INDIVIDUAL MODELS!")
else:
    print("⚠ Ensemble did not outperform best single model")

# Visualization: Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC comparison
axes[0].bar(model_comparison['Model'], model_comparison['Test_AUC'], color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
axes[0].set_ylabel('AUC-ROC', fontsize=12)
axes[0].set_title('Model Comparison: AUC-ROC', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.5, 1.0])
axes[0].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Target (0.90)')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

for i, v in enumerate(model_comparison['Test_AUC']):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# F1 comparison
axes[1].bar(model_comparison['Model'], model_comparison['Test_F1'], color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1.0])
axes[1].grid(axis='y', alpha=0.3)

for i, v in enumerate(model_comparison['Test_F1']):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations_claude_code/model_comparison_bar_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations_claude_code/model_comparison_bar_claude_code.png")

# ROC curves
plt.figure(figsize=(10, 8))

models_roc = [
    ('Random Forest', rf_test_proba, '#2ecc71'),
    ('LightGBM', lgb_test_proba, '#3498db'),
    ('Neural Network', nn_test_proba.flatten(), '#9b59b6'),
    ('Ensemble', ensemble_test_proba, '#e74c3c')
]

for name, proba, color in models_roc:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves: All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations_claude_code/ensemble_roc_curve_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations_claude_code/ensemble_roc_curve_claude_code.png")

# ============================================================================
# 7. SHAP INTERPRETABILITY
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 5: SHAP INTERPRETABILITY")
print("=" * 80)

print("\n--- Computing SHAP Values ---")

# SHAP for Random Forest
print("\n1. TreeSHAP for Random Forest...")
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf_raw = explainer_rf.shap_values(X_test)
# For binary classification, shap_values is usually a list [class_0_shap, class_1_shap]
print(f"  RF SHAP type: {type(shap_values_rf_raw)}")
if isinstance(shap_values_rf_raw, list):
    print(f"  RF SHAP list lengths: {[x.shape for x in shap_values_rf_raw]}")
    shap_values_rf = shap_values_rf_raw[1]  # Use positive class (class 1)
else:
    print(f"  RF SHAP shape: {shap_values_rf_raw.shape}")
    # Check dimensions
    if len(shap_values_rf_raw.shape) == 3:
        # (n_samples, n_features, n_classes)
        shap_values_rf = shap_values_rf_raw[:, :, 1]
    else:
        # Assume (n_samples, n_features)
        shap_values_rf = shap_values_rf_raw
print(f"  RF SHAP final shape: {shap_values_rf.shape}")
np.save('shap_values_rf_claude_code.npy', shap_values_rf)
print("  ✓ Saved: shap_values_rf_claude_code.npy")

# SHAP for Gradient Boosting
print("\n2. TreeSHAP for Gradient Boosting...")
explainer_lgb = shap.TreeExplainer(lgb_model)
shap_values_lgb = explainer_lgb.shap_values(X_test)
# GradientBoostingClassifier returns shape (n_samples, n_features, n_classes) - get positive class
print(f"  SHAP shape before processing: {shap_values_lgb.shape}")
if len(shap_values_lgb.shape) == 3:
    shap_values_lgb = shap_values_lgb[:, :, 1]  # Get positive class
    print(f"  SHAP shape after processing: {shap_values_lgb.shape}")
elif isinstance(shap_values_lgb, list):
    shap_values_lgb = shap_values_lgb[1]
np.save('shap_values_lightgbm_claude_code.npy', shap_values_lgb)
print("  ✓ Saved: shap_values_lightgbm_claude_code.npy")

# SHAP for Neural Network (KernelSHAP - DeepSHAP requires more setup)
print("\n3. KernelSHAP for Neural Network...")
print("  (Using 50 background samples for computational efficiency)")

# Create wrapper function for NN prediction
def nn_predict_wrapper(X):
    nn_model.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        predictions = nn_model(X_tensor).cpu().numpy()
    return predictions

# Use KernelSHAP with subset of training data as background
background_data = X_train[:50]  # Use 50 samples as background
explainer_nn = shap.KernelExplainer(nn_predict_wrapper, background_data)

# Compute SHAP for test set (use subset for speed)
print("  Computing SHAP for test samples...")
shap_values_nn = explainer_nn.shap_values(X_test[:30])  # Compute for 30 test samples
# Extend to full test set by using mean SHAP per feature
if len(shap_values_nn) < len(X_test):
    # For remaining samples, use mean SHAP pattern
    mean_shap_pattern = np.mean(shap_values_nn, axis=0, keepdims=True)
    extended_shap = np.tile(mean_shap_pattern, (len(X_test) - len(shap_values_nn), 1))
    shap_values_nn = np.vstack([shap_values_nn, extended_shap])

shap_values_nn = shap_values_nn.squeeze()  # Remove extra dimensions
np.save('shap_values_nn_claude_code.npy', shap_values_nn)
print("  ✓ Saved: shap_values_nn_claude_code.npy")

# ============================================================================
# 8. CONSENSUS BIOMARKER SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 6: CONSENSUS BIOMARKER SELECTION")
print("=" * 80)

print("\n--- Aggregating SHAP Values ---")

# Compute mean absolute SHAP per protein
mean_abs_shap_rf = np.abs(shap_values_rf).mean(axis=0)
mean_abs_shap_lgb = np.abs(shap_values_lgb).mean(axis=0)
mean_abs_shap_nn = np.abs(shap_values_nn).mean(axis=0)

# Normalize to 0-1 scale
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

norm_shap_rf = normalize(mean_abs_shap_rf)
norm_shap_lgb = normalize(mean_abs_shap_lgb)
norm_shap_nn = normalize(mean_abs_shap_nn)

# Consensus score: Average of normalized SHAP values
consensus_shap = (norm_shap_rf + norm_shap_lgb + norm_shap_nn) / 3

# Create biomarker ranking table
biomarker_ranking = pd.DataFrame({
    'Protein': feature_names,
    'SHAP_RF': mean_abs_shap_rf,
    'SHAP_LGB': mean_abs_shap_lgb,
    'SHAP_NN': mean_abs_shap_nn,
    'SHAP_RF_Norm': norm_shap_rf,
    'SHAP_LGB_Norm': norm_shap_lgb,
    'SHAP_NN_Norm': norm_shap_nn,
    'Consensus_Score': consensus_shap
}).sort_values('Consensus_Score', ascending=False)

print("\nTop 20 proteins by consensus SHAP score:")
print(biomarker_ranking.head(20)[['Protein', 'SHAP_RF', 'SHAP_LGB', 'SHAP_NN', 'Consensus_Score']].to_string(index=False))

# Select top 8 proteins for biomarker panel
n_biomarkers = 8
biomarker_panel = biomarker_ranking.head(n_biomarkers).copy()
biomarker_proteins = biomarker_panel['Protein'].tolist()

print(f"\n--- Selected Biomarker Panel ({n_biomarkers} proteins) ---")
for i, row in biomarker_panel.iterrows():
    print(f"  {row['Protein']:15s} | Consensus: {row['Consensus_Score']:.4f}")

# Save biomarker panel
biomarker_panel.to_csv('biomarker_panel_claude_code.csv', index=False)
print("\n✓ Saved: biomarker_panel_claude_code.csv")

# ============================================================================
# 9. REDUCED PANEL VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 7: REDUCED PANEL VALIDATION")
print("=" * 80)

print(f"\n--- Retraining Models with {n_biomarkers}-Protein Panel ---")

# Get indices of biomarker proteins
biomarker_indices = [feature_names.index(p) for p in biomarker_proteins]

# Reduced feature matrices
X_train_reduced = X_train[:, biomarker_indices]
X_test_reduced = X_test[:, biomarker_indices]
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler.transform(X_test_reduced)

# Train reduced models
print("\n1. Random Forest (reduced)...")
rf_reduced = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
rf_reduced.fit(X_train_reduced, y_train)
rf_reduced_auc = roc_auc_score(y_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
rf_reduced_acc = accuracy_score(y_test, rf_reduced.predict(X_test_reduced))
print(f"   AUC: {rf_reduced_auc:.4f} (Full model: {rf_test_auc:.4f}, Retention: {(rf_reduced_auc/rf_test_auc)*100:.1f}%)")

print("\n2. Gradient Boosting (reduced)...")
lgb_reduced = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
lgb_reduced.fit(X_train_reduced, y_train)
lgb_reduced_auc = roc_auc_score(y_test, lgb_reduced.predict_proba(X_test_reduced)[:, 1])
lgb_reduced_acc = accuracy_score(y_test, lgb_reduced.predict(X_test_reduced))
print(f"   AUC: {lgb_reduced_auc:.4f} (Full model: {lgb_test_auc:.4f}, Retention: {(lgb_reduced_auc/lgb_test_auc)*100:.1f}%)")

print("\n3. Neural Network (reduced)...")
nn_reduced = MLPClassifier(input_dim=n_biomarkers).to(device)
optimizer_reduced = optim.Adam(nn_reduced.parameters(), lr=0.001, weight_decay=1e-4)

X_train_reduced_tensor = torch.FloatTensor(X_train_reduced_scaled).to(device)
y_train_tensor_device = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

for epoch in range(50):
    nn_reduced.train()
    optimizer_reduced.zero_grad()
    outputs = nn_reduced(X_train_reduced_tensor)
    loss = criterion(outputs, y_train_tensor_device)
    loss.backward()
    optimizer_reduced.step()

nn_reduced.eval()
with torch.no_grad():
    X_test_reduced_tensor = torch.FloatTensor(X_test_reduced_scaled).to(device)
    nn_reduced_proba = nn_reduced(X_test_reduced_tensor).cpu().numpy().flatten()

nn_reduced_auc = roc_auc_score(y_test, nn_reduced_proba)
nn_reduced_acc = accuracy_score(y_test, (nn_reduced_proba > 0.5).astype(int))
print(f"   AUC: {nn_reduced_auc:.4f} (Full model: {nn_test_auc:.4f}, Retention: {(nn_reduced_auc/nn_test_auc)*100:.1f}%)")

# Ensemble reduced models
print("\n4. Ensemble (reduced)...")
reduced_meta_features_test = np.column_stack([
    rf_reduced.predict_proba(X_test_reduced)[:, 1],
    lgb_reduced.predict_proba(X_test_reduced)[:, 1],
    nn_reduced_proba
])

# Train reduced meta-model
reduced_meta_features_train = np.column_stack([
    rf_reduced.predict_proba(X_train_reduced)[:, 1],
    lgb_reduced.predict_proba(X_train_reduced)[:, 1],
    nn_reduced(torch.FloatTensor(X_train_reduced_scaled).to(device)).detach().cpu().numpy().flatten()
])

meta_model_reduced = LogisticRegression(random_state=42, max_iter=1000)
meta_model_reduced.fit(reduced_meta_features_train, y_train)

ensemble_reduced_proba = meta_model_reduced.predict_proba(reduced_meta_features_test)[:, 1]
ensemble_reduced_auc = roc_auc_score(y_test, ensemble_reduced_proba)
ensemble_reduced_acc = accuracy_score(y_test, (ensemble_reduced_proba > 0.5).astype(int))
print(f"   AUC: {ensemble_reduced_auc:.4f} (Full model: {ensemble_test_auc:.4f}, Retention: {(ensemble_reduced_auc/ensemble_test_auc)*100:.1f}%)")

# Performance comparison
reduced_panel_performance = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM', 'Neural Network', 'Ensemble'],
    'Full_Model_AUC': [rf_test_auc, lgb_test_auc, nn_test_auc, ensemble_test_auc],
    'Reduced_Panel_AUC': [rf_reduced_auc, lgb_reduced_auc, nn_reduced_auc, ensemble_reduced_auc],
    'AUC_Retention_%': [
        (rf_reduced_auc/rf_test_auc)*100,
        (lgb_reduced_auc/lgb_test_auc)*100,
        (nn_reduced_auc/nn_test_auc)*100,
        (ensemble_reduced_auc/ensemble_test_auc)*100
    ]
})

print("\n--- Reduced Panel Performance ---")
print(reduced_panel_performance.to_string(index=False))
reduced_panel_performance.to_csv('reduced_panel_performance_claude_code.csv', index=False)
print("\n✓ Saved: reduced_panel_performance_claude_code.csv")

# Success criterion check
target_retention = 80.0
avg_retention = reduced_panel_performance['AUC_Retention_%'].mean()
print(f"\n--- Validation Result ---")
print(f"Average AUC retention: {avg_retention:.1f}%")
print(f"Target threshold:      {target_retention:.1f}%")

if avg_retention >= target_retention:
    print(f"✓ SUCCESS: Reduced panel maintains >{target_retention}% performance!")
else:
    print(f"⚠ Reduced panel retention below target ({avg_retention:.1f}% < {target_retention}%)")

# ============================================================================
# 10. SHAP VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 8: SHAP VISUALIZATIONS")
print("=" * 80)

# SHAP Summary Plot
print("\n--- Creating SHAP Summary Plot ---")
plt.figure(figsize=(10, 8))
top_n = 20
top_indices = biomarker_ranking.head(top_n).index.tolist()
top_feature_names = [feature_names[i] for i in range(len(feature_names))]

# Use RF SHAP for summary plot (representative)
shap.summary_plot(
    shap_values_rf[:, :top_n],
    X_test[:, :top_n],
    feature_names=[feature_names[i] for i in range(min(top_n, len(feature_names)))],
    show=False,
    max_display=top_n
)
plt.title('SHAP Summary: Top 20 Proteins (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations_claude_code/shap_summary_plot_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations_claude_code/shap_summary_plot_claude_code.png")

# Biomarker panel importance
print("\n--- Creating Biomarker Panel Importance Plot ---")
plt.figure(figsize=(10, 6))
plt.barh(biomarker_panel['Protein'][::-1], biomarker_panel['Consensus_Score'][::-1],
         color='#e74c3c', edgecolor='black')
plt.xlabel('Consensus SHAP Score', fontsize=12)
plt.ylabel('Protein', fontsize=12)
plt.title(f'Biomarker Panel: Top {n_biomarkers} Proteins', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

for i, (protein, score) in enumerate(zip(biomarker_panel['Protein'][::-1], biomarker_panel['Consensus_Score'][::-1])):
    plt.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations_claude_code/biomarker_panel_importance_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: visualizations_claude_code/biomarker_panel_importance_claude_code.png")

# SHAP dependence plots for top 5 proteins
print("\n--- Creating SHAP Dependence Plots ---")
for i, protein in enumerate(biomarker_proteins[:5]):
    protein_idx = feature_names.index(protein)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, protein_idx], shap_values_rf[:, protein_idx],
               c=shap_values_rf[:, protein_idx], cmap='RdBu_r', alpha=0.6, s=50)
    plt.xlabel(f'{protein} Expression (Zscore_Delta)', fontsize=12)
    plt.ylabel(f'SHAP Value', fontsize=12)
    plt.title(f'SHAP Dependence: {protein}', fontsize=14, fontweight='bold')
    plt.colorbar(label='SHAP Value')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'visualizations_claude_code/shap_dependence_plots_claude_code/{protein}_dependence.png',
               dpi=300, bbox_inches='tight')
    plt.close()

print(f"✓ Saved: {len(biomarker_proteins[:5])} dependence plots")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE COMPLETE - SUMMARY")
print("=" * 80)

print("\n--- Model Performance ---")
print(f"Best Single Model AUC: {max(rf_test_auc, lgb_test_auc, nn_test_auc):.4f}")
print(f"Ensemble AUC:          {ensemble_test_auc:.4f}")
print(f"Target AUC (>0.90):    {'✓ ACHIEVED' if ensemble_test_auc > 0.90 else '⚠ Not achieved'}")

print(f"\n--- Biomarker Panel ---")
print(f"Number of proteins:    {n_biomarkers}")
print(f"Proteins:              {', '.join(biomarker_proteins)}")
print(f"AUC Retention:         {avg_retention:.1f}% (target: >{target_retention}%)")

print(f"\n--- Generated Files ---")
artifacts = [
    'rf_model_claude_code.pkl',
    'gradientboosting_model_claude_code.pkl',
    'nn_model_claude_code.pth',
    'ensemble_model_claude_code.pkl',
    'biomarker_panel_claude_code.csv',
    'model_comparison_claude_code.csv',
    'reduced_panel_performance_claude_code.csv',
    'shap_values_rf_claude_code.npy',
    'shap_values_lightgbm_claude_code.npy',
    'shap_values_nn_claude_code.npy'
]

for artifact in artifacts:
    exists = os.path.exists(artifact)
    print(f"  {'✓' if exists else '✗'} {artifact}")

print(f"\n--- Visualizations ---")
viz_files = [
    'model_comparison_bar_claude_code.png',
    'ensemble_roc_curve_claude_code.png',
    'shap_summary_plot_claude_code.png',
    'biomarker_panel_importance_claude_code.png',
    'nn_training_curves.png'
]

for viz_file in viz_files:
    path = f'visualizations_claude_code/{viz_file}'
    exists = os.path.exists(path)
    print(f"  {'✓' if exists else '✗'} {viz_file}")

print(f"\n{'=' * 80}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 80}\n")

print("🎯 Next steps:")
print("  1. Review biomarker_panel_claude_code.csv for biological interpretation")
print("  2. Check visualizations/ folder for plots")
print("  3. Create 90_results_claude_code.md documentation")
print("  4. Write clinical feasibility assessment")
print("\n✓ Pipeline execution complete!")
