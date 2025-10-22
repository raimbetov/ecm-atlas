#!/usr/bin/env python3
"""
Comprehensive Training & Evaluation Pipeline for Multimodal Aging Predictor

Includes:
1. Baseline models (Ridge, Random Forest)
2. Ablation studies (AE-only, AE+LSTM, Full model)
3. Training with multi-task loss
4. Evaluation metrics (R², MAE, RMSE)
5. Visualizations
6. Interpretability (attention weights, feature importance)
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our architectures
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import importlib.util
spec = importlib.util.spec_from_file_location("arch", "02_multimodal_architecture_claude_code.py")
arch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arch)

MultiModalAgingPredictor = arch.MultiModalAgingPredictor
BaselineRidge = arch.BaselineRidge
AutoencoderOnly = arch.AutoencoderOnly
AutoencoderLSTM = arch.AutoencoderLSTM

print("=" * 80)
print("MULTIMODAL AGING PREDICTOR - COMPREHENSIVE TRAINING & EVALUATION")
print("=" * 80)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Load preprocessed data
print("\n1. Loading preprocessed data...")
data_dir = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_18_multimodal_integration/claude_code'

X_train = np.load(f'{data_dir}/X_train_claude_code.npy')
X_val = np.load(f'{data_dir}/X_val_claude_code.npy')
X_test = np.load(f'{data_dir}/X_test_claude_code.npy')
y_train = np.load(f'{data_dir}/y_train_claude_code.npy')
y_val = np.load(f'{data_dir}/y_val_claude_code.npy')
y_test = np.load(f'{data_dir}/y_test_claude_code.npy')

with open(f'{data_dir}/data_metadata_claude_code.pkl', 'rb') as f:
    metadata = pickle.load(f)

s100_indices = metadata['s100_indices']
protein_names = metadata['protein_names']

print(f"   Train: {X_train.shape}, y range [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"   Val:   {X_val.shape}, y range [{y_val.min():.2f}, {y_val.max():.2f}]")
print(f"   Test:  {X_test.shape}, y range [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"   S100 pathway features: {len(s100_indices)}")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
X_val_t = torch.FloatTensor(X_val)
X_test_t = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# Extract S100 features
X_train_s100 = X_train_t[:, s100_indices]
X_val_s100 = X_val_t[:, s100_indices]
X_test_s100 = X_test_t[:, s100_indices]

# Dummy edge index (protein correlation network - simplified)
print("\n2. Creating protein interaction graph...")
# For small dataset, create a simple correlation-based graph
corr_matrix = np.corrcoef(X_train.T)  # (n_proteins, n_proteins)
threshold = 0.7  # High correlation threshold
edge_pairs = np.argwhere(np.abs(corr_matrix) > threshold)
# Remove self-loops
edge_pairs = edge_pairs[edge_pairs[:, 0] != edge_pairs[:, 1]]
edge_index = torch.LongTensor(edge_pairs[:5000].T)  # Limit edges for memory

print(f"   Created graph with {edge_index.shape[1]} edges (correlation > {threshold})")

########################################
# BASELINE MODELS                      #
########################################

print("\n" + "=" * 80)
print("PHASE 1: BASELINE MODELS")
print("=" * 80)

results = {}

# Baseline 1: Ridge Regression
print("\n1.1. Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)

r2_ridge = r2_score(y_val, y_pred_ridge)
mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))

results['Ridge'] = {
    'val_r2': r2_ridge,
    'val_mae': mae_ridge,
    'val_rmse': rmse_ridge,
    'model': ridge
}

print(f"   Val R²:   {r2_ridge:.4f}")
print(f"   Val MAE:  {mae_ridge:.4f}")
print(f"   Val RMSE: {rmse_ridge:.4f}")

# Baseline 2: Random Forest
print("\n1.2. Random Forest...")
rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

r2_rf = r2_score(y_val, y_pred_rf)
mae_rf = mean_absolute_error(y_val, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))

results['Random Forest'] = {
    'val_r2': r2_rf,
    'val_mae': mae_rf,
    'val_rmse': rmse_rf,
    'model': rf
}

print(f"   Val R²:   {r2_rf:.4f}")
print(f"   Val MAE:  {mae_rf:.4f}")
print(f"   Val RMSE: {rmse_rf:.4f}")

########################################
# DEEP LEARNING TRAINING FUNCTION      #
########################################

def train_deep_model(model, model_name, num_epochs=500, lr=0.001, weight_decay=1e-3,
                      use_reconstruction_loss=False, use_s100=False):
    """Generic training function for all deep models"""

    print(f"\nTraining {model_name}...")
    print(f"   Epochs: {num_epochs}, LR: {lr}, Weight Decay: {weight_decay}")
    print(f"   Reconstruction loss: {use_reconstruction_loss}, S100 features: {use_s100}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_age = nn.MSELoss()
    criterion_recon = nn.MSELoss()

    best_val_r2 = -999
    best_epoch = 0
    patience = 100
    patience_counter = 0

    train_losses = []
    val_r2_scores = []

    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        if 'MultiModal' in model_name:
            age_pred, stiffness, attn_weights, reconstructed = model(
                X_train_t, edge_index, X_train_s100
            )
            loss_age = criterion_age(age_pred, y_train_t)
            loss_recon = criterion_recon(reconstructed, X_train_t)
            loss = loss_age + 0.1 * loss_recon  # Multi-task loss

        elif 'AutoencoderLSTM' in model_name:
            age_pred, reconstructed = model(X_train_t)
            loss_age = criterion_age(age_pred, y_train_t)
            loss_recon = criterion_recon(reconstructed, X_train_t)
            loss = loss_age + 0.1 * loss_recon

        elif 'AutoencoderOnly' in model_name:
            age_pred, reconstructed = model(X_train_t)
            loss_age = criterion_age(age_pred, y_train_t)
            loss_recon = criterion_recon(reconstructed, X_train_t)
            loss = loss_age + 0.1 * loss_recon

        elif 'Baseline' in model_name:
            age_pred = model(X_train_t)
            loss = criterion_age(age_pred, y_train_t)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if 'MultiModal' in model_name:
                val_pred, _, _, _ = model(X_val_t, edge_index, X_val_s100)
            elif 'AutoencoderLSTM' in model_name or 'AutoencoderOnly' in model_name:
                val_pred, _ = model(X_val_t)
            else:
                val_pred = model(X_val_t)

            val_r2 = r2_score(y_val, val_pred.numpy())

        train_losses.append(loss.item())
        val_r2_scores.append(val_r2)

        # Early stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f'{data_dir}/best_{model_name.replace(" ", "_")}_claude_code.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch} (best: {best_epoch}, R²={best_val_r2:.4f})")
            break

        if epoch % 100 == 0:
            print(f"   Epoch {epoch:3d}: Loss={loss.item():.4f}, Val R²={val_r2:.4f}")

    # Load best model
    model.load_state_dict(torch.load(f'{data_dir}/best_{model_name.replace(" ", "_")}_claude_code.pth'))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        if 'MultiModal' in model_name:
            val_pred, _, _, _ = model(X_val_t, edge_index, X_val_s100)
            test_pred, _, _, _ = model(X_test_t, edge_index, X_test_s100)
        elif 'AutoencoderLSTM' in model_name or 'AutoencoderOnly' in model_name:
            val_pred, _ = model(X_val_t)
            test_pred, _ = model(X_test_t)
        else:
            val_pred = model(X_val_t)
            test_pred = model(X_test_t)

    val_r2 = r2_score(y_val, val_pred.numpy())
    val_mae = mean_absolute_error(y_val, val_pred.numpy())
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred.numpy()))

    test_r2 = r2_score(y_test, test_pred.numpy())
    test_mae = mean_absolute_error(y_test, test_pred.numpy())
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred.numpy()))

    print(f"   ✓ Best Val R²:  {val_r2:.4f} (epoch {best_epoch})")
    print(f"   ✓ Test R²:      {test_r2:.4f}")
    print(f"   ✓ Test MAE:     {test_mae:.4f}")

    return {
        'val_r2': val_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_r2_scores': val_r2_scores,
        'model': model
    }

########################################
# ABLATION STUDIES                     #
########################################

print("\n" + "=" * 80)
print("PHASE 2: ABLATION STUDIES")
print("=" * 80)

# Ablation 1: Baseline Neural Network
print("\n2.1. Baseline Neural Network (Linear)")
baseline_nn = BaselineRidge(n_proteins=X_train.shape[1])
results['Baseline NN'] = train_deep_model(
    baseline_nn, 'Baseline NN', num_epochs=500, lr=0.01, weight_decay=1e-3
)

# Ablation 2: Autoencoder Only
print("\n2.2. Autoencoder Only")
ae_only = AutoencoderOnly(n_proteins=X_train.shape[1], latent_dim=32, dropout=0.5)
results['AE Only'] = train_deep_model(
    ae_only, 'AutoencoderOnly', num_epochs=500, lr=0.001, weight_decay=1e-3,
    use_reconstruction_loss=True
)

# Ablation 3: Autoencoder + LSTM
print("\n2.3. Autoencoder + LSTM")
ae_lstm = AutoencoderLSTM(n_proteins=X_train.shape[1], latent_dim=32, dropout=0.5)
results['AE + LSTM'] = train_deep_model(
    ae_lstm, 'AutoencoderLSTM', num_epochs=500, lr=0.001, weight_decay=1e-3,
    use_reconstruction_loss=True
)

# Full Model: AE + GNN + LSTM + S100
print("\n2.4. FULL MODEL (AE + GNN + LSTM + S100)")
full_model = MultiModalAgingPredictor(
    n_proteins=X_train.shape[1],
    latent_dim=32,
    s100_dim=len(s100_indices),
    dropout=0.5
)

# Try to load H04 autoencoder weights
h04_path = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_02/hypothesis_04_deep_protein_embeddings/claude_code/autoencoder_weights_claude_code.pth'
print(f"\n   Attempting to load H04 autoencoder from: {h04_path}")
full_model.load_pretrained_autoencoder(h04_path)

results['Full Model'] = train_deep_model(
    full_model, 'MultiModal Full', num_epochs=500, lr=0.001, weight_decay=1e-3,
    use_reconstruction_loss=True, use_s100=True
)

########################################
# RESULTS SUMMARY                      #
########################################

print("\n" + "=" * 80)
print("PHASE 3: RESULTS SUMMARY")
print("=" * 80)

# Create summary table
summary_data = []
for model_name, res in results.items():
    summary_data.append({
        'Model': model_name,
        'Val R²': res['val_r2'],
        'Val MAE': res['val_mae'],
        'Val RMSE': res['val_rmse'],
        'Test R²': res.get('test_r2', np.nan),
        'Test MAE': res.get('test_mae', np.nan),
        'Test RMSE': res.get('test_rmse', np.nan)
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Val R²', ascending=False)

print("\n" + summary_df.to_string(index=False))

# Save results
summary_df.to_csv(f'{data_dir}/model_performance_claude_code.csv', index=False)
print(f"\n✓ Saved performance summary to model_performance_claude_code.csv")

# Save detailed results
with open(f'{data_dir}/all_results_claude_code.pkl', 'wb') as f:
    pickle.dump(results, f)

########################################
# VISUALIZATIONS                       #
########################################

print("\n" + "=" * 80)
print("PHASE 4: VISUALIZATIONS")
print("=" * 80)

viz_dir = f'{data_dir}/visualizations_claude_code'

# Ablation bar chart
fig, ax = plt.subplots(figsize=(10, 6))
models = summary_df['Model'].tolist()
r2_scores = summary_df['Val R²'].tolist()

colors = ['#e74c3c' if 'Ridge' in m or 'Forest' in m else '#3498db' if 'Baseline' in m else '#2ecc71' for m in models]

ax.barh(models, r2_scores, color=colors)
ax.axvline(0.85, color='red', linestyle='--', linewidth=2, label='Target R²=0.85')
ax.set_xlabel('Validation R² Score', fontsize=12)
ax.set_title('Ablation Study: Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{viz_dir}/ablation_study_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: ablation_study_claude_code.png")
plt.close()

# Training curves (for deep models)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (model_name, res) in enumerate([(k, v) for k, v in results.items() if 'train_losses' in v][:4]):
    row = idx // 2
    col = idx % 2

    ax = axes[row, col]

    epochs = range(len(res['train_losses']))
    ax.plot(epochs, res['train_losses'], label='Train Loss', alpha=0.7)

    ax2 = ax.twinx()
    ax2.plot(epochs, res['val_r2_scores'], color='green', label='Val R²', alpha=0.7)
    ax2.axhline(0.85, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='blue')
    ax2.set_ylabel('R² Score', color='green')
    ax.set_title(model_name, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{viz_dir}/training_curves_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: training_curves_claude_code.png")
plt.close()

# Predicted vs Actual (best model)
best_model_name = summary_df.iloc[0]['Model']
best_model_res = results[best_model_name]

if 'model' in best_model_res:
    print(f"\nGenerating predictions for best model: {best_model_name}")

    model = best_model_res['model']

    if isinstance(model, (Ridge, RandomForestRegressor)):
        y_pred_test = model.predict(X_test)
    else:
        model.eval()
        with torch.no_grad():
            if 'MultiModal' in best_model_name:
                y_pred_test, _, _, _ = model(X_test_t, edge_index, X_test_s100)
            elif 'Autoencoder' in best_model_name:
                y_pred_test, _ = model(X_test_t)
            else:
                y_pred_test = model(X_test_t)

            y_pred_test = y_pred_test.numpy().flatten()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred_test, alpha=0.6, s=100, edgecolors='black')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Age (or Age Group)', fontsize=12)
    ax.set_ylabel('Predicted Age', fontsize=12)
    ax.set_title(f'Predicted vs Actual ({best_model_name})\nTest R²={best_model_res.get("test_r2", 0):.3f}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{viz_dir}/predicted_vs_actual_claude_code.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: predicted_vs_actual_claude_code.png")
    plt.close()

print("\n" + "=" * 80)
print("TRAINING & EVALUATION COMPLETE")
print("=" * 80)
print(f"\nBest Model: {best_model_name}")
print(f"Val R²:     {summary_df.iloc[0]['Val R²']:.4f}")
print(f"Test R²:    {summary_df.iloc[0]['Test R²']:.4f}")
print(f"Test MAE:   {summary_df.iloc[0]['Test MAE']:.4f}")
print("=" * 80)
