#!/usr/bin/env python3
"""
H11 - LSTM Benchmarking for All Pseudo-Time Methods
Author: claude_code
Date: 2025-10-21

This script trains LSTM Seq2Seq models on temporal sequences created by each
pseudo-time method and compares their predictive performance.

GOAL: Find which pseudo-time method produces most predictable aging trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LSTM BENCHMARKING: Testing All Pseudo-Time Methods")
print("="*80)

# ============================================================================
# 1. LOAD DATA & PSEUDOTIME ORDERINGS
# ============================================================================

print("\n[1] Loading data and pseudotime orderings...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Load pseudotime orderings
pseudotime_df = pd.read_csv('pseudotime_orderings_claude_code.csv', index_col=0)
methods = pseudotime_df.columns.tolist()

print(f"   Pseudo-time methods to test: {methods}")

# Create tissue × protein matrix
tissue_protein_matrix = df.pivot_table(
    values='Zscore_Delta',
    index='Tissue',
    columns='Gene_Symbol',
    aggfunc='median'
).fillna(0)

tissues_list = tissue_protein_matrix.index.tolist()
proteins_list = tissue_protein_matrix.columns.tolist()
X_full = tissue_protein_matrix.values  # (17, 910)

print(f"   Data matrix shape: {X_full.shape}")
print(f"   Tissues: {len(tissues_list)}, Proteins: {len(proteins_list)}")

# ============================================================================
# 2. LSTM SEQ2SEQ MODEL ARCHITECTURE
# ============================================================================

class LSTMSeq2Seq(nn.Module):
    """Encoder-Decoder LSTM for sequence forecasting"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, forecast_horizon=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               batch_first=True)

        # Output projection
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_input, target_len):
        """
        x_input: (batch, seq_len, input_dim) - input sequence
        target_len: number of future steps to predict
        """
        batch_size = x_input.size(0)

        # Encode input sequence
        _, (hidden, cell) = self.encoder(x_input)

        # Initialize decoder input with last input timestep
        decoder_input = x_input[:, -1:, :]  # (batch, 1, input_dim)

        outputs = []
        for t in range(target_len):
            # Decode one step
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # Project to output space
            prediction = self.fc(decoder_output)  # (batch, 1, input_dim)
            outputs.append(prediction)

            # Use prediction as next input (teacher forcing disabled for robustness)
            decoder_input = prediction

        # Stack predictions
        outputs = torch.cat(outputs, dim=1)  # (batch, target_len, input_dim)
        return outputs

# ============================================================================
# 3. CREATE TEMPORAL SEQUENCES FOR EACH METHOD
# ============================================================================

def create_sequences(protein_matrix, tissue_order, window_size=4, forecast_horizon=3):
    """
    Create training sequences from temporal ordering

    Args:
        protein_matrix: (n_tissues, n_proteins) array
        tissue_order: list of tissue indices in temporal order
        window_size: number of past steps to use as input
        forecast_horizon: number of future steps to predict

    Returns:
        X: (n_samples, window_size, n_proteins) input sequences
        Y: (n_samples, forecast_horizon, n_proteins) target sequences
    """
    n_tissues = len(tissue_order)
    n_proteins = protein_matrix.shape[1]

    # Order protein profiles by pseudotime
    ordered_profiles = protein_matrix[tissue_order, :]

    X, Y = [], []

    for i in range(n_tissues - window_size - forecast_horizon + 1):
        x_seq = ordered_profiles[i:i+window_size, :]
        y_seq = ordered_profiles[i+window_size:i+window_size+forecast_horizon, :]

        X.append(x_seq)
        Y.append(y_seq)

    return np.array(X), np.array(Y)

# ============================================================================
# 4. TRAIN & EVALUATE LSTM FOR EACH METHOD
# ============================================================================

def train_lstm(X_train, Y_train, X_val, Y_val, input_dim, epochs=100,
               hidden_dim=64, num_layers=2, dropout=0.2, lr=1e-3):
    """Train LSTM model and return best validation loss"""

    forecast_horizon = Y_train.shape[1]

    model = LSTMSeq2Seq(input_dim, hidden_dim, num_layers, dropout, forecast_horizon)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t, forecast_horizon)
        loss = criterion(predictions, Y_train_t)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t, forecast_horizon)
            val_loss = criterion(val_predictions, Y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # Early stopping
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_t, forecast_horizon)
        mse = criterion(val_predictions, Y_val_t).item()

        # Calculate R²
        y_true_flat = Y_val_t.reshape(-1).numpy()
        y_pred_flat = val_predictions.reshape(-1).numpy()
        ss_res = np.sum((y_true_flat - y_pred_flat)**2)
        ss_tot = np.sum((y_true_flat - y_true_flat.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return mse, r2, model

print("\n[4] Benchmarking LSTM for each pseudo-time method...")
print("-"*80)

window_size = 4
forecast_horizon = 3
n_splits = 3  # Use 3-fold CV due to small number of tissues

results = []

for method in methods:
    print(f"\n>>> Method: {method}")
    print("-"*40)

    # Get tissue ordering for this method
    tissue_ranks = pseudotime_df[method].to_dict()
    tissue_order = sorted(range(len(tissues_list)),
                         key=lambda i: tissue_ranks[tissues_list[i]])

    print(f"   Tissue ordering: {[tissues_list[i] for i in tissue_order[:5]]} ... {[tissues_list[i] for i in tissue_order[-2:]]}")

    # Create sequences
    X, Y = create_sequences(X_full, tissue_order, window_size, forecast_horizon)

    print(f"   Sequences created: X={X.shape}, Y={Y.shape}")

    if len(X) < n_splits:
        print(f"   ⚠ Not enough sequences ({len(X)}) for {n_splits}-fold CV. Using simple train/val split.")
        # Simple 70/30 split
        split_idx = int(0.7 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]

        mse, r2, model = train_lstm(X_train, Y_train, X_val, Y_val,
                                    input_dim=X.shape[2],
                                    epochs=100, hidden_dim=64,
                                    num_layers=2, dropout=0.2)

        print(f"   Results: MSE={mse:.4f}, R²={r2:.4f}")

        results.append({
            'Method': method,
            'MSE': mse,
            'R2': r2,
            'n_sequences': len(X)
        })

    else:
        # K-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_mses, fold_r2s = [], []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]

            mse, r2, model = train_lstm(X_train, Y_train, X_val, Y_val,
                                        input_dim=X.shape[2],
                                        epochs=100, hidden_dim=64,
                                        num_layers=2, dropout=0.2)

            fold_mses.append(mse)
            fold_r2s.append(r2)

            print(f"   Fold {fold_idx+1}/{n_splits}: MSE={mse:.4f}, R²={r2:.4f}")

        mean_mse = np.mean(fold_mses)
        mean_r2 = np.mean(fold_r2s)
        std_mse = np.std(fold_mses)
        std_r2 = np.std(fold_r2s)

        print(f"   >>> Mean: MSE={mean_mse:.4f}±{std_mse:.4f}, R²={mean_r2:.4f}±{std_r2:.4f}")

        results.append({
            'Method': method,
            'MSE': mean_mse,
            'R2': mean_r2,
            'MSE_std': std_mse,
            'R2_std': std_r2,
            'n_sequences': len(X)
        })

# ============================================================================
# 5. COMPARE RESULTS
# ============================================================================

print("\n" + "="*80)
print("[5] LSTM Performance Comparison")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('R2', ascending=False)

print("\nRanked by R² (descending):")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('lstm_performance_claude_code.csv', index=False)
print("\n✓ Saved to: lstm_performance_claude_code.csv")

# Key findings
print("\n" + "-"*80)
print("KEY FINDINGS:")
print("-"*80)

best_method = results_df.iloc[0]['Method']
best_r2 = results_df.iloc[0]['R2']
worst_method = results_df.iloc[-1]['Method']
worst_r2 = results_df.iloc[-1]['R2']

print(f"1. BEST METHOD: {best_method} (R² = {best_r2:.4f})")
print(f"2. WORST METHOD: {worst_method} (R² = {worst_r2:.4f})")
print(f"3. Performance difference: {best_r2/abs(worst_r2) if worst_r2 < 0 else best_r2 - worst_r2:.2f}× / {best_r2 - worst_r2:.4f} absolute")

# Check if this explains H09 disagreement
if 'Velocity (H03)' in results_df['Method'].values and 'PCA (Codex)' in results_df['Method'].values:
    velocity_r2 = results_df[results_df['Method'] == 'Velocity (H03)']['R2'].values[0]
    pca_r2 = results_df[results_df['Method'] == 'PCA (Codex)']['R2'].values[0]

    print(f"\n4. H09 DISAGREEMENT ANALYSIS:")
    print(f"   - Velocity (Claude): R² = {velocity_r2:.4f}")
    print(f"   - PCA (Codex): R² = {pca_r2:.4f}")
    print(f"   - Difference: {velocity_r2 - pca_r2:.4f}")

    if velocity_r2 > pca_r2:
        print(f"   ✓ CONFIRMED: Velocity method superior, explaining Claude's R²=0.81 vs Codex R²=0.011")
    else:
        print(f"   ✗ UNEXPECTED: PCA better in this test, but Codex reported R²=0.011. Investigate further.")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n[6] Generating visualizations...")

# (A) Bar chart: R² by method
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['green' if r2 > 0.6 else 'orange' if r2 > 0.3 else 'red'
          for r2 in results_df['R2']]

bars = ax.bar(results_df['Method'], results_df['R2'], color=colors, edgecolor='black', linewidth=1.5)

# Add error bars if std available
if 'R2_std' in results_df.columns:
    ax.errorbar(results_df['Method'], results_df['R2'],
                yerr=results_df['R2_std'], fmt='none', ecolor='black',
                capsize=5, capthick=2)

ax.axhline(0.70, color='green', linestyle='--', linewidth=2, label='Target R²=0.70', alpha=0.7)
ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

ax.set_xlabel('Pseudo-Time Method', fontsize=12, fontweight='bold')
ax.set_ylabel('LSTM R² (Test Set)', fontsize=12, fontweight='bold')
ax.set_title('LSTM Forecasting Performance by Pseudo-Time Method', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('visualizations_claude_code/lstm_performance_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations_claude_code/lstm_performance_claude_code.png")

# (B) Scatter: MSE vs R²
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(results_df['MSE'], results_df['R2'], s=200, c=range(len(results_df)),
                     cmap='viridis', edgecolors='black', linewidths=2)

for i, row in results_df.iterrows():
    ax.annotate(row['Method'], (row['MSE'], row['R2']),
                fontsize=9, ha='left', va='bottom',
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('MSE (Test Set)', fontsize=12, fontweight='bold')
ax.set_ylabel('R² (Test Set)', fontsize=12, fontweight='bold')
ax.set_title('LSTM Performance: MSE vs R² Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linestyle='-', linewidth=1)
ax.axvline(0.3, color='red', linestyle='--', alpha=0.5, label='Target MSE<0.3')
ax.axhline(0.7, color='green', linestyle='--', alpha=0.5, label='Target R²>0.7')
ax.legend()

plt.tight_layout()
plt.savefig('visualizations_claude_code/lstm_mse_vs_r2_claude_code.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations_claude_code/lstm_mse_vs_r2_claude_code.png")

print("\n" + "="*80)
print("LSTM Benchmarking Complete!")
print("="*80)
print(f"\nWinner: {best_method} with R²={best_r2:.4f}")
print("This method should be standardized for all future temporal modeling.")
print("="*80)
