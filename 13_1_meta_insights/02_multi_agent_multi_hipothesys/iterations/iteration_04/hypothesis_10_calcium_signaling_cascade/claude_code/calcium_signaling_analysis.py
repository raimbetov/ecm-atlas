#!/usr/bin/env python3
"""
Calcium Signaling Cascade Analysis - Hypothesis 10
Multi-Agent Multi-Hypothesis Framework - Iteration 04

Analysis of S100 proteins -> LOX/TGM crosslinkers pathway
Author: Claude Code Agent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Installing...")
    import subprocess
    subprocess.run(['python3', '-m', 'pip', 'install', 'torch', '--quiet'])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Paths
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_10_calcium_signaling_cascade/claude_code'
VIZ_DIR = f'{OUTPUT_DIR}/visualizations_claude_code'

# Create output directories
import os
os.makedirs(VIZ_DIR, exist_ok=True)

print("="*80)
print("CALCIUM SIGNALING CASCADE ANALYSIS - HYPOTHESIS 10")
print("="*80)
print()

# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print()

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================
print("STEP 1: Preparing protein matrix...")
print("-" * 80)

# Identify proteins (case-insensitive)
def find_proteins(df, pattern):
    """Find proteins matching pattern (case-insensitive)"""
    mask = df['Gene_Symbol'].str.contains(pattern, case=False, na=False)
    proteins = df[mask]['Gene_Symbol'].str.upper().unique()
    return sorted(set(proteins))

s100_proteins = find_proteins(df, 'S100')
lox_proteins = find_proteins(df, 'LOX')
tgm_proteins = find_proteins(df, 'TGM')

# Normalize gene symbols to uppercase
df['Gene_Symbol_Upper'] = df['Gene_Symbol'].str.upper()

print(f"S100 proteins found: {len(s100_proteins)}")
print(f"  {s100_proteins[:5]}... (showing first 5)")
print()
print(f"LOX proteins found: {len(lox_proteins)}")
print(f"  {lox_proteins}")
print()
print(f"TGM proteins found: {len(tgm_proteins)}")
print(f"  {tgm_proteins}")
print()

# Create protein matrix (samples x proteins)
# Group by Study_ID and Tissue to create samples
df['Sample_ID'] = df['Study_ID'] + '_' + df['Tissue']

# Pivot to wide format: rows=samples, columns=proteins, values=z-scores
protein_matrix = df.pivot_table(
    index='Sample_ID',
    columns='Gene_Symbol_Upper',
    values='Zscore_Old',  # Using old age z-scores
    aggfunc='mean'
)

# Filter to relevant proteins
all_proteins = s100_proteins + lox_proteins + tgm_proteins
available_proteins = [p for p in all_proteins if p in protein_matrix.columns]
protein_matrix = protein_matrix[available_proteins]

# Fill NaN with 0 (missing measurements)
protein_matrix_filled = protein_matrix.fillna(0)

print(f"Protein matrix shape: {protein_matrix_filled.shape}")
print(f"  Samples: {protein_matrix_filled.shape[0]}")
print(f"  Proteins: {protein_matrix_filled.shape[1]}")
print()

# Separate S100 and crosslinkers
s100_cols = [c for c in protein_matrix_filled.columns if c in s100_proteins]
lox_cols = [c for c in protein_matrix_filled.columns if c in lox_proteins]
tgm_cols = [c for c in protein_matrix_filled.columns if c in tgm_proteins]
crosslinker_cols = lox_cols + tgm_cols

print(f"S100 proteins in matrix: {len(s100_cols)}")
print(f"LOX proteins in matrix: {len(lox_cols)}")
print(f"TGM proteins in matrix: {len(tgm_cols)}")
print(f"Total crosslinkers: {len(crosslinker_cols)}")
print()

# ============================================================================
# STEP 2: CORRELATION ANALYSIS
# ============================================================================
print("STEP 2: Computing S100-crosslinker correlations...")
print("-" * 80)

correlations = []

for s100 in s100_cols:
    for crosslinker in crosslinker_cols:
        # Get data
        x = protein_matrix_filled[s100].values
        y = protein_matrix_filled[crosslinker].values

        # Spearman correlation (robust to outliers)
        rho, pval = stats.spearmanr(x, y)

        correlations.append({
            'S100': s100,
            'Crosslinker': crosslinker,
            'Spearman_rho': rho,
            'P_value': pval
        })

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('Spearman_rho', key=abs, ascending=False)

# Save correlations
corr_output = f'{OUTPUT_DIR}/s100_crosslinker_correlations_claude.csv'
corr_df.to_csv(corr_output, index=False)
print(f"Saved correlations to: {corr_output}")
print()

# Top correlations
print("Top 10 S100-crosslinker correlations:")
print(corr_df.head(10).to_string(index=False))
print()

# ============================================================================
# STEP 3: HEATMAP VISUALIZATION
# ============================================================================
print("STEP 3: Creating correlation heatmap...")
print("-" * 80)

# Pivot to heatmap format
heatmap_data = corr_df.pivot(index='S100', columns='Crosslinker', values='Spearman_rho')

# Plot
plt.figure(figsize=(12, 14))
sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', cbar_kws={'label': 'Spearman ρ'})
plt.title('S100 Proteins vs LOX/TGM Crosslinkers\nSpearman Correlation Heatmap',
          fontsize=14, fontweight='bold')
plt.xlabel('Crosslinker Proteins', fontsize=12)
plt.ylabel('S100 Calcium Signaling Proteins', fontsize=12)
plt.tight_layout()
heatmap_path = f'{VIZ_DIR}/calcium_network_heatmap_claude.png'
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved heatmap to: {heatmap_path}")
print()

# ============================================================================
# STEP 4: DEEP NEURAL NETWORK - PREDICT STIFFNESS PROXY
# ============================================================================
print("STEP 4: Building Deep Neural Network...")
print("-" * 80)

# Create stiffness proxy target
# Using weighted average of available crosslinkers
lox_available = [c for c in lox_cols if c in protein_matrix_filled.columns]
tgm_available = [c for c in tgm_cols if c in protein_matrix_filled.columns]

if 'TGM2' in tgm_available and 'LOX' in lox_available:
    # Preferred: LOX + TGM2 weighted
    stiffness_proxy = 0.5 * protein_matrix_filled['LOX'] + 0.3 * protein_matrix_filled['TGM2']
    print("Stiffness proxy: 0.5×LOX + 0.3×TGM2")
else:
    # Fallback: mean of all crosslinkers
    stiffness_proxy = protein_matrix_filled[crosslinker_cols].mean(axis=1)
    print(f"Stiffness proxy: mean of {len(crosslinker_cols)} crosslinkers")

# Features: S100 proteins
X = protein_matrix_filled[s100_cols].values
y = stiffness_proxy.values

print(f"Feature matrix: {X.shape}")
print(f"Target vector: {y.shape}")
print()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print()

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).reshape(-1, 1)

# Define neural network architecture
class CalciumSignalingNet(nn.Module):
    def __init__(self, input_dim):
        super(CalciumSignalingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model
input_dim = X_train.shape[1]
model = CalciumSignalingNet(input_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Network architecture:")
print(f"  Input layer: {input_dim} features (S100 proteins)")
print(f"  Hidden layers: 128 -> 64 -> 32")
print(f"  Output layer: 1 (stiffness proxy)")
print()

# Training
print("Training neural network...")
epochs = 500
train_losses = []
test_losses = []

for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_loss = criterion(test_outputs, y_test_t)
        test_losses.append(test_loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

print()

# Evaluate model
model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t).numpy()
    y_test_pred = model(X_test_t).numpy()

# Metrics
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Deep Learning Model Performance:")
print(f"  Training R²: {r2_train:.4f}")
print(f"  Test R²: {r2_test:.4f}")
print(f"  Training MAE: {mae_train:.4f}")
print(f"  Test MAE: {mae_test:.4f}")
print(f"  Training RMSE: {rmse_train:.4f}")
print(f"  Test RMSE: {rmse_test:.4f}")
print()

# Save model weights
model_path = f'{VIZ_DIR}/calcium_signaling_model_claude.pth'
torch.save(model.state_dict(), model_path)
print(f"Saved model weights to: {model_path}")
print()

# ============================================================================
# STEP 5: TRAINING LOSS CURVE
# ============================================================================
print("STEP 5: Plotting training curves...")
print("-" * 80)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', alpha=0.7)
plt.plot(test_losses, label='Validation Loss', alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Deep Learning Training Curve\nS100 → Stiffness Proxy Prediction',
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
loss_path = f'{VIZ_DIR}/training_loss_claude.png'
plt.savefig(loss_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved loss curve to: {loss_path}")
print()

# ============================================================================
# STEP 6: PREDICTIONS SCATTER PLOT
# ============================================================================
print("STEP 6: Creating predictions scatter plot...")
print("-" * 80)

plt.figure(figsize=(10, 8))

# Training data
plt.scatter(y_train, y_train_pred, alpha=0.6, s=100,
           label=f'Training (R²={r2_train:.3f})', color='blue')

# Test data
plt.scatter(y_test, y_test_pred, alpha=0.6, s=100,
           label=f'Test (R²={r2_test:.3f})', color='red')

# Perfect prediction line
all_vals = np.concatenate([y_train, y_test])
min_val, max_val = all_vals.min(), all_vals.max()
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

plt.xlabel('True Stiffness Proxy', fontsize=12)
plt.ylabel('Predicted Stiffness Proxy', fontsize=12)
plt.title('Deep Learning Predictions\nS100 Proteins → ECM Stiffness',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
scatter_path = f'{VIZ_DIR}/predictions_scatter_claude.png'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved scatter plot to: {scatter_path}")
print()

# ============================================================================
# STEP 7: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("STEP 7: Computing Random Forest feature importance...")
print("-" * 80)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'S100_Protein': s100_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Save
importance_path = f'{OUTPUT_DIR}/s100_feature_importance_claude.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"Saved feature importance to: {importance_path}")
print()

print("Top 15 S100 proteins by importance:")
print(feature_importance.head(15).to_string(index=False))
print()

# RF performance
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)
r2_train_rf = r2_score(y_train, y_train_rf)
r2_test_rf = r2_score(y_test, y_test_rf)

print(f"Random Forest R² (Train): {r2_train_rf:.4f}")
print(f"Random Forest R² (Test): {r2_test_rf:.4f}")
print()

# ============================================================================
# STEP 8: FEATURE IMPORTANCE BAR CHART
# ============================================================================
print("STEP 8: Plotting feature importance...")
print("-" * 80)

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue')
plt.yticks(range(len(top_features)), top_features['S100_Protein'].values)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('S100 Protein', fontsize=12)
plt.title('Top 15 S100 Proteins for Predicting ECM Stiffness\nRandom Forest Feature Importance',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
importance_plot_path = f'{VIZ_DIR}/feature_importance_claude.png'
plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved importance plot to: {importance_plot_path}")
print()

# ============================================================================
# STEP 9: RESULTS SUMMARY
# ============================================================================
print("STEP 9: Generating results summary...")
print("-" * 80)

# Top correlations
top_corr = corr_df.head(10).to_dict('records')

# Summary
results_summary = {
    'analysis_type': 'Calcium Signaling Cascade - S100 to LOX/TGM Pathway',
    'hypothesis': 'H10: S100 calcium sensors regulate ECM stiffness via LOX/TGM crosslinkers',
    'dataset': {
        'samples': int(protein_matrix_filled.shape[0]),
        'total_proteins': int(protein_matrix_filled.shape[1]),
        's100_proteins': len(s100_cols),
        'lox_proteins': len(lox_cols),
        'tgm_proteins': len(tgm_cols),
        'total_crosslinkers': len(crosslinker_cols)
    },
    'correlations': {
        'total_pairs': len(corr_df),
        'significant_positive': int((corr_df['Spearman_rho'] > 0.3).sum()),
        'significant_negative': int((corr_df['Spearman_rho'] < -0.3).sum()),
        'top_10_correlations': top_corr
    },
    'deep_learning': {
        'architecture': '128-64-32',
        'input_features': int(input_dim),
        'epochs': epochs,
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'train_mae': float(mae_train),
        'test_mae': float(mae_test),
        'train_rmse': float(rmse_train),
        'test_rmse': float(rmse_test)
    },
    'random_forest': {
        'n_estimators': 200,
        'train_r2': float(r2_train_rf),
        'test_r2': float(r2_test_rf),
        'top_5_features': feature_importance.head(5).to_dict('records')
    },
    'outputs': {
        'correlations_csv': corr_output,
        'feature_importance_csv': importance_path,
        'model_weights': model_path,
        'heatmap': heatmap_path,
        'loss_curve': loss_path,
        'scatter_plot': scatter_path,
        'importance_plot': importance_plot_path
    },
    'notes': [
        'CALM and CAMK proteins are MISSING from dataset',
        'Analysis focused on DIRECT S100 -> LOX/TGM pathway',
        'Small dataset (18 samples) - results are preliminary',
        'Z-scores used from old age samples'
    ]
}

# Save summary
summary_path = f'{OUTPUT_DIR}/results_summary_claude.json'
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"Saved results summary to: {summary_path}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("KEY FINDINGS:")
print()
print("1. TOP 5 S100-CROSSLINKER CORRELATIONS:")
for i, row in corr_df.head(5).iterrows():
    print(f"   {row['S100']} ↔ {row['Crosslinker']}: ρ={row['Spearman_rho']:.3f}, p={row['P_value']:.4f}")
print()

print("2. DEEP LEARNING PERFORMANCE:")
print(f"   Training R²: {r2_train:.4f}")
print(f"   Test R²: {r2_test:.4f}")
print(f"   Test MAE: {mae_test:.4f}")
print(f"   Status: {'SUCCESS' if r2_test > 0.5 else 'MODERATE'}")
print()

print("3. TOP 5 S100 PROTEINS BY IMPORTANCE:")
for i, row in feature_importance.head(5).iterrows():
    print(f"   {row['S100_Protein']}: {row['Importance']:.4f}")
print()

print("4. FILES CREATED:")
all_files = [
    corr_output,
    importance_path,
    model_path,
    heatmap_path,
    loss_path,
    scatter_path,
    importance_plot_path,
    summary_path
]
for f in all_files:
    print(f"   ✓ {f}")
print()
print("="*80)
