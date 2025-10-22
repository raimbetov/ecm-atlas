# H18 – Multi-Modal Aging Predictor: Unified Deep Learning Architecture

## Scientific Question
Can integrating LSTM temporal trajectories (H11), GNN relationship networks (H05), Autoencoder feature compression (H04), and S100 pathway knowledge (H08) into a unified deep learning architecture achieve R²>0.85 for tissue age prediction, surpassing individual model performance?

## Background & Rationale

**Individual Models Achieved Strong Performance:**
- **H11 LSTM:** PCA pseudo-time prediction, R²=0.29 (Claude)
- **H05 GNN:** 103,037 hidden relationships discovered
- **H04 Autoencoder:** 648→32 dimensions, preserves 89% variance
- **H08 S100 Pathway:** S100→stiffness, R²=0.75-0.81

**Problem: Models Are Isolated**
- Each model captures ONE aspect (time, network, compression, mechanism)
- No integration → information loss
- Biology is MULTI-MODAL (temporal + spatial + mechanistic + dimensional)

**Hypothesis: Integration > Sum of Parts**
- Autoencoder (H04) → compress 648 proteins to 32 latent features
- GNN (H05) → capture protein-protein interactions in latent space
- LSTM (H11) → model temporal dynamics on GNN-enriched features
- S100 pathway (H08) → mechanistic constraints (stiffness = S100 + LOX + TGM2)

**Expected Synergy:**
- Autoencoder removes noise (89% variance in 32D)
- GNN adds relationship context (103k edges)
- LSTM captures aging trajectories
- S100 pathway provides biological interpretability

**Clinical Impact:**
If unified model achieves R²>0.85 → can predict tissue age from proteomics alone, enabling:
- Personalized aging clocks (tissue-specific, not epigenetic)
- Intervention efficacy testing (does drug reverse biological age?)
- Clinical trial endpoints (reduce predicted age by 5 years)

## Objectives

### Primary Objective
Design and train a multi-modal deep learning architecture (Autoencoder → GNN → LSTM → S100 pathway fusion) for tissue age prediction, targeting R²>0.85 and MAE<3 years.

### Secondary Objectives
1. Ablation studies (test each module independently to quantify contribution)
2. Attention mechanism visualization (which proteins/edges/timepoints drive predictions?)
3. Transfer learning to external datasets (from H16 validation)
4. Interpretability analysis (SHAP values, biological pathway enrichment)
5. Comparison to baseline models (linear regression, random forest, individual deep models)

## Hypotheses to Test

### H18.1: Multi-Modal Superiority
Unified architecture achieves R²>0.85 for age prediction, outperforming best individual model (H08 R²=0.81) by ≥5%.

### H18.2: Synergistic Gains
Ablation studies show each module contributes ≥10% performance gain; removing any module drops R² below 0.75.

### H18.3: Biological Interpretability
Attention weights align with known aging pathways (S100, LOX, TGM2 receive highest attention); SHAP values correlate with H06 biomarker panel.

### H18.4: External Generalization
Model maintains R²≥0.75 on external datasets (from H16), demonstrating robustness beyond training data.

## Required Analyses

### 1. DATA PREPARATION

**Load Merged Dataset:**
```python
import pandas as pd
import numpy as np

data = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Filter: Only samples with age metadata
data_aged = data.dropna(subset=['Age'])

# Pivot to wide format (samples × proteins)
X_wide = data_aged.pivot_table(
    index=['Sample_ID', 'Age', 'Tissue'],
    columns='Gene_Symbol',
    values='Z_score'
).fillna(0)  # Zero for missing proteins

X = X_wide.values  # (n_samples, 648)
y = X_wide.index.get_level_values('Age').values  # (n_samples,)
tissues = X_wide.index.get_level_values('Tissue').values

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} proteins")
```

**Train/Validation/Test Split:**
```python
from sklearn.model_selection import train_test_split

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=tissues)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
```

### 2. ARCHITECTURE DESIGN

**Multi-Modal Pipeline:**

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data

# MODULE 1: Autoencoder (648 → 32 latent dimensions)
class ProteinAutoencoder(nn.Module):
    def __init__(self, input_dim=648, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# MODULE 2: GNN (graph-aware feature enrichment)
class ProteinGNN(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        # x: (n_nodes, 32) latent features
        # edge_index: (2, n_edges) from H05
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x  # (n_nodes, 32) graph-enriched features

# MODULE 3: LSTM (temporal trajectory modeling)
class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 32)

    def forward(self, x):
        # x: (batch, seq_len, 32) — sequence of latent features over pseudo-time
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.fc(h_n[-1])  # (batch, 32)
        return out

# MODULE 4: S100 Pathway Fusion (mechanistic constraints)
class S100PathwayFusion(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Pathway-specific attention
        self.pathway_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4)
        self.fc_stiffness = nn.Linear(latent_dim, 1)
        self.fc_age = nn.Linear(latent_dim + 1, 1)  # latent + stiffness → age

    def forward(self, x_latent, x_s100_features):
        # x_latent: (batch, 32) from LSTM
        # x_s100_features: (batch, 20) S100 protein levels

        # Attention over latent features
        x_attn, attn_weights = self.pathway_attention(
            x_latent.unsqueeze(0),
            x_latent.unsqueeze(0),
            x_latent.unsqueeze(0)
        )
        x_attn = x_attn.squeeze(0)

        # Predict stiffness from S100 pathway
        stiffness = self.fc_stiffness(x_attn)

        # Combine latent + stiffness → age
        x_combined = torch.cat([x_attn, stiffness], dim=1)
        age_pred = self.fc_age(x_combined)

        return age_pred, stiffness, attn_weights

# UNIFIED MODEL
class MultiModalAgingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder = ProteinAutoencoder(input_dim=648, latent_dim=32)
        self.gnn = ProteinGNN(input_dim=32, hidden_dim=64, output_dim=32)
        self.lstm = TemporalLSTM(input_dim=32, hidden_dim=64)
        self.s100_fusion = S100PathwayFusion(latent_dim=32)

    def forward(self, x_full, edge_index, x_sequence, x_s100):
        # Step 1: Autoencoder (compress 648 → 32)
        latent, reconstructed = self.autoencoder(x_full)

        # Step 2: GNN (enrich with graph structure)
        latent_gnn = self.gnn(latent, edge_index)

        # Step 3: LSTM (temporal dynamics)
        # x_sequence: pseudo-time ordered samples (batch, seq_len, 648)
        # First encode sequence through autoencoder
        seq_latent = self.autoencoder.encoder(x_sequence)  # (batch, seq_len, 32)
        lstm_out = self.lstm(seq_latent)

        # Step 4: S100 pathway fusion
        age_pred, stiffness, attn_weights = self.s100_fusion(lstm_out, x_s100)

        return age_pred, stiffness, attn_weights, reconstructed
```

### 3. TRAINING PIPELINE

**Load Pre-trained Components:**
```python
# Load H04 Autoencoder weights (if available)
# Load H05 GNN edge_index
edge_index = torch.load('/iterations/iteration_02/hypothesis_05_hidden_relationships/claude_code/gnn_edges_claude_code.pt')

# Load H08 S100 pathway genes
s100_genes = ['S100A1', 'S100A4', 'S100A6', 'S100A8', 'S100A9', 'S100A10', 'S100B', 'S100A11', 'S100A12', 'S100A13']
s100_indices = [protein_list.index(g) for g in s100_genes if g in protein_list]
```

**Training Loop:**
```python
from torch.optim import Adam
from torch.nn import MSELoss

model = MultiModalAgingPredictor()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion_age = MSELoss()
criterion_recon = MSELoss()

num_epochs = 200
best_val_r2 = -999

for epoch in range(num_epochs):
    model.train()

    # Mini-batch training
    for batch_idx in range(0, len(X_train), 32):
        batch_x = torch.FloatTensor(X_train[batch_idx:batch_idx+32])
        batch_y = torch.FloatTensor(y_train[batch_idx:batch_idx+32]).unsqueeze(1)
        batch_s100 = batch_x[:, s100_indices]

        # Create pseudo-time sequence (sliding window)
        # For simplicity, use same sample repeated (or sort by age for real sequence)
        batch_seq = batch_x.unsqueeze(1).repeat(1, 5, 1)  # (batch, 5, 648)

        optimizer.zero_grad()

        # Forward pass
        age_pred, stiffness, attn_weights, reconstructed = model(
            batch_x, edge_index, batch_seq, batch_s100
        )

        # Multi-task loss
        loss_age = criterion_age(age_pred, batch_y)
        loss_recon = criterion_recon(reconstructed, batch_x)
        loss = loss_age + 0.1 * loss_recon  # Weight reconstruction loss lower

        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_x = torch.FloatTensor(X_val)
        val_y = torch.FloatTensor(y_val).unsqueeze(1)
        val_s100 = val_x[:, s100_indices]
        val_seq = val_x.unsqueeze(1).repeat(1, 5, 1)

        val_pred, _, _, _ = model(val_x, edge_index, val_seq, val_s100)

        val_r2 = r2_score(val_y.numpy(), val_pred.numpy())

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            torch.save(model.state_dict(), f'multimodal_aging_best_{agent}.pth')

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val R²={val_r2:.4f}")
```

**Success Criteria:**
- Validation R²>0.85 (target)
- Test R²>0.80 (generalization)
- MAE<3 years

### 4. ABLATION STUDIES

**Test Each Module's Contribution:**
```python
# Baseline: Linear regression
from sklearn.linear_model import Ridge
baseline_model = Ridge(alpha=1.0)
baseline_model.fit(X_train, y_train)
r2_baseline = r2_score(y_val, baseline_model.predict(X_val))

# Ablation 1: Autoencoder only
model_ae_only = ProteinAutoencoder()
# Train and evaluate

# Ablation 2: Autoencoder + GNN
# (skip LSTM, S100 fusion)

# Ablation 3: Full model WITHOUT GNN
# (skip GNN module)

# Ablation 4: Full model WITHOUT S100 pathway
# (skip S100 fusion)

ablation_results = {
    'Baseline (Ridge)': r2_baseline,
    'Autoencoder Only': 0.45,  # Example
    'AE + GNN': 0.68,
    'AE + GNN + LSTM': 0.79,
    'Full (AE + GNN + LSTM + S100)': 0.87  # Target
}

# Plot ablation
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(list(ablation_results.keys()), list(ablation_results.values()))
plt.xlabel('R² Score')
plt.title('Ablation Study: Module Contributions')
plt.axvline(0.85, color='red', linestyle='--', label='Target R²=0.85')
plt.legend()
plt.tight_layout()
plt.savefig(f'visualizations_{agent}/ablation_study_{agent}.png', dpi=300)
```

**Success Criteria:**
- Full model outperforms baseline by ≥40% (0.87 vs 0.45)
- Each module contributes ≥10% gain

### 5. ATTENTION MECHANISM VISUALIZATION

**Extract Attention Weights:**
```python
model.eval()
with torch.no_grad():
    test_x = torch.FloatTensor(X_test[:10])  # 10 samples
    test_s100 = test_x[:, s100_indices]
    test_seq = test_x.unsqueeze(1).repeat(1, 5, 1)

    _, _, attn_weights, _ = model(test_x, edge_index, test_seq, test_s100)

# attn_weights: (batch, num_heads, seq_len, seq_len)
# Average over heads and batch
attn_avg = attn_weights.mean(dim=0).mean(dim=0).numpy()  # (32, 32)

# Heatmap
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(attn_avg, cmap='viridis', xticklabels=range(32), yticklabels=range(32))
plt.title('Attention Weights (Latent Dimensions)')
plt.xlabel('Key Dimension')
plt.ylabel('Query Dimension')
plt.savefig(f'visualizations_{agent}/attention_heatmap_{agent}.png', dpi=300)
```

**Map Attention to Proteins:**
```python
# Which proteins receive highest attention?
# (Requires mapping latent dimensions back to original proteins via autoencoder decoder)

# Use gradient-based attribution
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(test_x, target=0)  # Age prediction target

# Top proteins by attribution
protein_attributions = attributions.mean(dim=0).numpy()
top_proteins_idx = np.argsort(np.abs(protein_attributions))[-20:]
top_proteins = [protein_list[i] for i in top_proteins_idx]

print("Top 20 proteins by attention:")
print(top_proteins)
# Expected: S100A9, S100A10, LOX, TGM2, F13B (from H06/H08)
```

**Success Criteria:**
- S100, LOX, TGM2 in top 20 (mechanistic alignment)
- Overlap ≥50% with H06 biomarker panel

### 6. SHAP INTERPRETABILITY

**SHAP Values for Explainability:**
```python
import shap

# Use TreeSHAP approximation (treat neural net as black box)
explainer = shap.KernelExplainer(
    lambda x: model(torch.FloatTensor(x), edge_index,
                    torch.FloatTensor(x).unsqueeze(1).repeat(1, 5, 1),
                    torch.FloatTensor(x)[:, s100_indices]).detach().numpy(),
    X_train[:100]  # Background dataset
)

shap_values = explainer.shap_values(X_test[:50])

# Summary plot
shap.summary_plot(shap_values, X_test[:50], feature_names=protein_list, show=False)
plt.savefig(f'visualizations_{agent}/shap_summary_{agent}.png', dpi=300, bbox_inches='tight')

# Force plot (individual sample)
shap.force_plot(explainer.expected_value, shap_values[0], X_test[0], feature_names=protein_list, matplotlib=True, show=False)
plt.savefig(f'visualizations_{agent}/shap_force_{agent}.png', dpi=300, bbox_inches='tight')
```

**Pathway Enrichment of Top SHAP Features:**
```python
from scipy.stats import hypergeom

# Top 50 proteins by |SHAP|
shap_importance = np.abs(shap_values).mean(axis=0)
top_shap_idx = np.argsort(shap_importance)[-50:]
top_shap_proteins = [protein_list[i] for i in top_shap_idx]

# Test enrichment in H08 S100 pathway
s100_pathway_proteins = s100_genes + ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4', 'TGM2', 'TGM3']
overlap = set(top_shap_proteins) & set(s100_pathway_proteins)

M = 648  # Total proteins
n = len(s100_pathway_proteins)
N = 50  # Top SHAP
k = len(overlap)

p_value = hypergeom.sf(k-1, M, n, N)
print(f"S100 Pathway Enrichment: {k}/{N} overlap, p={p_value:.2e}")
```

**Success Criteria:**
- SHAP enrichment in S100 pathway: p<0.01
- Top 10 SHAP proteins overlap ≥60% with H06 panel

### 7. EXTERNAL VALIDATION (H16 DATASETS)

**Transfer to External Data:**
```python
# Load external datasets from H16
external_datasets = [
    'external_datasets/dataset_1/processed_zscore.csv',
    'external_datasets/dataset_2/processed_zscore.csv',
    # ... (from H16)
]

external_r2_scores = []

for dataset_path in external_datasets:
    ext_data = pd.read_csv(dataset_path)

    # Preprocess (same as training)
    X_ext = ...  # (n_samples, 648)
    y_ext = ...  # (n_samples,)

    # Predict
    model.eval()
    with torch.no_grad():
        ext_x = torch.FloatTensor(X_ext)
        ext_s100 = ext_x[:, s100_indices]
        ext_seq = ext_x.unsqueeze(1).repeat(1, 5, 1)

        ext_pred, _, _, _ = model(ext_x, edge_index, ext_seq, ext_s100)

    r2_ext = r2_score(y_ext, ext_pred.numpy())
    external_r2_scores.append(r2_ext)
    print(f"{dataset_path}: R²={r2_ext:.3f}")

# Meta-analysis
mean_external_r2 = np.mean(external_r2_scores)
print(f"Mean External R²: {mean_external_r2:.3f}")
```

**Success Criteria:**
- Mean external R²≥0.75 (acceptable generalization)
- ≥4/6 external datasets R²>0.70

## Deliverables

### Code & Models
- `multimodal_aging_model_{agent}.py` — Full architecture (AE+GNN+LSTM+S100)
- `train_multimodal_{agent}.py` — Training pipeline with multi-task loss
- `ablation_studies_{agent}.py` — Module contribution analysis
- `attention_visualization_{agent}.py` — Attention weights, SHAP, attribution
- `external_validation_{agent}.py` — Transfer to H16 datasets

### Data Tables
- `model_performance_{agent}.csv` — Train/val/test R², MAE, RMSE
- `ablation_results_{agent}.csv` — Performance of each module combination
- `attention_top_proteins_{agent}.csv` — Top 50 proteins by attention weights
- `shap_importance_{agent}.csv` — SHAP values for all 648 proteins
- `external_validation_{agent}.csv` — R² on each external dataset
- `pathway_enrichment_{agent}.csv` — Enrichment of SHAP features in known pathways

### Visualizations
- `visualizations_{agent}/architecture_diagram_{agent}.png` — Model flowchart (AE→GNN→LSTM→S100)
- `visualizations_{agent}/training_curves_{agent}.png` — Loss and R² over epochs
- `visualizations_{agent}/ablation_study_{agent}.png` — Bar chart of module contributions
- `visualizations_{agent}/predicted_vs_actual_{agent}.png` — Scatter plot (test set)
- `visualizations_{agent}/attention_heatmap_{agent}.png` — Attention weights visualization
- `visualizations_{agent}/shap_summary_{agent}.png` — SHAP feature importance
- `visualizations_{agent}/shap_force_{agent}.png` — Individual prediction explanation
- `visualizations_{agent}/external_validation_boxplot_{agent}.png` — R² distribution across datasets

### Report
- `90_results_{agent}.md` — CRITICAL findings:
  - **Performance:** Does multi-modal achieve R²>0.85? MAE<3 years?
  - **Ablation:** Which modules contribute most? Synergy quantified?
  - **Interpretability:** Do attention/SHAP align with H06/H08 biology?
  - **External Validation:** Does model generalize (R²≥0.75 on external)?
  - **Comparison:** How much better than baseline (Ridge, RF)?
  - **FINAL VERDICT:** Is this the BEST aging predictor to date?
  - **Recommendations:** Deploy as clinical tool? Further architecture refinements?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Validation R² | ≥0.85 | Multi-modal model |
| Test R² | ≥0.80 | Hold-out test set |
| Test MAE | <3 years | Age prediction error |
| Ablation: Full vs Baseline | ≥40% gain | Ridge regression comparison |
| Module contribution (each) | ≥10% gain | Ablation studies |
| SHAP-S100 pathway overlap | ≥60% (top 10) | Pathway enrichment |
| Attention biological alignment | S100/LOX/TGM2 in top 20 | Attribution analysis |
| External validation R² | ≥0.75 (mean) | H16 datasets |
| Overall SUCCESS | ≥7/8 criteria met | Comprehensive assessment |

## Expected Outcomes

### Scenario 1: BREAKTHROUGH (Best Aging Predictor)
- R²=0.87 (val), R²=0.83 (test), MAE=2.1 years
- Ablation: Each module contributes 12-18% gain, full model 95% better than baseline
- SHAP: 8/10 top proteins from H06 panel, S100 pathway p=10⁻⁸
- External: Mean R²=0.78 across 6 datasets
- **Action:** Publish in Nature/Science, deploy as open-source aging clock, clinical trials

### Scenario 2: STRONG (Improved but Not Best)
- R²=0.81 (val), R²=0.76 (test), MAE=3.5 years
- Matches H08 best single model but doesn't surpass
- SHAP alignment moderate (50% overlap with H06)
- External: Mean R²=0.68
- **Action:** Investigate architecture refinements (Transformers, graph attention), more data

### Scenario 3: WEAK (Integration Fails)
- R²=0.65 (val) — WORSE than H08 S100 model alone (R²=0.81)
- Modules interfere (negative synergy)
- Overfitting on training set
- **Action:** Simplify architecture, use ensemble instead of integration

### Scenario 4: PARTIAL SUCCESS (Some Modules Work)
- Full model R²=0.78, but ablation shows GNN contributes nothing
- AE+LSTM sufficient (R²=0.76), S100 fusion adds 2%
- **Action:** Remove GNN module, focus on AE+LSTM+S100 streamlined architecture

## Dataset

**Primary:**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Models to Load:**
- H04 Autoencoder: `/iterations/iteration_02/hypothesis_04_low_dimensional_embeddings/{claude_code,codex}/autoencoder_{agent}.pth`
- H05 GNN edges: `/iterations/iteration_02/hypothesis_05_hidden_relationships/{claude_code,codex}/gnn_edges_{agent}.pt`
- H08 S100 model: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/{claude_code,codex}/s100_stiffness_model_{agent}.pth`
- H11 PCA pseudo-time: `/iterations/iteration_04/hypothesis_11_standardized_temporal_trajectories/{claude_code,codex}/pca_pseudotime_{agent}.csv`

**External (from H16):**
- `/iterations/iteration_05/hypothesis_16_h13_validation_completion/{claude_code,codex}/external_datasets/`

## References

1. **H04 Autoencoder**: `/iterations/iteration_02/hypothesis_04_low_dimensional_embeddings/`
2. **H05 GNN**: `/iterations/iteration_02/hypothesis_05_hidden_relationships/`
3. **H08 S100 Pathway**: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/`
4. **H11 LSTM Trajectories**: `/iterations/iteration_04/hypothesis_11_standardized_temporal_trajectories/`
5. **Multi-Head Attention**: Vaswani et al. 2017, "Attention is All You Need"
6. **Graph Neural Networks**: Kipf & Welling 2017, "Semi-Supervised Classification with GCNs"
7. **SHAP Interpretability**: Lundberg & Lee 2017, "A Unified Approach to Interpreting Model Predictions"
