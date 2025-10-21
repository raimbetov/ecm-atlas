# Hypothesis 04: Deep Protein Embeddings Reveal Hidden Aging Modules

## Scientific Question

Can deep learning autoencoders and pre-trained protein language models (ESM-2, ProtBERT) discover latent aging factors and non-linear protein modules invisible to traditional correlation analysis, revealing hierarchical organization of ECM aging?

## Background Context

**Limitation of Traditional Methods:**
- Linear correlation (Spearman, Pearson) misses complex non-linear relationships
- PCA assumes linearity
- Hierarchical clustering uses simple distance metrics

**ML Hypothesis:** Deep autoencoders learn compressed representations capturing non-linear aging patterns. Pre-trained protein language models (ESM-2) provide biological priors from evolutionary sequences.

**Expected Discovery:** 5-10 latent factors representing fundamental aging mechanisms (e.g., "collagen degradation factor", "serpin dysregulation factor") with superior biological interpretability vs PCA.

## Data Source

**Primary Dataset:**
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Additional Resources:**
- Protein sequences from UniProt (for ESM-2 embeddings)
- Pre-trained ESM-2 model: `facebook/esm2_t33_650M_UR50D` (HuggingFace)
- ProtBERT model: `Rostlab/prot_bert` (optional)

## ML Requirements (MANDATORY)

### Must Use At Least 3 of These:

1. **Autoencoder (Required):**
   - Input: Proteins × Tissues z-score matrix
   - Architecture: ≥3 hidden layers, bottleneck latent dimension 5-20
   - Loss: MSE reconstruction
   - Output: Latent factors + reconstructed data

2. **Variational Autoencoder (VAE):**
   - Probabilistic latent space
   - Sample from learned distribution
   - Generate "synthetic" aged protein profiles

3. **Pre-trained Protein Embeddings:**
   - Download ESM-2 from HuggingFace
   - Get embeddings for all ECM proteins
   - Cluster embeddings → protein families
   - Compare ESM-2 clusters with aging-based clusters

4. **Deep Neural Network Regression:**
   - Predict aging velocity (tissue-specific Δz) from protein expression
   - Multi-layer perceptron with dropout, batch norm
   - SHAP for feature importance

5. **t-SNE / UMAP on Latent Space:**
   - Visualize autoencoder latent space
   - Color by: Tissue, Matrisome category, Aging velocity
   - Identify clusters = aging modules

6. **Attention Mechanisms:**
   - Build attention-based autoencoder
   - Attention weights reveal "important" proteins
   - Compare with correlation-based importance

## Success Criteria

### Criterion 1: Deep Autoencoder Training (40 pts)

**Required:**
1. Build autoencoder with architecture:
   - Encoder: Input → 128 → 64 → Latent (5-20 dims)
   - Decoder: Latent → 64 → 128 → Output
   - Activation: ReLU or LeakyReLU
   - Optional: Dropout (0.2-0.3), Batch Normalization
2. Train on protein z-score matrix (proteins × tissues)
3. Monitor training:
   - Plot loss curve (train + validation)
   - Early stopping if validation loss plateaus
   - Target: Reconstruction MSE < 0.5
4. Extract latent factors from bottleneck layer
5. Statistical validation:
   - Explained variance per latent factor
   - Compare with PCA: Autoencoder explains MORE variance?

**Deliverables:**
- `autoencoder_weights_[agent].pth` - Trained model
- `latent_factors_[agent].csv` - Latent embeddings for all proteins
- `training_loss_curve_[agent].png` - Train/val loss
- `latent_variance_explained_[agent].csv` - Variance per factor

### Criterion 2: Latent Factor Interpretation (30 pts)

**Required:**
1. For each latent factor (dimension):
   - Identify top 10 proteins with highest absolute loadings
   - Annotate biological function (collagens? serpins? enzymes?)
   - Name the factor based on biology (e.g., "Collagen Degradation Axis")
2. Correlation with known pathways:
   - Do latent factors align with Matrisome categories?
   - Do they separate by tissue?
3. Visualization:
   - Heatmap: Proteins × Latent Factors
   - UMAP: Proteins colored by dominant latent factor
   - Network: Proteins connected if similar latent embedding

**Deliverables:**
- `latent_factor_interpretation_[agent].md` - Biological annotation
- `protein_latent_heatmap_[agent].png`
- `latent_umap_[agent].png`

### Criterion 3: Pre-trained Protein Embeddings (20 pts)

**Required:**
1. Download ESM-2 model from HuggingFace
2. Get protein sequences from UniProt for all ECM proteins
3. Generate ESM-2 embeddings (650M parameter model)
4. Cluster ESM-2 embeddings (HDBSCAN or k-means)
5. Compare ESM-2 clusters with autoencoder latent clusters:
   - Adjusted Rand Index (ARI)
   - Do evolutionary patterns match aging patterns?

**Deliverables:**
- `esm2_embeddings_[agent].npy` - Saved embeddings
- `esm2_vs_aging_clusters_[agent].csv` - Comparison table
- `esm2_umap_[agent].png` - ESM-2 embedding visualization

### Criterion 4: Novel Discoveries (10 pts)

**Required:**
1. Identify protein modules missed by traditional clustering
2. Find non-linear relationships: Proteins correlated in latent space but NOT in raw data
3. Predict aging: Use autoencoder latent factors → Predict tissue aging velocity

**Deliverables:**
- `novel_modules_[agent].csv` - New protein groups
- `nonlinear_pairs_[agent].csv` - Protein pairs with non-linear relationships
- `aging_prediction_performance_[agent].csv` - R², MAE metrics

## Required Artifacts

All files in `claude_code/` or `codex/`:

### Mandatory:
1. **01_plan_[agent].md** - Analysis plan
2. **analysis_ml_[agent].py** - Full ML pipeline code
3. **autoencoder_weights_[agent].pth** - Trained autoencoder
4. **latent_factors_[agent].csv** - Latent embeddings (proteins × latent dims)
5. **training_loss_curve_[agent].png** - Loss monitoring
6. **latent_factor_interpretation_[agent].md** - Biological interpretation
7. **visualizations_[agent]/** - Folder with:
   - protein_latent_heatmap_[agent].png
   - latent_umap_[agent].png
   - esm2_umap_[agent].png (if ESM-2 used)
   - attention_weights_[agent].png (if attention used)
8. **90_results_[agent].md** - Final report in Knowledge Framework format

### Optional Advanced:
- VAE weights and samples
- Attention mechanism visualizations
- Predictive model for aging velocity

## ML Implementation Template

```python
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# 2. Prepare protein × tissue matrix
pivot = df.pivot_table(values='Zscore_Delta', index='Gene_Symbol', columns='Tissue')
X = pivot.fillna(0).values  # Shape: (n_proteins, n_tissues)

# 3. Split train/val
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# 4. Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 5. Define Autoencoder
class ProteinAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# 6. Training loop
model = ProteinAutoencoder(input_dim=X_train.shape[1], latent_dim=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    recon, _ = model(torch.tensor(X_train_scaled, dtype=torch.float32))
    loss = criterion(recon, torch.tensor(X_train_scaled, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        recon_val, _ = model(torch.tensor(X_val_scaled, dtype=torch.float32))
        val_loss = criterion(recon_val, torch.tensor(X_val_scaled, dtype=torch.float32))
        val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}")

# 7. Extract latent factors
model.eval()
with torch.no_grad():
    _, latent = model(torch.tensor(scaler.transform(X), dtype=torch.float32))
    latent_factors = latent.numpy()

# Save
torch.save(model.state_dict(), 'autoencoder_weights_[agent].pth')
pd.DataFrame(latent_factors, index=pivot.index, columns=[f'Latent_{i}' for i in range(10)]).to_csv('latent_factors_[agent].csv')
```

## Documentation Standards

Follow Knowledge Framework (TD diagrams for architecture, LR for training process).

Reference: `/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`

**Also reference:** `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/ADVANCED_ML_REQUIREMENTS.md`

## Expected Results

**Hypothesis Confirmation:**
- Autoencoder discovers 5-10 latent factors
- Each factor represents biological aging module
- Latent factors explain >80% variance (vs PCA ~60-70%)
- ESM-2 clusters align with aging-based clusters (ARI > 0.4)

**Novel Discovery:**
- Non-linear protein relationships (e.g., synergistic effects)
- Hidden modules not visible in correlation networks
- Improved aging velocity prediction (R² > 0.7)

---

**Task Created:** 2025-10-21
**Hypothesis ID:** H04
**Iteration:** 02
**Predicted Scores:** Novelty 10/10, Impact 9/10
**ML Focus:** ✅ Autoencoders, ESM-2, Deep Learning
