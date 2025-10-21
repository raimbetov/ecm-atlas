# Hypothesis 07: Coagulation Cascade as Central Aging Hub

## Scientific Question

Can deep neural networks trained on coagulation cascade proteins predict tissue aging velocity (R² >0.85) and identify coagulation state (hypercoagulable vs hyperfibrinolytic) as the primary ECM aging driver, surpassing serpins and inflammation as central mechanisms, with LSTMs revealing temporal precedence where coagulation dysregulation leads ECM remodeling by ≥3 months?

## Background Context

**Convergent Discovery from Iterations 01-02:**

Coagulation proteins appeared in ALL 6 completed hypotheses despite NO pre-specified focus:
- **H06 Biomarker Panel:** F13B, GAS6 in top 8 proteins (AUC=1.0, SHAP consensus)
- **H03 Tissue Clocks:** Shared coagulation proteins (F2, SERPINB6A) across fast-aging tissues
- **H02 Serpins:** Coagulation serpins (SERPINC1, SERPINF2) highly dysregulated
- **H01 Antagonism:** F13B magnitude 7.80 SD (rank #2 antagonistic protein)
- **H05 GNN:** Coagulation factors present but NOT identified as top master regulators (gap to explore)
- **H04 Deep Embeddings:** Latent factors likely include coagulation module (not explicitly annotated)

**Central Hypothesis:** Coagulation cascade dysregulation is THE unifying mechanism of ECM aging, with thrombotic/fibrinolytic balance determining tissue-specific aging velocities.

**Clinical Translation:** Anticoagulants (warfarin, DOACs) and antiplatelets already FDA-approved, enabling immediate therapeutic testing if hypothesis confirmed.

## Data Source

```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Coagulation Proteins to Focus:** F2 (thrombin), F13B (fibrin stabilizing factor), GAS6 (growth arrest specific 6), SERPINC1 (antithrombin), PLAU (urokinase), PLAUR (urokinase receptor), FGA/FGB/FGG (fibrinogen chains), VWF (von Willebrand factor), PROC (protein C), PROS1 (protein S), THBD (thrombomodulin), and all related serpins (SERPINE1/PAI-1, SERPINF2).

## ML Requirements (MANDATORY)

### Must Use All of These:

1. **Deep Neural Network Classifier (Required):**
   - Task: Predict tissue aging velocity (regression) or fast/slow class (classification)
   - Features: Coagulation cascade proteins ONLY (20-30 proteins)
   - Architecture: 3-4 hidden layers (128-64-32-16), Dropout, BatchNorm
   - Performance target: R² >0.85 (regression) or AUC >0.90 (classification)

2. **LSTM Temporal Ordering (Required):**
   - Task: Model temporal sequences to test causality (coagulation → downstream ECM)
   - Input: Protein expression time-series (if age metadata available) OR create pseudo-temporal ordering via latent traversal
   - Output: Predict whether coagulation proteins change BEFORE collagens/MMPs
   - Early-change proteins: First quartile of aging trajectory

3. **Transfer Learning (Required):**
   - Pre-trained thrombosis/hemostasis models (if available via AlphaFold, STRING, or PubMed embeddings)
   - Fine-tune on ECM aging dataset
   - Hypothesis: Thrombosis pathways generalize to aging

4. **Network Centrality Re-Analysis:**
   - Rebuild protein correlation network focusing on coagulation proteins
   - Calculate betweenness centrality for coagulation module vs serpin module vs collagen module
   - Test: Does coagulation module have HIGHEST betweenness? (contradicts H05 findings)

5. **SHAP Interpretability:**
   - Apply SHAP to deep NN to identify which specific coagulation proteins drive predictions
   - Compare with H06 biomarker panel to validate F13B/GAS6 importance

6. **State Classification:**
   - Define coagulation states:
     - Hypercoagulable: F2↑, SERPINC1↓, FGA↑, PLAU↓
     - Hyperfibrinolytic: PLAU↑, SERPINE1↓, F13B↓
   - Classify tissues into states and correlate with aging velocity

## Success Criteria

### Criterion 1: Coagulation-Based Aging Velocity Prediction (40 pts)

**Required:**
1. Train deep NN using ONLY coagulation cascade proteins (20-30 proteins)
2. Predict tissue aging velocity (mean |Δz| per tissue) or binary fast/slow class
3. Performance targets:
   - Regression: R² >0.85, MAE <0.3, RMSE <0.5
   - Classification: Accuracy >85%, AUC >0.90
4. Compare with full protein set baseline:
   - Does coagulation-only model achieve ≥80% of full-model performance?
5. Cross-validation: 5-fold tissue-level CV

**Deliverables:**
- `coagulation_nn_model_[agent].pth` - Trained model
- `aging_velocity_predictions_[agent].csv` - Tissue-level predictions
- `model_performance_[agent].csv` - R², MAE, RMSE, AUC metrics
- `feature_importance_shap_[agent].png` - SHAP summary plot

### Criterion 2: Temporal Precedence Analysis (LSTM) (30 pts)

**Required:**
1. Create pseudo-temporal sequences:
   - Option A: If age metadata exists, use true time-series
   - Option B: Order tissues by aging velocity (slow → fast) as proxy for time
   - Option C: Latent space traversal (use H04 autoencoder latent factors)
2. Train LSTM to predict future protein states (t+1, t+2, ..., t+k)
3. Identify early-change proteins:
   - Extract first quartile proteins by temporal gradient
   - Test: Are coagulation proteins enriched in early-change group? (χ² test)
4. Granger causality test:
   - Test: Do coagulation proteins Granger-cause collagen/MMP changes?
   - Null hypothesis: Coagulation changes do NOT precede ECM remodeling

**Deliverables:**
- `lstm_model_[agent].pth` - Trained LSTM
- `early_change_proteins_[agent].csv` - First quartile proteins with enrichment stats
- `granger_causality_[agent].csv` - Coagulation → ECM causality p-values
- `temporal_trajectory_plot_[agent].png` - Protein change over pseudo-time

### Criterion 3: Coagulation State Classification (20 pts)

**Required:**
1. Define coagulation states using protein signatures:
   - Hypercoagulable: High thrombin (F2), low antithrombin (SERPINC1), high fibrinogen
   - Hyperfibrinolytic: High urokinase (PLAU), low PAI-1 (SERPINE1), low F13B
   - Balanced: Intermediate levels
2. Classify 16-17 tissues into states using thresholds or k-means clustering
3. Correlate coagulation state with aging velocity:
   - Hypothesis: Hypercoagulable tissues age fastest (Spearman ρ, p<0.05)
   - Visualize: Scatter plot (coagulation score vs velocity)
4. Compare with serpin-based classification (H02):
   - Does coagulation state predict velocity BETTER than serpin dysregulation? (R² comparison)

**Deliverables:**
- `coagulation_states_[agent].csv` - Tissue classifications + scores
- `state_velocity_correlation_[agent].csv` - Spearman ρ, p-value, R²
- `coagulation_state_scatter_[agent].png` - State vs velocity plot

### Criterion 4: Network Centrality Showdown (10 pts)

**Required:**
1. Build protein correlation network (|Spearman ρ| >0.5)
2. Identify modules via community detection:
   - Coagulation module (F2, F13B, GAS6, serpins)
   - Serpin module (all serpins)
   - Collagen module (COL1A1, COL3A1, etc.)
3. Calculate average betweenness centrality per module
4. Test: Coagulation module > Serpin module > Collagen module (pairwise Mann-Whitney)
5. If coagulation wins → hypothesis supported
6. If serpins win → H02 was correct, coagulation is downstream

**Deliverables:**
- `network_modules_[agent].csv` - Module assignments + centrality scores
- `centrality_comparison_[agent].csv` - Module-level betweenness statistics
- `network_visualization_[agent].png` - Network colored by module

## Required Artifacts

All in `claude_code/` or `codex/`:

1. **01_plan_[agent].md**
2. **analysis_coagulation_[agent].py** - Full ML pipeline
3. **coagulation_nn_model_[agent].pth**
4. **lstm_model_[agent].pth** (if temporal analysis performed)
5. **aging_velocity_predictions_[agent].csv**
6. **coagulation_states_[agent].csv**
7. **early_change_proteins_[agent].csv**
8. **visualizations_[agent]/**:
   - model_performance_[agent].png (ROC/regression scatter)
   - feature_importance_shap_[agent].png
   - temporal_trajectory_plot_[agent].png
   - coagulation_state_scatter_[agent].png
   - network_visualization_[agent].png
9. **90_results_[agent].md** - Knowledge Framework format

## ML Implementation Template

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
import shap

# 1. Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# 2. Filter coagulation cascade proteins
coagulation_proteins = [
    'F2', 'F13B', 'GAS6', 'SERPINC1', 'PLAU', 'PLAUR',
    'FGA', 'FGB', 'FGG', 'VWF', 'PROC', 'PROS1', 'THBD',
    'SERPINE1', 'SERPINF2', 'F7', 'F10', 'F11', 'F12', 'F13A1'
]

pivot = df[df['Gene_Symbol'].isin(coagulation_proteins)].pivot_table(
    values='Zscore_Delta', index='Tissue', columns='Gene_Symbol'
)
X = pivot.fillna(0).values  # Features: coagulation proteins per tissue
y_velocity = np.abs(pivot).mean(axis=1).values  # Target: aging velocity

# 3. Deep NN
class CoagulationNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)  # Regression output
        )

    def forward(self, x):
        return self.network(x)

model = CoagulationNN(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop (simplified)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    pred = model(torch.tensor(X, dtype=torch.float32))
    loss = criterion(pred, torch.tensor(y_velocity, dtype=torch.float32).unsqueeze(1))
    loss.backward()
    optimizer.step()

# 4. SHAP
model.eval()
explainer = shap.DeepExplainer(model, torch.tensor(X, dtype=torch.float32))
shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32))

# 5. Coagulation state classification
F2_z = pivot['F2'].fillna(0)
SERPINC1_z = pivot['SERPINC1'].fillna(0)
coag_score = F2_z - SERPINC1_z  # High score = hypercoagulable

pd.DataFrame({
    'Tissue': pivot.index,
    'Coagulation_Score': coag_score,
    'Aging_Velocity': y_velocity
}).to_csv('coagulation_states_[agent].csv', index=False)

# 6. Correlation test
from scipy.stats import spearmanr
rho, p = spearmanr(coag_score, y_velocity)
print(f"Coagulation state vs velocity: ρ={rho:.3f}, p={p:.4f}")
```

## Documentation Standards

Follow Knowledge Framework (TD diagrams for architecture, LR for workflow).

Reference: `/Users/Kravtsovd/projects/ecm-atlas/ADVANCED_ML_REQUIREMENTS.md`

## Expected Results

**If Hypothesis Confirmed:**
- Coagulation-only NN predicts aging velocity with R² >0.85
- Coagulation proteins are early-change proteins (enrichment p<0.05)
- Hypercoagulable tissues age fastest (ρ >0.6, p<0.01)
- Coagulation module has highest betweenness centrality

**If Hypothesis Rejected:**
- R² <0.70 (serpins or collagens perform better)
- Coagulation proteins are late-change (downstream of other mechanisms)
- No correlation between coagulation state and velocity

**Clinical Impact:**
- If confirmed: Anticoagulant therapy trials for aging intervention
- Biomarker: F13B/GAS6 levels predict tissue aging risk

---

**Hypothesis ID:** H07
**Iteration:** 03
**Predicted Scores:** Novelty 10/10, Impact 10/10
**ML Focus:** ✅ Deep NN, LSTM, Transfer Learning, SHAP, Network Centrality
