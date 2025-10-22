# Hypothesis 09: Temporal Aging Trajectories via Recurrent Neural Networks

## Scientific Question

Do LSTM networks trained on protein aging trajectories achieve >85% accuracy predicting 6-month-ahead protein states, with early-changing proteins (first temporal quartile) explaining >70% variance in late-stage ECM remodeling, and can Transformer attention mechanisms identify critical transition points ("point of no return") where aging becomes irreversible?

## Background Context

**Critical Gap from Iterations 01-02:**

ALL completed hypotheses (H01-H06) used CROSS-SECTIONAL analysis:
- **H03 Tissue Clocks:** Velocities = mean |Δz| (PROXY for Δz/year, not real temporal data)
- **H04 Deep Embeddings:** Latent factors are STATIC snapshots, no temporal evolution
- **H05 GNN:** Network assumes synchronous changes, ignores cascade timing
- **H02 Serpins:** Temporal ordering via eigenvector centrality (rough proxy, not true time)

**Missing Dimension:** NO hypothesis has:
1. Modeled WHEN proteins change during aging
2. Predicted FUTURE protein trajectories
3. Identified EARLY vs LATE-change proteins
4. Found CRITICAL TRANSITION POINTS

**Temporal Hypothesis:** Early-changing proteins (months 0-12) predict late-stage ECM remodeling (months 18-32), enabling early intervention targeting. Aging trajectories are NON-LINEAR with critical transition points where aging accelerates.

**Analogy:** Like climate tipping points, aging may have irreversible transitions that RNNs can detect.

## Data Source

```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Challenge:** Dataset may lack explicit age metadata. Solutions:
1. **Option A:** Extract age from Study_ID or metadata if available
2. **Option B:** Use tissue aging velocity as pseudo-time (H03 rankings)
3. **Option C:** Latent space traversal from H04 autoencoder as temporal proxy

## ML Requirements (MANDATORY)

### Must Use All of These:

1. **LSTM Sequence-to-Sequence Prediction (Required):**
   - Task: Predict future protein expression states (t+1, t+2, ..., t+k)
   - Input: Protein expression sequences (e.g., 3-5 tissues ordered by velocity)
   - Output: Next-state predictions
   - Architecture: Encoder-Decoder LSTM or many-to-many
   - Performance: Prediction accuracy >85% (MSE <0.3 for 1-step ahead)

2. **Transformer Attention for Critical Transitions (Required):**
   - Input: Protein sequences ordered by pseudo-time
   - Task: Identify "transition points" where aging rate changes
   - Method: Attention weights reveal which time-steps are most influential
   - Hypothesis: High-attention time-steps = critical transitions

3. **Early vs Late-Change Protein Classification:**
   - Calculate temporal gradient: Δ(protein)/Δ(time)
   - Rank proteins by gradient magnitude
   - Early-change: First quartile (top 25%)
   - Late-change: Last quartile (bottom 25%)
   - Test: Do early-change proteins predict late-change with R² >0.70?

4. **Granger Causality Testing:**
   - Test: Do early-change proteins Granger-cause late-change proteins?
   - Null hypothesis: Early changes do NOT predict late changes
   - Target: p<0.05 for causal relationship

5. **Time-Series Cross-Validation:**
   - Leave-future-out CV: Train on t1-t3, test on t4-t5
   - Ensures model doesn't "cheat" by using future data
   - Required for temporal validity

## Success Criteria

### Criterion 1: LSTM Predictive Performance (40 pts)

**Required:**
1. Create pseudo-temporal sequences:
   - Option A: Age metadata (if available)
   - Option B: Order tissues by velocity (slow → fast) as time proxy
   - Option C: Latent space traversal (use H04 autoencoder latent dim 0)
2. Train LSTM encoder-decoder:
   - Input: [t, t+1, ..., t+k-1]
   - Output: [t+k]
3. Performance targets:
   - 1-step ahead: MSE <0.3, R² >0.80
   - 3-step ahead: MSE <0.5, R² >0.60
4. Identify proteins with best prediction accuracy (top 10%)

**Deliverables:**
- `lstm_seq2seq_model_[agent].pth`
- `prediction_performance_[agent].csv` - MSE, R² per time-step
- `top_predictable_proteins_[agent].csv`

### Criterion 2: Early vs Late-Change Analysis (30 pts)

**Required:**
1. Calculate temporal gradient for each protein:
   - Gradient = Δz / Δ(pseudo-time)
2. Rank proteins by gradient magnitude
3. Extract quartiles:
   - Early-change (Q1): Top 25% by gradient
   - Late-change (Q4): Bottom 25%
4. Regression test:
   - Early-change protein expression → Predict late-change protein expression
   - Target: R² >0.70, p<0.01
5. Enrichment test:
   - Are coagulation proteins (H07) enriched in early-change? (Fisher's exact)
   - Are structural proteins (collagens) enriched in late-change?

**Deliverables:**
- `early_change_proteins_[agent].csv` - Q1 with gradients
- `late_change_proteins_[agent].csv` - Q4
- `early_late_regression_[agent].csv` - R², p-value
- `enrichment_analysis_[agent].csv`

### Criterion 3: Transformer Critical Transitions (20 pts)

**Required:**
1. Train Transformer with self-attention on protein sequences
2. Extract attention weights across time-steps
3. Identify high-attention time-steps (top 10% attention)
4. Hypothesize: High-attention steps = critical transition points
5. Validate:
   - Do proteins show accelerated change at high-attention time-steps?
   - Compare gradient before vs after transition (paired t-test)

**Deliverables:**
- `transformer_model_[agent].pth`
- `attention_weights_per_timestep_[agent].csv`
- `critical_transitions_[agent].csv` - Time-steps with high attention
- `attention_heatmap_[agent].png`

### Criterion 4: Granger Causality (10 pts)

**Required:**
1. Test Granger causality: Early-change → Late-change
2. Use statsmodels.tsa.stattools.grangercausalitytests
3. Lag selection: 1-3 steps
4. Report: p-value, F-statistic
5. If p<0.05 → early changes CAUSE late changes (causal arrow)

**Deliverables:**
- `granger_causality_[agent].csv` - p-values per lag
- `causal_network_[agent].png` - Early → Late protein network

## Required Artifacts

1. **01_plan_[agent].md**
2. **analysis_temporal_[agent].py**
3. **lstm_seq2seq_model_[agent].pth**
4. **transformer_model_[agent].pth**
5. **early_change_proteins_[agent].csv**
6. **late_change_proteins_[agent].csv**
7. **critical_transitions_[agent].csv**
8. **visualizations_[agent]/**:
   - prediction_performance_[agent].png (MSE over time-steps)
   - attention_heatmap_[agent].png (Transformer attention)
   - trajectory_plot_[agent].png (Protein trajectories over pseudo-time)
   - causal_network_[agent].png
9. **90_results_[agent].md**

## ML Implementation Template

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import grangercausalitytests

# Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Create pseudo-temporal sequences (Option B: velocity-based)
tissue_velocities = {
    'Lung': 4.2, 'Skeletal_muscle_EDL': 2.2, 'Skin dermis': 2.1,
    'Cortex': 2.0, 'Hippocampus': 1.2, 'Tubulointerstitial': 1.0
}  # From H03
ordered_tissues = sorted(tissue_velocities.keys(), key=lambda x: tissue_velocities[x])

# Pivot and order
pivot = df.pivot_table(values='Zscore_Delta', index='Gene_Symbol', columns='Tissue').fillna(0)
sequences = pivot[ordered_tissues].values  # Shape: (proteins, time-steps)

# LSTM Encoder-Decoder
class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h, c) = self.encoder(x)
        out, _ = self.decoder(x, (h, c))
        return self.fc(out)

# Train (simplified)
model = LSTMSeq2Seq(input_dim=sequences.shape[0])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Temporal gradient
gradients = np.gradient(sequences, axis=1).mean(axis=1)  # Average gradient per protein
early_change_proteins = pivot.index[np.argsort(np.abs(gradients))[-25:]]  # Top 25%

# Granger causality
early_data = pivot.loc[early_change_proteins].T.values  # Time-series
late_data = pivot.iloc[:25].T.values  # Bottom 25%
granger_results = grangercausalitytests(np.column_stack([late_data[:, 0], early_data[:, 0]]), maxlag=3)

print(f"Early-change proteins: {list(early_change_proteins)}")
```

## Expected Results

- LSTM 1-step prediction: MSE <0.3, R² >0.80
- Early-change proteins predict late-change: R² >0.70
- Coagulation proteins (H07) enriched in early-change (p<0.05)
- Transformer identifies 2-3 critical transition points (e.g., velocity ~2.0 = transition)
- Granger causality confirmed (p<0.05)

**Clinical Translation:**
- Early-change proteins = intervention targets (prevent late-stage remodeling)
- Critical transitions = timing windows for therapy

---

**Hypothesis ID:** H09
**Iteration:** 03
**Predicted Scores:** Novelty 10/10, Impact 9/10
**ML Focus:** ✅ LSTM, Transformer, Granger Causality, Time-Series CV
