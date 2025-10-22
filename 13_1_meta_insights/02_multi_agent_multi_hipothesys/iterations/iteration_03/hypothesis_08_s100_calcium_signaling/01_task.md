# Hypothesis 08: S100 Family Calcium Signaling as Inflammation-Independent Aging Mechanism

## Scientific Question

Can transfer learning from pre-trained calcium signaling models identify S100-mediated mechanosensitive pathways that predict tissue stiffness (correlation ρ>0.7) and aging velocity independent of inflammation markers (p<0.05), with S100B/S100A8/S100A9 acting as calcium rheostats modulating ECM crosslinking enzymes (LOX family, transglutaminases)?

## Background Context

**Paradoxical Discovery from Iterations 01-02:**

S100 proteins selected by 3 independent ML methods across 4 hypotheses, yet inflammation rejected as mechanism:
- **H04 Deep Embeddings:** S100A8/S100A9 define Latent Factor 3 (Inflammation module)
- **H06 Biomarker Panel:** S100A9 in top 8 proteins (SHAP consensus)
- **H03 Tissue Markers:** S100B (dermis TSI=50.74), S100a5 (hippocampus TSI=3.60)
- **H03 Mechanism Test:** Fast-aging tissues do NOT share inflammatory signatures (p=0.41-0.63)

**Resolution Hypothesis:** S100 proteins act via calcium-dependent mechanotransduction and ECM crosslinking pathways, NOT classical inflammation. This explains selection by ML without inflammatory phenotype.

**S100 Family Members:** S100A8, S100A9, S100B, S100A1, S100A4, S100A6, S100P, S100A10, S100A11, and others (23 total in humans).

**Calcium-ECM Crosslinking Hypothesis:**
- S100 proteins are calcium sensors with 2 EF-hand motifs
- Calcium binding triggers conformational changes exposing protein interaction sites
- S100-calcium complexes activate crosslinking enzymes: LOX (lysyl oxidase), LOXL1-4, TGM2 (transglutaminase 2)
- Crosslinking increases tissue stiffness → mechanotransduction feedback loop

## Data Source

```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Focus Proteins:**
- S100 family: All S100A/B/P proteins
- Crosslinking enzymes: LOX, LOXL1, LOXL2, LOXL3, LOXL4, TGM2, TGM1, TGM3
- Calcium signaling: CALM1-3 (calmodulin), CAMK2A/B, SLC8A1 (Na/Ca exchanger)

## ML Requirements (MANDATORY)

### Must Use At Least 3 of These:

1. **Transfer Learning from AlphaFold Structures (Required):**
   - Download AlphaFold structures for S100 proteins
   - Extract structural features (EF-hand conformations, protein interfaces)
   - Fine-tune on ECM aging data
   - Hypothesis: Calcium-binding conformation predicts aging dysregulation

2. **Deep NN: S100 Expression → Tissue Stiffness Prediction:**
   - Input: S100 family expression profiles per tissue
   - Output: Predicted tissue stiffness (Young's modulus proxy via collagen crosslinking)
   - Proxy for stiffness: LOX + TGM2 expression (crosslinking activity)
   - Performance target: R² >0.70

3. **Attention Mechanisms: S100-LOX-TGM Network Discovery:**
   - Build multi-head attention network to learn S100 → crosslinking enzyme relationships
   - Extract attention weights to identify which S100 proteins regulate which enzymes
   - Hypothesis: S100B → LOX, S100A8/A9 → TGM2

4. **Correlation Network Analysis:**
   - Compute S100-LOX-TGM correlation network (Spearman)
   - Test: S100 proteins more correlated with crosslinking enzymes than with inflammatory markers
   - Inflammatory markers: IL6, IL1B, TNF, CXCL8, CCL2

5. **Mechanistic Pathway Enrichment:**
   - Test if S100 expression correlates with:
     - Mechanotransduction pathways (YAP/TAZ, ROCK, MLC2)
     - ECM stiffness markers (collagen I/III ratio, fibronectin EDA)
     - NOT inflammation (IL-6, TNF-α, NF-κB targets)

## Success Criteria

### Criterion 1: S100-Stiffness Prediction (40 pts)

**Required:**
1. Define tissue stiffness proxy:
   - Stiffness Score = α·LOX + β·TGM2 + γ·(COL1/COL3 ratio)
   - Calibrate weights via literature (e.g., α=0.5, β=0.3, γ=0.2)
2. Train deep NN: S100 expression → Stiffness Score
3. Performance: R² >0.70, MAE <0.3
4. Validate: S100 expression does NOT correlate with inflammation score (IL6+TNF, p>0.05)

**Deliverables:**
- `s100_stiffness_model_[agent].pth`
- `stiffness_predictions_[agent].csv`
- `s100_vs_inflammation_[agent].csv` - Correlation test

### Criterion 2: S100-Crosslinking Network (30 pts)

**Required:**
1. Build correlation matrix: S100 proteins × Crosslinking enzymes (LOX family, TGM family)
2. Extract top S100-enzyme pairs (|ρ| >0.6, p<0.05)
3. Attention network: Learn which S100 proteins regulate which enzymes
4. Compare with S100-inflammation correlations:
   - Hypothesis: S100-crosslinking correlations > S100-inflammation (paired t-test)

**Deliverables:**
- `s100_crosslinking_network_[agent].csv` - Correlation matrix
- `attention_weights_[agent].npy` - Attention scores
- `s100_enzyme_heatmap_[agent].png`

### Criterion 3: Mechanotransduction Pathway Enrichment (20 pts)

**Required:**
1. Test S100 correlation with mechanotransduction genes:
   - YAP1, TAZ (WWTR1), ROCK1, ROCK2, MYL2 (myosin light chain 2)
2. Enrichment test: S100 proteins co-expressed with mechanotransduction > inflammation (Fisher's exact)
3. Visualize: Network diagram showing S100 → mechanotransduction → crosslinking → stiffness

**Deliverables:**
- `mechanotransduction_enrichment_[agent].csv`
- `pathway_network_[agent].png`

### Criterion 4: AlphaFold Transfer Learning (10 pts)

**Required:**
1. Download AlphaFold structures for S100A8, S100A9, S100B (if available)
2. Extract structural features (optional: use ESM-2 embeddings as proxy)
3. Fine-tune on aging data
4. Compare structural vs expression-based models

**Deliverables:**
- `alphafold_transfer_model_[agent].pth` (or ESM-2 embeddings)
- `structural_vs_expression_[agent].csv`

## Required Artifacts

1. **01_plan_[agent].md**
2. **analysis_s100_[agent].py**
3. **s100_stiffness_model_[agent].pth**
4. **stiffness_predictions_[agent].csv**
5. **s100_crosslinking_network_[agent].csv**
6. **visualizations_[agent]/**:
   - s100_enzyme_heatmap_[agent].png
   - pathway_network_[agent].png
   - stiffness_scatter_[agent].png
7. **90_results_[agent].md**

## ML Implementation Template

```python
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn as nn

# Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Filter S100 proteins
s100_proteins = [p for p in df['Gene_Symbol'].unique() if p.startswith('S100')]
crosslinking = ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4', 'TGM2', 'TGM1', 'TGM3']
inflammation = ['IL6', 'IL1B', 'TNF', 'CXCL8', 'CCL2']

# Pivot
pivot_s100 = df[df['Gene_Symbol'].isin(s100_proteins)].pivot_table(
    values='Zscore_Delta', index='Tissue', columns='Gene_Symbol'
).fillna(0)

pivot_cross = df[df['Gene_Symbol'].isin(crosslinking)].pivot_table(
    values='Zscore_Delta', index='Tissue', columns='Gene_Symbol'
).fillna(0)

# Stiffness proxy
stiffness_score = 0.5 * pivot_cross.get('LOX', 0) + 0.3 * pivot_cross.get('TGM2', 0)

# Deep NN
class S100StiffnessNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = S100StiffnessNN(input_dim=len(s100_proteins))
# Training loop...

# Correlation test
s100_inflammation_corr = []
for s100 in s100_proteins:
    for inflam in inflammation:
        if s100 in pivot_s100.columns:
            rho, p = spearmanr(pivot_s100[s100], pivot_cross.get(inflam, np.zeros(len(pivot_s100))))
            s100_inflammation_corr.append({'S100': s100, 'Inflammation': inflam, 'rho': rho, 'p': p})

pd.DataFrame(s100_inflammation_corr).to_csv('s100_vs_inflammation_[agent].csv', index=False)
```

## Expected Results

- S100 expression predicts stiffness (R² >0.70)
- S100-crosslinking correlations > S100-inflammation (p<0.01)
- S100B → LOX, S100A8/A9 → TGM2 (attention weights >0.5)
- Mechanotransduction pathway enriched (Fisher's p<0.05)

---

**Hypothesis ID:** H08
**Iteration:** 03
**Predicted Scores:** Novelty 9/10, Impact 8/10
**ML Focus:** ✅ Transfer Learning, Deep NN, Attention, Network Analysis
