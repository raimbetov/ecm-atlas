# H11 – Standardized Temporal Trajectories: Resolving H09 Agent Disagreement

## Scientific Question
Which pseudo-time construction method produces robust, reproducible LSTM predictions of ECM aging trajectories: tissue velocity ranking (H03), PCA-based ordering, or alternative trajectory inference methods?

## Background & Rationale

**CRITICAL DISAGREEMENT from Iteration 03 (H09):**
- **Claude Code:** LSTM R²=0.81, MSE=0.165 → CONFIRMED temporal predictability
- **Codex:** LSTM R²=0.011, MSE=0.381 → REJECTED (near-baseline performance)
- **Root cause:** Different pseudo-time construction methods

**Agent Methods:**
1. **Claude (R²=0.81):** Used H03 tissue velocities as temporal axis
   - Ordering: Skeletal muscle (v=1.02) → Lung (v=4.29)
   - Justification: Velocity = aging rate proxy
2. **Codex (R²=0.011):** Used PCA (PC1) on ΔZ profiles
   - Ordering: Skeletal muscle → Intervertebral disc IAF
   - Justification: Principal component captures aging trajectory variance

**Why This Matters:**
- If LSTM performance depends on pseudo-time method → temporal modeling is fragile
- If one method is universally better → standardize for all future hypotheses
- If both fail on external data → need REAL longitudinal data (not cross-sectional proxies)

## Objectives

### Primary Objective
Systematically compare 4+ pseudo-time construction methods on the SAME dataset and identify the most robust approach for temporal ECM aging analysis.

### Secondary Objectives
1. **Literature validation:** Find best practices for pseudo-time inference in aging proteomics
2. **New datasets:** Identify LONGITUDINAL aging proteomics (real time-series, not cross-sectional)
3. **Benchmarking:** Test all methods on external data if longitudinal datasets found
4. **Recommendation:** Standardize one method for Iterations 05-07

## Hypotheses to Test

### H11.1: Velocity-Based Ordering (Claude Method)
Tissue velocity ranking (H03) produces highest LSTM R² (>0.70).

### H11.2: PCA-Based Ordering (Codex Method)
PCA (PC1) on ΔZ profiles produces comparable or superior R² vs velocity.

### H11.3: Diffusion Pseudotime
Diffusion maps (DESTINY R package) outperform both velocity and PCA.

### H11.4: Slingshot Trajectory Inference
Slingshot algorithm (Bioconductor) identifies branching aging trajectories better than linear methods.

### H11.5: Real Longitudinal Data
If longitudinal data found → pseudo-time methods correlate with real time (ρ>0.70).

## Required Analyses

### 1. LITERATURE SEARCH (MANDATORY)

**Search queries:**
```
1. "pseudo-time construction aging proteomics"
2. "trajectory inference methods comparison"
3. "temporal modeling best practices omics"
4. "diffusion maps aging"
5. "Slingshot trajectory inference validation"
6. "longitudinal aging proteomics"
```

**Tasks:**
- Search PubMed, bioRxiv, Nature Methods, Cell Systems
- Download methodological papers (trajectory inference benchmarks)
- Extract: Recommended methods, validation strategies, pitfalls
- Save: `literature_pseudotime.md`

### 2. NEW DATASET SEARCH (MANDATORY - HIGHEST PRIORITY)

**Search targets:**
```
- GEO: "longitudinal aging proteomics" OR "time series aging"
- PRIDE: "aging" + "time course"
- ProteomeXchange: Filter by "Homo sapiens" + "aging" + multiple timepoints
- Published cohorts: Baltimore Longitudinal Study of Aging (BLSA), Framingham
```

**Criteria:**
- **MUST HAVE:** ≥2 timepoints per subject (longitudinal, not cross-sectional)
- **Preferred:** 3+ timepoints, human, ≥50 subjects
- **Tissues:** Any (blood/plasma preferred for availability)
- **Proteins:** Overlap with ECM proteins (≥50 shared genes)

**If found:**
- Download raw data
- Normalize with universal_zscore_function.py
- Create "ground truth" temporal ordering (real time)
- Compare: Do pseudo-time methods recapitulate real time?

### 3. PSEUDO-TIME METHODS COMPARISON (4 METHODS MINIMUM)

#### Method A: Tissue Velocity Ranking (Claude Approach — H03)

**Implementation:**
```python
# From H03 results
tissue_velocities = {
    'Lung': 4.29,
    'Tubulointerstitial': 3.45,
    # ... all 17 tissues
    'Skeletal_muscle_Gastrocnemius': 1.02
}
pseudo_time = rank(tissue_velocities)
```

**Rationale:** Velocity = rate of aging → faster = later in timeline

#### Method B: PCA-Based Ordering (Codex Approach)

**Implementation:**
```python
from sklearn.decomposition import PCA

# Tissue × ΔZ matrix
pca = PCA(n_components=1)
pc1_scores = pca.fit_transform(tissue_delta_matrix)
pseudo_time = rank(pc1_scores)
```

**Rationale:** PC1 captures maximum variance in aging direction

#### Method C: Diffusion Pseudotime (NEW)

**Implementation:**
```R
library(destiny)

# Tissue × protein matrix
dm <- DiffusionMap(data = tissue_matrix)
pseudo_time <- rank(dm@eigenvectors[, 1])  # DC1
```

**Rationale:** Diffusion maps preserve manifold structure, robust to noise

#### Method D: Slingshot Trajectory Inference (NEW)

**Implementation:**
```R
library(slingshot)
library(SingleCellExperiment)

sce <- SingleCellExperiment(assays = list(counts = tissue_matrix))
sce <- slingshot(sce, reducedDim = 'PCA')
pseudo_time <- slingPseudotime(sce)[, 1]
```

**Rationale:** Detects branching trajectories (e.g., disc vs lung endpoints)

#### Method E: Latent Space Traversal (NEW - from H04)

**Implementation:**
```python
# Use H04 autoencoder latent factors
latent_coords = autoencoder.encode(tissue_matrix)
# Order by latent factor 1 (or factor most correlated with age metadata)
pseudo_time = rank(latent_coords[:, 0])
```

**Rationale:** Nonlinear dimensionality reduction may capture aging better than PCA

### 4. LSTM BENCHMARKING (ALL METHODS)

**Shared LSTM architecture (from H09):**
```python
LSTMSeq2Seq(
    input_dim=num_proteins,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2,
    forecast_horizon=3
)
```

**Evaluation:**
- **Train/Val/Test split:** Leave-future-out (train ≤ t10, val ≤ t13, test > t13)
- **Metrics:** MSE, R², MAE for forecast horizons 1/2/3
- **Repeat:** 5-fold CV with different temporal splits

**Output table:**
| Method | MSE (h=1) | MSE (h=2) | MSE (h=3) | R² (mean) | Rank |
|--------|-----------|-----------|-----------|-----------|------|
| Velocity (Claude) | ? | ? | ? | ? | ? |
| PCA (Codex) | ? | ? | ? | ? | ? |
| Diffusion | ? | ? | ? | ? | ? |
| Slingshot | ? | ? | ? | ? | ? |
| Autoencoder | ? | ? | ? | ? | ? |

**Success criteria:**
- At least ONE method achieves R²>0.70 consistently (5/5 folds)
- Winner identified (highest mean R²)

### 5. CRITICAL TRANSITIONS VALIDATION

Both Claude and Codex identified critical transitions (Ovary Cortex, Heart) despite disagreement on R².

**Test:**
- Do ALL pseudo-time methods identify the same critical transition points?
- Transformer attention analysis for each method
- Quantify: Spearman correlation of attention weights across methods

**If consistent:**
- Critical transitions are ROBUST (independent of pseudo-time method)

**If inconsistent:**
- Transitions are artifacts of specific orderings

### 6. GRANGER CAUSALITY CONSISTENCY

**Test:**
- Run Granger causality (lags 1-3) for ALL methods
- Compare: Do different pseudo-times produce different causal networks?
- Metric: Jaccard similarity of significant edges (p<0.05)

**Expected:**
- If Jaccard >0.50 → causal relationships robust
- If Jaccard <0.30 → pseudo-time choice confounds causality

### 7. EXTERNAL VALIDATION (IF LONGITUDINAL DATA FOUND)

**Ground truth test:**
```python
# Longitudinal data: Subject X Time X Protein tensor
# Real pseudo-time = actual time (months, years)

# For each method:
spearman_corr(method_pseudo_time, real_time)
```

**Performance on real data:**
- Train LSTM on cross-sectional data with method X pseudo-time
- Test LSTM on longitudinal data (predict future timepoints)
- Report: R² on prospective predictions

**Success criteria:**
- Method with highest R² on cross-sectional ALSO has highest R² on longitudinal (consistency)

### 8. SENSITIVITY ANALYSIS

**Robustness tests:**
1. **Tissue subset:** Remove 1 tissue at a time, recompute pseudo-time, measure ordering stability
2. **Protein subset:** Use only top 100 genes (by variance), compare pseudo-time orderings
3. **Noise injection:** Add Gaussian noise (σ=0.1, 0.2, 0.5), test LSTM degradation

**Metric:** Kendall's τ (rank correlation) between original and perturbed pseudo-time

**Success criteria:**
- Robust method: τ>0.80 under all perturbations

## Deliverables

### Code & Models
- `analysis_pseudotime_comparison_{agent}.py` — main comparison script
- `lstm_benchmark_{agent}.py` — LSTM training for all methods
- `diffusion_pseudotime_{agent}.R` — Diffusion maps script
- `slingshot_trajectory_{agent}.R` — Slingshot inference
- `literature_search_pseudotime_{agent}.py` — automated queries

### Data Tables
- `pseudotime_orderings_{agent}.csv` — Tissue orderings for all 5 methods
- `lstm_performance_{agent}.csv` — R², MSE, MAE for each method × forecast horizon
- `critical_transitions_comparison_{agent}.csv` — Attention weights across methods
- `granger_causality_consistency_{agent}.csv` — Edge overlap (Jaccard matrix)
- `sensitivity_analysis_{agent}.csv` — Kendall's τ under perturbations
- `longitudinal_validation_{agent}.csv` — Correlation with real time (if data found)
- `literature_recommendations_{agent}.csv` — Best practices from papers

### Visualizations
- `visualizations_{agent}/pseudotime_comparison_{agent}.png` — Parallel coordinates plot of tissue orderings
- `visualizations_{agent}/lstm_performance_{agent}.png` — Bar chart: R² by method
- `visualizations_{agent}/attention_consistency_{agent}.png` — Heatmap: attention weights correlation
- `visualizations_{agent}/sensitivity_heatmap_{agent}.png` — τ under perturbations
- `visualizations_{agent}/slingshot_trajectory_{agent}.png` — Branching trajectory plot (if detected)

### Report
- `90_results_{agent}.md` — comprehensive findings with:
  - Literature synthesis (recommended methods, citations)
  - Method comparison (which pseudo-time is best?)
  - Robustness analysis (sensitivity to noise, tissue subsets)
  - Critical transitions consistency
  - External validation (if longitudinal data found)
  - **RECOMMENDATION:** Standardized method for Iterations 05-07
  - **ROOT CAUSE ANALYSIS:** Why did Claude/Codex disagree in H09?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Best method R² | >0.70 | LSTM forecast accuracy |
| Method consistency | Jaccard >0.50 | Granger causality edges |
| Sensitivity (Kendall τ) | >0.80 | Tissue subset robustness |
| Attention correlation | ρ>0.60 | Critical transitions agreement |
| Literature papers | ≥5 relevant | Trajectory inference reviews |
| Longitudinal datasets | ≥1 found | GEO/PRIDE search |
| Prospective R² | >0.60 | Longitudinal validation (if data) |

## Expected Outcomes

### Scenario 1: Velocity Method Wins
Claude's H03 velocity approach is superior → Standardize for all temporal modeling.

### Scenario 2: Diffusion/Slingshot Wins
Novel methods outperform both agents → Update H09 with best method, re-run analysis.

### Scenario 3: All Methods Fail on Longitudinal Data
Pseudo-time proxies are unreliable → Require REAL longitudinal cohorts for Iteration 05+.

### Scenario 4: Branching Detected (Slingshot)
Aging is NOT linear → Some tissues follow "disc-like" trajectory, others "lung-like". Update modeling to multi-trajectory framework.

## Clinical Translation

- **Precision medicine:** Use best pseudo-time method to predict individual aging trajectories
- **Intervention timing:** Identify "critical windows" (transitions) for maximal therapeutic effect
- **Longitudinal monitoring:** If prospective R²>0.60 → deploy models clinically for aging risk prediction

## Dataset

**Primary:**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**External (if found):**
- Save to: `external_longitudinal_datasets/` within workspace
- Required columns: Subject_ID, Timepoint, Protein_ID, Abundance

## References

1. H09 Results: `/iterations/iteration_03/hypothesis_09_temporal_rnn_trajectories/{claude_code,codex}/90_results_{agent}.md`
2. H03 Velocity Results: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/{claude_code,codex}/90_results_{agent}.md`
3. ADVANCED_ML_REQUIREMENTS.md
4. Haghverdi et al. (2016). "Diffusion pseudotime robustly reconstructs lineage branching." Nature Methods.
5. Street et al. (2018). "Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics." BMC Genomics.
6. Saelens et al. (2019). "A comparison of single-cell trajectory inference methods." Nature Biotechnology. (Benchmarking framework)
