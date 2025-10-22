# H16 – External Validation COMPLETION (H13 Continuation)

## Scientific Question
Do the top discoveries from Iterations 01-04 (H08 S100 pathway R²=0.75-0.81, H06 biomarker panel AUC=1.0, H03 tissue velocities 4.2× range) replicate on INDEPENDENT external datasets, validating robustness or exposing single-dataset overfitting?

## Background & Rationale

**CRITICAL GAP FROM H13:**
- **Claude agent:** Found 6 independent datasets but **FAILED TO ANALYZE** them
- **Codex agent:** Completely failed (no results)
- **Risk:** ALL 15 hypotheses trained on SAME merged_ecm_aging_zscore.csv → overfitting undetected
- **FDA requirement:** Multi-cohort validation mandatory for biomarker approval

**This is THE MOST IMPORTANT hypothesis for Iteration 05:**
Without external validation, H01-H15 conclusions remain unverified. This hypothesis determines which discoveries are ROBUST vs SPURIOUS.

**H13 Identified 6 Datasets (Claude):**
Claude found these datasets but stopped before analysis. Your job: COMPLETE the validation!

## Objectives

### Primary Objective
Transfer H08 S100 models, H06 biomarker panel, and H03 tissue velocities to EXTERNAL datasets (already identified by H13 Claude) WITHOUT retraining, measuring performance drop to assess overfitting.

### Secondary Objectives
1. Download and preprocess all 6 external datasets identified by H13 Claude
2. Test H08 S100→stiffness models (both Claude R²=0.81 and Codex R²=0.75 versions)
3. Test H06 biomarker panel (8-protein classifier)
4. Compare H03 tissue velocity rankings
5. Meta-analysis across old + new data (I² heterogeneity)
6. Identify STABLE proteins (consistent across cohorts) vs VARIABLE proteins

## Hypotheses to Test

### H16.1: S100 Model Generalization
H08 S100→stiffness models maintain external R²≥0.60 (allowable drop: -0.15 to -0.21 from training 0.75-0.81).

### H16.2: Biomarker Panel Robustness
H06 8-protein panel achieves external AUC≥0.80 (down from training AUC=1.0, acceptable generalization).

### H16.3: Velocity Ranking Consistency
External tissue velocities correlate ρ>0.70 with H03 internal rankings (Lung fast, Muscle slow).

### H16.4: Low Heterogeneity
Meta-analysis I²<50% for top 20 aging proteins → consistent effects across cohorts.

## Required Analyses

### 1. LOCATE H13 CLAUDE IDENTIFIED DATASETS

**Claude H13 output location:**
`/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code/90_results_claude_code.md`

**Your first task:**
- Read this file to extract the 6 dataset IDs/names Claude found
- If dataset download links provided → use them
- If only dataset IDs → search PRIDE/GEO/ProteomeXchange with those IDs

**Expected datasets (based on H13 task):**
- PRIDE accessions (PXD...)
- GEO series (GSE...)
- ProteomeXchange IDs

### 2. DOWNLOAD & PREPROCESS EXTERNAL DATASETS

**For EACH of the 6 datasets:**

```python
# Download
# Use wget/curl if direct links, or API if PRIDE/GEO

# Preprocessing pipeline (MATCH our methods exactly)
from universal_zscore_function import calculate_zscore

external_data = pd.read_csv("external_raw.csv")

# Normalize SAME way as merged_ecm_aging_zscore.csv
external_zscore = calculate_zscore(
    external_data,
    group_by=['Tissue', 'Compartment'],  # IF metadata available
    value_col='Abundance'
)

# Gene symbol mapping
from uniprot_api import map_gene_symbols
external_zscore['Gene_Symbol'] = map_gene_symbols(external_zscore['Protein_ID'])

# Overlap analysis
overlap = set(external_zscore['Gene_Symbol']) & set(our_proteins)
overlap_percent = len(overlap) / len(our_proteins) * 100
print(f"Overlap: {overlap_percent:.1f}%")  # Target: ≥70%
```

**Save preprocessed data:**
`external_datasets/dataset_{1-6}/processed_zscore.csv`

### 3. TRANSFER LEARNING: H08 S100 MODELS

**Load pre-trained models:**
```python
import torch

# Claude model
claude_model = torch.load(
    '/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/s100_stiffness_model_claude_code.pth'
)

# Codex model
codex_model = torch.load(
    '/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/codex/s100_stiffness_model_codex.pth'
)
```

**Test on EACH external dataset (NO retraining!):**
```python
# Extract S100 features (20 genes from H08)
s100_genes = ['S100A8', 'S100A9', 'S100B', 'S100A1', 'S100A4', 'S100A6', 'S100A10', ...]
X_external = external_data[s100_genes]  # Only genes present in external data

# Predict stiffness
claude_model.eval()
y_pred_claude = claude_model(X_external)

codex_model.eval()
y_pred_codex = codex_model(X_external)

# Ground truth stiffness (if LOX/TGM/collagens available in external)
stiffness_external = 0.5*external['LOX'] + 0.3*external['TGM2'] + 0.2*(external['COL1A1']/external['COL3A1'])

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error

r2_claude_external = r2_score(stiffness_external, y_pred_claude)
r2_codex_external = r2_score(stiffness_external, y_pred_codex)

# Compare to training performance
print(f"Claude: Train R²=0.81 → External R²={r2_claude_external:.3f} (Δ={r2_claude_external-0.81:.3f})")
print(f"Codex:  Train R²=0.75 → External R²={r2_codex_external:.3f} (Δ={r2_codex_external-0.75:.3f})")
```

**Success criteria:**
- External R²≥0.60 → VALIDATED
- External R²=0.30-0.60 → PARTIAL (moderate overfitting)
- External R²<0.30 → FAILED (severe overfitting)

### 4. TRANSFER LEARNING: H06 BIOMARKER PANEL

**Panel from H06:**
F13B, SERPINF1, S100A9, FSTL1, GAS6, CTSA, COL1A1, BGN

**Classification task:**
```python
# Create labels (if velocity proxy available in external data)
# Fast aging (top tertile velocity) vs Slow aging (bottom tertile)

X_external = external_data[biomarker_panel]  # 8 proteins
y_external_labels = ...  # Derive from external metadata or compute velocity

# Load H06 model (RandomForest from Codex)
from joblib import load
rf_h06 = load('/iterations/iteration_02/hypothesis_06_biomarker_panel/codex/biomarker_classifier.pkl')

# Predict
y_pred_proba = rf_h06.predict_proba(X_external)[:, 1]

# Metrics
from sklearn.metrics import roc_auc_score, roc_curve
auc_external = roc_auc_score(y_external_labels, y_pred_proba)

print(f"H06: Train AUC=1.0 → External AUC={auc_external:.3f} (Δ={auc_external-1.0:.3f})")
```

**Success criteria:**
- AUC≥0.85 → EXCELLENT generalization
- AUC=0.75-0.85 → GOOD (acceptable drop from perfect 1.0)
- AUC<0.75 → POOR (overfitting)

### 5. H03 TISSUE VELOCITY CROSS-VALIDATION

**H03 internal velocities:**
```python
h03_velocities = {
    'Lung': 4.29,
    'Tubulointerstitial': 3.45,
    'Liver': 1.34,
    'Skeletal_muscle': 1.02,
    # ... all 17 tissues
}
```

**Compute external velocities:**
```python
# For each tissue in external data
for tissue in external_tissues:
    tissue_data = external_data[external_data['Tissue'] == tissue]

    # Velocity = mean |ΔZ| (same as H03 method)
    velocity_external = tissue_data['Z_score'].abs().mean()

    external_velocities[tissue] = velocity_external
```

**Correlation analysis:**
```python
# Match tissues (e.g., external 'lung' vs our 'Lung')
matched_tissues = []
our_v = []
ext_v = []

for tissue in matched_tissues:
    our_v.append(h03_velocities[tissue])
    ext_v.append(external_velocities[tissue])

# Correlation
from scipy.stats import spearmanr
rho, p = spearmanr(our_v, ext_v)

print(f"H03 Velocity Correlation: ρ={rho:.3f}, p={p:.4f}")
```

**Success criteria:**
- ρ>0.75 → STRONG consistency
- ρ=0.50-0.75 → MODERATE
- ρ<0.50 → WEAK (tissue ranking unstable)

### 6. META-ANALYSIS (COMBINE OLD + NEW DATA)

**For top 20 proteins (from H06, H08, H03):**
```python
from statsmodels.stats.meta_analysis import combine_effects

proteins_to_test = ['F13B', 'S100A9', 'S100A10', 'FSTL1', 'GAS6', 'LOX', 'TGM2', ...]

for protein in proteins_to_test:
    # Our study effect
    delta_our = mean(ΔZ[protein])  # Aging effect size
    se_our = std(ΔZ[protein]) / sqrt(n_our)

    # External study effect (across 6 datasets)
    delta_ext_list = []
    se_ext_list = []

    for dataset in external_datasets:
        delta_ext = mean(dataset[protein + '_ΔZ'])
        se_ext = std(dataset[protein + '_ΔZ']) / sqrt(n_ext)
        delta_ext_list.append(delta_ext)
        se_ext_list.append(se_ext)

    # Random-effects meta-analysis
    combined_delta, combined_se, I2, Q, p_heterogeneity = combine_effects(
        [delta_our] + delta_ext_list,
        [se_our] + se_ext_list
    )

    results.append({
        'Protein': protein,
        'Delta_combined': combined_delta,
        'I2_heterogeneity': I2,
        'p_heterogeneity': p_heterogeneity
    })
```

**I² interpretation:**
- I²<25%: Low heterogeneity (consistent across studies) → STABLE protein
- I²=25-50%: Moderate
- I²>50%: High (study-specific effects) → VARIABLE protein

**Success criteria:**
- ≥15/20 proteins with I²<50% → findings are ROBUST

### 7. STABLE vs VARIABLE PROTEIN CLASSIFICATION

**Criteria for STABLE proteins:**
1. Same direction (+ or -) in our data AND all external datasets
2. Magnitude within ±0.5 ΔZ units
3. I²<30% heterogeneity
4. Meta-analysis p<0.05 combined effect

**Output table:**
```csv
Protein, Delta_our, Delta_ext_mean, Direction_match, Magnitude_ratio, I2, Classification
F13B, 2.1, 1.9, TRUE, 0.90, 12%, STABLE
S100A9, 1.5, 1.7, TRUE, 1.13, 8%, STABLE
SERPINC1, -0.8, 0.3, FALSE, -, 85%, VARIABLE
...
```

**Clinical implication:**
- **STABLE proteins** → prioritize for biomarker development (robust across cohorts)
- **VARIABLE proteins** → investigate context-dependence (tissue, technique, species)

## Deliverables

### Code & Models
- `h13_completion_{agent}.py` — main validation script
- `dataset_download_{agent}.py` — automated download from PRIDE/GEO
- `transfer_h08_{agent}.py` — S100 model testing
- `transfer_h06_{agent}.py` — biomarker panel testing
- `meta_analysis_{agent}.py` — combine old + new data

### Data Tables
- `external_datasets_summary_{agent}.csv` — Metadata for all 6 datasets (n, overlap %, techniques)
- `h08_external_validation_{agent}.csv` — S100 model R²/MAE on each external dataset
- `h06_external_validation_{agent}.csv` — Biomarker panel AUC/accuracy on external data
- `h03_velocity_correlation_{agent}.csv` — Our vs external tissue velocities
- `meta_analysis_results_{agent}.csv` — Combined effect sizes, I², p for top 20 proteins
- `stable_proteins_{agent}.csv` — Classification: STABLE vs VARIABLE
- `validation_summary_{agent}.csv` — Overall pass/fail for H08/H06/H03

### Visualizations
- `visualizations_{agent}/h08_transfer_scatter_{agent}.png` — Predicted vs actual stiffness (external)
- `visualizations_{agent}/h06_roc_external_{agent}.png` — ROC curves on external data
- `visualizations_{agent}/velocity_correlation_{agent}.png` — Our vs external velocities
- `visualizations_{agent}/meta_forest_plot_{agent}.png` — Forest plot for top 10 proteins
- `visualizations_{agent}/heterogeneity_heatmap_{agent}.png` — I² heatmap (proteins × datasets)
- `visualizations_{agent}/dataset_venn_{agent}.png` — Gene overlap across datasets

### Report
- `90_results_{agent}.md` — CRITICAL findings:
  - **H08 validation:** Does S100 pathway replicate? (R² external ≥ 0.60?)
  - **H06 validation:** Does biomarker panel replicate? (AUC external ≥ 0.80?)
  - **H03 validation:** Are tissue velocities consistent? (ρ > 0.70?)
  - **Meta-analysis:** Which proteins are STABLE (I²<50%)?
  - **OVERALL VERDICT:** Are Iterations 01-04 findings ROBUST or OVERFIT?
  - **Recommendations:** Which hypotheses need re-evaluation? Which are FDA-ready?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| External datasets downloaded | 6/6 (from H13 Claude) | Dataset retrieval |
| Gene overlap (mean) | ≥70% | UniProt mapping |
| H08 external R² (Claude model) | ≥0.60 | Transfer learning |
| H08 external R² (Codex model) | ≥0.60 | Transfer learning |
| H06 external AUC | ≥0.80 | Transfer learning |
| H03 velocity correlation | ρ>0.70 | Cross-cohort comparison |
| Stable proteins (I²<50%) | ≥15/20 | Meta-analysis |
| Overall validation | PASS | ≥3/4 metrics above thresholds |

## Expected Outcomes

### Scenario 1: STRONG VALIDATION (Best Case)
- H08 R² external ≥0.65 → S100 pathway ROBUST
- H06 AUC external ≥0.85 → Biomarkers FDA-ready
- H03 ρ>0.75 → Tissue ranking universal
- I²<40% for most proteins → Consistent aging signatures
- **Action:** Publish, proceed to clinical trials

### Scenario 2: MODERATE VALIDATION
- H08 R²=0.50-0.60 → S100 holds but weaker
- H06 AUC=0.75-0.80 → Biomarkers useful but imperfect
- Some proteins stable (I²<30%), others variable (I²>70%)
- **Action:** Focus on STABLE proteins only, investigate context-dependence

### Scenario 3: POOR VALIDATION (Failure)
- H08 R²<0.40 → S100 model overfit
- H06 AUC<0.70 → Biomarkers fail external test
- I²>60% → High heterogeneity
- **Action:** Re-evaluate ALL H01-H15, mandatory external validation for future work

### Scenario 4: NO DATASETS ACCESSIBLE
- H13 Claude identified datasets but links broken / paywalled
- **Action:** Search for alternative datasets, contact study authors, or generate new data

## Dataset

**Primary (for comparison):**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**External (to be downloaded):**
- Datasets identified by H13 Claude (read from H13 results file)
- Save to: `external_datasets/dataset_{1-6}/`

## References

1. H13 Claude Results: `/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code/90_results_claude_code.md`
2. H08 S100 Models: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/{claude_code,codex}/s100_stiffness_model_{agent}.pth`
3. H06 Biomarker Panel: `/iterations/iteration_02/hypothesis_06_biomarker_panel/codex/biomarker_classifier.pkl`
4. H03 Tissue Velocities: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/`
5. PRIDE: https://www.ebi.ac.uk/pride/
6. GEO: https://www.ncbi.nlm.nih.gov/geo/
7. ProteomeXchange: http://www.proteomexchange.org/
