# H13 Independent Dataset Validation - Data Acquisition Plan
**Agent:** claude_code
**Status:** IN PROGRESS
**Priority:** CRITICAL

## Executive Summary

**PROBLEM:** All hypotheses H01-H12 trained on SAME dataset → Risk of overfitting
**SOLUTION:** Obtain ≥2 independent datasets, test models WITHOUT retraining
**GOAL:** Prove findings are ROBUST, not dataset-specific artifacts

---

## 1.0 Dataset Inventory

### 1.1 HIGH PRIORITY - Validated Datasets (Ready for Download)

| Dataset | Repository | Tissue | Age Groups | n | Proteins | Status |
|---------|-----------|--------|------------|---|----------|---------|
| **PXD011967** | PRIDE | Skeletal muscle | 20-34, 35-49, 50-64, 65-79, 80+ | 58 | 4,380 | ✅ VALIDATED |
| **PXD015982** | PRIDE | Skin (3 sites) | Young (26.7), Aged (84.0) | 6 | 229 ECM | ✅ VALIDATED |

### 1.2 MEDIUM PRIORITY - Additional Datasets

| Dataset | Repository | Tissue | Priority | Notes |
|---------|-----------|--------|----------|--------|
| PXD007048 | PRIDE | Bone marrow | MEDIUM | Cell-specific, ECM niche |
| MSV000082958 | MassIVE | Lung | MEDIUM | In vitro fibrosis, collagen PTMs |
| MSV000096508 | MassIVE | Brain | MEDIUM | Mouse model, cognitive aging |

### 1.3 PENDING - High-Impact Multi-Tissue Study

**Cell 2025 Study** (Ding et al., PMID: 40713952):
- **Tissues:** 13 tissues (skin, muscle, heart, aorta, lung, liver, spleen, etc.)
- **Samples:** 516 samples across 5 decades
- **Proteins:** Up to 12,771 proteins
- **DOI:** 10.1016/j.cell.2025.06.047
- **Status:** ⚠️ ACCESSION NUMBER NOT YET FOUND

**ACTION REQUIRED:** Contact corresponding author or check supplementary materials when available

---

## 2.0 Data Acquisition Strategy

### 2.1 PXD011967 - Skeletal Muscle Aging (HIGHEST PRIORITY)

**Why this dataset?**
- **Excellent age stratification:** 5 age bins, 58 subjects
- **Large protein coverage:** 4,380 proteins
- **TMT quantification:** Batch-corrected, high quality
- **Publication:** eLife 2019, well-documented

**Data Access Options:**

**Option A: PRIDE FTP (Raw files - LARGE)**
```bash
ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2019/11/PXD011967/
# Contains: .raw, .mzML files (hundreds of GB)
# ❌ Too large, not necessary for our validation
```

**Option B: Processed Quantification Tables (PREFERRED)**
- **Source:** eLife article supplementary data
- **URL:** https://elifesciences.org/articles/49874
- **Files needed:**
  - `Figure 1—source data 1.xlsx` - 5,891 proteins quantified
  - `Figure 1—source data 3.xlsx` - Age-associated proteins
- **Format:** TMT abundance values per sample
- **Size:** ~few MB

**Option C: Request from Authors**
- **Contact:** Corresponding author (check paper)
- **Request:** Processed abundance matrix (proteins × samples)

**RECOMMENDED ACTION:**
1. Download supplementary Excel files from eLife
2. Extract protein IDs, gene symbols, TMT abundances
3. Map to age groups
4. Calculate z-scores per our method

### 2.2 PXD015982 - Skin Aging ECM (HIGH PRIORITY)

**Why this dataset?**
- **ECM-focused:** 229 matrisome proteins (115 core)
- **Absolute quantification:** QconCAT method
- **Young vs Old:** Clear age comparison
- **Photoaging analysis:** 3 anatomical sites

**Data Access:**

**Option A: PRIDE Web Interface**
```
URL: https://www.ebi.ac.uk/pride/archive/projects/PXD015982
Files: Check for processed abundance tables
```

**Option B: Matrix Biology Plus Supplementary**
- **Paper:** PMC7852213
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7852213/
- **Supplementary Tables:** Should contain ECM protein abundances

**RECOMMENDED ACTION:**
1. Access PRIDE project page
2. Download processed files (look for .txt, .csv, .xlsx)
3. If not available → download from PMC supplementary
4. Extract young vs aged abundances for ECM proteins

---

## 3.0 Data Processing Pipeline

### 3.1 Harmonization to Our Format

**Target Schema:**
```
Protein_ID | Gene_Symbol | Tissue | Species | Age | Abundance | Z_score | Study_ID
-----------|-------------|--------|---------|-----|-----------|---------|----------
P12345     | COL1A1      | Muscle | Human   | 25  | 1000      | 0.5     | PXD011967
```

**Processing Steps:**
1. **Gene Symbol Mapping**
   - UniProt API for Protein ID → Gene Symbol
   - Handle isoforms, synonyms, deprecated symbols
   - Target: ≥70% overlap with our 648 ECM genes

2. **Z-score Calculation**
   ```python
   # Use same method as our merged dataset
   from universal_zscore_function import calculate_zscore

   Z = (Abundance - mean_per_tissue) / std_per_tissue
   Delta_Z = Z_old - Z_young
   ```

3. **Quality Control**
   - Check for batch effects
   - Verify age group separation
   - Compare distributions to our data

### 3.2 Expected Overlap with Our ECM Gene List

**Our gene list:** 648 ECM-centric genes (from H03, H06, H08)

**Expected overlaps:**
- PXD011967 (muscle): ~200-300 ECM genes (estimated 30-50%)
- PXD015982 (skin): ~150-200 ECM genes (matrisome-focused, ~25-30%)

**Key proteins for validation:**
- **H08 S100 family:** S100A4, S100A6, S100A8, S100A9, S100A10, S100A11 (20 total)
- **H06 biomarkers:** F13B, SERPINF1, S100A9, FSTL1, GAS6, CTSA, COL1A1, BGN
- **H03 fast-aging:** COL1A1, COL3A1, FN1, LOX, TGM2

---

## 4.0 Validation Experiments

### 4.1 H08: S100 → Stiffness Model Transfer Learning

**Hypothesis:** S100 pathway model maintains R² > 0.60 on external data

**Method:**
```python
# Load pre-trained model (from Iteration 03)
model = torch.load('hypothesis_08/.../s100_stiffness_model_claude_code.pth')

# Extract S100 features from PXD011967
s100_genes = ['S100A4', 'S100A6', 'S100A8', 'S100A9', ...] # 20 S100 proteins
X_external = external_muscle_data[s100_genes]

# Predict stiffness proxy
y_pred = model(X_external)

# Calculate stiffness from ECM proteins
stiffness_proxy = 0.5*LOX + 0.3*TGM2 + 0.2*(COL1A1/COL3A1)

# Evaluate
from sklearn.metrics import r2_score
r2_external = r2_score(stiffness_proxy, y_pred)

# Success criteria:
# - Training R²: 0.81 (Claude), 0.75 (Codex)
# - Target R² external: ≥ 0.60 (allowable drop: -0.15)
# - Failure threshold: R² < 0.40
```

**Expected Challenges:**
- LOX, TGM2 may not be in skeletal muscle dataset
- Alternative: use collagen ratios only
- Tissue difference: muscle vs original training tissues

### 4.2 H06: Biomarker Panel Classification

**Hypothesis:** F13B/S100A9/FSTL1/GAS6 panel achieves AUC > 0.80 on external data

**Method:**
```python
# Load H06 biomarker panel
biomarkers = ['F13B', 'SERPINF1', 'S100A9', 'FSTL1', 'GAS6', 'CTSA', 'COL1A1', 'BGN']

# Create age-based labels (since we don't have velocity for external data)
# Young = slow aging, Old = fast aging
labels_external = (age_group >= 65).astype(int)

# Test classifier
X_external = external_data[biomarkers]
y_pred_proba = rf_classifier.predict_proba(X_external)[:, 1]

# Evaluate
from sklearn.metrics import roc_auc_score
auc_external = roc_auc_score(labels_external, y_pred_proba)

# Success criteria:
# - Training AUC: 1.0 (H06 Codex) - likely overfit
# - Realistic external AUC: 0.80-0.90
# - Failure threshold: AUC < 0.70
```

### 4.3 H03: Tissue Velocity Correlation

**Hypothesis:** External muscle velocity matches our muscle velocity (ρ > 0.70)

**Method:**
```python
# Compute aging velocity for external muscle
velocity_external_muscle = mean(abs(Delta_Z_external))

# Our muscle velocity (from H03)
velocity_our_muscle = 2.5  # example value, check H03 results

# If multiple tissues available in Cell 2025 dataset:
# - Compute velocity for each tissue
# - Correlate with our velocity ranking

from scipy.stats import spearmanr
rho, p = spearmanr(velocities_external, velocities_ours)

# Success: rho > 0.70, p < 0.05
```

### 4.4 Meta-Analysis: I² Heterogeneity

**Hypothesis:** Top aging proteins show I² < 50% (low heterogeneity)

**Method:**
```python
# For each protein in top 20 (from H06, H08)
from statsmodels.stats.meta_analysis import combine_effects

for protein in top_proteins:
    # Our study
    delta_our = mean_delta_z[protein]
    se_our = std_delta_z[protein] / sqrt(n_our)

    # External study
    delta_ext = mean_delta_z_external[protein]
    se_ext = std_delta_z_external[protein] / sqrt(n_ext)

    # Fixed-effect meta-analysis
    combined, se_combined, I2 = combine_effects(
        [delta_our, delta_ext],
        [se_our, se_ext],
        method='fixed'
    )

    # Interpret I²:
    # < 25%: Low heterogeneity (GOOD - consistent)
    # 25-50%: Moderate
    # > 50%: High (BAD - study-specific)
```

---

## 5.0 Success Criteria Summary

| Validation Test | Metric | Target | Source | Interpretation |
|-----------------|--------|--------|--------|----------------|
| H08 S100 model | R² | ≥ 0.60 | Transfer learning | Model generalizes well |
| H06 Biomarkers | AUC | ≥ 0.80 | Transfer learning | Panel validated for clinical use |
| H03 Velocities | Spearman ρ | > 0.70 | Cross-cohort | Tissue ranking universal |
| Meta-analysis | I² | < 50% | Top 20 proteins | Aging signatures stable |

**Scenarios:**

**✅ Scenario 1: Strong Validation (BEST CASE)**
- All targets met → Findings ROBUST → Proceed to clinical translation

**⚠️ Scenario 2: Moderate Validation**
- Some targets met → Focus on STABLE proteins only → Investigate context-dependence

**❌ Scenario 3: Poor Validation (FAILURE)**
- Targets missed → Acknowledge overfitting → Re-evaluate all hypotheses → Require external validation for future work

---

## 6.0 Immediate Next Steps

### Priority 1: Data Download (THIS WEEK)
1. [ ] Download PXD011967 supplementary from eLife
2. [ ] Download PXD015982 data from PRIDE/PMC
3. [ ] Organize in `external_datasets/` folder

### Priority 2: Data Processing (NEXT)
1. [ ] Run harmonization script
2. [ ] Calculate z-scores
3. [ ] Verify gene overlap (target: ≥70%)

### Priority 3: Validation Testing (AFTER DATA READY)
1. [ ] Test H08 S100 model
2. [ ] Test H06 biomarker panel
3. [ ] Compute H03 velocity correlation
4. [ ] Run meta-analysis

### Priority 4: Reporting
1. [ ] Document results in `90_results_claude_code.md`
2. [ ] Create visualizations
3. [ ] Update merged database if validation successful

---

## 7.0 Alternative Strategies (If Data Access Fails)

**Plan B: Simulated Validation**
- Use statistical resampling (bootstrap) on our data
- NOT ideal, but shows robustness to sampling

**Plan C: Request Collaboration**
- Contact authors directly for processed data
- Offer co-authorship on validation paper

**Plan D: Wait for Future Publications**
- Monitor new proteomics papers
- ProteomeXchange auto-alerts for "aging ECM"

---

## 8.0 References

1. **PXD011967 Paper:** Ferri et al. (2019). eLife. DOI: 10.7554/eLife.49874
2. **PXD015982 Paper:** Richter et al. (2021). Matrix Biol Plus. PMID: 33543036
3. **Cell 2025 Paper:** Ding et al. (2025). Cell. PMID: 40713952
4. **PRIDE API:** https://www.ebi.ac.uk/pride/ws/archive/v2/
5. **Our H08 Results:** `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/`
6. **Our H06 Results:** `/iterations/iteration_02/hypothesis_06_biomarker_discovery_panel/`
7. **Our H03 Results:** `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/`

---

**Last Updated:** 2025-10-21
**Next Review:** After data download completion
**Owner:** claude_code agent
