# H13 – Independent Dataset Validation: Testing Generalizability

## Scientific Question
Do the strongest findings from Iterations 01-03 (H08 S100 pathway, H06 biomarker panel, H03 tissue velocities) replicate on INDEPENDENT external datasets, or are they overfit to our merged ECM aging database?

## Background & Rationale

**Critical Risk: Single-Dataset Overfitting**
- ALL hypotheses H01-H12 trained and tested on `/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Even with train/test splits → data from same 13 studies, same processing pipeline
- High R² and AUC may reflect study-specific artifacts, not biological signal

**Gold Standard: External Validation**
- Find NEW proteomics data (NOT in our 13 studies)
- Test models WITHOUT retraining
- If performance holds → findings are robust
- If performance drops → findings are dataset-specific

**High-Value Targets for Validation:**
1. **H08 S100→stiffness model:** R²=0.81 (Claude), 0.75 (Codex) — STRONGEST SIGNAL
2. **H06 Biomarker panel:** F13B, S100A9, FSTL1, GAS6 — AUC=1.0 (needs external test)
3. **H03 Tissue velocities:** Do external tissues follow same fast/slow ranking?

## Objectives

### Primary Objective
Obtain ≥2 independent ECM aging proteomics datasets (NOT in our 13 studies) and test transfer learning performance of H08, H06, H03 models without retraining.

### Secondary Objectives
1. **Comprehensive dataset search:** Systematically query all proteomics repositories (PRIDE, ProteomeXchange, GEO, MassIVE)
2. **Meta-analysis:** Combine old + new data, test heterogeneity (I² statistic)
3. **Cross-cohort comparison:** Identify proteins with stable vs variable aging effects across studies
4. **Update merged database:** Add validated external datasets to canonical ECM aging resource

## Hypotheses to Test

### H13.1: S100 Pathway Replication
H08 S100→stiffness model maintains R²>0.60 on external data (allowable drop: -0.15 from training).

### H13.2: Biomarker Panel Generalization
H06 F13B/S100A9/FSTL1/GAS6 panel achieves AUC>0.80 on external fast/slow aging classification.

### H13.3: Tissue Velocity Consistency
External tissues cluster with matched tissue types from H03 (e.g., external lung ~ our lung velocity).

### H13.4: Low Heterogeneity
Meta-analysis I²<50% for top aging proteins → consistent effects across cohorts.

## Required Analyses

### 1. COMPREHENSIVE DATASET SEARCH (MANDATORY - HIGHEST PRIORITY)

#### Repository Queries

**PRIDE (Proteomics Identifications Database):**
```
Search terms:
- "aging" + "extracellular matrix"
- "aging" + "tissue" + "proteomics"
- "senescence" + "collagen"
- "fibrosis" + "proteome"

Filters:
- Species: Homo sapiens (OR Mus musculus if human scarce)
- Techniques: LFQ, TMT, iTRAQ, SILAC
- Publication year: 2018-2025 (recent, likely NOT in our 13 studies)
```

**ProteomeXchange:**
```
Advanced search:
- Keywords: "aging", "ECM", "matrix", "fibrosis", "tissue"
- NOT in our dataset accessions: [list 13 study IDs]
- Download: ≥2 datasets with ≥50 ECM proteins overlap
```

**GEO (for proteomics/transcriptomics):**
```
DataSets filter: "proteomics" OR "protein abundance"
Organism: "Homo sapiens"
Study type: Expression profiling by mass spectrometry
Keywords: "aging", "elderly", "old vs young"
```

**MassIVE (UCSD):**
```
Search: "aging human tissue"
Download: .mzTab or .txt quantification files
Priority: Multi-tissue studies (like ours)
```

**Published cohorts (Literature search):**
```
PubMed: "(aging proteomics) AND (tissue OR ECM OR extracellular matrix)"
Filter: 2020-2025, Full text available
Extract: Supplementary data links, raw data repositories
```

#### Inclusion Criteria

**MUST HAVE:**
- Quantitative proteomics (abundance values, not just IDs)
- Age comparison (young vs old, OR age as continuous variable)
- Human OR mouse tissues (compatible with our dataset)
- ≥50 proteins overlap with our 648 ECM-centric genes

**PREFERRED:**
- Multiple tissues (for H03 velocity validation)
- Longitudinal (for H09/H11 temporal validation)
- Published in high-impact journals (Nature, Cell, Science, eLife)

**EXCLUSION:**
- Cell lines (not tissue)
- Plasma/serum only (prefer solid tissue for ECM focus)
- Already in our 13 studies (check Study_ID)

#### Download & Documentation

For each dataset:
```
1. Download raw quantification tables
2. Record metadata:
   - Study ID (ProteomeXchange PXD, PRIDE accession)
   - Publication (PMID, DOI)
   - Sample size (n tissues, n age groups)
   - Technique (LFQ, TMT, etc.)
   - Tissue types
3. Save to: external_datasets/dataset_X/
4. Create: external_datasets/dataset_X/metadata.json
```

**Success criteria:**
- ≥2 independent datasets downloaded
- ≥1 multi-tissue dataset (for H03 validation)
- ≥1 with S100 family proteins (for H08 validation)

### 2. DATA PREPROCESSING & HARMONIZATION

**Apply universal z-score function:**
```python
# Use same normalization as merged_ecm_aging_zscore.csv
from universal_zscore_function import calculate_zscore

external_zscore = calculate_zscore(
    external_data,
    group_by=['Tissue', 'Compartment'],  # match our method
    value_col='Abundance'
)
```

**Gene symbol matching:**
- Use UniProt API to map external IDs → our Gene_Symbol
- Handle: Isoforms, synonyms, outdated symbols
- Report: % overlap (target: ≥70% of our 648 ECM genes)

**Delta calculation (if age groups available):**
```python
delta_Z = Z_old - Z_young  # match our ΔZ metric
```

**Output:**
- `external_datasets/dataset_X/processed_zscore.csv` (same schema as our main file)

### 3. TRANSFER LEARNING: H08 S100 MODEL

**Model source:**
- `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/{claude_code,codex}/s100_stiffness_model_{agent}.pth`

**Test WITHOUT retraining:**
```python
import torch

# Load pre-trained model
model = torch.load('s100_stiffness_model_claude_code.pth')
model.eval()

# Extract S100 features from external data
X_external = external_data[s100_genes]  # 20-dim

# Predict stiffness
y_pred = model(X_external)

# Compare to stiffness proxy (if LOX/TGM/collagens available)
stiffness_external = 0.5*LOX + 0.3*TGM2 + 0.2*(COL1A1/COL3A1)

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error
r2_external = r2_score(stiffness_external, y_pred)
mae_external = mean_absolute_error(stiffness_external, y_pred)
```

**Benchmarks:**
- Training R²: 0.81 (Claude), 0.75 (Codex)
- Acceptable drop: -0.15 (R² external ≥ 0.60)
- Catastrophic drop: R² < 0.30 → overfitting confirmed

**Repeat for both agents:**
- Claude Code model → test on external
- Codex model → test on external
- Compare: Which agent generalizes better?

### 4. TRANSFER LEARNING: H06 BIOMARKER PANEL

**Panel:** F13B, SERPINF1, S100A9, FSTL1, GAS6, CTSA, COL1A1, BGN (top 8 from H06)

**Classification task:**
```python
# Create fast/slow labels for external tissues (if velocity proxy available)
# OR use age groups (old = fast, young = slow)

from sklearn.ensemble import RandomForestClassifier

# Load H06 trained model (if saved) OR retrain on our data
rf = RandomForestClassifier()  # from H06
X_external = external_data[biomarker_panel]
y_external_pred = rf.predict_proba(X_external)[:, 1]

# Metrics
from sklearn.metrics import roc_auc_score
auc_external = roc_auc_score(y_external_true, y_external_pred)
```

**Benchmarks:**
- Training AUC: 1.0 (H06 Codex)
- Realistic external AUC: 0.80-0.90
- Failure: AUC < 0.70

### 5. CROSS-COHORT TISSUE VELOCITY (H03 VALIDATION)

**Goal:** Do external tissues have similar aging velocities to our tissues?

**Method:**
```python
# For each external tissue type
external_lung_velocity = compute_velocity(external_lung_data)  # mean |ΔZ|
our_lung_velocity = 4.29  # from H03

# Correlation across matched tissues
matched_tissues = ['Lung', 'Liver', 'Heart', ...]
external_velocities = [...]
our_velocities = [4.29, 1.34, 1.58, ...]

from scipy.stats import spearmanr
rho, p = spearmanr(external_velocities, our_velocities)
```

**Success criteria:**
- Spearman ρ > 0.70 (strong consistency)
- p < 0.05

**Visualization:**
- Scatter plot: Our velocity (x) vs External velocity (y)
- Identity line (y=x)
- Color by tissue type

### 6. META-ANALYSIS (COMBINE OLD + NEW DATA)

**For top aging proteins (from H03, H06, H08):**
```python
# Per protein, per tissue
effect_sizes = []
for protein in top_proteins:
    # Our study
    delta_our = mean(ΔZ[protein, tissue])
    se_our = std(ΔZ[protein, tissue]) / sqrt(n)

    # External study
    delta_ext = mean(ΔZ_ext[protein, tissue])
    se_ext = std(ΔZ_ext[protein, tissue]) / sqrt(n_ext)

    # Fixed-effect meta-analysis
    from statsmodels.stats.meta_analysis import combine_effects
    combined_delta, combined_se, I2 = combine_effects([delta_our, delta_ext], ...)

    effect_sizes.append({
        'Protein': protein,
        'Delta_combined': combined_delta,
        'I2_heterogeneity': I2
    })
```

**Interpret I² (heterogeneity):**
- I² < 25%: Low (consistent across studies)
- I² 25-50%: Moderate
- I² > 50%: High (study-specific effects)

**Success criteria:**
- Top 20 proteins: I² < 50% (stable aging signatures)

### 7. CROSS-COHORT PROTEIN STABILITY

**Identify proteins with STABLE vs VARIABLE effects:**

**Stable proteins:**
- Same direction (+ or -) in our data and external data
- Similar magnitude (ΔZ within ±0.5)
- Low heterogeneity (I² < 30%)

**Variable proteins:**
- Opposite direction in external data
- OR magnitude differs >2-fold
- High I² (>70%)

**Output:**
```csv
Protein, Delta_our, Delta_external, Direction_match, Magnitude_ratio, I2, Classification
F13B, 2.1, 1.9, TRUE, 0.90, 12%, STABLE
S100A9, 1.5, 1.7, TRUE, 1.13, 8%, STABLE
SERPINC1, -0.8, 0.3, FALSE, -, 85%, VARIABLE
...
```

**Recommendation:**
- **STABLE proteins** → prioritize for clinical translation (robust across cohorts)
- **VARIABLE proteins** → investigate context-dependence (tissue, species, technique)

### 8. UPDATE CANONICAL DATABASE (IF VALIDATION SUCCESSFUL)

**If external datasets pass validation:**
- Merge into `/08_merged_ecm_dataset/merged_ecm_aging_zscore_v2.csv`
- Add Study_ID for new datasets
- Recompute cross-study percentiles
- Update metadata: `dataset_metadata_v2.json`

**Version control:**
- v1: Original 13 studies (current)
- v2: +2-3 validated external studies (Iteration 04)

**Documentation:**
- Update `04_compilation_of_papers/00_README_compilation.md`
- Add new papers to bibliography
- Note: External validation performed, results in H13

## Deliverables

### Code & Models
- `dataset_search_{agent}.py` — automated PRIDE/ProteomeXchange queries
- `data_harmonization_{agent}.py` — external data preprocessing
- `transfer_learning_h08_{agent}.py` — S100 model validation
- `transfer_learning_h06_{agent}.py` — biomarker panel validation
- `meta_analysis_{agent}.py` — combine old + new data

### Data Tables
- `external_datasets_summary_{agent}.csv` — Metadata for all found datasets
- `h08_external_validation_{agent}.csv` — R², MAE on external data (S100 model)
- `h06_external_validation_{agent}.csv` — AUC on external data (biomarker panel)
- `h03_velocity_comparison_{agent}.csv` — Our vs external tissue velocities
- `meta_analysis_results_{agent}.csv` — Combined effect sizes, I² for top proteins
- `protein_stability_{agent}.csv` — Stable vs variable protein classification
- `gene_overlap_{agent}.csv` — % overlap with our 648 ECM genes per dataset

### Visualizations
- `visualizations_{agent}/dataset_venn_{agent}.png` — Overlap: our genes ∩ external genes
- `visualizations_{agent}/h08_transfer_scatter_{agent}.png` — Predicted vs actual stiffness (external)
- `visualizations_{agent}/h06_roc_external_{agent}.png` — ROC curve (external data)
- `visualizations_{agent}/velocity_correlation_{agent}.png` — Our vs external velocities
- `visualizations_{agent}/meta_forest_plot_{agent}.png` — Forest plot for top 10 proteins
- `visualizations_{agent}/heterogeneity_heatmap_{agent}.png` — I² for all proteins × tissues

### External Datasets
- `external_datasets/dataset_1/` (raw + processed + metadata)
- `external_datasets/dataset_2/`
- `external_datasets/merged_external_zscore.csv` (combined external data)

### Report
- `90_results_{agent}.md` — comprehensive findings with:
  - Dataset search summary (how many found, sources)
  - Preprocessing steps (overlap %, normalization)
  - H08 transfer learning (does S100 model generalize?)
  - H06 transfer learning (does biomarker panel generalize?)
  - H03 velocity replication (consistent tissue ranking?)
  - Meta-analysis (I², stable proteins)
  - **CRITICAL:** Are Iteration 01-03 findings ROBUST or OVERFIT?
  - **RECOMMENDATION:** Which hypotheses are validated, which need re-evaluation?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Independent datasets found | ≥2 | PRIDE/ProteomeXchange search |
| Gene overlap | ≥70% | UniProt mapping |
| H08 external R² (S100 model) | ≥0.60 | Transfer learning |
| H06 external AUC (biomarkers) | ≥0.80 | Transfer learning |
| H03 velocity correlation | ρ>0.70 | Cross-cohort comparison |
| Meta-analysis I² (top 20 proteins) | <50% | Heterogeneity test |
| Stable proteins identified | ≥15/20 | Cross-cohort consistency |

## Expected Outcomes

### Scenario 1: Strong Validation (Best Case)
- H08 R² external ≥ 0.65 → S100 pathway CONFIRMED as robust
- H06 AUC external ≥ 0.85 → Biomarker panel validated for clinical use
- H03 velocities correlate ρ>0.75 → Tissue ranking universal
- I² < 40% → Aging signatures stable across cohorts
- **Action:** Publish findings, proceed to clinical trials

### Scenario 2: Moderate Validation
- H08 R² = 0.50-0.60 → S100 pathway holds but weaker
- H06 AUC = 0.70-0.80 → Biomarkers useful but not perfect
- Some proteins stable (I²<30%), others variable (I²>70%)
- **Action:** Focus on STABLE proteins only, investigate context-dependence of variable ones

### Scenario 3: Poor Validation (Failure)
- H08 R² < 0.40 → S100 model overfit to our dataset
- H06 AUC < 0.70 → Biomarkers fail external test
- I² > 60% for most proteins → High heterogeneity
- **Action:** Re-evaluate all hypotheses, require external validation for ALL future work

### Scenario 4: No Datasets Found
- Comprehensive search yields <2 usable datasets
- **Action:** Collect NEW data (collaboration, funding for cohort), OR wait for future publications

## Clinical Translation

**If validation successful:**
- FDA submission requires multi-cohort validation → H13 provides this evidence
- Biomarker assay (F13B, S100A9) can cite independent replication
- S100→stiffness model deployable in clinical decision support

**If validation fails:**
- Recognize limitations, avoid premature clinical translation
- Refocus on mechanistic understanding (not predictive models)

## Dataset

**Primary (for comparison):**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**External (to be found):**
- PRIDE, ProteomeXchange, GEO, MassIVE repositories
- Target: 2-5 independent cohorts

## References

1. H08 Results: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/{claude_code,codex}/90_results_{agent}.md`
2. H06 Results: `/iterations/iteration_02/hypothesis_06_biomarker_discovery_panel/{codex}/90_results_codex.md`
3. H03 Results: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/{claude_code,codex}/`
4. ADVANCED_ML_REQUIREMENTS.md
5. ProteomeXchange: http://www.proteomexchange.org/
6. PRIDE: https://www.ebi.ac.uk/pride/
7. GEO: https://www.ncbi.nlm.nih.gov/geo/
8. MassIVE: https://massive.ucsd.edu/
9. Higgins et al. (2003). "Measuring inconsistency in meta-analyses." BMJ. (I² statistic)
