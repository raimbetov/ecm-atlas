# Task: Batch Correction for ECM-Atlas Proteomics Data

**Priority:** CRITICAL
**Context:** Fix statistical analysis pipeline by standardizing data scales and applying appropriate batch correction

---

## Problem Statement

Current z-score calculations are unreliable due to:
- **ICC = 0.29** (SEVERE batch effects, target >0.50)
- Mixed data scales (linear, log2, log10) across 12 studies
- No distribution normality checking before z-score application
- **70% unreliable data** - invalid cross-study comparisons

---

## Your Task

Complete ALL 5 steps below and document your work in Knowledge Framework format.
BE sure to create  a new CSV aretefact in agent fodler!!!!! 

### Step 1: Verify Abundance Transformations ✅ (Already Complete)

**Reference:** `/Users/Kravtsovd/projects/ecm-atlas/04_compilation_of_papers/ABUNDANCE_TRANSFORMATIONS_METADATA.md`

The metadata shows:
- **LINEAR scale:** 4 studies (Randles, Dipali, Ouni, LiDermis) = 5,750 rows (61.5%)
- **LOG2 scale:** 7 studies (Angelidis, Tam, Tsumagari, Schuler, Santinha×3) = 3,550 rows (38%)
- **EXCLUDE:** Caldeira (ratios, incompatible) = 43 rows (0.5%)

### Step 2: Standardize to Unified Log2 Scale

**Implementation:**
```python
# Apply log2(x + 1) transformation to LINEAR studies only:
studies_to_transform = ['Randles_2021', 'Dipali_2023', 'Ouni_2022', 'LiDermis_2021']

# Keep as-is (already LOG2):
studies_keep_asis = ['Angelidis_2019', 'Tam_2020', 'Tsumagari_2023', 'Schuler_2021',
                     'Santinha_2024_Human', 'Santinha_2024_Mouse_DT', 'Santinha_2024_Mouse_NT']

# Exclude from batch correction:
exclude = ['Caldeira_2017']
```

**Validation:**
- Verify all medians in range 15-30 after transformation
- Check global median ~18-22 (uniform log2 scale)

### Step 3: Check Distribution Normality

For EACH study after standardization, apply normality tests:

```python
from scipy.stats import shapiro, normaltest

# Test normality
stat, p_value = shapiro(abundance_values)
is_normal = p_value > 0.05

# Document findings per study
```

### Step 4: Apply Batch Correction Method

**Task Specs:**
- Full task: `/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/BATCH_CORRECTION_TASK.md`

**Based on normality results:**

- **If NORMAL (p > 0.05):** Use ComBat parametric
- **If NON-NORMAL (p ≤ 0.05):** Use quantile normalization or percentile-based methods

**ComBat Implementation:**
```python
# Example using pyComBat or R's sva::ComBat
from combat.pycombat import pycombat

batch_corrected = pycombat(
    data=expr_matrix_log2,  # genes × samples
    batch=batch_labels,      # study IDs
    mod=covariate_matrix     # Age_Group + Tissue_Compartment
)
```

**CRITICAL:** Preserve Age_Group and Tissue_Compartment as biological covariates

### Step 5: Validate Results

**Success Criteria (MUST ACHIEVE):**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **ICC** | 0.29 | **>0.50** | ❌ |
| **Driver recovery** | 20% | **≥66.7%** | ❌ |
| **FDR-significant proteins** | 0 | **≥5** | ❌ |
| **Median (global)** | 1,173 (mixed) | **15-30 (log2)** | ❌ |

**Validation metrics to calculate:**
- Intraclass Correlation Coefficient (ICC)
- Known aging driver recovery rate
- FDR-corrected significant proteins (q < 0.05)

---

## Required Deliverables

**Create these files in YOUR agent folder:**

1. **01_plan_[agent_name].md** - Your approach and progress tracking
2. **batch_correction_pipeline.py** - Complete implementation script
3. **merged_ecm_aging_STANDARDIZED.csv** - Data after log2 standardization
4. **merged_ecm_aging_COMBAT_CORRECTED.csv** - Final batch-corrected dataset
5. **normality_test_results.csv** - Normality test per study (p-values, test stats)
6. **validation_metrics.json** - ICC, driver recovery, FDR counts
7. **90_results_[agent_name].md** - Final report with self-evaluation

**Document Format:** Use Knowledge Framework (see `/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`)
- Thesis (1 sentence) → Overview (1 paragraph) → Mermaid diagrams → MECE sections

---

## Important References

**Data & Metadata:**
- Main DB: `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Metadata: `/Users/Kravtsovd/projects/ecm-atlas/04_compilation_of_papers/ABUNDANCE_TRANSFORMATIONS_METADATA.md`

**Task Specs:**
- Full task: `/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/BATCH_CORRECTION_TASK.md`
- Documentation standards: `/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`

**Scripts to Reference:**
- Processing scripts: `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/*/`
- Z-score function: `/Users/Kravtsovd/projects/ecm-atlas/11_subagent_for_LFQ_ingestion/universal_zscore_function.py`

---

## Self-Evaluation Template (for 90_results_[agent].md)

```markdown
## Self-Evaluation

### Criterion 1: Log2 Standardization Applied
**Status:** ✅/❌/⚠️
**Evidence:** Global median = X.XX (target: 15-30)
**Details:** Applied log2(x+1) to 4 studies (Randles, Dipali, Ouni, LiDermis)

### Criterion 2: Normality Tests Completed
**Status:** ✅/❌/⚠️
**Evidence:** X/12 studies normal (p>0.05)
**Details:** [Attach normality_test_results.csv]

### Criterion 3: Batch Correction Applied
**Status:** ✅/❌/⚠️
**Evidence:** ComBat/Quantile method used
**Details:** Method selection based on normality tests

### Criterion 4: ICC Improved
**Status:** ✅/❌/⚠️
**Evidence:** ICC = X.XX (before: 0.29, target: >0.50)
**Details:** [How measured]

### Criterion 5: Driver Recovery Improved
**Status:** ✅/❌/⚠️
**Evidence:** Driver recovery = XX% (before: 20%, target: ≥66.7%)
**Details:** [Which drivers recovered]

### Criterion 6: FDR-Significant Proteins Found
**Status:** ✅/❌/⚠️
**Evidence:** X proteins (before: 0, target: ≥5)
**Details:** [Top proteins list]

## Overall: X/6 criteria met | Grade: ✅/❌/⚠️
```

---

## Key Principles

✅ **Check distribution FIRST** → Then select method
✅ **Unified scale** → All data must be log2
✅ **Preserve biology** → Age_Group + Tissue as covariates
✅ **Non-destructive** → Keep original, create new files
✅ **Validate** → Must hit target metrics

❌ **Don't assume normal distribution**
❌ **Don't mix scales**
❌ **Don't apply z-score to non-normal data**
❌ **Don't include Caldeira_2017** (ratio data incompatible)

---

**DO NOT ask questions - all information provided above. Execute the full pipeline and document results.**

**Workspace:** All files MUST be created in your agent folder (claude_1/, claude_2/, or codex/)
