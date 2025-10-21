# Batch Correction Task: Fix Statistical Analysis Pipeline

**Created:** 2025-10-18  
**Priority:** CRITICAL  
**Context:** Statistical validation revealed that current z-score calculations are unreliable due to non-normal distribution and mixed data scales.

---

## Problem Statement

Current z-score calculations assume normal distribution, but many datasets have **non-normal distributions**. Simple z-score formula (mean ± std) **distorts the picture** when applied to non-normal data, leading to:
- **ICC = 0.29** (SEVERE batch effects, should be >0.75)
- **70% unreliable data** according to statistical tests
- Invalid cross-study comparisons

**Root causes:**
1. Mixed data scales (linear, log2, log10) across studies
2. Z-scores applied without checking distribution normality
3. No proper batch correction method selected based on distribution type

---

## Your Task

### Step 1: Verify All Abundance Transformations ✅ 

**Reference:** `/Users/Kravtsovd/projects/ecm-atlas/04_compilation_of_papers/ABUNDANCE_TRANSFORMATIONS_METADATA.md`

For EACH of the 12 studies, verify:
- What transformation was applied in the source data (log2, log10, linear)?
- Cross-reference with original papers in `/Users/Kravtsovd/projects/ecm-atlas/04_compilation_of_papers/`
- Check processing scripts in `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/*/`
- Update ABUNDANCE_TRANSFORMATIONS_METADATA.md if any information is missing

**Success criteria:** Complete table with transformation metadata for all 12 studies

---

### Step 2: Standardize to Unified Scale

**Goal:** All data must be on the SAME scale before batch correction

**Actions:**
1. Apply **reverse transformation** where needed to get back to linear scale
2. Apply **uniform log2 transformation** to all datasets: `log2(x + 1)`
3. Verify all medians are in consistent range (15-30 for log2)

**Reference logic from ABUNDANCE_TRANSFORMATIONS_METADATA.md:**
- Already log2 → Keep as-is
- Already log10 → Convert to log2: `log2_value = log10_value * log2(10)`
- Linear → Apply: `log2(x + 1)`

---

### Step 3: Check Distribution Normality

**For EACH study after standardization:**

Apply **Shapiro-Wilk test** or **Kolmogorov-Smirnov test** to check if distribution is normal:

```python
from scipy.stats import shapiro, kstest, normaltest

# Check normality
stat, p_value = shapiro(abundance_values)
is_normal = p_value > 0.05  # If p > 0.05, distribution is normal
```

**Document findings:**
- Which studies have normal distribution?
- Which studies need non-parametric methods?

---

### Step 4: Apply Appropriate Batch Correction Method

**Based on distribution test results:**

#### If distribution IS normal (p > 0.05):
→ Use **ComBat** (parametric method)
```python
# ComBat works best with normal distribution
# Preserves Age_Group + Tissue_Compartment as biological covariates
```

#### If distribution is NOT normal (p ≤ 0.05):
→ Use **non-parametric methods**:
- Percentile-based normalization
- Quantile normalization
- Rank-based methods

**DO NOT apply simple z-score to non-normal data!**

---

### Step 5: Validate Results

**Success criteria (from ACTION_PLAN_REMEDY.md):**

| Metric | Current | Target |
|--------|---------|--------|
| ICC | 0.29 | >0.50 |
| Driver recovery | 20% | ≥66.7% |
| FDR-significant proteins | 0 | ≥5 |

**If targets NOT met → revise method selection**

---

## Important References

1. **Data scale metadata:** `04_compilation_of_papers/ABUNDANCE_TRANSFORMATIONS_METADATA.md`
2. **Current approach:** `14_exploratory_batch_correction/reports/ACTION_PLAN_REMEDY.md`
3. **Processing scripts:** `05_papers_to_csv/*/` folders
4. **Original papers:** `04_compilation_of_papers/*_comprehensive_analysis.md`

---

## Key Principles

✅ **Check distribution FIRST** → Then select method  
✅ **Unified scale** → All data must be comparable  
✅ **Preserve biology** → Age_Group and Tissue must remain as covariates  
✅ **Non-destructive** → Keep original data, create new batch-corrected version  
✅ **Validate** → Must improve ICC and driver recovery metrics  

❌ **Don't assume normal distribution**  
❌ **Don't mix scales**  
❌ **Don't apply z-score to non-normal data**  

---

## Output Deliverables

1. **Updated ABUNDANCE_TRANSFORMATIONS_METADATA.md** with complete transformation info
2. **Distribution analysis report** for each study (normal vs non-normal)
3. **Batch-corrected dataset:** `merged_ecm_aging_COMBAT_CORRECTED.csv`
4. **Validation report** with ICC, driver recovery, and FDR metrics
5. **Methodology document** explaining which method was used for which study and why

---

**DO NOT ask questions until you have:**
- Verified ALL transformations from source papers
- Checked normality for ALL studies  
- Selected appropriate method for EACH study based on distribution
- Generated validated batch-corrected dataset

**This is for proper statistical analysis to enable valid cross-study comparisons and publishable results.**

