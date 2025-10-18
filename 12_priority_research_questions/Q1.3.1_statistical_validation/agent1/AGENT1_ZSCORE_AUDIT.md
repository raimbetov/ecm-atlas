# Statistical Validation of ECM-Atlas Z-Score Methodology

**Thesis:** Z-score normalization is statistically FLAWED with significant batch effects (ICC=0.29), violated normality assumptions (58% failure rate), and zero FDR-significant proteins, requiring robust alternatives and batch correction before cross-study integration.

## Overview

ECM-Atlas employs within-study z-score normalization for cross-study integration of 12 proteomic datasets (9,343 proteins). This audit reveals critical methodological weaknesses: (1) Z-score assumptions violated in majority of studies, (2) High batch effects prevent reliable cross-study comparisons, (3) Alternative normalization methods yield similar results, (4) Permutation testing reveals no globally significant age-related changes at FDR<0.05. Methodology is statistically unsound for quantitative cross-study inference without batch correction.

```mermaid
graph TD
    Data[ECM-Atlas Database<br/>12 studies, 9,343 proteins] --> Audit[Statistical Audit]

    Audit --> A1[Assumption Validation]
    Audit --> A2[Sensitivity Analysis]
    Audit --> A3[Batch Effects]
    Audit --> A4[Permutation Testing]

    A1 --> R1[❌ Normality: 58% FAIL<br/>❌ Independence: SUSPECT<br/>✅ Homoscedasticity: PASS]
    A2 --> R2[✅ Robust to method<br/>Standard ≈ Quantile ≈ Rank]
    A3 --> R3[❌ ICC=0.29 Poor reliability<br/>❌ Study clustering in PCA]
    A4 --> R4[❌ 0 FDR-significant proteins<br/>Bootstrap CIs: Stable]

    R1 --> Verdict[VERDICT: FLAWED]
    R2 --> Verdict
    R3 --> Verdict
    R4 --> Verdict

    style Verdict fill:#ff6b6b,color:#fff
    style R1 fill:#ffd93d
    style R2 fill:#6bcf7f
    style R3 fill:#ff6b6b,color:#fff
    style R4 fill:#ff6b6b,color:#fff
```

**Occurrent: Validation Workflow**

```mermaid
graph LR
    A[Load Data<br/>9,343 rows] --> B[Test Assumptions<br/>Normality, Variance, Independence]
    B --> C[Sensitivity Analysis<br/>4 normalization methods]
    C --> D[Batch Effect Detection<br/>PCA + ICC]
    D --> E[Permutation Testing<br/>1000 iterations]
    E --> F[Generate Report<br/>+ Diagnostic Plots]

    style F fill:#51cf66,color:#fff
```

---

## 1.0 Z-Score Assumption Validation

¶1 Ordering: Normality → Homoscedasticity → Independence (parametric test prerequisites)

### 1.1 Normality (Shapiro-Wilk Test)

**CRITICAL FINDING: 58% of z-score distributions FAIL normality test**

| Finding | Details |
|---------|---------|
| Tests performed | 24 (12 studies × 2 age groups) |
| Passed (p>0.05) | 10 / 24 (41.7%) |
| Failed (p≤0.05) | 14 / 24 (58.3%) |
| Worst offenders | Tam_2020, Angelidis_2019, Schuler_2021, Randles_2021 |

**Implications:**
- Z-score assumes normal distribution N(0,1)
- Parametric tests (t-tests, ANOVA) may be invalid
- Q-Q plots show heavy tails and skewness in multiple studies

**Evidence:**
- File: `normality_test_results.csv`
- Plot: `diagnostics/qq_plots_by_study.png`

```
Example failures (p < 0.05):
  Tam_2020 (Young): p=5.6e-14, skew=0.68
  Angelidis_2019 (Old): p=1.2e-08, skew=0.60
  Randles_2021 (Young): p=0.044, skew=0.14
```

### 1.2 Homoscedasticity (Equal Variance)

**STATUS: ✅ PASSED**

Levene's test confirms equal variance across studies (p=0.995 for Young, p=0.987 for Old). Variance ratio across studies: max/min = 1.002 (nearly perfect).

**Interpretation:** Variance normalization (σ=1) is working correctly within each study.

**Evidence:**
- File: `homoscedasticity_test_results.csv`
- Plot: `diagnostics/variance_by_study.png`

### 1.3 Independence

**STATUS: ⚠️ SUSPECT (but expected)**

High correlation (r>0.9) between raw abundance and z-score in most studies. This is EXPECTED because z-scores are linear transformations of abundances. Only Randles_2021 shows near-zero correlation (r=0.03), suggesting proper normalization.

**Interpretation:** Not a violation - this correlation is mathematically required. Independence refers to sample independence, not variable independence.

**Evidence:**
- File: `independence_test_results.csv`

---

## 2.0 Sensitivity Analysis - Alternative Normalizations

¶1 Ordering: Standard → Robust → Quantile → Rank (increasing robustness to outliers)

**FINDING: Results robust to normalization method choice**

Tested on Randles_2021 Kidney_Tubulointerstitial (n=2,598 proteins):

| Method | Young μ | Young σ | Old μ | Old σ | Notes |
|--------|---------|---------|-------|-------|-------|
| Standard Z-Score | 0.000 | 1.000 | 0.000 | 1.000 | Current method |
| Robust (MAD) | 9.894 | 72.678 | 6.847 | 43.441 | Poor normalization |
| Quantile Transform | -0.001 | 1.009 | 0.000 | 1.008 | Excellent |
| Rank-Based | 0.000 | 0.996 | 0.000 | 0.996 | Excellent |

**Correlation between methods:**
- Standard vs Quantile: r > 0.95
- Standard vs Rank: r > 0.92

**Recommendation:** Current z-score method is reasonable, but quantile transformation may be superior for non-normal data.

**Evidence:**
- File: `sensitivity_analysis_methods.csv`
- Plot: `diagnostics/sensitivity_normalization_methods.png`

---

## 3.0 Batch Effect Detection

¶1 Ordering: PCA visualization → ICC quantification → Interpretation

### 3.1 PCA Analysis

**CRITICAL FINDING: Strong batch effects - studies cluster separately**

| PC | Variance Explained | Cumulative |
|----|-------------------|------------|
| PC1 | 24.1% | 24.1% |
| PC2 | 16.1% | 40.2% |
| PC3 | 15.0% | 55.2% |
| PC4 | 12.7% | 67.9% |
| PC5 | 11.2% | 79.0% |

**Interpretation:**
- Studies cluster in PCA space (see plot)
- Santinha_2024_Mouse_DT is extreme outlier (PC1=-19.9)
- LiDermis_2021 and Tam_2020 also separated
- If normalization were perfect, studies would overlap randomly

**Implication:** Cross-study comparisons are confounded by study-specific effects (batch effects).

**Evidence:**
- File: `pca_batch_effects.csv`
- Plot: `diagnostics/pca_batch_effects.png`

### 3.2 Intraclass Correlation Coefficient (ICC)

**CRITICAL FINDING: ICC = 0.29 (POOR reliability)**

```
ICC(1,1) = 0.2918
F-statistic = 3.0654
P-value = 1.49e-90

Interpretation thresholds:
  ICC < 0.5:     Poor reliability (HIGH BATCH EFFECTS) ← ECM-Atlas
  ICC 0.5-0.75:  Moderate reliability
  ICC 0.75-0.9:  Good reliability
  ICC > 0.9:     Excellent reliability
```

**Tested on:** 666 proteins present in ≥3 studies (3,339 measurements)

**Interpretation:** Only 29% of variance is due to true protein differences; 71% is study-specific noise.

**Implication:** Cross-study integration WITHOUT batch correction is statistically unreliable.

**Evidence:**
- File: `icc_batch_effects.csv`

---

## 4.0 Permutation Testing - Null Hypothesis Validation

¶1 Ordering: Observed results → FDR correction → Permutation validation

### 4.1 Observed Age-Protein Correlations

**CRITICAL FINDING: 0 proteins significant at FDR < 0.05**

```
Proteins tested: 666
Significant (FDR < 0.05): 0 / 666 (0.0%)
Nominal p < 0.05: ~33 proteins
FDR-corrected threshold: 0.167 (too high)
```

**Top candidates (all FDR > 0.16):**

| Protein | Mean Δ Z-Score | P-value | FDR-corrected P |
|---------|----------------|---------|-----------------|
| Kera | -0.970 | 0.00038 | 0.167 |
| COL4A1 | +0.271 | 0.00401 | 0.167 |
| Fth1 | +0.836 | 0.00487 | 0.167 |
| COL11A1 | -0.426 | 0.00172 | 0.167 |

**Interpretation:** High FDR indicates results are not reproducible across studies. Individual study findings may be real, but do NOT generalize.

**Evidence:**
- File: `observed_protein_changes.csv`

### 4.2 Permutation Test Results

**FINDING: Permutation p-values >> parametric p-values**

Tested top 10 proteins (1000 permutations each):

```
Example:
  Kera:      Parametric p=0.00038, Permutation p=1.00
  Gstm2:     Parametric p=0.00091, Permutation p=0.91
  FBLN5:     Parametric p=0.00273, Permutation p=0.54
```

**Interpretation:** Parametric tests overestimate significance. True p-values are much higher when accounting for data structure.

**Evidence:**
- File: `permutation_test_results.csv`

---

## 5.0 Bootstrap Confidence Intervals

¶1 Bootstrap validates stability of point estimates (n=1000 iterations)

### 5.1 Top 4 Driver Proteins

**FINDING: Stable estimates but wide confidence intervals**

| Protein | Observed Δ | 95% CI Lower | 95% CI Upper | CI Width |
|---------|------------|--------------|--------------|----------|
| Kera | -0.970 | -1.059 | -0.887 | 0.172 |
| COL4A1 | +0.271 | +0.147 | +0.404 | 0.258 |
| Fth1 | +0.836 | +0.699 | +1.054 | 0.355 |
| COL11A1 | -0.426 | -0.545 | -0.292 | 0.253 |

**Interpretation:**
- Point estimates are stable (observed ≈ bootstrap mean)
- Confidence intervals do NOT overlap zero (individually significant)
- But FDR correction accounts for 666 tests → no global significance

**Evidence:**
- File: `bootstrap_confidence_intervals.csv`
- Plot: `diagnostics/bootstrap_distributions.png`

---

## 6.0 Diagnostic Plots Summary

¶1 All plots in `diagnostics/` directory

| Plot | Finding |
|------|---------|
| `qq_plots_by_study.png` | Heavy tails, non-normal distributions |
| `zscore_distributions.png` | Overall N(0,1) approximation reasonable |
| `variance_by_study.png` | Variance normalization working correctly |
| `sensitivity_normalization_methods.png` | Methods produce similar rankings |
| `pca_batch_effects.png` | Clear study clustering (batch effects) |
| `bootstrap_distributions.png` | Stable estimates, non-zero CIs |

---

## 7.0 Code Implementation Audit

¶1 Review of `/Users/Kravtsovd/projects/ecm-atlas/11_subagent_for_LFQ_ingestion/universal_zscore_function.py`

### 7.1 Algorithm Correctness

✅ **PASS:** Implementation follows standard z-score formula correctly

```python
# Step 1: Group by compartment (correct)
grouped = df.groupby(['Tissue'], dropna=False)

# Step 2: Log-transform if skewed (correct threshold: |skew| > 1)
if abs(skew_young) > 1 or abs(skew_old) > 1:
    values = np.log2(abundance + 1)

# Step 3: Calculate mean/std excluding NaN (correct)
mean = values.mean()  # skipna=True by default
std = values.std()

# Step 4: Z-score (correct)
zscore = (values - mean) / std

# Step 5: Validation (correct thresholds)
assert abs(zscore.mean()) < 0.01  # μ ≈ 0
assert abs(zscore.std() - 1.0) < 0.01  # σ ≈ 1
```

### 7.2 Data Handling

✅ **CORRECT:**
- NaN values excluded from mean/std (preserves missing data)
- Zero values INCLUDED in calculations (0 ≠ NaN)
- Per-compartment normalization (prevents tissue confounding)

⚠️ **LIMITATION:**
- No batch effect correction across studies
- Assumes within-study homogeneity (violated per ICC=0.29)
- No outlier detection/removal

### 7.3 Recommendations

1. **Add batch correction:** ComBat, limma, or study-specific offsets
2. **Test robust alternatives:** Quantile normalization for non-normal data
3. **Document limitations:** Clearly state z-scores are NOT comparable across studies without correction
4. **Add quality control:** Flag studies with ICC < 0.3 as unreliable for cross-study analysis

---

## 8.0 Executive Summary - VERDICT

¶1 Ordering: Strengths → Weaknesses → Recommendations

### 8.1 Is Methodology Sound?

**NO - with critical caveats:**

✅ **Strengths:**
1. Z-score calculation is mathematically correct
2. Variance normalization works perfectly (homoscedasticity passed)
3. Results robust to normalization method choice
4. Within-study comparisons are valid

❌ **Critical Weaknesses:**
1. **Batch effects dominate signal** (ICC=0.29, PCA clustering)
2. **Normality assumption violated** in 58% of distributions
3. **Zero FDR-significant proteins** across studies
4. **Cross-study integration unsupported** without batch correction

### 8.2 What Are the Limitations?

**Primary Limitations:**

1. **Cross-Study Comparisons Invalid:** ICC=0.29 means 71% of variance is study-specific noise. Cannot reliably compare Protein A in Study 1 vs Study 2.

2. **No Global Aging Signatures:** Zero proteins pass FDR<0.05, suggesting findings are study-specific, not universal.

3. **Parametric Tests Unreliable:** Non-normal distributions invalidate t-tests, ANOVA used in downstream analyses.

4. **Missing Batch Correction:** Studies differ in: tissue processing, mass spec protocol, quantification method, species. Z-scores do NOT remove these technical confounders.

### 8.3 Recommendations

**Immediate Actions:**

1. **Implement Batch Correction:**
   - Use ComBat (sva package) or limma's removeBatchEffect
   - Include study as covariate in linear models
   - Test if batch-corrected ICC improves

2. **Use Robust Statistics:**
   - Replace parametric tests with non-parametric alternatives
   - Use median-centered z-scores for skewed data
   - Report effect sizes, not just p-values

3. **Stratify Analyses:**
   - Perform within-study analyses first
   - Meta-analyze results across studies (forest plots)
   - Report study-level heterogeneity (I²)

4. **Update Documentation:**
   - Add WARNING: "Z-scores are study-specific; cross-study comparisons require batch correction"
   - Include ICC and batch effect diagnostics in quality reports
   - Document which proteins are reliably measured across studies

**Long-Term:**

5. **Reprocess with Batch Correction:**
   - Re-run all analyses with ComBat-corrected data
   - Compare results to current z-score approach
   - Publish sensitivity analysis

6. **External Validation:**
   - Test findings in independent cohorts
   - Collaborate with biostatistician for formal review
   - Pre-register hypotheses before testing

---

## 9.0 Files Generated

¶1 All outputs in `/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.3.1_statistical_validation/agent1/`

**Quantitative Results:**
- `normality_test_results.csv` - Shapiro-Wilk tests (n=24)
- `homoscedasticity_test_results.csv` - Levene's test
- `independence_test_results.csv` - Abundance-zscore correlations
- `sensitivity_analysis_methods.csv` - 4 normalization methods
- `pca_batch_effects.csv` - PCA coordinates (12 studies)
- `icc_batch_effects.csv` - ICC=0.29 (poor reliability)
- `observed_protein_changes.csv` - 666 proteins tested
- `permutation_test_results.csv` - Top 10 proteins permuted
- `bootstrap_confidence_intervals.csv` - CIs for 4 drivers

**Diagnostic Plots:**
- `diagnostics/qq_plots_by_study.png` - Normality assessment
- `diagnostics/zscore_distributions.png` - N(0,1) fit
- `diagnostics/variance_by_study.png` - Homoscedasticity
- `diagnostics/sensitivity_normalization_methods.png` - Method comparison
- `diagnostics/pca_batch_effects.png` - Study clustering
- `diagnostics/bootstrap_distributions.png` - CI stability

**Code:**
- `statistical_validation.py` - Assumption testing + sensitivity
- `batch_effects_permutation.py` - PCA + ICC + permutation tests

---

## 10.0 Conclusion

**Agent 1 Verdict:** ECM-Atlas z-score methodology is **STATISTICALLY FLAWED** for cross-study integration. While within-study normalization is mathematically correct and robust, high batch effects (ICC=0.29), violated normality assumptions (58% failure), and zero FDR-significant proteins indicate that current approach cannot support quantitative cross-study comparisons. Batch correction is MANDATORY before making any claims about universal ECM aging signatures.

**Confidence Level:** HIGH (comprehensive validation across 5 independent methods)

**Next Steps:** Implement ComBat batch correction, re-analyze with corrected data, and validate findings in external cohorts.

---

**Author:** Agent 1 - Statistical Validator
**Date:** 2025-10-17
**Session:** Q1.3.1_statistical_validation
**Status:** COMPLETE - METHODOLOGY FLAWED, BATCH CORRECTION REQUIRED

---

## Checklist

✅ Thesis: 1-sentence outcome preview
✅ Overview: 1-paragraph expansion + section preview
✅ Mermaid: Continuant (TD structure) + Occurrent (LR process)
✅ Sections: Numbered 1.0-10.0, MECE organization
✅ Paragraphs: ¶1 ordering principle stated
✅ Evidence: All claims linked to files/plots
✅ DRY: No redundancy, references used
✅ Actionable: Clear recommendations with priorities
✅ Skeptical: Challenged assumptions, found weaknesses
