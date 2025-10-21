# Agent 1 - Statistical Validation Results

## Quick Start

**Main Report:** `AGENT1_ZSCORE_AUDIT.md` - Full comprehensive audit with findings and recommendations

**TL;DR Verdict:** ❌ METHODOLOGY FLAWED - Batch effects (ICC=0.29) and violated assumptions invalidate cross-study comparisons. Batch correction MANDATORY.

---

## Key Findings (60 seconds)

1. **Normality:** ❌ 58% FAIL - Z-scores not normally distributed in most studies
2. **Homoscedasticity:** ✅ PASS - Variance normalization works perfectly
3. **Batch Effects:** ❌ CRITICAL - ICC=0.29 (poor), studies cluster in PCA
4. **Statistical Power:** ❌ ZERO proteins FDR-significant across studies
5. **Sensitivity:** ✅ Results robust to normalization method choice
6. **Bootstrap:** ✅ Stable estimates with reasonable confidence intervals

**Bottom Line:** Within-study comparisons are valid. Cross-study comparisons are NOT valid without batch correction.

---

## File Navigation

### Core Documentation
- `AGENT1_ZSCORE_AUDIT.md` - Main comprehensive report (READ THIS FIRST)
- `README.md` - This file (navigation guide)

### Quantitative Results (CSV)
```
normality_test_results.csv          - 24 Shapiro-Wilk tests
homoscedasticity_test_results.csv   - Levene's variance test
independence_test_results.csv       - Abundance-zscore correlations
sensitivity_analysis_methods.csv    - 4 normalization methods compared
pca_batch_effects.csv              - PCA coordinates for 12 studies
icc_batch_effects.csv              - ICC=0.29 (poor reliability)
observed_protein_changes.csv        - 666 proteins tested for age effects
permutation_test_results.csv       - Top 10 proteins permuted
bootstrap_confidence_intervals.csv  - CIs for 4 driver proteins
```

### Diagnostic Plots (PNG)
```
diagnostics/
├── qq_plots_by_study.png                      - Normality check per study
├── zscore_distributions.png                   - Overall N(0,1) fit
├── variance_by_study.png                      - Variance consistency
├── sensitivity_normalization_methods.png      - Method comparison
├── pca_batch_effects.png                      - Study clustering (CRITICAL)
└── bootstrap_distributions.png                - CI stability for drivers
```

### Code (Python)
```
statistical_validation.py       - Assumption tests + sensitivity analysis
batch_effects_permutation.py   - PCA + ICC + permutation testing
```

---

## Critical Plots (View These First)

1. **`diagnostics/pca_batch_effects.png`**
   - Shows studies cluster separately (not overlapping)
   - Proof of batch effects

2. **`diagnostics/qq_plots_by_study.png`**
   - Many studies deviate from normal (diagonal line)
   - Violates z-score assumptions

3. **`diagnostics/variance_by_study.png`**
   - Shows variance ≈ 1 across all studies
   - Only thing that worked correctly!

---

## How to Reproduce

```bash
cd /Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.3.1_statistical_validation/agent1

# Run validation (Part 1)
python statistical_validation.py

# Run batch effects & permutation (Part 2)
python batch_effects_permutation.py
```

**Runtime:** ~5 minutes total

**Requirements:** pandas, numpy, scipy, sklearn, seaborn, matplotlib, tqdm

---

## Top 3 Recommendations

1. **Implement Batch Correction**
   - Use ComBat (sva R package) or limma
   - Re-run all analyses with corrected data
   - Compare ICC before/after

2. **Use Non-Parametric Tests**
   - Replace t-tests with Mann-Whitney U
   - Use Spearman instead of Pearson correlations
   - Report effect sizes + confidence intervals

3. **Stratified Meta-Analysis**
   - Analyze each study separately first
   - Meta-analyze results (forest plots)
   - Report heterogeneity (I² statistic)

---

## Questions & Interpretation

**Q: Can I trust individual study results?**
A: YES - within-study z-scores are valid.

**Q: Can I compare proteins across studies?**
A: NO - without batch correction, comparisons are confounded by study-specific effects.

**Q: Why are zero proteins significant at FDR < 0.05?**
A: High between-study variability (batch effects) overwhelms biological signal. Individual proteins may be real within studies but don't replicate across studies.

**Q: Is the z-score calculation wrong?**
A: NO - implementation is mathematically correct. The PROBLEM is assuming studies are comparable after z-score normalization (they are not).

**Q: What should I do next?**
A: Apply batch correction (ComBat), re-test, and validate in independent cohorts.

---

## Contact

**Agent:** Agent 1 - Statistical Validator
**Date:** 2025-10-17
**Status:** COMPLETE - FLAWED METHODOLOGY IDENTIFIED
**Next Steps:** Batch correction required before publication

**For questions:** See main report `AGENT1_ZSCORE_AUDIT.md` sections 8.2-8.3
