# ECM-Atlas Database Quality Assessment & Imputation Strategy

**Document Version:** 2.0
**Date:** 2025-10-16
**Assessment of:** `merged_ecm_aging_zscore.csv`
**Total Records:** 4,584 protein-tissue-study combinations
**Objective:** Comprehensive missing data analysis and scientifically sound imputation strategy for cross-organ aging signature discovery

---

## Executive Summary

The ECM-Atlas database contains **4,584 measurements** across **1,376 unique proteins** from **12 studies** covering **8 organs** and **17 tissue compartments**. While 93.89% of rows have complete critical data, strategic imputation could increase cross-organ signature discovery by **11.2%** (from 463 to 515 proteins with multi-organ coverage at >=75% completeness).

**Current State:**
- 4,584 total measurements
- 4,304 complete (93.89%)
- 280 incomplete (6.11%)
- 6 non-imputable (0.13% - both abundance values missing)

**Imputation Potential:**
- 274 rows can be imputed (99.87% of incomplete data)
- Expected gain: +52 cross-organ signature proteins (+11.2%)
- Risk: Low (only 6.11% of data requires imputation)

---

## Table of Contents

1. [Missing Data Analysis](#1-missing-data-analysis)
2. [Protein Coverage Patterns](#2-protein-coverage-patterns)
3. [Impact on Cross-Organ Signature Discovery](#3-impact-on-cross-organ-signature-discovery)
4. [Imputation Strategy](#4-imputation-strategy)
5. [Implementation Plan](#5-implementation-plan)
6. [Risk Assessment](#6-risk-assessment)
7. [Quality Control Metrics](#7-quality-control-metrics)

---

## 1. Missing Data Analysis

### 1.1 Overall Missing Data Statistics

**Database Overview:**
- **Total rows:** 4,584
- **Total columns:** 26
- **Unique proteins:** 1,376
- **Unique genes:** 1,167
- **Studies:** 12
- **Organs:** 8
- **Compartments:** 17
- **Tissues:** 18

**Overall Missing Data:**
- Total cells: 119,184
- Missing cells: 12,419 (10.42%)
- **Complete rows (all critical columns):** 4,304 (93.89%)
- **Incomplete rows:** 280 (6.11%)

### 1.2 Missing Data by Column

Critical columns for cross-organ signature analysis:

| Column | Missing Count | Missing % | Impact |
|--------|--------------|-----------|--------|
| `Abundance_Old` | 85 | 1.85% | HIGH - Cannot compute z-score without abundance |
| `Abundance_Young` | 201 | 4.38% | HIGH - Cannot compute z-score without abundance |
| `Zscore_Old` | 85 | 1.85% | HIGH - Required for age comparison |
| `Zscore_Young` | 201 | 4.38% | HIGH - Required for age comparison |
| `Zscore_Delta` | 280 | 6.11% | CRITICAL - Primary metric for aging signatures |

**Non-critical columns with missing data:**
- `Abundance_Old_transformed`: 2,125 (46.36%) - Can be recalculated
- `Abundance_Young_transformed`: 2,119 (46.23%) - Can be recalculated
- `N_Profiles_Old`: 1,575 (34.36%) - Metadata extraction needed
- `N_Profiles_Young`: 1,575 (34.36%) - Metadata extraction needed
- `Protein_Name`: 1,723 (37.59%) - Can be retrieved from UniProt
- `Data_Quality`: 2,407 (52.51%) - New column, needs backfill
- `Canonical_Gene_Symbol`: 41 (0.89%) - Low priority

### 1.3 Missing Data Patterns

**Asymmetric Detection:**
- **Only `Abundance_Old` missing:** 79 rows (1.72%)
- **Only `Abundance_Young` missing:** 195 rows (4.25%)
- **Both abundances missing:** 6 rows (0.13%)

**Interpretation:**
- Young samples have 2.5x more missing data than old samples
- Likely due to lower protein abundance in young tissues (developmental state)
- 99.87% of incomplete rows are imputable (only one abundance missing)

### 1.4 Missing Data by Study

| Study | Total | Complete | Missing | Missing % |
|-------|-------|----------|---------|-----------|
| Schuler_2021 | 1,290 | 1,290 | 0 | 0.0% |
| Randles_2021 | 458 | 458 | 0 | 0.0% |
| Tsumagari_2023 | 423 | 423 | 0 | 0.0% |
| LiDermis_2021 | 262 | 262 | 0 | 0.0% |
| Santinha_2024_Human | 207 | 207 | 0 | 0.0% |
| Santinha_2024_Mouse_NT | 191 | 191 | 0 | 0.0% |
| Dipali_2023 | 173 | 173 | 0 | 0.0% |
| Santinha_2024_Mouse_DT | 155 | 155 | 0 | 0.0% |
| Ouni_2022 | 98 | 98 | 0 | 0.0% |
| Caldeira_2017 | 43 | 39 | 4 | 9.3% |
| **Tam_2020** | **993** | **739** | **254** | **25.6%** |
| Angelidis_2019 | 291 | 269 | 22 | 7.6% |

**Key Findings:**
- **Tam_2020 accounts for 90.7% of all missing data** (254/280 incomplete rows)
- 9 studies have 0% missing data in critical columns
- 2 studies have <10% missing data
- Problem is highly localized to one study

**Tam_2020 Missing Data Breakdown:**
- Study type: Mouse skeletal muscle aging (multiple muscle types)
- Missing pattern: Asymmetric detection (18.9% only-old, 6.6% only-young)
- Likely cause: Label-free quantification with low-abundance proteins at detection limit
- Imputation feasibility: HIGH (all missing values are single-group, not both)

---

## 2. Protein Coverage Patterns

### 2.1 Proteins by Coverage Completeness

**Before Imputation:**

| Coverage Threshold | Protein Count | Percentage |
|-------------------|---------------|------------|
| >=0% (all proteins) | 1,376 | 100.0% |
| <=0% missing | 1,159 | 84.2% |
| <=25% missing | 1,211 | 88.0% |
| <=50% missing | 1,280 | 93.0% |
| <=75% missing | 1,293 | 94.0% |

**After Imputation (Projected):**

| Coverage Threshold | Protein Count | Improvement | % Improvement |
|-------------------|---------------|-------------|---------------|
| <=0% missing | 1,370 | +211 | +18.2% |
| <=25% missing | 1,374 | +163 | +13.5% |
| <=50% missing | 1,374 | +94 | +7.3% |
| <=75% missing | 1,373 | +80 | +6.2% |

**Interpretation:**
- 84.2% of proteins already have perfect coverage
- Imputation would achieve near-complete coverage (99.6% at <=0% missing)
- Largest gains at high-coverage thresholds (perfect coverage: +18.2%)

### 2.2 Cross-Organ Distribution

**Proteins by Number of Organs:**

| Organs | Protein Count | Percentage |
|--------|---------------|------------|
| 1 | 860 | 62.5% |
| 2 | 171 | 12.4% |
| 3 | 133 | 9.7% |
| 4 | 121 | 8.8% |
| 5 | 89 | 6.5% |
| 6 | 2 | 0.1% |

**Multi-organ proteins:** 516 (37.5%)

### 2.3 Cross-Compartment Distribution

**Proteins by Number of Compartments:**

| Compartments | Protein Count | Percentage |
|--------------|---------------|------------|
| 1 | 380 | 27.6% |
| 2 | 255 | 18.5% |
| 3 | 180 | 13.1% |
| 4 | 260 | 18.9% |
| 5-6 | 123 | 8.9% |
| 7-8 | 131 | 9.5% |
| 9+ | 47 | 3.4% |

**Multi-compartment proteins:** 996 (72.4%)

### 2.4 Cross-Organ AND Cross-Compartment Proteins

**Proteins with >1 organ AND >1 compartment:** 516 (37.5%)

This is the **critical subset for cross-organ aging signature discovery**.

### 2.5 Top Proteins with Best Cross-Tissue Coverage

**Top 10 proteins with broadest coverage:**

| Gene | Protein_ID | Organs | Compartments | Studies | Complete Measurements |
|------|-----------|--------|--------------|---------|---------------------|
| Dcn | P28654 | 6 | 11 | 7 | 11 |
| Lama4 | P97927 | 6 | 11 | 7 | 11 |
| Fga | E9PU48 | 5 | 10 | 6 | 10 |
| Nid2 | Q14112 | 5 | 10 | 6 | 10 |
| Col4a1 | P02462 | 5 | 10 | 6 | 10 |
| Anxa2 | P07355 | 5 | 10 | 6 | 10 |
| Col3a1 | P02461 | 5 | 10 | 6 | 10 |
| Col4a2 | P08572 | 5 | 10 | 6 | 10 |
| Anxa1 | P04083 | 5 | 10 | 6 | 10 |
| Nid1 | P14543 | 5 | 10 | 6 | 10 |

**Characteristics:**
- Core matrisome proteins (collagens, laminins, nidogens)
- Consistently detected across organs and compartments
- Perfect coverage (0% missing data)
- Ideal candidates for cross-organ aging signatures

### 2.6 Proteins with Highest Missing Data

**Top 10 proteins with poorest coverage (>1 measurement):**

All proteins with 100% missing data have only 2-3 total measurements and are from single organs (Tam_2020 or Santinha studies). These are low-priority for imputation.

---

## 3. Impact on Cross-Organ Signature Discovery

### 3.1 Current State: Proteins Discoverable Without Imputation

**Cross-organ proteins (>1 organ) with >=75% coverage:**
- **463 proteins**
- Represents proteins reliably measured across multiple organs
- Sufficient for initial cross-organ aging signature analysis

### 3.2 Projected State: Proteins Discoverable With Imputation

**Cross-organ proteins (>1 organ) with >=75% coverage after imputation:**
- **515 proteins**
- **Gain: +52 proteins (+11.2%)**
- Increases statistical power for cross-organ comparisons

**Cross-organ + cross-compartment proteins (>1 organ AND >1 compartment) with >=75% coverage:**
- Before: 463 proteins
- After: 515 proteins
- **Gain: +52 proteins (+11.2%)**

### 3.3 Scientific Value of Imputation

**Benefits:**
1. **Increased statistical power:** 11.2% more proteins for cross-organ aging signature discovery
2. **Reduced bias:** Currently excluding proteins with minor missing data penalizes biologically real low-abundance proteins
3. **Study-level completeness:** Tam_2020 (largest study) would go from 74.4% to ~99% complete
4. **Publication readiness:** Near-complete datasets are more defensible for peer review

**Limitations:**
1. **Modest gains:** Only 11.2% improvement (not transformative)
2. **Localized problem:** 90.7% of missing data is in one study (Tam_2020)
3. **Alternative solution:** Could simply exclude Tam_2020 incomplete rows (but loses data)

### 3.4 Comparison: Imputation vs. Exclusion

**Option A: Complete Case Analysis (No Imputation)**
- Analyze only 4,304 complete rows (93.89%)
- Pros: No assumptions, conservative
- Cons: Loses 280 measurements (6.11% of data)

**Option B: Imputation**
- Impute 274 rows (99.87% of incomplete data)
- Pros: Retains 99.87% of incomplete data, +11.2% cross-organ proteins
- Cons: Introduces uncertainty (though quantifiable)

**Option C: Study Exclusion**
- Exclude Tam_2020 entirely (993 rows)
- Pros: Removes 90.7% of missing data problem
- Cons: Loses entire study (21.7% of database), reduces cross-organ power

**Recommendation:** **Option B (Imputation)** - Best balance of data retention and scientific rigor

---

## 4. Imputation Strategy

### 4.1 Recommended Approach: Hybrid Imputation Framework

Given that missing data is highly localized (90.7% in Tam_2020) and mechanistically interpretable (left-censored detection limit), we recommend a **three-tier imputation strategy**:

#### Tier 1: Complete Case Analysis (Primary Analysis)
- Use only 4,304 complete rows (93.89%)
- No imputation
- Most conservative, publication-ready

#### Tier 2: Selective Imputation (Sensitivity Analysis)
- Impute only Tam_2020 missing values (254 rows)
- Use study-specific method (see 4.2)
- Compare results with Tier 1 to assess robustness

#### Tier 3: Comprehensive Imputation (Exploratory Analysis)
- Impute all missing values (280 rows)
- Use KNN or quantile regression
- Flag imputed values in `Data_Quality` column

### 4.2 Imputation Methods by Missing Data Mechanism

#### Method A: Left-Censored Imputation (for MNAR data)

**When to use:** Missing values where protein is detected in one age group but not the other, AND detected abundance is <25th percentile

**Rationale:** Protein is below detection limit (not truly absent)

**Method:** Quantile Regression Imputation of Left-Censored data (QRILC)

```python
from sklearn.impute import IterativeImputer
from scipy.stats import norm
import numpy as np

def qrilc_imputation(df, column, study_col='Study_ID'):
    """
    Quantile Regression Imputation of Left-Censored data.

    Method:
    1. Identify left-censored values (missing when detected value is low)
    2. Estimate distribution of values below detection limit
    3. Sample from truncated normal distribution

    Reference: Wei et al. (2018) "Missing Value Imputation Approach for Mass
               Spectrometry-based Metabolomics Data"
    """
    imputed_df = df.copy()

    for study in df[study_col].unique():
        mask = (df[study_col] == study) & (df[column].isnull())

        # Get distribution of detected values in this study
        detected = df[(df[study_col] == study) & (df[column].notna())][column]

        # Estimate parameters for truncated normal below 25th percentile
        q25 = detected.quantile(0.25)
        mean_low = detected[detected < q25].mean()
        std_low = detected[detected < q25].std()

        # Sample from truncated normal (0, q25)
        n_missing = mask.sum()
        imputed_values = norm.rvs(loc=mean_low, scale=std_low, size=n_missing)
        imputed_values = np.clip(imputed_values, 0, q25)  # Truncate at q25

        imputed_df.loc[mask, column] = imputed_values
        imputed_df.loc[mask, 'Data_Quality'] = 'Imputed_QRILC'

    return imputed_df
```

**Applicability:** ~80% of Tam_2020 missing values

#### Method B: K-Nearest Neighbors Imputation (for MAR data)

**When to use:** Missing values that appear random within study (not clearly left-censored)

**Rationale:** Similar proteins (by abundance profile) likely have similar missing values

**Method:** KNN imputation using within-study protein abundance profiles

```python
from sklearn.impute import KNNImputer

def knn_imputation(df, n_neighbors=5, study_col='Study_ID'):
    """
    K-Nearest Neighbors imputation within each study.

    Method:
    1. Group by study to preserve study-specific batch effects
    2. Use k=5 nearest neighbors by Euclidean distance in abundance space
    3. Impute missing value as weighted mean of k neighbors

    Reference: Troyanskaya et al. (2001) Bioinformatics 17:520-525
    """
    imputed_df = df.copy()

    for study in df[study_col].unique():
        study_mask = df[study_col] == study
        study_data = df[study_mask].copy()

        # Extract abundance columns
        abundance_cols = ['Abundance_Old', 'Abundance_Young']
        abundance_matrix = study_data[abundance_cols].values

        # KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
        imputed_matrix = imputer.fit_transform(abundance_matrix)

        # Update dataframe
        imputed_df.loc[study_mask, abundance_cols] = imputed_matrix

        # Mark imputed values
        for col in abundance_cols:
            was_missing = study_data[col].isnull()
            imputed_df.loc[study_mask & was_missing, 'Data_Quality'] = 'Imputed_KNN'

    return imputed_df
```

**Applicability:** ~18% of Tam_2020 missing values, all other studies

#### Method C: Minimum Value Imputation (for biological absence)

**When to use:** Proteins biologically absent in one age group (e.g., developmental markers)

**Rationale:** True biological absence, not detection limit issue

**Method:** Impute with minimum detected value in dataset (conservative)

```python
def minimum_value_imputation(df, column, percentile=0.01):
    """
    Impute missing values with minimum detected value (or lower percentile).

    Method:
    1. Calculate 1st percentile of all detected values
    2. Impute missing values with this minimum
    3. Conservative approach for true biological absence
    """
    imputed_df = df.copy()
    min_value = df[column].quantile(percentile)

    missing_mask = df[column].isnull()
    imputed_df.loc[missing_mask, column] = min_value
    imputed_df.loc[missing_mask, 'Data_Quality'] = 'Imputed_MinValue'

    return imputed_df
```

**Applicability:** ~2% of missing values (identified by biological context)

### 4.3 Recommended Imputation Workflow

```python
def impute_ecm_atlas(df):
    """
    Comprehensive imputation workflow for ECM-Atlas database.

    Steps:
    1. Classify missing data mechanism
    2. Apply appropriate imputation method
    3. Recalculate z-scores for imputed abundances
    4. Validate imputation quality
    """
    df_imputed = df.copy()

    # Step 1: Identify left-censored values
    # (detected in one group but not other, AND detected value is low)
    for idx, row in df_imputed[df_imputed['Abundance_Old'].isnull()].iterrows():
        if pd.notna(row['Abundance_Young']):
            # Check if detected value is <25th percentile in this study
            study_data = df_imputed[df_imputed['Study_ID'] == row['Study_ID']]
            q25 = study_data['Abundance_Young'].quantile(0.25)

            if row['Abundance_Young'] < q25:
                # Left-censored: use QRILC
                df_imputed = qrilc_imputation(df_imputed, 'Abundance_Old')
            else:
                # Not left-censored: use KNN
                df_imputed = knn_imputation(df_imputed, 'Abundance_Old')

    # Repeat for Abundance_Young
    # (same logic)

    # Step 2: Recalculate z-scores for imputed values
    df_imputed = recalculate_zscores(df_imputed)

    # Step 3: Validate imputation
    validate_imputation(df_imputed, df)

    return df_imputed

def recalculate_zscores(df):
    """
    Recalculate z-scores for imputed abundances.
    Use within-study normalization.
    """
    df_recalc = df.copy()

    for study in df['Study_ID'].unique():
        study_mask = df['Study_ID'] == study

        # Recalculate z-scores for this study
        old_mean = df.loc[study_mask, 'Abundance_Old'].mean()
        old_std = df.loc[study_mask, 'Abundance_Old'].std()
        df_recalc.loc[study_mask, 'Zscore_Old'] = (
            (df.loc[study_mask, 'Abundance_Old'] - old_mean) / old_std
        )

        # Repeat for Young
        young_mean = df.loc[study_mask, 'Abundance_Young'].mean()
        young_std = df.loc[study_mask, 'Abundance_Young'].std()
        df_recalc.loc[study_mask, 'Zscore_Young'] = (
            (df.loc[study_mask, 'Abundance_Young'] - young_mean) / young_std
        )

        # Recalculate delta
        df_recalc['Zscore_Delta'] = (
            df_recalc['Zscore_Old'] - df_recalc['Zscore_Young']
        )

    return df_recalc

def validate_imputation(df_imputed, df_original):
    """
    Validate imputation quality.

    Checks:
    1. Imputed values are in plausible range
    2. Distribution of imputed values matches observed
    3. No extreme outliers introduced
    """
    imputed_mask = df_imputed['Data_Quality'].str.contains('Imputed', na=False)

    print("Imputation Validation:")
    print(f"  Total imputed values: {imputed_mask.sum()}")

    # Check Abundance_Old imputed values
    imputed_old = df_imputed[imputed_mask & df_original['Abundance_Old'].isnull()]
    observed_old = df_original['Abundance_Old'].dropna()

    print(f"\n  Abundance_Old:")
    print(f"    Observed range: [{observed_old.min():.2f}, {observed_old.max():.2f}]")
    print(f"    Imputed range:  [{imputed_old['Abundance_Old'].min():.2f}, {imputed_old['Abundance_Old'].max():.2f}]")
    print(f"    Observed mean:  {observed_old.mean():.2f}")
    print(f"    Imputed mean:   {imputed_old['Abundance_Old'].mean():.2f}")

    # Check for outliers (imputed values >3 SD from observed mean)
    outliers = imputed_old[
        (imputed_old['Abundance_Old'] > observed_old.mean() + 3*observed_old.std()) |
        (imputed_old['Abundance_Old'] < observed_old.mean() - 3*observed_old.std())
    ]
    print(f"    Outliers (>3 SD): {len(outliers)}")

    # Repeat for Abundance_Young
    # (same logic)
```

### 4.4 Alternative: Simple Mean Imputation (Not Recommended)

**Method:** Replace missing values with study-specific mean

**Pros:** Simple, fast, deterministic

**Cons:**
- Reduces variance (artificially increases power)
- Ignores missing data mechanism
- Not scientifically defensible for MNAR data
- **NOT RECOMMENDED for proteomics data**

---

## 5. Implementation Plan

### 5.1 Phase 1: Data Extraction and Preparation (Week 1)

**Objective:** Complete missing metadata before imputation

**Tasks:**
1. Extract sample sizes for Randles_2021 from supplementary materials
   - Source: PRIDE repository PXD019345
   - Target: Fill 1,575 missing `N_Profiles_Old` and `N_Profiles_Young` values

2. Extract standard deviations from all studies
   - Required for statistical significance testing
   - New columns: `SD_Old`, `SD_Young`

3. Retrieve missing protein names from UniProt API
   - Target: Fill 1,723 missing `Protein_Name` values
   - Script: `scripts/retrieve_protein_metadata.py`

**Deliverables:**
- Updated CSV with complete metadata
- Documentation of extraction methods

### 5.2 Phase 2: Missing Data Mechanism Classification (Week 1-2)

**Objective:** Classify each missing value by mechanism (MCAR, MAR, MNAR)

**Tasks:**
1. Implement missing data diagnostic tests
   - Little's MCAR test
   - Logistic regression for MAR
   - Abundance threshold analysis for MNAR (left-censored)

2. Create missing data classification column
   - Values: 'MCAR', 'MAR', 'MNAR_LeftCensored', 'MNAR_Biological'

3. Generate missing data report
   - Study-specific patterns
   - Protein-specific patterns
   - Abundance distribution analysis

**Deliverables:**
- `missing_data_classification.csv`
- `missing_data_diagnostic_report.md`

### 5.3 Phase 3: Imputation Implementation (Week 2-3)

**Objective:** Implement and validate imputation methods

**Tasks:**
1. Implement QRILC imputation for left-censored data
   - Target: ~80% of Tam_2020 missing values
   - Validation: Check imputed values are <25th percentile

2. Implement KNN imputation for MAR data
   - Target: ~18% of missing values
   - Parameter tuning: k=3, 5, 7, 10 (cross-validation)

3. Implement minimum value imputation for biological absence
   - Target: ~2% of missing values
   - Manual review of candidates

4. Create imputation pipeline script
   - `scripts/impute_ecm_atlas.py`
   - Command-line interface with method selection

**Deliverables:**
- Imputed database: `merged_ecm_aging_zscore_imputed.csv`
- Imputation flag column: `Data_Quality` ('Original', 'Imputed_QRILC', 'Imputed_KNN', 'Imputed_MinValue')
- Imputation report: `imputation_validation_report.md`

### 5.4 Phase 4: Validation and Sensitivity Analysis (Week 3-4)

**Objective:** Ensure imputation does not introduce bias or artifacts

**Tasks:**
1. **Validation Test 1:** Distribution comparison
   - Compare imputed vs. observed abundance distributions
   - Kolmogorov-Smirnov test (p > 0.05 = distributions match)

2. **Validation Test 2:** Leave-one-out cross-validation
   - Artificially remove 10% of observed values
   - Impute them
   - Compare imputed vs. true values (RMSE, MAE, R-squared)

3. **Validation Test 3:** Biological consistency
   - Re-run cross-organ signature discovery on:
     a) Complete cases only (4,304 rows)
     b) Complete + imputed (4,578 rows)
   - Compare top 50 aging signatures (overlap should be >80%)

4. **Sensitivity Analysis:** Test robustness
   - Vary imputation parameters (k in KNN, quantile in QRILC)
   - Measure impact on key biological findings
   - Document parameter sensitivity

**Acceptance Criteria:**
- Imputed distributions match observed (KS test p > 0.05)
- Cross-validation RMSE < 20% of mean abundance
- Top 50 aging signatures: >80% overlap between imputed vs. complete-case analysis
- No extreme outliers introduced (all imputed values within 3 SD)

**Deliverables:**
- `imputation_validation_results.csv`
- `sensitivity_analysis_report.md`
- Decision matrix: Use imputed vs. complete-case for publication

### 5.5 Phase 5: Documentation and Versioning (Week 4)

**Objective:** Create publication-ready documentation

**Tasks:**
1. Update database schema documentation
   - Document `Data_Quality` column values
   - Add imputation method descriptions

2. Create methods section for manuscript
   - Detailed imputation methodology
   - Justification for method selection
   - Validation results

3. Version control
   - `merged_ecm_aging_zscore_v1.0.csv` (original, complete-case)
   - `merged_ecm_aging_zscore_v2.0_imputed.csv` (with imputation)

4. Create reproducibility package
   - All imputation scripts
   - Random seeds for reproducibility
   - Input data checksums

**Deliverables:**
- `DATABASE_IMPUTATION_METHODS.md`
- `REPRODUCIBILITY_GUIDE.md`
- Manuscript methods section draft

---

## 6. Risk Assessment

### 6.1 Risks of Imputation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Introduce bias in aging signatures** | Low | High | Use conservative methods (QRILC), validate with leave-one-out CV |
| **Reduce variance (false positives)** | Medium | High | Flag imputed values, run sensitivity analysis, use Tier 1 for primary analysis |
| **Reviewer skepticism** | Medium | Medium | Comprehensive documentation, three-tier analysis, show robustness |
| **Computational complexity** | Low | Low | Imputation is fast (<1 min for 280 values) |
| **Incorrect mechanism classification** | Low | Medium | Manual review of classifications, conservative default (QRILC) |

### 6.2 Risks of NOT Imputing

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Lose 11.2% cross-organ proteins** | High | Medium | Already accepted in Tier 1 analysis |
| **Bias against low-abundance proteins** | High | High | Complete-case analysis preferentially excludes proteins near detection limit (biological bias) |
| **Reduce statistical power** | High | Medium | Smaller sample size = larger confidence intervals |
| **Data waste** | High | Low | Losing 6.11% of hard-won proteomic data |

### 6.3 Risk Mitigation Strategy

**Primary Recommendation:** Three-tier analysis framework

1. **Tier 1 (Conservative):** Complete cases only (4,304 rows)
   - Use for primary biological claims
   - Publication-ready, no imputation concerns

2. **Tier 2 (Sensitivity):** Imputed data (4,578 rows)
   - Use to show robustness of Tier 1 findings
   - Increase power for borderline signals

3. **Tier 3 (Exploratory):** Alternative imputation methods
   - Test different assumptions (KNN vs. QRILC)
   - Identify method-dependent vs. robust findings

**Reporting Strategy:**
- Present Tier 1 results in main text
- Present Tier 2 vs. Tier 1 comparison in supplementary
- Use Tier 2 to claim "robust to missing data handling"

---

## 7. Quality Control Metrics

### 7.1 Pre-Imputation Checklist

- [ ] All metadata extracted (sample sizes, SDs)
- [ ] Missing data mechanism classified for each missing value
- [ ] Proteins with 100% missing data excluded from imputation
- [ ] Study-specific patterns documented
- [ ] Imputation methods selected per missing value type

### 7.2 Post-Imputation Validation Checklist

- [ ] No missing values remain in critical columns (Abundance_Old, Abundance_Young)
- [ ] Imputed values flagged in `Data_Quality` column
- [ ] Distribution comparison: KS test p > 0.05
- [ ] Cross-validation RMSE < 20% of mean abundance
- [ ] No extreme outliers (all values within 3 SD)
- [ ] Z-scores recalculated correctly (mean ≈ 0, std ≈ 1 per study)
- [ ] Top 50 aging signatures: >80% overlap with complete-case analysis

### 7.3 Publication Readiness Checklist

- [ ] Three-tier analysis complete
- [ ] Imputation methods documented in manuscript methods
- [ ] Validation results in supplementary materials
- [ ] Sensitivity analysis shows robustness
- [ ] Complete vs. imputed results concordance >80%
- [ ] Data availability statement includes both versions
- [ ] Code deposited in GitHub with reproducibility guide

---

## 8. Expected Outcomes and Timeline

### 8.1 Expected Improvements

**Quantitative Gains:**
- Complete data: 93.89% → 99.87% (+6.1%)
- Proteins with perfect coverage: 1,159 → 1,370 (+18.2%)
- Cross-organ proteins (>1 organ, >=75% coverage): 463 → 515 (+11.2%)
- Cross-organ + cross-compartment proteins: 463 → 515 (+11.2%)

**Qualitative Gains:**
- Reduced bias against low-abundance proteins
- Increased confidence in cross-organ aging signatures
- Publication-ready documentation
- Reviewer-defensible methodology

### 8.2 Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Data Extraction | 1 week | Complete metadata |
| Phase 2: Mechanism Classification | 1 week | Missing data report |
| Phase 3: Imputation | 2 weeks | Imputed database |
| Phase 4: Validation | 2 weeks | Validation report, sensitivity analysis |
| Phase 5: Documentation | 1 week | Methods documentation, reproducibility guide |
| **Total** | **7 weeks** | Publication-ready database with imputation |

### 8.3 Implementation Complexity

**Complexity: MODERATE**

**Easy components:**
- Data extraction (standard API calls)
- QRILC imputation (existing R packages)
- KNN imputation (scikit-learn)

**Moderate components:**
- Missing data mechanism classification (requires domain knowledge)
- Validation framework (multiple tests)
- Three-tier analysis comparison

**Hard components:**
- None (all methods are established, not novel development)

**Skills required:**
- Python/R programming (intermediate)
- Proteomics data analysis (intermediate)
- Statistical inference (intermediate)

**Estimated effort:** 40-60 hours over 7 weeks (part-time, ~1 day/week)

---

## 9. Conclusions and Recommendations

### 9.1 Summary of Key Findings

1. **Database Quality:** 93.89% complete, high quality overall
2. **Missing Data:** Highly localized (90.7% in Tam_2020), mechanistically interpretable
3. **Imputation Potential:** 99.87% of incomplete data is imputable
4. **Expected Gain:** +11.2% cross-organ signature proteins (moderate but meaningful)
5. **Risk:** Low (conservative methods, three-tier validation)

### 9.2 Final Recommendations

**Recommendation 1: Implement Three-Tier Analysis Framework**
- Tier 1: Complete cases (primary analysis)
- Tier 2: With imputation (sensitivity analysis)
- Tier 3: Alternative methods (robustness check)

**Recommendation 2: Use QRILC for Tam_2020 Left-Censored Data**
- Most missing data is left-censored (below detection limit)
- QRILC is scientifically defensible for this mechanism
- Conservative approach minimizes bias

**Recommendation 3: Validate Extensively Before Publication**
- Leave-one-out cross-validation
- Distribution comparison tests
- Biological concordance (aging signature overlap)
- Sensitivity to imputation parameters

**Recommendation 4: Comprehensive Documentation**
- Methods section for manuscript
- Supplementary validation report
- Reproducibility guide with code
- Deposit both complete-case and imputed databases

### 9.3 Decision Matrix

| Scenario | Recommendation |
|----------|----------------|
| **For publication (primary analysis)** | Use Tier 1 (complete cases, no imputation) |
| **For sensitivity analysis** | Use Tier 2 (QRILC imputation) |
| **For exploratory discovery** | Use Tier 3 (KNN imputation) |
| **For reviewer concerns** | Show Tier 1 vs. Tier 2 concordance |
| **For maximizing cross-organ power** | Use Tier 2 (gains +11.2% proteins) |

### 9.4 Next Steps

1. **Immediate (Week 1):** Extract missing metadata (sample sizes, SDs)
2. **Short-term (Weeks 1-3):** Implement imputation pipeline
3. **Medium-term (Weeks 3-4):** Validate and sensitivity analysis
4. **Long-term (Week 4+):** Document for publication

---

## Appendix A: Missing Data Diagnostic Tests

### A.1 Little's MCAR Test

**Null hypothesis:** Data is missing completely at random (MCAR)

**Test statistic:** Chi-square test comparing observed vs. expected missing patterns

**Interpretation:**
- p > 0.05: Cannot reject MCAR (missing data is random)
- p < 0.05: Reject MCAR (missing data is not random)

**Expected result for ECM-Atlas:** p < 0.001 (NOT MCAR, missing data is informative)

### A.2 Logistic Regression for MAR

**Model:** Predict missingness from observed variables

**Variables:** Study_ID, Tissue, Organ, Compartment, Mean_Abundance_Study

**Interpretation:**
- Significant predictors → MAR (missing depends on observed variables)
- No significant predictors → MNAR (missing depends on unobserved variables)

**Expected result:** Study_ID and Mean_Abundance_Study are significant (MAR/MNAR mixture)

### A.3 Abundance Threshold Analysis for Left-Censored MNAR

**Method:** For each missing value, check if detected abundance (in opposite group) is below detection threshold

**Threshold:** 25th percentile of within-study abundance distribution

**Classification:**
- Detected abundance < 25th percentile → Left-censored MNAR
- Detected abundance >= 25th percentile → MAR

**Expected result:** ~80% of Tam_2020 missing values are left-censored MNAR

---

## Appendix B: Imputation Code Examples

### B.1 QRILC Imputation in R

```r
library(imputeLCMD)

# Load data
df <- read.csv("merged_ecm_aging_zscore.csv")

# Prepare abundance matrix (rows = proteins, columns = samples)
abundance_matrix <- df %>%
  select(Protein_ID, Study_ID, Abundance_Old, Abundance_Young) %>%
  pivot_wider(names_from = Study_ID, values_from = c(Abundance_Old, Abundance_Young))

# QRILC imputation
imputed_matrix <- impute.QRILC(abundance_matrix, tune.sigma = 1)

# Convert back to long format
df_imputed <- imputed_matrix[[1]] %>%
  as.data.frame() %>%
  pivot_longer(cols = -Protein_ID, names_to = "Study_Sample", values_to = "Abundance")
```

### B.2 KNN Imputation in Python

```python
from sklearn.impute import KNNImputer
import pandas as pd

# Load data
df = pd.read_csv("merged_ecm_aging_zscore.csv")

# KNN imputation within each study
for study in df['Study_ID'].unique():
    study_mask = df['Study_ID'] == study
    study_data = df[study_mask][['Abundance_Old', 'Abundance_Young']]

    # Impute with k=5 neighbors
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputed_data = imputer.fit_transform(study_data)

    df.loc[study_mask, ['Abundance_Old', 'Abundance_Young']] = imputed_data
```

---

## Appendix C: References

1. Wei et al. (2018) "Missing Value Imputation Approach for Mass Spectrometry-based Metabolomics Data" *Scientific Reports* 8:663

2. Troyanskaya et al. (2001) "Missing value estimation methods for DNA microarrays" *Bioinformatics* 17:520-525

3. Lazar et al. (2016) "Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies" *Journal of Proteome Research* 15:1116-1125

4. Little & Rubin (2019) "Statistical Analysis with Missing Data" 3rd Edition, Wiley

5. Hrydziuszko & Viant (2012) "Missing values in mass spectrometry based metabolomics: an undervalued step in the data processing pipeline" *Metabolomics* 8:161-174

---

**Document End**

*For questions or implementation support, contact: daniel@improvado.io*
