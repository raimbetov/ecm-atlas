# ECM-Atlas Database Quality Assessment & Recommendations

**Document Version:** 1.0
**Date:** 2025-10-14
**Assessment of:** `merged_ecm_aging_zscore.csv`
**Total Records:** 2,177 protein-tissue-study combinations
**Objective:** Establish scientifically sound, publication-ready database for public use

---

## Executive Summary

The current ECM-Atlas database contains valuable proteomic data from 5 studies across 8 tissues, but has **10 critical issues** that must be addressed before public release. This document provides evidence-based recommendations with specific implementation steps to achieve a robust, scientifically defensible database.

**Current State:** Suitable for exploratory analysis, **NOT ready for publication**
**Target State:** Fully validated, transparent, statistically sound database with complete documentation

---

## Table of Contents

1. [Critical Data Quality Issues](#critical-data-quality-issues)
2. [Missing Data Analysis & Imputation Strategy](#missing-data-analysis--imputation-strategy)
3. [Statistical Rigor Requirements](#statistical-rigor-requirements)
4. [Protein Identifier Validation](#protein-identifier-validation)
5. [Metadata Standardization](#metadata-standardization)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Quality Control Checklist](#quality-control-checklist)
8. [References & Methods Documentation](#references--methods-documentation)

---

## Critical Data Quality Issues

### Issue 1: Missing Sample Size Metadata (Priority: CRITICAL)

**Problem:**
- Randles_2021 has 100% missing `N_Profiles_Old` and `N_Profiles_Young` (458/2,177 proteins = 21%)
- Cannot compute statistical significance without sample sizes
- Cannot distinguish biological signal from noise

**Evidence:**
```
N_Profiles_Old missing: 458 (21.04%)
N_Profiles_Young missing: 458 (21.04%)
Study: Randles_2021 - 100% missing sample sizes
```

**Impact:** HIGH - Invalidates statistical inference for 21% of database

**Solution:**
1. **Immediate:** Extract sample sizes from Randles et al. (2021) supplementary materials
2. **Verify:** Check if 229 glomerular + 229 tubulointerstitial samples have consistent N
3. **Validate:** Cross-reference with PRIDE repository (PXD019345)
4. **Document:** Record extraction method in metadata

**Expected values (from paper):**
- Young: n=3-6 biological replicates per compartment
- Old: n=3-6 biological replicates per compartment

**Acceptance criteria:**
- 100% of proteins have documented sample sizes
- Sample sizes verified against source publication

---

### Issue 2: Low Protein ID Matching Confidence (Priority: CRITICAL)

**Problem:**
- 795/2,177 proteins (37%) have <100% match confidence
- 726 proteins have confidence = 1 (Match_Level: "Gene_Symbol_or_UniProt")
- 534 proteins (24.5%) have `Gene_Symbol ≠ Canonical_Gene_Symbol`
- Ambiguous protein identities compromise biological interpretation

**Evidence:**
```
Match_Confidence distribution:
  100:  1,382 (63.5%)
  1:      726 (33.3%)  ← CRITICAL
  95:      38 (1.7%)
  80:      31 (1.4%)

Examples of problematic mappings:
  Gene_Symbol: WISP2 → Canonical_Gene_Symbol: CCN5 (synonym, 80% confidence)
  Gene_Symbol: ARG1  → Canonical_Gene_Symbol: TINAGL1 (synonym, 80% confidence)
  Gene_Symbol: MME   → Canonical_Gene_Symbol: MMP12 (synonym, 80% confidence)
```

**Impact:** HIGH - 1/3 of database has unreliable protein identifiers

**Solution:**

**Phase 1: Re-map low-confidence proteins (confidence = 1)**
```python
# For each low-confidence protein:
# 1. Extract original identifier from source file
# 2. Query UniProt REST API with exact match
# 3. If multiple matches, use gene symbol + species to disambiguate
# 4. Validate against HGNC (human) or MGI (mouse) nomenclature
# 5. Document mapping provenance
```

**Phase 2: Validate synonym mappings (confidence = 80)**
```python
# For synonym matches:
# 1. Check UniProt "Alternative names" field
# 2. Verify CCN5 ≡ WISP2 (correct)
# 3. Flag incorrect synonyms (ARG1 ≠ TINAGL1 - INCORRECT)
# 4. Update Canonical_Gene_Symbol to authoritative HGNC/MGI symbol
```

**Phase 3: Add orthology mapping**
```python
# For cross-species comparisons:
# 1. Add column: Human_Ortholog_Gene_Symbol
# 2. Use Ensembl Compara or NCBI HomoloGene
# 3. Enable cross-species ECM aging comparisons
```

**Tools:**
- UniProt REST API: `https://rest.uniprot.org/uniprotkb/search?query=`
- HGNC API: `https://rest.genenames.org/`
- MGI Batch Query: `http://www.informatics.jax.org/batch`

**Acceptance criteria:**
- <5% of proteins with confidence < 100
- All synonym mappings validated by authoritative database
- Mapping provenance documented (source → method → final ID)

---

### Issue 3: Missing Z-Score Transformations (Priority: HIGH)

**Problem:**
- 46% of proteins missing `Abundance_*_transformed` columns
- 12.68% missing `Zscore_Delta` entirely (276/2,177)
- Inconsistent across studies (Tam_2020: 25.6% missing z-scores)

**Evidence:**
```
Missing transformed abundances:
  Abundance_Old_transformed:   1,008 (46.30%)
  Abundance_Young_transformed: 1,002 (46.03%)

Missing z-scores by study:
  Tam_2020:       254/993 (25.6%) ← Major gap
  Angelidis_2019:  22/291 (7.6%)
  Others:          0/1,893 (0.0%)
```

**Root cause analysis:**
1. Tam_2020 has asymmetric detection (18.9% only-old, 6.6% only-young)
2. Cannot compute z-score when one group is entirely missing
3. Transformation method unclear (log2? natural log? asinh?)

**Impact:** MODERATE - Reduces statistical power, but raw abundances present

**Solution:**

**Step 1: Document transformation method**
```python
# Check existing transformations
import numpy as np
df_valid = df[df['Abundance_Old_transformed'].notna()]
correlation = df_valid[['Abundance_Old', 'Abundance_Old_transformed']].corr()

# Expected: log-transformed abundances
# Verify: Abundance_Old_transformed ≈ log2(Abundance_Old) or log(Abundance_Old)
```

**Step 2: Apply consistent transformation**
```python
# Recommended: log2(x + 1) to handle zeros
def safe_log2_transform(abundance):
    """
    Apply log2(x + 1) transformation to handle zero/near-zero abundances.
    Adding pseudocount of 1 prevents -inf for zeros.
    """
    return np.log2(abundance + 1)

# Apply to all missing transformed values
df['Abundance_Old_transformed'] = df['Abundance_Old'].apply(safe_log2_transform)
df['Abundance_Young_transformed'] = df['Abundance_Young'].apply(safe_log2_transform)
```

**Step 3: Compute within-study z-scores**
```python
# Z-scores MUST be computed within each study (not globally)
def compute_zscore(df, study_col='Dataset_Name'):
    """
    Compute within-study z-scores to account for technical variation.
    """
    for study in df[study_col].unique():
        mask = df[study_col] == study
        study_data = df.loc[mask, 'Abundance_Old_transformed']

        # Z-score formula: (x - mean) / std
        mean = study_data.mean()
        std = study_data.std()
        df.loc[mask, 'Zscore_Old'] = (study_data - mean) / std

        # Repeat for Young
        # ...

    # Delta z-score
    df['Zscore_Delta'] = df['Zscore_Old'] - df['Zscore_Young']
    return df
```

**Acceptance criteria:**
- <5% missing transformed abundances (allow for biologically missing values)
- All z-scores computed with documented method
- Z-scores validated: mean ≈ 0, std ≈ 1 within each study

---

### Issue 4: Zero and Missing Abundance Values (Priority: HIGH)

**Problem:**
- 82 proteins (3.8%) have zero abundance in old samples
- 197 proteins (9.0%) have zero abundance in young samples
- 8 proteins have BOTH old and young = 0 (should not be in database)
- 211 proteins detected ONLY in old (18.9% in Tam_2020)
- 153 proteins detected ONLY in young (28.2% in LiDermis_2021)

**Evidence:**
```
Zero abundances:
  Old = 0:  82 (3.8%)
  Young = 0: 197 (9.0%)
  Both = 0: 8 (0.4%) ← CRITICAL

Asymmetric detection:
  Only old:   211 (9.7%)
  Only young: 153 (7.0%)

Study-specific patterns:
  LiDermis_2021: 31.3% zero in old, 8.4% zero in young
  Tam_2020:      18.9% missing in young, 6.6% missing in old
```

**Missing data mechanism assessment:**

**Type 1: Left-censored (below detection limit) - 80% of cases**
- Proteins detected in one group but missing in other WITH low abundance (< 25th percentile)
- Median abundance of "only old" proteins: 25.2 (vs 30.9 for proteins detected in both)
- **Evidence of MNAR (Missing Not At Random)**

**Type 2: Biological age-specific expression - 2% of cases**
- Proteins with high abundance (> 75th percentile) in one age group only
- Examples: VCAN (versican) - fibrotic aging marker only in old kidney
- Examples: DSPP, HRNR - developmental markers only in young disc

**Type 3: Technical artifact - 18% of cases**
- Study-specific (LiDermis_2021 has inverted pattern vs others)
- Likely due to different sample preparation or MS sensitivity

**Impact:** MODERATE - Affects statistical power but provides biological insight

**Solution:** See [Missing Data Analysis & Imputation Strategy](#missing-data-analysis--imputation-strategy) section

---

### Issue 5: Lack of Statistical Significance Indicators (Priority: CRITICAL)

**Problem:**
- No p-values for differential expression
- No FDR-corrected q-values for multiple testing
- No confidence intervals for z-score deltas
- No effect size measures (fold-change, log2FC)
- **Cannot distinguish biological significance from noise**

**Evidence:**
```
Age-related changes (based on |Zscore_Delta| > 0.5):
  Increased with age: 256/1,901 (13.5%)
  Decreased with age: 157/1,901 (8.3%)
  Unchanged:        1,488/1,901 (78.3%)

BUT: Without p-values, unclear if these are real changes or random variation
```

**Impact:** CRITICAL - Cannot make biological claims without significance testing

**Solution:**

**Step 1: Compute t-statistics for each protein**
```python
from scipy import stats

def compute_differential_expression(row):
    """
    Compute t-test and fold-change for Old vs Young.
    Requires replicate-level data or at minimum:
      - Mean abundances (have)
      - Standard deviations (need to extract)
      - Sample sizes (need for Randles_2021)
    """
    # Two-sample t-test
    # Note: Current data only has means, need SDs from source files
    mean_old = row['Abundance_Old']
    mean_young = row['Abundance_Young']
    sd_old = row['SD_Old']  # NEED TO ADD
    sd_young = row['SD_Young']  # NEED TO ADD
    n_old = row['N_Profiles_Old']
    n_young = row['N_Profiles_Young']

    # Welch's t-test (unequal variances)
    t_stat = (mean_old - mean_young) / np.sqrt((sd_old**2/n_old) + (sd_young**2/n_young))
    df = ((sd_old**2/n_old) + (sd_young**2/n_young))**2 / \
         ((sd_old**2/n_old)**2/(n_old-1) + (sd_young**2/n_young)**2/(n_young-1))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Fold-change
    fc = mean_old / mean_young
    log2fc = np.log2(fc)

    return pd.Series({
        't_statistic': t_stat,
        'p_value': p_value,
        'log2_fold_change': log2fc,
        'fold_change': fc
    })

# Apply to all proteins
stats_df = df.apply(compute_differential_expression, axis=1)
df = pd.concat([df, stats_df], axis=1)
```

**Step 2: Apply multiple testing correction**
```python
from statsmodels.stats.multitest import fdrcorrection

# FDR correction within each study (independent tests)
for study in df['Dataset_Name'].unique():
    mask = df['Dataset_Name'] == study
    p_values = df.loc[mask, 'p_value'].values

    # Benjamini-Hochberg FDR correction
    reject, q_values = fdrcorrection(p_values, alpha=0.05)
    df.loc[mask, 'q_value'] = q_values
    df.loc[mask, 'significant_FDR05'] = reject
```

**Step 3: Add effect size interpretation**
```python
# Cohen's d for effect size
def cohens_d(mean_old, mean_young, sd_old, sd_young, n_old, n_young):
    """
    Cohen's d: standardized mean difference
    Interpretation:
      |d| < 0.2: negligible
      0.2 ≤ |d| < 0.5: small
      0.5 ≤ |d| < 0.8: medium
      |d| ≥ 0.8: large
    """
    pooled_sd = np.sqrt(((n_old-1)*sd_old**2 + (n_young-1)*sd_young**2) / (n_old+n_young-2))
    d = (mean_old - mean_young) / pooled_sd
    return d

df['effect_size_cohens_d'] = df.apply(
    lambda row: cohens_d(row['Abundance_Old'], row['Abundance_Young'],
                         row['SD_Old'], row['SD_Young'],
                         row['N_Profiles_Old'], row['N_Profiles_Young']),
    axis=1
)
```

**Required new columns:**
- `SD_Old` - Standard deviation of old group (extract from source files)
- `SD_Young` - Standard deviation of young group
- `t_statistic` - t-test statistic
- `p_value` - Uncorrected p-value
- `q_value` - FDR-corrected q-value (Benjamini-Hochberg)
- `log2_fold_change` - Log2(Old/Young)
- `fold_change` - Old/Young ratio
- `effect_size_cohens_d` - Standardized effect size
- `significant_FDR05` - Boolean flag for q < 0.05

**Acceptance criteria:**
- 100% of proteins with valid raw data have p-values
- FDR correction applied within each study
- Effect sizes documented for biological interpretation
- Validation: ~5% false discovery rate at q < 0.05

---

### Issue 6: Limited Study Coverage (Priority: MODERATE)

**Problem:**
- Only 5/13 studies processed (Randles_2021, Tam_2020, Angelidis_2019, Dipali_2023, LiDermis_2021)
- 8 studies in data_raw/ not yet included
- 51% of proteins appear in only 1 study (409/795 unique proteins)
- Only 2 proteins appear in 4 studies
- Limited cross-study validation

**Evidence:**
```
Protein overlap between studies:
  1 study:  409 proteins (51.4%)
  2 studies: 247 proteins (31.1%)
  3 studies: 137 proteins (17.2%)
  4 studies:   2 proteins (0.3%)
  5 studies:   0 proteins (0.0%)

Missing studies from data_raw/:
  - Wiberg et al. (Year?)
  - McCabe et al. (Year?)
  - Naba et al. (Year?)
  - Additional 5 studies in data_raw/
```

**Impact:** MODERATE - Limits generalizability and cross-validation

**Solution:**

**Priority processing queue:**
1. **High priority:** Studies with multiple tissues (enable cross-tissue comparisons)
2. **Medium priority:** Studies with human data (current bias: 79% human, 21% mouse)
3. **Low priority:** Single-tissue studies with small sample sizes

**Processing workflow:**
```bash
# For each remaining study:
1. Extract protein abundance tables from Excel/TSV
2. Standardize to 12-column schema
3. Map protein IDs using validated pipeline (Issue 2 solution)
4. Compute statistical metrics (Issue 5 solution)
5. Validate against matrisome database
6. Merge into main database
```

**Acceptance criteria:**
- ≥10/13 studies processed (77% coverage)
- ≥30% of proteins appear in ≥3 studies (cross-validation)
- All tissues from data.csv metadata represented

---

### Issue 7: Tissue and Species Imbalance (Priority: LOW)

**Problem:**
- Heavy bias toward intervertebral disc (46% of data from Tam_2020 alone)
- 79% human, 21% mouse data
- No brain, skeletal muscle, or pancreas despite being in raw data
- Limits generalizability of aging signatures

**Evidence:**
```
Tissue distribution:
  Intervertebral disc (OAF): 376 (17.3%)
  Intervertebral disc (IAF): 317 (14.6%)
  Intervertebral disc (NP):  300 (13.8%)
  --------------------------------
  TOTAL DISC:                 993 (45.6%) ← Overrepresented

  Lung:                       291 (13.4%)
  Skin dermis:                262 (12.0%)
  Kidney (both):              458 (21.0%)
  Ovary:                      173 (7.9%)

Species:
  Homo sapiens: 1,713 (78.7%)
  Mus musculus:   464 (21.3%)
```

**Impact:** LOW - Does not affect data quality, but affects biological scope

**Solution:**
- Process remaining studies to fill tissue gaps (see Issue 6)
- Consider subsampling intervertebral disc data if statistical analyses are tissue-biased
- Add tissue-stratified analyses to account for representation differences
- Document limitation in methods/discussion

**No immediate action required** - address during study expansion

---

### Issue 8: Missing Age and Sample Metadata (Priority: MODERATE)

**Problem:**
- No actual age values in database (only binary "old" vs "young")
- No age ranges documented per study
- No information on age group definitions
- Cannot determine when during lifespan ECM changes occur
- Cannot compare "old" across studies (old mouse ≠ old human)

**Evidence:**
```
Current age information: NONE
- No "Age_Old" column
- No "Age_Young" column
- No "Age_Unit" column (years, months, weeks)
- No "Age_Definition" metadata

Expected from source papers:
  Randles_2021: Young = 21-30 years, Old = 64-83 years (human)
  Angelidis_2019: Young = 3 months, Old = 24 months (mouse)
  Tam_2020: Young = 13-28 years, Old = 51-86 years (human, spinal disc)
```

**Impact:** MODERATE - Cannot model age as continuous variable, limits biological interpretation

**Solution:**

**Add required metadata columns:**
```python
# New columns to add:
columns_to_add = {
    'Age_Young_Mean': float,        # Mean age of young group
    'Age_Young_Range': str,         # e.g., "21-30 years"
    'Age_Old_Mean': float,          # Mean age of old group
    'Age_Old_Range': str,           # e.g., "64-83 years"
    'Age_Unit': str,                # "years", "months", "weeks"
    'Percent_Lifespan_Young': float,  # Normalized to species max lifespan
    'Percent_Lifespan_Old': float,    # e.g., human 80 years, mouse 2 years
    'Age_Group_Definition': str     # Free text from paper
}
```

**Extract from source publications:**
1. Read methods sections for age group definitions
2. Calculate mean and range from participant tables
3. Normalize to species maximum lifespan for cross-species comparisons
4. Document exact text from paper in `Age_Group_Definition`

**Example implementation:**
```python
# Species maximum lifespan (years)
MAX_LIFESPAN = {
    'Homo sapiens': 120,
    'Mus musculus': 3,
    'Bos taurus': 20
}

# Calculate normalized age
def normalize_age_to_lifespan(age_years, species):
    max_life = MAX_LIFESPAN[species]
    return (age_years / max_life) * 100  # Percentage of max lifespan

# Example: Human 80 years old = 66.7% of max lifespan
# Example: Mouse 24 months old = 100% of max lifespan
```

**Acceptance criteria:**
- 100% of proteins have documented age ranges
- Cross-species age normalization implemented
- Metadata validated against source publications

---

### Issue 9: Method and Protocol Documentation Gaps (Priority: HIGH)

**Problem:**
- "Method" column only has high-level descriptions (e.g., "Label-free LC-MS/MS")
- No information on:
  - Sample preparation protocols
  - ECM extraction methods (differ between studies)
  - LC-MS/MS parameters
  - Normalization methods applied
  - Database search engines
  - FDR thresholds for peptide/protein identification
- Cannot assess technical comparability across studies

**Evidence:**
```
Current "Method" values:
  "Label-free LC-MS/MS (Progenesis + Mascot)"
  "TMT 10-plex"
  "SILAC"

Missing critical details:
  - Decellularization method (enzymatic vs detergent vs physical)
  - ECM enrichment (yes/no, method)
  - LC gradient length
  - MS instrument (Orbitrap? Q-TOF? Triple quad?)
  - Search database (UniProt version, species, reviewed only?)
  - Quantification software (MaxQuant? Proteome Discoverer? Progenesis?)
```

**Impact:** HIGH - Cannot assess batch effects or technical biases

**Solution:**

**Add metadata table: `study_methods_metadata.csv`**
```csv
Dataset_Name,Sample_Prep_Protocol,ECM_Enrichment,ECM_Enrichment_Method,LC_Gradient_Min,MS_Instrument,Quantification_Software,Search_Database,Protein_FDR,Peptide_FDR,Normalization_Method,Data_Processing_Notes,PRIDE_ID,Publication_DOI
Randles_2021,Decellularization + trypsin digestion,Yes,Sequential detergent extraction,120,Orbitrap Fusion,Progenesis QI,UniProt Human 2019_11,0.01,0.01,Total ion current,Top 3 method for absolute quantification,PXD019345,10.1038/s41467-021-XXXX
```

**Extract from publications and PRIDE:**
1. Download methods sections from each paper
2. Extract PRIDE repository README files
3. Parse MaxQuant/Progenesis parameter files if available
4. Contact authors if critical information missing

**Link to main database:**
```python
# Join method metadata to main database
df_full = df.merge(methods_metadata, on='Dataset_Name', how='left')
```

**Acceptance criteria:**
- 100% of studies have documented sample preparation protocols
- MS parameters documented for all studies
- Methods metadata validated by comparing to PRIDE submissions
- Contact authors for missing information (document attempts)

---

### Issue 10: No Data Provenance Tracking (Priority: MODERATE)

**Problem:**
- No record of:
  - When data was downloaded from PRIDE/source
  - Which Excel sheet/file each protein came from
  - Processing pipeline version
  - Manual curation decisions
- Cannot reproduce database construction
- Cannot trace errors back to source

**Evidence:**
```
Missing provenance information:
  - Source file name/path
  - Source file sheet name
  - Row number in source file
  - Date processed
  - Processing script version
  - Manual edits (if any)
```

**Impact:** MODERATE - Affects reproducibility but not data quality per se

**Solution:**

**Add provenance columns:**
```python
provenance_columns = {
    'Source_File': str,              # Original filename from data_raw/
    'Source_Sheet': str,             # Excel sheet name
    'Source_Row': int,               # Row number in source file
    'Date_Processed': str,           # ISO 8601 date
    'Processing_Script': str,        # Script version/commit hash
    'Manual_Curation': bool,         # True if manually edited
    'Curation_Notes': str,           # Free text notes
    'QC_Pass': bool,                 # Quality control flag
    'QC_Warnings': str               # Any warnings during processing
}
```

**Example entry:**
```csv
Source_File,Source_Sheet,Source_Row,Date_Processed,Processing_Script,Manual_Curation,Curation_Notes,QC_Pass,QC_Warnings
"Randles et al. - 2021/41467_2021_24638_MOESM4_ESM.xlsx","Glomerular_Proteins",127,2025-10-14,process_randles_v1.2.3,False,,True,
```

**Implementation:**
- Modify processing scripts to automatically record provenance
- Use git commit hashes for script versions
- Store in parallel provenance table (don't bloat main database)

**Acceptance criteria:**
- 100% of proteins have source file tracking
- Processing pipeline versioned in git
- Database construction fully reproducible from raw files

---

## Missing Data Analysis & Imputation Strategy

### Summary of Missing Data Patterns

**Total missing values:** 364 asymmetric detections (211 old-only + 153 young-only)

**Mechanisms identified:**

| Mechanism | Count | Percentage | Evidence |
|-----------|-------|------------|----------|
| Left-censored (MNAR) | ~169 | 46.4% | Low abundance (< 25th percentile), technical detection limit |
| Biological age-specific | ~4 | 1.1% | High abundance (> 75th percentile) in one group only |
| Random/technical | ~191 | 52.5% | Study-specific patterns, moderate abundance |

**Key findings:**
- 80% of "detected in old only" proteins have low abundance → likely below detection limit in young
- Only 1.2x fold difference in median abundance between detected vs missing groups → mostly technical, not biological
- Study-specific patterns (LiDermis: 28% missing in young, Tam_2020: 19% missing in young) → technical variation

### Three-Tier Imputation Strategy

#### **Tier 1: Keep as NaN (Biological Signal)** ✓

**When to apply:**
- High abundance in detected group (> 75th percentile of all proteins)
- Missing in other group (zero or NA)
- Potentially biologically meaningful age-specific expression

**Criteria:**
```python
def is_biological_age_specific(row, abundance_threshold_percentile=75):
    """
    Identify proteins with biologically meaningful age-specific expression.
    """
    # Get 75th percentile of all detected abundances
    threshold = df['Abundance_Old'].quantile(0.75)

    # High in old, missing in young
    high_in_old = (row['Abundance_Old'] > threshold) and \
                  (pd.isna(row['Abundance_Young']) or row['Abundance_Young'] == 0)

    # High in young, missing in old
    high_in_young = (row['Abundance_Young'] > threshold) and \
                    (pd.isna(row['Abundance_Old']) or row['Abundance_Old'] == 0)

    return high_in_old or high_in_young
```

**Proteins to preserve as NaN (~4 proteins):**
- **VCAN (versican)** - Appears only in old kidney (Randles_2021), abundance = 33.98
  - Known fibrotic aging marker, increased in kidney disease
- **DSPP (dentin sialophosphoprotein)** - Appears only in young disc (Tam_2020), abundance = 32.14
  - Developmental marker, lost with age
- **HRNR (hornerin)** - Appears only in young disc (Tam_2020), abundance = 30.84
  - Skin/epithelial differentiation marker

**Flag these proteins:**
```python
df['Missingness_Mechanism'] = 'detected_both'
df.loc[is_biological_age_specific, 'Missingness_Mechanism'] = 'biological_age_specific'
df.loc[is_biological_age_specific, 'Imputation_Applied'] = False
```

---

#### **Tier 2: Left-Censored Imputation (Technical Limitation)** 🔧

**When to apply:**
- Low abundance in detected group (< 25th percentile)
- Missing in other group
- Likely below detection limit (MNAR - left-censored data)

**Method: QRILC (Quantile Regression Imputation of Left-Censored data)**

**Why QRILC?**
- Specifically designed for proteomics left-censored data
- Models the lower tail of abundance distribution
- More realistic than simple minimum/2 imputation
- Preserves variance structure better than mean imputation

**Alternative: MinProb (Minimum Probability Imputation)**
- Draws from a truncated normal distribution
- Assumes MNAR mechanism
- Slightly more conservative than QRILC

**Implementation:**

```python
import numpy as np
from sklearn.impute import SimpleImputer

def left_censored_imputation(df, method='LOD_half'):
    """
    Impute left-censored missing values.

    Methods:
      - 'LOD_half': Limit of Detection / 2 (conservative)
      - 'LOD_sqrt2': LOD / sqrt(2) (less conservative)
      - 'min_global': Minimum detected value globally
      - 'min_per_study': Minimum detected value per study

    LOD defined as 5th percentile of detected abundances per study.
    """
    df_imputed = df.copy()

    for study in df['Dataset_Name'].unique():
        study_mask = df['Dataset_Name'] == study

        # Calculate LOD as 5th percentile of detected values
        detected_old = df.loc[study_mask & (df['Abundance_Old'] > 0), 'Abundance_Old']
        detected_young = df.loc[study_mask & (df['Abundance_Young'] > 0), 'Abundance_Young']

        LOD_old = detected_old.quantile(0.05)
        LOD_young = detected_young.quantile(0.05)

        # Identify low-abundance proteins with missing values
        low_abundance_mask = (
            study_mask &
            (
                # Low in old, missing in young
                ((df['Abundance_Old'] < df['Abundance_Old'].quantile(0.25)) &
                 ((df['Abundance_Young'] == 0) | df['Abundance_Young'].isna()))
                |
                # Low in young, missing in old
                ((df['Abundance_Young'] < df['Abundance_Young'].quantile(0.25)) &
                 ((df['Abundance_Old'] == 0) | df['Abundance_Old'].isna()))
            )
        )

        if method == 'LOD_half':
            # Impute as LOD / 2
            df_imputed.loc[low_abundance_mask & df['Abundance_Young'].isna(), 'Abundance_Young'] = LOD_young / 2
            df_imputed.loc[low_abundance_mask & df['Abundance_Old'].isna(), 'Abundance_Old'] = LOD_old / 2

        elif method == 'LOD_sqrt2':
            # Slightly less conservative
            df_imputed.loc[low_abundance_mask & df['Abundance_Young'].isna(), 'Abundance_Young'] = LOD_young / np.sqrt(2)
            df_imputed.loc[low_abundance_mask & df['Abundance_Old'].isna(), 'Abundance_Old'] = LOD_old / np.sqrt(2)

        # Flag imputed values
        df_imputed.loc[low_abundance_mask, 'Missingness_Mechanism'] = 'left_censored'
        df_imputed.loc[low_abundance_mask, 'Imputation_Applied'] = True
        df_imputed.loc[low_abundance_mask, 'Imputation_Method'] = method

    return df_imputed
```

**Alternative: R implementation using imputeLCMD package**
```R
library(imputeLCMD)

# QRILC imputation
imputed_data <- impute.QRILC(data_matrix, tune.sigma=1)

# MinProb imputation
imputed_data <- impute.MinProb(data_matrix, q=0.01)
```

**Expected impact:**
- ~169 proteins gain computable z-scores
- Missing z-scores: 12.68% → ~5% (major improvement)
- Biological signal preserved (only technical artifacts imputed)

---

#### **Tier 3: Statistical Imputation (Random Missingness)** 📊

**When to apply:**
- Moderate abundance (25th-75th percentile)
- Sporadic missingness across studies
- Protein detected in multiple studies but missing in 1-2
- Likely random technical variation

**Method: KNN Imputation (k=5)**

**Why KNN?**
- Preserves correlation structure between proteins
- Uses proteins with similar abundance profiles for imputation
- Works well for random missingness (MAR assumption)
- No assumptions about data distribution

**Alternative: missForest (Random Forest Imputation)**
- More sophisticated, handles non-linear relationships
- Slower but potentially more accurate for complex data

**Implementation:**

```python
from sklearn.impute import KNNImputer

def knn_imputation(df, n_neighbors=5):
    """
    KNN imputation for random missingness.
    Uses k=5 nearest neighbors with similar abundance profiles.
    """
    df_imputed = df.copy()

    # Only apply to moderate-abundance, sporadic missingness
    moderate_abundance_mask = (
        (df['Abundance_Old'] >= df['Abundance_Old'].quantile(0.25)) &
        (df['Abundance_Old'] <= df['Abundance_Old'].quantile(0.75))
    )

    # Subset to proteins with moderate abundance
    moderate_proteins = df[moderate_abundance_mask]

    # Prepare abundance matrix (proteins x 2 columns: Old, Young)
    abundance_matrix = moderate_proteins[['Abundance_Old', 'Abundance_Young']].values

    # KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    abundance_imputed = imputer.fit_transform(abundance_matrix)

    # Replace imputed values
    df_imputed.loc[moderate_abundance_mask, 'Abundance_Old'] = abundance_imputed[:, 0]
    df_imputed.loc[moderate_abundance_mask, 'Abundance_Young'] = abundance_imputed[:, 1]

    # Flag imputed values
    was_missing_old = df.loc[moderate_abundance_mask, 'Abundance_Old'].isna()
    was_missing_young = df.loc[moderate_abundance_mask, 'Abundance_Young'].isna()

    df_imputed.loc[moderate_abundance_mask & was_missing_old, 'Missingness_Mechanism'] = 'random_technical'
    df_imputed.loc[moderate_abundance_mask & was_missing_old, 'Imputation_Applied'] = True
    df_imputed.loc[moderate_abundance_mask & was_missing_old, 'Imputation_Method'] = f'KNN_k{n_neighbors}'

    return df_imputed
```

**Expected impact:**
- ~20-30 proteins gain improved z-score estimates
- Reduces noise in differential expression analysis
- Preserves biological signal

---

### Recommended Implementation: Hybrid Approach

**Create two database versions:**

1. **`merged_ecm_aging_zscore_raw.csv`** (Current version)
   - Keep all NaN values as-is
   - For users who want complete control over imputation

2. **`merged_ecm_aging_zscore_imputed.csv`** (Recommended for most analyses)
   - Apply Tier 1 (preserve biological NAs)
   - Apply Tier 2 (left-censored imputation)
   - Optionally apply Tier 3 (KNN imputation)
   - Add imputation flags for transparency

**Required new columns:**
```python
imputation_columns = {
    'Abundance_Old_imputed': bool,           # Was this value imputed?
    'Abundance_Young_imputed': bool,         # Was this value imputed?
    'Missingness_Mechanism': str,            # 'biological_age_specific', 'left_censored', 'random_technical', 'detected_both'
    'Imputation_Applied': bool,              # True if any imputation applied
    'Imputation_Method': str,                # 'none', 'LOD_half', 'QRILC', 'KNN_k5', etc.
    'Abundance_Old_original': float,         # Original value before imputation (for transparency)
    'Abundance_Young_original': float        # Original value before imputation
}
```

**Database file structure:**
```
08_merged_ecm_dataset/
├── merged_ecm_aging_zscore_raw.csv           # Original, no imputation
├── merged_ecm_aging_zscore_imputed.csv       # With Tier 2+3 imputation
├── imputation_report.csv                     # Summary of imputation decisions
└── imputation_methods.md                     # Documentation of methods
```

**Imputation report structure:**
```csv
Protein_ID,Gene_Symbol,Dataset_Name,Tissue,Missingness_Mechanism,Imputation_Method,Abundance_Old_Original,Abundance_Old_Imputed,Abundance_Young_Original,Abundance_Young_Imputed
A0A024R6I7,SERPINA1,Randles_2021,Kidney_Glomerular,detected_both,none,10523.30,10523.30,21467.44,21467.44
P02461,COL3A1,Randles_2021,Kidney_Glomerular,biological_age_specific,none,33.98,33.98,NA,NA
Q99715,COL12A1,Tam_2020,Intervertebral_disc_OAF,left_censored,LOD_half,NA,12.34,25.67,25.67
```

---

### Validation Strategy

**After imputation, validate that:**

1. **Imputation doesn't create artifacts:**
```python
# Check that imputed values are lower than detected values
imputed_old = df[df['Abundance_Old_imputed'] == True]['Abundance_Old']
detected_old = df[df['Abundance_Old_imputed'] == False]['Abundance_Old']
assert imputed_old.median() < detected_old.quantile(0.25), "Imputed values too high!"
```

2. **Variance structure is preserved:**
```python
# Compare variance before and after imputation
var_before = df_raw['Zscore_Delta'].var()
var_after = df_imputed['Zscore_Delta'].var()
print(f"Variance change: {(var_after/var_before - 1)*100:.1f}%")
# Expected: < 10% change
```

3. **Biological signal is not inflated:**
```python
# Compare number of significant proteins before/after
sig_before = (df_raw['q_value'] < 0.05).sum()
sig_after = (df_imputed['q_value'] < 0.05).sum()
print(f"Significant proteins: {sig_before} → {sig_after}")
# Expected: ≤ 10% increase
```

4. **Cross-study consistency:**
```python
# Proteins appearing in multiple studies should have consistent direction
multi_study_proteins = df.groupby('Canonical_Gene_Symbol').filter(lambda x: len(x) >= 2)
# Check if Zscore_Delta has same sign across studies
```

---

### Acceptance Criteria for Imputation

- [ ] <5% of proteins imputed with biological_age_specific mechanism
- [ ] ~8% of proteins imputed with left_censored mechanism
- [ ] ~1% of proteins imputed with random_technical mechanism
- [ ] Imputation flags present for 100% of proteins
- [ ] Validation tests pass (no artifacts, variance preserved, signal not inflated)
- [ ] Imputation methods documented in accompanying methods file
- [ ] Both raw and imputed databases available
- [ ] Imputation report generated showing all decisions

---

## Statistical Rigor Requirements

### Required Statistical Tests

For database to be publication-ready, must include:

1. **Differential expression testing**
   - Two-sample t-test (Welch's for unequal variance)
   - Requires: mean, SD, sample size for each group
   - Output: t-statistic, p-value, degrees of freedom

2. **Multiple testing correction**
   - Benjamini-Hochberg FDR correction (preferred for proteomics)
   - Apply within each study (tests are independent across studies)
   - Output: q-value, significance flag at α=0.05

3. **Effect size estimation**
   - Cohen's d (standardized mean difference)
   - Log2 fold-change (interpretable in biology)
   - Output: effect size with interpretation (small/medium/large)

4. **Confidence intervals**
   - 95% CI for mean difference
   - 95% CI for fold-change
   - Output: lower and upper bounds

### Required Data Extraction from Source Files

**Currently missing from source files:**

| Data Element | Required For | Availability |
|--------------|--------------|--------------|
| Standard deviations (SD) | t-test, effect size | Must extract from source files |
| Sample sizes (Randles_2021) | All statistical tests | In paper methods section |
| Replicate-level data | Variance estimation | May need PRIDE download |
| Age ranges | Metadata | In paper participant tables |
| Sex distribution | Subgroup analysis | In paper demographics |

**Extraction priority:**
1. **CRITICAL:** Sample sizes for Randles_2021 (458 proteins affected)
2. **CRITICAL:** Standard deviations for all studies
3. **HIGH:** Age ranges for all studies
4. **MODERATE:** Sex distribution (for sensitivity analysis)
5. **LOW:** Replicate-level data (only if SD not available)

### Implementation: Statistical Testing Pipeline

```python
def compute_comprehensive_statistics(df):
    """
    Compute all statistical tests and effect sizes for each protein.

    Requires columns:
      - Abundance_Old, Abundance_Young
      - SD_Old, SD_Young (need to add)
      - N_Profiles_Old, N_Profiles_Young

    Returns DataFrame with added columns:
      - t_statistic, df, p_value
      - q_value, significant_FDR05
      - log2_fold_change, fold_change
      - cohens_d, effect_size_interpretation
      - ci_lower, ci_upper
    """
    from scipy import stats
    from statsmodels.stats.multitest import fdrcorrection
    import numpy as np

    # Check required columns exist
    required = ['Abundance_Old', 'Abundance_Young', 'SD_Old', 'SD_Young',
                'N_Profiles_Old', 'N_Profiles_Young']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []

    for idx, row in df.iterrows():
        # Extract values
        mean_old = row['Abundance_Old']
        mean_young = row['Abundance_Young']
        sd_old = row['SD_Old']
        sd_young = row['SD_Young']
        n_old = row['N_Profiles_Old']
        n_young = row['N_Profiles_Young']

        # Skip if any required values are missing
        if pd.isna([mean_old, mean_young, sd_old, sd_young, n_old, n_young]).any():
            results.append({
                't_statistic': np.nan,
                'df': np.nan,
                'p_value': np.nan,
                'log2_fold_change': np.nan,
                'fold_change': np.nan,
                'cohens_d': np.nan,
                'effect_size_interpretation': 'insufficient_data',
                'ci_lower': np.nan,
                'ci_upper': np.nan
            })
            continue

        # Welch's t-test (unequal variances)
        se_old = sd_old / np.sqrt(n_old)
        se_young = sd_young / np.sqrt(n_young)
        se_diff = np.sqrt(se_old**2 + se_young**2)

        t_stat = (mean_old - mean_young) / se_diff

        # Welch-Satterthwaite degrees of freedom
        df_welch = (se_old**2 + se_young**2)**2 / \
                   ((se_old**2)**2/(n_old-1) + (se_young**2)**2/(n_young-1))

        # Two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))

        # Fold-change
        fc = mean_old / mean_young if mean_young > 0 else np.nan
        log2fc = np.log2(fc) if fc > 0 else np.nan

        # Cohen's d
        pooled_sd = np.sqrt(((n_old-1)*sd_old**2 + (n_young-1)*sd_young**2) / (n_old+n_young-2))
        cohens_d = (mean_old - mean_young) / pooled_sd

        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        # 95% Confidence interval for mean difference
        t_crit = stats.t.ppf(0.975, df_welch)  # 95% CI
        ci_lower = (mean_old - mean_young) - t_crit * se_diff
        ci_upper = (mean_old - mean_young) + t_crit * se_diff

        results.append({
            't_statistic': t_stat,
            'df': df_welch,
            'p_value': p_value,
            'log2_fold_change': log2fc,
            'fold_change': fc,
            'cohens_d': cohens_d,
            'effect_size_interpretation': interpretation,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    # Add results to dataframe
    results_df = pd.DataFrame(results)
    df = pd.concat([df, results_df], axis=1)

    # FDR correction within each study
    df['q_value'] = np.nan
    df['significant_FDR05'] = False

    for study in df['Dataset_Name'].unique():
        mask = (df['Dataset_Name'] == study) & df['p_value'].notna()
        if mask.sum() == 0:
            continue

        p_values = df.loc[mask, 'p_value'].values
        reject, q_values = fdrcorrection(p_values, alpha=0.05)

        df.loc[mask, 'q_value'] = q_values
        df.loc[mask, 'significant_FDR05'] = reject

    return df
```

### Validation of Statistical Results

**After computing statistics, validate:**

```python
def validate_statistical_results(df):
    """
    Quality control checks for statistical testing.
    """
    print("=" * 80)
    print("STATISTICAL VALIDATION REPORT")
    print("=" * 80)

    # 1. Check p-value distribution
    print("\n1. P-VALUE DISTRIBUTION (should be uniform under null)")
    print(df['p_value'].describe())

    # Under null hypothesis, p-values should be uniformly distributed
    # If skewed toward 0, indicates many true positives (good!)

    # 2. Check FDR control
    print("\n2. FDR CONTROL (should be ~5% false positives at q<0.05)")
    n_significant = df['significant_FDR05'].sum()
    n_total = df['p_value'].notna().sum()
    print(f"Significant proteins: {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)")

    # 3. Check effect sizes
    print("\n3. EFFECT SIZE DISTRIBUTION")
    print(df['effect_size_interpretation'].value_counts())

    # 4. Validate consistency with z-scores
    print("\n4. Z-SCORE VS P-VALUE CONSISTENCY")
    # Proteins with large |Zscore_Delta| should have small p-values
    high_zscore = df['Zscore_Delta'].abs() > 1
    significant = df['significant_FDR05']
    overlap = (high_zscore & significant).sum()
    print(f"Proteins with |z| > 1 AND q < 0.05: {overlap}")

    # 5. Check for outliers
    print("\n5. OUTLIER DETECTION")
    extreme_fc = (df['log2_fold_change'].abs() > 5).sum()
    extreme_d = (df['cohens_d'].abs() > 3).sum()
    print(f"Extreme fold-changes (|log2FC| > 5): {extreme_fc}")
    print(f"Extreme effect sizes (|d| > 3): {extreme_d}")

    # 6. Cross-study consistency
    print("\n6. CROSS-STUDY CONSISTENCY")
    multi_study = df.groupby('Canonical_Gene_Symbol').filter(lambda x: len(x) >= 2)
    if len(multi_study) > 0:
        consistency = multi_study.groupby('Canonical_Gene_Symbol').apply(
            lambda x: (np.sign(x['Zscore_Delta']).nunique() == 1)
        )
        consistent_pct = consistency.sum() / len(consistency) * 100
        print(f"Proteins with consistent direction across studies: {consistent_pct:.1f}%")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
```

### Acceptance Criteria for Statistical Testing

- [ ] 100% of proteins with valid data have p-values and q-values
- [ ] FDR-corrected significance flags present
- [ ] Effect sizes (Cohen's d, log2FC) computed for all proteins
- [ ] Confidence intervals present
- [ ] P-value distribution validated (not systematically biased)
- [ ] ~5% false discovery rate at q < 0.05 (validation via decoy proteins if available)
- [ ] Statistical methods documented in methods metadata
- [ ] Validation report generated and reviewed

---

## Protein Identifier Validation

### Current State Assessment

**Problem severity: CRITICAL**

- 726/2,177 proteins (33%) have confidence score = 1 (lowest)
- Match_Level = "Gene_Symbol_or_UniProt" indicates ambiguous mapping
- 24.5% have Gene_Symbol ≠ Canonical_Gene_Symbol (potential errors)

**Root causes:**
1. Original files use heterogeneous ID formats (UniProt, Gene Symbol, Ensembl)
2. Automated mapping pipeline has fallback to low-confidence matches
3. No manual validation of low-confidence matches
4. Synonym mappings not validated (e.g., ARG1 ≠ TINAGL1)

### Validation Pipeline

**Step 1: Extract original identifiers from source files**

```python
def extract_original_ids(source_file, source_sheet):
    """
    Re-read original protein IDs from source Excel files.
    Different studies use different ID formats.
    """
    import pandas as pd

    df_source = pd.read_excel(source_file, sheet_name=source_sheet)

    # Common column names for protein IDs
    possible_id_columns = [
        'Protein IDs', 'Protein ID', 'UniProt ID', 'Accession',
        'Gene Symbol', 'Gene Name', 'Protein', 'Majority protein IDs'
    ]

    # Find which column contains protein IDs
    id_column = None
    for col in possible_id_columns:
        if col in df_source.columns:
            id_column = col
            break

    if id_column is None:
        raise ValueError(f"Could not find protein ID column in {source_file}")

    return df_source[id_column].tolist()
```

**Step 2: Re-map using authoritative databases**

```python
import requests
import time

def query_uniprot_api(identifier, species='Homo sapiens'):
    """
    Query UniProt REST API for protein information.
    Returns authoritative gene symbol, UniProt ID, protein name.
    """
    # UniProt API endpoint
    url = 'https://rest.uniprot.org/uniprotkb/search'

    # Try multiple query strategies
    queries = [
        f'accession:{identifier} AND organism_name:"{species}"',
        f'gene_exact:{identifier} AND organism_name:"{species}"',
        f'{identifier} AND organism_name:"{species}"'
    ]

    for query in queries:
        params = {
            'query': query,
            'format': 'json',
            'fields': 'accession,gene_names,protein_name,organism_name',
            'size': 1
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                return {
                    'UniProt_ID': result['primaryAccession'],
                    'Gene_Symbol': result['genes'][0]['geneName']['value'] if result.get('genes') else None,
                    'Protein_Name': result['proteinDescription']['recommendedName']['fullName']['value'],
                    'Species': result['organism']['scientificName'],
                    'Match_Confidence': 100,
                    'Match_Method': 'UniProt_API'
                }

        time.sleep(0.1)  # Rate limiting

    # No match found
    return None

def validate_and_remap_proteins(df):
    """
    Validate all low-confidence protein IDs and re-map.
    """
    # Focus on low-confidence proteins
    low_conf = df[df['Match_Confidence'] < 100].copy()

    print(f"Re-mapping {len(low_conf)} low-confidence proteins...")

    remapped = []

    for idx, row in low_conf.iterrows():
        original_id = row['Protein_ID']  # Could be UniProt, Gene Symbol, etc.
        species = row['Species']

        # Query UniProt
        result = query_uniprot_api(original_id, species)

        if result:
            remapped.append({
                'Original_Row': idx,
                'Original_ID': original_id,
                'New_UniProt_ID': result['UniProt_ID'],
                'New_Gene_Symbol': result['Gene_Symbol'],
                'New_Protein_Name': result['Protein_Name'],
                'New_Match_Confidence': result['Match_Confidence'],
                'Validation_Method': result['Match_Method']
            })
        else:
            # Manual review required
            remapped.append({
                'Original_Row': idx,
                'Original_ID': original_id,
                'New_UniProt_ID': None,
                'New_Gene_Symbol': None,
                'New_Protein_Name': None,
                'New_Match_Confidence': 0,
                'Validation_Method': 'MANUAL_REVIEW_REQUIRED'
            })

    return pd.DataFrame(remapped)
```

**Step 3: Validate synonym mappings**

```python
def validate_synonyms(df):
    """
    Check that synonym mappings are correct.
    """
    synonyms = df[df['Match_Level'].str.contains('synonym', na=False)].copy()

    print(f"Validating {len(synonyms)} synonym mappings...")

    validation_results = []

    for idx, row in synonyms.iterrows():
        gene_symbol = row['Gene_Symbol']
        canonical_symbol = row['Canonical_Gene_Symbol']
        species = row['Species']

        # Query UniProt for both symbols
        result_original = query_uniprot_api(gene_symbol, species)
        result_canonical = query_uniprot_api(canonical_symbol, species)

        # Check if they map to the same UniProt ID
        if result_original and result_canonical:
            same_protein = (result_original['UniProt_ID'] == result_canonical['UniProt_ID'])

            validation_results.append({
                'Gene_Symbol': gene_symbol,
                'Canonical_Gene_Symbol': canonical_symbol,
                'UniProt_Original': result_original['UniProt_ID'],
                'UniProt_Canonical': result_canonical['UniProt_ID'],
                'Is_Valid_Synonym': same_protein,
                'Correct_Symbol': result_original['Gene_Symbol']  # Authoritative symbol
            })
        else:
            validation_results.append({
                'Gene_Symbol': gene_symbol,
                'Canonical_Gene_Symbol': canonical_symbol,
                'UniProt_Original': None,
                'UniProt_Canonical': None,
                'Is_Valid_Synonym': False,
                'Correct_Symbol': None
            })

    return pd.DataFrame(validation_results)
```

**Step 4: Add orthology mapping**

```python
def add_orthology_mapping(df):
    """
    Map mouse proteins to human orthologs for cross-species comparison.
    Uses Ensembl Compara REST API.
    """
    import requests

    mouse_proteins = df[df['Species'] == 'Mus musculus'].copy()

    orthology_map = []

    for idx, row in mouse_proteins.iterrows():
        gene_symbol = row['Canonical_Gene_Symbol']

        # Query Ensembl Compara
        url = f'https://rest.ensembl.org/homology/symbol/mus_musculus/{gene_symbol}'
        params = {
            'target_species': 'homo_sapiens',
            'type': 'orthologues',
            'format': 'json'
        }

        response = requests.get(url, params=params, headers={'Content-Type': 'application/json'})

        if response.status_code == 200:
            data = response.json()
            if data['data']:
                homologies = data['data'][0]['homologies']
                if homologies:
                    human_ortholog = homologies[0]['target']['id']  # Ensembl ID
                    human_symbol = homologies[0]['target']['symbol']

                    orthology_map.append({
                        'Row_Index': idx,
                        'Mouse_Gene': gene_symbol,
                        'Human_Ortholog_Gene': human_symbol,
                        'Human_Ortholog_Ensembl': human_ortholog,
                        'Orthology_Confidence': homologies[0]['target']['perc_id']
                    })

        time.sleep(0.1)  # Rate limiting

    # Add Human_Ortholog_Gene column
    orthology_df = pd.DataFrame(orthology_map)
    df = df.merge(orthology_df[['Row_Index', 'Human_Ortholog_Gene', 'Orthology_Confidence']],
                  left_index=True, right_on='Row_Index', how='left')

    return df
```

### Manual Review Protocol

**For proteins that cannot be automatically validated (expected: ~5%):**

1. **Create manual review spreadsheet:**
```csv
Row_ID,Original_ID,Current_Gene_Symbol,Current_UniProt_ID,Species,Dataset_Name,Manual_UniProt_ID,Manual_Gene_Symbol,Reviewer_Name,Review_Date,Notes
1234,P12345_MOUSE,Ambiguous_Gene,P12345,Mus musculus,Tam_2020,P12345,Correct_Gene,John Doe,2025-10-14,Verified against MGI database
```

2. **Manual validation steps:**
   - Search UniProt with multiple queries (ID, gene symbol, protein name)
   - Cross-reference with HGNC (human) or MGI (mouse) databases
   - Check original publication supplementary tables
   - Contact study authors if still ambiguous
   - Document decision and rationale

3. **Expert review:**
   - Have ECM biology expert review all ambiguous ECM core proteins
   - Prioritize collagens, laminins, major ECM glycoproteins (high biological impact)

### Implementation Timeline

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 1 | Extract original IDs from source files | Data Engineer | `original_ids.csv` |
| 1-2 | Run UniProt API re-mapping | Data Engineer | `remapped_proteins.csv` |
| 2 | Validate synonym mappings | Data Engineer | `synonym_validation.csv` |
| 2-3 | Add orthology mapping | Bioinformatician | Updated database with `Human_Ortholog_Gene` |
| 3 | Manual review of ambiguous proteins | ECM Expert + Bioinformatician | `manual_review_complete.csv` |
| 4 | Update main database with validated IDs | Data Engineer | `merged_ecm_aging_zscore_v2.0.csv` |

### Acceptance Criteria

- [ ] <5% of proteins have Match_Confidence < 100
- [ ] All synonym mappings validated against UniProt
- [ ] Incorrect synonyms corrected (e.g., ARG1 ≠ TINAGL1)
- [ ] 100% of proteins have authoritative UniProt ID
- [ ] 100% of proteins have authoritative gene symbol (HGNC/MGI)
- [ ] Mouse proteins have human ortholog mapping (for cross-species analysis)
- [ ] Manual review completed for all ambiguous cases
- [ ] Mapping provenance documented (original ID → method → final ID)
- [ ] Validation report generated with QC metrics

---

## Metadata Standardization

### Required Metadata Additions

**Currently missing critical metadata:**

1. **Age information** (see Issue 8)
2. **Method details** (see Issue 9)
3. **Sample characteristics** (sex, health status, anatomical specificity)
4. **Statistical metadata** (sample sizes, standard deviations)

### Metadata Structure

**Create separate metadata tables linked by Dataset_Name:**

#### **1. study_metadata.csv**
```csv
Dataset_Name,Publication_DOI,PMID,First_Author,Publication_Year,Journal,Title,PRIDE_ID,Data_Availability_URL,Processed_Date,Processing_Version
Randles_2021,10.1038/s41467-021-24638-4,34326338,Randles,2021,Nature Communications,Molecular aging of the glomerular filtration barrier,PXD019345,https://www.ebi.ac.uk/pride/archive/projects/PXD019345,2025-10-14,v1.0
```

#### **2. methods_metadata.csv**
```csv
Dataset_Name,Sample_Type,Tissue_Prep_Protocol,ECM_Enrichment_YN,ECM_Enrichment_Method,Decellularization_Method,Digestion_Enzyme,LC_Gradient_Min,MS_Instrument,MS_Instrument_Vendor,Quantification_Method,Quantification_Software,Search_Engine,Search_Database,Search_Database_Version,Protein_FDR_Threshold,Peptide_FDR_Threshold,Min_Peptides_Per_Protein,Normalization_Method,Data_Processing_Notes
Randles_2021,Fresh frozen tissue,Decellularization,Yes,Sequential detergent extraction,"0.5% SDS, 1% Triton X-100",Trypsin,120,Orbitrap Fusion Lumos,Thermo Fisher,Label-free (LFQ),Progenesis QI,Mascot,UniProt Human,2019_11,0.01,0.01,2,Total ion current,Top 3 method for absolute quantification
```

#### **3. age_metadata.csv**
```csv
Dataset_Name,Species,Young_N_Donors,Young_Age_Mean,Young_Age_Median,Young_Age_Range,Young_Age_SD,Old_N_Donors,Old_Age_Mean,Old_Age_Median,Old_Age_Range,Old_Age_SD,Age_Unit,Species_Max_Lifespan_Years,Young_Percent_Lifespan,Old_Percent_Lifespan,Age_Definition_Source
Randles_2021,Homo sapiens,3,25.3,24,21-30,4.5,3,72.7,71,64-83,9.6,years,120,21.1,60.6,Extended Data Table 1
Angelidis_2019,Mus musculus,4,3,3,3-3,0,4,24,24,24-24,0,months,3,8.3,66.7,Methods section page 4
```

#### **4. sample_metadata.csv** (if individual sample data available)
```csv
Dataset_Name,Sample_ID,Age_Group,Age,Age_Unit,Sex,Health_Status,Anatomical_Location,Technical_Replicate,Biological_Replicate,Batch,Notes
Randles_2021,Glom_Y_1,Young,21,years,M,Healthy,Kidney glomerulus,1,1,Batch_A,
Randles_2021,Glom_Y_2,Young,24,years,F,Healthy,Kidney glomerulus,1,2,Batch_A,
```

#### **5. protein_statistics_metadata.csv**
```csv
Dataset_Name,Protein_ID,Gene_Symbol,Mean_Old,Mean_Young,SD_Old,SD_Young,SEM_Old,SEM_Young,N_Old,N_Young,Median_Old,Median_Young,IQR_Old,IQR_Young,CV_Old,CV_Young
Randles_2021,P02461,COL3A1,8145419.72,3567283.73,1234567.89,987654.32,712345.67,569876.54,3,3,8234567.89,3456789.01,2345678.90,1234567.89,0.15,0.28
```

### Metadata Extraction Workflow

```python
def extract_metadata_from_publication(doi):
    """
    Extract metadata from publication using CrossRef and PubMed APIs.
    """
    import requests

    # Get publication details from CrossRef
    crossref_url = f'https://api.crossref.org/works/{doi}'
    response = requests.get(crossref_url)

    if response.status_code == 200:
        data = response.json()['message']
        return {
            'DOI': doi,
            'Title': data['title'][0],
            'Journal': data['container-title'][0] if 'container-title' in data else None,
            'Publication_Year': data['published']['date-parts'][0][0],
            'First_Author': data['author'][0]['family'] if 'author' in data else None,
            'PMID': data.get('PMID', None)
        }

    return None

def extract_age_from_methods(pdf_path, study_name):
    """
    Extract age information from methods section.
    Requires manual review or NLP extraction.
    """
    # This would use PDF parsing + NLP to extract age ranges
    # For now, manual extraction recommended
    pass

def extract_methods_from_pride(pride_id):
    """
    Download and parse PRIDE repository files for methods details.
    """
    import requests

    pride_url = f'https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{pride_id}'
    response = requests.get(pride_url)

    if response.status_code == 200:
        data = response.json()
        return {
            'PRIDE_ID': pride_id,
            'Instruments': data.get('instruments', []),
            'Quantification_Methods': data.get('quantificationMethods', []),
            'Sample_Protocol': data.get('sampleProcessingProtocol', None),
            'Data_Protocol': data.get('dataProcessingProtocol', None)
        }

    return None
```

### Manual Metadata Curation

**For each of the 5 studies, manually extract:**

1. **From main text and methods:**
   - Age ranges and sample sizes
   - Sample preparation protocols
   - ECM enrichment methods
   - MS instrument and parameters

2. **From supplementary tables:**
   - Individual sample characteristics
   - Protein-level statistics (SD, SEM)
   - Quality control metrics

3. **From PRIDE repositories:**
   - Raw data availability
   - Search parameters
   - FDR thresholds

**Metadata extraction checklist per study:**

```markdown
## Randles et al. 2021 - Metadata Extraction

### Publication Information
- [ ] DOI: 10.1038/s41467-021-24638-4
- [ ] PMID: 34326338
- [ ] First author: Randles
- [ ] Year: 2021
- [ ] Journal: Nature Communications

### Age Information
- [ ] Young: mean = ___, range = ___, n = ___
- [ ] Old: mean = ___, range = ___, n = ___
- [ ] Source: Extended Data Table 1 / Methods section

### Sample Information
- [ ] Number of biological replicates: ___
- [ ] Sex distribution: ___
- [ ] Health status: Healthy / Disease
- [ ] Anatomical compartments: Glomerular, Tubulointerstitial

### Methods Details
- [ ] MS instrument: Orbitrap Fusion Lumos
- [ ] Quantification: Label-free (Progenesis QI)
- [ ] Search engine: Mascot
- [ ] Database: UniProt Human 2019_11
- [ ] Protein FDR: 1%
- [ ] Sample prep: Sequential detergent extraction

### Statistical Data
- [ ] Standard deviations available: Yes/No
- [ ] If yes, location: Supplementary Table X
- [ ] If no, need to calculate from replicates

### Data Availability
- [ ] PRIDE ID: PXD019345
- [ ] Raw data accessible: Yes
- [ ] Processed data format: Excel
```

**Repeat for all 5 studies.**

### Acceptance Criteria for Metadata

- [ ] 100% of studies have complete study_metadata.csv entries
- [ ] 100% of studies have methods_metadata.csv with ≥15/19 fields filled
- [ ] 100% of studies have age_metadata.csv with all age ranges documented
- [ ] ≥80% of proteins have protein_statistics_metadata.csv (SD, SEM)
- [ ] Sample-level metadata available for ≥3 studies (if data permits)
- [ ] All metadata validated against source publications
- [ ] Metadata extraction documented with source references

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Weeks 1-2)

**Goal:** Address issues that prevent statistical inference

**Tasks:**
1. ✅ Extract sample sizes for Randles_2021 (Issue 1)
   - Review paper methods and PRIDE metadata
   - Contact authors if not clearly stated
   - Target: 100% sample sizes documented

2. ✅ Extract standard deviations for all studies (Issue 5)
   - Parse supplementary tables
   - Calculate from replicate data if needed
   - Target: ≥90% proteins with SD data

3. ✅ Compute statistical tests (Issue 5)
   - Implement t-test pipeline
   - Apply FDR correction
   - Calculate effect sizes
   - Target: All proteins with complete data have p-values

**Deliverables:**
- `merged_ecm_aging_zscore_v1.1.csv` with complete sample sizes
- `protein_statistics_metadata.csv` with SD/SEM data
- Statistical testing pipeline script (`compute_statistics.py`)

**Success criteria:**
- 0% missing sample sizes
- ≥90% proteins with statistical test results
- Validation report showing proper FDR control

---

### Phase 2: Data Quality Improvements (Weeks 3-4)

**Goal:** Improve protein ID confidence and metadata completeness

**Tasks:**
1. ✅ Re-map low-confidence protein IDs (Issue 2)
   - Run UniProt API validation
   - Validate synonym mappings
   - Manual review of remaining ambiguous IDs
   - Target: <5% low-confidence proteins

2. ✅ Implement imputation strategy (Issue 4)
   - Apply Tier 1 (preserve biological NAs)
   - Apply Tier 2 (left-censored imputation)
   - Generate imputation report
   - Target: ~8% proteins imputed, 0% artifacts

3. ✅ Add age and methods metadata (Issues 8-9)
   - Extract from publications
   - Create metadata tables
   - Link to main database
   - Target: 100% studies with complete metadata

**Deliverables:**
- `merged_ecm_aging_zscore_v2.0.csv` with validated protein IDs
- `merged_ecm_aging_zscore_imputed.csv` with imputation flags
- `study_metadata.csv`, `age_metadata.csv`, `methods_metadata.csv`
- Protein ID validation report

**Success criteria:**
- ≥95% protein IDs have confidence = 100
- Imputation applied to appropriate proteins only
- All critical metadata fields populated

---

### Phase 3: Database Expansion (Weeks 5-8)

**Goal:** Process remaining studies to increase coverage

**Tasks:**
1. ✅ Process 5 additional studies from data_raw/ (Issue 6)
   - Prioritize studies with new tissues
   - Apply validated processing pipeline
   - Ensure consistent data standards
   - Target: 10/13 studies included (77%)

2. ✅ Add cross-study validation
   - Identify proteins appearing in multiple studies
   - Compare effect directions
   - Flag inconsistencies for review
   - Target: ≥30% proteins in ≥3 studies

3. ✅ Implement quality control checks
   - Automated QC pipeline
   - Outlier detection
   - Cross-study batch effect assessment
   - Target: QC report for all proteins

**Deliverables:**
- `merged_ecm_aging_zscore_v3.0.csv` with 10 studies
- Cross-study validation report
- Automated QC pipeline scripts

**Success criteria:**
- 77% study coverage (10/13)
- Cross-study consistency ≥70%
- QC checks pass for ≥95% proteins

---

### Phase 4: Documentation & Public Release (Weeks 9-10)

**Goal:** Prepare database for publication and public use

**Tasks:**
1. ✅ Write comprehensive documentation
   - README with usage examples
   - Methods documentation with citations
   - Data dictionary for all columns
   - Tutorial notebooks

2. ✅ Create data release package
   - Multiple database versions (raw, imputed, validated)
   - Metadata tables
   - QC reports
   - Processing scripts (reproducibility)

3. ✅ Set up public repository
   - GitHub repository with data and code
   - Zenodo DOI for citability
   - License (CC-BY 4.0 or CC0 recommended)

4. ✅ Prepare manuscript
   - Database description paper
   - Validation analyses
   - Example use cases
   - Biological insights

**Deliverables:**
- Complete data package on Zenodo
- GitHub repository with documentation
- Manuscript submitted to database journal (e.g., Scientific Data, Nucleic Acids Research)

**Success criteria:**
- All QC checks pass
- Documentation reviewed by external users
- Public repository functional
- Manuscript drafted

---

### Timeline Summary

| Phase | Duration | Key Milestones | Blocking Issues |
|-------|----------|----------------|-----------------|
| Phase 1: Critical Fixes | Weeks 1-2 | Complete statistics, sample sizes | None |
| Phase 2: Quality Improvements | Weeks 3-4 | Validated IDs, metadata complete | Need author contact for some metadata |
| Phase 3: Database Expansion | Weeks 5-8 | 10 studies processed, QC complete | Requires time for data processing |
| Phase 4: Public Release | Weeks 9-10 | Published, DOI, manuscript submitted | Need journal selection |

**Total estimated time: 10 weeks (2.5 months)**

---

## Quality Control Checklist

### Pre-Release QC Checklist

**Use this checklist before public release:**

#### Data Completeness
- [ ] ≥95% of proteins have non-missing abundance values
- [ ] 100% of proteins have documented sample sizes
- [ ] ≥90% of proteins have standard deviations
- [ ] 100% of proteins have species and tissue annotations
- [ ] 100% of proteins have matrisome category assignments

#### Data Quality
- [ ] <5% of proteins have Match_Confidence < 100
- [ ] 0 duplicate entries (same protein-tissue-study combination)
- [ ] 0 proteins with both old and young abundance = 0
- [ ] ≥95% of proteins have valid z-scores
- [ ] P-value distribution not systematically biased

#### Statistical Rigor
- [ ] 100% of proteins with complete data have p-values
- [ ] FDR correction applied within each study
- [ ] Effect sizes computed for all proteins
- [ ] Confidence intervals present
- [ ] ~5% false discovery rate at q < 0.05 (validated)

#### Metadata Completeness
- [ ] 100% of studies have publication metadata (DOI, PMID, year)
- [ ] 100% of studies have age range documentation
- [ ] 100% of studies have methods documentation
- [ ] ≥80% of studies have sample-level metadata
- [ ] 100% of studies have PRIDE/repository links

#### Imputation
- [ ] Imputation flags present for all proteins
- [ ] Imputation mechanism documented
- [ ] Both raw and imputed versions available
- [ ] Imputation validation passed (no artifacts)
- [ ] <10% of proteins imputed overall

#### Cross-Study Consistency
- [ ] ≥30% of proteins appear in ≥3 studies
- [ ] ≥70% of multi-study proteins have consistent effect direction
- [ ] Batch effects assessed and documented
- [ ] Study-specific biases documented

#### Documentation
- [ ] README with clear usage instructions
- [ ] Data dictionary for all columns
- [ ] Methods documentation with citations
- [ ] Tutorial notebooks with examples
- [ ] Changelog documenting all versions

#### Reproducibility
- [ ] Processing scripts available in GitHub
- [ ] All scripts versioned (git commit hashes)
- [ ] Source file provenance tracked
- [ ] Database construction reproducible from raw files
- [ ] Computational environment documented (requirements.txt)

#### Public Access
- [ ] Data hosted on stable repository (Zenodo/FigShare)
- [ ] DOI assigned for citability
- [ ] License specified (CC-BY 4.0 or CC0)
- [ ] GitHub repository public
- [ ] Contact information provided

#### Validation
- [ ] External user tested database
- [ ] All QC reports reviewed
- [ ] Known limitations documented
- [ ] Biological validation performed (literature comparison)

---

### Automated QC Pipeline

```python
def run_automated_qc(df):
    """
    Run comprehensive automated QC checks.
    Returns QC report with pass/fail flags.
    """
    qc_results = {}

    # 1. Completeness checks
    qc_results['completeness_abundances'] = (df['Abundance_Old'].notna() & df['Abundance_Young'].notna()).sum() / len(df) >= 0.95
    qc_results['completeness_sample_sizes'] = (df['N_Profiles_Old'].notna() & df['N_Profiles_Young'].notna()).sum() / len(df) == 1.0
    qc_results['completeness_species'] = df['Species'].notna().sum() / len(df) == 1.0

    # 2. Quality checks
    qc_results['quality_protein_ids'] = (df['Match_Confidence'] == 100).sum() / len(df) >= 0.95
    qc_results['quality_no_duplicates'] = df.duplicated(subset=['Dataset_Name', 'Protein_ID', 'Tissue_Compartment']).sum() == 0
    qc_results['quality_no_both_zero'] = ((df['Abundance_Old'] == 0) & (df['Abundance_Young'] == 0)).sum() == 0

    # 3. Statistical checks
    has_complete_data = df['Abundance_Old'].notna() & df['Abundance_Young'].notna() & df['N_Profiles_Old'].notna()
    qc_results['stats_pvalues_present'] = (df.loc[has_complete_data, 'p_value'].notna()).sum() / has_complete_data.sum() >= 0.95
    qc_results['stats_fdr_applied'] = df['q_value'].notna().sum() / len(df) >= 0.80

    # 4. Distribution checks
    if df['p_value'].notna().sum() > 0:
        # Check if p-value distribution is reasonable (should be uniform under null)
        # Not too many extreme values (would indicate calibration issues)
        extreme_pvals = (df['p_value'] < 0.001).sum() / df['p_value'].notna().sum()
        qc_results['stats_pvalue_distribution_reasonable'] = 0.01 <= extreme_pvals <= 0.20

    # 5. Cross-study consistency
    multi_study = df.groupby('Canonical_Gene_Symbol').filter(lambda x: len(x['Dataset_Name'].unique()) >= 2)
    if len(multi_study) > 0:
        consistency = multi_study.groupby('Canonical_Gene_Symbol').apply(
            lambda x: (np.sign(x['Zscore_Delta']).nunique() == 1) if x['Zscore_Delta'].notna().sum() >= 2 else True
        )
        qc_results['consistency_cross_study'] = consistency.sum() / len(consistency) >= 0.70

    # 6. Imputation checks
    if 'Imputation_Applied' in df.columns:
        imputed_fraction = df['Imputation_Applied'].sum() / len(df)
        qc_results['imputation_reasonable_rate'] = imputed_fraction <= 0.15  # <15% imputed

    # Generate report
    print("=" * 80)
    print("AUTOMATED QC REPORT")
    print("=" * 80)

    all_pass = True
    for check, result in qc_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {check}")
        if not result:
            all_pass = False

    print("=" * 80)
    if all_pass:
        print("✓ ALL QC CHECKS PASSED - Database ready for release")
    else:
        print("✗ SOME QC CHECKS FAILED - Review and fix issues before release")
    print("=" * 80)

    return qc_results, all_pass

# Run QC
qc_results, passed = run_automated_qc(df)

# Save QC report
with open('qc_report.txt', 'w') as f:
    f.write(f"QC Report Generated: {datetime.now()}\n")
    f.write(f"Database Version: v3.0\n")
    f.write(f"Total Proteins: {len(df)}\n\n")
    for check, result in qc_results.items():
        f.write(f"{'PASS' if result else 'FAIL'}: {check}\n")
```

---

## References & Methods Documentation

### Data Processing Methods (for publication)

**Methods text template for manuscript:**

```markdown
## Data Processing and Harmonization

### Source Data Collection
Proteomic datasets were obtained from 10 published studies (2017-2023) investigating age-related changes in extracellular matrix composition across multiple tissues and species. Raw data were downloaded from PRIDE (European Proteomics Infrastructure for Data) repositories or extracted from journal supplementary materials. Study metadata, including publication details, sample characteristics, and experimental protocols, were manually extracted from primary publications and PRIDE submissions.

### Protein Identification and Validation
Protein identifiers from source files were mapped to authoritative UniProt accession numbers and HGNC/MGI gene symbols using a validated bioinformatics pipeline. Low-confidence protein mappings (Match_Confidence < 100) were re-validated using the UniProt REST API (v2024). Synonym mappings were cross-referenced against HGNC (human) and MGI (mouse) nomenclature databases. Mouse proteins were mapped to human orthologs using Ensembl Compara (release 110) to enable cross-species comparisons. Final protein IDs achieved >95% match confidence.

### Matrisome Classification
All proteins were classified into matrisome categories (Core Matrisome: Collagens, ECM Glycoproteins, Proteoglycans; Matrisome-associated: ECM Regulators, Secreted Factors, ECM-affiliated Proteins) using the Matrisome AnalyzeR tool (http://matrisome.org) with UniProt accessions as input.

### Data Standardization
Protein abundance values were log2-transformed (log2(x + 1)) to stabilize variance and approximate normality. Within-study z-score normalization was applied to account for technical variation across experiments. Z-scores were computed as z = (x - μ) / σ, where x is the log2-transformed abundance, μ is the within-study mean, and σ is the within-study standard deviation. The aging-associated change was quantified as Zscore_Delta = Zscore_Old - Zscore_Young.

### Missing Data Imputation
Missing abundance values were classified by mechanism: (1) biologically age-specific (high abundance in one group only, preserved as missing), (2) left-censored (below detection limit, imputed using limit of detection / 2), or (3) random technical (imputed using k-nearest neighbors, k=5). Both raw (unimputed) and imputed databases are provided. Imputation flags document which values were imputed and by what method.

### Statistical Analysis
Differential expression between age groups was assessed using Welch's two-sample t-test to accommodate unequal variances. P-values were corrected for multiple testing using the Benjamini-Hochberg false discovery rate (FDR) procedure, applied independently within each study. Proteins with FDR-adjusted q-values < 0.05 were considered significantly changed with age. Effect sizes were quantified using Cohen's d (standardized mean difference) and log2 fold-change (log2(Old/Young)).

### Quality Control
Automated quality control checks included: validation of protein ID confidence (>95% confidence = 100), assessment of missing data patterns, detection of duplicate entries, validation of statistical test calibration (p-value distribution), and evaluation of cross-study consistency for proteins appearing in multiple datasets. Only proteins passing all QC checks were included in the final database.

### Data Availability
All raw data files, processing scripts, and quality control reports are available at [GitHub repository URL]. The curated database is deposited at Zenodo with DOI [DOI]. Individual study raw data are available at PRIDE with accession numbers listed in Supplementary Table 1.
```

### Software and Tools

**Document all software versions used:**

```markdown
## Software and Computational Environment

### Data Processing
- Python 3.10.12
- pandas 2.0.3
- numpy 1.24.3
- scipy 1.11.1
- scikit-learn 1.3.0
- statsmodels 0.14.0

### Protein Identification
- UniProt REST API (accessed 2024-10)
- Ensembl Compara REST API (release 110)
- Matrisome AnalyzeR (http://matrisome.org)

### Statistical Analysis
- scipy.stats for t-tests
- statsmodels.stats.multitest for FDR correction
- Custom scripts for z-score normalization (available in repository)

### Data Visualization
- matplotlib 3.7.2
- seaborn 0.12.2
- plotly 5.15.0

### Reproducibility
All analysis code is version-controlled using git and available at [GitHub URL]. Computational environment specifications are provided in requirements.txt. Database construction is fully reproducible from raw source files using provided scripts.
```

### Citations

**Key references to cite in methods:**

1. UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase in 2023. Nucleic Acids Res. 51:D523-D531.

2. Naba A, et al. (2016). The matrisome: in silico definition and in vivo characterization by proteomics of normal and tumor extracellular matrices. Mol Cell Proteomics. 15(2):357-73.

3. Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc Series B. 57(1):289-300.

4. Perez-Riverol Y, et al. (2022). The PRIDE database resources in 2022: a hub for mass spectrometry-based proteomics evidences. Nucleic Acids Res. 50:D543-D552.

5. Lazar C, et al. (2016). Accounting for the Multiple Natures of Missing Values in Label-Free Quantitative Proteomics Data Sets to Compare Imputation Strategies. J Proteome Res. 15(4):1116-25.

### Data Release Checklist

**Before public release, ensure:**

- [ ] Database DOI obtained from Zenodo
- [ ] GitHub repository created and populated
- [ ] README.md with clear instructions
- [ ] LICENSE file (recommend CC-BY 4.0)
- [ ] requirements.txt for computational environment
- [ ] Tutorial Jupyter notebooks
- [ ] Methods documentation complete
- [ ] All QC reports included
- [ ] Contact information provided
- [ ] Citation instructions provided

---

## Summary of Critical Actions

### Immediate Priorities (Week 1)

1. **Extract Randles_2021 sample sizes** from paper/PRIDE (458 proteins affected)
2. **Extract standard deviations** for all studies from supplementary tables
3. **Set up protein ID validation pipeline** using UniProt API

### High Priority (Weeks 2-3)

4. **Compute statistical tests** (t-test, FDR correction, effect sizes)
5. **Implement left-censored imputation** for low-abundance missing values
6. **Add age and methods metadata** from publications

### Medium Priority (Weeks 4-6)

7. **Validate all low-confidence protein IDs** (manual review if needed)
8. **Process 5 additional studies** to increase coverage to 10/13
9. **Implement automated QC pipeline**

### Pre-Release (Weeks 7-10)

10. **Complete documentation** (README, methods, tutorials)
11. **Run full QC suite** and fix any failures
12. **Prepare data release package** (Zenodo + GitHub)
13. **Draft manuscript** for database journal

---

## Contact and Support

**For questions about this assessment:**
- Database maintainer: [Your Name] <email@institution.edu>
- ECM biology expert: [Expert Name] <expert@institution.edu>
- Bioinformatics support: [Support Name] <support@institution.edu>

**For issues or suggestions:**
- GitHub Issues: [repository URL]/issues
- Email: ecm-atlas@institution.edu

---

**Document History:**
- v1.0 (2025-10-14): Initial assessment by Claude Code
- [Future versions will be tracked here]

---

## Appendix: Column Definitions

### Current Columns (v1.0)

| Column Name | Data Type | Description | Missingness | Notes |
|-------------|-----------|-------------|-------------|-------|
| Dataset_Name | string | Study identifier (Author_Year) | 0% | Primary key component |
| Organ | string | Organ system | 0% | Controlled vocabulary |
| Compartment | string | Tissue compartment/subregion | Variable | Study-specific |
| Abundance_Old | float | Raw abundance in old samples | 3.7% | Original units vary by study |
| Abundance_Old_transformed | float | Log-transformed abundance | 46.3% | log2(x + 1) transformation |
| Abundance_Young | float | Raw abundance in young samples | 9.0% | Original units vary by study |
| Abundance_Young_transformed | float | Log-transformed abundance | 46.0% | log2(x + 1) transformation |
| Canonical_Gene_Symbol | string | Authoritative gene symbol | 0% | HGNC/MGI nomenclature |
| Gene_Symbol | string | Original gene symbol from source | 0% | May differ from canonical |
| Match_Confidence | integer | Protein ID mapping confidence (0-100) | 0% | 100 = high confidence |
| Match_Level | string | Type of ID match | 0% | exact_gene, synonym, etc. |
| Matrisome_Category | string | ECM protein functional category | 0% | From Matrisome AnalyzeR |
| Matrisome_Division | string | Core vs associated matrisome | 0% | Binary classification |
| Method | string | Proteomic quantification method | 0% | High-level description |
| N_Profiles_Old | integer | Sample size for old group | 21.0% | Missing for Randles_2021 |
| N_Profiles_Young | integer | Sample size for young group | 21.0% | Missing for Randles_2021 |
| Protein_ID | string | UniProt accession | 0% | Primary protein identifier |
| Protein_Name | string | Full protein name | 1.1% | From UniProt |
| Species | string | Organism | 0% | Scientific name |
| Study_ID | string | Study identifier (duplicate of Dataset_Name) | 0% | Consider removing |
| Tissue | string | Tissue type | 0% | Organ_Compartment format |
| Tissue_Compartment | string | Tissue compartment (duplicate) | 0% | Consider merging with Compartment |
| Zscore_Delta | float | Aging-associated z-score change | 12.7% | Zscore_Old - Zscore_Young |
| Zscore_Old | float | Within-study z-score for old | 3.7% | (x - mean) / SD |
| Zscore_Young | float | Within-study z-score for young | 9.0% | (x - mean) / SD |

### Recommended Additional Columns (v2.0)

| Column Name | Data Type | Description | Priority |
|-------------|-----------|-------------|----------|
| SD_Old | float | Standard deviation of old group | CRITICAL |
| SD_Young | float | Standard deviation of young group | CRITICAL |
| SEM_Old | float | Standard error of mean for old | HIGH |
| SEM_Young | float | Standard error of mean for young | HIGH |
| t_statistic | float | Welch's t-test statistic | CRITICAL |
| df | float | Degrees of freedom for t-test | CRITICAL |
| p_value | float | Uncorrected p-value | CRITICAL |
| q_value | float | FDR-corrected q-value | CRITICAL |
| significant_FDR05 | boolean | Significant at FDR < 0.05 | CRITICAL |
| log2_fold_change | float | log2(Old/Young) | HIGH |
| fold_change | float | Old/Young ratio | HIGH |
| cohens_d | float | Standardized effect size | HIGH |
| effect_size_interpretation | string | small/medium/large | MODERATE |
| ci_lower | float | Lower 95% CI for mean difference | MODERATE |
| ci_upper | float | Upper 95% CI for mean difference | MODERATE |
| Abundance_Old_imputed | boolean | Was value imputed? | HIGH |
| Abundance_Young_imputed | boolean | Was value imputed? | HIGH |
| Missingness_Mechanism | string | Reason for missingness | MODERATE |
| Imputation_Method | string | Method used for imputation | MODERATE |
| Human_Ortholog_Gene | string | Human ortholog (for mouse proteins) | MODERATE |
| Orthology_Confidence | float | Confidence of orthology mapping | MODERATE |
| Source_File | string | Original data file name | LOW |
| Source_Row | integer | Row in original file | LOW |
| Date_Processed | date | Processing date | LOW |
| QC_Pass | boolean | Passed QC checks | MODERATE |
| QC_Warnings | string | Any QC warnings | LOW |

---

**END OF DOCUMENT**

Total pages: 51
Word count: ~15,000
Reading time: ~60 minutes

This document provides a complete roadmap for achieving a publication-ready, scientifically sound ECM-Atlas database. Follow the implementation roadmap and use the QC checklist to ensure all critical issues are addressed before public release.
