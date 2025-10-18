# VALIDATION REPORT: Santinha_2024_Mouse_NT Data Scale Analysis

**Status:** COMPLETED - Data Scale Confirmed as LOG2-TRANSFORMED  
**Date:** 2025-10-17  
**Task:** Verify data scale (LINEAR or LOG2) for Santinha Mouse Non-Treated cardiac tissue  
**Key Finding:** Database abundances are **LOG2-TRANSFORMED**

---

## PHASE 1: Paper & Methods Verification

### Study Information
- **Title:** Remodeling of the Cardiac Extracellular Matrix Proteome During Chronological and Pathological Aging
- **Authors:** Santinha et al.
- **Journal:** Molecular & Cellular Proteomics (MCP), 2024
- **DOI:** 10.1016/j.mcpro.2023.100706
- **PMID:** 38141925
- **Method:** TMT-10plex LC-MS/MS (Tandem Mass Tags isobaric labeling)

### Dataset Details - Mouse Non-Treated (NT)
- **Species:** Mus musculus (C57BL/6 mice)
- **Tissue:** Cardiac left ventricle (LV)
- **Age Groups:** 
  - Young: 3 months (n=5)
  - Old: 20 months (n=5)
- **Source Data Format:** Differential expression (logFC, AveExpr) from supplementary table mmc2.xlsx
- **Sheet Name:** "MICE_NT_old_vs_young"

### Methods Section - TMT Quantification (From Paper)
From Methods section (page 2-3):

> "The reconstituted peptides were used for TMT labeling... For each animal, we obtained a quantitative proteome profile using Tandem Mass Tags (TMT)-based quantitative MS after peptide fractionation at high pH to maximize proteome coverage."

### Processing Consistency: Same as Human & Mouse_DT
- **All three Santinha datasets use identical TMT-10plex protocol**
- **Same supplementary data format:** logFC + AveExpr (differential expression)
- **Same back-calculation method applied:**
  ```
  Abundance_Young = AveExpr - (logFC / 2)
  Abundance_Old = AveExpr + (logFC / 2)
  ```
- **Same species-specific matrisome annotations:** Mouse reference (1,110 genes)

---

## PHASE 2: Processing Script Verification

### TMT Adapter Implementation
**Script:** `tmt_adapter_santinha2024.py`  
**Lines 48-65:** Back-calculation function

```python
def back_calculate_abundances(df):
    """
    Back-calculate Young and Old abundances from logFC and AveExpr.
    
    Given:
        logFC = log2(Old) - log2(Young)
        AveExpr = (log2(Young) + log2(Old)) / 2
    
    Solve:
        log2(Old) = AveExpr + (logFC / 2)
        log2(Young) = AveExpr - (logFC / 2)
    
    Returns log2-transformed abundances (suitable for z-score normalization).
    """
    df = df.copy()
    df['Abundance_Young'] = df['AveExpr'] - (df['logFC'] / 2)
    df['Abundance_Old'] = df['AveExpr'] + (df['logFC'] / 2)
    return df
```

### Processing Pipeline Confirmation
1. **Phase 1 (TMT Adapter):** Back-calculated Young/Old from logFC + AveExpr ✓
2. **Phase 2 (Merge):** Added to unified database ✓
3. **Phase 3 (Z-scores):** Calculated without log2 transformation ✓

---

## PHASE 3: Source Data Inspection

### File Location
`/Users/Kravtsovd/projects/ecm-atlas/data_raw/Santinha et al. - 2024/mmc2.xlsx`

### Data Format Verification
- **Input:** Supplementary table with columns: `gene.name`, `ID`, `logFC`, `AveExpr`, `description`
- **logFC column:** Contains log2 fold-changes (Old vs Young)
- **AveExpr column:** Contains average expression in log2 scale

### Sample Abundances (First 15 Mouse_NT ECM Proteins)

| Protein_ID | Gene_Symbol | Abundance_Young (log2) | Abundance_Old (log2) | Mean      | Notes                    |
|------------|-------------|----------------------|----------------------|-----------|--------------------------|
| Q9R045     | Angptl2     | 16.749                | 17.801               | 17.275    | Angiopoietin-like protein|
| Q3UQ28     | Pxdn        | 16.763                | 16.158               | 16.461    | Peroxidasin homolog      |
| Q02788     | Col6a2      | 15.988                | 16.781               | 16.385    | Collagen VI α2           |
| Q8C6K9     | Col6a6      | 15.929                | 16.870               | 16.400    | Collagen VI α6           |
| P16110     | Lgals3      | 16.247                | 17.055               | 16.651    | Galectin-3               |
| P21956     | Mfge8       | 15.951                | 16.588               | 16.270    | **Lactadherin** (aging marker)|
| Q04857     | Col6a1      | 15.465                | 16.173               | 15.819    | Collagen VI α1           |
| Q9R013     | Ctsf        | 14.218                | 15.196               | 14.707    | Cathepsin F              |
| P14069     | S100a6      | 17.563                | 18.022               | 17.793    | S100 calcium-binding     |
| A6X935     | Itih4       | 16.401                | 17.046               | 16.724    | Inter-alpha trypsin      |
| Q8CFG0     | Sulf2       | 16.429                | 17.129               | 16.779    | Extracellular sulfatase  |
| P82198     | Tgfbi       | 17.019                | 17.470               | 17.245    | TGF-beta-induced protein |
| Q61704     | Itih3       | 16.163                | 16.687               | 16.425    | Inter-alpha trypsin      |
| Q8K4G1     | Ltbp4       | 14.567                | 14.064               | 14.316    | LTBP-4                   |
| P18242     | Ctsd        | 16.878                | 17.300               | 17.089    | Cathepsin D              |

### Distribution Statistics - Mouse_NT

**Young Age Group (3 months):**
- Mean: **15.8824** log2 units
- Median: **15.9778** log2 units
- Std Dev: 1.2189
- Range: 12.48 - 19.43 log2 units
- Q1: 15.10, Q3: 16.68

**Old Age Group (20 months):**
- Mean: **16.0277** log2 units
- Median: **16.1511** log2 units
- Std Dev: 1.2476
- Range: 12.74 - 19.87 log2 units
- Q1: 15.22, Q3: 16.87

**Age Difference:** Δ = 0.1453 log2 units (~1.1-fold linear increase)

---

## PHASE 4: Cross-Study Comparison

### All Three Santinha Datasets - Abundance Scales

| Dataset              | Tissue                      | Young Mean | Old Mean | Median Range | Linear Range |
|---------------------|----------------------------|------------|----------|--------------|--------------|
| Santinha_2024_Mouse_NT  | Heart Native Tissue         | 15.88      | 16.03    | 15.98-16.15  | 2^15.88 to 2^16.03 |
| Santinha_2024_Mouse_DT  | Heart Decellularized Tissue | 16.65      | 16.82    | 16.77-16.92  | 2^16.65 to 2^16.82 |
| Santinha_2024_Human     | Heart Native Tissue         | 14.80      | 15.12    | 14.81-15.17  | 2^14.80 to 2^15.12 |

### Key Observations

1. **Consistent Scale Across Datasets:** All three use log2 transform
   - Mouse_NT: Young median 15.98, Old median 16.15 (log2 units)
   - Mouse_DT: Young median 16.77, Old median 16.92 (log2 units)
   - Human: Young median 14.81, Old median 15.17 (log2 units)

2. **Mouse_DT vs Mouse_NT Offset:** +0.76 log2 units
   - Biologically sensible: Decellularized tissue enriched for ECM proteins
   - Higher abundance in DT reflects ECM enrichment (as paper documents)

3. **Human vs Mouse_NT Offset:** -1.08 log2 units
   - Biologically sensible: Different ECM composition between species
   - Lower scale in human consistent with species-specific protein expression

4. **All use identical back-calculation:** Confirms consistency across compartments

---

## PHASE 5: Z-Score Metadata Validation

### Zscore_Metadata Summary

**Mouse_NT (Heart_Native_Tissue):**
```json
{
  "mean_young": 15.8824,
  "std_young": 1.2189,
  "mean_old": 16.0277,
  "std_old": 1.2476,
  "skew_young": -0.073,
  "skew_old": -0.117,
  "log2_transformed": false,  // ← Input was already log2
  "z_mean_young": 0.0,
  "z_std_young": 1.0,
  "z_mean_old": 0.0,
  "z_std_old": 1.0,
  "validation_passed": true
}
```

**Interpretation:**
- `log2_transformed: false` means script did NOT apply log2 transformation
- Reason: Skewness < 1 for both Young (-0.073) and Old (-0.117)
- Input data already in log2 space (confirmed by back-calculation method)
- Z-scores properly standardized: μ ≈ 0, σ ≈ 1

---

## PHASE 6: FINAL DETERMINATION

### Database Scale: LOG2-TRANSFORMED

**Evidence Supporting LOG2:**

1. **Source Data Format:**
   - Supplementary table uses logFC (log2 fold-change)
   - AveExpr provided as average in log2 space
   - Paper Methods: "TMT-based quantitative MS" → log2 reporter intensities

2. **Back-Calculation Formula:**
   - Math explicitly assumes log2: `log2(Young) = AveExpr - (logFC / 2)`
   - Result: Abundances in log2 units, not linear

3. **Abundance Range Verification:**
   - Mouse_NT: 15.98-16.15 (median log2 range)
   - If linear: would be ~2^15.98 to 2^16.15 ≈ 65,000-70,000 intensity units
   - TMT reporter intensities typically in 1,000-100,000 range ✓
   - If incorrectly assumed linear, median would be ~16 (nonsensical intensity)

4. **Consistency Across Compartments:**
   - Mouse_NT: 15.88-16.03 mean (log2)
   - Mouse_DT: 16.65-16.82 mean (log2)
   - Human: 14.80-15.12 mean (log2)
   - All consistent with log2 scale

5. **Skewness Analysis:**
   - Both Young and Old: skewness -0.073 to -0.117
   - Slight negative skew typical of log-transformed proteomics data
   - Confirms log2 transformation already applied

---

## BATCH CORRECTION IMPLICATIONS

### Current State: CORRECT FOR BATCH CORRECTION

**Do NOT apply log2(x+1):**
- Data already in log2 space
- Applying second log would produce nonsensical values
- Z-scores already computed correctly

**For Batch Correction Framework:**
1. Data arrives as: Abundance (log2 units), z-score
2. Use z-scores directly for batch correction (already standardized)
3. If reverting to linear space, use: 2^Abundance
4. If batch-correcting in linear space, convert to linear FIRST

### Handling Within Batch Correction Pipeline
```python
# Data is already log2-transformed
# Use as-is for standardized comparisons
z_score_corrected = apply_batch_correction(z_scores)

# If needed to return to linear space:
abundance_linear = 2 ** abundance_log2
```

---

## SUMMARY TABLE

| Aspect | Finding | Evidence |
|--------|---------|----------|
| **Data Scale** | LOG2-TRANSFORMED | Back-calculation formula, supplementary data format |
| **Database Median** | 15.98-16.15 (log2) | Verified from processed CSV |
| **Processing Method** | TMT adapter (same as Mouse_DT, Human) | Confirmed in tmt_adapter_santinha2024.py |
| **Z-Score Validation** | PASSED (μ≈0, σ≈1) | Metadata confirms validation_passed: true |
| **Consistency** | All 3 datasets identical protocol | Same back-calculation, species-specific annotation |
| **Apply log2(x+1)** | **NO** | Already log2-transformed |
| **Batch Correction Ready** | YES | Use z-scores directly |

---

## RECOMMENDATIONS

### For Batch Correction Analysis

1. **Input Format:** Use z-scores (already standardized)
   - Source: Columns Zscore_Young, Zscore_Old in unified CSV

2. **Scale Compatibility:** Log2 scale shared with all other Santinha datasets
   - No cross-dataset scale mismatch issues

3. **Compartment Specificity:** Separate z-scores per tissue compartment
   - Mouse_NT: Heart_Native_Tissue
   - Mouse_DT: Heart_Decellularized_Tissue  
   - Human: Heart_Native_Tissue
   - Prevents spurious cross-compartment batch effects

4. **No Additional Transformation Needed**
   - Data already optimally scaled
   - Skip log2(x+1) for this study

### Comparative Context

This dataset is part of the broader Santinha 2024 study which includes:
- **191 ECM proteins** in Mouse_NT (4.0% of 4,827 detected proteins)
- **155 ECM proteins** in Mouse_DT (3.8% of 4,089 detected proteins)
- **207 ECM proteins** in Human (5.3% of 3,922 detected proteins)

All processed with identical methodology and available for cross-tissue, cross-species aging signature analysis.

---

## FILES VALIDATED

- `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/14_Santinha_2024_paper_to_csv/Santinha_2024_wide_format.csv` ✓
- `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/14_Santinha_2024_paper_to_csv/tmt_adapter_santinha2024.py` ✓
- `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/14_Santinha_2024_paper_to_csv/zscore_metadata_Santinha_2024_Mouse_NT.json` ✓
- `/Users/Kravtsovd/projects/ecm-atlas/pdf/Santinha et al. - 2024.pdf` ✓
- `/Users/Kravtsovd/projects/ecm-atlas/data_raw/Santinha et al. - 2024/mmc2.xlsx` (Source data) ✓

---

**Validation Completed:** 2025-10-17  
**Validated By:** Claude Code Analysis Agent  
**Status:** READY FOR BATCH CORRECTION ANALYSIS
