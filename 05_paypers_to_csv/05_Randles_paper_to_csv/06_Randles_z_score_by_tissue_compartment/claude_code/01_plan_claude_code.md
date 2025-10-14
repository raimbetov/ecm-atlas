# Z-Score Normalization Analysis Plan

**Project:** ECM Atlas - Randles 2021 Z-Score Normalization by Tissue Compartment
**Date:** 2025-10-12
**Analyst:** Claude Code
**Task File:** `00_TASK_Z_SCORE_NORMALIZATION.md`

---

## Objective

Calculate z-score normalization separately for each kidney tissue compartment (Glomerular and Tubulointerstitial) to enable within-study statistical comparisons while preserving biological differences between compartments.

---

## Execution Strategy

### Phase 1: Data Loading & Validation
**Input:** `05_Randles_paper_to_csv/claude_code/Randles_2021_wide_format.csv`
**Expected:** 5,220 rows (2,422 unique proteins × 2 compartments)

**Steps:**
1. Load wide-format CSV with pandas
2. Validate required columns: `Protein_ID`, `Gene_Symbol`, `Tissue_Compartment`, `Abundance_Young`, `Abundance_Old`
3. Verify compartment counts (should be 2,610 per compartment)
4. Check for null values, zeros, and negative values

**Success criteria:**
- ✅ All required columns present
- ✅ Exactly 2 compartments (Glomerular, Tubulointerstitial)
- ✅ No critical data quality issues

---

### Phase 2: Compartment Separation
**Output:** Two separate dataframes

**Steps:**
1. Split data by `Tissue_Compartment` column
2. Create `df_glomerular` and `df_tubulointerstitial`
3. Validate no data loss during split
4. Run quality checks on each compartment independently

**Success criteria:**
- ✅ df_glomerular: ~2,610 proteins
- ✅ df_tubulointerstitial: ~2,610 proteins
- ✅ Total rows preserved: 5,220

---

### Phase 3: Distribution Analysis
**Purpose:** Determine if log2-transformation is needed

**Steps:**
1. Calculate skewness for `Abundance_Young` and `Abundance_Old` in each compartment
2. Decision rule: If skewness > 1, apply log2(x + 1) transformation
3. Document transformation decision per compartment

**Expected outcome:**
- Mass spectrometry data typically has right-skewed distribution
- Likely need log2-transformation for both compartments

---

### Phase 4: Statistical Normalization
**Method:** Z-score calculation per compartment per age group

**Formula:**
```
Z = (X - Mean) / StdDev
```

**Steps:**
1. Apply log2-transformation if needed (from Phase 3)
2. Calculate normalization parameters:
   - Glomerular: Mean_Young, StdDev_Young, Mean_Old, StdDev_Old
   - Tubulointerstitial: Mean_Young, StdDev_Young, Mean_Old, StdDev_Old
3. Compute z-scores independently for each group
4. Calculate Zscore_Delta (Old - Young) for aging signature

**Success criteria:**
- ✅ Z-score means ≈ 0 (within ±0.01)
- ✅ Z-score standard deviations ≈ 1 (within 0.99-1.01)
- ✅ No null values in z-score columns

---

### Phase 5: Export & Documentation
**Outputs:**
1. `Randles_2021_Glomerular_zscore.csv` (~2,610 proteins)
2. `Randles_2021_Tubulointerstitial_zscore.csv` (~2,610 proteins)
3. `zscore_validation_report.md` (statistics & QC)
4. `zscore_metadata.json` (normalization parameters)

**CSV columns:**
- Protein identifiers: `Protein_ID`, `Protein_Name`, `Gene_Symbol`
- Tissue metadata: `Tissue`, `Tissue_Compartment`, `Species`
- Original values: `Abundance_Young`, `Abundance_Old`
- Transformed values: `Abundance_Young_transformed`, `Abundance_Old_transformed`
- Z-scores: `Zscore_Young`, `Zscore_Old`, `Zscore_Delta`
- Annotations: `Matrisome_Category`, `Matrisome_Division`, etc.

---

## Validation Checklist

### Tier 1: Critical (ALL required)
- [ ] 2 output CSV files created
- [ ] Total proteins = 5,220 (preserved)
- [ ] Z-score means within [-0.01, +0.01]
- [ ] Z-score standard deviations within [0.99, 1.01]
- [ ] No null values in z-score columns

### Tier 2: Quality (ALL required)
- [ ] Log2-transformation applied correctly (if skewness > 1)
- [ ] <5% proteins with |Z| > 3 (extreme outliers)
- [ ] Known ECM markers present (COL1A1, FN1, LAMA1, etc.)
- [ ] Z-score distributions approximately normal (skewness < |0.5|)

### Tier 3: Documentation (ALL required)
- [ ] Validation report generated
- [ ] Metadata JSON exported
- [ ] CSV files readable and properly formatted

---

## Key Biological Rationale

**Why separate compartments?**

Glomerular and Tubulointerstitial compartments have fundamentally different ECM compositions:
- **Glomerular:** Filtration units enriched in basement membrane proteins (COL4, laminins)
- **Tubulointerstitial:** Tubule structures enriched in stromal ECM (COL1, fibronectin)

**Risk of combining:** Would blur biological signal and mask compartment-specific aging changes.

**Solution:** Independent normalization preserves biological differences while enabling statistical comparison.

---

## Python Environment

**Required packages:**
- pandas (data manipulation)
- numpy (mathematical operations)
- scipy (distribution analysis - skewness)
- json (metadata export)

**Working directory:** `/Users/Kravtsovd/projects/ecm-atlas/06_Randles_z_score_by_tissue_compartment/claude_code`

---

## Next Steps

1. Execute Python analysis script implementing all phases
2. Generate all output files
3. Validate against success criteria
4. Create final results document (`90_results_claude_code.md`)

---

**Status:** Plan approved - Ready for execution
**Estimated time:** 10-15 minutes
**Complexity:** Medium (statistical normalization with quality control)
