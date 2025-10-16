# Processing Log: Zero-to-NaN Conversion Fix
**Dataset:** Randles_2021 (Kidney aging proteomics)
**Date:** 2025-10-15
**Task:** Convert zero values to NaN to align with proteomics standards

---

## Executive Summary

Successfully implemented zero-to-NaN conversion in Randles_2021 dataset processing pipeline. This fix corrects a data quality issue where zeros (representing "not detected") were being treated as true abundance values, causing statistical bias in downstream analyses.

**Impact:** 483 zeros (1.54% of source measurements) converted to NaN, resulting in more accurate protein abundance calculations.

---

## 1. Problem Statement

### Issue Identified
- **Source data:** 483 zero values across 12 samples in Excel file
- **Processed data:** 25 zero values in averaged wide-format dataset
- **Biological concern:** Zeros represent detection failure, not true "zero abundance"
- **Statistical concern:** Zeros artificially deflate means and inflate standard deviations

### Example Case: Protein Q9H553 (ALG2)
**Before fix (Young Glomerular):**
- Replicate values: G15=105.9, G29=0.0, G37=33.0
- Average with zero: 46.30
- Interpretation: "Protein detected with very low abundance"

**After fix (Young Glomerular):**
- Replicate values: G15=105.9, G29=NaN, G37=33.0
- Average excluding NaN: 69.44
- Interpretation: "Protein detected in 2/3 replicates with mean 69.44"

**Change:** +50.0% increase (correct statistical treatment of missing data)

---

## 2. Implementation Details

### Code Changes

**File 1:** `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/randles_conversion.py`

**Change 1 - Added numpy import:**
```python
import numpy as np
```

**Change 2 - Added zero-to-NaN conversion after Excel read (line 41-60):**
```python
# ============================================================================
# ZERO-TO-NAN CONVERSION (Data Quality Fix)
# ============================================================================
# Convert zero values to NaN in abundance columns
# Zero in proteomics = not detected (missing), not true zero abundance
# This fix prevents zero-inflation bias in statistical analyses
intensity_cols = ['G15', 'T15', 'G29', 'T29', 'G37', 'T37', 'G61', 'T61', 'G67', 'T67', 'G69', 'T69']
zeros_before = sum((df_wide[col] == 0).sum() for col in intensity_cols)
print(f"\n[DATA QUALITY FIX] Converting zeros to NaN in abundance columns")
print(f"   - Zero values before conversion: {zeros_before} ({zeros_before/(df_wide.shape[0]*len(intensity_cols))*100:.2f}% of measurements)")

for col in intensity_cols:
    df_wide[col] = df_wide[col].replace(0, np.nan)

zeros_after = sum((df_wide[col] == 0).sum() for col in intensity_cols)
nan_count = sum(df_wide[col].isna().sum() for col in intensity_cols)
print(f"   - Zero values after conversion: {zeros_after}")
print(f"   - NaN values after conversion: {nan_count}")
print(f"   ✅ Zero-to-NaN conversion complete (aligns with proteomics standards)")
# ============================================================================
```

**File 2:** `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/pivot_to_wide_format.py`

**Change 3 - Explicit skipna=True in averaging (line 71):**
```python
'Abundance': lambda x: x.mean(skipna=True),  # Average across replicates, excluding NaN (zeros converted to NaN)
```

---

## 3. Validation Results

### Before Fix (Original Data)
```
Source Excel file:
  Zero values: 483 (1.54% of 31,320 measurements)
  NaN values: 0

Wide-format CSV:
  Abundance_Young zeros: 18
  Abundance_Old zeros: 7
  Total zeros: 25 (0.24% of 10,440 measurements)
```

### After Fix (Current Data)
```
Source Excel file (after conversion):
  Zero values: 0
  NaN values: 483 (1.54% of measurements)

Long-format CSV (Randles_2021_parsed.csv):
  Total rows: 30,837 (was 31,320)
  Removed rows with null Abundance: 483

Wide-format CSV (Randles_2021_wide_format.csv):
  Abundance_Young zeros: 0 (was 18)
  Abundance_Old zeros: 0 (was 7)
  Abundance_Young NaN: 15
  Abundance_Old NaN: 4
  Total rows: 5,217
```

### Example Protein Validation

**Q9H553 (ALG2) - Kidney_Glomerular:**
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Young Mean | 46.30 | 69.44 | +50.0% |
| Replicates | [105.9, 0.0, 33.0] | [105.9, NaN, 33.0] | 0→NaN |
| Valid N | 3 | 2 | -1 |

**Interpretation:** The increase is expected and correct - we now average only the detected values, not treating "not detected" as zero abundance.

---

## 4. Impact Assessment

### Data Volume Impact
- **Source:** 483 measurements (1.54%) changed from 0 to NaN
- **Long-format:** 483 rows removed (had null Abundance after conversion)
- **Wide-format:** 19 measurements changed from 0 to NaN (15 Young + 4 Old)

### Statistical Impact
**Positive effects:**
1. Mean abundance calculations exclude missing values (statistically correct)
2. Standard deviations reflect true variance, not detection gaps
3. Z-score normalization uses accurate distribution parameters
4. Downstream tests account for varying sample sizes per protein

**Proteins affected:**
- ~25 proteins in wide-format dataset (0.48% of 5,217 protein-compartment pairs)
- Most affected: proteins with 1/3 or 2/3 detection rate in replicate groups

### Alignment with Standards
- **PRIDE database:** Recommends NaN for undetected proteins
- **MaxQuant output:** Uses NaN by default for missing values
- **Perseus software:** Requires NaN for proper missing data handling
- **Community practice:** Zero imputation discouraged for LFQ data

---

## 5. Files Modified

### Scripts (Permanent Changes)
1. `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/randles_conversion.py`
   - Added numpy import
   - Added zero-to-NaN conversion block after Excel read
   - Updated paths to use absolute references

2. `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/pivot_to_wide_format.py`
   - Explicit `skipna=True` in mean aggregation (documentation)

### Data Files (Regenerated)
1. `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/Randles_2021_parsed.csv`
   - Rows: 30,837 (was 31,320)
   - Size: 10.06 MB
   - Status: Regenerated with zero-to-NaN fix

2. `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/Randles_2021_wide_format.csv`
   - Rows: 5,217
   - Size: 1.21 MB
   - Status: Regenerated with correct averaging (excludes NaN)

3. `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/Randles_2021_metadata.json`
   - Status: Updated with fix details

4. `/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/Randles_2021_wide_format_metadata.json`
   - Status: Updated with new statistics

---

## 6. Next Steps

### Immediate
- [x] Apply zero-to-NaN conversion to source data parsing
- [x] Regenerate Randles_2021_parsed.csv
- [x] Regenerate Randles_2021_wide_format.csv
- [x] Validate with example protein Q9H553
- [x] Document changes in processing log

### Follow-up
- [ ] Update merged ECM database (`08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`)
- [ ] Re-run merge_to_unified.py with new Randles data
- [ ] Verify dashboard displays updated values
- [ ] Update `04_compilation_of_papers/00_README_compilation.md` with processing notes
- [ ] Check if other datasets need similar zero-to-NaN fixes

---

## 7. Audit Trail

### Analysis
- **Date:** 2025-10-15
- **Audit script:** `analyze_zeros.py`
- **Report:** `ZERO_VALUE_AUDIT_REPORT.md`
- **Finding:** 483 zeros identified, conversion recommended

### Implementation
- **Date:** 2025-10-15
- **Modified by:** Claude Code (Autonomous Agent)
- **Scripts updated:** randles_conversion.py, pivot_to_wide_format.py
- **Data regenerated:** All Randles_2021 CSV outputs
- **Validation:** Q9H553 example confirms expected behavior

### Results
- **Source zeros:** 483 → 0 (converted to NaN)
- **Long-format rows:** 31,320 → 30,837 (removed 483 null Abundance)
- **Wide-format zeros:** 25 → 0 (Young: 18→0, Old: 7→0)
- **Wide-format NaN:** 0 → 19 (Young: 15, Old: 4)
- **Example protein Q9H553 Young mean:** 46.30 → 69.44 (+50.0%)

---

## 8. Conclusion

Zero-to-NaN conversion successfully implemented and validated. The fix improves data quality by:

1. **Biological accuracy:** Distinguishes "not detected" from "detected at low level"
2. **Statistical correctness:** Excludes missing values from mean/SD calculations
3. **Standard compliance:** Aligns with proteomics community best practices
4. **Transparency:** Honest representation of measurement limitations

**Status:** COMPLETED ✅

**Recommendation:** Apply this fix to all future dataset ingestions and audit existing datasets for similar issues.

---

**Generated:** 2025-10-15
**Analyst:** Claude Code
**Dataset:** Randles_2021 (Kidney aging proteomics, PMID: 34049963)
**Source:** ASN.2020101442-File027.xlsx
