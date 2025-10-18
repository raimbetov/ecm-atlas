# CRITICAL BUG: Incorrect Data Scale Assumption in parse_lidermis.py

**Status:** 🚨 CRITICAL - REQUIRES IMMEDIATE FIX
**Discovered:** 2025-10-17
**Impact:** HIGH - Affects z-score calculations and cross-study normalization
**Affected File:** `parse_lidermis.py`

---

## Problem Summary

**parse_lidermis.py incorrectly assumes Li et al. 2021 dermis data is log2-transformed when it is actually LINEAR scale FOT-normalized intensities.**

This causes incorrect z-score calculations in the ECM-Atlas pipeline because:
1. Data is treated as log2 when it's actually linear
2. universal_zscore_function.py applies another log2 transformation
3. Result: **double log2 transformation** → incorrect z-scores

---

## Evidence from Paper

**Source:** Li et al. 2021 - "Time-Resolved Extracellular Matrix Atlas of the Developing Human Skin Dermis"

**Quote from Methods (Page 3):**
> "The quantification values of identified proteins were normalized by taking the fraction of total, followed by multiplication by 10^6."

**What is FOT (Fraction of Total)?**
- **Linear normalization method**
- Formula: (Protein intensity ÷ Total protein intensity) × 10^6
- Produces **linear scale** values representing relative abundance proportions
- **NOT log2-transformed**

**Confusion Source:**
- Figure 2B caption states: "according to log2 normalized protein intensity"
- This refers to **heatmap visualization transformation ONLY**, not raw data in Table S2
- Table S2 contains **linear FOT-normalized values** (the raw data we're parsing)

---

## Current Bug Location

**File:** `05_papers_to_csv/11_LiDermis_2021_paper_to_csv/parse_lidermis.py`

**Line 290 (and Line 312):**
```python
'Abundance_Unit': 'log2_normalized_intensity'  # ❌ INCORRECT ASSUMPTION
```

**Also in header comment (Lines 8-10):**
```python
\"\"\"
Key features:
- Already log2-normalized data  # ❌ INCORRECT
\"\"\"
```

---

## Required Fix

### Fix 1: Update Abundance_Unit (Line 290 and 312)

**Before:**
```python
'Abundance_Unit': 'log2_normalized_intensity',
```

**After:**
```python
'Abundance_Unit': 'FOT_normalized_intensity',  # Fraction of Total × 10^6 (LINEAR scale)
```

### Fix 2: Update Header Comment (Lines 8-10)

**Before:**
```python
\"\"\"
LiDermis 2021 Data Parser
Based on: 04_compilation_of_papers/06_LiDermis_2021_comprehensive_analysis.md

Key features:
- 4 age groups → 2 bins (exclude Adult 40yr middle-aged)
- 10 samples total: Toddler(2), Teenager(3), Adult(2), Elderly(3)
- Protein_Name via UniProt lookup (not in source file)
- Already log2-normalized data  # ❌ REMOVE THIS LINE
\"\"\"
```

**After:**
```python
\"\"\"
LiDermis 2021 Data Parser
Based on: 04_compilation_of_papers/06_LiDermis_2021_comprehensive_analysis.md

Key features:
- 4 age groups → 2 bins (exclude Adult 40yr middle-aged)
- 10 samples total: Toddler(2), Teenager(3), Adult(2), Elderly(3)
- Protein_Name via UniProt lookup (not in source file)
- FOT-normalized data (Fraction of Total × 10^6, LINEAR scale)
- Log2 transformation applied by universal_zscore_function.py (not in source data)
\"\"\"
```

### Fix 3: Update Parsing_Notes (Lines 294 and 316)

**Before:**
```python
'Parsing_Notes': f"Age={age}yr... Abundance: log2-normalized intensity (FOT fraction of total)..."
```

**After:**
```python
'Parsing_Notes': f"Age={age}yr... Abundance: FOT-normalized intensity (Fraction of Total × 10^6, LINEAR scale)..."
```

---

## Impact Assessment

### What Happens If Not Fixed

1. **universal_zscore_function.py processes LiDermis data:**
   - Assumes data is LINEAR (correct assumption)
   - Applies log2 transformation: `np.log2(Abundance + 1)`
   - **BUT** if Abundance_Unit says "log2_normalized_intensity", downstream analysis may skip log2 transformation
   - Result: Incorrect z-scores

2. **Cross-study normalization:**
   - LiDermis z-scores don't align with other studies
   - Percentile normalization produces incorrect rankings
   - Dashboard heatmaps show wrong values

3. **Data integrity:**
   - Merged database contains mislabeled data
   - Future analyses inherit the error
   - Scientific conclusions may be invalid

### Current Status in Database

**Check Required:**
- Inspect `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Filter rows where `Study_ID = 'LiDermis_2021'`
- Check if `Abundance_Unit` column contains 'log2_normalized_intensity'
- If YES → database needs regeneration after fix

---

## Action Plan

### Immediate (Priority 1)

1. ✅ **Fix parse_lidermis.py** (3 locations: header, line 290, line 312)
2. ⬜ **Regenerate LiDermis_2021_long_format.csv**
   ```bash
   cd 05_papers_to_csv/11_LiDermis_2021_paper_to_csv
   python parse_lidermis.py
   ```
3. ⬜ **Re-run merge to unified database**
   ```bash
   cd 08_merged_ecm_dataset
   python merge_to_unified.py
   ```
4. ⬜ **Verify universal_zscore_function.py handles FOT data correctly**
   - Check if log2 transformation is applied to LiDermis rows
   - Inspect z-score distribution for LiDermis vs other studies

### Validation (Priority 2)

5. ⬜ **Compare before/after z-scores**
   - Extract LiDermis z-scores from current database
   - Extract LiDermis z-scores from regenerated database
   - Document differences

6. ⬜ **Dashboard QA**
   - Load updated database in dashboard
   - Check LiDermis heatmap values
   - Verify cross-study comparison accuracy

### Documentation (Priority 3)

7. ⬜ **Update 04_compilation_of_papers/06_LiDermis_2021_comprehensive_analysis.md**
   - Correct data scale assumption
   - Reference DATA_SCALE_VALIDATION_FROM_PAPERS.md

8. ⬜ **Update 00_README_compilation.md**
   - Add "Data Scale" column to processing table
   - Mark LiDermis as "FOT (LINEAR)"

---

## Prevention for Future Datasets

### Recommended Changes to Autonomous Agent

1. **Add data scale detection heuristic:**
   ```python
   # In autonomous_agent.py PHASE 1
   def detect_data_scale(df, abundance_cols):
       \"\"\"Heuristic to detect if data is LINEAR or LOG2 scale.\"\"\"
       median_val = df[abundance_cols].median().median()

       if median_val > 100:
           return "LINEAR (likely raw intensities or FOT)"
       elif median_val < 20:
           return "LOG2 (likely pre-transformed)"
       else:
           return "AMBIGUOUS (manual verification required)"
   ```

2. **Add metadata validation:**
   - Check paper Methods section for keywords: "log2", "log transformation", "FOT", "normalized intensity"
   - Flag ambiguous cases for manual review

3. **Standardize metadata schema:**
   - Add `Data_Scale` field: "LINEAR" | "LOG2" | "LOG10"
   - Add `Software_Used` field
   - Add `Normalization_Method` field

---

## Related Documents

- **Root cause analysis:** `/Users/Kravtsovd/projects/ecm-atlas/DATA_SCALE_VALIDATION_FROM_PAPERS.md`
- **Zero-to-NaN fix (Randles):** `05_papers_to_csv/05_Randles_paper_to_csv/PROCESSING_LOG_ZERO_FIX_2025-10-15.md`
- **LiDermis comprehensive analysis:** `04_compilation_of_papers/06_LiDermis_2021_comprehensive_analysis.md`

---

## Acknowledgment

**Discovered during:** Zero-to-NaN conversion investigation (Randles 2021)
**Validation method:** Parallel Task agents with paper Methods section extraction
**Confidence:** HIGH (direct quote from Li et al. 2021 Methods section)

---

**Status:** 🚨 AWAITING FIX - DO NOT MERGE NEW DATA UNTIL RESOLVED
**Created:** 2025-10-17
**Analyst:** Claude Code
