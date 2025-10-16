# Zero-to-NaN Fix: Impact Assessment Across 11 Datasets

**Date:** 2025-10-15
**Analysis:** Parallel audit by 11 specialized agents

---

## Executive Summary

**Datasets requiring fix:** 3 of 11 (27%)
**Datasets already clean:** 7 of 11 (64%)
**Datasets not applicable:** 1 of 11 (9%)

**Total impact:** ~1,500 zero values across 3 datasets will convert to NaN, affecting 100-250 proteins in merged database.

---

## Impact by Dataset (Ranked by Severity)

### 🔴 HIGH IMPACT (Fix Required)

#### 1. LiDermis_2021 - **CRITICAL**
- **Zeros in source:** 996 / 3,682 values (27.1%)
- **Zeros after averaging:** 82 Old (31.3%), 22 Young (8.4%)
- **Proteins with both=0:** 8 proteins (will be removed)
- **Example impact:** Protein with [0,0,0,1500] replicates: current mean=375 → fixed mean=1500 (+300%)
- **Root cause:** Elderly samples have 45-55% zeros (detection limit issue)
- **Action:** MANDATORY fix - current averaging heavily distorts Old group data

#### 2. Randles_2021 - **MODERATE**
- **Zeros in source:** 483 / 31,320 values (1.54%)
- **Zeros after averaging:** 25 proteins (0.24%)
- **Example impact:** AGRN protein [0,0,0,40,91] → current mean=26 → fixed mean=65 (+150%)
- **Technical replicates:** 3 donors per age/compartment
- **Action:** FIX recommended - affects kidney glomerular/tubulointerstitial comparisons

#### 3. Ouni_2022 - **LOW**
- **Zeros in source:** 22 / 1,530 values (1.44%)
- **Zeros after averaging:** 0 (diluted across 5 replicates)
- **Example impact:** PLG protein [0,242,171,0,193] → current mean=121 → fixed mean=202 (+67%)
- **Technical replicates:** 5 replicates per age group (TMT)
- **Action:** FIX recommended - improves accuracy for 5+ proteins

---

### ✅ ZERO IMPACT (Already Clean)

#### 4. Caldeira_2017
- **Zeros:** 0 (iTRAQ ratios already handle missing as NaN)
- **Action:** None required

#### 5. Tam_2020
- **Zeros:** 0 (MaxQuant LFQ uses NaN for missing)
- **Missing values:** 76.5% as NaN (correct)
- **Action:** None required

#### 6. Angelidis_2019
- **Zeros:** 0 (log2-transformed LFQ, NaN for missing)
- **Action:** None required

#### 7. Dipali_2023
- **Zeros:** 0 (DIA with 100% quantification coverage)
- **Action:** None required

#### 8. Tsumagari_2023
- **Zeros:** 0 (TMT multiplexing ensures complete data)
- **Action:** None required

#### 9. Schuler_2021
- **Zeros:** 0 (pre-filtered, publication-quality)
- **Action:** None required

#### 10. Santinha_2024
- **Zeros:** 0 (TMT-10plex with aggregated statistics)
- **Action:** None required

---

### ⚠️ NOT APPLICABLE

#### 11. Lofaro_2021
- **Status:** Only aggregated statistics (Log2FC, p-values) available
- **Issue:** No sample-level abundances in published data
- **Action:** Excluded from zero-to-NaN task (requires RAW data reprocessing)

---

## Statistical Impact on Merged Database

**Current database:** 2,177 proteins

**Expected changes after 0→NaN fix:**

| Dataset | Proteins Affected | Current Zeros | After Fix (NaN) | Mean Change |
|---------|------------------|---------------|-----------------|-------------|
| LiDermis_2021 | ~100 | 104 | 104 NaN | +50-300% |
| Randles_2021 | ~25 | 25 | 25 NaN | +50-150% |
| Ouni_2022 | ~5 | 0 (diluted) | varies | +25-67% |
| **TOTAL** | **~130** | **~130** | **~130 NaN** | **+25-300%** |

**Percentage of database:** ~6% of proteins will have more accurate abundance estimates

---

## Technical Implementation Priority

### Phase 1: High Priority (Execute First)
1. **LiDermis_2021** - 27% zeros, critical distortion in Old group
   - Modify adapter: `li_dermis_adapter.py`
   - Add 0→NaN at line ~40 (after read_excel)
   - Reprocess: 262 proteins

### Phase 2: Medium Priority
2. **Randles_2021** - 1.5% zeros, kidney-specific
   - Modify adapter: `randles_adapter.py`
   - Add 0→NaN during sample reading
   - Reprocess: 458 proteins

3. **Ouni_2022** - 1.4% zeros, ovarian tissue
   - Modify adapter: `ouni_adapter.py`
   - Add 0→NaN in TMT intensity columns
   - Reprocess: 98 proteins

### Phase 3: Validation
4. **All other datasets** - verify no regression (should remain unchanged)

---

## Code Template for Fix

```python
# Add to each adapter script immediately after reading source file

def convert_zeros_to_nan(df, abundance_columns):
    """
    Convert zero values to NaN in abundance/LFQ/intensity columns.

    Zero in proteomics = protein not detected (missing measurement),
    not true zero abundance (biologically impossible).
    """
    for col in abundance_columns:
        df[col] = df[col].replace(0, np.nan)

    return df

# Usage in adapter:
df = pd.read_excel(source_file)
abundance_cols = [c for c in df.columns if any(x in c.lower() for x in ['abundance', 'lfq', 'intensity', 'tmt'])]
df = convert_zeros_to_nan(df, abundance_cols)
```

---

## Validation Checklist

After implementing fix:

**Per-dataset validation:**
- [ ] LiDermis_2021: 996 zeros → NaN in source, 104 → NaN in wide_format
- [ ] Randles_2021: 483 zeros → NaN in source, 25 → NaN in wide_format
- [ ] Ouni_2022: 22 zeros → NaN in source, recalculate means

**Database-level validation:**
- [ ] merged_ecm_aging_zscore.csv: 130 proteins have updated abundances
- [ ] Zero records with "Both old=0 AND young=0" (should drop 8 → ~0)
- [ ] Z-scores recalculated: mean≈0, std≈1 within each study
- [ ] Dashboard loads without errors

**Statistical validation:**
- [ ] Compare old vs new z-scores for affected proteins
- [ ] Verify LiDermis Old group no longer has 31% zeros
- [ ] Confirm mean increases by expected % for test cases

---

## Estimated Effort

**Development:** 2-4 hours
- Modify 3 adapter scripts
- Add zero_to_nan() function
- Update processing metadata

**Execution:** 10-15 minutes
- Reprocess 3 datasets (818 proteins total)
- Merge to unified database
- Recalculate z-scores

**Validation:** 1-2 hours
- Run validation checks
- Compare before/after statistics
- Update dashboard

**Total:** ~4-7 hours for complete implementation and validation

---

## Risk Assessment

**Low Risk:**
- Only 3 datasets affected (27% of total)
- Clear test cases with expected outcomes
- Easy to validate (compare old/new z-scores)
- Reversible (backups exist in 08_merged_ecm_dataset/backups/)

**High Reward:**
- Fixes critical issue in LiDermis (31% zeros in Old group)
- Improves biological accuracy (no false "zero abundance")
- Aligns with proteomics best practices
- Enables proper statistical inference

---

## Recommendation

**PROCEED with zero-to-NaN fix in priority order:**

1. Start with LiDermis_2021 (highest impact)
2. Validate results before proceeding to Randles_2021 and Ouni_2022
3. Skip clean datasets (no changes needed)
4. Defer Lofaro_2021 (requires RAW data reprocessing)

**Expected outcome:** More accurate abundance estimates for ~6% of database with minimal risk.
