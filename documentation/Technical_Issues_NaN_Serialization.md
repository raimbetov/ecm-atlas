# Technical Issue: NaN JSON Serialization Bug (October 2025)

## Executive Summary

**Date Discovered:** October 14, 2025
**Severity:** Critical (Dashboard tab completely non-functional)
**Affected Component:** Compare Datasets tab (cross-study comparison)
**Status:** ✅ **RESOLVED**

**Root Cause:** Pandas `NaN` values in Gene_Symbol and Protein_Name fields were serialized as literal JavaScript `NaN` instead of JSON `null`, causing `SyntaxError: Unexpected token 'N'` in browser.

**Impact:** 16.4% of dataset entries (423/2575) contained `NaN` in Protein_Name field, with 1 critical entry having `NaN` in both Gene_Symbol and Protein_Name.

---

## Problem Description

### The Bug

When users clicked the "Compare Datasets" tab and selected Schuler 2021 studies, the heatmap failed to load with JavaScript console error:

```
SyntaxError: Unexpected token 'N', ..."ein_name":NaN},"HGF""... is not valid JSON
```

**Why it happened:**
1. Flask API endpoint `/api/compare/heatmap` returned JSON with literal `NaN` text
2. JavaScript `JSON.parse()` cannot parse `NaN` (only `null` is valid)
3. Frontend remained stuck on "Loading heatmap..." forever

### Visual Evidence

**Before fix:**
- Compare tab shows "Loading heatmap..." indefinitely
- Browser console shows JSON parse error
- API returns 200 OK but invalid JSON body

**After fix:**
- Heatmap renders correctly with 958 proteins across 12 compartments
- No JavaScript errors
- All filters work properly

---

## Data Quality Analysis

### NaN Statistics

| Column | NaN Count | % of Total | Impact |
|--------|-----------|------------|--------|
| **Gene_Symbol** | 1 / 2,575 | 0.04% | Critical - breaks JSON array |
| **Protein_Name** | 423 / 2,575 | 16.4% | High - breaks protein metadata |
| Protein_ID | 0 / 2,575 | 0% | None |
| Compartment | 0 / 2,575 | 0% | None |

### Source of NaN Values

#### 1. Schuler et al. 2021 (Skeletal Muscle Aging)

**Problem:** Original Excel files (`mmc2.xls`, `mmc3.xls`, etc.) have no `Protein_Name` column at all.

**Available fields:**
- `Protein IDs` (UniProt accessions)
- `Gene names` (Gene Symbols)
- Abundance data

**Missing field:**
- `Protein_Name` (full protein names)

**Affected records:**
```
Schuler_2021_EDL:           278 / 278  (100.0%)
Schuler_2021_TA:             65 / 65   (100.0%)
Schuler_2021_Soleus:         37 / 37   (100.0%)
Schuler_2021_Gastrocnemius:  18 / 18   (100.0%)
─────────────────────────────────────────────
Total:                      398 / 398  (100.0%)
```

**Data processing decision:** When converting Excel to unified CSV format, missing Protein_Name fields were left as `NaN` rather than attempting UniProt lookups (to preserve original data integrity).

#### 2. Angelidis et al. 2019 (Lung Aging)

**Problem:** Some UniProt IDs lack full protein names in the source data.

**Examples:**
- `A0A087WSN6` → Gene: FN1, Name: NaN (splice variant)
- `D3YXG0` → Gene: HMCN1, Name: NaN (partial sequence)
- `D3Z689` → Gene: ADAMTSL5, Name: NaN (isoform)

**Affected records:** 25 / 291 (8.6%)

**Explanation:** These are often:
- Unreviewed UniProt entries
- Splice variants or isoforms
- TrEMBL (computationally annotated) rather than Swiss-Prot (manually reviewed)

#### 3. Critical Record: P01635 (IGKC)

**The only record with NaN in BOTH Gene_Symbol AND Protein_Name:**

```
Protein_ID:        P01635
Protein_Name:      NaN
Gene_Symbol:       NaN
Canonical_Name:    Ig kappa chain C region (IGKC)
Dataset:           Schuler_2021_TA
Organ:             Skeletal muscle
Compartment:       TA (Tibialis anterior)
Match_Level:       Unmatched
Matrisome:         Non-ECM
```

**Why this happened:**
- IGKC is an immunoglobulin constant region (antibody component)
- Likely a contaminant from mouse serum/blood during tissue extraction
- Original Schuler 2021 Excel file had empty `Gene names` column for this entry
- Not a real skeletal muscle protein

**Scientific note:** Mass spectrometry often picks up immunoglobulins as background noise, especially in tissue samples with blood contamination.

---

## Technical Root Cause

### Old Code (Buggy)

```python
# api_server.py, line 340 (before fix)
proteins = filtered_df['Gene_Symbol'].unique()
compartments = sorted(filtered_df['Compartment'].unique())

# Line 355-359 (before fix)
protein_metadata[protein] = {
    "protein_id": first_row['Protein_ID'],
    "protein_name": first_row['Protein_Name'],  # ← NaN becomes literal "NaN"
    "matrisome_category": first_row['Matrisome_Category'] if pd.notna(...) else None
}

# Line 374-380 (before fix)
heatmap_data[protein][compartment] = {
    "zscore_delta": float(row['Zscore_Delta']) if pd.notna(...) else None,
    "zscore_young": float(row['Zscore_Young']) if pd.notna(...) else None,
    "zscore_old": float(row['Zscore_Old']) if pd.notna(...) else None,
    "dataset": row['Dataset_Name'],  # ← NaN becomes literal "NaN"
    "organ": row['Organ'],
    ...
}
```

### What Happened

1. **NumPy array with NaN:**
   ```python
   >>> proteins = df['Gene_Symbol'].unique()
   >>> proteins
   array(['Col1a1', 'Fn1', ..., nan, 'Eln'], dtype=object)
   ```

2. **Flask JSON serialization:**
   ```python
   >>> jsonify({"proteins": list(proteins)})
   {"proteins": ["Col1a1", "Fn1", ..., NaN, "Eln"]}
   #                                       ↑ Invalid JSON!
   ```

3. **JavaScript parse error:**
   ```javascript
   fetch('/api/compare/heatmap')
     .then(r => r.json())  // ← Throws SyntaxError here
   ```

---

## Solution Implemented

### Three-Layer Defense

#### Layer 1: Filter NaN from Arrays

```python
# api_server.py, line 339-340 (after fix)
compartments = sorted([c for c in filtered_df['Compartment'].unique() if pd.notna(c)])
proteins = [p for p in filtered_df['Gene_Symbol'].unique() if pd.notna(p)]
```

**Effect:** P01635 (IGKC) is excluded from protein list entirely since Gene_Symbol is NaN.

#### Layer 2: JSON-Safe Wrapper Function

```python
# api_server.py, line 17-25 (helper function)
def to_json_safe(value):
    """Convert pandas/numpy values to JSON-safe format (replace NaN with None)"""
    if pd.isna(value):
        return None  # ← Becomes JSON 'null'
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value) if isinstance(value, np.floating) else int(value)
    return value
```

**Usage:**
```python
# Line 355-359 (after fix)
protein_metadata[protein] = {
    "protein_id": to_json_safe(first_row['Protein_ID']),
    "protein_name": to_json_safe(first_row['Protein_Name']),  # NaN → null
    "matrisome_category": to_json_safe(first_row['Matrisome_Category'])
}

# Line 374-380 (after fix)
heatmap_data[protein][compartment] = {
    "zscore_delta": to_json_safe(row['Zscore_Delta']),
    "zscore_young": to_json_safe(row['Zscore_Young']),
    "zscore_old": to_json_safe(row['Zscore_Old']),
    "dataset": to_json_safe(row['Dataset_Name']),
    "organ": to_json_safe(row['Organ']),
    "isoforms": int(len(compartment_data))
}
```

#### Layer 3: Empty Data Guard

```python
# api_server.py, line 348-350 (after fix)
for protein in proteins:
    protein_data = filtered_df[filtered_df['Gene_Symbol'] == protein]

    # Skip if no data for this protein after filtering
    if len(protein_data) == 0:
        continue

    first_row = protein_data.iloc[0]  # Safe now
```

**Effect:** Prevents `IndexError: single positional indexer is out-of-bounds` when DataFrame is empty.

---

## Testing & Verification

### Headless Browser Tests

**Test script:** `10_unified_dashboard_2_tabs/test_compare_tab.py`

**Results after fix:**
```
✅ Valid JSON: 958 proteins, 12 compartments
✅ No JavaScript errors detected
✅ Heatmap renders correctly
✅ All filters functional
```

**Screenshot evidence:**
- `test_screenshots/01_individual_tab_*.png` - First tab working
- `test_screenshots/02_after_click_*.png` - Tab switch successful
- `test_screenshots/03_final_state_*.png` - Heatmap fully rendered

### API Validation

**Before fix:**
```bash
$ curl 'http://localhost:5004/api/compare/heatmap?...' | grep -c 'NaN'
1  # ← Found literal NaN in JSON
```

**After fix:**
```bash
$ curl 'http://localhost:5004/api/compare/heatmap?...' | python -m json.tool
✅ Valid JSON: 958 proteins, 12 compartments
```

---

## Impact Assessment

### What Works Now

✅ **Compare Datasets tab fully functional**
- Filters: Organs, Compartments, Categories, Studies
- Heatmap: 958 proteins × 12 compartments
- Tooltips: Protein names, z-scores, datasets
- Sorting: By relevance, alphabetical

✅ **Data integrity preserved**
- No data loss (only 1 protein excluded: P01635 IGKC)
- All 397 Schuler 2021 proteins with valid Gene_Symbol displayed
- Protein_Name shown as "null" in tooltips when missing (acceptable UX)

### What Changed

⚠️ **1 protein excluded from visualization:**
- **P01635 (IGKC)** - Ig kappa chain C region
- Reason: No Gene_Symbol available
- Impact: Negligible (was likely a contaminant anyway)

ℹ️ **398 proteins display with Protein_Name = null:**
- All Schuler 2021 records (expected, source data lacks this field)
- Gene_Symbol still displayed correctly
- Scientific validity unchanged

---

## Lessons Learned

### 1. Data Quality Assumptions

**Mistake:** Assumed all records would have Gene_Symbol (primary identifier).

**Reality:** 1 in 2,575 records (P01635) had NaN Gene_Symbol.

**Learning:** Always filter `NaN` from arrays before JSON serialization, even for "required" fields.

### 2. Pandas → JSON Serialization

**Mistake:** Relied on Flask's default JSON encoder to handle pandas types.

**Reality:** Pandas `NaN` serializes as JavaScript `NaN` (invalid JSON), not `null`.

**Learning:** Always use explicit conversion helpers (`to_json_safe()`) for pandas DataFrames.

### 3. Silent Failures

**Mistake:** API returned 200 OK with invalid JSON body.

**Reality:** JavaScript failed silently, no server-side error logged.

**Learning:** Add JSON validation middleware or use stricter serializers (e.g., Marshmallow, Pydantic).

### 4. Testing Gaps

**Mistake:** No automated tests for API JSON validity.

**Reality:** Bug only caught when user manually tested Compare tab.

**Learning:** Implement:
- JSON schema validation tests
- Headless browser E2E tests
- API contract tests with edge cases

---

## Recommendations

### Short-Term (Completed ✅)

- [x] Filter NaN from protein/compartment arrays
- [x] Wrap all DataFrame values in `to_json_safe()`
- [x] Add empty data guards
- [x] Verify with headless browser tests
- [x] Document issue in this file

### Medium-Term (TODO)

- [ ] Add pytest suite for API endpoints
  - Test with NaN values explicitly
  - Test with empty DataFrames
  - Validate JSON schema compliance

- [ ] Consider enriching Schuler 2021 data
  - Fetch Protein_Name from UniProt API
  - Store in separate enrichment CSV
  - Merge at runtime

- [ ] Add API response validation middleware
  - Check for literal `NaN` in JSON strings
  - Log warnings for missing critical fields
  - Return 500 error instead of invalid JSON

### Long-Term (Consideration)

- [ ] Migrate to proper API framework
  - FastAPI with Pydantic models (auto-validates types)
  - Marshmallow schemas for Flask
  - OpenAPI spec generation

- [ ] Database backend
  - PostgreSQL with proper NULL handling
  - Constraints on required fields
  - Data quality checks at ingestion

- [ ] Improve data pipeline
  - Standardize all datasets to uniform schema
  - Enrich missing fields at processing stage
  - Flag low-quality records with metadata

---

## References

### Related Files

- **Bug fix commit:** `c7c4623` - "fix: resolve Compare tab JSON serialization errors"
- **API server:** `/10_unified_dashboard_2_tabs/api_server.py`
- **Test script:** `/10_unified_dashboard_2_tabs/test_compare_tab.py`
- **Source data:** `/data_raw/Schuler et al. - 2021/mmc2.xls` (and others)
- **Merged dataset:** `/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

### External Resources

- [RFC 8259 - JSON Specification](https://datatracker.ietf.org/doc/html/rfc8259)
  - Section 3: "The only values allowed are `true`, `false`, and `null`" (not `NaN`)

- [Pandas to_json() documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)
  - Default behavior: `NaN` values are converted to `NaN` (not `null`)

- [Flask jsonify() source code](https://github.com/pallets/flask/blob/main/src/flask/json/__init__.py)
  - Uses Python's `json.dumps()` which doesn't handle pandas types

### Protein References

- **P01635 (IGKC):** [UniProt Entry](https://www.uniprot.org/uniprotkb/P01635)
  - Name: Ig kappa chain C region
  - Organism: Mus musculus (Mouse)
  - Function: Immunoglobulin constant region

---

## Appendix: Full Error Stack Trace

### Browser Console Error (Before Fix)

```
[log] ECM Atlas Dashboard v1.2.0
[log] Individual Dataset module initialized
[log] Compare Datasets module initialized
[error] Failed to load resource: the server responded with a status of 500 (INTERNAL SERVER ERROR)
[error] Error loading heatmap: SyntaxError: Unexpected token 'N', ..."ein_name":NaN},"HGF""... is not valid JSON
    at Object.fetchAPI (http://localhost:8080/static/main.js:151:15)
    at async loadHeatmap (http://localhost:8080/static/compare_datasets.js:238:26)
    at async loadCompareTab (http://localhost:8080/static/compare_datasets.js:24:13)
```

### Python Stack Trace (Before Empty Data Guard)

```
[2025-10-14 16:26:08,808] ERROR in app: Exception on /api/compare/heatmap [GET]
Traceback (most recent call last):
  File "/venv/lib/python3.11/site-packages/flask/app.py", line 1511, in wsgi_app
    response = self.full_dispatch_request()
  File "/venv/lib/python3.11/site-packages/flask/app.py", line 919, in full_dispatch_request
    rv = self.handle_user_exception(e)
  File "/venv/lib/python3.11/site-packages/flask_cors/extension.py", line 176, in wrapped_function
    return cors_after_request(app.make_response(f(*args, **kwargs)))
  File "/venv/lib/python3.11/site-packages/flask/app.py", line 917, in full_dispatch_request
    rv = self.dispatch_request()
  File "/venv/lib/python3.11/site-packages/flask/app.py", line 902, in dispatch_request
    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)
  File "/api_server.py", line 349, in get_compare_heatmap
    first_row = protein_data.iloc[0]
  File "/venv/lib/python3.11/site-packages/pandas/core/indexing.py", line 1192, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)
  File "/venv/lib/python3.11/site-packages/pandas/core/indexing.py", line 1753, in _getitem_axis
    self._validate_integer(key, axis)
  File "/venv/lib/python3.11/site-packages/pandas/core/indexing.py", line 1686, in _validate_integer
    raise IndexError("single positional indexer is out-of-bounds")
IndexError: single positional indexer is out-of-bounds
```

---

**Document Version:** 1.0
**Last Updated:** October 14, 2025
**Author:** Daniel Kravtsov (with assistance from Claude Code)
**Status:** Final
