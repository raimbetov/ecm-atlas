# NaN JSON Serialization Bug - Fixed

**Date:** October 14, 2025
**Status:** ✅ RESOLVED
**Severity:** Critical (Dashboard Compare tab completely broken)

---

## Problem

### Symptom
Compare Datasets tab showed "Loading heatmap..." forever with JavaScript error:
```
SyntaxError: Unexpected token 'N', ..."ein_name":NaN}... is not valid JSON
```

### Root Cause
- Flask API returned pandas `NaN` values as literal JavaScript `NaN` in JSON
- JavaScript `JSON.parse()` only accepts `null`, not `NaN`
- **16.4% of dataset (423/2575 records)** had missing `Protein_Name`

### Data Source Issue
**Schuler 2021 datasets:** Source Excel files have NO `Protein_Name` column
- Schuler_2021_EDL: 278/278 missing (100%)
- Schuler_2021_TA: 65/65 missing (100%)
- Schuler_2021_Soleus: 37/37 missing (100%)
- Schuler_2021_Gastrocnemius: 18/18 missing (100%)

**Critical case:** Protein P01635 (IGKC) had NaN in BOTH Gene_Symbol AND Protein_Name

---

## Solution: 3-Layer Defense Architecture

### Layer 1: API Code Fixes (Immediate)
**Location:** `10_unified_dashboard_2_tabs/api_server.py`

```python
# Filter NaN from arrays (line 363-364)
compartments = sorted([c for c in filtered_df['Compartment'].unique() if pd.notna(c)])
proteins = [p for p in filtered_df['Gene_Symbol'].unique() if pd.notna(p)]

# JSON-safe wrapper for all values (line 379-383)
protein_metadata[protein] = {
    "protein_id": to_json_safe(first_row['Protein_ID']),
    "protein_name": to_json_safe(first_row['Protein_Name']),  # NaN → null
    "matrisome_category": to_json_safe(first_row['Matrisome_Category'])
}

# Global NaN-safe JSON encoder (line 15-34)
class NaNSafeJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if pd.isna(obj):
            return None  # Convert NaN to JSON null
        # ... handle numpy types
```

### Layer 2: Data Enrichment (Root Fix)
**Location:** `08_merged_ecm_dataset/enrich_missing_metadata.py`

Fetched missing metadata from UniProt REST API:
- Queried **422 unique proteins** with missing metadata
- Successfully enriched **412 proteins** (97.6% success rate)
- **10 proteins** remain NaN (UniProt lacks data for these)
- `Protein_Name`: 423 records → 10 NaN (fixed 413 records)
- `Gene_Symbol`: 1 NaN → 0 NaN (fixed all)

**Process:**
1. Query UniProt API: `https://rest.uniprot.org/uniprotkb/{protein_id}.json`
2. Extract protein name and gene symbol
3. Save enrichment table: `merged_ecm_enrichment.csv`
4. Replace main dataset with enriched version

### Layer 3: Pipeline Documentation (Future Prevention)
**Location:** `11_subagent_for_LFQ_ingestion/07_PROTEIN_METADATA_ENRICHMENT.md`

Added enrichment step to data ingestion pipeline:
- Run after merging all studies
- Prevents future NaN issues at data source
- Documents enrichment process for reproducibility

---

## Results

### Before Fix
- ❌ Compare tab: Infinite loading spinner
- ❌ Console error: JSON parse failure
- ❌ Missing metadata: 423 proteins (16.4%)

### After Fix
- ✅ Compare tab: Heatmap renders 958 proteins × 12 compartments
- ✅ No JavaScript errors
- ✅ Missing metadata: 10 proteins (<1%)
- ✅ All filters and tooltips functional

### Testing
```bash
# Headless browser test
python3 test_compare_tab.py
✅ Valid JSON: 958 proteins, 12 compartments
✅ No JavaScript errors
✅ Screenshots captured in test_screenshots/
```

---

## Key Learnings

1. **Never trust data completeness** - Even "required" fields can have NaN
2. **Pandas NaN ≠ JSON null** - Always convert explicitly with `pd.notna()` or custom encoder
3. **Fix at data layer** - Enrichment at ETL prevents API workarounds
4. **Defense in depth** - Multiple protection layers catch edge cases

---

## Files Changed

**Fixes:**
- `10_unified_dashboard_2_tabs/api_server.py` - NaN filtering + JSON encoder
- `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` - Replaced with enriched version

**New files:**
- `08_merged_ecm_dataset/enrich_missing_metadata.py` - UniProt enrichment script
- `08_merged_ecm_dataset/merged_ecm_enrichment.csv` - Lookup table (422 proteins)
- `11_subagent_for_LFQ_ingestion/07_PROTEIN_METADATA_ENRICHMENT.md` - Pipeline docs

**Commit:** `96267d1` - feat: enrich protein metadata from UniProt API to fix NaN serialization

---

## References

- **UniProt REST API:** https://rest.uniprot.org/help/api
- **RFC 8259 (JSON spec):** NaN is NOT valid JSON, only `null`
- **Example protein:** [P01635 (IGKC)](https://www.uniprot.org/uniprotkb/P01635) - The critical case with double NaN

---

**Document Version:** 2.0 (Condensed)
**Last Updated:** October 14, 2025
**Author:** Daniel Kravtsov
