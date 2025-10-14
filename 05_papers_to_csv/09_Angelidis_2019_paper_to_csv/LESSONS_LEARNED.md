# Lessons Learned: Angelidis 2019 Processing

**Date:** 2025-10-13
**Task:** Manual processing to improve autonomous agent
**Result:** ✅ SUCCESS - All 3 phases completed

---

## 📊 Summary

**Input:** `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`
**Output:** 291 ECM proteins in unified CSV with z-scores

**Processing time:**
- PHASE 1: ~3 minutes (manual scripting + debugging)
- PHASE 2: ~5 seconds
- PHASE 3: ~3 seconds

---

## ✅ What Worked Well

### 1. Comprehensive Analysis File
- **File:** `04_compilation_of_papers/01_Angelidis_2019_comprehensive_analysis.md`
- **Value:** Contains almost all necessary parameters:
  - Data file path ✅
  - Sheet name ✅
  - Column mappings ✅
  - Age groups ✅
  - Expected output rows ✅

**For Agent:** Always start by reading comprehensive analysis!

### 2. Auto-Detection
- **Project root detection:** Works via `references/human_matrisome_v2.csv` check
- **Largest file selection:** Good heuristic when multiple files present
- **Sheet detection:** Keywords "data", "protein", "abundance", "matrix" work

### 3. Logging System
- Real-time timestamps
- Step-by-step progress
- Error tracebacks captured

### 4. Missing Value Handling
- `skipna=True` in aggregation preserved NaN
- 3-5% missing abundances is normal for LFQ
- No imputation needed

---

## ❌ Problems Encountered

### Problem 1: Column Name Mismatch

**Error:**
```
KeyError: "['Category', 'Division'] not in index"
```

**Root Cause:**
- Comprehensive analysis said: `"Category"`, `"Division"`, `"UniProt ID"`
- Reality: `"Matrisome Category"`, `"Matrisome Division"`, `"UniProt_IDs"`

**Solution:**
```python
# Always inspect actual columns first!
log(f"Matrisome columns: {df_matrisome.columns.tolist()}")
```

**For Agent:**
- ⚠️ **Never trust documentation** - always inspect real column names
- **Step 0.5:** Read reference files and log actual columns
- **Validation:** Check expected columns exist before using them

---

### Problem 2: Multiple UniProt IDs per Gene

**Discovery:**
- Mouse matrisome `UniProt_IDs` column contains multiple IDs separated by colons
- Example: `"Q60994:Q8BRW2:E9PWU4"`

**Solution:**
```python
uniprot_lookup = {}
for _, row in df_matrisome.iterrows():
    uniprots = str(row['UniProt_IDs']).split(':')
    for up_id in uniprots:
        if up_id and up_id != 'nan':
            uniprot_lookup[up_id.strip()] = row_data
```

**For Agent:**
- UniProt matching requires parsing colon-separated lists
- Create lookup dict from ALL individual IDs

---

### Problem 3: F-string Formatting Error

**Error:**
```
ValueError: Invalid format specifier '.2f if pd.notna(...) else 'NaN''
```

**Root Cause:**
- Can't use conditional in same f-string expression as format spec

**Solution:**
```python
# WRONG:
f"{value:.2f if pd.notna(value) else 'NaN'}"

# CORRECT:
val_str = f"{value:.2f}" if pd.notna(value) else 'NaN'
f"Value: {val_str}"
```

**For Agent:**
- Pre-format conditional values before f-string

---

### Problem 4: Pandas FutureWarning

**Warning:**
```
FutureWarning: A value is trying to be set on a copy of a DataFrame
```

**Root Cause:**
```python
df_annotated['Column'].fillna(value, inplace=True)
```

**Solution:**
```python
df_annotated = df_annotated.assign(
    Column=df_annotated['Column'].fillna(value)
)
```

**For Agent:**
- Use `.assign()` or explicit assignment instead of `inplace=True`

---

### Problem 5: Excel Header Row

**Discovery:**
- Column 0 was: `"Differential protein abundance in total lung tissue proteomes..."`
- This was a descriptive header, but pandas read it as column name

**Actual columns:**
- Column 1: `Majority protein IDs`
- Column 2: `Protein names`
- Column 3: `Gene names`

**For Agent:**
- ⚠️ **Check for descriptive headers** - may need `skiprows=1`
- Log first 10 columns to detect this

---

### Problem 6: Path Resolution

**Error:**
```
FileNotFoundError: ../09_Angelidis_2019_paper_to_csv/Angelidis_2019_wide_format.csv
```

**Root Cause:**
- Relative path used from wrong directory

**Solution:**
- Always use project root for path resolution
- Convert to absolute paths internally

---

## 📋 Agent Improvements Needed

### 1. Add Column Inspection Step

**Where:** PHASE 0 (Reconnaissance)

```python
def _inspect_columns(self):
    """Inspect actual columns in data files and references."""
    # Check data file
    df_preview = pd.read_excel(data_file, sheet_name=sheet, nrows=0)
    self._log(f"Data columns: {df_preview.columns.tolist()}")

    # Check matrisome reference
    df_matrisome = pd.read_csv(matrisome_path, nrows=0)
    self._log(f"Matrisome columns: {df_matrisome.columns.tolist()}")

    # Validate expected columns exist
    required_cols = ['Matrisome Category', 'Matrisome Division', 'Gene Symbol']
    missing = [col for col in required_cols if col not in df_matrisome.columns]
    if missing:
        self._log_error(f"Missing matrisome columns: {missing}")
```

---

### 2. Add Comprehensive Analysis Parser

**Where:** PHASE 0

```python
def _parse_comprehensive_analysis(self, paper_folder):
    """Extract parameters from comprehensive analysis file."""
    # Look for file in 04_compilation_of_papers/
    analysis_files = list(Path("04_compilation_of_papers").glob("*comprehensive_analysis.md"))

    # Match by study name
    for f in analysis_files:
        if study_id.lower() in f.name.lower():
            # Parse markdown, extract:
            # - Data file path
            # - Sheet name
            # - Column mappings
            # - Age groups
            # - Expected rows
            return config_dict
```

---

### 3. Add Validation Step

**Where:** After each phase

```python
def _validate_phase1_output(self, df_wide):
    """Validate wide format output."""
    errors = []

    # Check expected columns
    expected_cols = [
        'Protein_ID', 'Protein_Name', 'Gene_Symbol',
        'Canonical_Gene_Symbol', 'Matrisome Category', 'Matrisome Division',
        'Tissue', 'Tissue_Compartment', 'Species',
        'Abundance_Young', 'Abundance_Old',
        'Method', 'Study_ID', 'Match_Level', 'Match_Confidence'
    ]

    missing_cols = [c for c in expected_cols if c not in df_wide.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check ECM protein count (should be > 0)
    ecm_count = (df_wide['Match_Confidence'] > 0).sum()
    if ecm_count == 0:
        errors.append("No ECM proteins found!")

    # Check expected row count (if available from comprehensive analysis)
    if 'expected_ecm_proteins' in config:
        if abs(ecm_count - config['expected_ecm_proteins']) > 50:
            errors.append(f"ECM count {ecm_count} differs from expected {config['expected_ecm_proteins']}")

    return errors
```

---

### 4. Add Error Recovery

**Where:** Each phase

```python
try:
    result = self._normalize_data()
except ColumnNotFoundError as e:
    # Try to auto-correct common column name issues
    self._log(f"Column error: {e}")
    self._log("Attempting auto-correction...")

    # Try variations
    variations = {
        'Category': ['Matrisome Category', 'ECM_Category', 'category'],
        'Division': ['Matrisome Division', 'ECM_Division', 'division'],
        'UniProt ID': ['UniProt_IDs', 'UniProt_ID', 'UniProtID']
    }

    # Retry with corrected column names
    result = self._normalize_data(column_corrections=corrections)
```

---

## 📝 Instructions for Autonomous Agent

### Phase 0: Reconnaissance (Enhanced)

```
1. Find paper folder
2. Look for comprehensive analysis in 04_compilation_of_papers/
3. Parse comprehensive analysis to extract:
   - Data file path
   - Sheet name
   - Column mappings (BUT DON'T TRUST THEM!)
   - Age groups
   - Species
   - Tissue
4. Load data file (first 0 rows to get columns)
5. Log actual column names
6. Load matrisome reference (first 0 rows)
7. Log actual matrisome column names
8. Compare comprehensive analysis columns vs actual columns
9. If mismatch detected, log warning and use actual columns
10. Generate config with ACTUAL column names
```

### Phase 1: Data Normalization (Enhanced)

```
1. Load Excel with error handling for header rows
2. Inspect first 10 columns - check for descriptive headers
3. If column[0] looks like description (>50 chars), try skiprows=1
4. Extract protein IDs (handle semicolon-separated)
5. Transform to long format
6. Load matrisome reference
7. Build UniProt lookup (handle colon-separated IDs)
8. Annotate ECM (4-level hierarchy)
9. Filter to ECM only (Match_Confidence > 0)
10. Aggregate by age (skipna=True)
11. Convert to wide format
12. Validate output
13. Save with logging
```

---

## 🎯 Success Metrics

### PHASE 1 Output
- ✅ 291 ECM proteins (5.6% of 5213 total)
- ✅ Wide format CSV created
- ✅ All required columns present
- ✅ Missing values preserved (3-5%)

### PHASE 2 Output
- ✅ Merged to unified CSV
- ✅ Metadata updated
- ✅ Backup created

### PHASE 3 Output
- ✅ Z-scores calculated (μ=0, σ=1)
- ✅ No log-transformation needed
- ✅ No outliers (|z| > 3)

---

## 📦 Files Created

```
09_Angelidis_2019_paper_to_csv/
├── agent_log.md                          # Complete execution log
├── parse_angelidis.py                    # PHASE 1a: Long format
├── annotate_ecm.py                       # PHASE 1b: ECM annotation
├── convert_to_wide.py                    # PHASE 1c: Wide format
├── Angelidis_2019_long_format.csv        # 41,704 rows (all proteins)
├── Angelidis_2019_long_annotated.csv     # 41,704 rows (with ECM annotation)
├── Angelidis_2019_wide_format.csv        # 291 ECM proteins ✅
└── LESSONS_LEARNED.md                    # This file

08_merged_ecm_dataset/
├── ECM_Atlas_Unified.csv                 # 291 rows (Angelidis only)
├── unified_metadata.json                 # Study metadata
├── zscore_metadata_Angelidis_2019.json   # Z-score parameters
└── backups/
    └── ECM_Atlas_Unified_*.csv           # Auto-backup
```

---

## 🚀 Next Steps for Agent Development

1. **Implement enhanced PHASE 0**
   - Comprehensive analysis parser
   - Column inspection
   - Auto-correction of column name mismatches

2. **Add validation after each phase**
   - Schema validation
   - ECM protein count check
   - Expected row count comparison

3. **Improve error recovery**
   - Column name variations
   - Header row detection
   - Retry logic

4. **Test on another study**
   - Use Randles 2021 (already processed)
   - Compare agent output vs manual output
   - Measure success rate

5. **Create universal templates**
   - MaxQuant LFQ template (Angelidis, Tam, Randles)
   - TMT template (Tsumagari)
   - Species-specific templates (Human vs Mouse)

---

## 💡 Key Insights

1. **Comprehensive analysis is valuable but not perfect**
   - Use as starting point
   - Always validate against real data
   - Don't skip inspection steps

2. **Column name standardization is critical**
   - Different matrisome versions use different names
   - Need flexible matching ("Category" → "Matrisome Category")

3. **LLM would help here!**
   - Could read comprehensive analysis
   - Could inspect actual columns
   - Could auto-correct mismatches
   - Could generate parsing code on the fly

4. **Manual process took ~10 minutes**
   - But taught us exactly what agent needs to do
   - Now we know all the edge cases
   - Agent can be built to handle these automatically

---

**Status:** Ready to implement enhanced autonomous agent
**Next:** Apply lessons to improve `autonomous_agent.py`
