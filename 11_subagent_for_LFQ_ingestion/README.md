# LFQ Dataset Ingestion: Sub-Agent Instructions

This directory contains universal algorithms and documentation for processing any LFQ proteomics dataset and adding it to the ECM Atlas unified database.

---

## Contents

### 📖 Documentation

1. **`01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md`**
   - How to process a new LFQ dataset (PHASE 1)
   - How to merge it into unified CSV (PHASE 2)
   - Complete step-by-step algorithm
   - Missing value handling guidelines
   - Compartment separation requirements

2. **`02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md`**
   - How z-scores are calculated (PHASE 3)
   - Universal function documentation
   - Grouping strategies
   - Validation procedures

### 🔧 Code

3. **`universal_zscore_function.py`**
   - Ready-to-use Python function
   - Command-line interface
   - Automatic backup creation
   - Metadata generation

---

## Quick Start

### Scenario: Adding a new study to ECM Atlas

```bash
# Step 1: Process new study (PHASE 1)
# Follow instructions in 01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md
# Output: Study_YYYY_wide_format.csv

# Step 2: Merge to unified CSV (PHASE 2)
python merge_study.py Study_YYYY_wide_format.csv
# Output: ECM_Atlas_Unified.csv updated

# Step 3: Calculate z-scores (PHASE 3)
cd /Users/Kravtsovd/projects/ecm-atlas/11_subagent_for_LFQ_ingestion
python universal_zscore_function.py Study_YYYY Tissue
# Output: ECM_Atlas_Unified.csv with z-scores
```

---

## File Paths

### Input Data
- Raw files: `/Users/Kravtsovd/projects/ecm-atlas/data_raw/Study et al. - YYYY/`
- Reference lists: `/Users/Kravtsovd/projects/ecm-atlas/references/`
  - `human_matrisome_v2.csv`
  - `mouse_matrisome_v2.csv`

### Output Data
- Study results: `/Users/Kravtsovd/projects/ecm-atlas/XX_Study_YYYY_paper_to_csv/`
  - `Study_YYYY_wide_format.csv`
  - `Study_YYYY_metadata.json`
  - `Study_YYYY_annotation_report.md`
  - `Study_YYYY_validation_log.txt`

- Unified database: `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/`
  - `ECM_Atlas_Unified.csv`
  - `unified_metadata.json`
  - `zscore_metadata_*.json`
  - `backups/ECM_Atlas_Unified_*.csv`

---

## Key Principles

### ✅ Always Do
1. **Keep compartments separate** in Tissue column (e.g., "Kidney_Glomerular" not "Kidney")
2. **Filter to ECM proteins only** before wide-format conversion (Match_Confidence > 0)
3. **Preserve NaN values** throughout pipeline (missing = not detected)
4. **Create backups** before modifying unified CSV
5. **Validate schema** when merging studies
6. **Document everything** in metadata JSONs

### ❌ Never Do
1. Merge compartments into generic tissue names
2. Impute or fill missing values
3. Remove proteins with NaN abundances (only null IDs)
4. Modify existing studies when adding new one
5. Calculate z-scores before adding to unified CSV

### ⚠️ Missing Values
- 50-80% null abundances are **NORMAL** for LFQ proteomics
- NaN = protein not detected in sample (biological reality)
- Use `mean(skipna=True)` when aggregating (pandas default)
- NaN preserved: Abundance NaN → Zscore NaN

---

## Expected Results

### PHASE 1 Output (Study Processing)
```
XX_Study_YYYY_paper_to_csv/
├── Study_YYYY_wide_format.csv          (PRIMARY OUTPUT)
├── Study_YYYY_metadata.json
├── Study_YYYY_annotation_report.md
└── Study_YYYY_validation_log.txt
```

**Wide-format schema (15 core columns):**
- Protein_ID, Protein_Name, Gene_Symbol
- Canonical_Gene_Symbol, Matrisome_Category, Matrisome_Division
- Tissue, Tissue_Compartment, Species
- Abundance_Young, Abundance_Old
- Method, Study_ID
- Match_Level, Match_Confidence

### PHASE 2 Output (Merge)
```
08_merged_ecm_dataset/
├── ECM_Atlas_Unified.csv               (UPDATED)
├── unified_metadata.json               (UPDATED)
└── backups/
    └── ECM_Atlas_Unified_TIMESTAMP.csv
```

### PHASE 3 Output (Z-Scores)
```
08_merged_ecm_dataset/
├── ECM_Atlas_Unified.csv               (UPDATED with z-scores)
├── zscore_metadata_Study_YYYY.json     (NEW)
└── backups/
    └── ECM_Atlas_Unified_TIMESTAMP.csv
```

**Z-score columns added:**
- Zscore_Young
- Zscore_Old
- Zscore_Delta

---

## Validation Checklist

### After PHASE 1 (Study Processing)
- [ ] CSV file created with correct schema
- [ ] ECM proteins filtered (Match_Confidence > 0)
- [ ] Compartments kept separate in Tissue column
- [ ] Known ECM markers found (COL1A1, FN1, etc.)
- [ ] NaN abundances preserved (not removed)
- [ ] Metadata JSON generated

### After PHASE 2 (Merge)
- [ ] Unified CSV row count increased by study size
- [ ] No duplicate rows (check with Study_ID + Protein_ID + Tissue)
- [ ] Schema consistent across studies
- [ ] Backup created
- [ ] unified_metadata.json updated

### After PHASE 3 (Z-Scores)
- [ ] Z-score columns added for new study
- [ ] Other studies' z-scores unchanged
- [ ] Z-score validation passed (mean ≈ 0, std ≈ 1)
- [ ] NaN abundances → NaN z-scores
- [ ] Metadata JSON created
- [ ] Backup created

---

## Troubleshooting

### Low annotation coverage (<20%)
**Diagnosis:** Expected for whole-proteome studies
**Action:** Normal - most proteins are non-ECM

### Schema mismatch when merging
**Diagnosis:** Column differences between studies
**Action:** Add missing columns as NaN in both dataframes

### High % missing abundances (>80%)
**Diagnosis:** Normal for spatial LFQ proteomics
**Action:** Proceed - do NOT impute or remove

### Compartments accidentally merged
**Diagnosis:** Tissue column used generic name
**Action:** Fix in PHASE 1 - use "Organ_Compartment" format

### Z-score validation WARNING
**Diagnosis:** Mean/std deviate from (0, 1)
**Possible causes:** Very small group (<10 proteins), all identical values
**Action:** Check group size - may be acceptable

---

## Examples

### Example 1: Randles 2021 (Human Kidney)
- **Species:** Homo sapiens
- **Tissue:** Kidney
- **Compartments:** Glomerular, Tubulointerstitial (2 separate)
- **Age groups:** Young (15,29,37) vs Old (61,67,69)
- **Method:** Label-free LC-MS/MS (Progenesis + Mascot)
- **Proteins:** 2,610 total → 229 ECM proteins
- **Groupby:** `['Tissue']`
- **Result:** 2 z-score groups

### Example 2: Tam 2020 (Human Intervertebral Disc)
- **Species:** Homo sapiens
- **Tissue:** Intervertebral disc
- **Compartments:** NP, IAF, OAF (3 separate)
- **Age groups:** Young (16) vs Aged (59)
- **Method:** Label-free LC-MS/MS (MaxQuant LFQ)
- **Proteins:** 3,101 total → 426 ECM proteins
- **Groupby:** `['Tissue']`
- **Result:** 3 z-score groups

---

## Support

### Documentation
- Full pipeline: Read documents 01 and 02
- Z-score function: See `universal_zscore_function.py` docstring
- Examples: Check `07_Tam_2020_paper_to_csv/` and `05_Randles_paper_to_csv/`

### Reference Materials
- Original task specs: `/Users/Kravtsovd/projects/ecm-atlas/01_TASK_DATA_STANDARDIZATION.md`
- Annotation guidelines: `/Users/Kravtsovd/projects/ecm-atlas/02_TASK_PROTEIN_ANNOTATION_GUIDELINES.md`
- Pipeline overview: `/Users/Kravtsovd/projects/ecm-atlas/00_ECM_ATLAS_PIPELINE_OVERVIEW.md`

---

## Version History

- **2025-10-13:** Initial documentation created
  - Document 01: LFQ Dataset Normalization and Merge
  - Document 02: Z-Score Calculation Universal Function
  - Python function: universal_zscore_function.py

---

**Last updated:** 2025-10-13
**Maintainer:** Daniel Kravtsov (daniel@improvado.io)
