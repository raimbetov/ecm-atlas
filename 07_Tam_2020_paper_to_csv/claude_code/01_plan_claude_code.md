# Tam 2020 Dataset Conversion - Execution Plan

**Task:** Convert Tam et al. 2020 intervertebral disc (IVD) aging proteomics dataset into standardized CSV format

**Created:** 2025-10-12

## Overview

This plan implements a 7-phase approach to convert spatially-resolved proteomics data from Tam et al. 2020 into standardized CSV format with compartment-specific z-score normalization.

### Key Requirements
- **Source:** `data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx`
- **Target format:** Wide-format CSV with separate Abundance_Young and Abundance_Old columns
- **Tissue compartments:** NP (Nucleus Pulposus), IAF (Inner Annulus Fibrosus), OAF (Outer Annulus Fibrosus)
- **Age groups:** Young (16yr) vs Old (59yr)
- **Spatial resolution:** 66 profiles across 3 disc levels (L3/4, L4/5, L5/S1)
- **Annotation target:** ≥90% coverage using human matrisome reference

### Critical Design Decisions
1. **Compartment separation:** Keep NP, IAF, OAF as separate tissues (not combined)
2. **Z-score normalization:** Calculate separately per compartment (3 separate files)
3. **Spatial aggregation:** Average LFQ intensities across spatial profiles within each compartment-age combination
4. **Wide-format output:** Final deliverable with Abundance_Young and Abundance_Old columns

## Execution Phases

### Phase 1: File Reconnaissance (Est: 5 min)
**Objective:** Validate data availability and structure before parsing

**Tasks:**
- [ ] Verify Excel file exists at specified path
- [ ] Inspect sheet names and structure
- [ ] Validate dimensions (3,158 proteins × 80 columns, 66 profiles)
- [ ] Check column names (Protein identifiers, LFQ intensity columns)
- [ ] Test metadata join alignment between Raw data and Sample information sheets
- [ ] Check for critical data quality issues (null protein IDs, empty columns)

**Success Criteria:**
- File accessible
- Both sheets present (Raw data, Sample information)
- Profile names align between sheets
- <10% missing protein identifiers

**Abort Criteria:**
- File missing
- Wrong sheet structure
- >50% missing identifiers
- Profile count mismatch

---

### Phase 2: Data Parsing (Est: 15 min)
**Objective:** Extract and reshape data from Excel to long-format table

**Tasks:**
- [ ] Load Excel sheets (Raw data + Sample information)
- [ ] Extract protein identifiers (Majority protein IDs, Protein names, Gene names)
- [ ] Extract LFQ intensity columns (66 columns)
- [ ] Reshape wide matrix to long format (melt operation)
- [ ] Join with sample metadata on profile names
- [ ] Parse age groups (young → 16yr, old → 59yr)
- [ ] Parse tissue compartments (NP, IAF, OAF)
- [ ] Extract spatial metadata (disc level, anatomical direction, distance)
- [ ] Create Sample_ID and Tissue columns

**Expected Output:**
- Long-format dataframe: ~208,428 rows (3,158 proteins × 66 profiles)
- Columns: Protein identifiers, Abundance, Age, Age_Bin, Tissue_Compartment, spatial metadata

**Validation:**
- No rows lost during metadata join
- All profiles matched
- Age and compartment distributions correct

---

### Phase 3: Schema Standardization (Est: 10 min)
**Objective:** Map columns to unified 14-column format

**Tasks:**
- [ ] Map protein identifiers (Protein_ID, Protein_Name, Gene_Symbol)
- [ ] Create Tissue column combining organ and compartment (format: "Intervertebral_disc_NP")
- [ ] Add Tissue_Compartment column for explicit compartment tracking
- [ ] Set Species to "Homo sapiens"
- [ ] Map abundance values with units
- [ ] Add Method metadata
- [ ] Set Study_ID to "Tam_2020"
- [ ] Add spatial metadata columns
- [ ] Create Parsing_Notes with profile context
- [ ] Remove rows with null Protein_ID or Abundance
- [ ] Validate schema compliance

**Critical Validation:**
- Tissue column has 3 distinct values (Intervertebral_disc_NP, Intervertebral_disc_IAF, Intervertebral_disc_OAF)
- No nulls in required columns
- Correct data types

---

### Phase 4: Protein Annotation (Est: 15 min)
**Objective:** Harmonize protein identifiers using human matrisome reference

**Tasks:**
- [ ] Load human matrisome reference (`references/human_matrisome_v2.csv`)
- [ ] Build lookup dictionaries (gene symbol, UniProt ID, synonyms)
- [ ] Apply hierarchical matching:
  - Level 1: Exact gene symbol match (confidence 100%)
  - Level 2: UniProt ID match (confidence 95%)
  - Level 3: Synonym match (confidence 80%)
  - Level 4: Unmatched (requires review)
- [ ] Add annotation columns (Canonical_Gene_Symbol, Matrisome_Category, Matrisome_Division, Match_Level, Match_Confidence)
- [ ] Calculate coverage statistics
- [ ] Validate known markers (COL1A1, COL2A1, FN1, ACAN)
- [ ] Generate annotation report

**Success Criteria:**
- Coverage ≥90%
- Known markers present and correctly categorized

**Warning (not failure):**
- Coverage <90% → document and flag for review

---

### Phase 5: Wide-Format Conversion (Est: 10 min)
**Objective:** Aggregate spatial profiles and convert to wide format

**Tasks:**
- [ ] Group by protein and compartment
- [ ] Calculate mean LFQ intensity for Young profiles (per compartment)
- [ ] Calculate mean LFQ intensity for Old profiles (per compartment)
- [ ] Count number of profiles per group (N_Profiles_Young, N_Profiles_Old)
- [ ] Validate row count (~9,474 rows = 3,158 proteins × 3 compartments)
- [ ] Validate compartment distribution
- [ ] Export `Tam_2020_wide_format.csv`

**Expected Output:**
- Wide-format CSV: ~9,474 rows × 15 columns
- Columns: Protein identifiers, Tissue, Abundance_Young, Abundance_Old, annotations, method metadata

**Validation:**
- Each compartment has ~3,158 proteins
- No data loss during aggregation
- All compartments present

---

### Phase 6: Z-Score Normalization (Est: 20 min)
**Objective:** Calculate compartment-specific z-scores for statistical comparison

**Tasks:**
- [ ] Split wide-format data by compartment (NP, IAF, OAF)
- [ ] For each compartment:
  - Check skewness of Abundance_Young and Abundance_Old distributions
  - Apply log2(x+1) transformation if skewness > 1
  - Calculate z-scores for Young (within compartment)
  - Calculate z-scores for Old (within compartment)
  - Calculate z-score delta (Old - Young)
  - Validate normalization (mean ≈ 0, std ≈ 1)
- [ ] Export 3 separate CSV files:
  - `Tam_2020_NP_zscore.csv`
  - `Tam_2020_IAF_zscore.csv`
  - `Tam_2020_OAF_zscore.csv`
- [ ] Validate z-score parameters across all compartments

**Critical Validation:**
- Z-score mean within ±0.01 of 0
- Z-score std within ±0.01 of 1
- All 3 compartment files created

---

### Phase 7: Quality Validation and Export (Est: 10 min)
**Objective:** Verify completeness and generate documentation

**Tasks:**
- [ ] Run Tier 1 validation (6 critical checks)
- [ ] Run Tier 2 validation (6 quality checks)
- [ ] Generate annotation report (Markdown)
- [ ] Generate metadata JSON with:
  - Dataset information
  - Parsing results
  - Z-score normalization parameters
  - File list
- [ ] Create validation summary log
- [ ] Verify all output files created
- [ ] Calculate final score (17/17 criteria)

**Tier 1 Critical Checks:**
1. File parsing successful
2. Row count reasonable (long ≥100K, wide ≥9K)
3. Zero null critical fields
4. Age bins correct (16yr, 59yr)
5. Compartments separate (3 distinct Tissue values)
6. Spatial metadata preserved

**Tier 2 Quality Checks:**
7. Annotation coverage ≥90%
8. Known markers present
9. Species consistency
10. Schema compliance
11. Compartment validation (~3,158 proteins each)
12. Z-score validation (all compartments)

**Tier 3 Documentation:**
13. Wide-format CSV created
14. 3 z-score CSVs created
15. Metadata JSON created
16. Annotation report created
17. Validation log created

**Final Status:**
- PASS: 17/17 criteria met
- FAIL: <17 criteria met

---

## Output Files

### Required Deliverables
1. `Tam_2020_wide_format.csv` - Wide-format data with Abundance_Young and Abundance_Old
2. `Tam_2020_NP_zscore.csv` - NP compartment with z-scores
3. `Tam_2020_IAF_zscore.csv` - IAF compartment with z-scores
4. `Tam_2020_OAF_zscore.csv` - OAF compartment with z-scores
5. `Tam_2020_metadata.json` - Dataset metadata and parsing parameters
6. `Tam_2020_annotation_report.md` - Annotation coverage and validation
7. `Tam_2020_validation_log.txt` - Validation check results

### Optional Diagnostics
- `Tam_2020_low_coverage_diagnostic.json` (if coverage <90%)
- `progress_log.md` (phase-by-phase execution log)

---

## Execution Notes

### Working Directory
`/Users/Kravtsovd/projects/ecm-atlas/07_Tam_2020_paper_to_csv`

### Key Data Paths
- Input: `../data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx`
- Reference: `../references/human_matrisome_v2.csv`
- Output: Current directory

### Error Handling Strategy
1. **Annotation coverage <90%:** Continue but flag as REVIEW status
2. **Spatial profile mismatch >5:** Abort with clear error message
3. **Missing critical fields:** Remove rows and document
4. **Z-score validation failure:** Investigate skewness and transformation

### Time Estimate
- Total: ~85 minutes (1.5 hours)
- Critical path: Parsing → Annotation → Z-score normalization

---

## References
- Task specification: `00_TASK_TAM_2020_CSV_CONVERSION.md`
- Source analysis: `../04_compilation_of_papers/10_Tam_2020_comprehensive_analysis.md`
- Schema definition: `../01_TASK_DATA_STANDARDIZATION.md`
- Annotation guidelines: `../02_TASK_PROTEIN_ANNOTATION_GUIDELINES.md`
- Z-score methodology: `../06_Randles_z_score_by_tissue_compartment/00_TASK_Z_SCORE_NORMALIZATION.md`
