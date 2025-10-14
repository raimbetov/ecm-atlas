# Task: ECM Atlas Dataset Parsing - Version 2 (Knowledge-First Approach)

**🔴 CRITICAL: NEVER MOCK DATA! All data must come from real files in `data_raw/` directory.**

**Date:** 2025-10-12
**Previous Attempt:** v1 scored 10/21 (48%) - only 3/13 datasets parsed
**Goal:** Parse 11 ECM proteomic datasets into unified CSV with full traceability

---

## Task Overview

Parse 11 proteomic datasets (2017-2023) from `data_raw/` into unified 12-column schema with complete reasoning trail and knowledge base documentation.

**Key Requirements:**
1. **Knowledge-First:** Analyze ALL papers and datasets BEFORE writing any parsing code
2. **Full Traceability:** Every row must have reasoning for age/abundance values
3. **11 Datasets:** Excel/TSV only (exclude Lofaro PDF, McCabe DOCX)
4. **Output:** Unified CSV (150k+ rows) with Parsing_Notes column

---

## Datasets to Parse (11 studies)

| # | Study | Files | Format | Est. Rows |
|---|-------|-------|--------|-----------|
| 1 | Angelidis et al. 2019 | MOESM5_ESM.xlsx | Excel | 38,057 |
| 2 | Ariosa-Morejon et al. 2021 | [Excel files] | Excel | ~5,000 |
| 3 | Caldeira et al. 2017 | MOESM2_ESM.xls | Excel | 1,078 |
| 4 | Chmelova et al. 2023 | [Excel files] | Excel | ~8,000 |
| 5 | Dipali et al. 2023 | Candidates.xlsx | Excel | 8,168 |
| 6 | Li et al. 2021 - dermis | Table 1-4.xlsx | Excel | ~3,000 |
| 7 | Li et al. 2021 - pancreas | MOESM4-13.xlsx | Excel | ~15,000 |
| 8 | Ouni et al. 2022 | Supp Table 1-4.xlsx | Excel | ~10,000 |
| 9 | Randles et al. 2021 | index.html | HTML | ~5,000 |
| 10 | Tam et al. 2020 | [Excel files] | Excel | ~12,000 |
| 11 | Tsumagari et al. 2023 | MOESM2-8.xlsx | Excel | ~8,000 |

**Excluded:**
- ❌ Lofaro et al. 2021 (PDF tables only)
- ❌ McCabe et al. 2020 (DOCX only)

**Expected Total:** ~150,000-200,000 rows

---

## Unified CSV Schema (13 columns)

```csv
Protein_ID,Protein_Name,Gene_Symbol,Tissue,Species,Age,Age_Unit,Abundance,Abundance_Unit,Method,Study_ID,Sample_ID,Parsing_Notes
Q9JLC8,Sacsin,Sacs,Lung,Mus musculus,24,months,32.32026,log2_intensity,LC-MS/MS,Angelidis_2019,old_1,"Age=24mo from col name 'old_1' per Methods p.2; Abundance from MOESM5 sheet 'Proteome' col AF; log2 already applied per Methods p.3"
```

**Column Descriptions:**
- **Parsing_Notes:** Reasoning for this row (age source, abundance source, transformations)
- All other columns same as v1 (Protein_ID, Abundance, etc.)

---

## Knowledge Base Structure (MANDATORY)

**CRITICAL:** Create knowledge base BEFORE any parsing code!

```
knowledge_base/
├── 00_dataset_inventory.md           # All Excel/TSV files per study
├── 01_paper_analysis/                # Analysis of 11 papers
│   ├── Angelidis_2019_analysis.md
│   ├── Dipali_2023_analysis.md
│   └── ...                           # 11 files total
├── 02_column_mapping_strategy.md     # How to map columns across studies
├── 03_normalization_strategy.md      # Abundance normalization approach
└── 04_implementation_plan.md         # Detailed parsing plan (AFTER analysis)
```

**Paper Analysis Template:**
```markdown
# [Study] et al. [Year] - Analysis

## 1. Paper Overview
- Title: [exact title]
- PMID: [ID]
- Tissue: [organ]
- Species: [organism]
- Age groups: [definitions from paper]

## 2. Data Files Available
- File: [name]
- Sheet: [sheet name]
- Rows: [count]
- Columns: [list]

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | "Majority protein IDs" | Methods p.3 | UniProt IDs, semicolon-separated, take first |
| Age | Column name "old_1" | Methods p.2 | "old" = 24mo per paper |
| Abundance | Column "old_1" value | Methods p.3 | log2(LFQ) already applied |

## 4. Abundance Calculation
- Formula from paper: [exact quote]
- Unit: [log2_intensity / LFQ / etc]
- Already normalized: [YES/NO]
- Reasoning: [why you believe this]

## 5. Ambiguities & Decisions
- Ambiguity #1: [describe issue]
  - Decision: [what you decided]
  - Reasoning: [why this decision]
```

---

## SUCCESS CRITERIA (25 Total - ALL Required)

**Scoring:** 25/25 required for PASS (100%)

---

### ✅ TIER 0: KNOWLEDGE BASE (4 criteria - COMPLETE FIRST!)

**🔴 CRITICAL: Complete Tier 0 BEFORE writing ANY parsing code!**

**Criterion 0.1: Dataset inventory complete**
- ✅ **Target:** List all Excel/TSV files per study with row counts
- **Evidence:** `knowledge_base/00_dataset_inventory.md` with table of all files
- **Format:**
  ```markdown
  | Study | File | Format | Sheet | Est. Rows | Notes |
  |-------|------|--------|-------|-----------|-------|
  | Angelidis 2019 | MOESM5_ESM.xlsx | Excel | Proteome | 38,057 | Main data |
  ```

**Criterion 0.2: Paper analysis complete (11 papers)**
- ✅ **Target:** Analyze all 11 papers for:
  - Age definitions (exact values + units from paper Methods)
  - Abundance calculation formulas (exact quote from Methods section)
  - Column mappings (which Excel columns → unified schema)
- **Evidence:** 11 files in `knowledge_base/01_paper_analysis/` using template above
- **Pass threshold:** All 11 papers analyzed

**Criterion 0.3: Normalization strategy documented**
- ✅ **Target:** Document how to normalize abundance across studies BEFORE coding
- **Required content:**
  - Which studies use log2 vs raw intensities
  - Whether to apply transformations or preserve as-is
  - How to handle different units (LFQ vs spectral counts)
- **Evidence:** `knowledge_base/03_normalization_strategy.md` with strategy

**Criterion 0.4: Implementation plan approved**
- ✅ **Target:** Detailed parsing plan AFTER complete analysis
- **Required content:**
  - Order of dataset parsing (easiest first)
  - Per-study parsing logic (which columns to extract)
  - Common parsing functions (reusable code structure)
- **Evidence:** `knowledge_base/04_implementation_plan.md` with step-by-step plan
- **CRITICAL:** No code written until this is complete!

---

### ✅ TIER 1: DATA COMPLETENESS (4 criteria - ALL required)

**Criterion 1.1: Parse ALL 11 datasets**
- ✅ **Target:** 11/11 studies parsed successfully (100% coverage)
- ❌ **Failure:** <11 studies parsed (partial solutions not accepted)
- **Evidence:** List of 11 parsed CSV files in `data_processed/` directory
- **Note:** Lofaro (PDF) and McCabe (DOCX) excluded from scope

**Criterion 1.2: Maximum row count achieved**
- ✅ **Target:** ≥150,000 total rows across all datasets (proteins × samples)
- ⚠️ **Warning:** 100,000-150,000 rows (acceptable but investigate missing data)
- ❌ **Failure:** <100,000 rows (indicates major parsing issues)
- **Evidence:** `validation_report.md` showing row counts per study
- **Benchmark:** v1 achieved 47,303 rows from only 3 studies

**Criterion 1.3: Maximum protein coverage**
- ✅ **Target:** ≥12,000 unique proteins across all datasets
- ⚠️ **Warning:** 8,000-12,000 proteins (acceptable)
- ❌ **Failure:** <8,000 proteins (too few for meaningful analysis)
- **Evidence:** Unique Protein_ID count in validation report
- **Benchmark:** v1: 9,350 proteins from 3 studies only

**Criterion 1.4: No mock data**
- ✅ **Required:** All data comes from real files in `data_raw/` directory
- ❌ **Disqualification:** Any mocked, placeholder, or fabricated data
- **Verification:** Spot-check 10 random rows against original Excel files

---

### ✅ TIER 2: DATA QUALITY (6 criteria - ALL required)

**Criterion 2.1: Zero critical nulls**
- ✅ **Target:** 0 null values in required fields (Protein_ID, Abundance, Study_ID, Sample_ID)
- ❌ **Failure:** Any null in critical columns
- **Evidence:** Null count report per column per study

**Criterion 2.2: Protein IDs standardized**
- ✅ **Target:** All Protein_IDs mapped to UniProt OR Gene symbols consistently
- **Format rules:**
  - UniProt: P02452 (6-character alphanumeric)
  - Gene symbol: COL1A1 (uppercase, standard HGNC/MGI nomenclature)
- **Evidence:** Sample 100 random Protein_IDs, verify format compliance

**Criterion 2.3: Age groups correctly identified**
- ✅ **Target:** 100% of samples labeled with correct age group from paper
- **Age mapping examples:**
  - Angelidis 2019: Young=3 months, Old=24 months (from paper Methods section)
  - Caldeira 2017: Foetus / Young / Old (three-point scale)
- **Evidence:** Age mapping table per study with source references from papers

**Criterion 2.4: Abundance units documented with paper citations**
- ✅ **Target:** Every `Abundance_Unit` value has citation to source paper section
- **Required documentation format:**
  ```markdown
  ### Study: Angelidis et al. 2019
  - Abundance_Unit: log2_intensity
  - Source: Paper Methods, paragraph 3: "LFQ intensities were log2-transformed..."
  - File location: 41467_2019_8831_MOESM5_ESM.xlsx, sheet "Proteome", columns old_1...young_4
  - Calculation: log2(sum of peptide intensities per protein)
  ```
- **Evidence:** Citations in paper_analysis files

**Criterion 2.5: Tissue metadata complete**
- ✅ **Target:** All samples have Tissue and Species filled (no nulls)
- **Source hierarchy:**
  1. If in data file columns → extract directly
  2. If in paper title/abstract → use paper-level annotation
  3. If missing → flag for manual review (document in issues log)
- **Evidence:** Tissue/Species distribution table

**Criterion 2.6: Sample IDs preserve biological structure**
- ✅ **Target:** Sample_ID reflects biological vs technical replicates
- **Naming convention:**
  - Biological replicates: `young_1`, `young_2`, `old_1`, `old_2`
  - Technical replicates: `young_1_tech_a`, `young_1_tech_b`
  - Pooled samples: `young_pooled`, `old_pooled`
- **Evidence:** Sample naming logic documented per study

---

### ✅ TIER 3: COLUMN TRACEABILITY (4 criteria - ALL required)

**🔴 CRITICAL: Every column must be traceable to source paper**

**Criterion 3.1: Column mapping documentation**
- ✅ **Required:** For EACH of 12 schema columns, document where data comes from in source paper
- **Template per study:** (included in paper_analysis files)
  ```markdown
  ## Column Mappings
  | Schema Column | Source File | Source Column/Section | Paper Citation | Notes |
  |---------------|-------------|----------------------|----------------|-------|
  | Protein_ID | MOESM5_ESM.xlsx | "Majority protein IDs" | Methods, page 3 | UniProt IDs |
  | Age | Column headers | "old_1" → 24mo | Methods, para 2 | "24-month-old" |
  ```
- **Evidence:** Complete mapping tables in all 11 paper_analysis files

**Criterion 3.2: Paper PDF references**
- ✅ **Required:** All 11 papers cited with PMID or DOI
- **Evidence:** List of papers with identifiers in `00_dataset_inventory.md`
- **Note:** PDFs don't need to be present, just cited correctly

**Criterion 3.3: Abundance calculation formulas documented**
- ✅ **Required:** For each study, extract exact formula for how abundance values were calculated
- **Examples:**
  - Angelidis 2019: "LFQ intensities = sum of peptide intensities, then log2-transformed"
  - Dipali 2023: "DirectDIA output, LFQ normalized by total protein amount"
- **Evidence:** Formulas in paper_analysis files (section 4)

**Criterion 3.4: Ambiguities documented**
- ✅ **Required:** Create ambiguities log in each paper_analysis file
- **Format per ambiguity:**
  ```markdown
  ### Ambiguity #1: Li et al. 2021 | pancreas - Age definition unclear
  - Issue: Paper mentions "young adult" but no specific age stated
  - Data file: Table 1.xlsx does not have age columns
  - Paper search: Methods section says "8-12 weeks" on page 4, paragraph 2
  - Resolution: Used Age=10 weeks (midpoint), Age_Unit=weeks
  - Confidence: Medium (inferred from range)
  ```
- **Evidence:** Ambiguities section in all relevant paper_analysis files

---

### ✅ TIER 4: PROGRESS TRACKING (3 criteria - ALL required)

**🔴 CRITICAL: Agent must document progress in real-time during execution**

**Criterion 4.1: Task decomposition before starting**
- ✅ **Required:** Create detailed task breakdown BEFORE writing any parsing code
- **Format:** `01_task_decomposition_[agent].md` with:
  ```markdown
  ## Task Breakdown

  ### Phase 0: Knowledge Base (Est: 45 min)
  - [ ] Create dataset inventory (10 min)
  - [ ] Analyze all 11 papers (30 min)
  - [ ] Document normalization strategy (5 min)
  - [ ] Write implementation plan (5 min)
  - Success metric: Complete knowledge_base/ folder

  ### Phase 1: Per-Study Parsing (Est: 2 hours)
  - [ ] Study 1: Angelidis 2019 (Est: 10 min)
  - [ ] Study 2: Ariosa-Morejon 2021 (Est: 8 min)
  ...
  - [ ] Study 11: Tsumagari 2023 (Est: 10 min)

  ### Phase 2: Validation & Documentation (Est: 30 min)
  - [ ] Generate validation report
  - [ ] Verify Parsing_Notes completeness
  - [ ] Create unified CSV

  Total estimated time: 3h 15min
  ```
- **Evidence:** Task decomposition file created within first 5 minutes

**Criterion 4.2: Real-time progress logging**
- ✅ **Required:** Update `progress_log_[agent].md` after completing each study
- **Update frequency:** After every study parsed (11 updates minimum)
- **Format per update:**
  ```markdown
  ## [Timestamp] Study 1 Complete: Angelidis 2019
  - Status: ✅ Success
  - Rows parsed: 38,057
  - Unique proteins: 5,189
  - Time taken: 12 minutes
  - Issues encountered: None
  - Next: Starting Study 2 (Ariosa-Morejon 2021)
  ```
- **Evidence:** Progress log with 11+ timestamped updates

**Criterion 4.3: Intermediate validation checkpoints**
- ✅ **Required:** Run validation after every 3 studies parsed
- **Checkpoints:** After studies 3, 6, 9, 11
- **Validation checks per checkpoint:**
  - Row count so far vs. expected
  - Null count in critical columns
  - Protein ID format compliance
  - Sample spot-check against original files
- **Evidence:** Validation reports at each checkpoint in `checkpoints/` folder

---

### ✅ TIER 5: DELIVERABLES (4 criteria - ALL required)

**Criterion 5.1: Complete file set**
- ✅ **Required files in `data_processed/`:**
  ```
  data_processed/
  ├── Angelidis_2019_parsed.csv
  ├── Ariosa-Morejon_2021_parsed.csv
  ├── ...                              # 11 individual CSVs
  ├── ecm_atlas_unified.csv            # All studies combined
  ├── metadata.json                    # Study metadata
  └── validation_report.md             # Statistics and QC
  ```
- **Evidence:** All 13 files present with non-zero size

**Criterion 5.2: Validation report completeness**
- ✅ **Required sections in `validation_report.md`:**
  1. Executive summary (total rows, proteins, studies)
  2. Per-study statistics table
  3. Null value analysis (per column, per study)
  4. Protein ID format compliance check
  5. Age group coverage
  6. Tissue/Species distribution
  7. Sample count per study
  8. Known ECM markers presence check (COL1A1, FN1, LAMA2, etc.)
- **Evidence:** Validation report with all 8 sections

**Criterion 5.3: Parsing reasoning logged (NEW)**
- ✅ **Required:** Every row in CSVs has Parsing_Notes column with reasoning
- **Content per row:**
  - Age source (which column/paper section)
  - Abundance source (which column/calculation)
  - Any transformations applied
- **Example:** `"Age=24mo from col 'old_1' per Methods p.2; Abundance from sheet 'Proteome' col AF; log2 already applied"`
- **Evidence:** Spot-check 20 random rows, verify Parsing_Notes completeness

**Criterion 5.4: Reproducible code**
- ✅ **Required:** Parsing code that can be re-run to reproduce exact same output
- **Code requirements:**
  - Config-driven (adding new studies only requires config update, not code changes)
  - Error handling (graceful failures with clear error messages)
  - Logging (debug logs showing file reads, column mappings, row counts)
  - Comments (explain non-obvious logic, especially format-specific hacks)
- **Evidence:** Successfully re-run code from scratch, verify output matches

---

## 📊 SCORING RUBRIC

| Tier | Criteria | Weight | Pass Threshold |
|------|----------|--------|----------------|
| **Tier 0: Knowledge Base** | 4 | 20% | 4/4 required |
| **Tier 1: Completeness** | 4 | 25% | 4/4 required |
| **Tier 2: Quality** | 6 | 25% | 6/6 required |
| **Tier 3: Traceability** | 4 | 15% | 4/4 required |
| **Tier 4: Progress** | 3 | 10% | 3/3 required |
| **Tier 5: Deliverables** | 4 | 5% | 4/4 required |
| **TOTAL** | **25** | **100%** | **25/25 required** |

**Final Grade:**
- ✅ **Pass:** 25/25 criteria met (100%)
- ❌ **Fail:** <25 criteria met (any missing criterion = task failure)

**Previous Attempt (v1) Results:**
- Claude Code CLI: 10/21 (48%) - ❌ Failed (only 3/13 studies, no knowledge base)
- Codex CLI: 2/21 (10%) - ❌ Failed (reconnaissance only)
- Gemini CLI: 3/21 (14%) - ❌ Failed (skeleton code only)

---

## Agents Artifact Requirements

### Required Files per Agent:

**Planning:**
- `01_task_decomposition_[agent].md` - Task breakdown with time estimates
- `01_plan_[agent].md` - Approach and strategy

**Progress:**
- `progress_log_[agent].md` - Real-time updates after each study

**Results:**
- `90_results_[agent].md` - Self-evaluation with evidence for all 25 criteria

**Deliverables:**
- `knowledge_base/` folder with all analysis files
- `data_processed/` folder with CSVs and docs
- `parse_datasets.py` (or equivalent) - Parsing code

### Self-Evaluation Format:

```markdown
# Self-Evaluation: [Agent Name]

## TIER 0: KNOWLEDGE BASE

### Criterion 0.1: Dataset inventory complete
**Status:** ✅/❌/⚠️
**Evidence:** `knowledge_base/00_dataset_inventory.md` created with 11 studies
**Details:** Listed all Excel/TSV files with row counts
**Pass:** YES/NO

### Criterion 0.2: Paper analysis complete
**Status:** ✅/❌/⚠️
**Evidence:** 11 files in `knowledge_base/01_paper_analysis/`
**Details:** Analyzed age definitions, formulas, column mappings for all 11 papers
**Pass:** YES/NO

[Continue for all 25 criteria...]

## FINAL SCORE: X/25 criteria met
## GRADE: ✅ PASS / ❌ FAIL
```

---

## Technical Notes

**File Formats:**
- **Excel:** Use `openpyxl` for .xlsx, `xlrd` for .xls
- **TSV:** Use `pandas.read_csv(sep='\t')`
- **HTML:** Use `pandas.read_html()` or `BeautifulSoup`

**Python Environment:**
```python
# Required packages
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
xlrd>=2.0.1
```

**Data Location:**
- Repository root: `/Users/Kravtsovd/projects/ecm-atlas/`
- Raw data: `data_raw/[Study]/*.xlsx`
- Output: `data_processed/*.csv`

---

**CRITICAL REMINDERS:**
1. ✅ Complete Tier 0 (Knowledge Base) BEFORE any code
2. ✅ Every row must have Parsing_Notes with reasoning
3. ✅ 11 studies only (no PDF/DOCX parsing)
4. ✅ Real-time progress logging after each study
5. ❌ NO MOCK DATA - all from real files
