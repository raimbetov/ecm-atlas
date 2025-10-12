# ECM Atlas Dataset Parsing - Analysis Plan (Claude Code)

**Date:** 2025-10-12
**Agent:** Claude Code CLI
**Task:** Version 2 - Knowledge-First Approach
**Goal:** Parse 11 ECM proteomic datasets with complete traceability

---

## Executive Summary

This document provides a comprehensive analysis and implementation plan for parsing 11 ECM proteomic datasets into a unified 12-column schema. This is Version 2 of the task, incorporating lessons learned from Version 1 where Claude Code achieved 10/21 criteria (48%) by parsing only 3/13 datasets.

**Key Improvements for V2:**
- ✅ Knowledge-First approach: Complete analysis BEFORE coding
- ✅ Full traceability: Every data point documented with source
- ✅ Parsing_Notes column: Reasoning for each row
- ✅ 11 datasets only: Exclude PDF/DOCX (Lofaro, McCabe)
- ✅ 25 criteria scoring rubric (up from 21)

---

## Task Context

### Previous Attempt (V1) Summary

**Claude Code V1 Results: 10/21 (48%) - FAILED**
- ✅ Parsed: 3/13 datasets (Angelidis 2019, Dipali 2023, Caldeira 2017)
- ✅ Output: 47,303 rows, 9,350 unique proteins
- ✅ Code quality: Production-ready OOP architecture
- ❌ Missing: 10 datasets not parsed
- ❌ No knowledge base documentation
- ❌ No parsing reasoning trail

**Key Learnings:**
1. Config-driven architecture was correct approach
2. Code quality was excellent but incomplete coverage
3. Lacked systematic paper analysis before coding
4. No traceability for age/abundance interpretations

---

## Datasets Overview

### ✅ Parseable Datasets (11 studies)

| # | Study | File | Format | Est. Rows | Complexity |
|---|-------|------|--------|-----------|------------|
| 1 | **Angelidis 2019** | MOESM5_ESM.xlsx | Excel | 38,057 | Moderate (multi-sample) |
| 2 | **Ariosa-Morejon 2021** | elife-66635-fig2-data1-v1.xlsx | Excel | ~5,000 | Moderate (multi-sheet, ratio-based) |
| 3 | **Caldeira 2017** | MOESM2_ESM.xls | Excel | 1,078 | Simple (3 age groups) |
| 4 | **Chmelova 2023** | Data Sheet 1.XLSX | Excel | ~8,000 | Complex (transposed) |
| 5 | **Dipali 2023** | Candidates.xlsx | Excel | 8,168 | Simple (2 age groups) |
| 6 | **Li 2021 - dermis** | Table 2.xlsx | Excel | ~3,000 | Moderate (4 age groups, skip rows) |
| 7 | **Li 2021 - pancreas** | MOESM6_ESM.xlsx | Excel | ~15,000 | Moderate (4 age groups F/J/Y/O) |
| 8 | **Ouni 2022** | Supp Table 1-4.xlsx | Excel | ~10,000 | Unknown (needs investigation) |
| 9 | **Randles 2021** | File027.xlsx or index.html | Excel/HTML | ~5,000 | Moderate (2 tissues, age in cols) |
| 10 | **Tam 2020** | elife-64940-supp1-v3.xlsx | Excel | ~12,000 | Complex (hierarchical columns) |
| 11 | **Tsumagari 2023** | MOESM3_ESM.xlsx | Excel | ~8,000 | Simple (3 age groups) |

**Expected Total: 150,000-200,000 rows**

### ❌ Excluded Datasets (2 studies)

- **Lofaro 2021**: PDF tables only (no Excel)
- **McCabe 2020**: DOCX only (no Excel)

---

## Critical Issues Identified

### 1. Data File Inventory Gap

**Issue:** Task specifies 11 datasets but only 7 were analyzed in existing docs:
- ✅ Documented: Ariosa-Morejon, Chmelova, Li dermis, Li pancreas, Randles, Tam, Tsumagari
- ❓ **Missing analysis:** Angelidis, Caldeira, Dipali, Ouni

**Action Required:** Investigate all 11 studies, especially:
- Angelidis 2019: Which file? (10 xlsx files available)
- Ouni 2022: Is this really proteomics data? (PARSER_CONFIG says skip for "literature mining")
- Caldeira 2017: Need to verify file and structure

### 2. Conflicting Information

**PARSER_CONFIG_SUMMARY.md** says:
- "Total datasets: 10, Parseable: 7, Skip: 3"
- Marks Ouni 2022 as "SKIP - Literature mining data"

**Task v2** says:
- "11 datasets to parse"
- Lists Ouni 2022 as parseable (#8)

**Resolution Needed:** Clarify if Ouni 2022 should be parsed or excluded

### 3. Paper Analysis Requirements

Task requires analysis of 11 papers, but existing docs don't have:
- PMID/DOI citations for all studies
- Exact age definitions from Methods sections
- Abundance calculation formulas (exact quotes)
- Paper section references for column mappings

---

## Data Standardization Issues

### Age Encoding Patterns (9 different methods)

| Study | Age Encoding Method | Example | Extraction Logic |
|-------|-------------------|---------|------------------|
| Angelidis 2019 | Column name (old/young) | `old_1`, `young_3` | Direct mapping: old=24mo, young=3mo |
| Ariosa-Morejon 2021 | Isotope ratio (H/L) | `Ratio H/L A1` | H=Heavy=Old, L=Light=Young |
| Caldeira 2017 | Group names | Foetus/Young/Old | 3-point categorical scale |
| Chmelova 2023 | Sample name prefix | `X3m_ctrl_A` | Extract `3m` or `18m` from name |
| Dipali 2023 | Column names | Old/Young | 2-point categorical |
| Li dermis 2021 | Sample name prefix | `Toddler-Sample1` | 4 groups: Toddler/Teenager/Adult/Elderly |
| Li pancreas 2021 | Column prefix code | `F_7`, `J_22`, `Y_31`, `O_54` | F=Fetal, J=Juvenile(wks), Y=Young(yrs), O=Old(yrs) |
| Randles 2021 | Numeric in col name | `G15`, `T67` | Number = patient age in years (15-69) |
| Tam 2020 | Word in col name | `LFQ intensity...old...` | "old" vs "Young" (note capitalization) |
| Tsumagari 2023 | Age with unit in col | `Cx_3mo_1` | Extract `3mo`, `15mo`, `24mo` |

**Normalization Challenge:** How to map diverse age representations to unified Age + Age_Unit columns?

### Abundance Unit Complexity

| Study | Unit | Transformation | Already Normalized? | Citation Needed |
|-------|------|----------------|---------------------|-----------------|
| Angelidis 2019 | log2_intensity | log2(LFQ) | YES | Methods p.3 (need exact quote) |
| Ariosa-Morejon 2021 | iBAQ ratio | H/L isotope ratio | YES | Methods (need quote) |
| Caldeira 2017 | normalized_ratio | Unknown | Unknown | Need paper |
| Chmelova 2023 | log2 | log2-transformed | YES | Need paper |
| Dipali 2023 | LFQ | DirectDIA output | ? | Need paper |
| Li dermis 2021 | log2_FOT | log2(Fraction of Total) | YES | Need paper |
| Li pancreas 2021 | Unknown | ? | ? | Need paper |
| Randles 2021 | Hi-N normalized | ? | YES | Need paper |
| Tam 2020 | LFQ intensity | ? | ? | Need paper |
| Tsumagari 2023 | Unknown | ? | ? | Need paper |

**Normalization Strategy Decision:** Preserve as-is or transform to common scale?

---

## Unified Schema (13 columns)

```csv
Protein_ID,Protein_Name,Gene_Symbol,Tissue,Species,Age,Age_Unit,Abundance,Abundance_Unit,Method,Study_ID,Sample_ID,Parsing_Notes
```

### NEW: Parsing_Notes Column

**Purpose:** Document reasoning for each row's values

**Format:** Concise string with 3 elements:
1. Age source and reasoning
2. Abundance source and location
3. Any transformations applied

**Example:**
```
"Age=24mo from col name 'old_1' per Methods p.2; Abundance from MOESM5 sheet 'Proteome' col AF; log2 already applied per Methods p.3"
```

**Implementation:** Generate programmatically from config + paper citations

---

## Implementation Strategy

### Phase 0: Knowledge Base (MANDATORY - Complete FIRST!)

**Tier 0 Criteria (4 criteria - ALL required before coding):**

#### 0.1 Dataset Inventory
- **Task:** List all Excel/TSV files per study with row counts
- **Deliverable:** `knowledge_base/00_dataset_inventory.md`
- **Content:**
  - Table with: Study | File | Format | Sheet | Est. Rows | Notes
  - Verify all 11 datasets have parseable files
  - Resolve Angelidis/Ouni ambiguities

#### 0.2 Paper Analysis (11 papers)
- **Task:** Analyze all 11 papers using template
- **Deliverable:** 11 files in `knowledge_base/01_paper_analysis/`
- **Template sections:**
  1. Paper Overview (PMID, tissue, species, age groups)
  2. Data Files Available (sheets, rows, columns)
  3. Column Mapping to Schema (table with reasoning)
  4. Abundance Calculation (exact formula from paper)
  5. Ambiguities & Decisions (issues and resolutions)

**Critical:** Extract exact quotes from Methods sections for:
- Age definitions (e.g., "24-month-old mice" from Methods p.2)
- Abundance formulas (e.g., "LFQ intensities were log2-transformed..." from Methods p.3)

#### 0.3 Normalization Strategy
- **Task:** Document abundance normalization approach
- **Deliverable:** `knowledge_base/03_normalization_strategy.md`
- **Decisions:**
  - Which studies use log2 vs raw intensities
  - Whether to apply transformations or preserve as-is
  - How to handle different units (LFQ vs spectral counts)

#### 0.4 Implementation Plan
- **Task:** Detailed parsing plan AFTER complete analysis
- **Deliverable:** `knowledge_base/04_implementation_plan.md`
- **Content:**
  - Order of dataset parsing (easiest first)
  - Per-study parsing logic (which columns to extract)
  - Common parsing functions (reusable code structure)
- **CRITICAL:** No code written until this is complete!

### Phase 1: Per-Study Parsing

**Parse 11 datasets in order of complexity:**

**Easy (3 studies - start here):**
1. Caldeira 2017 (simple 3-group)
2. Dipali 2023 (simple 2-group)
3. Tsumagari 2023 (standard format)

**Moderate (5 studies):**
4. Angelidis 2019 (multi-sample, already done in v1)
5. Ariosa-Morejon 2021 (multi-sheet, ratio-based)
6. Li dermis 2021 (skip rows, 4 age groups)
7. Li pancreas 2021 (age prefix codes)
8. Randles 2021 (dual tissue, age in numbers)

**Complex (2 studies):**
9. Tam 2020 (hierarchical columns: disc × age × region × tissue)
10. Chmelova 2023 (TRANSPOSED matrix)

**Unknown (1 study):**
11. Ouni 2022 (need to investigate if parseable)

### Phase 2: Validation & Documentation

**Checkpoints:** After studies 3, 6, 9, 11
**Per checkpoint:**
- Row count vs. expected
- Null count in critical columns
- Protein ID format compliance
- Sample spot-check against original files

**Final deliverables:**
- `ecm_atlas_unified.csv` (all studies combined)
- `validation_report.md` (statistics and QC)
- `metadata.json` (study metadata)
- Reproducible parsing code

---

## Success Criteria Checklist (25 total)

### ✅ TIER 0: KNOWLEDGE BASE (4 criteria)

- [ ] 0.1: Dataset inventory complete (00_dataset_inventory.md)
- [ ] 0.2: Paper analysis complete (11 files in 01_paper_analysis/)
- [ ] 0.3: Normalization strategy documented (03_normalization_strategy.md)
- [ ] 0.4: Implementation plan approved (04_implementation_plan.md)

### ✅ TIER 1: DATA COMPLETENESS (4 criteria)

- [ ] 1.1: Parse ALL 11 datasets (100% coverage)
- [ ] 1.2: ≥150,000 total rows achieved
- [ ] 1.3: ≥12,000 unique proteins
- [ ] 1.4: No mock data (all from real files)

### ✅ TIER 2: DATA QUALITY (6 criteria)

- [ ] 2.1: Zero critical nulls (Protein_ID, Abundance, Study_ID, Sample_ID)
- [ ] 2.2: Protein IDs standardized (UniProt or Gene symbols)
- [ ] 2.3: Age groups correctly identified (100% accuracy)
- [ ] 2.4: Abundance units documented with paper citations
- [ ] 2.5: Tissue metadata complete (no nulls)
- [ ] 2.6: Sample IDs preserve biological structure

### ✅ TIER 3: COLUMN TRACEABILITY (4 criteria)

- [ ] 3.1: Column mapping documentation (all 12 columns per study)
- [ ] 3.2: Paper PDF references (PMID or DOI for all 11)
- [ ] 3.3: Abundance formulas documented (exact quotes from papers)
- [ ] 3.4: Ambiguities documented (issues log in paper_analysis files)

### ✅ TIER 4: PROGRESS TRACKING (3 criteria)

- [ ] 4.1: Task decomposition before starting
- [ ] 4.2: Real-time progress logging (11+ timestamped updates)
- [ ] 4.3: Intermediate validation checkpoints (after 3, 6, 9, 11)

### ✅ TIER 5: DELIVERABLES (4 criteria)

- [ ] 5.1: Complete file set (11 individual CSVs + unified + metadata + validation)
- [ ] 5.2: Validation report completeness (8 required sections)
- [ ] 5.3: Parsing reasoning logged (Parsing_Notes in every row)
- [ ] 5.4: Reproducible code (config-driven, error handling, logging)

**PASS THRESHOLD: 25/25 criteria (100%)**

---

## Risk Assessment

### High Risk Items

**1. Missing Paper Access**
- **Risk:** Cannot cite exact Methods sections without papers
- **Mitigation:** Use available supplementary files, infer from data structure
- **Impact:** May fail Tier 3 (Traceability) criteria

**2. Ouni 2022 Dataset Ambiguity**
- **Risk:** Unclear if it's proteomics data or literature mining
- **Mitigation:** Investigate files, exclude if not proteomic
- **Impact:** May reduce total to 10 datasets (150k row target still achievable)

**3. Complex Format Parsing (Chmelova, Tam)**
- **Risk:** Transposed matrix and hierarchical columns are error-prone
- **Mitigation:** Parse these last, extra validation
- **Impact:** Could delay completion or introduce errors

### Medium Risk Items

**4. Age Normalization Ambiguity**
- **Risk:** How to standardize 9 different age encoding methods?
- **Mitigation:** Document strategy in normalization_strategy.md
- **Impact:** Affects comparability across studies

**5. Abundance Unit Diversity**
- **Risk:** Different units (LFQ, iBAQ, log2, ratios) - how to preserve/transform?
- **Mitigation:** Preserve as-is, document source clearly
- **Impact:** Affects downstream analysis capability

### Low Risk Items

**6. Code Reusability from V1**
- **Risk:** V1 code may not fit new requirements (Parsing_Notes)
- **Mitigation:** V1 architecture is solid, extend don't rebuild
- **Impact:** Minimal - minor modifications needed

---

## Time Estimates

### Phase 0: Knowledge Base (Est: 45-60 min)

- Dataset inventory: 10 min
- Paper analysis (11 papers × 3 min): 30-40 min
- Normalization strategy: 5 min
- Implementation plan: 5 min

### Phase 1: Per-Study Parsing (Est: 2-3 hours)

- Study 1-3 (easy): 30 min (10 min each)
- Study 4-8 (moderate): 60-90 min (12-15 min each)
- Study 9-10 (complex): 30-40 min (15-20 min each)
- Study 11 (unknown): 15-20 min

### Phase 2: Validation & Documentation (Est: 30 min)

- Validation checkpoints: 15 min
- Final validation report: 10 min
- Unified CSV generation: 5 min

**Total Estimated Time: 3h 45min - 4h 30min**

---

## Key Decisions for Implementation

### Decision 1: Parsing_Notes Generation

**Options:**
A. Hardcode reasoning strings per study in config
B. Generate programmatically from config metadata
C. Hybrid: config provides templates, code fills values

**Recommendation:** Option C (Hybrid)
- Config provides citation templates per study
- Code fills in specific column/sheet names
- Example config:
```python
'Angelidis_2019': {
    'age_reasoning_template': 'Age={age}{unit} from col name {col_name} per Methods p.2',
    'abundance_reasoning_template': 'Abundance from MOESM5 sheet Proteome col {col}; log2 already applied per Methods p.3',
}
```

### Decision 2: Normalization Strategy

**Options:**
A. Convert all to common scale (e.g., log2)
B. Preserve original units, document clearly
C. Provide both raw and normalized

**Recommendation:** Option B (Preserve original)
- Less error-prone (no transformation bugs)
- Maintains fidelity to source papers
- Clear documentation in Abundance_Unit + Parsing_Notes
- Downstream users can normalize if needed

### Decision 3: Ouni 2022 Resolution

**Investigation needed:**
- Check file `Supp Table 1-4.xlsx` structure
- If proteomics → parse (target 11 datasets)
- If literature mining → exclude (target 10 datasets)

**Adjust row count target accordingly:**
- 11 datasets: ≥150,000 rows
- 10 datasets: ≥140,000 rows (acceptable)

### Decision 4: Code Architecture

**Options:**
A. Extend V1 config-driven OOP code
B. Rewrite from scratch with new requirements
C. Hybrid: V1 core + new Parsing_Notes layer

**Recommendation:** Option A (Extend V1)
- V1 architecture is proven and solid
- Add Parsing_Notes generation to existing parsers
- Minimal refactoring needed

---

## Next Steps (Immediate Actions)

### Step 1: Investigate Ambiguous Datasets (15 min)

**Angelidis 2019:**
- List all 10 xlsx files in directory
- Identify which is main proteomics data (likely MOESM5 based on V1)
- Confirm row count and structure

**Ouni 2022:**
- Examine Supp Table files
- Determine if proteomics or literature mining
- Decide include/exclude

### Step 2: Create Knowledge Base Structure (5 min)

```bash
mkdir -p knowledge_base/01_paper_analysis
touch knowledge_base/00_dataset_inventory.md
touch knowledge_base/02_column_mapping_strategy.md
touch knowledge_base/03_normalization_strategy.md
touch knowledge_base/04_implementation_plan.md
```

### Step 3: Start Paper Analysis (40 min)

**Priority order (analyze papers for these studies first):**
1. Studies with missing citations (Angelidis, Caldeira, Dipali, Ouni)
2. Complex studies (Chmelova, Tam)
3. Already-documented studies (verify and complete)

**Per paper, extract:**
- PMID/DOI
- Exact age definitions from Methods
- Exact abundance formulas from Methods
- Column mapping reasoning

### Step 4: Complete Tier 0 Before Any Coding! (60 min total)

**DO NOT WRITE PARSING CODE UNTIL:**
- ✅ All 11 papers analyzed
- ✅ Normalization strategy decided
- ✅ Implementation plan approved
- ✅ All Tier 0 criteria met (4/4)

---

## Success Metrics

### Minimum Viable Success (PASS):
- ✅ 25/25 criteria met (100%)
- ✅ 11 datasets parsed (or 10 if Ouni excluded)
- ✅ ≥150,000 rows (or ≥140,000 if 10 datasets)
- ✅ Complete knowledge base with paper citations
- ✅ Parsing_Notes in every row

### Stretch Goals (Excellence):
- 🎯 >200,000 rows (maximum data extraction)
- 🎯 >15,000 unique proteins
- 🎯 Zero ambiguities (all decisions paper-backed)
- 🎯 <3 hours total execution time

---

## Appendix: V1 vs V2 Comparison

| Aspect | V1 (Failed - 10/21) | V2 (Current Plan) |
|--------|-------------------|-------------------|
| **Approach** | Code-first | Knowledge-first |
| **Documentation** | None | Complete knowledge base |
| **Traceability** | None | Parsing_Notes per row |
| **Paper Analysis** | None | 11 papers × 5 sections |
| **Datasets** | 3/13 parsed | 11/11 target |
| **Rows** | 47,303 | 150,000+ target |
| **Criteria** | 21 total | 25 total |
| **Pass Threshold** | Partial OK | 100% required |

**Key Improvement:** Systematic analysis BEFORE implementation prevents V1 mistakes.

---

## Document Control

**Version:** 1.0
**Created:** 2025-10-12
**Author:** Claude Code CLI
**Status:** DRAFT - Pending execution
**Next Review:** After Tier 0 completion

**Changes from V1:**
- Added knowledge base requirement (Tier 0)
- Added Parsing_Notes column to schema
- Excluded PDF/DOCX datasets (Lofaro, McCabe)
- Increased criteria from 21 to 25
- Changed pass threshold from partial to 100%

---

**END OF ANALYSIS PLAN**
