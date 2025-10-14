# Claude Code - Age Bin Normalization Analysis Plan

**Date:** 2025-10-12
**Agent:** Claude Code CLI
**Task:** Age bin normalization + column mapping verification (LFQ focus)

---

## Execution Plan

### Phase 1: LFQ Study Identification (5 min)
**Objective:** Identify which of 11 studies use label-free quantification

**Actions:**
1. Read all 11 paper analyses from `knowledge_base/01_paper_analysis/`
2. Extract quantification method from each study
3. Classify as LFQ-compatible (✅) or non-LFQ (❌)
4. Expected: 6 LFQ studies, 5 non-LFQ exclusions

**Output:** Classification list for cross-study summary

---

### Phase 2: Age Bin Analysis (10 min)
**Objective:** Create detailed age bin normalization for 6 LFQ studies

**Actions:**
1. For each LFQ study, create individual analysis file:
   - `{Study}_age_bin_analysis.md`
2. Apply user-approved decisions:
   - Exclude middle-aged groups (conservative approach)
   - Species-specific cutoffs (mouse ≤4mo/≥18mo, human ≤30yr/≥55yr)
   - Exclude fetal/embryonic samples
   - Verify ≥66% data retention

**Output:** 6 individual study analyses in `claude_code/`

---

### Phase 3: Column Mapping Verification (10 min)
**Objective:** Verify 13-column schema mapping completeness

**Actions:**
1. For each LFQ study, verify all 13 schema columns:
   - Identify source file (proteomic data, not metadata)
   - Map each column to source or document gap
   - Propose solutions for missing columns
2. Embed verification in individual study analyses

**Output:** Complete mapping tables in each study analysis

---

### Phase 4: Paper Analyses Update (10 min)
**Objective:** Copy all 11 paper analyses and add Section 6

**Actions:**
1. Copy all 11 files from `knowledge_base/01_paper_analysis/` to `claude_code/paper_analyses_updated/`
2. Add Section 6 to each file:
   - **LFQ studies (6):** Document young/old age bin mapping
   - **Non-LFQ studies (5):** Mark as "EXCLUDED - non-LFQ method"
3. Verify no modifications to original files

**Output:** 11 updated paper analyses in `claude_code/paper_analyses_updated/`

---

### Phase 5: Integration & Self-Evaluation (5 min)
**Objective:** Create cross-study summary and evaluate success

**Actions:**
1. Create `00_cross_study_age_bin_summary.md`:
   - LFQ study classification table
   - Excluded studies list
   - Age bin normalization summary
   - Column mapping verification results
   - Recommendations for Phase 2 parsing
2. Create `90_results_claude_code.md`:
   - Self-evaluation against 13 success criteria
   - Evidence with file paths and line numbers
   - Final pass/fail determination

**Output:** Summary and self-evaluation files

---

## User-Approved Decisions (MUST FOLLOW)

1. ✅ **Intermediate groups:** EXCLUDE (not combine)
2. ✅ **Method focus:** ONLY LFQ (exclude TMT/iTRAQ/SILAC)
3. ✅ **Fetal samples:** EXCLUDE
4. ✅ **Species cutoffs:** SPECIES-SPECIFIC
5. ✅ **Data retention:** ≥66%

---

## Expected LFQ Studies (6)

Based on task description:
1. Angelidis 2019 - MaxQuant LFQ
2. Chmelova 2023 - Log2 LFQ
3. Dipali 2023 - DirectDIA
4. Li Dermis 2021 - Log2 normalized (confirm LFQ)
5. Randles 2021 - Progenesis Hi-N
6. Tam 2020 - MaxQuant LFQ

---

## Success Criteria (13 Total)

### Tier 1: LFQ Identification (3)
- 1.1: LFQ studies correctly identified
- 1.2: Method verification documented
- 1.3: Non-LFQ studies explicitly excluded

### Tier 2: Age Bin Normalization (4)
- 2.1: Species-specific cutoffs applied
- 2.2: Middle-aged groups excluded
- 2.3: Embryonic/fetal samples excluded
- 2.4: Data retention ≥66%

### Tier 3: Column Mapping (4)
- 3.1: All 13 columns verified per study
- 3.2: Source files identified
- 3.3: Mapping gaps documented
- 3.4: Implementation-ready mappings

### Tier 4: Integration (2)
- 4.1: All deliverables in workspace
- 4.2: Ready for Phase 2 parsing

---

## Timeline

- Phase 1: 5 min (LFQ identification)
- Phase 2: 10 min (age bin analyses)
- Phase 3: 10 min (column mapping)
- Phase 4: 10 min (paper analyses update)
- Phase 5: 5 min (integration)
- **Total: 40 minutes**

---

**Status:** Plan created ✅
**Next:** Begin Phase 1 - LFQ study identification
