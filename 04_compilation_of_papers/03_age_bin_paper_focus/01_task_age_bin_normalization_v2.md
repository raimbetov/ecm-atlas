# Task: Age Bin Normalization + Column Mapping Verification (LFQ Focus)

**Date:** 2025-10-12
**Version:** 2.0 (Updated with user approval)
**Goal:** Normalize LFQ datasets to 2 age bins (young vs old) + verify column mapping completeness
**Method Focus:** ⚠️ **ONLY Label-Free Quantification (LFQ)** - exclude TMT/iTRAQ/isobaric methods
**Input:** 11 paper analyses from `knowledge_base/01_paper_analysis/`
**Output:** Age bin analyses + column mapping verification + updated paper analyses

---

## ⚠️ CRITICAL: Agent Workspace Isolation

**EACH AGENT MUST CREATE ITS OWN WORKSPACE:**

```
03_age_bin_paper_focus/
├── claude_code/           ← Claude Code CLI workspace
│   ├── {Study}_age_bin_analysis.md (6 files)
│   ├── 00_cross_study_age_bin_summary.md
│   ├── 90_results_claude_code.md (self-evaluation)
│   └── progress_log.md
├── codex_cli/             ← Codex CLI workspace
│   ├── {Study}_age_bin_analysis.md (6 files)
│   ├── 00_cross_study_age_bin_summary.md
│   ├── 90_results_codex.md (self-evaluation)
│   └── progress_log.md
└── gemini/                ← Gemini CLI workspace
    ├── {Study}_age_bin_analysis.md (6 files)
    ├── 00_cross_study_age_bin_summary.md
    ├── 90_results_gemini.md (self-evaluation)
    └── progress_log.md
```

**⚠️ DO NOT write to shared `03_age_bin_paper_focus/` root folder!**
- Each agent creates its own subfolder first
- No file conflicts between agents
- Parallel execution safe

**Agent-specific workspace:**
- Claude Code: `03_age_bin_paper_focus/claude_code/`
- Codex: `03_age_bin_paper_focus/codex_cli/`
- Gemini: `03_age_bin_paper_focus/gemini/`

---

## ⚠️ USER-APPROVED DECISIONS (MUST FOLLOW)

### 1. Intermediate Group Handling: **Option A - EXCLUDE** ✅
- Middle-aged groups will be EXCLUDED (not combined)
- Example: Tsumagari 15-month → EXCLUDED
- Rationale: Stronger young/old contrast > data quantity

### 2. Method Focus: **ONLY LFQ** ✅
- **INCLUDE:** Label-Free Quantification (LFQ, DIA, Hi-N) methods
- **EXCLUDE:** TMT, iTRAQ, iBAQ, isobaric labeled methods
- Rationale: LFQ methods are directly comparable across studies

### 3. Fetal/Embryonic Samples: **EXCLUDE** ✅
- Fetal samples (Li Pancreas, Caldeira) → EXCLUDED
- Rationale: Embryonic development ≠ aging biology

### 4. Species Cutoffs: **SPECIES-SPECIFIC** ✅
- Mouse: young ≤4mo, old ≥18mo
- Human: young ≤30yr, old ≥55yr
- Cow: young ≤3yr, old ≥15yr

### 5. Data Retention Threshold: **≥66%** ✅
- Accept up to 34% data exclusion for signal quality
- Document all excluded samples explicitly

---

## 1. TASK OVERVIEW

### Two-Part Task

**Part A: Age Bin Normalization (Primary)**
- Normalize 3-4 age groups → 2 age bins (young vs old)
- Focus ONLY on LFQ-compatible studies
- Exclude non-LFQ methods from this analysis

**Part B: Column Mapping Verification (Secondary)**
- Verify all 13 schema columns are mapped for each study
- Identify missing/incomplete mappings
- Document which source files contain proteomic data (not metadata)

### Problem Statement

**Challenge 1: Inconsistent age groupings**
- Current: 5 studies have 2 groups, 6 studies have 3-4 groups
- Solution: Normalize to young vs old, exclude middle-aged

**Challenge 2: Method heterogeneity**
- Current: Mix of LFQ, TMT, iTRAQ, DIA methods
- Solution: Focus ONLY on LFQ-compatible studies for Phase 1

**Challenge 3: Column mapping gaps**
- Current: Some studies may have incomplete column mappings
- Solution: Verify 13-column schema coverage, document gaps

---

## 2. LFQ METHOD IDENTIFICATION

### What is Label-Free Quantification (LFQ)?

**LFQ Definition:**
- Quantification based on MS signal intensity WITHOUT isotope labels
- No SILAC, TMT, iTRAQ, or other chemical/metabolic labels
- Methods: MaxQuant LFQ, DIA (directDIA, DIA-NN), Progenesis Hi-N, Skyline

**Why LFQ Focus:**
- Directly comparable across studies (same measurement scale)
- No batch effects from labeling chemistry
- Most common method in aging proteomics
- Simpler normalization (log2 intensity)

### Study Classification by Method

| Study | Method | LFQ Compatible? | Priority | Notes |
|-------|--------|----------------|----------|-------|
| **Angelidis 2019** | MaxQuant LFQ | ✅ YES | **HIGH** | Label-free, 2 age groups already |
| **Ariosa 2021** | In vivo SILAC (iBAQ) | ❌ NO | EXCLUDE | Isotope labeling (heavy/light) |
| **Caldeira 2017** | iTRAQ LC-MS/MS | ❌ NO | EXCLUDE | Isobaric labeling |
| **Chmelova 2023** | Log2 LFQ | ✅ YES | **HIGH** | Label-free, 2 age groups already |
| **Dipali 2023** | DirectDIA | ✅ YES | **HIGH** | Label-free DIA, 2 age groups already |
| **Li Dermis 2021** | Log2 normalized | ✅ YES (likely) | **MEDIUM** | Need to confirm LFQ in paper |
| **Li Pancreas 2021** | DiLeu reporter | ❌ NO | EXCLUDE | Isobaric labeling (DiLeu) |
| **Ouni 2022** | TMTpro normalized | ❌ NO | EXCLUDE | Isobaric labeling (TMT) |
| **Randles 2021** | Progenesis Hi-N LFQ | ✅ YES | **HIGH** | Label-free, 2 age groups already |
| **Tam 2020** | MaxQuant LFQ | ✅ YES | **HIGH** | Label-free, 2 age groups already |
| **Tsumagari 2023** | TMTpro | ❌ NO | EXCLUDE | Isobaric labeling (TMT) |

### LFQ-Compatible Studies (Phase 1)

**✅ 6 studies qualify for LFQ-focused analysis:**

1. **Angelidis 2019** - Mouse lung, 3mo vs 24mo, MaxQuant LFQ
2. **Chmelova 2023** - Mouse brain, 3mo vs 18mo, log2 LFQ
3. **Dipali 2023** - Mouse ovary, 6-12wk vs 10-12mo, directDIA
4. **Li Dermis 2021** - Human dermis, 4 age groups (needs normalization), log2 normalized (confirm LFQ)
5. **Randles 2021** - Human kidney, 15-37yr vs 61-69yr, Progenesis Hi-N
6. **Tam 2020** - Human spine, 16yr vs 59yr, MaxQuant LFQ

**❌ 5 studies EXCLUDED (non-LFQ):**
- Ariosa 2021 (SILAC)
- Caldeira 2017 (iTRAQ)
- Li Pancreas 2021 (DiLeu)
- Ouni 2022 (TMTpro)
- Tsumagari 2023 (TMTpro)

**⚠️ Li Dermis 2021:** Need to verify from paper if truly label-free (marked as "log2 normalized" but method unclear)

---

## 3. AGE BIN NORMALIZATION (LFQ STUDIES ONLY)

### Studies Requiring Normalization

From 6 LFQ studies, only **1 study needs age bin normalization:**

| Study | Current Age Groups | Action Required |
|-------|-------------------|-----------------|
| Angelidis 2019 | 3mo, 24mo | ✅ No action (already 2 groups) |
| Chmelova 2023 | 3mo, 18mo | ✅ No action (already 2 groups) |
| Dipali 2023 | 6-12wk, 10-12mo | ✅ No action (already 2 groups) |
| **Li Dermis 2021** | **2yr, 14yr, 40yr, 65yr** | ❌ **NEEDS NORMALIZATION (4 → 2 groups)** |
| Randles 2021 | 15-37yr, 61-69yr | ✅ No action (already 2 groups) |
| Tam 2020 | 16yr, 59yr | ✅ No action (already 2 groups) |

### Li Dermis 2021 - Normalization Strategy (USER-APPROVED)

**Current Age Groups:** Toddler (2yr), Teenager (14yr), Adult (40yr), Elderly (65yr)

**Proposed Mapping (Conservative - Option A):**
- **Young:** Toddler (2yr) + Teenager (14yr)
  - Rationale: Both pre-reproductive-peak, ≤30yr cutoff
  - Ages: 2 years, 14 years
  - Sample count: 2 + 3 = 5 samples

- **Old:** Elderly (65yr) ONLY
  - Rationale: Clearly post-reproductive, >55yr cutoff
  - Ages: 65 years (5 samples)
  - Sample count: 5 samples

- **EXCLUDED:** Adult (40yr) - middle-aged
  - Rationale: Between 30-55yr cutoff (ambiguous aging state)
  - Data loss: 3 samples (20% of 15 total)

**Impact:**
- Rows before: 264 proteins × 15 samples = 3,960 rows
- Rows after: 264 proteins × 10 samples = 2,640 rows (-33%)
- Data retention: 67% ✅ (meets ≥66% threshold)
- Signal quality: IMPROVED (young ≤14yr vs old 65yr clear contrast)

---

## 4. COLUMN MAPPING VERIFICATION

### Purpose

Verify that all 13 schema columns can be mapped from source files for each LFQ study:

**13-Column Schema:**
1. Protein_ID
2. Protein_Name
3. Gene_Symbol
4. Tissue
5. Species
6. Age
7. Age_Unit
8. Abundance
9. Abundance_Unit
10. Method
11. Study_ID
12. Sample_ID
13. Parsing_Notes

### Verification Task

For **each of 6 LFQ studies**, verify:

**A. Source File Identification**
- Which file contains proteomic data? (e.g., MOESM5_ESM.xlsx)
- Which sheet/tab within file? (e.g., "Proteome")
- File size and row/column dimensions?
- File format (Excel, TSV, HTML)?

**B. Column Mapping Completeness**

Check each of 13 schema columns:

```markdown
## Column Mapping Verification: {Study Name}

| Schema Column | Source Column/Logic | Status | Notes |
|---------------|---------------------|--------|-------|
| Protein_ID | "Majority protein IDs" (col 0) | ✅ MAPPED | UniProt, semicolon-sep |
| Protein_Name | "Protein names" (col 1) | ✅ MAPPED | Direct mapping |
| Gene_Symbol | "Gene names" (col 2) | ✅ MAPPED | Direct mapping |
| Tissue | Constant "Lung" | ✅ MAPPED | Hardcoded per study |
| Species | Constant "Mus musculus" | ✅ MAPPED | Hardcoded per study |
| Age | Parse from column name "old_1" → 24 | ✅ MAPPED | Lookup table |
| Age_Unit | Constant "months" | ✅ MAPPED | Hardcoded per study |
| Abundance | "old_1" intensity (col 10) | ✅ MAPPED | Direct column value |
| Abundance_Unit | Constant "LFQ_intensity" | ✅ MAPPED | Hardcoded per study |
| Method | Constant "Label-free LC-MS/MS" | ✅ MAPPED | Hardcoded per study |
| Study_ID | Constant "Angelidis_2019" | ✅ MAPPED | Hardcoded |
| Sample_ID | Template "{tissue}_{age}_{rep}" | ✅ MAPPED | Composite field |
| Parsing_Notes | Template "Age={age} from col..." | ✅ MAPPED | Template generation |
```

**C. Gap Identification**

If any column is ❌ MISSING or ⚠️ AMBIGUOUS:
1. Document which column(s) are problematic
2. Explain why mapping failed (no source column? ambiguous logic?)
3. Propose solution (alternative source? derivation logic? external lookup?)

**Example Gap:**
```markdown
❌ **Gene_Symbol** - MISSING
- Problem: Source file has only UniProt IDs, no gene symbols column
- Impact: Cannot populate Gene_Symbol directly
- Proposed Solution:
  - Option A: Map UniProt ID → Gene Symbol via UniProt API/reference table
  - Option B: Leave Gene_Symbol as NULL and document in Parsing_Notes
  - Recommendation: Option A (add external lookup step)
```

---

## 5. TASK INSTRUCTIONS FOR AGENTS

### Step 1: LFQ Study Identification (CRITICAL FIRST STEP)

**Action:** Read all 11 paper analyses and identify which use LFQ methods

**Output:** List of LFQ-compatible studies (expect 5-6 studies)

**Criteria:**
- ✅ LFQ: MaxQuant LFQ, DIA, directDIA, Hi-N, Skyline, label-free
- ❌ NON-LFQ: TMT, iTRAQ, SILAC, iBAQ (heavy/light), DiLeu, any isobaric

**⚠️ CRITICAL:** If study uses TMT/iTRAQ/SILAC → EXCLUDE from this analysis

---

### Step 2: Age Bin Analysis (LFQ Studies Only)

For **each LFQ study**, create analysis document:

**⚠️ CRITICAL:** Create files in YOUR agent workspace, NOT shared root folder!

**File Path Pattern:**
- Claude Code: `03_age_bin_paper_focus/claude_code/{StudyName}_age_bin_analysis.md`
- Codex: `03_age_bin_paper_focus/codex_cli/{StudyName}_age_bin_analysis.md`
- Gemini: `03_age_bin_paper_focus/gemini/{StudyName}_age_bin_analysis.md`

**Required Content:**

```markdown
# {Study Name} - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: [exact method from paper]
- LFQ compatible: [YES/NO with reasoning]
- If NO → EXCLUDE from this analysis
- If YES → Continue below

## 2. Current Age Groups
- List all age groups in original data
- Exact ages with units (months/years/weeks)
- Sample sizes per group (if available)
- File/sheet containing age metadata

## 3. Species Context
- Species: [Mouse/Human/Cow]
- Lifespan reference: [X months/years]
- Aging cutoffs applied:
  - Mouse: young ≤4mo, old ≥18mo
  - Human: young ≤30yr, old ≥55yr
  - Cow: young ≤3yr, old ≥15yr

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** [which original groups qualify as ≤cutoff]
  - Ages: [X, Y]
  - Justification: [biological reasoning]
  - Sample count: [N]
- **Old group:** [which original groups qualify as ≥cutoff]
  - Ages: [Z]
  - Justification: [biological reasoning]
  - Sample count: [M]
- **EXCLUDED:** [groups between cutoffs]
  - Ages: [middle-aged]
  - Rationale: Between cutoffs, ambiguous aging state
  - Sample count: [K]

### Impact Assessment
- **Data retained:** [X% of original samples]
- **Data excluded:** [Y%]
- **Meets ≥66% threshold?** [YES/NO]
- **Signal strength:** [Will young/old contrast be clear?]

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: [filename]
- Sheet/tab name: [name]
- File size: [rows × columns]
- Format: [Excel/TSV/HTML]

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | [source col] | ✅/❌ | [details] |
| Protein_Name | [source col] | ✅/❌ | [details] |
| Gene_Symbol | [source col] | ✅/❌ | [details] |
| Tissue | [constant/derived] | ✅/❌ | [details] |
| Species | [constant] | ✅/❌ | [details] |
| Age | [parsing logic] | ✅/❌ | [details] |
| Age_Unit | [constant/derived] | ✅/❌ | [details] |
| Abundance | [source col] | ✅/❌ | [details] |
| Abundance_Unit | [constant] | ✅/❌ | [details] |
| Method | [constant] | ✅/❌ | [details] |
| Study_ID | [constant] | ✅/❌ | [details] |
| Sample_ID | [template] | ✅/❌ | [details] |
| Parsing_Notes | [template] | ✅/❌ | [details] |

### Mapping Gaps (if any)

**❌ Gap 1:** [Column name]
- Problem: [why mapping failed]
- Proposed solution: [how to fix]

**❌ Gap 2:** [Column name]
- Problem: [why mapping failed]
- Proposed solution: [how to fix]

**✅ All columns mapped:** [if no gaps]

## 6. Implementation Notes
- Column name mapping: [original → schema]
- Sample_ID format: [template with example]
- Parsing_Notes template: [example]
- Special handling: [any edge cases]
```

---

### Step 3: Cross-Study Summary

**⚠️ CRITICAL:** Create in YOUR agent workspace!

**File Path Pattern:**
- Claude Code: `03_age_bin_paper_focus/claude_code/00_cross_study_age_bin_summary.md`
- Codex: `03_age_bin_paper_focus/codex_cli/00_cross_study_age_bin_summary.md`
- Gemini: `03_age_bin_paper_focus/gemini/00_cross_study_age_bin_summary.md`

```markdown
# LFQ-Focused Age Bin Normalization Summary

## Executive Summary
- **Total studies analyzed:** 11
- **LFQ-compatible studies:** [X]
- **Non-LFQ studies (excluded):** [Y]
- **Studies needing age normalization:** [Z]
- **All column mappings complete:** [YES/NO]

## LFQ Study Classification

| Study | Method | LFQ? | Age Groups | Normalization Needed? | Column Mapping Complete? |
|-------|--------|------|------------|----------------------|--------------------------|
| Angelidis 2019 | MaxQuant LFQ | ✅ | 3mo, 24mo | No (2 groups) | ✅ YES |
| ... | ... | ... | ... | ... | ... |

## Excluded Studies (Non-LFQ)

| Study | Method | Reason for Exclusion |
|-------|--------|---------------------|
| Ariosa 2021 | In vivo SILAC | Isotope labeling (not label-free) |
| ... | ... | ... |

## Age Bin Normalization Summary (LFQ Only)

### Studies Already 2-Group (No Action)
1. Angelidis 2019: 3mo vs 24mo ✅
2. ... ✅

### Studies Requiring Normalization
1. Li Dermis 2021: 4 groups → 2 groups (exclude 40yr middle-aged)
   - Young: 2yr + 14yr (5 samples)
   - Old: 65yr (5 samples)
   - Excluded: 40yr (3 samples, 20% loss)
   - Impact: 67% retention ✅

## Column Mapping Verification Results

### Complete Mappings (✅)
- [List studies with all 13 columns mapped]

### Incomplete Mappings (❌)
- **Study X:** Missing Gene_Symbol
  - Problem: [description]
  - Solution: [proposal]

## Recommendations for Phase 2 Parsing

1. **Parse immediately (ready):**
   - [Studies with 2 groups + complete mapping]

2. **Parse after age normalization:**
   - [Studies needing normalization]

3. **Defer to Phase 3 (non-LFQ):**
   - [TMT/iTRAQ studies]

## Data Retention Summary

| Study | Original Samples | After Normalization | Retention % |
|-------|-----------------|---------------------|-------------|
| Li Dermis 2021 | 15 | 10 | 67% ✅ |
| ... | ... | ... | ... |
```

---

### Step 4: Self-Evaluation & Results

**⚠️ MANDATORY:** Create self-evaluation file in YOUR agent workspace:

**File Path Pattern:**
- Claude Code: `03_age_bin_paper_focus/claude_code/90_results_claude_code.md`
- Codex: `03_age_bin_paper_focus/codex_cli/90_results_codex.md`
- Gemini: `03_age_bin_paper_focus/gemini/90_results_gemini.md`

**Required Content:**
```markdown
# Self-Evaluation: {Agent Name}

## Summary
- LFQ studies identified: [X]
- Age bin analyses created: [X]
- Column mapping verifications: [X]
- Paper analyses updated: [X]

## Criterion-by-Criterion Evaluation

### Tier 1: LFQ Identification (3 criteria)
**Criterion 1.1: LFQ studies correctly identified**
- Status: ✅ PASS / ❌ FAIL
- Evidence: [specific files/line numbers]
- Details: [1-2 sentences]

[... repeat for all 13 criteria ...]

## FINAL SCORE: X/13 criteria met
## GRADE: ✅ PASS / ❌ FAIL
```

---

### Step 5: Copy and Update ALL Paper Analyses (MANDATORY)

**⚠️ CRITICAL:** DO NOT modify original files in `knowledge_base/01_paper_analysis/`!

**INSTEAD: Copy to YOUR workspace and update copies:**

**Process:**
1. Copy ALL 11 paper analyses from `knowledge_base/01_paper_analysis/` to YOUR workspace
2. Add Section 6 "Age Bin Normalization Strategy" to EACH copied file
3. For LFQ studies: Document young/old mapping
4. For non-LFQ studies: Document "EXCLUDED - non-LFQ method"

**File Structure:**
```
03_age_bin_paper_focus/{agent_name}/
├── paper_analyses_updated/          ← Create this folder
│   ├── Angelidis_2019_analysis.md   (copy + add Section 6)
│   ├── Ariosa_2021_analysis.md      (copy + add Section 6 - EXCLUDED)
│   ├── Caldeira_2017_analysis.md    (copy + add Section 6 - EXCLUDED)
│   ├── Chmelova_2023_analysis.md    (copy + add Section 6)
│   ├── Dipali_2023_analysis.md      (copy + add Section 6)
│   ├── LiDermis_2021_analysis.md    (copy + add Section 6)
│   ├── LiPancreas_2021_analysis.md  (copy + add Section 6 - EXCLUDED)
│   ├── Ouni_2022_analysis.md        (copy + add Section 6 - EXCLUDED)
│   ├── Randles_2021_analysis.md     (copy + add Section 6)
│   ├── Tam_2020_analysis.md         (copy + add Section 6)
│   └── Tsumagari_2023_analysis.md   (copy + add Section 6 - EXCLUDED)
```

**Add Section 6 to ALL 11 files:**

**Template for LFQ studies (6 studies):**

```markdown
## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### LFQ Method Confirmation
- Method: [MaxQuant LFQ / DirectDIA / Progenesis Hi-N / log2 normalized]
- LFQ compatible: ✅ YES
- Included in Phase 1 parsing: ✅ YES

### Original Age Structure
- [List original groups with ages from original analysis]
  - Example: Toddler (2 years): 2 samples
  - Example: Teenager (14 years): 3 samples
  - Example: Adult (40 years): 3 samples
  - Example: Elderly (65 years): 5 samples

### Normalized to Young vs Old (Conservative Approach)
- **Young:** [which groups qualify as ≤cutoff]
  - Ages: [list ages]
  - Rationale: [≤30yr for human / ≤4mo for mouse cutoff, pre-reproductive/early adult]
  - Sample count: [N samples]
- **Old:** [which groups qualify as ≥cutoff]
  - Ages: [list ages]
  - Rationale: [≥55yr for human / ≥18mo for mouse cutoff, post-reproductive/geriatric]
  - Sample count: [M samples]
- **EXCLUDED:** [middle-aged groups between cutoffs]
  - Ages: [list ages]
  - Rationale: Between cutoffs, ambiguous aging state
  - Sample count: [K samples]
  - Data loss: [X%]

### Impact on Parsing
- Column mapping: [describe which sample columns map to young/old]
- Expected row count: [proteins × retained samples]
- Data retention: [X%] (meets ≥66% threshold: YES/NO)
- Signal quality: [expected young/old contrast]
```

**Template for non-LFQ studies (5 studies):**

```markdown
## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### EXCLUDED FROM LFQ-FOCUSED ANALYSIS

**Method:** [TMTpro / iTRAQ / SILAC / iBAQ / DiLeu]
- Method type: [Isobaric labeling / Isotope labeling]
- LFQ compatible: ❌ NO
- Reason for exclusion: [Specific reason - TMT tags, isotope labels, etc]

**Status:** DEFERRED TO PHASE 3
- This study uses [isobaric/isotope] labeling, not label-free quantification
- Will be analyzed in future phase focusing on [TMT/iTRAQ/SILAC] methods
- Age bin normalization will be defined when parsing this method group
- Current LFQ-focused Phase 1 does not include this study

**Original Age Groups (for reference):**
- [List original age groups from sections above]

**Note:** No age bin mapping performed in this analysis. Age normalization will be addressed in Phase 3 when this study is parsed alongside other [method type] studies.
```

---

## 6. SUCCESS CRITERIA (13 Total - All Required)

### Tier 1: LFQ Study Identification (3 criteria)

**✅ 1.1: LFQ studies correctly identified**
- Evidence: List of 5-6 LFQ-compatible studies
- All non-LFQ methods (TMT/iTRAQ/SILAC) excluded
- Clear justification for each inclusion/exclusion

**✅ 1.2: Method verification documented**
- Evidence: Each study analysis includes "Method Verification" section
- Exact quantification method quoted from paper
- LFQ compatibility reasoning provided

**✅ 1.3: Non-LFQ studies explicitly excluded**
- Evidence: Cross-study summary lists excluded studies
- Reason for exclusion documented per study
- No non-LFQ studies included in age bin analysis

### Tier 2: Age Bin Normalization (4 criteria)

**✅ 2.1: Species-specific cutoffs applied**
- Evidence: Mouse studies use ≤4mo vs ≥18mo
- Human studies use ≤30yr vs ≥55yr
- Cutoffs documented with biological reasoning

**✅ 2.2: Middle-aged groups excluded (conservative)**
- Evidence: Groups between cutoffs marked as EXCLUDED
- No combining of middle-aged with young/old
- Exclusion rationale provided per study

**✅ 2.3: Embryonic/fetal samples excluded**
- Evidence: Li Pancreas fetal samples, Caldeira fetal samples marked as EXCLUDED
- Clear distinction between development vs aging

**✅ 2.4: Data retention ≥66% achieved**
- Evidence: Each analysis calculates % retention
- All studies meet or exceed 66% threshold
- Excluded sample counts documented

### Tier 3: Column Mapping Verification (4 criteria)

**✅ 3.1: All 13 columns verified per study**
- Evidence: Each study analysis includes 13-column mapping table
- Status (✅/❌) provided for each column
- Source column or logic documented

**✅ 3.2: Source files identified (proteomic data only)**
- Evidence: Primary proteomic file named per study
- Sheet/tab name specified
- File dimensions (rows × cols) documented
- No metadata-only files included

**✅ 3.3: Mapping gaps documented and resolved**
- Evidence: If any column is ❌ MISSING, gap is documented
- Problem description provided
- Proposed solution offered (e.g., external lookup, derivation logic)
- If no gaps, explicitly state "All columns mapped"

**✅ 3.4: Implementation-ready mappings**
- Evidence: Column mappings are actionable (not vague)
- Parser can implement mechanically
- Sample_ID templates provided with examples

### Tier 4: Integration & Deliverables (2 criteria)

**✅ 4.1: All deliverable files created in YOUR workspace**
- Evidence:
  - 6 individual LFQ study analyses in `03_age_bin_paper_focus/{agent_name}/`
  - 1 cross-study summary: `03_age_bin_paper_focus/{agent_name}/00_cross_study_age_bin_summary.md`
  - 1 self-evaluation: `03_age_bin_paper_focus/{agent_name}/90_results_{agent}.md`
  - ⚠️ MANDATORY: 11 updated paper analyses in `03_age_bin_paper_focus/{agent_name}/paper_analyses_updated/`
    - Each has Section 6 added (LFQ: age mapping, non-LFQ: EXCLUDED notice)

**✅ 4.2: Ready for Phase 2 parsing**
- Evidence: Age bin decisions are clear and actionable
- Column mappings enable mechanical parsing
- No ambiguities requiring manual per-sample decisions
- LFQ subset ready for immediate parsing

---

## 7. DELIVERABLES CHECKLIST

### Primary Deliverables (in YOUR agent workspace)

**⚠️ File Structure - EACH AGENT:**
```
03_age_bin_paper_focus/{agent_name}/
├── Angelidis_2019_age_bin_analysis.md
├── Chmelova_2023_age_bin_analysis.md
├── Dipali_2023_age_bin_analysis.md
├── LiDermis_2021_age_bin_analysis.md
├── Randles_2021_age_bin_analysis.md
├── Tam_2020_age_bin_analysis.md
├── 00_cross_study_age_bin_summary.md
├── 90_results_{agent_name}.md (self-evaluation)
└── paper_analyses_updated/          ⚠️ NEW - MANDATORY
    ├── Angelidis_2019_analysis.md   (copy + Section 6)
    ├── Ariosa_2021_analysis.md      (copy + Section 6 - EXCLUDED)
    ├── Caldeira_2017_analysis.md    (copy + Section 6 - EXCLUDED)
    ├── Chmelova_2023_analysis.md    (copy + Section 6)
    ├── Dipali_2023_analysis.md      (copy + Section 6)
    ├── LiDermis_2021_analysis.md    (copy + Section 6)
    ├── LiPancreas_2021_analysis.md  (copy + Section 6 - EXCLUDED)
    ├── Ouni_2022_analysis.md        (copy + Section 6 - EXCLUDED)
    ├── Randles_2021_analysis.md     (copy + Section 6)
    ├── Tam_2020_analysis.md         (copy + Section 6)
    └── Tsumagari_2023_analysis.md   (copy + Section 6 - EXCLUDED)
```

**Checklist:**
- [ ] 6 LFQ study analyses in YOUR workspace
- [ ] 1 cross-study summary in YOUR workspace
- [ ] 1 self-evaluation file (90_results_*.md)
- [ ] Column mapping verification for all 6 LFQ studies
- [ ] ⚠️ MANDATORY: 11 copied & updated paper analyses in `{agent_name}/paper_analyses_updated/`
  - [ ] 6 LFQ studies with age bin mapping (Section 6)
  - [ ] 5 non-LFQ studies marked as "EXCLUDED" (Section 6)

### Quality Checks
- [ ] All non-LFQ studies explicitly excluded from analysis
- [ ] Middle-aged groups EXCLUDED (not combined)
- [ ] Embryonic/fetal samples EXCLUDED
- [ ] Species-specific cutoffs applied consistently
- [ ] Data retention ≥66% for all studies
- [ ] All 13 columns mapped (or gaps documented with solutions)
- [ ] Source proteomic files identified (not metadata files)

### Integration Checks
- [ ] Age bin strategy aligns with column mapping strategy
- [ ] No conflicts with implementation plan
- [ ] LFQ subset clearly prioritized for Phase 2 parsing

---

## 8. AGENT-SPECIFIC GUIDANCE

### What You MUST Do

**⚠️ FIRST STEP: Create your workspace folders!**
```bash
# If Claude Code:
mkdir -p 03_age_bin_paper_focus/claude_code/
mkdir -p 03_age_bin_paper_focus/claude_code/paper_analyses_updated/

# If Codex:
mkdir -p 03_age_bin_paper_focus/codex_cli/
mkdir -p 03_age_bin_paper_focus/codex_cli/paper_analyses_updated/

# If Gemini:
mkdir -p 03_age_bin_paper_focus/gemini/
mkdir -p 03_age_bin_paper_focus/gemini/paper_analyses_updated/
```

Then proceed:

1. ✅ **Work in YOUR workspace** - All files go in your agent subfolder
2. ✅ **Copy ALL 11 paper analyses** - From `knowledge_base/01_paper_analysis/` to YOUR `paper_analyses_updated/`
3. ✅ **Start with LFQ filtering** - Identify LFQ studies FIRST, exclude non-LFQ immediately
4. ✅ **Apply user-approved decisions** - Exclude middle-aged, exclude fetal, species-specific cutoffs
5. ✅ **Verify 13-column mapping** - Check completeness, document gaps
6. ✅ **Identify proteomic source files** - Primary data files, not metadata
7. ✅ **Document all exclusions** - Middle-aged groups, non-LFQ studies, fetal samples
8. ✅ **Calculate data retention** - Must be ≥66% after exclusions
9. ✅ **Update ALL 11 copied paper analyses** - Add Section 6 to each
10. ✅ **Create self-evaluation** - 90_results_{agent}.md with criterion-by-criterion scoring

### What You MUST NOT Do

❌ **Don't write to shared root folder** - Use YOUR agent workspace only
❌ **Don't overwrite other agents' files** - Each agent has separate folder
❌ **Don't modify original paper analyses** - Copy to YOUR workspace first, then update copies
❌ **Don't skip copying paper analyses** - ALL 11 files must be copied and updated (mandatory)
❌ **Don't include non-LFQ studies in age bin analysis** - TMT/iTRAQ/SILAC excluded from mapping
❌ **Don't combine middle-aged groups** - Conservative approach: EXCLUDE
❌ **Don't include fetal samples** - Embryonic development ≠ aging
❌ **Don't use non-species-specific cutoffs** - Mouse ≠ human aging timeline
❌ **Don't analyze metadata files** - Only proteomic data files (large protein lists)
❌ **Don't skip column mapping verification** - Must check all 13 columns
❌ **Don't skip self-evaluation** - 90_results file is mandatory

### Expected Output (Per Agent)

**Files in YOUR workspace (`03_age_bin_paper_focus/{agent_name}/`):**

**For 6 LFQ studies:**
- 6 individual age bin analyses ({Study}_age_bin_analysis.md)
- 1 cross-study summary (00_cross_study_age_bin_summary.md)
- 1 self-evaluation (90_results_{agent}.md) ⚠️ MANDATORY
- Column mapping verification for all 6 (embedded in analyses)

**For 5 non-LFQ studies:**
- Listed in "Excluded Studies" section of cross-study summary
- No individual age bin analyses (excluded from LFQ focus)

**For ALL 11 studies (LFQ + non-LFQ):** ⚠️ MANDATORY
- Copy all 11 paper analyses from `knowledge_base/01_paper_analysis/` to YOUR `paper_analyses_updated/`
- Add Section 6 to ALL 11 copied files:
  - 6 LFQ studies: Document age bin mapping (young/old)
  - 5 non-LFQ studies: Mark as "EXCLUDED - non-LFQ method, deferred to Phase 3"
- DO NOT modify original files in `knowledge_base/` - only YOUR copies!

---

## 9. EXAMPLE: Li Dermis 2021 (Complete Analysis)

```markdown
# Li Dermis 2021 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: Label-free LC-MS/MS with log2 normalization (from Methods p.3)
- LFQ compatible: ✅ YES - No isotope labeling, intensity-based quantification
- Priority: HIGH - LFQ method, human tissue, multiple age groups

## 2. Current Age Groups
- Toddler (1-3 years, midpoint 2yr): 2 samples
- Teenager (8-20 years, midpoint 14yr): 3 samples
- Adult (30-50 years, midpoint 40yr): 3 samples
- Elderly (>60 years, midpoint 65yr): 5 samples
- **Total:** 15 samples
- **Source:** Table 2.xlsx, sheet "Table S2"

## 3. Species Context
- Species: Homo sapiens
- Lifespan reference: ~75-85 years
- Aging cutoffs applied: young ≤30yr, old ≥55yr
- Biological markers:
  - Young: reproductive peak, collagen synthesis high
  - Old: post-reproductive, ECM degradation, wrinkles

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** Toddler (2yr) + Teenager (14yr)
  - Ages: 2 years, 14 years
  - Justification: Both ≤30yr cutoff, pre-reproductive-peak, no aging markers
  - Sample count: 2 + 3 = 5 samples

- **Old group:** Elderly (65yr) only
  - Ages: 65 years
  - Justification: >55yr cutoff, clearly post-reproductive, aging phenotype
  - Sample count: 5 samples

- **EXCLUDED:** Adult (40yr) - middle-aged
  - Ages: 40 years
  - Rationale: Between 30-55yr cutoffs, ambiguous aging state (pre-senescent but not young)
  - Sample count: 3 samples

### Impact Assessment
- **Data retained:** 10/15 samples = 67% ✅ (meets ≥66% threshold)
- **Data excluded:** 3/15 samples = 20%
- **Signal strength:** Strong contrast (2-14yr vs 65yr) - 43-51 year gap
- **Biological relevance:** Captures pre-reproductive vs geriatric comparison

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `Table 2.xlsx`
- Sheet/tab name: "Table S2"
- File size: 266 rows × 22 columns
- Format: Excel (.xlsx)
- Skip rows: 3 (header rows with merged cells)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "Protein ID" (col 0, after skip) | ✅ MAPPED | UniProt accessions |
| Protein_Name | Not in file, derive from UniProt | ⚠️ DERIVED | Requires external lookup |
| Gene_Symbol | "Gene symbol" (col 1, after skip) | ✅ MAPPED | Direct mapping |
| Tissue | Constant "Dermis" | ✅ MAPPED | Hardcoded per study |
| Species | Constant "Homo sapiens" | ✅ MAPPED | Hardcoded per study |
| Age | Parse from column group prefix | ✅ MAPPED | "Toddler-Sample1-2" → 2yr lookup |
| Age_Unit | Constant "years" | ✅ MAPPED | Hardcoded per study |
| Abundance | Sample columns (cols 4-18) | ✅ MAPPED | Log2 normalized intensities |
| Abundance_Unit | Constant "log2_normalized_intensity" | ✅ MAPPED | From Methods section |
| Method | Constant "Label-free LC-MS/MS" | ✅ MAPPED | From Methods p.3 |
| Study_ID | Constant "LiDermis_2021" | ✅ MAPPED | Hardcoded |
| Sample_ID | Template "{age_group}_{sample_num}" | ✅ MAPPED | "Toddler_1", "Elderly_3" |
| Parsing_Notes | Template "Age={age}yr from {col}..." | ✅ MAPPED | Generate per row |

### Mapping Gaps

**⚠️ Gap 1: Protein_Name**
- Problem: Source file doesn't include protein name column
- Proposed solution:
  - Option A: Map Protein_ID (UniProt) → Protein_Name via UniProt API
  - Option B: Query local UniProt reference table
  - Recommendation: Option B (faster, offline-capable)
- Impact: Requires pre-processing step before parsing

**✅ All other columns:** Direct mapping or hardcoded constants

## 6. Implementation Notes
- Column name parsing: "Toddler-Sample1-2" → extract "Toddler" → lookup age 2yr
- Sample_ID format: `Dermis_{age_group}_{sample_num}` (e.g., "Dermis_Toddler_1")
- Parsing_Notes template: "Age={age}yr (midpoint of {range}) from column '{col_name}' per Supplementary Table S1; Abundance from Table 2 sheet 'Table S2' col {col}; log2 normalized per Methods p.3"
- Special handling: Skip first 3 rows before reading data; filter samples to exclude Adult (40yr) group
- UniProt mapping: Pre-process Protein_ID list to fetch Protein_Name before parsing
```

---

## 10. EXPECTED TIMELINE

**Phase 1: LFQ Identification (5 min)**
- Read 11 paper analyses
- Classify by method (LFQ vs non-LFQ)
- Create exclusion list

**Phase 2: Age Bin Analysis (10 min)**
- Analyze 6 LFQ studies
- Apply user-approved decisions
- Create individual analyses

**Phase 3: Column Mapping Verification (10 min)**
- Verify 13 columns per study
- Document gaps
- Propose solutions

**Phase 4: Paper Analyses Update (10 min)** ⚠️ NEW
- Copy all 11 paper analyses to YOUR workspace
- Add Section 6 to all 11 files (LFQ mapping or non-LFQ excluded)
- Verify all copied files have new section

**Phase 5: Integration (5 min)**
- Create cross-study summary
- Final quality check
- Self-evaluation

**Total: 35 minutes** (was 30, now +5 for paper analyses copying/updating)

---

## 11. USER-APPROVED DECISIONS (CONFIRMED) ✅

**These decisions are FINAL and must be followed:**

1. ✅ **Intermediate groups:** EXCLUDE (Option A - Conservative)
2. ✅ **Method focus:** ONLY LFQ (exclude TMT/iTRAQ/SILAC)
3. ✅ **Fetal samples:** EXCLUDE (embryonic ≠ aging)
4. ✅ **Species cutoffs:** SPECIES-SPECIFIC (mouse/human/cow different)
5. ✅ **Data retention:** ≥66% (accept up to 34% exclusion)

---

**Status:** ⏸️ READY FOR MULTI-AGENT LAUNCH
**Awaiting:** User confirmation to proceed
**Estimated Time:** 30 minutes
**Output:** 6 LFQ analyses + 1 summary + 1 updated paper analysis + column mapping verification

