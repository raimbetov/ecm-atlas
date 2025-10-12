# Angelidis et al. 2019 - Comprehensive Analysis

**Compilation Date:** 2025-10-12
**Study Type:** LFQ Proteomics (Phase 1 Ready)
**Sources Synthesized:** Original paper analysis + Claude Code + Codex CLI age bin analyses

---

## 1. Paper Overview

- **Title:** An atlas of the aging lung mapped by single cell transcriptomics and deep tissue proteomics
- **PMID:** 30814501
- **Tissue:** Lung (whole tissue homogenate)
- **Species:** Mus musculus (C57BL/6J cohorts)
- **Age groups:** Young 3-month mice vs Old 24-month mice (four replicates each)
- **Reference:** Methods page 3, Abstract page 1

---

## 2. Method Classification & Quality Control

### Quantification Method
- **Method:** MaxQuant Label-Free Quantification (LFQ) on Thermo Orbitrap Fusion
- **Reference:** Methods p.14
- **Workflow:** Label-free LC-MS/MS with MaxQuant pipeline, match-between-runs enabled

### LFQ Compatibility Verification
- **Claude Code assessment:** ✅ YES - Label-free LC-MS/MS with MaxQuant LFQ intensities
- **Codex CLI assessment:** ✅ YES - No isotopic or isobaric labeling; LFQ intensities per sample
- **Final verdict:** ✅ **LFQ COMPATIBLE** - Included in Phase 1 parsing

### Quality Control
- **Agent agreement:** Both agents correctly identified as LFQ method
- **No classification conflicts:** ✅ Unanimous classification
- **Data integrity:** High - MaxQuant standardized output with internal normalization

---

## 3. Age Bin Normalization Strategy

### Original Age Design
- **Young cohort:** 3 months (4 biological replicates)
  - Columns: `young_1`, `young_2`, `young_3`, `young_4`
  - Biological context: Pre-reproductive peak, young adult baseline
- **Old cohort:** 24 months (4 biological replicates)
  - Columns: `old_1`, `old_2`, `old_3`, `old_4`
  - Biological context: Geriatric age for C57BL/6J mice (lifespan ~26-30 months)
- **Total samples:** 8

### Species-Specific Cutoffs (User-Approved)
- **Species:** Mus musculus
- **Lifespan reference:** ~24-30 months (laboratory C57BL/6J)
- **Aging cutoffs applied:**
  - Young: ≤4 months
  - Old: ≥18 months

### Normalization Assessment
- **Current design:** Already binary (2 age groups)
- **Young group mapping:**
  - Ages: 3 months
  - Justification: Below 4-month cutoff (3 ≤ 4) ✅
  - Sample count: 4 replicates
- **Old group mapping:**
  - Ages: 24 months
  - Justification: Above 18-month cutoff (24 ≥ 18) ✅
  - Sample count: 4 replicates
- **EXCLUDED groups:** None - no middle-aged samples in this study

### Data Retention Analysis
- **Data retained:** 100% (8/8 samples)
- **Data excluded:** 0%
- **Meets ≥66% threshold?** ✅ YES (exceeds requirement)
- **Signal strength:** Excellent - 21-month age gap provides clear young/old contrast
- **Conclusion:** **NO NORMALIZATION REQUIRED** - proceed directly to parsing

---

## 4. Column Mapping to 13-Column Schema

### Source File Details
- **Primary file:** `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`
- **Target sheet:** `Proteome`
- **File dimensions:** 5,214 rows × 36 columns
- **Format:** Excel (.xlsx) from Nature Communications supplement

### Alternative Data Files (Not Used for Atlas)
- `41467_2019_8831_MOESM7_ESM.xlsx` - QDSP detergent fractionation profiles (5,119 rows × 64 columns)
- `41467_2019_8831_MOESM6_ESM.xlsx` - ELISA/immunoblot validation panel (109 rows × 8 columns)
- **Decision:** Restrict parsing to `Proteome` sheet for primary proteomic data

### Complete Schema Mapping

| Schema Column | Source Location | Status | Reasoning & Notes |
|---------------|----------------|--------|-------------------|
| **Protein_ID** | `Protein IDs` (col 0) | ✅ MAPPED | MaxQuant label-free pipeline outputs UniProt accessions; use **first accession** for canonical ID if semicolon-separated (Methods p.14) |
| **Protein_Name** | `Protein names` (col 1) | ✅ MAPPED | Human-readable names aligned with UniProt, required for provenance |
| **Gene_Symbol** | `Gene names` (col 2) | ✅ MAPPED | MaxQuant gene annotation per protein group |
| **Tissue** | Constant `Lung` | ✅ MAPPED | Study profiled whole mouse lungs; no other tissues (Abstract p.1) |
| **Species** | Constant `Mus musculus` | ✅ MAPPED | All samples from C57BL/6J mouse cohorts (Methods p.3) |
| **Age** | Derived from column prefix `young_` / `old_` | ✅ MAPPED | Map to numeric: `young_*` → 3 months, `old_*` → 24 months |
| **Age_Unit** | Constant `months` | ✅ MAPPED | Ages explicitly described as months (Methods p.3) |
| **Abundance** | Sample columns: `young_1-4`, `old_1-4` | ✅ MAPPED | Label-free quantification (LFQ) intensities from MaxQuant |
| **Abundance_Unit** | Constant `LFQ_intensity` | ✅ MAPPED | MaxQuant LFQ output with internal normalization (Methods p.14) |
| **Method** | Constant `Label-free LC-MS/MS (MaxQuant LFQ)` | ✅ MAPPED | Shotgun proteomics with label-free quantification workflow |
| **Study_ID** | Constant `Angelidis_2019` | ✅ MAPPED | Unique identifier for downstream joins |
| **Sample_ID** | Use column header | ✅ MAPPED | E.g., `young_1`, `old_3` - encodes cohort + replicate |
| **Parsing_Notes** | Template (see below) | ✅ MAPPED | Capture age mapping, LFQ context, replicate info |

### Mapping Quality
- ✅ **All 13 columns mapped** - No gaps identified
- ✅ **Agent agreement:** Both Claude Code and Codex CLI confirmed complete mapping
- ✅ **Source clarity:** All mappings have clear source columns or derivation logic

### Important Column Selection Decision
- **`Protein IDs` vs `Majority protein IDs`:**
  - **Decision:** Use `Protein IDs` column (not `Majority protein IDs`)
  - **Reasoning:** `Protein IDs` captures primary accession per MaxQuant documentation; `Majority` retains grouped evidence but is redundant for canonical ID (Methods p.14)
  - **Implementation:** Select first semicolon-separated accession if multiple IDs present

---

## 5. Parsing Implementation Guide

### Data Extraction Steps

**Step 1: File Loading**
```python
# Read Excel file, target Proteome sheet
file_path = "data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx"
sheet_name = "Proteome"

# Option: Skip header row if needed (Codex note)
df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=1)  # if header descriptive
```

**Step 2: Sample Columns**
- **Young samples:** `young_1`, `young_2`, `young_3`, `young_4`
- **Old samples:** `old_1`, `old_2`, `old_3`, `old_4`
- **Column order:** Maintain young→old ordering for reproducibility

**Step 3: Age Mapping**
```python
# Derive age from column name
if 'young_' in column_name:
    age = 3
    age_unit = 'months'
elif 'old_' in column_name:
    age = 24
    age_unit = 'months'
```

**Step 4: Sample_ID Format**
- **Template:** Use column header directly
- **Examples:** `young_1`, `young_2`, `old_1`, `old_2`
- **Format:** `{cohort}_{replicate_number}`

**Step 5: Parsing_Notes Template**
```python
parsing_notes = f"Age={age}mo from column '{column_name}'; LFQ intensity from MaxQuant (Methods p.14); 4 replicates per age group; C57BL/6J cohorts"
```

**Step 6: Protein ID Processing**
```python
# Handle semicolon-separated accessions
protein_id = row['Protein IDs'].split(';')[0]  # Select first/canonical accession
```

### Expected Output
- **Format:** Long-format CSV with 13 columns
- **Expected rows:** 5,214 proteins × 8 samples = **41,712 rows**
- **Validation:** Check for 4 replicates per age group, no missing LFQ values in critical samples

### Preprocessing Requirements
- ✅ **None required** - Data ready for direct parsing
- ⚠️ **Optional:** Consider log2-transforming LFQ intensities downstream (values currently linear); document transformation in `Parsing_Notes`
- ⚠️ **Optional:** Filter out low-quality proteins (e.g., based on MaxQuant reverse/contaminant flags if available)

### Implementation Notes (Synthesized from Both Agents)
1. **Protein ID selection:** Use `Protein IDs` column; split on semicolon and select first entry for canonical UniProt accession
2. **Header handling:** Check for descriptive header row; skip if necessary (`skiprows=1`)
3. **Column order:** Preserve `young_1-4`, `old_1-4` ordering for replicate reproducibility
4. **LFQ transformation:** Values are linear LFQ intensities; consider log2 transform for downstream analysis (document in notes)
5. **Ambiguity resolution:** Methods cite MaxQuant defaults where `Protein IDs` captures primary accession

---

## 6. Quality Assurance & Biological Context

### Study Design Strengths
- ✅ **Optimal binary design:** Already young vs old, no normalization needed
- ✅ **Adequate replication:** 4 biological replicates per age group
- ✅ **Clear age separation:** 21-month gap (3mo → 24mo) captures young adult vs geriatric states
- ✅ **Biological relevance:** 24-month mice near end of typical C57BL/6J lifespan (~26-30mo)

### Known Limitations & Considerations
1. **Replicate metadata ambiguity:**
   - Workbook lacks explicit mapping from replicate IDs (1-4) to individual mouse metadata (sex, exact age at sacrifice)
   - Recommendation: Cross-reference Methods supplementary text if precise per-sample metadata needed
2. **Transcriptomics co-presence:**
   - Same workbook contains RNA-seq data (other sheets)
   - Decision: Restrict to `Proteome` sheet only; RNA-seq outside proteomic atlas scope
3. **Fractionation profiles:**
   - QDSP sheet contains detergent fractionation data (MOESM7)
   - Decision: Defer to specialized analysis; primary `Proteome` sheet sufficient for atlas

### Cross-Study Comparisons
- **Similar studies in atlas:** Dipali 2023 (mouse ovary, LFQ), Tsumagari 2023 (mouse brain, TMT - Phase 3)
- **Species consistency:** Mouse aging cutoffs (≤4mo, ≥18mo) consistently applied across all mouse studies
- **Method comparison:** MaxQuant LFQ also used in Tam 2020 (human spine) - allows direct cross-species comparison

### Biological Insights (From Paper)
- **Aging signature:** Paper reports ECM remodeling and immune infiltration at 24 months
- **Validation:** ELISA/immunoblot data in MOESM6 confirms key protein changes
- **Multi-omics:** Transcriptomics available for validation (though not included in atlas)

---

## 7. Ready for Phase 2 Parsing

### Parsing Status
- ✅ **READY FOR IMMEDIATE PARSING**
- ✅ No preprocessing required
- ✅ No age bin normalization needed (already binary)
- ✅ All 13 columns mapped with clear sources
- ✅ Expected output: 41,712 rows (5,214 proteins × 8 samples)

### Parsing Priority
- **Priority:** HIGH (Tier 1)
- **Recommendation:** Use as **pilot study** for Phase 2 parser development
- **Rationale:** Simple binary design, complete mappings, no preprocessing - ideal for testing pipeline

### Next Steps
1. Implement parser for Angelidis 2019
2. Validate output against expected row count (41,712)
3. Check data types and missing value handling
4. Use as template for other LFQ studies (Dipali, Randles, Tam)

---

**Compilation Notes:**
- **Primary source for column mapping:** Original knowledge_base paper analysis (most detailed reasoning)
- **Age bin strategy:** Synthesized from Claude Code + Codex CLI (identical conclusions)
- **Implementation details:** Combined best practices from both agents (Codex log2 note + Claude skiprows note)
- **No agent conflicts:** Both Claude Code and Codex CLI unanimous on all classifications and mappings

**Agent Contributions:**
- 🔵 **Claude Code:** Age bin analysis, method verification, column mapping validation
- 🟢 **Codex CLI:** Age bin analysis, implementation notes (skiprows, log2 transform)
- 📚 **Knowledge Base:** Original detailed column mapping with paper references
