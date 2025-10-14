# Angelidis 2019 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: MaxQuant Label-Free Quantification (LFQ)
- LFQ compatible: YES - Label-free LC-MS/MS workflow with MaxQuant LFQ intensities (Methods p.14)
- Included in Phase 1 parsing: YES

## 2. Current Age Groups
- Young: 3 months (4 replicates)
- Old: 24 months (4 replicates)
- Total samples: 8
- File/sheet: `41467_2019_8831_MOESM5_ESM.xlsx`, sheet "Proteome"

## 3. Species Context
- Species: Mus musculus (C57BL/6J cohorts)
- Lifespan reference: ~24-30 months
- Aging cutoffs applied:
  - Mouse: young ≤4 months, old ≥18 months
  - Study groups: 3 months (young) vs 24 months (old)

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Already 2 Groups)
- **Young group:** 3 months
  - Ages: 3 months
  - Justification: Below 4-month cutoff, pre-reproductive peak
  - Sample count: 4 replicates
- **Old group:** 24 months
  - Ages: 24 months
  - Justification: Above 18-month cutoff, geriatric age for mice
  - Sample count: 4 replicates
- **EXCLUDED:** None
  - No middle-aged groups in this study

### Impact Assessment
- **Data retained:** 100% (8/8 samples)
- **Data excluded:** 0%
- **Meets ≥66% threshold?** YES (100%)
- **Signal strength:** Excellent - 21-month age gap provides clear young/old contrast

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `41467_2019_8831_MOESM5_ESM.xlsx`
- Sheet/tab name: "Proteome"
- File size: 5,214 rows × 36 columns
- Format: Excel (.xlsx)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "Protein IDs" (col 0) | ✅ MAPPED | UniProt accessions from MaxQuant |
| Protein_Name | "Protein names" (col 1) | ✅ MAPPED | Human-readable names |
| Gene_Symbol | "Gene names" (col 2) | ✅ MAPPED | MaxQuant gene annotation |
| Tissue | Constant "Lung" | ✅ MAPPED | Whole mouse lung homogenate |
| Species | Constant "Mus musculus" | ✅ MAPPED | C57BL/6J mouse cohorts |
| Age | Derived from column names "young_*"/"old_*" | ✅ MAPPED | young=3mo, old=24mo |
| Age_Unit | Constant "months" | ✅ MAPPED | Ages described in months |
| Abundance | Sample columns (young_1-4, old_1-4) | ✅ MAPPED | LFQ intensities |
| Abundance_Unit | Constant "LFQ_intensity" | ✅ MAPPED | MaxQuant LFQ output |
| Method | Constant "Label-free LC-MS/MS" | ✅ MAPPED | MaxQuant label-free pipeline |
| Study_ID | Constant "Angelidis_2019" | ✅ MAPPED | Unique identifier |
| Sample_ID | Column header (e.g., "young_1", "old_1") | ✅ MAPPED | Encodes cohort + replicate |
| Parsing_Notes | Template | ✅ MAPPED | Age mapping, LFQ context |

### Mapping Gaps (if any)

✅ **All columns mapped** - No gaps identified

## 6. Implementation Notes
- Column name mapping: "young_1" → Age=3mo, "old_1" → Age=24mo
- Sample_ID format: Use column header directly (e.g., "young_1", "old_2", "old_3", "old_4")
- Parsing_Notes template: "Age={age}mo from column '{col_name}'; LFQ intensity from MaxQuant (Methods p.14); 4 replicates per age group"
- Special handling: Use "Protein IDs" column (not "Majority protein IDs") for canonical accession per MaxQuant documentation
