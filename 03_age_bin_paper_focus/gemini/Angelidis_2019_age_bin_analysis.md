# Angelidis 2019 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: Label-free quantification via MaxQuant
- LFQ compatible: ✅ YES - The study uses MaxQuant's label-free quantification (LFQ) which is a standard LFQ method.

## 2. Current Age Groups
- Young: 3-month-old mice (4 replicates)
- Old: 24-month-old mice (4 replicates)
- Source: `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`, sheet `Proteome`

## 3. Species Context
- Species: Mus musculus
- Lifespan reference: ~24-30 months
- Aging cutoffs applied:
  - Mouse: young ≤4mo, old ≥18mo

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** 3-month-old mice
  - Ages: 3 months
  - Justification: ≤4mo cutoff for young mice.
  - Sample count: 4
- **Old group:** 24-month-old mice
  - Ages: 24 months
  - Justification: ≥18mo cutoff for old mice.
  - Sample count: 4
- **EXCLUDED:** None.

### Impact Assessment
- **Data retained:** 100% of original samples
- **Data excluded:** 0%
- **Meets ≥66% threshold?** ✅ YES
- **Signal strength:** The study already uses a clear young vs. old comparison.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `41467_2019_8831_MOESM5_ESM.xlsx`
- Sheet/tab name: "Proteome"
- File size: 5,214 rows × 36 columns
- Format: Excel (.xlsx)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "Protein IDs" | ✅ MAPPED | UniProt accessions. |
| Protein_Name | "Protein names" | ✅ MAPPED | Direct mapping. |
| Gene_Symbol | "Gene names" | ✅ MAPPED | Direct mapping. |
| Tissue | Constant "Lung" | ✅ MAPPED | Hardcoded per study. |
| Species | Constant "Mus musculus" | ✅ MAPPED | Hardcoded per study. |
| Age | Derived from column names "young_*" / "old_*" | ✅ MAPPED | 3 or 24. |
| Age_Unit | Constant "months" | ✅ MAPPED | Hardcoded per study. |
| Abundance | "young_*" and "old_*" columns | ✅ MAPPED | LFQ intensity. |
| Abundance_Unit | Constant "LFQ_intensity" | ✅ MAPPED | Hardcoded per study. |
| Method | Constant "LC-MS/MS (label-free)" | ✅ MAPPED | Hardcoded per study. |
| Study_ID | Constant "Angelidis_2019" | ✅ MAPPED | Hardcoded. |
| Sample_ID | Column header (e.g., "old_1") | ✅ MAPPED | Composite field. |
| Parsing_Notes | Constructed | ✅ MAPPED | Template generation. |

### Mapping Gaps (if any)

✅ All columns mapped.

## 6. Implementation Notes
- No special handling required. The data is already in a clean format with 2 age groups.
