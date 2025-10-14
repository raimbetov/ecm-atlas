# Chmelova 2023 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: Log2 LFQ
- LFQ compatible: ✅ YES - The task description explicitly states this study is LFQ compatible.

## 2. Current Age Groups
- Young: 3-month-old mice
- Old: 18-month-old mice
- Source: `Data Sheet 1.XLSX`, sheet `protein.expression.imputed.new(`

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
  - Sample count: Not specified in analysis, but not needed for mapping.
- **Old group:** 18-month-old mice
  - Ages: 18 months
  - Justification: ≥18mo cutoff for old mice.
  - Sample count: Not specified in analysis, but not needed for mapping.
- **EXCLUDED:** None.

### Impact Assessment
- **Data retained:** 100% of original samples
- **Data excluded:** 0%
- **Meets ≥66% threshold?** ✅ YES
- **Signal strength:** The study already uses a clear young vs. old comparison.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `Data Sheet 1.XLSX`
- Sheet/tab name: "protein.expression.imputed.new("
- File size: 17 rows
- Format: Excel (.xlsx)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | Column headers (Gene Symbols) | ✅ MAPPED | Using Gene Symbols as Protein IDs as no protein IDs are provided. |
| Protein_Name | Not in file, derive from Gene Symbol | ⚠️ DERIVED | Requires external lookup. |
| Gene_Symbol | Column headers (Gene Symbols) | ✅ MAPPED | Direct mapping. |
| Tissue | Constant "Brain" | ✅ MAPPED | Hardcoded per study. |
| Species | Constant "Mus musculus" | ✅ MAPPED | Hardcoded per study. |
| Age | Sample names in the first column | ✅ MAPPED | "3m" = 3, "18m" = 18. |
| Age_Unit | Constant "months" | ✅ MAPPED | Hardcoded per study. |
| Abundance | Values in the table | ✅ MAPPED | log2Norm.counts. |
| Abundance_Unit | Constant "log2Norm.counts" | ✅ MAPPED | Hardcoded per study. |
| Method | Constant "Log2 LFQ" | ✅ MAPPED | Hardcoded per study. |
| Study_ID | Constant "Chmelova_2023" | ✅ MAPPED | Hardcoded. |
| Sample_ID | Sample names in the first column | ✅ MAPPED | e.g., "X3m_ctrl_A". |
| Parsing_Notes | Constructed | ✅ MAPPED | Template generation. |

### Mapping Gaps (if any)

**⚠️ Gap 1: Protein_Name**
- Problem: Source file doesn't include protein name column.
- Proposed solution:
  - Option A: Map Gene_Symbol → Protein_Name via UniProt API or a local reference file.
  - Option B: Leave Protein_Name as NULL and document in Parsing_Notes.
  - Recommendation: Option A.

## 6. Implementation Notes
- The data is in a wide format and needs to be melted to a long format.
- Gene Symbols are used as Protein IDs.
- Protein Names need to be derived from Gene Symbols.
