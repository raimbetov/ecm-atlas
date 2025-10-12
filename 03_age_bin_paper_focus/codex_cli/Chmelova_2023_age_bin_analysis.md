# Chmelova 2023 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: NanoLC-MS/MS on Thermo Orbitrap Fusion with MaxQuant label-free quantification pipeline (Section 2.6–2.7, Chmelova et al. 2023).
- LFQ compatible: ✅ YES — Dataset reports LFQ intensities; no isotopic or isobaric labels used.

## 2. Current Age Groups
- Young (3 months) — samples labelled `X3m_*` (ctrl and MCAO replicates across 0, 3, and 7 day time points); 9 total samples.
- Old (18 months) — samples labelled `X18m_*` (ctrl and MCAO replicates); 8 total samples.
- Sample identifiers supplied in first column of `protein.expression.imputed.new(` sheet.

## 3. Species Context
- Species: Mus musculus.
- Lifespan reference: C57BL/6J laboratory mice ~26–30 months.
- Aging cutoffs applied: young ≤4 months, old ≥18 months.

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** All samples with prefix `X3m_`
  - Ages: 3 months.
  - Justification: Falls within ≤4 month young cutoff; includes control and ischemia (MCAO) replicates captured in dataset.
  - Sample count: 9.
- **Old group:** All samples with prefix `X18m_`
  - Ages: 18 months.
  - Justification: Meets ≥18 month old cutoff; captures aging cohort defined in study.
  - Sample count: 8.
- **EXCLUDED:** None (dataset provides only 3 m and 18 m cohorts).

### Impact Assessment
- **Data retained:** 17 / 17 samples = 100% ✅
- **Data excluded:** 0%.
- **Meets ≥66% threshold?** ✅ YES.
- **Signal strength:** Strong age separation; note additional MCAO condition metadata should be tracked to allow filtering to control-only comparisons when desired.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `data_raw/Chmelova et al. - 2023/Data Sheet 1.XLSX`.
- Sheet/tab name: `protein.expression.imputed.new(`.
- File size: 17 rows × 3,830 columns (samples × proteins).
- Format: Excel (.xlsx); matrix is transposed relative to standard schema (samples in rows, genes as columns).

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | Column headers (gene symbols) | ⚠️ DERIVED | Need mapping Gene Symbol → UniProt accession using reference table.
| Protein_Name | External lookup via UniProt given accession | ⚠️ DERIVED | No protein name column provided; must enrich after ID mapping.
| Gene_Symbol | Column headers | ✅ | Gene symbols supplied for each protein feature.
| Tissue | Constant `Brain cortex` | ✅ | Dataset collects cortical tissue punches.
| Species | Constant `Mus musculus` | ✅ | Single-species study.
| Age | Parse sample ID prefix (`X3m`, `X18m`) | ✅ | Map to numeric ages 3 and 18 months.
| Age_Unit | Constant `months` | ✅ | Ages reported in months.
| Abundance | Cell value for given sample × gene | ✅ | LFQ intensity already log2-transformed.
| Abundance_Unit | Constant `log2_LFQ_intensity` | ✅ | Supplementary text states values are LFQ intensities.
| Method | Constant `Label-free LC-MS/MS (MaxQuant LFQ)` | ✅ | Documented workflow.
| Study_ID | Constant `Chmelova_2023` | ✅ | Parser identifier.
| Sample_ID | Value from first column (e.g., `X3m_ctrl_A`) | ✅ | Unique per sample.
| Parsing_Notes | Template capturing condition (`ctrl` vs `MCAO`) and timepoint | ✅ | Derive from sample suffix and encode in notes for downstream filtering.

### Mapping Gaps (if any)
- ⚠️ **Protein_ID & Protein_Name** — Supplementary file lacks UniProt accessions and protein names. Plan to map gene symbols to UniProt/Protein names via external reference prior to parsing to satisfy schema.

## 6. Implementation Notes
- Transpose matrix to long format: sample IDs become rows, gene symbols melt into `Protein_ID`/`Gene_Symbol` columns.
- Capture ischemia status by parsing substrings `_ctrl`, `_MCAO_3d`, `_MCAO_7d`; retain as covariates in `Parsing_Notes`.
- Confirm whether additional metadata file lists donor sex or stroke timepoint to enrich Sample_ID definitions.
- After UniProt mapping, store unresolved symbols explicitly in `Parsing_Notes` for manual review.
