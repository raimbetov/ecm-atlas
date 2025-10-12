# Angelidis 2019 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: MaxQuant label-free quantification (LFQ intensities) on Thermo Orbitrap Fusion (Methods p.14).
- LFQ compatible: ✅ YES — No isotopic or isobaric labeling; log-transformed LFQ intensities supplied per sample.

## 2. Current Age Groups
- Young (3 months) — columns `young_1` … `young_4`; 4 biological replicates (C57BL/6J) (Methods p.3).
- Old (24 months) — columns `old_1` … `old_4`; 4 biological replicates (C57BL/6J) (Methods p.3).
- Age metadata embedded in column names within `Proteome` sheet of MOESM5.

## 3. Species Context
- Species: Mus musculus.
- Lifespan reference: Laboratory C57BL/6J mice typically 26–30 months.
- Aging cutoffs applied: young ≤4 months, old ≥18 months (user-approved mouse thresholds).

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** `young_1-4`
  - Ages: 3 months (all samples).
  - Justification: Within ≤4 month mouse cutoff; represents young adult baseline.
  - Sample count: 4.
- **Old group:** `old_1-4`
  - Ages: 24 months (all samples).
  - Justification: ≥18 month cutoff; captures geriatric lung proteome.
  - Sample count: 4.
- **EXCLUDED:** None — dataset already 2 age bins.

### Impact Assessment
- **Data retained:** 8 / 8 samples = 100% ✅
- **Data excluded:** 0%.
- **Meets ≥66% threshold?** ✅ YES.
- **Signal strength:** Strong — 21 month gap between cohorts, matching aging phenotype described in paper.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`.
- Sheet/tab name: `Proteome`.
- File size: 5,213 rows × 36 columns.
- Format: Excel (.xlsx) exported from Nature Communications supplement.

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | `Protein IDs` | ✅ | Use leading UniProt accession per row.
| Protein_Name | `Protein names` | ✅ | Canonical UniProt protein label supplied.
| Gene_Symbol | `Gene names` | ✅ | Gene symbols from MaxQuant export.
| Tissue | Constant `Lung` | ✅ | Whole lung homogenates only.
| Species | Constant `Mus musculus` | ✅ | Single species cohort.
| Age | Derive from column prefix `young_`/`old_` | ✅ | Map to numeric ages: 3 vs 24 months.
| Age_Unit | Constant `months` | ✅ | Ages explicitly reported in months.
| Abundance | Sample intensity columns (`young_*`, `old_*`) | ✅ | LFQ intensity per sample.
| Abundance_Unit | Constant `LFQ_intensity` | ✅ | Direct MaxQuant LFQ output.
| Method | Constant `Label-free LC-MS/MS (MaxQuant LFQ)` | ✅ | Documented workflow.
| Study_ID | Constant `Angelidis_2019` | ✅ | Parser identifier.
| Sample_ID | Column header (`young_1`, etc.) | ✅ | Encodes age group + replicate.
| Parsing_Notes | Template string capturing age (3 vs 24 mo) and LFQ context | ✅ | Populate during ETL.

### Mapping Gaps (if any)
- ✅ All columns mapped — no gaps identified.

## 6. Implementation Notes
- Strip the leading descriptive header row before reading (`skiprows=1` or select columns explicitly).
- Ensure `Protein IDs` values with semicolon-separated accessions select the first entry to match canonical UniProt.
- Maintain column order `young_1-4`, `old_1-4` to preserve reproducibility of replicates.
- Consider log2-transforming LFQ intensities downstream (values currently linear); document any transformation in `Parsing_Notes`.
