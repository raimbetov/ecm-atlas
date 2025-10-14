# Li Dermis 2021 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: LC-MS/MS on Orbitrap with label-free log2-normalized intensities (Methods & Supplementary Table S2).
- LFQ compatible: ✅ YES — No labeling; study reports log2 normalized intensities per donor replicate.

## 2. Current Age Groups
- Toddler (2 years; reported range 1–3) — `Toddler-Sample1/2`; 2 replicates.
- Teenager (14 years; range 8–20) — `Teenager-Sample1/2/3`; 3 replicates.
- Adult (40 years; range 30–50) — `Adult-Sample1/2`; 2 replicates.
- Elderly (65 years; >60) — `Elderly-Sample1/2/3`; 3 replicates.
- Sample values located in `Table 2.xlsx`, sheet `Table S2` (log2 normalized intensities).

## 3. Species Context
- Species: Homo sapiens.
- Lifespan reference: ~75–85 years in developed populations.
- Aging cutoffs applied: young ≤30 years, old ≥55 years (user-approved human thresholds).

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** Toddler + Teenager columns
  - Ages: 2 years (Toddler midpoint), 14 years (Teenager midpoint).
  - Justification: Both ≤30-year cutoff; represent developing to adolescent dermis with high ECM synthesis.
  - Sample count: 5 (2 toddler + 3 teenager).
- **Old group:** Elderly columns only
  - Ages: 65 years (reported >60 cohort; midpoint 65 used).
  - Justification: ≥55-year cutoff; captures post-menopausal dermis remodeling.
  - Sample count: 3.
- **EXCLUDED:** Adult group (40 years)
  - Ages: 40-year midpoint.
  - Rationale: Between 30–55 cutoff; ambiguous transitional aging state.
  - Sample count: 2 (20% of 10 samples).

### Impact Assessment
- **Data retained:** 8 / 10 samples = 80% ✅
- **Data excluded:** 20% (Adult group) — meets ≥66% retention rule.
- **Meets ≥66% threshold?** ✅ YES.
- **Signal strength:** Strong young vs old contrast (2–14 vs 65 years) with clear developmental vs geriatric phenotypes.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `data_raw/Li et al. - 2021 | dermis/Table 2.xlsx`.
- Sheet/tab name: `Table S2` (log2 normalized protein intensities).
- File size: 266 rows × 22 columns (after header cleanup).
- Format: Excel (.xlsx) with merged header rows (requires skip of first 2 rows).

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | `Protein ID` column (after header normalization) | ✅ | UniProt accession per protein.
| Protein_Name | External lookup (Table S3 or UniProt) | ⚠️ DERIVED | Supplement lacks explicit protein name; map via accession.
| Gene_Symbol | `Gene symbol` | ✅ | Provided alongside protein ID.
| Tissue | Constant `Skin dermis` | ✅ | All samples dermal ECM scaffolds.
| Species | Constant `Homo sapiens` | ✅ | Human donors only.
| Age | Map from column group (Toddler/Teenager/Elderly) to midpoint ages | ✅ | 2, 14, 65 years; store numeric age per sample.
| Age_Unit | Constant `years` | ✅ | Age ranges reported in years.
| Abundance | Sample-specific columns (`Toddler-Sample1` … `Elderly-Sample3`) | ✅ | Log2 normalized intensities.
| Abundance_Unit | Constant `log2_normalized_intensity` | ✅ | Authors state values are log2 normalized.
| Method | Constant `Label-free LC-MS/MS` | ✅ | Documented in methods workflow.
| Study_ID | Constant `LiDermis_2021` | ✅ | Parser identifier.
| Sample_ID | Compose `{group}_{sample}` (e.g., `Teenager_Sample3`) | ✅ | Derived from column header.
| Parsing_Notes | Template documenting original age ranges, log2 scaling, exclusion of Adult group | ✅ | Provide context during ETL.

### Mapping Gaps (if any)
- ⚠️ **Protein_Name** absent in Table S2; plan to enrich via UniProt lookup or Table S3 before final load.

## 6. Implementation Notes
- Read sheet with `skiprows=2`; row 0 then holds column labels; drop first row after promoting headers.
- Trim whitespace from column names (some include trailing spaces).
- Maintain mapping table for midpoint ages (Toddler=2, Teenager=14, Adult=40, Elderly=65) and document original ranges in `Parsing_Notes`.
- Ensure Adult columns are dropped during load to enforce conservative two-bin strategy; record exclusion count in results metadata.
- Consider pre-fetching UniProt accession → protein name mapping to close `Protein_Name` gap prior to ingestion.
