# Chmelova et al. 2023 - Analysis

## 1. Paper Overview
- Title: A view of the genetic and proteomic profile of extracellular matrix molecules in aging and stroke
- PMID: 38045306
- Tissue: Brain (Cortex)
- Species: Mus musculus
- Age groups: Young adult (3-month-old), Aged (18-month-old)

## 2. Data Files Available
- File: Data Sheet 1.XLSX
- Sheet: protein.expression.imputed.new(
- Rows: 17
- Columns: See previous analysis.

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---|---|---|---|
| Protein_ID | Column headers (Gene Symbols) | - | Using Gene Symbols as Protein IDs as no protein IDs are provided. |
| Gene_Symbol | Column headers (Gene Symbols) | - | Gene symbols are provided as column headers. |
| Age | Sample names in the first column | Figure 1 | "3m" = 3 months, "18m" = 18 months. |
| Abundance | Values in the table | Figure 1 | log2Norm.counts. |
| Abundance_Unit | - | Figure 1 | "log2Norm.counts". |
| Tissue | - | Paper Abstract | "Brain" is the tissue of interest. |
| Species | - | Abstract | "Mus musculus" (mice) are the species studied. |
| Method | - | Abstract | "RNA-Seq". |
| Study_ID | - | - | "Chmelova_2023" |
| Sample_ID | Sample names in the first column | - | e.g., "X3m_ctrl_A". |
| Parsing_Notes | - | - | Will be generated during parsing. |

## 4. Abundance Calculation
- Formula from paper: Not specified, but data is log2 normalized.
- Unit: log2Norm.counts
- Already normalized: YES
- Reasoning: The data is provided as log2 normalized counts.

## 5. Ambiguities & Decisions
- Ambiguity #1: No Protein IDs are provided.
  - Decision: I will use the Gene Symbols as Protein IDs.
  - Reasoning: This is the only identifier available for the proteins.
- Ambiguity #2: The data is in a wide format.
  - Decision: I will melt the dataframe to a long format.
  - Reasoning: This is necessary to transform the data into the required schema.
## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### LFQ Method Confirmation
- Method: NanoLC-MS/MS with MaxQuant label-free quantification (LFQ intensities).
- LFQ compatible: ✅ YES
- Included in Phase 1 parsing: ✅ YES

### Original Age Structure
- 3-month cohort: `X3m_ctrl_A/B/C`, `X3m_MCAO_3d_A/B/C`, `X3m_MCAO_7d_A/B/C` (9 samples).
- 18-month cohort: `X18m_ctrl_A/B`, `X18m_MCAO_3d_A/B/D`, `X18m_MCAO_7d_A/B/C` (8 samples).
- Values reported as log2 LFQ intensities in Supplementary Data Sheet 1.

### Normalized to Young vs Old (Conservative Approach)
- **Young:** All samples prefixed `X3m_`
  - Ages: 3 months.
  - Rationale: Meets ≤4-month mouse young cutoff; retains control and ischemia replicates for ≥66% retention.
  - Sample count: 9.
- **Old:** All samples prefixed `X18m_`
  - Ages: 18 months.
  - Rationale: Meets ≥18-month old cutoff.
  - Sample count: 8.
- **EXCLUDED:** None.
  - Data loss: 0%.

### Impact on Parsing
- Column mapping: transpose matrix and melt sample IDs (`X3m_*`, `X18m_*`) with gene columns; attach condition metadata (`ctrl`, `MCAO_3d`, `MCAO_7d`).
- Expected row count: ~3,830 proteins × 17 samples.
- Data retention: 100% (meets ≥66% threshold: YES).
- Signal quality: Strong age contrast; downstream models should allow covariate for ischemia status.
