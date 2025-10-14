# Li et al. 2021 (Dermis) - Analysis

## 1. Paper Overview
- Title: Time-Resolved Extracellular Matrix Atlas of the Developing Human Skin Dermis
- PMID: 34901026
- Tissue: Human skin dermis (decellularized scaffold)
- Species: Homo sapiens
- Age groups: Toddler (1–3 years), Teenager (8–20 years), Adult (30–50 years), Elderly (>60 years) (Supplementary Table S1)

## 2. Data Files Available
- File: `data_raw/Li et al. - 2021 | dermis/Table 2.xlsx`
  - Sheet: `Table S2` — 266×22 (log2 normalized protein intensities per donor group)
  - Columns: `Protein ID`, `Gene symbol`, `Division`, `Category`, sample columns `Toddler-Sample1`...`Elderly-Sample3`, averaged columns (`Ave_*`), trend annotations
- File: `Table 1.xlsx`
  - Sheet: `Table S1` — 10×10 (donor demographics and age ranges)
- File: `Table 3.xlsx`
  - Sheet: `Table S3` — 265×9 (differential expression summary)
- File: `Table 4.xlsx`
  - Sheet: `Table S4` — 53×9 (highlighted biomarkers by age)

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Protein ID` (UniProt accession) | Supplementary Table S2 header | Required canonical identifier for ECM atlas |
| Protein_Name | Map via UniProt lookup or `Protein Description` (Table S3) | Figure 1 caption (workflow) | Protein names referenced in figures/tables |
| Gene_Symbol | `Gene symbol` column | Supplementary Table S2 | Provides official gene symbols used in analysis |
| Tissue | Constant `Skin dermis` | Figure 1A (workflow) | All samples derived from dermal scaffolds |
| Species | `Homo sapiens` | Methods/Donor demographics | Human donors only |
| Age | Derived from sample column group; assign midpoint ages (Toddler=2yr, Teenager=14yr, Adult=40yr, Elderly=65yr) | Supplementary Table S1 | Age buckets reported as ranges |
| Age_Unit | `years` | Supplementary Table S1 | Age ranges in years |
| Abundance | Individual sample columns (`Toddler-Sample1`, etc.) | Figure 5B (heatmap of log2 normalized intensity) | Numeric values correspond to log2-normalized intensities |
| Abundance_Unit | `log2_normalized_intensity` | Figure 5B caption | Data expressed as log2 normalized protein intensity |
| Method | `LC-MS/MS label-free quantification` | Figure 1A & text (workflow describes mass spectrometry-based quantitative proteomics) | Label-free quantification applied to decellularized dermis |
| Study_ID | `LiDermis_2021` | Internal | Unique identifier |
| Sample_ID | Combine group + sample index (e.g., `Toddler_Sample1`) | Supplementary Table S2 | Maintains replicate provenance |
| Parsing_Notes | Constructed | — | Document midpoint age assumptions, confirm log2 scaling, note decellularization protocol |

## 4. Abundance Calculation
- Formula from paper: Mass spectrometry-based quantitative proteomics with log2 normalization applied to protein intensities (Figure 5B).
- Unit: Log2-normalized intensity (dimensionless) per donor sample.
- Already normalized: YES — Data provided as normalized log2 intensities; averages per age stage are precomputed.
- Reasoning: Authors compare abundance trends via heatmaps and average columns; using supplied normalized values retains their analytical frame.

## 5. Ambiguities & Decisions
- Ambiguity #1: Precise mass-spectrometry platform/normalization parameters not captured in PDF text extraction.
  - Decision: Note requirement to inspect supplementary methods or PRIDE entry (if available) before coding; placeholder flagged in `Parsing_Notes`.
  - Reasoning: Ensures future implementers verify instrument settings when loading raw data.
- Ambiguity #2: Age represented as categorical bins with ranges.
  - Decision: Use midpoint ages for numeric analysis and preserve original ranges in notes.
  - Reasoning: Enables numeric modeling while maintaining interpretability.
- Ambiguity #3: Duplicate columns `Unnamed: 19-21` carrying trend annotations.
  - Decision: Map `Expression trend` and `P Value` to auxiliary metadata table rather than main abundance frame.
  - Reasoning: Keeps abundance matrix tidy while retaining statistical context.
