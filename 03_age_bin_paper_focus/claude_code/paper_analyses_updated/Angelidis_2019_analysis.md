# Angelidis et al. 2019 - Analysis

## 1. Paper Overview
- Title: An atlas of the aging lung mapped by single cell transcriptomics and deep tissue proteomics
- PMID: 30814501
- Tissue: Lung (whole tissue homogenate)
- Species: Mus musculus (C57BL/6J cohorts)
- Age groups: Young 3-month mice vs Old 24-month mice (four replicates each; Methods page 3)

## 2. Data Files Available
- File: `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`
  - Sheet: Proteome — 5,214 rows × 36 columns
  - Columns: `Majority protein IDs`, `Protein names`, `Gene names`, `old_1-4`, `young_1-4`, LFQ statistics, pathway annotations
- File: `41467_2019_8831_MOESM7_ESM.xlsx`
  - Sheet: QDSP — 5,119 rows × 64 columns (detergent fractionation profiles)
- File: `41467_2019_8831_MOESM6_ESM.xlsx`
  - Sheet: Sheet1 — 109 rows × 8 columns (ELISA/immunoblot validation panel)

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Protein IDs` (Proteome sheet) | Methods p.14 (MaxQuant output) | MaxQuant label-free pipeline outputs UniProt accessions in `Protein IDs`; select first accession for canonical ID |
| Protein_Name | `Protein names` | Supplementary Data description | Human-readable names aligned with UniProt, required for provenance |
| Gene_Symbol | `Gene names` | Supplementary Data description | MaxQuant gene annotation per protein group |
| Tissue | Constant `Lung` | Abstract p.1; Methods p.14 | Study profiled whole mouse lungs; no other tissues |
| Species | Constant `Mus musculus` | Abstract p.1 | All samples are from mouse cohorts |
| Age | Derived from column names `young_*` / `old_*` | Methods p.3 | Column naming encodes 3 mo vs 24 mo cohorts |
| Age_Unit | `months` | Methods p.3 | Ages explicitly described as months |
| Abundance | Numeric value from sample column (e.g., `old_1`) | Methods p.14 | Columns contain label-free quantification (LFQ) intensities |
| Abundance_Unit | `LFQ_intensity` | Methods p.14 | Label-free quantification performed in MaxQuant |
| Method | `LC-MS/MS (label-free)` | Methods p.14 | Shotgun proteomics with label-free quantification |
| Study_ID | `Angelidis_2019` | Internal convention | Unique identifier for downstream joins |
| Sample_ID | Use column header (e.g., `old_1`) | Methods p.3 | Column encodes cohort + replicate |
| Parsing_Notes | Constructed | — | Capture age mapping (3 vs 24 mo), column origin, LFQ context |

## 4. Abundance Calculation
- Formula from paper: Label-free quantification via MaxQuant with minimum ratio count of 2 (Methods p.14).
- Unit: MaxQuant LFQ intensity (log2 not applied; raw LFQ values).
- Already normalized: YES — MaxQuant LFQ includes internal normalization across runs.
- Reasoning: Methods detail label-free workflow and match-between-runs; LFQ is standardized output, therefore treat as normalized intensity requiring no further scaling before log transforms.

## 5. Ambiguities & Decisions
- Ambiguity #1: `Majority protein IDs` vs `Protein IDs` columns.
  - Decision: Use `Protein IDs` for canonical accession, as it reflects the leading protein in the group per MaxQuant documentation.
  - Reasoning: Methods cite MaxQuant defaults (p.14) where `Protein IDs` captures the primary accession; `Majority` retains grouped evidence.
- Ambiguity #2: Presence of transcriptomics sheets in same workbook.
  - Decision: Restrict parsing to `Proteome` sheet for Tier 1 deliverable.
  - Reasoning: Other sheets are RNA-seq derived and outside proteomic scope required for ECM atlas.
- Ambiguity #3: Determining replicate metadata (sex/strain) per sample.
  - Decision: Encode only age (3 vs 24 months) in parsing notes; additional metadata to be linked separately from Methods if needed.
  - Reasoning: Workbook lacks explicit mapping from replicate IDs to cohort metadata; requires cross-referencing supplementary text beyond current scope.

## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### LFQ Method Confirmation
- Method: MaxQuant LFQ
- LFQ compatible: ✅ YES
- Included in Phase 1 parsing: ✅ YES

### Original Age Structure
- Young: 3 months (4 replicates)
- Old: 24 months (4 replicates)

### Normalized to Young vs Old (Conservative Approach)
- **Young:** 3 months
  - Ages: 3 months
  - Rationale: ≤4mo cutoff for mouse studies
  - Sample count: 4 samples (young_1, young_2, young_3, young_4)
- **Old:** 24 months
  - Ages: 24 months
  - Rationale: ≥18mo cutoff for mouse studies
  - Sample count: 4 samples (old_1, old_2, old_3, old_4)
- **EXCLUDED:** None
  - Ages: N/A
  - Rationale: Study already has binary young/old design
  - Sample count: 0 samples
  - Data loss: 0%

### Impact on Parsing
- Column mapping: Direct 1:1 mapping of young_* and old_* columns to Young/Old bins
- Expected row count: 5,214 proteins × 8 samples = 41,712 rows
- Data retention: 100% (meets ≥66% threshold: YES)
- Signal quality: Excellent young/old contrast with 21-month age gap
