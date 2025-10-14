# Randles et al. 2021 - Analysis

## 1. Paper Overview
- Title: Identification of an Altered Matrix Signature in Kidney Aging and Disease
- PMID: 34049963
- Tissue: Human kidney cortex (glomerular and tubulointerstitial fractions)
- Species: Homo sapiens
- Age groups: Young donors (15, 29, 37 years) vs aged donors (61, 67, 69 years) (Methods p.3)

## 2. Data Files Available
- File: `data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx`
  - Sheet: `Human data matrix fraction` — 2,611×31 (label-free intensities for glomerular and tubulointerstitial samples `G15-69`, `T15-69`)
  - Additional sheets: `Col4a*` mouse matrix fractions (validation datasets)
- File: `ASN.2020101442-File024.xlsx`
  - Sheet: `Human kidney` — 326×3 (signature protein lists)
- File: `ASN.2020101442-File023.xlsx`
  - Sheet: `Mouse kidney` — 378×3 (mouse signature proteins)
- File: `ASN.2020101442-File026.xlsx`
  - Sheet: `NephroSeq UP and DOWN` — 52×36 (transcript validation data)

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Accession` column | Methods p.5 (Progenesis LC-MS + Mascot) | UniProt accession from Mascot search |
| Protein_Name | `Description` | Supplementary data | Canonical protein name for interpretability |
| Gene_Symbol | `Gene name` | Supplementary data | Needed for matrisome cross-reference |
| Tissue | Derived from column prefix (`G`=glomerular, `T`=tubulointerstitial) | Methods p.3 | Each intensity column corresponds to a compartment |
| Species | `Homo sapiens` | Methods p.3 | Human donor kidneys |
| Age | Numeric value embedded in column name (e.g., `G15` → 15 years) | Methods p.3 | Donor ages explicitly listed |
| Age_Unit | `years` | Methods p.3 | Ages given in years |
| Abundance | Intensity values in `Gxx`/`Txx` columns | Methods p.5 | Hi-N normalized label-free intensities |
| Abundance_Unit | `HiN_LFQ_intensity` | Methods p.5 | Progenesis Hi-N (top-3 peptide) normalization |
| Method | `Label-free LC-MS/MS (Progenesis + Mascot)` | Methods p.5 | Workflow described (alignment, Hi-N quant) |
| Study_ID | `Randles_2021` | Internal | Unique identifier |
| Sample_ID | `{compartment}_{age}` (e.g., `G_15`, `T_61`) | Column naming convention | Distinguishes compartment and donor age |
| Parsing_Notes | Constructed | — | Capture compartment assignment, Hi-N normalization, donor sex if referenced |

## 4. Abundance Calculation
- Formula from paper: Label-free LC-MS/MS processed in Progenesis LC-MS with chromatogram alignment and Hi-N (top-3 peptide) quantification; Mascot used for identification (Methods p.5).
- Unit: Hi-N normalized intensity (linear scale) per protein per sample.
- Already normalized: YES — Progenesis applies inter-run alignment and Hi-N scaling; no additional normalization applied before statistical analysis.
- Reasoning: Using provided intensities preserves authors’ quantitation scheme; optional median scaling can be documented if applied downstream.

## 5. Ambiguities & Decisions
- Ambiguity #1: Duplicate `.1` suffix columns in exported sheet (e.g., `G15.1`).
  - Decision: Retain only primary intensity columns (`G15`, `T15`) and treat `.1` columns as presence/absence flags described in supplementary notes.
  - Reasoning: `.1` columns contain binary detection data rather than quantitative intensities.
- Ambiguity #2: Missing donor sex for one aged kidney (67 years).
  - Decision: Note in `Parsing_Notes` that sex is unspecified; maintain age-based grouping.
  - Reasoning: Ensures transparency without fabricating metadata.
- Ambiguity #3: Mouse validation sheets present but not part of human atlas deliverable.
  - Decision: Exclude mouse sheets from primary parsing but reference them for cross-species validation documentation.
  - Reasoning: Keeps human ECM atlas focused while acknowledging supporting evidence.

## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### LFQ Method Confirmation
- Method: Progenesis Hi-N label-free quantification
- LFQ compatible: ✅ YES
- Included in Phase 1 parsing: ✅ YES

### Original Age Structure
- Young donors: 15, 29, 37 years (3 donors)
- Aged donors: 61, 67, 69 years (3 donors)
- Two compartments per donor: Glomerular (G) and Tubulointerstitial (T)
- Total: 6 young samples (3 donors × 2 compartments), 6 old samples (3 donors × 2 compartments)

### Normalized to Young vs Old (Conservative Approach)
- **Young:** All three young donors
  - Ages: 15, 29, 37 years (all ≤30yr except 37yr which is close)
  - Rationale: All donors ≤40yr, well within pre-aging phase for human kidney
  - Sample count: 6 samples (G15, T15, G29, T29, G37, T37)
- **Old:** All three aged donors
  - Ages: 61, 67, 69 years (all ≥55yr cutoff for human studies)
  - Rationale: All donors ≥55yr cutoff, representing aged kidney pathology
  - Sample count: 6 samples (G61, T61, G67, T67, G69, T69)
- **EXCLUDED:** None
  - Ages: N/A
  - Rationale: Study already has binary young/old design with clear age separation
  - Sample count: 0 samples
  - Data loss: 0%

### Impact on Parsing
- Column mapping: G/T columns for ages 15, 29, 37 → Young bin; G/T columns for ages 61, 67, 69 → Old bin
- Expected row count: 2,611 proteins × 12 samples = 31,332 rows
- Data retention: 100% (meets ≥66% threshold: YES)
- Signal quality: Excellent young/old contrast with minimum 24-year age gap (37→61yr)
