# Tsumagari et al. 2023 - Analysis

## 1. Paper Overview
- Title: Proteomic characterization of aging-driven changes in the mouse brain by co-expression network analysis
- PMID: 37875604
- Tissue: Mouse cerebral cortex and hippocampus
- Species: Mus musculus (C57BL/6J male)
- Age groups: 3-month, 15-month, and 24-month mice (n=6 per age) (Materials & Methods)

## 2. Data Files Available
- File: `data_raw/Tsumagari et al. - 2023/41598_2023_45570_MOESM3_ESM.xlsx`
  - Sheet: `expression` — 6,822×32 (cortex TMT intensities `Cx_{age}_{rep}`)
  - Sheet: `Welch's test` — statistical outputs (log2 ratios, q-values)
- File: `41598_2023_45570_MOESM4_ESM.xlsx`
  - Sheet: `expression` — 6,910×32 (hippocampus TMT intensities `Hip_{age}_{rep}`)
  - Sheet: `Welch's test` — cortex vs hippocampus comparisons
- File: `41598_2023_45570_MOESM6_ESM.xlsx`
  - Sheets: `Cx_upregulated`, `Cx_downregulated`, `Hipp_upregulated` — marker lists
- File: `41598_2023_45570_MOESM7_ESM.xlsx`
  - Sheet: `TableS4` — peptide-level data
- File: `41598_2023_45570_MOESM8_ESM.xlsx`
  - Sheet: `TableS5` — transcriptional validation

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `UniProt accession` column | Methods p.2 (MaxQuant search) | UniProt entries supplied alongside gene names |
| Protein_Name | Map via UniProt (if not provided) | Methods | Data tables provide gene; use UniProt reference for names |
| Gene_Symbol | `Gene name` column | Supplementary data | Direct gene symbol for each protein |
| Tissue | Parsed from column prefix (`Cx`=cortex, `Hip`=hippocampus) | Experimental design | Distinguishes the two brain regions |
| Species | `Mus musculus` | Methods | Mouse study |
| Age | Extract numeric age (3, 15, 24) from column name | Materials & Methods | Column naming embeds age in months |
| Age_Unit | `months` | Methods | Age specified in months |
| Abundance | TMT reporter intensity (log2-normalized by MaxQuant) | Methods | `expression` sheets contain normalized intensities |
| Abundance_Unit | `TMTpro_normalized_intensity` | Methods (TMT 11-plex) | MaxQuant Reporter intensity after normalization |
| Method | `TMT 11-plex LC-MS/MS (Orbitrap Fusion Lumos)` | Methods p.2 | Detailed instrument workflow |
| Study_ID | `Tsumagari_2023` | Internal | Unique identifier |
| Sample_ID | Construct `{tissue}_{age}mo_rep{n}` | Column naming (e.g., `Cx_15mo_4`) | Ensures uniqueness and clarity |
| Parsing_Notes | Constructed | — | Document two-batch TMT design, reference channels, Welch's test references |

## 4. Abundance Calculation
- Formula from paper: TMT-11 plex labeling with internal references; MaxQuant reporter intensity normalization using TMT-126 for batch bridging and TMT-131C for technical correction (Methods p.2).
- Unit: Normalized TMT reporter intensity (log-scaled) per protein per replicate.
- Already normalized: YES — MaxQuant normalization plus internal reference channels applied; bridging handled per tissue.
- Reasoning: Use provided normalized intensities; optional log2 conversions already inherent in dataset.

## 5. Ambiguities & Decisions
- Ambiguity #1: `Welch's test` sheets provide statistical summaries; not primary abundance.
  - Decision: Use `expression` sheets for raw intensities; treat `Welch's test` as derived metadata for QC.
  - Reasoning: Maintains measurement-level detail required for unified atlas.
- Ambiguity #2: Column order mixes replicate groups; need mapping to batches.
  - Decision: Capture batch ID from supplemental Table S1 (TMT channel assignment) and store in auxiliary metadata file.
  - Reasoning: Enables batch-aware normalization and bridging.
- Ambiguity #3: Peptide-level tables (TableS4) large; not required for initial atlas.
  - Decision: Reference only if peptide QC needed; do not ingest into main schema.
  - Reasoning: Focus on protein-level quant consistent with other studies.
