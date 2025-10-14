# Randles 2021 – CSV Conversion Analysis

## Data Verification
- Input file `data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx` (1.3 MB) exists; target sheet `Human data matrix fraction` confirmed alongside six non-human validation sheets.
- Sheet requires `header=1` to skip a duplicated label row; yields 2,610 protein records × 31 columns (task brief cited 2,611, so one expected row is absent—no aggregate row at tail).
- Primary identifiers (`Gene name`, `Accession`, `Description`) are complete and mutually unique (`Accession` count = 2,610).

## Column Structure Findings
- Metadata columns: `Peptide count`, `Unique peptides`, `Confidence score`, `P value` provide search metrics; `P value` spans 1e-6–1.0, `Confidence score` 15–16,323.
- Intensity columns: 12 totals (`G15…G69`, `T15…T69`) with no nulls, min 0, max 6.0e9; zero counts per sample range 19–82.
- Suffix `.1` columns (12) store peptide detection tallies (0–170+), not binary flags; treat as optional QC metrics rather than presence booleans.
- No negative intensities; 0 rows have all-zero intensities, so every protein detected in at least one compartment/age.

## Sample & Age Metadata
| Sample | Compartment | Age (years) | Age bin |
|--------|-------------|-------------|---------|
| G15 | Glomerular | 15 | Young |
| T15 | Tubulointerstitial | 15 | Young |
| G29 | Glomerular | 29 | Young |
| T29 | Tubulointerstitial | 29 | Young |
| G37 | Glomerular | 37 | Young |
| T37 | Tubulointerstitial | 37 | Young |
| G61 | Glomerular | 61 | Old |
| T61 | Tubulointerstitial | 61 | Old |
| G67 | Glomerular | 67 | Old |
| T67 | Tubulointerstitial | 67 | Old |
| G69 | Glomerular | 69 | Old |
| T69 | Tubulointerstitial | 69 | Old |

- Age bins align with brief: ≤40 → Young, ≥55 → Old, retaining all 12 samples; expected long-format rows = 2,610 × 12 = 31,320 (12 fewer than task assumption owing to row count difference).

## Execution Workflow Outline
1. Reconnaissance: load Excel with `header=1`, assert 12 intensity + 12 detection columns, confirm null-free IDs, log row count variance (2610 vs 2611).
2. Parsing: select identifier + metadata + 12 intensities, melt to long format with fields (`Protein_ID`, `Gene_Symbol`, `Protein_Name`, `Sample_ID`, `Abundance`); keep `.1` columns optionally for QC by pivoting to auxiliary table.
3. Sample enrichment: derive `Compartment`, `Donor_Age`, `Age_Bin`, `Age_Unit='years'`, `Species='Homo sapiens'`, `Tissue='Kidney cortex'`, `Tissue_Subtype` distinguishing `Glomerular` vs `Tubulointerstitial`.
4. Schema mapping: populate standardized 14-column frame (per `01_TASK_DATA_STANDARDIZATION.md`) including `Method='Label-free LC-MS/MS (Progenesis Hi-N + Mascot)'`, `Study_ID='Randles_2021'`, `Sample_ID` pattern `Randles2021_{Compartment}_{Age}`.
5. Matrisome annotation: use `references/human_matrisome_v2.csv`; exact gene-symbol match first, fallback to UniProt accession; log coverage ≥90% target, store unmatched list.
6. Validation: verify row count (31,320), null checks on required columns, check key ECM markers (COL1A1, FN1) present in both compartments, summarize zero intensity distribution, ensure `.1` columns excluded from final export but archived.
7. Export & docs: write `data_processed/Randles_2021_parsed.csv`, metadata JSON with counts + coverage, unmatched proteins CSV if needed, annotation/validation markdown.

## Analysis Needs & Open Decisions
- Confirm final schema field names and ordering against `01_TASK_DATA_STANDARDIZATION.md` (12-column baseline vs 14-column requirement in task brief) and capture any additional required columns (e.g., `Age_Bin`, `Compartment`, `File_Source`).
- Decide retention strategy for peptide-count metrics (`Peptide count`, `Unique peptides`, `.1` columns): exclude from core schema but archive in supplemental QC export or add optional columns per rubric.
- Validate assumption that `P value` column corresponds to Progenesis ANOVA; if unused, document omission.
- Ensure matrisome reference version hash captured (per annotation guidelines) and note any human gene synonyms needed for coverage.

## Next Actions
1. Implement parsing script/notebook with the workflow above, logging structural assertions and saving intermediate QC artifacts.
2. Run annotation pipeline against human matrisome list, capture coverage metrics, and generate unmatched report.
3. Produce validation + metadata docs satisfying 15-point rubric: include row counts, null checks, zero distribution, marker presence, age-bin confirmation, and documentation of `.1` column handling.
4. Review outputs with task owner for confirmation on handling of peptide detection columns and the 2610-row discrepancy before committing results.
