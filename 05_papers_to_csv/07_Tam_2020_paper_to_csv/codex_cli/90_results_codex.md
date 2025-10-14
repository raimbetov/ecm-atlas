# Tam 2020 Dataset Analysis (Codex)

## Source File Reconnaissance
- File `data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx` contains sheets `Raw data` (3,157 × 80) and `Sample information` (66 × 6) with header offset by one row in both sheets.
- `Raw data` exposes 66 LFQ intensity columns (`LFQ intensity <disc> <age> <direction> <compartment>`) plus 14 identifier/metric columns (e.g., `T: Majority protein IDs`, `T: Gene names`, `N: Score`). All 3,157 rows have non-null majority protein IDs; 26 protein names are blank but gene symbols are complete.
- LFQ matrix holds 208,362 cells with 159,401 NaNs (76.5%) and zero 0-values, so downstream parsing must drop null abundances and beware of sparse coverage.
- `Sample information` rows enumerate spatial metadata (`Disc level`, `Direction`, `Disc Compartments`) and use lower-case age groups (`aged`, `young`). Profile names include the full LFQ column header strings.

## Metadata Alignment Findings
- Join key is exact string match between `Sample information`.`Profile names` and the LFQ column headers; all 66 columns match with no extras or omissions.
- Per-age distribution: each donor contributes 33 spatial profiles (aged vs young). Per-compartment counts per age: OAF 12, IAF 6, NP 3, IAF/NP transition 12. Disc levels L3/4, L4/5, L5/S1 each supply 11 profiles per age.
- One LFQ column uses `Young` (capital Y); others use `young`. Age tokens in metadata (`aged` vs `young`) must be remapped consistently to `Old`/`Young` for schema compliance.
- NP profiles have `Direction = central`; OAF/IAF/IAF-NP use cardinal directions (L/A/P/R), which can be retained in parsing notes if spatial context is desired later.

## Schema & Transformation Mapping
- Column renames after trimming `T:` prefix: `T: Majority protein IDs` → `Protein_ID`, `T: Protein names` → `Protein_Name`, `T: Gene names` → `Gene_Symbol`.
- Additional static fields for long-format schema: `Tissue = "Intervertebral disc"`, `Species = "Homo sapiens"`, `Method = "Label-free LC-MS/MS (MaxQuant LFQ)"`, `Study_ID = "Tam_2020"`, `Age_Unit = "years"`.
- Long-format construction: melt 66 LFQ columns, split header tokens into `Disc_Level`, `Age_Label`, `Direction`, `Compartment_Raw`; normalize casing (`old`→`Old`, `young/Young`→`Young`). Merge with sample metadata to capture canonical compartment labels and spatial coordinates if needed.
- Wide-format target aligns with existing Randles implementation: one row per protein × tissue compartment with columns `[Protein_ID, Protein_Name, Gene_Symbol, Tissue, Tissue_Compartment, Species, Abundance_Young, Abundance_Old, Method, Study_ID, Canonical_Gene_Symbol, Matrisome_Category, Matrisome_Division, Match_Level, Match_Confidence]` after annotation.
- Dependencies: `references/human_matrisome_v2.csv` for annotation, prior workflow example `05_Randles_paper_to_csv/claude_code/randles_conversion.py` for scripting pattern, and validation criteria enumerated in `00_TASK_TAM_2020_CSV_CONVERSION.md`.

## Aggregation, Annotation, and Z-Score Considerations
- Age binning is binary (16 yr vs 59 yr). Mean-abundance aggregation should occur after converting to long format and filtering by `Age_Bin` + standardized compartment names. Handle 12 `IAF/NP` transition profiles explicitly—decide whether to (a) map them to IAF, (b) keep a fourth compartment, or (c) exclude from wide-format; deliverable list implies the final CSVs require only NP, IAF, OAF, so this needs design consensus.
- Preliminary matrisome coverage check using exact gene-symbol matches yields only 412/3,157 proteins (13.0%) in the human reference list, far below the ≥90% target. Multi-level matching (synonyms, UniProt accessions) will improve slightly but still unlikely to reach 90%; task allows continuation with diagnostics, so plan on producing unmatched reports and flagging low coverage.
- For z-scores, generate three compartment-specific files (`NP`, `IAF`, `OAF`) with separate normalization for Young and Old abundances; apply log2-transform first because LFQ intensities span several orders of magnitude. Ensure rows with insufficient data (e.g., all NaNs after aggregation) are pruned prior to z-score computation.

## Risks, Open Questions, Recommendations
- **Transition handling:** Need explicit rule for 12 `IAF/NP` profiles per age. Aggregating them into IAF will raise IAF replicates to 18 but may blur boundary biology; excluding them contradicts "100% data retention" guidance. Clarify expected compartment mapping before coding.
- **Annotation gap:** Even with synonym/UniProt matching, Tam 2020 likely remains <25% matrisome coverage. Prepare to emit `*_unmatched.csv` and a diagnostic JSON per guideline, and confirm whether non-matrisome proteins should persist in the main export.
- **Sparse intensities:** 76.5% missing LFQ values mean long-format tables will drop the majority of rows. Validate that averaging uses available values only and document the effective replicate count per protein/compartment.
- **Case sensitivity:** Standardize casing for age tokens (`Old`/`Young`) and compartment strings (`NP`, `IAF`, `OAF`) to avoid downstream mismatches.
- **Validation checklist:** Ensure plan covers the 17 QC checks (annotation coverage report, metadata JSON, validation log). Reuse Randles automation template but adjust for Tam-specific metadata fields (disc level, directions).

Recommended execution steps: (1) script Excel parsing with header offsets and column normalization, (2) produce long-format + merged metadata with explicit handling for transition profiles, (3) generate wide-format aggregates and annotation outputs, (4) compute compartmental z-scores post-log2, (5) complete validation artifacts including coverage diagnostics and documentation updates.
