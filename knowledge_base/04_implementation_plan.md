# Implementation Plan (Tier 0.2)

Goal: Build a reproducible parser that ingests all 11 proteomic datasets into the unified ECM Atlas schema with full traceability, validation, and configuration-driven flexibility.

## Architecture Overview
- **Language:** Python 3.11 (align with repo environment).
- **Key libraries:** `pandas`, `numpy`, `openpyxl`, `xlrd` (legacy `.xls`), `pyyaml` for configuration, `loguru` or standard `logging` for structured logs.
- **Project layout:**
  - `config/` per-study YAML files describing file paths, sheet names, column mappings, age rules, and unit metadata.
  - `ecm_parser/`
    - `loaders/` (study-specific readers)
    - `transformers/` (shared normalization & mapping utilities)
    - `validators/`
    - `cli.py` entry point, `config_loader.py`, `logging.py`.
  - `data_processed/` output CSVs (per study + unified master) and `logs/`.

## Workflow Steps
1. **Configuration Loading**
   - Global config enumerates studies, pointing to per-study YAML.
   - Each YAML defines: input files, sheets, data type (Excel/TSV/HTML), column mapping expressions, age derivation rules, abundance unit, method tag, supplemental metadata (e.g., transcript conditions).

2. **Extraction Layer (Loaders)**
   - Generic loader handles Excel/TSV/HTML with `pandas`; supports multi-row headers and metadata stripping via config directives (e.g., `skip_rows`, `header_row`, `transpose`).
   - Study-specific loader subclasses implement bespoke logic (e.g., Dipali Spectronaut header reconstruction, Tam direction parsing).
   - Loader returns tidy long-form dataframe with raw columns retained for audit.

3. **Transformation Layer**
   - Apply column mapping via config definitions (dictionary mapping raw fields to schema). Support lambda expressions or custom functions (e.g., `first_uniprot`, `parse_age_from_header`).
   - Attach metadata columns: `Study_ID`, `Method`, etc.
   - Generate `Sample_ID` by templating fields (e.g., `"{tissue}_{age_label}_{rep}"`).
   - Compose `Parsing_Notes` using f-string templates capturing: column source, age rationale, unit, paper citation. Provide per-study template in config (strings with placeholders).

4. **Validation Layer**
   - Schema validation: ensure required columns present and types correct.
   - Per-study QC:
     - Row counts vs expected (from dataset inventory).
     - Unique `Sample_ID` per row.
     - Check for missing `Protein_ID`/`Abundance` (> threshold triggers warning/log).
   - Global QC:
     - Aggregate stats (protein coverage per study, matrisome overlap).
     - Summaries required for Tier 4 validation report.

5. **Output Generation**
   - Write per-study CSV named `data_processed/{Study_ID}_parsed.csv`.
   - Concatenate to master file `data_processed/ecm_atlas_v2.csv` retaining study provenance.
   - Emit JSON metadata (row counts, timestamp, version hash).

6. **Logging & Reproducibility**
   - Structured logs (INFO/DEBUG) capturing file reads, sheet names, rows processed, unit conversions, exceptions.
   - All parameters sourced from config; no hard-coded paths.
   - Implement CLI flags: `--studies`, `--validate-only`, `--export-master`.

7. **Testing**
   - Unit tests for helper functions (`tests/` directory): e.g., semicolon accession parser, age normalization, `Parsing_Notes` formatting.
   - Snapshot tests for small sample subsets (commit limited rows) to detect regressions.
   - Continuous validation script verifying expected counts from inventory.

## Study-Specific Tasks
- **Angelidis:** Handle multi-sheet workbook; select `Proteome` sheet only. Age mapping dictionary {`old_*`: 24, `young_*`: 3}.
- **Ariosa:** Parse heavy/light columns; restructure to include `channel` dimension. Map group letters to week ages, encode heavy vs light in notes.
- **Caldeira:** Combine multiple `.xls` runs; unify columns; map pools; convert ages to months. Ensure for run duplicates we capture run label.
- **Chmelova:** Transpose matrix; merge with condition annotations (ctrl vs MCAO). Map gene to UniProt using reference CSV.
- **Dipali:** Use `ConditionSetup.tsv` to label columns; manage wide-format spreadsheets lacking headers. Parse both native and ECM datasets; align replicates by mouse ID.
- **Li dermis:** Skip header rows; rename columns; compute age midpoints.
- **Li pancreas:** Merge donor metadata (Data 1) to map `F_7` etc; convert log10 intensities to float.
- **Ouni:** Distinguish soluble/insoluble/TOT columns via regex; map cohort means; incorporate fraction info into `Sample_ID`.
- **Randles:** Remove `.1` flags; decode `G/T` tissues; convert ages to int. Optionally melt additional sheets for mouse validation.
- **Tam:** Use sample info sheet to parse `Profile names` into tokens (`level`, `age-group`, `direction`, `compartment`); map `age-group` to numeric age (16/59). Merge supplementary cohorts as separate modules (SILAC, degradome) flagged accordingly.

## Deliverables & Milestones
- **M1:** Implement configuration schema + loader scaffolding; parse Angelidis & Randles as pilot.
- **M2:** Complete mouse SILAC (Ariosa), iTRAQ (Caldeira), human dermis/pancreas ingestion; update normalization utilities.
- **M3:** Handle complex datasets (Dipali, Tam) with metadata reconstruction; finalize `Parsing_Notes` templating system.
- **M4:** Produce master CSV, run QC summary report (per Success Criteria Tier 4).
- **M5:** Document CLI usage and write README updates; generate Tier 5 validation report (8 sections).

All code changes must be version-controlled with clear commit messages referencing study and module. Endeavor to create reusable parsing primitives to simplify future dataset additions.
