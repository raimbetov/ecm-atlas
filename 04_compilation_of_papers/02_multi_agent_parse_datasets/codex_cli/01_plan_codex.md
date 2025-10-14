## My Approach (Codex)
- [x] Step 1: Analyze data_raw/ directory structure
- [ ] Step 2: Outline parsing architecture and schema mapping strategy
- [ ] Step 3: Prototype parsing on representative studies (Angelidis 2019, Dipali 2023)
- [ ] Step 4: Generalize parsing for all 13 studies
- [ ] Step 5: Define validation outputs (metadata.json, validation report)

## Progress Updates
✅ 2025-10-12 11:38:27 PDT Step 1 complete: Located 13 study directories and cataloged primary supplemental files per study.

## Key Findings So Far
- Nature-format studies provide multiple MOESM files; likely protein tables in MOESM4-8 (Angelidis), MOESM2/5 etc. (Li pancreas) requiring sheet inspection.
- eLife datasets (Ariosa-Morejon, Tam) supply tidy XLSX tables (`fig*-data`, `supp*`) with distinct tabs per experiment; column headers encode age groups.
- Several studies include only PDFs/Word docs (Lofaro, McCabe); need contingency plan to extract tabular data or confirm if protein tables exist elsewhere.
- Dipali 2023 offers TSV summaries and DIA reports; `Candidates.tsv` and `Report_*Protein+Quant_Pivot+(Pivot).xls` appear to contain protein-level quant.

## Risks / Questions
- Are there machine-readable protein tables for studies with only PDFs (Lofaro 2021, McCabe 2020)?
- Need consistent mapping for age metadata when only textual descriptors (e.g., "Young" vs numeric months) are provided.
- Should PDF-only sources be processed via OCR or excluded pending confirmation?
