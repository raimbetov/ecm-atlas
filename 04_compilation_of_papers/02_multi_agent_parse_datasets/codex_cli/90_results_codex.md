## Analysis Summary (Codex)
- Repository contains 13 study folders under `data_raw/`; each mapped to one publication year spanning 2017-2023 as required.
- Machine-readable protein tables appear in 10/13 studies (XLSX/XLS/TSV); three studies currently expose only PDF/DOCX artifacts (Lofaro 2021, McCabe 2020, portions of Randles 2021) and will need further investigation or data sourcing.
- Unified 12-column schema aligns with existing repository standards; no processed outputs (`data_processed/`) currently present, so full pipeline must create this structure from scratch.
- Parsing must handle Nature "MOESM" multi-sheet workbooks, eLife "fig*-data" spreadsheets, and DIA exports (`Report_*Pivot.xls`, `Candidates.tsv`).

## Study-Level File Reconnaissance
- **Angelidis et al. 2019 (lung, mouse):** Eleven supplements; protein abundances likely in `MOESM5_ESM.xlsx` (cell-type matrix) and `MOESM8_ESM.xlsx` (bulk). Requires sheet-level inspection to isolate protein IDs vs metadata.
- **Ariosa-Morejon et al. 2021 (lung organoids, human):** Six tidy `elife-66635-fig*-data*-v1.xlsx` files; expect per-figure tabs with standardized headers that encode age/treatment groups.
- **Caldeira et al. 2017 (cardiac, rat):** Four `.xls` workbooks (`MOESM2/3/6/7`); likely intensity sums per ECM component. Must confirm encoding (Excel 97 format) and convert to UTF-8 CSV.
- **Chmelova et al. 2023 (ovarian/aging, mouse):** `Data Sheet 1.XLSX` includes quantitative tables; multiple PDF tables provide context but not directly ingestible.
- **Dipali et al. 2023 (ovary, cow):** Rich DIA export set with `Candidates.tsv` (protein-level) and protein/peptide pivot reports; metadata in `.docx`/`.txt`. Need to map directDIA columns to abundance + Sample_ID.
- **Li et al. 2021 | dermis (skin, human):** Tables 1-4 in XLSX; determine which contain proteomics vs metadata; likely separate sheets for young vs old donors.
- **Li et al. 2021 | pancreas (pancreas, human):** Large MOESM* XLSX files; expect ECM fractions in MOESM4-9; must filter for matrisome subset and capture donor ages.
- **Lofaro et al. 2021 (kidney, human):** Only PDF tables (S1, S2); no obvious XLSX. Need plan for PDF table extraction or locate alternate supplementary datasets.
- **McCabe et al. 2020 (lung, human):** Single `.docx` supplement; likely contains embedded tables requiring conversion via python-docx or manual CSV reconstruction.
- **Ouni et al. 2022 (adipose, human):** Mix of HTML reports (`AGE_Report.htm`) and Excel supplements (Supp Table 1-4, Table 2). Determine if HTML tables = quantitative output; may need BeautifulSoup parsing.
- **Randles et al. 2021 (kidney, mouse/human):** Extensive PDF set plus five XLSX tables (`File023`-`File027`). Need to map which include quantitative proteomics vs metadata; ensure AVI/video files ignored.
- **Tam et al. 2020 (heart, mouse):** Five `elife-64940-supp*-v3.xlsx` files; likely straightforward tidy tables with age groups embedded in headers.
- **Tsumagari et al. 2023 (tissue TBD, mouse):** Eight MOESM XLSX files; identify which hold ECM abundance (likely MOESM2-6). Confirm presence of species/age metadata columns.

## Recommended Next Actions
1. Inventory each XLSX/XLS/TSV file for sheet names and key column headers to map candidate Protein_ID, Abundance, and Sample metadata fields.
2. Flag studies lacking machine-readable tables (Lofaro, McCabe, portions of Randles) and decide whether to apply table extraction (tabula/pandas-read_html) or request alternative sources.
3. Draft configurable parsing module (per study config YAML/JSON) to capture column mappings, units, and metadata defaults before implementing full ETL pipeline.
4. Define validation checklist ensuring each output CSV populates the 12 schema columns and logs missing metadata for follow-up.
