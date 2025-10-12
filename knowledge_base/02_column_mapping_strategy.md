# Column Mapping Strategy (Tier 0.2)

This document defines how each source dataset will be transformed into the unified ECM Atlas schema (`Protein_ID, Protein_Name, Gene_Symbol, Tissue, Species, Age, Age_Unit, Abundance, Abundance_Unit, Method, Study_ID, Sample_ID, Parsing_Notes`). Strategies incorporate per-study nuances captured in the analysis files.

## Common Principles
- **UniProt-first:** When multiple accessions are present (semicolon-separated), retain the first canonical UniProt ID; record additional IDs in `Parsing_Notes`.
- **Name & Gene hygiene:** Trim whitespace, standardize case, and map via UniProt/Matrisome references when raw tables only provide gene symbols (e.g., Chmelova 2023).
- **Sample granularity:** Each measured column (replicate, channel, or fraction) becomes one row with explicit `Sample_ID`. Multi-channel datasets (SILAC/TMT) yield separate rows per channel with metadata describing channel semantics.
- **Age metadata:** Convert categorical ages to numeric values with documented assumptions (midpoints, means). Preserve original descriptors in `Parsing_Notes`.
- **Abundance units:** Keep native measurement units (LFQ, iBAQ, TMT intensity, iTRAQ ratio). No unit homogenization before Tier 3; capture unit labels explicitly.

## Column-by-Column Mapping

| Schema Column | Strategy | Notes |
|---------------|----------|-------|
| `Protein_ID` | Extract from accession column per study. Use UniProt if available; otherwise derive via mapping (e.g., Chmelova gene → UniProt). | Angelidis `Protein IDs`; Ariosa `Majority protein IDs`; Caldeira `Accession Number`; Dipali `UniProtIds`; Li dermis/pancreas primary accession fields; Ouni `Accession`; Randles `Accession`; Tam `T: Majority protein IDs`. |
| `Protein_Name` | Use provided protein/description column; if absent, map from UniProt. | Standardize capitalization, remove prefixes like `T:`. |
| `Gene_Symbol` | Direct column where supplied; else map from mnemonic/gene column or reference table. | Caldeira requires trimming `_BOVIN`; Chmelova uses gene headers; Tam uses `T: Gene names`. |
| `Tissue` | Assign constant per study or derive from sample metadata. | Multi-tissue studies (Ariosa, Li pancreas, Tam) require values from sample descriptors (e.g., cartilage vs bone, NP vs OAF). |
| `Species` | Hard-coded per dataset (mouse vs human). | Angelidis/Ariosa/Chmelova/Dipali (mouse vs human) etc. |
| `Age` | Parse numeric value from sample metadata. For groups, convert to numeric (e.g., Ariosa group A = 7 weeks; Dipali young = 2.25 months). | Document conversion logic in `Parsing_Notes` including ranges/SD. |
| `Age_Unit` | Use natural unit per study (weeks for Ariosa, months for Dipali Caldeira (converted to months), years for human datasets, gestational weeks for fetal pancreas). | Mixed-unit studies store per-row units. |
| `Abundance` | Use the numeric measurement column for each sample/channel. Do not pre-aggregate. | Handle special cases: Ariosa heavy vs light iBAQ, Caldeira iTRAQ ratios, Dipali directDIA intensities, Li dermis log2 normalized values, Li pancreas DiLeu intensities, Ouni TMT normalized intensities, Randles Progenesis Hi-N intensities, Tam LFQ intensities, Chmelova log2 LFQ. |
| `Abundance_Unit` | Populate with specific measurement type (`LFQ_intensity`, `iBAQ_intensity`, `iTRAQ_ratio`, `directDIA_intensity`, `log2_normalized_intensity`, `TMTpro_normalized_intensity`, etc.). | Maintain consistent vocabulary for repeated units to facilitate downstream grouping. |
| `Method` | Capture experimental mode (e.g., `Label-free LC-MS/MS`, `In vivo SILAC`, `iTRAQ LC-MS/MS`, `DIA LC-MS/MS`). | Extend with fraction details if relevant (e.g., Ouni soluble vs insoluble). |
| `Study_ID` | Assign stable identifier per study (e.g., `Angelidis_2019`, `AriosaMorejon_2021`, `Caldeira_2017`, etc.). | Used for joins with metadata. |
| `Sample_ID` | Compose structured ID describing tissue, age cohort, replicate, and channel. | Examples: `Lung_old_1`; `Cartilage_A_H1`; `NP_old_L3L_A`; `prepub_tot_3`. Ensure uniqueness. |
| `Parsing_Notes` | Free-text traceability string summarizing column source, age justification, transformations, and references to paper sections. | Template snippet per study to ensure age & abundance provenance recorded. |

## Study-Specific Considerations

- **Angelidis 2019:** Map `old_*`/`young_*` columns to age 24 vs 3 months. Age unit `months`. Mention MaxQuant label-free pipeline (Methods p.14) in notes.
- **Ariosa-Morejon 2021:** Each tissue sheet yields heavy (`iBAQ H`) and light (`iBAQ L`) channels. Store age as harvest week (7, 10, 15, 45). Include isotope channel in `Sample_ID` (e.g., `Cartilage_A_H1`).
- **Caldeira 2017:** Convert ages to months (Foetus=7 gestational months; Young=12 months; Old=204 months). Retain pool columns with flagged notes (`Pool Foetus`).
- **Chmelova 2023:** Transpose matrix to long format. Map gene symbols using mouse matrisome reference to obtain UniProt & protein names. Age = 3 vs 18 months; include condition (ctrl vs MCAO D3/D7) in notes.
- **Dipali 2023:** Condition metadata from `ConditionSetup.tsv` drives sample parsing (Native vs ECM enriched, Young vs Old). `directDIA` intensities appear in sheets lacking headers; reconstruct column names from metadata. Age assigned as 2.25 vs 11 months (record range 6–12 wks, 10–12 m).
- **Li dermis 2021:** Skip first header rows; rename `Unnamed` columns. Age groups -> numeric midpoints (Toddler 2, Teenager 14, Adult 40, Elderly 65). Values are `log2Normalized` intensities; note in `Abundance_Unit`.
- **Li pancreas 2021:** Use `Data 3` reporter columns. Map `F_7` etc to donor-specific ages from supplementary metadata (`f7`=20 gestational weeks; `J_22`=11 years etc.). For fetal samples, set `Age_Unit=gestational_weeks`; for postnatal, `years`.
- **Ouni 2022:** Use `Q. Norm. of TOT_*` columns. Age equals cohort means (7, 26, 59). Distinguish soluble vs insoluble vs total intensities based on column prefix; include in sample ID and notes.
- **Randles 2021:** Remove `.1` boolean columns; treat `Gxx` and `Txx` as separate tissues (glomerular vs tubulointerstitial) with age embedded in column name.
- **Tam 2020:** Column names encode disc level, age group, direction, compartment. Parse via sample information sheet. Age = 16 vs 59 (cadaver atlas). Supplementary cohorts flagged separately in metadata tables.

## Metadata Augmentation
- Build lookup tables for:
  - **Age mapping:** Source-specific dictionary {column → (age value, unit, source reference)}.
  - **Tissue mapping:** Standardized names (e.g., `NP`, `IAF` → descriptive string).
  - **Method tags:** Controlled vocabulary for `Method` column to avoid duplicates.
  - **Study descriptors:** Title, DOI, PMID for quick reference when generating `Parsing_Notes`.

Implementations should load these lookup tables via configuration (e.g., YAML/JSON) to keep parsing code declarative.
