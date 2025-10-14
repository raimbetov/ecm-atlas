# Ouni et al. 2022 - Analysis

## 1. Paper Overview
- Title: Proteome-wide and matrisome-specific atlas of the human ovary computes fertility biomarker candidates and opens the way for precision oncofertility
- PMID: 35341935
- Tissue: Human ovarian cortex (soluble and insoluble ECM fractions)
- Species: Homo sapiens
- Age groups: Prepubertal (mean 7 ± 3 years, n=5), Reproductive age (mean 26 ± 5 years, n=5), Menopausal (mean 59 ± 8 years, n=5) (Methods, Experimental Procedures)

## 2. Data Files Available
- File: `data_raw/Ouni et al. - 2022/Supp Table 3.xlsx`
  - Sheet: `Matrisome Proteins` — 102×33 (normalized TMT reporter intensities for matrisome proteins)
  - Reporter columns: `Q. Norm. of TOT_prepub1-5`, `TOT_repro1-5`, `TOT_meno1-5`
- File: `Supp Table 1.xlsx`
  - Sheet: `Hydroxylation` — PTM-specific abundances split across soluble/insoluble fractions with age group replicates
- File: `Supp Table 2.xlsx`
  - Sheets: `Hippo_*`, `mTOR_*` — curated protein-pathway interaction tables derived from text mining
- File: `Supp Table 4.xlsx`
  - Sheets: `Fig 6 A/B`, `Fig 7` — validation metrics per pathway
- File: `Table 2.xlsx`
  - Sheet: `Sheet1` — qualitative mapping of age-specific biomarkers to pathways

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Accession` (Supp Table 3) | Methods (DC-MaP + TMT labeling) | Thermo output provides UniProt accession numbers |
| Protein_Name | `EntryName` / `EntryGeneSymbol` mapping | Supplementary table header | Supplies protein mnemonic and gene symbol |
| Gene_Symbol | `EntryGeneSymbol` | Supplementary table | Official gene symbols used in pathway analysis |
| Tissue | `Ovary_cortex` | Experimental design | All samples from human ovarian cortex |
| Species | `Homo sapiens` | Methods | Human donors |
| Age | Derive from reporter column group (`prepub`, `repro`, `meno`) | Methods (mean ages) | Map each replicate to cohort mean age (7, 26, 59 years) with SD noted |
| Age_Unit | `years` | Methods | Ages reported in years |
| Abundance | `Q. Norm. of TOT_*` columns | Methods & Supplementary Data | Normalized TMT reporter intensities post-DC-MaP pipeline |
| Abundance_Unit | `TMTpro_normalized_intensity` | Methods (16-plex TMTpro) | Intensities normalized using Thermo Sequest HT outputs |
| Method | `DC-MaP + 16-plex TMTpro LC-MS/MS` | Methods (Experimental Procedures) | Sample prep and quant approach described |
| Study_ID | `Ouni_2022` | Internal | Unique identifier |
| Sample_ID | Compose `{agegroup}_{replicate}` (e.g., `prepub_1`, `repro_4`) | Reporter column naming convention | Maintains replicate provenance |
| Parsing_Notes | Constructed | — | Capture cohort mean±SD ages, soluble/insoluble origin if applicable, normalization pipeline |

## 4. Abundance Calculation
- Formula from paper: Divide-and-conquer matrisome protein (DC-MaP) workflow with 16-plex TMTpro labeling; quantification via Sequest HT and normalized reporter intensities (Methods).
- Unit: Normalized TMT reporter intensity (dimensionless) per protein/per replicate.
- Already normalized: YES — Authors report quantile-like normalization (`Q. Norm.` columns) across replicates and shared reference channels.
- Reasoning: Utilize provided normalized reporter values to preserve relative abundance; additional scaling should respect existing normalization.

## 5. Ambiguities & Decisions
- Ambiguity #1: Reporter intensities aggregated over soluble vs insoluble fractions.
  - Decision: Track fraction source using column prefixes (`soluble`, `insoluble`, `TOT`) and document in `Parsing_Notes` when combining.
  - Reasoning: Ensures clarity when merging PTM-specific sheets with total matrisome table.
- Ambiguity #2: Exact variance within age cohorts (mean ± SD given, individual ages not provided).
  - Decision: Assign cohort mean as numeric age and record full mean ± SD and replicate count per cohort in notes.
  - Reasoning: Supports numeric analyses while acknowledging spread.
- Ambiguity #3: Interaction/pathway sheets (`Hippo_*`, `mTOR_*`) contain literature-mined data not directly measured.
  - Decision: Flag these as annotation resources only; exclude from primary abundance parsing but keep for `Parsing_Notes` citations.
  - Reasoning: Maintains focus on measured proteomic intensities while preserving contextual metadata.

## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### EXCLUDED FROM LFQ-FOCUSED ANALYSIS

**Method:** 16-plex TMTpro with DC-MaP workflow.
- Method type: Isobaric labeling.
- LFQ compatible: ❌ NO
- Reason for exclusion: Quantification relies on TMT reporter intensities; requires dedicated normalization distinct from LFQ pipeline.

**Status:** DEFERRED TO PHASE 3
- Will be revisited with other TMT/iTRAQ studies once reporter harmonization is in scope.
- Age bin decisions will be defined alongside reporter-based normalization procedures.

**Original Age Groups (for reference):**
- Prepubertal (mean 7 ± 3 years, n=5).
- Reproductive age (mean 26 ± 5 years, n=5).
- Menopausal (mean 59 ± 8 years, n=5).

**Note:** No Phase 1 LFQ age mapping; maintain metadata for future TMT parsing.
