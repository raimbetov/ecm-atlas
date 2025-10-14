# Tam et al. 2020 - Analysis

## 1. Paper Overview
- Title: DIPPER, a spatiotemporal proteomics atlas of human intervertebral discs for exploring ageing and degeneration dynamics
- PMID: 33382035
- Tissue: Human lumbar intervertebral discs (nucleus pulposus, inner/outer annulus fibrosus, transition zones)
- Species: Homo sapiens
- Age groups: Young cadaveric spine (16-year male) vs aged cadaveric spine (59-year male); additional surgical samples spanning teens (14–20 years) and adults (47–68 years) for validation (Table 1)

## 2. Data Files Available
- File: `data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx`
  - Sheet: `Raw data` — 3,158×80 (LFQ intensities for 66 spatial profiles across compartments/coordinates)
  - Sheet: `Sample information` — metadata (profile name, disc level, age-group, direction, compartment)
- File: `elife-64940-supp2-v3.xlsx` to `elife-64940-supp5-v3.xlsx`
  - Multiple sheets (statistics, functional annotations, degradome, SILAC incorporation) providing derived metrics

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `T: Majority protein IDs` (Raw data) | Methods p.29 (MaxQuant LFQ) | MaxQuant output with UniProt accessions |
| Protein_Name | `T: Protein names` | Supplementary data | Standard protein annotations |
| Gene_Symbol | `T: Gene names` | Supplementary data | Required for matrisome classification |
| Tissue | Derived from `Sample information` (compartment: NP, NP/IAF, IAF, OAF) | Methods p.5, Fig.1 | Each profile corresponds to a disc region |
| Species | `Homo sapiens` | Methods | Human cadaveric and surgical discs |
| Age | Map age-group + disc level using metadata (e.g., `old` = 59-year cadaver, `young` = 16-year cadaver); surgical validation cohorts supplied in Table 1 | Table 1 | Supports age annotation per profile |
| Age_Unit | `years` | Table 1 | Ages reported in years |
| Abundance | `LFQ intensity ...` columns (one per region/direction) | Methods p.29 | Label-free normalized intensities |
| Abundance_Unit | `LFQ_intensity` | Methods p.29 | MaxQuant LFQ outputs |
| Method | `Label-free LC-MS/MS (MaxQuant LFQ)` | Methods p.29 | Raw data generated via MaxQuant with LFQ option |
| Study_ID | `Tam_2020` | Internal | Unique identifier |
| Sample_ID | Use profile name (e.g., `L3/4_old_L_OAF`) | Sample information sheet | Encodes disc level, age, direction, compartment |
| Parsing_Notes | Constructed | — | Capture mapping from profile to cadaver ID, direction axes, replicate details |

## 4. Abundance Calculation
- Formula from paper: Trypsin-digested proteins analyzed by LC-MS/MS; MaxQuant LFQ intensities computed with default normalization (Methods p.29).
- Unit: LFQ intensity (log2 in downstream analyses; raw column values are linear).
- Already normalized: YES — MaxQuant LFQ applies run-wise normalization; authors further log2-transform for statistics (Methods p.31 for DEG criteria).
- Reasoning: Use raw LFQ intensities for quantitative matrix; apply log2 as needed while documenting transformation in parsing notes.

## 5. Ambiguities & Decisions
- Ambiguity #1: Profiles labeled `young`/`old` correspond to single cadaver donors, while supplementary datasets include broader patient ages.
  - Decision: Tag cadaver-derived LFQ profiles with exact ages (16, 59) and note that derived datasets incorporate additional cohorts; maintain separate metadata table.
  - Reasoning: Preserves clarity between atlas backbone and validation samples.
- Ambiguity #2: `Raw data` columns prefixed with `T:` include directional naming; additional sheets reuse similar naming without prefix.
  - Decision: Strip `T:` prefix during parsing but retain full descriptor in `Sample_ID` for traceability.
  - Reasoning: Simplifies downstream joins while preserving human-readable metadata.
- Ambiguity #3: Multiple supplementary workbooks provide high-level statistics (Supp2–Supp5).
  - Decision: Treat these as secondary derivatives; focus parsing on `supp1` for primary LFQ intensities, while capturing references to degenerative cohorts in notes.
  - Reasoning: Keeps Tier 1 dataset manageable and aligns with requirement for raw intensity-based atlas.
