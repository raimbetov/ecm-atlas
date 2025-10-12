# Caldeira et al. 2017 - Analysis

## 1. Paper Overview
- Title: Matrisome Profiling During Intervertebral Disc Development and Ageing
- PMID: 28912585
- Tissue: Bovine caudal nucleus pulposus
- Species: Bos taurus
- Age groups: Foetus (~7 months gestation), Young (12 months), Old (16–18 years) (Methods p.2)

## 2. Data Files Available
- File: `data_raw/Caldeira et al. - 2017/41598_2017_11960_MOESM2_ESM.xls`
  - Sheet: `1. Proteins` — 81×24 (iTRAQ protein ratios per replicate/pool)
  - Sheets: `4./5. Peptides Quant_batch 1/2` — 7,251×58 / 7,063×58 (reporter ion intensities)
- File: `41598_2017_11960_MOESM3_ESM.xls`
  - Sheet: `1. Proteins` — 104×25 (second iTRAQ experiment)
- File: `41598_2017_11960_MOESM6_ESM.xls`
  - Sheets: `1-3. Protein` — 96–146×11 (fraction-specific ECM extracts)
- File: `41598_2017_11960_MOESM7_ESM.xls`
  - Sheets: `Cluster 1B/2A/2B´` — 4–31×4 (expression clusters)

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Accession Number` | Methods p.3 (ProteinPilot search) | ProteinPilot reports UniProt accessions; required for downstream joins |
| Protein_Name | `Protein Name` | Supplementary header | Human-readable Matrisome protein names |
| Gene_Symbol | Derived from `Accession Name` (e.g., PGCA_BOVIN → ACAN) | Methods p.3 | UniProt mnemonic encodes gene symbol before species suffix |
| Tissue | Constant `Nucleus pulposus` | Abstract p.1 | All measurements from NP tissue |
| Species | Constant `Bos taurus` | Methods p.2 | Samples collected from bovine abattoir |
| Age | Column headers `Foetus`, `Young`, `Old` (+ replicate index) | Methods p.2 | Each column corresponds to an age-defined replicate |
| Age_Unit | `months` (Foetus=7 gestational months, Young=12 months, Old≈204 months) | Methods p.2 | Convert years to months (average 17 years = 204 months) for consistent numeric scale |
| Abundance | Reporter ratio columns (e.g., `Foetus 1`, `Young 3`, `Old 2`) | Methods p.3 | iTRAQ reporter ratios quantify relative abundance per replicate |
| Abundance_Unit | `iTRAQ_ratio` | Methods p.3 | Dimensionless normalized ratios post-ProteinPilot processing |
| Method | `iTRAQ LC-MS/MS` | Methods p.3 | 8-plex iTRAQ workflow described |
| Study_ID | `Caldeira_2017` | Internal | Unique identifier |
| Sample_ID | Compose `{AgeGroup}_{Replicate}` (e.g., `Foetus_1`, `Old_pool`) | Supplementary columns | Captures replicate/pool identity |
| Parsing_Notes | Constructed | — | Document age conversion to months, pool treatment, and reporter channel details |

## 4. Abundance Calculation
- Formula from paper: 8-plex iTRAQ labelling, ProteinPilot search with normalized reporter intensities (Methods p.3).
- Unit: Relative reporter ion ratios (dimensionless) per protein group.
- Already normalized: YES — ProteinPilot applies isotope-bias correction and channel normalization.
- Reasoning: Ratios directly represent fold-changes between age cohorts; additional scaling would distort ProteinPilot-calibrated values.

## 5. Ambiguities & Decisions
- Ambiguity #1: `Accession Name` includes species suffix (e.g., PGCA_BOVIN).
  - Decision: Strip suffix after underscore to recover gene mnemonic before mapping to official gene symbol.
  - Reasoning: Ensures consistent gene naming while preserving bovine-specific identifiers.
- Ambiguity #2: Pool columns (`Pool Foetus`, `Pool Old`).
  - Decision: Treat pools as separate sample entries flagged in `Parsing_Notes` as normalization references.
  - Reasoning: Pools capture averaged channel values used in figures (Results p.5); retaining them maintains traceability.
- Ambiguity #3: Age span for old animals (16–18 years).
  - Decision: Use midpoint (17 years → 204 months) for `Age` and record original range in `Parsing_Notes`.
  - Reasoning: Provides numeric value for analysis while preserving documented range.

## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### EXCLUDED FROM LFQ-FOCUSED ANALYSIS

**Method:** 8-plex iTRAQ LC-MS/MS.
- Method type: Isobaric labeling.
- LFQ compatible: ❌ NO
- Reason for exclusion: Reporter-ion quantification requires TMT/iTRAQ-specific normalization; not directly comparable to LFQ intensities.

**Status:** DEFERRED TO PHASE 3
- Will be analyzed with other isobaric labeling datasets (iTRAQ/TMT) in a later phase.
- Age bin strategy will be set when iTRAQ harmonization is addressed.
- Phase 1 LFQ parsing explicitly omits this study.

**Original Age Groups (for reference):**
- Foetus (~7 gestational months).
- Young (12 months).
- Old (16–18 years).

**Note:** No LFQ age mapping performed here; revisit once TMT/iTRAQ workflows are queued.
