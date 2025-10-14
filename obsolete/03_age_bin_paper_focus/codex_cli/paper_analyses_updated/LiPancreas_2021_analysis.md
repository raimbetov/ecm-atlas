# Li et al. 2021 (Pancreas) - Analysis

## 1. Paper Overview
- Title: Proteome-wide and matrisome-specific alterations during human pancreas development and maturation
- PMID: 32747717
- Tissue: Human pancreas (whole organ; acinar/islet compartments profiled)
- Species: Homo sapiens
- Age groups: Fetal (gestational weeks 18–20), Juvenile (5–16 years), Young adult (21–29 years), Older adult (50–61 years) (Supplementary Data 1; Methods)

## 2. Data Files Available
- File: `data_raw/Li et al. - 2021 | pancreas/41467_2021_21261_MOESM6_ESM.xlsx`
  - Sheet: `Data 3` — 2,066×53 (DiLeu-labeled reporter intensities across 24 samples)
  - Columns: `Accession`, `Description`, reporter columns (`F_7`–`O_68`), QC metrics (`Coverage`, `# Peptides`, etc.)
- File: `41467_2021_21261_MOESM4_ESM.xlsx`
  - Sheet: `Data 1` — donor metadata (IDs `f7`–`f12`, `8`–`68`) with sex, DCD/DBD status, age ranges
- File: `41467_2021_21261_MOESM5_ESM.xlsx`
  - Sheet: `Data 2` — Matrisome gene catalog (3,525×2)
- File: `41467_2021_21261_MOESM7_ESM.xlsx`
  - Sheet: `Data 4` — Differential abundance summaries between stages
- File: `41467_2021_21261_MOESM8_ESM.xlsx`
  - Sheet: `Data 5` — Pathway enrichment results

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Accession` (UniProt) | Methods p.3 (DiLeu LC-MS/MS pipeline) | Reporter rows tied to UniProt IDs |
| Protein_Name | `Description` | Supplementary Data 3 | Provides canonical protein names |
| Gene_Symbol | Derive via UniProt mapping (Accession → Gene) | Figures 4–6 | Gene symbols used for matrisome annotation |
| Tissue | Constant `Pancreas` (optionally annotate compartment if available) | Abstract p.1 | Whole pancreas datasets |
| Species | `Homo sapiens` | Methods p.3 | Human donors |
| Age | Map reporter column prefixes: `F_*`=fetal (gestational weeks listed), `J_*`=juvenile (years), `Y_*`=young adult, `O_*`=older adult | Supplementary Data 1 | Column suffix corresponds to donor ID tied to age group |
| Age_Unit | `weeks` for fetal (store gestational week), `years` for postnatal donors (use group-average or donor-specific when available) | Supplementary Data 1 | Documented units per cohort |
| Abundance | Reporter intensity columns (`F_7`, `J_8`, etc.) | Methods p.3 | DiLeu-labeled reporter channels per sample |
| Abundance_Unit | `log10_DiLeu_intensity` (values ~6–10) | Data 3 inspection + Methods | Intensities exported as log-scaled reporter abundances |
| Method | `12-plex DiLeu isobaric labeling + LC-MS/MS` | Methods p.3 | Paper details DiLeu tagging and LC-MS/MS acquisition |
| Study_ID | `LiPancreas_2021` | Internal | Unique key |
| Sample_ID | Combine group + donor ID (e.g., `F_7`, `J_22`, `Y_31`, `O_54`) | Supplementary Data 1 & Data 3 | Column headers match donor IDs |
| Parsing_Notes | Constructed | — | Record donor age ranges, DCD/DBD status, normalization details |

## 4. Abundance Calculation
- Formula from paper: Two batches of 12-plex DiLeu labeling with a shared reference channel; reporter intensities normalized across sets (Methods p.3).
- Unit: Log-transformed DiLeu reporter intensity (base 10) per protein group.
- Already normalized: YES — Shared reference channel enables cross-batch normalization; authors report quantification of 2,064 proteins across all samples.
- Reasoning: Using provided reporter values preserves relative quantitation; if linear scale needed, convert from log base 10.

## 5. Ambiguities & Decisions
- Ambiguity #1: Postnatal donor ages provided as ranges with single-number IDs.
  - Decision: Map numeric age using donor-specific value when available (e.g., `J_8` corresponds to 8-year donor); when only range provided, use group midpoint and note range in `Parsing_Notes`.
  - Reasoning: Maintains numeric age while documenting uncertainty.
- Ambiguity #2: Identification of reference channel used in normalization.
  - Decision: Inspect supplementary methods/PRIDE entry before coding; add placeholder note to confirm channel assignment.
  - Reasoning: Required to correctly re-scale reporter intensities if re-normalizing.
- Ambiguity #3: Whether intensities are log10 or log2.
  - Decision: Validate by comparing to raw Spectronaut export (values 6–10 suggest log10); document assumption and verify before final aggregation.
  - Reasoning: Ensures consistent scaling across studies.

## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### EXCLUDED FROM LFQ-FOCUSED ANALYSIS

**Method:** 12-plex DiLeu isobaric labeling with LC-MS/MS.
- Method type: Isobaric labeling.
- LFQ compatible: ❌ NO
- Reason for exclusion: Reporter-based quantification (DiLeu) requires different normalization and is not directly comparable to LFQ intensities.

**Status:** DEFERRED TO PHASE 3
- Will be processed alongside other isobaric datasets (TMT/DiLeu) in a later phase.
- Age bin mapping will be developed during that phase when reporter normalization strategy is chosen.

**Original Age Groups (for reference):**
- Fetal (gestational weeks 18–20).
- Juvenile (5–16 years).
- Young adult (21–29 years).
- Older adult (50–61 years).

**Note:** No LFQ age normalization performed in Phase 1; leave for reporter-based workflow integration.
