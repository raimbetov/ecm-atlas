# Dipali et al. 2023 - Analysis

## 1. Paper Overview
- Title: Proteomic quantification of native and ECM-enriched mouse ovaries reveals an age-dependent fibro-inflammatory signature
- PMID: 37903013
- Tissue: Ovary
- Species: Mus musculus
- Age groups: Reproductively young (6-12 weeks), Reproductively old (10–12 months)

## 2. Data Files Available
- File: Candidates.tsv
- Sheet: N/A
- Rows: 4909
- Columns: See previous analysis.

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---|---|---|---|
| Protein_ID | "ProteinGroups" | - | UniProt IDs. |
| Gene_Symbol | "Genes" | - | Gene symbols. |
| Age | "Condition Numerator" and "Condition Denominator" | - | "Young_Native" and "Old_Native". |
| Abundance | "AVG Log2 Ratio" | - | Log2 fold change. |
| Abundance_Unit | - | - | "log2_ratio". |
| Tissue | - | Paper Title | "Ovary". |
| Species | - | Abstract | "Mus musculus" (mouse) are the species studied. |
| Method | - | Abstract | "label-free quantitative proteomic methodology". |
| Study_ID | - | - | "Dipali_2023" |
| Sample_ID | - | - | Will be generated during parsing (e.g., "Young_Native_1"). |
| Parsing_Notes | - | - | Will be generated during parsing. |

## 4. Abundance Calculation
- Formula from paper: Not specified, but data is provided as log2 ratio.
- Unit: log2_ratio
- Already normalized: YES
- Reasoning: The data is provided as log2 ratio.

## 5. Ambiguities & Decisions
- Ambiguity #1: The data is already processed as a comparison between two groups.
  - Decision: I will have to extract the data for each group and then calculate the abundance. However, the file only provides the ratio. I will use the log2 ratio as the abundance for the "Old" group and 0 for the "Young" group.
  - Reasoning: This is a workaround to represent the relative abundance between the two groups.
## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### LFQ Method Confirmation
- Method: DIA-NN directDIA label-free quantification (Orbitrap Exploris 480).
- LFQ compatible: ✅ YES
- Included in Phase 1 parsing: ✅ YES (reproductive aging focus)

### Original Age Structure
- Young (6–12 weeks): runs `Y1L`–`Y5L`; decellularized ovary ECM replicates (5 samples).
- Old (10–12 months): runs `O1L`–`O5L`; decellularized ovary ECM replicates (5 samples).
- Intensities drawn from `*.PG.Quantity` columns in pivot TSV export.

### Normalized to Young vs Old (Conservative Approach)
- **Young:** `Y1L`–`Y5L`
  - Ages: 6–12 weeks (~1.5–3 months).
  - Rationale: Within ≤4-month cutoff; reproductive baseline.
  - Sample count: 5.
- **Old:** `O1L`–`O5L`
  - Ages: 10–12 months.
  - Rationale: Defined by authors as reproductive old; note this is <18-month whole-body cutoff but aligns with ovarian senescence window.
  - Sample count: 5.
- **EXCLUDED:** None.
  - Data loss: 0%.

### Impact on Parsing
- Column mapping: melt `.PG.Quantity` columns, map run IDs to age via `ConditionSetup.tsv`.
- Expected row count: ~3,903 proteins × 10 samples (before filtering by matrisome).
- Data retention: 100% (meets ≥66% threshold: YES).
- Signal quality: Pronounced reproductive aging signatures; document age-cutoff deviation in parser notes for cross-study comparisons.
