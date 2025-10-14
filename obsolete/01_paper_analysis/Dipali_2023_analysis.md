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