# Chmelova et al. 2023 - Analysis

## 1. Paper Overview
- Title: A view of the genetic and proteomic profile of extracellular matrix molecules in aging and stroke
- PMID: 38045306
- Tissue: Brain (Cortex)
- Species: Mus musculus
- Age groups: Young adult (3-month-old), Aged (18-month-old)

## 2. Data Files Available
- File: Data Sheet 1.XLSX
- Sheet: protein.expression.imputed.new(
- Rows: 17
- Columns: See previous analysis.

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---|---|---|---|
| Protein_ID | Column headers (Gene Symbols) | - | Using Gene Symbols as Protein IDs as no protein IDs are provided. |
| Gene_Symbol | Column headers (Gene Symbols) | - | Gene symbols are provided as column headers. |
| Age | Sample names in the first column | Figure 1 | "3m" = 3 months, "18m" = 18 months. |
| Abundance | Values in the table | Figure 1 | log2Norm.counts. |
| Abundance_Unit | - | Figure 1 | "log2Norm.counts". |
| Tissue | - | Paper Abstract | "Brain" is the tissue of interest. |
| Species | - | Abstract | "Mus musculus" (mice) are the species studied. |
| Method | - | Abstract | "RNA-Seq". |
| Study_ID | - | - | "Chmelova_2023" |
| Sample_ID | Sample names in the first column | - | e.g., "X3m_ctrl_A". |
| Parsing_Notes | - | - | Will be generated during parsing. |

## 4. Abundance Calculation
- Formula from paper: Not specified, but data is log2 normalized.
- Unit: log2Norm.counts
- Already normalized: YES
- Reasoning: The data is provided as log2 normalized counts.

## 5. Ambiguities & Decisions
- Ambiguity #1: No Protein IDs are provided.
  - Decision: I will use the Gene Symbols as Protein IDs.
  - Reasoning: This is the only identifier available for the proteins.
- Ambiguity #2: The data is in a wide format.
  - Decision: I will melt the dataframe to a long format.
  - Reasoning: This is necessary to transform the data into the required schema.