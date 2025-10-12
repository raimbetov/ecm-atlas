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

## 6. Age Bin Normalization Strategy (Added 2025-10-12)

### EXCLUDED FROM LFQ-FOCUSED ANALYSIS

**Method:** RNA-Seq
- Method type: Transcriptomics (NOT proteomics)
- LFQ compatible: ❌ NO
- Reason for exclusion: This is transcriptomic data (RNA-Seq), not proteomic data; outside scope of protein-focused ECM atlas

**Status:** EXCLUDED PERMANENTLY
- This study uses RNA-Seq (transcriptomics), not mass spectrometry proteomics
- Data represents gene expression (mRNA levels), not protein abundance
- Will NOT be included in proteomic ECM atlas (any phase)
- Marked as non-proteomic study for reference purposes only

**Original Age Groups (for reference):**
- Young: 3 months
- Old: 18 months
- Species: Mus musculus (mouse)
- Tissue: Brain (Cortex)

**Note:** No age bin mapping performed. This study is excluded from all phases of the proteomic ECM atlas due to incompatible data type (transcriptomics vs proteomics).