# Ariosa-Morejon et al. 2021 - Analysis

## 1. Paper Overview
- Title: Age-dependent changes in protein incorporation into collagen-rich tissues of mice by in vivo pulsed SILAC labelling
- PMID: 34581396
- Tissue: Plasma, Cartilage, Bone, Skin
- Species: Mus musculus
- Age groups: 4-7 weeks, 12-15 weeks, 42-45 weeks

## 2. Data Files Available
- File: elife-66635-fig2-data1-v1.xlsx
- Sheet: Plasma, Cartilage, Bone, Skin
- Rows: 173 (Plasma), 634 (Cartilage), 712 (Bone), 352 (Skin)
- Columns: See previous analysis.
- File: elife-66635-fig4-data1-v1.xlsx
- File: elife-66635-fig5-data1-v1.xlsx
- File: elife-66635-fig6-data1-v1.xlsx
- File: elife-66635-fig7-data1-v1.xlsx
- File: elife-66635-fig7-data2-v1.xlsx

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---|---|---|---|
| Protein_ID | "Majority protein IDs" | - | UniProt IDs. |
| Gene_Symbol | "Gene names" | - | Gene symbols corresponding to the protein IDs. |
| Age | Age groups from paper | Figure 1 | 4-7 weeks, 12-15 weeks, 42-45 weeks. |
| Abundance | "% Heavy isotope A", "% Heavy isotope B", etc. | Figure 2 | Percentage of heavy isotope incorporation. |
| Abundance_Unit | - | Figure 2 | "% heavy isotope". |
| Tissue | Sheet names | - | "Plasma", "Cartilage", "Bone", "Skin". |
| Species | - | Abstract | "Mus musculus" (mice) are the species studied. |
| Method | - | Abstract | "in vivo pulsed SILAC labelling". |
| Study_ID | - | - | "Ariosa-Morejon_2021" |
| Sample_ID | - | - | Will be generated during parsing (e.g., "4-7_weeks_1"). |
| Parsing_Notes | - | - | Will be generated during parsing. |

## 4. Abundance Calculation
- Formula from paper: (H/L/((H/L) + 1)) * 100
- Unit: % heavy isotope
- Already normalized: YES
- Reasoning: The paper provides the formula for calculating the percentage of heavy isotope incorporation.

## 5. Ambiguities & Decisions
- Ambiguity #1: Data is spread across multiple files and sheets.
  - Decision: I will need to read all the files and sheets and combine them into a single dataframe.
  - Reasoning: This is necessary to get all the data for all age groups and tissues.
- Ambiguity #2: The column names are not consistent across all files.
  - Decision: I will need to manually map the columns for each file.
  - Reasoning: This is necessary to correctly extract the data.
