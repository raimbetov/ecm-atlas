# Ariosa-Morejon et al. 2021 - Analysis

## 1. Paper Overview
- Title: Age-dependent changes in protein incorporation into collagen-rich tissues of mice by in vivo pulsed SILAC labelling
- PMID: 34581667
- Tissue: Articular cartilage, tibial bone, ventral skin, and plasma
- Species: Mus musculus (C57BL/6J)
- Age groups: Group A (heavy diet weeks 4–7 → harvest at 7 weeks), Group B (heavy weeks 4–7 followed by light chase to week 10), Group C (heavy weeks 12–15), Group D (heavy weeks 42–45) (Figures 1 & Methods, pages 3–7)

## 2. Data Files Available
- File: `data_raw/Ariosa-Morejon et al. - 2021/elife-66635-fig2-data1-v1.xlsx`
  - Sheets: Plasma (173×33), Cartilage (634×42), Bone (712×42), Skin (352×42)
  - Columns: `Ratio H/L A1-3`, `Ratio H/L B1-2`, `% Heavy isotope`, `iBAQ L/H A/B`, matrisome annotations (`Gene names`, ` Protein names`, `Majority protein IDs`)
- File: `elife-66635-fig7-data1-v1.xlsx`
  - Sheets: `Table_S5A_D` (tissue-specific differentially incorporated proteins; 182–600 rows)
- File: `elife-66635-fig7-data2-v1.xlsx`
  - Sheets: `cartilage`, `bone`, `skin`, `plasma` (candidate overlap lists)
- File: `elife-66635-fig6-data1-v1.xlsx`
  - Sheet: `Compiled - final` (68×110; glycoprotein incorporation heatmap source)

## 3. Column Mapping to Schema
| Schema Column | Source Location | Paper Section | Reasoning |
|---------------|----------------|---------------|-----------|
| Protein_ID | `Majority protein IDs` (each tissue sheet) | Methods p.16 | MaxQuant output provides UniProt accessions; per MaxQuant documentation `Majority` holds leading evidence |
| Protein_Name | ` Protein names` (note leading space) | Supplementary Data description | Column contains UniProt protein names used throughout figures |
| Gene_Symbol | `Gene names` | Supplementary Data description | Gene symbols used for STRING/IPA analysis (Results p.8) |
| Tissue | Sheet name (Plasma/Cartilage/Bone/Skin) | Figure 2 source data | Each sheet corresponds to a single tissue |
| Species | `Mus musculus` (constant) | Methods p.5–6 | All cohorts are C57BL/6J mice |
| Age | Derived from group letter in column name (`A1`, `B2`, `C1`, `D2` etc.) | Figure 1 schematic p.3 & text p.5–7 | Group design establishes week-based ages for each letter |
| Age_Unit | `weeks` | Figure 1 caption, p.3 | Ages reported in weeks |
| Abundance | Use `iBAQ H/L {Group}{rep}` columns | Methods p.16 | MaxQuant SILAC workflow outputs iBAQ intensities per heavy/light channel; these quantify incorporation |
| Abundance_Unit | `iBAQ_intensity` | Methods p.16 | iBAQ intensities exported from MaxQuant |
| Method | `In vivo pulsed SILAC + LC-MS/MS` | Abstract p.1; Methods p.15 | Heavy lysine diet with LC-MS/MS readout |
| Study_ID | `AriosaMorejon_2021` | Internal | Unique identifier |
| Sample_ID | Combination `{tissue}_{group}_{rep}` derived from column headers | Figure 1 & dataset columns | Captures tissue, age group, replicate |
| Parsing_Notes | Constructed | — | Store mapping between group letters, week ranges, isotope channel interpretation |

## 4. Abundance Calculation
- Formula from paper: MaxQuant (v1.6) SILAC search with second-peptide and match-between-runs enabled; heavy/light intensities summed to iBAQ (Methods p.16).
- Unit: iBAQ intensity per isotope channel; `Ratio H/L` columns provide heavy-to-light incorporation fractions.
- Already normalized: PARTIAL — iBAQ is normalized for protein length but not across samples; isotopic ratios allow relative normalization across ages.
- Reasoning: Authors interpret `% Heavy isotope` and `Ratio H/L` to assess incorporation (Results p.4–7); raw iBAQ values retain scale for absolute abundance.

## 5. Ambiguities & Decisions
- Ambiguity #1: Whether to represent age as midpoint of week range (e.g., 7 vs 10) or label by cohort (A/B/C/D).
  - Decision: Encode numeric age using harvest week (7, 10, 15, 45) and retain cohort letter in `Sample_ID`.
  - Reasoning: Harvest week is explicit in Figure 1 (p.3); storing numeric ages enables downstream comparisons while `Sample_ID` keeps provenance.
- Ambiguity #2: Choosing between heavy or light iBAQ values for abundance.
  - Decision: Record both channels as separate samples (heavy = newly synthesized, light = pre-existing) with channel captured in `Sample_ID` and `Parsing_Notes`.
  - Reasoning: Paper compares incorporation vs pre-existing pools; preserving both supports ratio recalculation.
- Ambiguity #3: Handling tissues absent in some groups (e.g., plasma lacks `C/D` sheets in certain files).
  - Decision: Use NA placeholder with explanatory `Parsing_Notes` when a group is missing to maintain schema consistency.
  - Reasoning: Maintains complete matrix while documenting source limitations noted in Results p.7–9.
