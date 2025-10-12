# Dataset Inventory (Tier 0.1)

All structured data files under `data_raw/` for the 11 target studies. Row and column counts measured from the native files (rounded where appropriate). Only Excel, TSV, CSV, and HTML assets are listed; PDF/DOCX attachments are excluded per requirements.

## Angelidis et al. 2019

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `41467_2019_8831_MOESM5_ESM.xlsx` | Proteome | 5,214 | 36 | Lung proteome LFQ intensities; samples `old_1-4`, `young_1-4` |
| `41467_2019_8831_MOESM4_ESM.xlsx` | Table S1_AllMarkersCelltypes | 17,399 | 8 | Marker gene matrix (transcriptomic reference) |
| `41467_2019_8831_MOESM6_ESM.xlsx` | Sheet1 | 109 | 8 | Protein validation panel |
| `41467_2019_8831_MOESM7_ESM.xlsx` | QDSP | 5,119 | 64 | Quantitative detergent solubility profiling (fractionation) |
| `41467_2019_8831_MOESM8_ESM.xlsx` | Sheet1 | 20,853 | 159 | Single-cell transcriptomic counts (context only) |
| `41467_2019_8831_MOESM9_ESM.xlsx` | 1D annotation enrichments_FDR10 | 1,880 | 10 | Enrichment statistics (context) |

*Primary proteomic source: `MOESM5` sheet `Proteome`.*

## Ariosa-Morejon et al. 2021

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `elife-66635-fig2-data1-v1.xlsx` | Plasma / Cartilage / Bone / Skin | 173 / 634 / 712 / 352 | 33-42 | iBAQ heavy/light ratios for four tissues (A vs B cohorts) |
| `elife-66635-fig4-data1-v1.xlsx` | Sheet1 | 28 | 114 | Matrisome remodeling metrics |
| `elife-66635-fig5-data1-v1.xlsx` | Sheet1 | 22 | 114 | Stable vs fast turnover summary |
| `elife-66635-fig6-data1-v1.xlsx` | Compiled - final | 68 | 110 | KEGG/GO enrichment signals |
| `elife-66635-fig7-data1-v1.xlsx` | Table_S5A/B/C/D | 600 / 561 / 456 / 182 | 16-18 | Top matrisome changes per tissue |
| `elife-66635-fig7-data2-v1.xlsx` | cartilage / bone / skin / plasma | 126 / 242 / 20 / 49 | 7 | Candidate biomarker lists |

*Primary quantitative abundances: `fig2-data1` tissue sheets (iBAQ & H/L ratios).* 

## Caldeira et al. 2017

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `41598_2017_11960_MOESM2_ESM.xls` | 1. Proteins | 81 | 24 | Batch 1 ECM proteome (young vs adult rat)|
| `41598_2017_11960_MOESM2_ESM.xls` | 4./5. Peptides Quant batch 1/2 | 7,251 / 7,063 | 58 | DIA peptide quant tables |
| `41598_2017_11960_MOESM3_ESM.xls` | 1. Proteins | 104 | 25 | Batch 2 ECM proteome (aged vs adult) |
| `41598_2017_11960_MOESM6_ESM.xls` | 1-3. Protein sheets | 146 / 96 / 96 | 11 | Fraction-specific ECM protein abundances |
| `41598_2017_11960_MOESM6_ESM.xls` | 4-6. Peptide sheets | 958 / 854 / 627 | 27 | Fraction peptide quantification |
| `41598_2017_11960_MOESM7_ESM.xls` | Cluster 1B / 2A / 2B´ | 27 / 4 / 31 | 4 | Cluster assignments |

*Primary tables: MOESM2 & MOESM3 `1. Proteins` sheets.*

## Chmelova et al. 2023

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `Data Sheet 1.XLSX` | protein.expression.imputed.new( | 17 | 3,830 | Transposed proteome (samples rows, genes columns; log2) |

## Dipali et al. 2023

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `Candidates.tsv` | - | 4,909 | 37 | DIA candidates merged from native/decell ovary |
| `Candidates_210823_SD7_Native_Ovary_v7_directDIA_v3.xlsx` | SD7_Native_2pept | 4,088 | 33 | Native ovary DIA (>=2 peptides) |
| `` | SD7_Native_2pept Qval / FC | 184 / 184 | 33 | Q-values & fold-change summaries |
| `` | Upregulated / Downregulated Proteins | 107 / 81 | 33 | Significant proteins |
| `Candidates_210915_SD7_Decellularized_Ovary_v8_directDIA.xlsx` | SD7_Decell_2pept | 3,164 | 33 | Decellularized ovary DIA |
| `` | SD7_Decell_2pept_OvsY_Qval / FC | 283 / 283 | 33 | Old vs young contrasts |
| `` | Upregulated / Downregulated Proteins | 129 / 158 | 33 | Significant subsets |
| `Candidates_220526_SD7_Native_Decell_Ovary_overlap.xlsx` | Overlap_Native / Overlap_Decell | 80 / 80 | 33 | Intersections |
| `` | Native&Decell_ratios_overlap | 80 | 19 | Ratio comparison |
| `` | Native_only / Decell_only | 108 / 207 | 33 | Unique proteins |
| `ConditionSetup.tsv` | - | 10 | 10 | Sample metadata |
| `Report_Birgit+Peptide+Quant+Pivot+(Pivot).xls` | (multiple) | — | — | Vendor pivot report (engine TBD) |

*Primary abundance tables: `SD7_Native_2pept` and `SD7_Decell_2pept` sheets; TSVs supply metadata.*

## Li et al. 2021 — Dermis

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `Table 1.xlsx` | Table S1 | 11 | 10 | Donor demographics |
| `Table 2.xlsx` | Table S2 | 266 | 22 | Dermal ECM proteomics (Toddler–Elderly) |
| `Table 3.xlsx` | Table S3 | 265 | 9 | Differentially expressed genes |
| `Table 4.xlsx` | Table S4 | 53 | 9 | ECM remodeling markers |

*Primary proteomic table: `Table 2.xlsx` sheet `Table S2`.*

## Li et al. 2021 — Pancreas

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `41467_2021_21261_MOESM4_ESM.xlsx` | Data 1 | 20 | 12 | Donor metadata |
| `41467_2021_21261_MOESM5_ESM.xlsx` | Data 2 | 3,525 | 2 | Matrisome gene catalog |
| `41467_2021_21261_MOESM6_ESM.xlsx` | Data 3 | 2,066 | 53 | Pancreatic matrisome proteomics (F, J, Y, O samples) |
| `41467_2021_21261_MOESM7_ESM.xlsx` | Data 4 | 179 | 13 | Comparisons across pancreata |
| `41467_2021_21261_MOESM8_ESM.xlsx` | Data 5 | 119 | 51 | ECM pathway enrichment |
| `41467_2021_21261_MOESM9_ESM.xlsx` | Data 6 | 202 | 40 | ECM remodeling statistics |
| `41467_2021_21261_MOESM10_ESM.xlsx` | Data 7 | 63 | 21 | RNA expression (supporting) |
| `41467_2021_21261_MOESM11_ESM.xlsx` | Data 8 | 28 | 7 | Donor list |
| `41467_2021_21261_MOESM13_ESM.xlsx` | Fig. 2 / 3 / 5 / SI tables | 26–2,970 | 11–23 | Supplementary figures |

*Primary proteomic source: `MOESM6` sheet `Data 3` (24 LC-MS/MS samples).* 

## Ouni et al. 2022

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `Supp Table 1.xlsx` | Hydroxylation / AGE | 264 / 5 | 64 / 62 | ECM PTM quantification (MS-based) |
| `Supp Table 2.xlsx` | Hippo_* / mTOR_* sheets | 2–2,666 | 27–30 | Literature-curated signaling interactions |
| `Supp Table 3.xlsx` | Matrisome Proteins / Cellular Proteins | 102 / 845 | 33 / 31 | Proteomic differential tables (LC-MS) |
| `Supp Table 4.xlsx` | Fig 6–7 sheets | 19–241 | 8 | Validation metrics |
| `Table 2.xlsx` | Sheet1 / Repro / Repro&Meno / Meno | 220 / 46 / 46 / 63 | 6–7 | Clinical cohort metadata |

*Primary proteomic outputs: `Supp Table 1` Hydroxylation sheet and `Supp Table 3` Matrisome Proteins.*

## Randles et al. 2021

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `ASN.2020101442-File027.xlsx` | Human data matrix fraction | 2,611 | 31 | Kidney ECM abundances (`G`/`T` columns encode compartment + age) |
| `` | Col4a* matrix fraction sheets | 132–708 | 19–27 | Mouse model validation |
| `ASN.2020101442-File024.xlsx` | Human kidney | 326 | 3 | Signature overlaps |
| `ASN.2020101442-File023.xlsx` | Mouse kidney | 378 | 3 | Reference signatures |
| `ASN.2020101442-File025.xlsx` | Signature proteins | 31 | 7 | ECM biomarker panel |
| `ASN.2020101442-File026.xlsx` | NephroSeq UP and DOWN | 52 | 36 | Gene expression validation |

*Primary proteomic table: `File027` sheet `Human data matrix fraction`.*

## Tam et al. 2020

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `elife-64940-supp1-v3.xlsx` | Raw data | 3,158 | 80 | LFQ intensities across lamina cribrosa regions (`young` vs `old`) |
| `` | Sample information | 67 | 6 | Metadata (disc, age, tissue) |
| `elife-64940-supp2-v3.xlsx` | Sheet1–46 | 4–170 | 8–12 | Region-specific statistics |
| `elife-64940-supp3-v3.xlsx` | Sheet1–50 | 2–343 | 8–12 | Supplemental comparisons |
| `elife-64940-supp4-v3.xlsx` | Sheet1–37 | 5–231 | 8–15 | Additional metrics |
| `elife-64940-supp5-v3.xlsx` | Sheet1 / Sheet2 / Y1–Y4 | 93–285 | 57–103 | Normalized validation datasets |

*Primary quantitative table: `supp1` sheet `Raw data`; other files provide derived metrics.*

## Tsumagari et al. 2023

| File | Sheet/Table | Rows | Columns | Notes |
|------|-------------|------|---------|-------|
| `41598_2023_45570_MOESM3_ESM.xlsx` | expression / Welch's test | 6,822 | 32 / 15 | Hippocampus matrisome proteomics (Cx vs Hipp) |
| `41598_2023_45570_MOESM4_ESM.xlsx` | expression / Welch's test | 6,911 | 32 / 15 | Cerebral cortex matrisome proteomics |
| `41598_2023_45570_MOESM6_ESM.xlsx` | Cx/Hipp up & down | 171–187 | 6 | Differential protein lists |
| `41598_2023_45570_MOESM7_ESM.xlsx` | TableS4 | 5,874 | 25 | ECM peptides |
| `41598_2023_45570_MOESM8_ESM.xlsx` | TableS5 | 830 | 7 | Gene expression validation |
| `41598_2023_45570_MOESM2_ESM.xlsx` | Sheet1 | 44 | 6 | Sample metadata |

*Primary proteomic matrices: `MOESM3` and `MOESM4` expression sheets.*

