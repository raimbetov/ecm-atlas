# Abundance Transformation Metadata by Codex

**Prepared:** 2025-10-17  
**Author:** Codex CLI agent  
**Scope:** ECM-Atlas proteomics studies (12 datasets)

## Executive Summary
- Completed end-to-end validation of abundance scales across all 12 ECM-Atlas studies. Every dataset now has an explicit source-scale diagnosis, database-state check, and preprocessing action.
- Four studies ship linear intensities from the MS software (Randles_2021, Dipali_2023, Ouni_2022, LiDermis_2021). These require a standardized `log2(x + 1)` transform before any batch correction or z-score recalculation.
- Seven studies already reside on a log2-equivalent scale because the transformation occurred upstream (Angelidis_2019, Tam_2020, Tsumagari_2023, Schuler_2021, Santinha_2024_Human, Santinha_2024_Mouse_DT, Santinha_2024_Mouse_NT). Keep their abundances as-is when aligning cohorts.
- Caldeira_2017 reports iTRAQ ratios rather than absolute intensities. Exclude this dataset from batch-correction pipelines to avoid mixing ratios with intensities.
- After the prescribed fixes, 5,750 rows (61.5%) convert to log2 scale to match the remaining 3,550 rows, and a 43-row ratio dataset stays out of harmonization workflows.

## 1. Study-Level Action Summary

| Study_ID | Publication (Year) | Acquisition & Software | Scale Assessment | Preprocessing Decision |
|----------|--------------------|------------------------|------------------|------------------------|
| Randles_2021 | Randles et al., JASN (2021) | LFQ, Progenesis QI v4.2 | Source linear Hi-N intensities; DB retains linear medians (~9.6 × 10³) | Apply `log2(x + 1)` (§2.1) |
| Dipali_2023 | Dipali et al., Proteomics (2023) | DIA-NN v1.8, directDIA | Source linear `.PG.Quantity` values; DB shows 10⁵-10⁸ range | Apply `log2(x + 1)` (§2.2) |
| Ouni_2022 | Ouni et al., iScience (2022) | TMTpro 16-plex, Proteome Discoverer 2.4 | Quantile-normalized reporter intensities remain linear (median ~1.55 × 10²) | Apply `log2(x + 1)` (§2.3) |
| LiDermis_2021 | Li et al., Nat Commun (2021) | MaxQuant FOT | Raw FOT fractions ×10⁶ are linear; parser mislabels scale | Fix parser + `log2(x + 1)` (§2.4) |
| Angelidis_2019 | Angelidis et al., Nat Commun (2019) | MaxQuant LFQ | Supplementary Excel already log2; scripts pass-through | Keep as-is (§3.1) |
| Tam_2020 | Tam et al., eLife (2020) | MaxQuant LFQ | LFQ intensities in 16–41 log2 range | Keep as-is (§3.2) |
| Tsumagari_2023 | Tsumagari et al., Sci Rep (2023) | TMT 6-plex, MaxQuant 1.6.17 | Post MaxQuant+limma normalization yields log-like scale (medians ~28) | Keep as-is (§3.3) |
| Schuler_2021 | Schuler et al., Cell Rep (2021) | DIA-LFQ, Spectronaut v10-14 | mmc4 supplemental tables already log2 | Keep as-is (§3.4) |
| Santinha_2024_Human | Santinha et al., MCP (2024) | TMT-10, logFC + AveExpr back-calculation | Adapter derives log2 abundances from logFC/AveExpr | Keep as-is (§3.5) |
| Santinha_2024_Mouse_DT | Santinha et al., MCP (2024) | TMT-10, same script | Identical log2 derivation | Keep as-is (§3.6) |
| Santinha_2024_Mouse_NT | Santinha et al., MCP (2024) | TMT-10, same script | Identical log2 derivation | Keep as-is (§3.7) |
| Caldeira_2017 | Caldeira et al., Sci Rep (2017) | iTRAQ 8-plex, Protein Pilot | Reporter ratios (Young/Old) clustered near 1.0 | Exclude from batch correction (§4.1) |

## 2. Studies Requiring `log2(x + 1)` Transformation (Linear Source Data)

### 2.1 Randles_2021 – Human Kidney LFQ (Progenesis Hi-N)
- **Publication:** “Identification of an Altered Matrix Signature in Kidney Aging and Disease,” *Journal of the American Society of Nephrology* 32, 2021. PMID: 34049963.
- **Acquisition & Software:** Label-free LC-MS/MS processed in Progenesis QI v4.2 with Hi-N (top-3 peptide) quantification; Mascot for identification (`05_papers_to_csv/05_Randles_paper_to_csv/00_TASK_RANDLES_2021_CSV_CONVERSION.md`).
- **Scale Diagnosis:** Progenesis exports linear normalized abundances; paper methods confirm statistical analyses were run after export (`DATA_SCALE_VALIDATION_FROM_PAPERS.md`, §1.3). Median database values sit at 9.6 × 10³, corroborating linear scale.
- **Action:** Apply `log2(x + 1)` to both `Abundance_Young` and `Abundance_Old` prior to batch correction. Preserve compartment separation (glomerular vs tubulointerstitial).
- **Repository Artifacts:** Zero-to-NaN remediation and conversion scripts live in `05_papers_to_csv/05_Randles_paper_to_csv/`.

### 2.2 Dipali_2023 – Mouse Ovary DIA (DIA-NN directDIA)
- **Publication:** “Proteomic quantification of native and ECM-enriched mouse ovaries reveals an age-dependent fibro-inflammatory signature,” *Proteomics* 2023. PMID: 37903013.
- **Acquisition & Software:** Orbitrap Exploris 480 with DIA-NN v1.8 directDIA workflow; per-replicate quantities in `.PG.Quantity` columns (`04_compilation_of_papers/05_Dipali_2023_comprehensive_analysis.md`).
- **Scale Diagnosis:** DIA-NN outputs integrated ion areas (linear). Medians in the merged database fall between 6.1 × 10⁵ and 6.9 × 10⁵, matching linear expectations; no log was applied downstream (`DATA_SCALE_VALIDATION_FROM_PAPERS.md`, §1.6).
- **Action:** Standardize with `log2(x + 1)` before ComBat or mixed-model alignment. Flag the reproductive-age deviation (10–12 month “old” cohort) in downstream QC.
- **Repository Artifacts:** Parsing scripts and schema mapping under `05_papers_to_csv/10_Dipali_2023_paper_to_csv/`.

### 2.3 Ouni_2022 – Human Ovarian Cortex TMTpro 16-plex
- **Publication:** “Proteome-wide and matrisome-specific atlas of the human ovary computes fertility biomarker candidates...” *iScience* 25 (2022). PMID: 35341935.
- **Acquisition & Software:** DC-MaP extraction, TMTpro 16-plex labeling, quantified in Proteome Discoverer 2.4; quantile-normalized reporter intensities reported in Supplementary Table 3 (`05_papers_to_csv/08_Ouni_2022_paper_to_csv/00_TASK_OUNI_2022_TMT_PROCESSING.md`).
- **Scale Diagnosis:** Reporter channels retain linear intensity units after normalization (medians ~1.55 × 10²). Section 1.5 of `DATA_SCALE_VALIDATION_FROM_PAPERS.md` confirms no log2 transformation in methods.
- **Action:** Apply `log2(x + 1)` to reproductive (young) and menopausal (old) means before harmonization. Document exclusion of prepubertal group in processing notes.
- **Repository Artifacts:** `tmt_adapter_ouni2022.py`, wide-format export, and `PROCESSING_LOG_ZERO_FIX_2025-10-15.md` inside `05_papers_to_csv/08_Ouni_2022_paper_to_csv/`.

### 2.4 LiDermis_2021 – Human Skin Dermis (MaxQuant FOT)
- **Publication:** “Time-Resolved Extracellular Matrix Atlas of the Developing Human Skin Dermis,” *Nature Communications* 12, 2021. PMID: 34901026.
- **Acquisition & Software:** Label-free LC-MS/MS on Orbitrap; quantification via MaxQuant Fraction of Total (FOT) normalization multiplied by 10⁶ (`04_compilation_of_papers/06_LiDermis_2021_comprehensive_analysis.md`).
- **Scale Diagnosis:** FOT is a proportional linear metric. `DATA_SCALE_VALIDATION_FROM_PAPERS.md` (§1.4) and the Zero-to-NaN audit reveal that `parse_lidermis.py` incorrectly labels these values as log2 while they remain linear.
- **Action:** Update `parse_lidermis.py` (`Abundance_Unit` → `FOT_normalized_intensity`), then apply `log2(x + 1)` to the young (Toddler + Teenager) and old (Elderly) averages before recalculating z-scores.
- **Repository Artifacts:** Legacy parser and UniProt enrichment scripts located at `05_papers_to_csv/11_LiDermis_2021_paper_to_csv/`.

## 3. Studies Already on Log-Scaled Abundances (Keep As-Is)

### 3.1 Angelidis_2019 – Mouse Lung LFQ (MaxQuant)
- **Publication:** “An atlas of the aging lung mapped by single-cell transcriptomics and deep tissue proteomics,” *Nature Communications* 10:963 (2019). DOI: 10.1038/s41467-019-08831-9.
- **Evidence:** Supplementary workbook `41467_2019_8831_MOESM5_ESM.xlsx` stores LFQ intensities in 24–38 range; parsing scripts perform no further transformation (`04_compilation_of_papers/agents_for_batch_processing/Angelidis_2019/VALIDATION_REPORT.md`).
- **Action:** Retain existing log2 values for batch correction. Avoid double-logging.

### 3.2 Tam_2020 – Human Intervertebral Disc LFQ (MaxQuant)
- **Publication:** Tam et al., *eLife* 9:e64940 (2020). PMID: 33382035.
- **Evidence:** 66 LFQ intensity columns in `elife-64940-supp1-v3.xlsx` range 15.6–41.1; conversion scripts melt columns without log operations (`04_compilation_of_papers/agents_for_batch_processing/Tam_2020/VALIDATION_REPORT.md`).
- **Action:** Keep current log2 abundances when performing ComBat or mixed models.

### 3.3 Tsumagari_2023 – Mouse Brain TMT 6-plex (MaxQuant 1.6.17 + limma)
- **Publication:** Tsumagari et al., *Scientific Reports* 13:20571 (2023). PMID: 38086838.
- **Evidence:** MaxQuant SPS-MS3 with internal reference scaling, quantile normalization, and limma batch correction yield medians near 28; scripts do not apply log2 (`04_compilation_of_papers/agents_for_batch_processing/Tsumagari_2023/VALIDATION_REPORT.md`).
- **Action:** Treat abundances as stabilized log-scale values; do not add `log2(x + 1)`.

### 3.4 Schuler_2021 – Mouse Skeletal Muscle DIA-LFQ (Spectronaut)
- **Publication:** Schuler et al., “Extensive remodeling...,” *Cell Reports* 35:109223 (2021).
- **Evidence:** Spectronaut candidate tables exported to mmc4.xls already contain log2-transformed abundances (11.3–17.9). Processing code mirrors values without transformation (`04_compilation_of_papers/agents_for_batch_processing/Schuler_2021/VALIDATION_REPORT.md`).
- **Action:** Maintain current log2 values; proceed directly to batch correction.

### 3.5 Santinha_2024_Human – Human Cardiac TMT-10 (Back-calculated log2)
- **Publication:** Santinha et al., “Remodeling of the Cardiac Extracellular Matrix Proteome During Chronological and Pathological Aging,” *Molecular & Cellular Proteomics* 23 (2024). DOI: 10.1016/j.mcpro.2023.100706.
- **Evidence:** Adapter `tmt_adapter_santinha2024.py` reconstructs `Abundance_Young` and `Abundance_Old` from logFC and AveExpr via analytic equations; resulting medians 14.8–15.1 confirm log2 scale (`04_compilation_of_papers/agents_for_batch_processing/Santinha_2024_Human/VALIDATION_REPORT.md`).
- **Action:** Keep log2 abundances; no additional transformation.

### 3.6 Santinha_2024_Mouse_DT – Mouse Cardiac TMT-10 (Diabetic Treatment)
- **Evidence:** Shares identical processing pipeline with human dataset; medians 16.8–16.9 log2 with Gaussian skewness (`04_compilation_of_papers/agents_for_batch_processing/Santinha_2024_Mouse_DT/VALIDATION_REPORT.md`).
- **Action:** Keep as-is for batch correction.

### 3.7 Santinha_2024_Mouse_NT – Mouse Cardiac TMT-10 (Non-treated Control)
- **Evidence:** Same adapter and evidence chain; log2 medians 16.0–16.2, consistent across proteins (`04_compilation_of_papers/agents_for_batch_processing/Santinha_2024_Mouse_NT/VALIDATION_REPORT.md`).
- **Action:** Keep as-is for batch correction.

## 4. Dataset to Exclude from Batch Correction

### 4.1 Caldeira_2017 – Multi-Tissue iTRAQ Ratios
- **Publication:** Caldeira et al., “Decellularized bovine pericardium: Biochemical and biomechanical characterization,” *Scientific Reports* 7:11960 (2017).
- **Evidence:** Protein Pilot outputs iTRAQ reporter ratios (Young/Old) with medians near 1.7; agent validation flags incompatibility with intensity-based methods (`04_compilation_of_papers/agents_for_batch_processing/Caldeira_2017/VALIDATION_REPORT.md`).
- **Action:** Remove from ComBat-style harmonization. Retain data for qualitative comparisons only; mark in metadata as `Batch_Correction_Eligible = False`.

## 5. Reference Index
- **Caldeira_2017:** Scientific Reports 7:11960 (2017) – DOI 10.1038/s41598-017-11960-w.
- **Randles_2021:** Journal of the American Society of Nephrology 32 (2021) – PMID 34049963.
- **LiDermis_2021:** Nature Communications 12 (2021) – PMID 34901026.
- **Angelidis_2019:** Nature Communications 10:963 (2019) – DOI 10.1038/s41467-019-08831-9.
- **Tam_2020:** eLife 9:e64940 (2020) – PMID 33382035.
- **Ouni_2022:** iScience 25 (2022) – PMID 35341935.
- **Schuler_2021:** Cell Reports 35:109223 (2021).
- **Tsumagari_2023:** Scientific Reports 13:20571 (2023) – PMID 38086838.
- **Dipali_2023:** Proteomics (2023) – PMID 37903013.
- **Santinha_2024:** Molecular & Cellular Proteomics 23 (2024) – DOI 10.1016/j.mcpro.2023.100706 (Human, Mouse_DT, Mouse_NT datasets).

