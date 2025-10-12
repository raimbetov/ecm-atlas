# Dipali 2023 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: DIA-NN directDIA workflow on Orbitrap Exploris 480 with label-free quantification (Methods, Dipali et al. 2023).
- LFQ compatible: ✅ YES — No labeling chemistry; intensities derived from DIA MS1/fragment ion areas.

## 2. Current Age Groups
- Reproductively young (6–12 weeks) — runs identified as `Y1L` … `Y5L` in pivot export; 5 biological replicates.
- Reproductively old (10–12 months) — runs `O1L` … `O5L`; 5 biological replicates.
- Run metadata stored in `ConditionSetup.tsv`; quant values in `Report_Birgit_Protein+Quant_Pivot+(Pivot).xls` (`*.PG.Quantity` columns).

## 3. Species Context
- Species: Mus musculus.
- Lifespan reference: Laboratory mice 26–30 months total lifespan; reproductive senescence begins ~10–12 months.
- Aging cutoffs applied: young ≤4 months, old ≥18 months (user standard). **Note:** Study defines “old” as 10–12 months (peri-reproductive decline); document as deviation risk.

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** Runs with prefix `Y` (Y1L–Y5L)
  - Ages: 6–12 weeks (~1.5–3 months).
  - Justification: Within ≤4 month cutoff; represent peak fertility cohort.
  - Sample count: 5.
- **Old group:** Runs with prefix `O` (O1L–O5L)
  - Ages: 10–12 months.
  - Justification: Study’s reproductive aging cohort; although <18 months, ovarian senescence markers justify treating as functional “old.” Flag in notes for cross-study comparisons.
  - Sample count: 5.
- **EXCLUDED:** None.

### Impact Assessment
- **Data retained:** 10 / 10 samples = 100% ✅
- **Data excluded:** 0%.
- **Meets ≥66% threshold?** ✅ YES.
- **Signal strength:** High — pronounced ovarian remodeling between early adult vs late reproductive mice; note age cutoff deviation in risk log.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `data_raw/Dipali et al. - 2023/Report_Birgit_Protein+Quant_Pivot+(Pivot).xls` (tab-delimited despite `.xls` extension).
- Sheet/tab name: N/A (flat file).
- File size: 3,903 rows × 38 columns.
- Format: TSV export from directDIA protein pivot report.

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | `PG.UniProtIds` (semicolon-separated) | ✅ | Use first UniProt accession per entry.
| Protein_Name | `PG.ProteinDescriptions` | ✅ | Contains canonical protein descriptions.
| Gene_Symbol | `PG.Genes` | ✅ | Gene symbols returned by DIA-NN.
| Tissue | Constant `Ovary ECM (decellularized)` | ✅ | Decellularized ovary ECM fractions only (study focus).
| Species | Constant `Mus musculus` | ✅ | Mouse cohort only.
| Age | Map run prefix `Y` → 2.25 months (mean of 6–12 wk); `O` → 11 months (mean of 10–12 mo) | ✅ | Store numeric month value; document range in notes.
| Age_Unit | Constant `months` | ✅ | Ages converted to months for consistency.
| Abundance | Columns ending `.PG.Quantity` | ✅ | DIA-NN normalized protein quantities per run.
| Abundance_Unit | Constant `directDIA_quantity` | ✅ | Represents integrated peptide ion areas.
| Method | Constant `Label-free DIA (directDIA)` | ✅ | Workflow documented in paper.
| Study_ID | Constant `Dipali_2023` | ✅ | Parser identifier.
| Sample_ID | Derive from run label (e.g., `Y1L`, `O3L`) | ✅ | Unique per run.
| Parsing_Notes | Template capturing exact run file, replicate number, age range | ✅ | Build from `ConditionSetup.tsv` metadata.

### Mapping Gaps (if any)
- ⚠️ Age mapping uses cohort means (ranges not per mouse). Include range ± context in notes for transparency; no additional data gap beyond this assumption.

## 6. Implementation Notes
- Load pivot file with `sep='\t'`; filter columns containing `.PG.Quantity` for intensities.
- Join with `ConditionSetup.tsv` to confirm group labels and replicate ordering; reference column `Is Reference` to identify bridging channels (young runs appear as references).
- Consider additional normalization to adjust for reference-channel flag if required (some runs flagged as reference = TRUE).
- Maintain dictionary translating run IDs to age (weeks/months) and experimental condition (native vs decellularized) before long-format melt.
