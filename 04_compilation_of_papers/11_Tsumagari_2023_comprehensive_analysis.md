# Tsumagari et al. 2023 - Comprehensive Analysis

**Compilation Date:** 2025-10-12
**Study Type:** Non-LFQ (TMTpro 11-plex - Deferred to Phase 3)
**Sources:** Original KB analysis only

---

## 1. Paper Overview

- **Title:** Proteomic characterization of aging-driven changes in the mouse brain by co-expression network analysis
- **PMID:** 37875604
- **Tissue:** Mouse cerebral cortex and hippocampus (2 brain regions)
- **Species:** Mus musculus (C57BL/6J male)
- **Age groups:** 3-month, 15-month, 24-month mice (n=6 per age per tissue)
- **Design:** 3 age groups × 2 tissues × 6 replicates = 36 samples

---

## 2. Method Classification & Phase Assignment

### Quantification Method
- **Method:** TMT 11-plex LC-MS/MS (Orbitrap Fusion Lumos)
- **Labeling chemistry:** TMT isobaric tags (11-plex)
- **Data unit:** MaxQuant Reporter intensity (log2-normalized)
- **Reference:** Methods p.2 (TMT workflow with batch bridging)

### Non-LFQ Classification
- **Label-free?** ❌ NO - Uses TMT isobaric labeling
- **Isobaric tags:** 11-plex TMT enables multiplexed quantification
- **Normalization:** MaxQuant reporter intensity with internal reference channels (TMT-126, TMT-131C)
- **Reason for exclusion:** Labeled method incompatible with Phase 1 LFQ normalization

### Phase Assignment
- **Phase 1 (LFQ):** ❌ EXCLUDED - TMT is labeled method
- **Phase 2 (Age bins):** ⏸️ SKIPPED - Not applicable to non-LFQ
- **Phase 3 (Labeled methods):** ✅ DEFERRED - Includes TMT in scope
- **Status:** Ready for Phase 3 parsing with isobaric normalization

---

## 3. Data Files & Structure

**Primary files:**
- `41598_2023_45570_MOESM3_ESM.xlsx` - **Cortex data**
  - Sheet: `expression` (6,822×32) - TMT intensities `Cx_{age}_{rep}`
  - Sheet: `Welch's test` - Statistical outputs (log2 ratios, q-values)
- `41598_2023_45570_MOESM4_ESM.xlsx` - **Hippocampus data**
  - Sheet: `expression` (6,910×32) - TMT intensities `Hip_{age}_{rep}`
  - Sheet: `Welch's test` - Cortex vs hippocampus comparisons

**Key columns:**
- Protein IDs: `UniProt accession`, `Gene name`
- Abundance: Tissue-age-replicate columns (e.g., `Cx_3mo_1`, `Hip_15mo_4`)
- QC: MaxQuant quality metrics

**Additional files:**
- `MOESM6_ESM.xlsx` - Upregulated/downregulated marker lists per tissue
- `MOESM7_ESM.xlsx` - Peptide-level data (TableS4)
- `MOESM8_ESM.xlsx` - Transcriptional validation (TableS5)

---

## 4. Why Excluded from Phase 1

### TMT vs LFQ Incompatibility

| Aspect | LFQ (Phase 1) | TMT (This Study) |
|--------|---------------|-------------------|
| **Labeling** | None | 11-plex TMT tags |
| **Quantification** | MS1 peak area | MS2 reporter ions |
| **Normalization** | Between samples | Between TMT channels |
| **Batch bridging** | Not needed | TMT-126/131C for bridging |
| **Integration risk** | Cannot mix with LFQ | Requires separate normalization |

### Biological Context
- **Mouse brain aging:** 3mo (young adult) → 15mo (middle-aged) → 24mo (geriatric)
- **Multi-region:** Cortex vs hippocampus provides regional aging comparison
- **Network analysis:** Co-expression modules identify aging-driven protein clusters
- **Value for Phase 3:** Mouse brain aging with regional specificity and large sample size

---

## 5. Column Mapping (Phase 3 Preview)

When parsed in Phase 3, will map to labeled-method schema:

| Schema Column | Source | Notes |
|--------------|--------|-------|
| Protein_ID | `UniProt accession` | MaxQuant-derived accessions |
| Protein_Name | Map via UniProt | Derive from UniProt reference |
| Gene_Symbol | `Gene name` | Direct gene symbols |
| Tissue | Parse from column prefix | Cx=Cortex, Hip=Hippocampus |
| Species | Constant `Mus musculus` | C57BL/6J male mice |
| Age | Extract from column name | 3mo, 15mo, 24mo |
| Abundance | TMT reporter columns | `Cx_3mo_1`, `Hip_24mo_6` etc |
| Abundance_Unit | `TMTpro_normalized_intensity` | MaxQuant normalized |
| Method | `TMT 11-plex LC-MS/MS` | Phase 3 labeled method |

**Expected output (Phase 3):** ~6,900 proteins × 36 samples = ~248,400 rows (combined cortex+hippocampus)

---

## 6. Deferred to Phase 3

**Reason:** TMT isobaric labeling incompatible with label-free normalization
**Status:** Complete KB analysis available; ready for Phase 3 parsing
**Priority:** High (mouse brain aging, multi-region, large dataset, network analysis)

---

**Agent Consensus:** Both Claude Code and Codex CLI correctly excluded from Phase 1 (LFQ)
