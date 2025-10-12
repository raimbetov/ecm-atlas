# Ouni et al. 2022 - Comprehensive Analysis

**Compilation Date:** 2025-10-12
**Study Type:** Non-LFQ (TMTpro 16-plex - Deferred to Phase 3)
**Sources:** Original KB analysis only

---

## 1. Paper Overview

- **Title:** Proteome-wide and matrisome-specific atlas of the human ovary computes fertility biomarker candidates and opens the way for precision oncofertility
- **PMID:** 35341935
- **Tissue:** Human ovarian cortex (soluble and insoluble ECM fractions)
- **Species:** Homo sapiens
- **Age groups:** Prepubertal (7±3 years, n=5), Reproductive age (26±5 years, n=5), Menopausal (59±8 years, n=5)
- **Design:** 3 age groups with 5 biological replicates per group (15 samples)

---

## 2. Method Classification & Phase Assignment

### Quantification Method
- **Method:** DC-MaP + 16-plex TMTpro LC-MS/MS
- **Labeling chemistry:** TMTpro isobaric tags (16-plex)
- **Data unit:** Quantile-normalized TMT reporter intensities
- **Reference:** Methods (Experimental Procedures) + DC-MaP workflow

### Non-LFQ Classification
- **Label-free?** ❌ NO - Uses TMTpro isobaric labeling
- **Isobaric tags:** 16-plex TMTpro enables high-multiplexed quantification
- **Normalization:** Quantile normalization (`Q. Norm.`) with shared reference channels
- **Reason for exclusion:** Labeled method incompatible with Phase 1 LFQ normalization

### Phase Assignment
- **Phase 1 (LFQ):** ❌ EXCLUDED - TMTpro is labeled method
- **Phase 2 (Age bins):** ⏸️ SKIPPED - Not applicable to non-LFQ
- **Phase 3 (Labeled methods):** ✅ DEFERRED - Includes TMT in scope
- **Status:** Ready for Phase 3 parsing with isobaric normalization

---

## 3. Data Files & Structure

**Primary file:** `Supp Table 3.xlsx`
- **Sheet:** `Matrisome Proteins` (102×33) - Normalized TMT reporter intensities for matrisome
- **Key columns:**
  - Protein IDs: `Accession`, `EntryName`, `EntryGeneSymbol`
  - Abundance: `Q. Norm. of TOT_prepub1-5`, `TOT_repro1-5`, `TOT_meno1-5`
  - Groups: Prepubertal, Reproductive age, Menopausal

**Additional files:**
- `Supp Table 1.xlsx` - PTM-specific abundances (hydroxylation) split by soluble/insoluble fractions
- `Supp Table 2.xlsx` - Hippo/mTOR pathway annotations (literature-mined)
- `Supp Table 4.xlsx` - Validation metrics per pathway

---

## 4. Why Excluded from Phase 1

### TMTpro vs LFQ Incompatibility

| Aspect | LFQ (Phase 1) | TMTpro (This Study) |
|--------|---------------|-------------------|
| **Labeling** | None | 16-plex TMTpro tags |
| **Quantification** | MS1 peak area | MS2 reporter ions |
| **Normalization** | Between samples | Between TMT channels |
| **Multiplexing** | Sequential runs | 16 samples per run |
| **Integration risk** | Cannot mix with LFQ | Requires separate normalization |

### Biological Context
- **Reproductive aging:** Prepubertal → Reproductive → Menopausal (human ovary lifespan)
- **ECM focus:** DC-MaP workflow enriches matrisome proteins specifically
- **Clinical relevance:** Fertility biomarkers and oncofertility applications
- **Value for Phase 3:** Human reproductive aging in ECM-rich tissue with PTM data

---

## 5. Column Mapping (Phase 3 Preview)

When parsed in Phase 3, will map to labeled-method schema:

| Schema Column | Source | Notes |
|--------------|--------|-------|
| Protein_ID | `Accession` | UniProt IDs |
| Protein_Name | `EntryName` | Protein mnemonics |
| Gene_Symbol | `EntryGeneSymbol` | Official gene symbols |
| Tissue | Constant `Ovary_cortex` | Cortical tissue |
| Species | Constant `Homo sapiens` | Human donors |
| Age | Derive from column group | prepub=7yr, repro=26yr, meno=59yr |
| Abundance | `Q. Norm. of TOT_*` | Normalized TMT intensities |
| Abundance_Unit | `TMTpro_normalized_intensity` | Quantile-normalized |
| Method | `DC-MaP + TMTpro 16-plex` | Phase 3 labeled method |

**Expected output (Phase 3):** 102 matrisome proteins × 15 samples = ~1,530 rows (matrisome subset)

---

## 6. Deferred to Phase 3

**Reason:** TMTpro isobaric labeling incompatible with label-free normalization
**Status:** Complete KB analysis available; ready for Phase 3 parsing
**Priority:** High (human reproductive aging, ECM-enriched, clinical biomarkers)

---

**Agent Consensus:** Both Claude Code and Codex CLI correctly excluded from Phase 1 (LFQ)
