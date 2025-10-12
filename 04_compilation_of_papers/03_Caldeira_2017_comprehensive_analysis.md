# Caldeira et al. 2017 - Comprehensive Analysis

**Compilation Date:** 2025-10-12
**Study Type:** Non-LFQ (iTRAQ 8-plex - Deferred to Phase 3)
**Sources:** Original KB analysis only

---

## 1. Paper Overview

- **Title:** Matrisome Profiling During Intervertebral Disc Development and Ageing
- **PMID:** 28912585
- **Tissue:** Bovine caudal nucleus pulposus
- **Species:** Bos taurus
- **Age groups:** Foetus (~7 months gestation), Young (12 months), Old (16-18 years)
- **Design:** 3 age groups with replicates/pools per group

---

## 2. Method Classification & Phase Assignment

### Quantification Method
- **Method:** iTRAQ LC-MS/MS (8-plex)
- **Labeling chemistry:** Isobaric iTRAQ tags
- **Data unit:** Normalized iTRAQ reporter ratios (dimensionless)
- **Reference:** Methods p.3 (ProteinPilot search)

### Non-LFQ Classification
- **Label-free?** ❌ NO - Uses iTRAQ isobaric labeling
- **Isobaric tags:** 8-plex iTRAQ labels enable multiplexed quantification
- **Quantification:** Reporter ion intensities per protein group
- **Reason for exclusion:** Labeled method incompatible with Phase 1 LFQ normalization

### Phase Assignment
- **Phase 1 (LFQ):** ❌ EXCLUDED - iTRAQ is labeled method
- **Phase 2 (Age bins):** ⏸️ SKIPPED - Not applicable to non-LFQ
- **Phase 3 (Labeled methods):** ✅ DEFERRED - Includes iTRAQ in scope
- **Status:** Ready for Phase 3 parsing with isobaric normalization

---

## 3. Data Files & Structure

**Primary file:** `41598_2017_11960_MOESM2_ESM.xls`
- **Sheet:** `1. Proteins` (81×24) - iTRAQ protein ratios per replicate/pool
- **Key columns:**
  - Protein IDs: `Accession Number`, `Protein Name`, `Accession Name`
  - Abundance: Reporter ratio columns (`Foetus 1-3`, `Young 1-3`, `Old 1-3`, pool columns)
  - Gene symbols: Derived from `Accession Name` (e.g., PGCA_BOVIN → ACAN)

**Additional files:**
- `MOESM3_ESM.xls` - Second iTRAQ experiment (104 proteins)
- `MOESM6_ESM.xls` - Fraction-specific ECM extracts (96-146 proteins)
- `MOESM7_ESM.xls` - Expression cluster data

---

## 4. Why Excluded from Phase 1

### iTRAQ vs LFQ Incompatibility

| Aspect | LFQ (Phase 1) | iTRAQ (This Study) |
|--------|---------------|-------------------|
| **Labeling** | None | 8-plex iTRAQ tags |
| **Quantification** | MS1 peak area | MS2 reporter ions |
| **Normalization** | Between samples | Between iTRAQ channels |
| **Dynamic range** | Wide | Compressed by reporter chemistry |
| **Integration risk** | Cannot mix with LFQ | Requires separate normalization |

### Biological Context
- **Developmental focus:** Foetus → Young → Old (spanning gestation to 16-18 years)
- **Species note:** Bos taurus (bovine) - different from mouse/human age cutoffs
- **Matrisome focus:** Specifically targets ECM proteins in nucleus pulposus
- **Value for Phase 3:** Developmental aging trajectory in ECM-rich tissue

---

## 5. Column Mapping (Phase 3 Preview)

When parsed in Phase 3, will map to labeled-method schema:

| Schema Column | Source | Notes |
|--------------|--------|-------|
| Protein_ID | `Accession Number` | UniProt accessions |
| Protein_Name | `Protein Name` | Human-readable names |
| Gene_Symbol | Derive from `Accession Name` | Strip _BOVIN suffix |
| Tissue | Constant `Nucleus pulposus` | IVD tissue |
| Species | Constant `Bos taurus` | Bovine samples |
| Age | Column headers | Foetus=7mo, Young=12mo, Old=204mo |
| Abundance | Reporter ratio columns | iTRAQ normalized ratios |
| Abundance_Unit | `iTRAQ_ratio` | Dimensionless |
| Method | `iTRAQ 8-plex LC-MS/MS` | Phase 3 labeled method |

**Expected output (Phase 3):** ~81 proteins × ~9 samples (replicates+pools) = ~729 rows

---

## 6. Deferred to Phase 3

**Reason:** iTRAQ isobaric labeling incompatible with label-free normalization
**Status:** Complete KB analysis available; ready for Phase 3 parsing
**Priority:** Medium (developmental aging, bovine ECM focus)

---

**Agent Consensus:** Both Claude Code and Codex CLI correctly excluded from Phase 1 (LFQ)
