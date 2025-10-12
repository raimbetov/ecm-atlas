# Li et al. 2021 (Pancreas) - Comprehensive Analysis

**Compilation Date:** 2025-10-12
**Study Type:** Non-LFQ (DiLeu 12-plex - Deferred to Phase 3)
**Sources:** Original KB analysis only

---

## 1. Paper Overview

- **Title:** Proteome-wide and matrisome-specific alterations during human pancreas development and maturation
- **PMID:** 32747717
- **Tissue:** Human pancreas (whole organ; acinar/islet compartments)
- **Species:** Homo sapiens
- **Age groups:** Fetal (18-20 gestational weeks), Juvenile (5-16 years), Young adult (21-29 years), Older adult (50-61 years)
- **Design:** 4 age groups with 6 donors per group (24 samples total)

---

## 2. Method Classification & Phase Assignment

### Quantification Method
- **Method:** 12-plex DiLeu isobaric labeling + LC-MS/MS
- **Labeling chemistry:** DiLeu (N,N-dimethyl leucine) isobaric tags
- **Data unit:** Log10-scaled DiLeu reporter intensities
- **Reference:** Methods p.3

### Non-LFQ Classification
- **Label-free?** ❌ NO - Uses DiLeu isobaric labeling
- **Isobaric tags:** 12-plex DiLeu enables multiplexed quantification
- **Normalization:** Shared reference channel across two batches
- **Reason for exclusion:** Labeled method incompatible with Phase 1 LFQ normalization

### Phase Assignment
- **Phase 1 (LFQ):** ❌ EXCLUDED - DiLeu is labeled method
- **Phase 2 (Age bins):** ⏸️ SKIPPED - Not applicable to non-LFQ
- **Phase 3 (Labeled methods):** ✅ DEFERRED - Includes DiLeu in scope
- **Status:** Ready for Phase 3 parsing with isobaric normalization

---

## 3. Data Files & Structure

**Primary file:** `41467_2021_21261_MOESM6_ESM.xlsx`
- **Sheet:** `Data 3` (2,066×53) - DiLeu reporter intensities across 24 samples
- **Key columns:**
  - Protein IDs: `Accession` (UniProt), `Description`
  - Abundance: Reporter columns (`F_7`–`O_68`) where F=Fetal, J=Juvenile, Y=Young adult, O=Older adult
  - QC metrics: `Coverage`, `# Peptides`

**Additional files:**
- `MOESM4_ESM.xlsx` - Donor metadata (age ranges, sex, DCD/DBD status)
- `MOESM5_ESM.xlsx` - Matrisome gene catalog (3,525 genes)
- `MOESM7_ESM.xlsx` - Differential abundance summaries

---

## 4. Why Excluded from Phase 1

### DiLeu vs LFQ Incompatibility

| Aspect | LFQ (Phase 1) | DiLeu (This Study) |
|--------|---------------|-------------------|
| **Labeling** | None | 12-plex DiLeu tags |
| **Quantification** | MS1 peak area | MS2 reporter ions |
| **Normalization** | Between samples | Between DiLeu channels |
| **Batch effects** | Run-specific | Shared reference channel |
| **Integration risk** | Cannot mix with LFQ | Requires separate normalization |

### Biological Context
- **Developmental span:** Fetal (gestational) → Juvenile → Young adult → Older adult
- **Human pancreas:** Captures maturation and aging across lifespan
- **Matrisome focus:** 2,066 proteins including ECM components
- **Value for Phase 3:** Human developmental aging trajectory in metabolic tissue

---

## 5. Column Mapping (Phase 3 Preview)

When parsed in Phase 3, will map to labeled-method schema:

| Schema Column | Source | Notes |
|--------------|--------|-------|
| Protein_ID | `Accession` | UniProt IDs |
| Protein_Name | `Description` | Canonical names |
| Gene_Symbol | Derive via UniProt | Map from accession |
| Tissue | Constant `Pancreas` | Whole organ |
| Species | Constant `Homo sapiens` | Human donors |
| Age | Map from column prefix | F=fetal weeks, J/Y/O=years |
| Abundance | Reporter columns | `F_7`, `J_8`, `Y_31`, `O_54` etc |
| Abundance_Unit | `log10_DiLeu_intensity` | Log-scaled reporter values |
| Method | `DiLeu 12-plex LC-MS/MS` | Phase 3 labeled method |

**Expected output (Phase 3):** 2,066 proteins × 24 samples = ~49,584 rows

---

## 6. Deferred to Phase 3

**Reason:** DiLeu isobaric labeling incompatible with label-free normalization
**Status:** Complete KB analysis available; ready for Phase 3 parsing
**Priority:** High (human developmental aging, large sample size, matrisome focus)

---

**Agent Consensus:** Both Claude Code and Codex CLI correctly excluded from Phase 1 (LFQ)
