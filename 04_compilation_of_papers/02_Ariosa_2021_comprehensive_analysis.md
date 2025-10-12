# Ariosa-Morejon et al. 2021 - Comprehensive Analysis

**Compilation Date:** 2025-10-12
**Study Type:** Non-LFQ (In vivo Pulsed SILAC - Deferred to Phase 3)
**Sources:** Original KB analysis only

---

## 1. Paper Overview

- **Title:** Age-dependent changes in protein incorporation into collagen-rich tissues of mice by in vivo pulsed SILAC labelling
- **PMID:** 34581667
- **Tissue:** Articular cartilage, tibial bone, ventral skin, plasma (4 tissues)
- **Species:** Mus musculus (C57BL/6J)
- **Age groups:** Group A (7 weeks), Group B (10 weeks), Group C (15 weeks), Group D (45 weeks)
- **Design:** 4 age groups with 1-3 replicates per tissue

---

## 2. Method Classification & Phase Assignment

### Quantification Method
- **Method:** In vivo pulsed SILAC + LC-MS/MS (MaxQuant v1.6)
- **Labeling chemistry:** Heavy lysine diet (stable isotope labeling)
- **Data unit:** iBAQ intensities per heavy/light channel, Ratio H/L
- **Reference:** Methods p.15-16

### Non-LFQ Classification
- **Label-free?** ❌ NO - Uses SILAC isotope labeling
- **Isotope labeling:** Heavy lysine (¹³C₆¹⁵N₂-Lys) incorporated in vivo
- **Quantification:** Heavy-to-light ratios measure protein incorporation dynamics
- **Reason for exclusion:** Labeled method incompatible with Phase 1 LFQ normalization

### Phase Assignment
- **Phase 1 (LFQ):** ❌ EXCLUDED - SILAC is labeled method
- **Phase 2 (Age bins):** ⏸️ SKIPPED - Not applicable to non-LFQ
- **Phase 3 (Labeled methods):** ✅ DEFERRED - Includes SILAC in scope
- **Status:** Ready for Phase 3 parsing with appropriate labeled-method normalization

---

## 3. Data Files & Structure

**Primary file:** `elife-66635-fig2-data1-v1.xlsx`
- **Sheets:** Plasma (173×33), Cartilage (634×42), Bone (712×42), Skin (352×42)
- **Key columns:** 
  - Protein IDs: `Majority protein IDs`, ` Protein names`, `Gene names`
  - Abundance: `iBAQ H/L {Group}{rep}` columns (heavy/light channels)
  - Ratios: `Ratio H/L A1-3`, `% Heavy isotope`

**Additional files:**
- `elife-66635-fig7-data1-v1.xlsx` - Differentially incorporated proteins per tissue
- `elife-66635-fig6-data1-v1.xlsx` - Glycoprotein incorporation heatmap source

---

## 4. Why Excluded from Phase 1

### SILAC vs LFQ Incompatibility

| Aspect | LFQ (Phase 1) | SILAC (This Study) |
|--------|---------------|-------------------|
| **Labeling** | None | Heavy lysine (in vivo) |
| **Quantification** | MS1 peak area | Heavy/light isotope ratios |
| **Normalization** | Between samples | Between heavy/light channels |
| **Interpretation** | Absolute abundance | Incorporation dynamics |
| **Integration risk** | Cannot mix with LFQ | Requires separate normalization |

### Biological Context
- **Unique design:** Measures protein **turnover** (newly synthesized vs pre-existing)
- **Heavy/light channels:** Both channels needed to calculate incorporation ratios
- **Age dynamics:** Paper focuses on incorporation rate changes across lifespan (7wk → 45wk)
- **Value for Phase 3:** Provides temporal protein dynamics not captured in static LFQ measurements

---

## 5. Column Mapping (Phase 3 Preview)

When parsed in Phase 3, will map to labeled-method schema:

| Schema Column | Source | Notes |
|--------------|--------|-------|
| Protein_ID | `Majority protein IDs` | MaxQuant UniProt accessions |
| Gene_Symbol | `Gene names` | Standard |
| Tissue | Sheet name | Plasma/Cartilage/Bone/Skin |
| Species | Constant `Mus musculus` | C57BL/6J mice |
| Age | Derived from group | A=7wk, B=10wk, C=15wk, D=45wk |
| Abundance | `iBAQ H/L {Group}{rep}` | Separate samples for heavy/light channels |
| Abundance_Unit | `iBAQ_intensity_SILAC` | Distinguish from LFQ |
| Method | `In vivo pulsed SILAC` | Phase 3 labeled method |

**Expected output (Phase 3):** ~2,500 proteins × 4 tissues × 4 ages × 2 channels = ~80,000 rows

---

## 6. Deferred to Phase 3

**Reason:** SILAC labeling incompatible with label-free normalization
**Status:** Complete KB analysis available; ready for Phase 3 parsing
**Priority:** Medium (unique turnover dynamics, multi-tissue design)

---

**Agent Consensus:** Both Claude Code and Codex CLI correctly excluded from Phase 1 (LFQ)
