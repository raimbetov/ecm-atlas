# Randles et al. 2021 - Comprehensive Analysis

**Compilation Date:** 2025-10-12  
**Study Type:** LFQ Proteomics (Phase 1 Ready)  
**Sources:** Original + Claude Code + Codex CLI (unanimous agreement)

---

## 1. Paper Overview
- **Title:** Identification of an Altered Matrix Signature in Kidney Aging and Disease
- **PMID:** 34049963
- **Tissue:** Human kidney cortex (glomerular and tubulointerstitial fractions)
- **Species:** Homo sapiens
- **Age groups:** Young (15, 29, 37 years) vs Aged (61, 67, 69 years)
- **Design:** Binary, 3 donors per group, 2 tissue compartments per donor = 12 samples total

---

## 2. Method & Age Bin Strategy
- **Method:** Progenesis Hi-N label-free LC-MS/MS ✅ LFQ compatible
- **Age normalization:** Already binary (young vs aged) - NO CHANGES NEEDED
- **Young:** 15, 29, 37yr (≤30yr cutoff; note: 37yr marginally above but author-classified as young)
- **Old:** 61, 67, 69yr (≥55yr cutoff)
- **Data retention:** 100% (12/12 samples)
- **Age gap:** 24-54 years between groups

---

## 3. Column Mapping (14 Schema Columns)
**Source:** `ASN.2020101442-File027.xlsx`, sheet "Human data matrix fraction" (2,611 × 31)

| Column | Source | Notes |
|--------|--------|-------|
| Protein_ID | "Accession" | UniProt from Mascot |
| Protein_Name | "Description" | ✅ Complete |
| Gene_Symbol | "Gene name" | ✅ |
| Tissue | Constant "Kidney" | Parent organ |
| Tissue_Compartment | Column prefix G/T | **✅ REQUIRED** - G=Glomerular, T=Tubulointerstitial |
| Species | Constant "Homo sapiens" | |
| Age | Column suffix number | G15→15yr, T67→67yr |
| Age_Unit | "years" | |
| Abundance | Gxx/Txx columns | Hi-N LFQ intensities |
| Abundance_Unit | "HiN_LFQ_intensity" | Top-3 peptide normalization |
| Method | "Label-free LC-MS/MS (Progenesis)" | |
| Study_ID | "Randles_2021" | PMID: 34049963 |
| Sample_ID | "{compartment}_{age}" | E.g., "G_15", "T_61" |
| Parsing_Notes | Generated | Document compartment separation, age groups |

**Mapping quality:** ✅ All 14 columns complete (including Tissue_Compartment); ⚠️ Filter .1 suffix columns (binary detection flags, not quant)

---

## 4. Parsing Implementation
**Expected output:** 2,611 proteins × 12 samples = **31,332 rows**

```python
# Filter columns: keep G15, G29, G37, T15, T29, T37 (young)
#                      G61, G67, G69, T61, T67, T69 (aged)
# Drop .1 suffix columns
young_samples = ['G15', 'G29', 'G37', 'T15', 'T29', 'T37']
aged_samples = ['G61', 'G67', 'G69', 'T61', 'T67', 'T69']
```

---

## 5. Ready for Parsing
- ✅ **READY IMMEDIATELY** - binary design, complete mappings
- **Priority:** HIGH (Tier 1)
- **Preprocessing:** None required
- **Note:** 37yr donor included per author classification (minor deviation from strict 30yr cutoff)

---

**Agent Consensus:** 100% agreement on all classifications and mappings.
