# Tam et al. 2020 - Comprehensive Analysis

**Compilation Date:** 2025-10-12  
**Study Type:** LFQ Proteomics (Phase 1 Ready)  
**Sources:** Original + Claude Code + Codex CLI (unanimous agreement)

---

## 1. Paper Overview
- **Title:** DIPPER, a spatiotemporal proteomics atlas of human intervertebral discs for exploring ageing and degeneration dynamics
- **PMID:** 33382035
- **Tissue:** Human lumbar intervertebral discs (nucleus pulposus, inner/outer annulus fibrosus, transition zones)
- **Species:** Homo sapiens
- **Age groups:** Young cadaver (16yr male) vs Aged cadaver (59yr male)
- **Design:** Binary, 1 donor per group, 66 spatial profiles total (multiple disc regions/coordinates)

---

## 2. Method & Age Bin Strategy
- **Method:** MaxQuant Label-Free Quantification ✅ LFQ compatible
- **Age normalization:** Already binary (young vs old cadavers) - NO CHANGES NEEDED
- **Young:** 16 years (~33 spatial profiles)
- **Old:** 59 years (~33 spatial profiles)
- **Data retention:** 100% (66/66 spatial profiles)
- **Age gap:** 43 years
- **Unique design:** Spatially resolved across disc compartments (NP, IAF, OAF, transitions)

---

## 3. Column Mapping (13 Schema Columns)
**Source:** `elife-64940-supp1-v3.xlsx`
- Sheet: "Raw data" (3,158 × 80)
- Metadata: "Sample information" (profile names, disc levels, ages, compartments)

| Column | Source | Notes |
|--------|--------|-------|
| Protein_ID | "T: Majority protein IDs" | MaxQuant UniProt accessions |
| Protein_Name | "T: Protein names" | ✅ Complete |
| Gene_Symbol | "T: Gene names" | ✅ |
| Tissue | Sample information "Compartment" | NP, IAF, OAF, NP/IAF transitions |
| Species | Constant "Homo sapiens" | |
| Age | Sample information "age-group" | young→16yr, old→59yr |
| Age_Unit | "years" | |
| Abundance | "LFQ intensity ..." columns | Each column = one spatial profile |
| Abundance_Unit | "LFQ_intensity" | MaxQuant normalized |
| Method | "Label-free LC-MS/MS (MaxQuant)" | |
| Sample_ID | Profile name | E.g., "L3/4_old_L_OAF" (disc level_age_direction_compartment) |

**Mapping quality:** ✅ All 13 columns complete; metadata join required for age/compartment annotation

---

## 4. Parsing Implementation
**Expected output:** 3,158 proteins × 66 profiles = **208,428 rows**

```python
# Join Raw data with Sample information on profile names
# Map age-group: "young" → 16, "old" → 59
# Parse compartment from Sample information
# Strip "T: " prefix from Raw data column names
```

**Special handling:** Spatially resolved data (66 profiles from 2 donors) vs typical replicate design

---

## 5. Ready for Parsing
- ✅ **READY IMMEDIATELY** - binary design, complete mappings
- **Priority:** HIGH (Tier 1)
- **Preprocessing:** None required (metadata join straightforward)
- **Note:** Single donor per age group but rich spatial sampling provides biological depth

---

**Agent Consensus:** 100% agreement on all classifications and mappings.
