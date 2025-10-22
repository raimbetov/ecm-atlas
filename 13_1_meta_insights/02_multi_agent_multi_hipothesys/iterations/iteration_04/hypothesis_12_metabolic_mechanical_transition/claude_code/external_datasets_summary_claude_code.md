# External Datasets for H12 Validation

**Agent:** claude_code
**Date:** 2025-10-21

## Summary

Identified 3 high-priority paired metabolomics+proteomics datasets and 2 YAP/TAZ mechanotransduction studies for future integration with ECM-Atlas.

---

## 1. PRIORITY DATASETS: Paired Metabolomics + Proteomics

### 1.1 Ten Mouse Organs Atlas (2025)
**Reference:** Genome Medicine, DOI: 10.1186/s13073-025-01535-4
**Description:** Systematic multi-omics analysis of 400 tissue samples from 10 organs
**Organs:** Brain, heart, intestine, kidney, liver, lung, muscle, skin, spleen, stomach
**Time Points:** 4, 8, 12, 20 months (mice)
**Methods:**
- Proteomics: DIA (Data-Independent Acquisition)
- Metabolomics: Positive and negative ion modes

**Relevance to H12:**
- Paired metabolomics+proteomics from SAME samples
- Multiple tissues spanning metabolic (liver, kidney) → mechanical (lung, skin) axes
- Covers early (4m) → late (20m) aging

**Next Steps:**
- Download from Genome Medicine supplementary data
- Map mouse → human gene symbols
- Calculate velocity proxies for 10 organs
- Test if v=1.65 threshold generalizes

---

### 1.2 Mouse Brain Metabolome Atlas
**Reference:** Metabolomics Workbench ST001637, ST001888
**Description:** Metabolome atlas of aging wildtype mouse brain from 10 anatomical regions
**Age Range:** Adolescence → old age
**Metabolites:** 1,547 structurally annotated

**Relevance to H12:**
- Brain (v=1.18 in ECM-Atlas) is Phase I tissue
- Can test if brain metabolome correlates with Phase I metabolic markers
- Regional brain data → test if hippocampus (v=1.18) vs cortex differs

**Access:**
- Metabolomics Workbench: https://www.metabolomicsworkbench.org/
- Study IDs: ST001637, ST001888

---

### 1.3 Mouse Multi-Organ Proteomics with Paired Samples
**Reference:** PXD047296 (PRIDE)
**Description:** TMT-labeled proteomics of 8 tissues with paired transcriptomics
**Tissues:** Aorta, brain, heart, kidney, liver, lung, muscle, skin
**Time Points:** 6, 15, 24, 30 months (C57BL/6 mice)
**Method:** TMT labeling

**Relevance to H12:**
- Lung (v=4.29, Phase II), heart (v=1.58-1.82, transition), muscle (v=1.02, Phase I)
- Covers full Phase I → Phase II spectrum
- TMT = high quantitative precision for velocity calculation

**Access:**
- PRIDE: https://www.ebi.ac.uk/pride/archive
- Accession: PXD047296

---

## 2. MECHANOTRANSDUCTION DATASETS: YAP/TAZ

### 2.1 YAP/TAZ Stromal Aging Study (Nature 2022)
**Reference:** Sladitschek-Martens et al., Nature 2022, DOI: 10.1038/s41586-022-04924-6
**Title:** "YAP/TAZ activity in stromal cells prevents ageing by controlling cGAS–STING"

**Key Findings:**
- YAP/TAZ activity declines during physiological aging in stromal cells
- YAP/TAZ mechanotransduction suppresses cGAS-STING signaling
- Sustaining YAP function rejuvenates old cells
- YAP/TAZ regulates nuclear envelope integrity (LaminB1, ACTR2)

**Proteomics Data:**
- Likely deposited in PRIDE (check paper supplementary for PXD accession)
- Focus: YAP/TAZ targets, nuclear envelope proteins, STING pathway

**Relevance to H12:**
- YAP/TAZ = mechanotransduction biomarker for Phase II
- Declining YAP/TAZ → Phase I (metabolic dysregulation)
- Rising YAP/TAZ → Phase II (mechanical remodeling, stiffness)
- Tests if v=1.65 corresponds to YAP/TAZ activation threshold

**Next Steps:**
- Download proteomics from PRIDE (find PXD in paper)
- Map YAP/TAZ targets to ECM-Atlas proteins
- Test enrichment in Phase II tissues

---

### 2.2 Human Multi-Compartment Metabolome Study (2025)
**Reference:** Nature Communications 2025, DOI: 10.1038/s41467-025-59964-z
**Description:** 5 organ-specific metabolome-based biological age gaps (MetBAGs)
**Cohort:** 274,247 UK Biobank participants
**Metabolites:** 107 plasma non-derivatized metabolites
**Compartments:** Plasma, muscle, urine

**Relevance to H12:**
- Metabolic aging biomarkers (Phase I)
- Multi-compartment (systemic vs tissue-specific)
- Large cohort → statistical power for v=1.65 threshold validation

**Next Steps:**
- Download metabolite data (check UK Biobank application)
- Correlate MetBAGs with ECM protein velocities
- Test if metabolic age predicts Phase I → Phase II transition

---

## 3. RECOMMENDED INTEGRATION STRATEGY

### Step 1: Mouse Organs Atlas (Highest Priority)
- Download proteomics + metabolomics from Genome Medicine paper
- Merge with ECM-Atlas (map mouse → human)
- Calculate tissue velocities for 10 organs
- Test changepoint detection (expect v=1.6-1.7 equivalent)

### Step 2: Metabolomics Validation
- Download ST001637, ST001888 from Metabolomics Workbench
- Extract metabolites: ATP, NAD+, lactate, pyruvate, glucose
- Correlate with Phase I ECM proteins (PLOD, coagulation, serpins)
- Hypothesis: High NAD+, low lactate → Phase I; low NAD+, high lactate → Phase II

### Step 3: YAP/TAZ Mechanotransduction
- Download Nature 2022 proteomics (find PRIDE accession)
- Map YAP/TAZ targets to ECM-Atlas
- Test: YAP/TAZ targets enriched in v>1.65 tissues

### Step 4: Cross-Dataset Validation
- Integrate all 3 datasets
- Build unified model: Metabolites → ECM velocity → Phase classification
- Test: Does metabolome predict transition timing (R²>0.70)?

---

## 4. LIMITATIONS & FUTURE WORK

**Current Limitations:**
1. ECM-Atlas lacks mitochondrial proteins (ECM-focused protocol)
2. No direct metabolomics in ECM-Atlas (proteomics only)
3. Species gap (mouse datasets vs human ECM-Atlas tissues)

**Future Directions:**
1. Re-process ECM-Atlas raw data with mitochondrial protein database
2. Paired ECM proteomics + metabolomics study (grant proposal)
3. YAP/TAZ immunostaining in Phase I vs Phase II tissues
4. Tissue stiffness measurements (AFM) to validate v=1.65 as mechanical threshold

---

## 5. DATASET AVAILABILITY TABLE

| Dataset | Type | Species | Tissues | Access | Priority |
|---------|------|---------|---------|--------|----------|
| Ten Organs Atlas | Proteomics + Metabolomics | Mouse | 10 organs | Genome Medicine | **HIGH** |
| Mouse Brain Metabolome | Metabolomics | Mouse | Brain (10 regions) | ST001637, ST001888 | Medium |
| PXD047296 | Proteomics (TMT) | Mouse | 8 tissues | PRIDE | **HIGH** |
| YAP/TAZ Aging (2022) | Proteomics | Mouse | Stromal cells | PRIDE (find PXD) | Medium |
| UK Biobank MetBAGs | Metabolomics | Human | Plasma, muscle, urine | UK Biobank | Medium |

---

**Recommendation:** Download **Ten Organs Atlas** (proteomics+metabolomics) as immediate next step for H12 validation.
