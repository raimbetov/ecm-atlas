# Lofaro et al. 2021 - Processing Status: BLOCKED

**Date:** 2025-10-14
**Status:** ❌ BLOCKED - Cannot process with current methodology

---

## Problem Identified

### Missing Sample-Level Quantification Data

**Issue:** Published supplementary tables contain only **aggregated statistics** (Log2 Fold Change, p-values), NOT individual replicate abundances.

**What we have:**
- `Table_S2_quantification.csv` - 154 rows with columns:
  - `Extraction_Method` (PBS, Urea/Thiourea, Guanidinium-HCl)
  - `Category` (ECM regulators, Collagens, etc.)
  - `Gene_Symbol`
  - `Log_Fold_Change` (aggregated)
  - `P_Value` (statistical test result)
  - `Protein_Name`

**What we need:**
- Individual abundance values for 18 samples:
  - 9 mice × 12 months (Adult)
  - 9 mice × 24 months (Old)
- Expected format: ~200 ECM proteins × 18 samples = 3,600 rows

**Example of published data:**
```csv
PBS,ECM regulators,FIBA,1.19,0.0009,Fibrinogen alpha chain
PBS,ECM regulators,TSP4,1.44,0.000,Thrombospondin-4
U/T,ECM Glycoproteins,FINC,1.35,0.036,Fibronectin
```

This is **statistical summary**, not raw quantification.

---

## Why This Blocks Processing

The autonomous agent pipeline (`11_subagent_for_LFQ_ingestion`) requires:
1. **Sample-level abundances** for each biological replicate
2. **Wide or long format** with individual intensity values
3. Ability to calculate **z-scores within age groups**

Without individual replicate data:
- Cannot normalize abundances
- Cannot calculate within-study z-scores
- Cannot merge with unified ECM Atlas dataset

---

## Study Details (For Reference)

### Publication
- **Title:** Age-Related Changes in the Matrisome of the Mouse Skeletal Muscle
- **Authors:** Francesco Demetrio Lofaro et al.
- **Journal:** IJMS, 2021
- **DOI:** 10.3390/ijms221910564
- **PMID:** 34686327
- **PRIDE:** PXD027895

### Biological Context
- **Species:** Mouse (BALB/c, male)
- **Tissue:** Gastrocnemius skeletal muscle
- **Ages:** 12 months (Adult) vs 24 months (Old)
- **Sample size:** n=9 per age group
- **Method:** Label-free LC-MS/MS with sequential extraction (PBS → Urea/Thiourea → Guanidinium-HCl)

### Age Normalization
✅ Binary design - no normalization needed (12mo → Young, 24mo → Old)

---

## Attempted Solutions

### 1. Check Published Supplementary Tables ❌
- **Table S1:** Protein identifications (3,650 proteins total)
- **Table S2:** ECM protein statistics (154 proteins, aggregated only)
- **Result:** No sample-level data in published materials

### 2. Check PRIDE Repository PXD027895 ⚠️
- **Available:** RAW mass spectrometry files
- **Issue:** Would require full reprocessing from scratch
  - Download RAW files (~GBs)
  - Run MaxQuant or similar tool
  - Map to matrisome proteins
  - Time-intensive (not feasible for immediate processing)

---

## Alternative Options

### Option A: Download and Reprocess RAW Data
**Pros:**
- Complete control over quantification
- Can extract sample-level abundances

**Cons:**
- Time-intensive (days of processing)
- Requires computational resources
- Need expertise in MS data processing
- Out of scope for current pipeline

**Verdict:** Not recommended for Phase 1

---

### Option B: Process Schüler 2021 Instead ✅ RECOMMENDED

**Study:** Schüler et al. (2021) Cell Reports - Muscle stem cell (MuSC) niche proteomics

**Why better:**
- Same tissue focus (skeletal muscle)
- DIA-LFQ method (compatible with pipeline)
- PRIDE: PXD015728 (likely has processed data)
- Interactive Shiny app with data: https://genome.leibniz-fli.de/shiny/orilab/muscle-aging/
- >4,500 proteins quantified

**Consideration:**
- Has **3 age groups** (3, 18, 26 months) - requires age normalization
- Need to exclude 18mo (middle-aged) and use only 3mo + 26mo
- Follows conservative strategy from compilation methodology

**Complementarity:**
- Lofaro = Bulk muscle ECM (gastrocnemius homogenate)
- Schüler = MuSC niche ECM (isolated satellite cell microenvironment)
- Both valuable but Schüler more feasible NOW

---

## Files in This Directory

```
05_paypers_to_csv/12_Lofaro_2021_paper_to_csv/
├── 00_STATUS_BLOCKED.md          (this file)
└── (empty - no processing performed)
```

---

## Decision Required

**Recommendation:**
1. **Defer Lofaro 2021** to Phase 3 (requires RAW data reprocessing)
2. **Proceed with Schüler 2021** (13_Schuler_2021) for immediate muscle ECM coverage
3. Mark Lofaro in compilation docs as "Deferred - awaiting PRIDE reprocessing"

---

## Related Files

- Comprehensive analysis: `/04_compilation_of_papers/12_Lofaro_2021_comprehensive_analysis.md`
- Raw data (PDFs/CSVs): `/data_raw/Lofaro et al. - 2021/`
- Methodology: `/11_subagent_for_LFQ_ingestion/00_START_HERE.md`
- Schüler 2021 analysis: `/04_compilation_of_papers/13_Schuler_2021_comprehensive_analysis.md`

---

**Conclusion:** Lofaro 2021 cannot be processed with current pipeline due to lack of sample-level quantification data in published materials. Alternative (Schüler 2021) identified and ready for processing after age normalization planning.

**Next Steps:**
1. Update compilation README to mark Lofaro as deferred
2. Create processing folder for Schüler 2021
3. Download Schüler 2021 data from PRIDE or Shiny app
4. Process with autonomous agent (excluding 18mo middle-aged group)
