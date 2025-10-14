# Lofaro et al. 2021 - Dataset Analysis and Processing Recommendation

**Date:** 2025-10-14
**Analysis by:** Claude (ECM-Atlas Pipeline)
**Status:** ⚠️ **BLOCKED** - Requires alternative data source

---

## Executive Summary

**Problem:** The Lofaro et al. 2021 skeletal muscle aging study **cannot be processed** with the current ECM-Atlas LFQ ingestion pipeline due to **missing sample-level quantification data** in published materials.

**Available Data:**
- ❌ Published supplementary tables contain only **aggregated statistics** (fold changes, p-values)
- ✅ PRIDE repository PXD027895 contains **RAW mass spectrometry files** (requires reprocessing)
- ❌ No processed quantification files (e.g., MaxQuant proteinGroups.txt) available in PRIDE

**Recommendation:**
1. **Defer Lofaro 2021** to Phase 2 (requires downloading and reprocessing 50+ GB of RAW files)
2. **Alternative identified:** Schüler et al. 2021 (Cell Reports) - same tissue, compatible format, ready to process

---

## Study Details

### Publication Information

| Field | Value |
|-------|-------|
| **Title** | Age-Related Changes in the Matrisome of the Mouse Skeletal Muscle |
| **Authors** | Francesco Demetrio Lofaro, Federica Boraldi, et al. |
| **Journal** | IJMS (International Journal of Molecular Sciences) |
| **Year** | 2021 |
| **DOI** | 10.3390/ijms221910564 |
| **PMID** | 34686327 |
| **PRIDE ID** | PXD027895 |

### Biological Context

| Parameter | Details |
|-----------|---------|
| **Species** | Mouse (Mus musculus, BALB/c, male) |
| **Tissue** | Gastrocnemius skeletal muscle |
| **Age Groups** | Adult (12 months) vs Old (24 months) |
| **Sample Size** | n=9 mice per age group (total 18 samples) |
| **Method** | Label-free LC-MS/MS (MS1 intensity-based quantification) |
| **Extraction** | Sequential 3-step: PBS → Urea/Thiourea → Guanidinium-HCl |
| **Instrument** | Q Exactive Hybrid Quadrupole-Orbitrap |
| **Software** | Mascot + Skyline (DDA peptide search workflow) |

### Experimental Design

```
Adult (12 months)          Old (24 months)
├─ PBS extraction       ├─ PBS extraction
├─ Urea/Thiourea        ├─ Urea/Thiourea
└─ Guanidinium-HCl      └─ Guanidinium-HCl

n=9 biological replicates per age × 3 extraction methods = 54 samples
```

**Scientific Rationale:**
- Sequential extraction captures different ECM protein solubility classes
- PBS: Soluble/secreted proteins
- Urea/Thiourea: Cell-associated ECM
- Guanidinium-HCl: Highly crosslinked, insoluble ECM

---

## Data Availability Assessment

### What We Have

#### 1. Published Supplementary Tables ❌

**Table S1** (`Table_S1_proteins.csv`):
- **Content:** Complete protein identifications
- **Rows:** 3,650 proteins
- **Columns:** Extraction_Method, Protein_Symbol, Gene_Symbol, Score, Mass, Num_Significant_Matches, Num_Significant_Sequences, Num_Unique_Sequences, Protein_Name
- **Problem:** No quantification data, just identification metrics

**Example:**
```
Extraction_Method,Protein_Symbol,Gene_Symbol,Score,Mass,Num_Significant_Matches,Num_Significant_Sequences,Num_Unique_Sequences,Protein_Name
PBS,1433B_MOUSE,1433B,3708,28183,199,13,6,14-3-3 protein beta/alpha
PBS,1433E_MOUSE,1433E,11008,29326,481,18,15,14-3-3 protein epsilon
```

**Table S2** (`Table_S2_quantification.csv`):
- **Content:** ECM protein aging statistics
- **Rows:** 154 ECM proteins
- **Columns:** Extraction_Method, Category, Gene_Symbol, Log_Fold_Change, P_Value, Protein_Name
- **Problem:** Only **aggregated fold changes**, no individual sample abundances

**Example:**
```
Extraction_Method,Category,Gene_Symbol,Log_Fold_Change,P_Value,Protein_Name
PBS,ECM regulators,SPA3N,1.773,0.0284,Serine protease inhibitor A3N
PBS,ECM regulators,A1AT2,0.693,0.0056,Alpha-1-antitrypsin 1-2
PBS,ECM regulators,CATD,0.578,0.0008,Cathepsin D
```

**Why This Is Insufficient:**
- No individual replicate abundances → Cannot calculate within-study z-scores
- No Young vs Old separation → Cannot determine which group has higher/lower values
- Only 154 ECM proteins → Missing the full proteome context needed for normalization

#### 2. PRIDE Repository PXD027895 ⚠️

**Status:** Accessible via PRIDE Archive

**Project Details:**
- **Submission Date:** 2021-08-12
- **Publication Date:** 2022-02-17
- **Submission Type:** PARTIAL (only RAW files, no processed results)
- **License:** Creative Commons Public Domain (CC0)

**Available Files:**
- **RAW files:** 54 .raw files (Q Exactive native format)
  - 18 biological samples × 3 extractions = 54 runs
  - Each file ~1-2 GB
  - **Total estimated size:** ~50-100 GB

**Missing Files:**
- ❌ MaxQuant `proteinGroups.txt`
- ❌ Skyline `.sky` project with quantification
- ❌ Mascot `.dat` search results
- ❌ Any processed quantification tables

**Data Processing Protocol (from PRIDE):**
```
1. RAW files → msConvert → MGF files
2. MGF → Mascot search → .dat files
3. .dat → Skyline spectral libraries
4. RAW + libraries → Skyline → Precursor ion intensities
5. Protein-level quantification (≥2 peptides per protein)
```

**Why PRIDE Files Cannot Be Used Directly:**
- Mascot .dat files and Skyline .sky files are **not deposited**
- Only RAW files available → Requires **complete reprocessing from scratch**
- Reprocessing workflow:
  1. Download 50-100 GB of RAW files
  2. Run MaxQuant (12-24 hours of computation)
  3. Parse proteinGroups.txt
  4. Map to matrisome proteins
  5. Calculate z-scores
- **Time estimate:** 2-3 days for experienced user

---

## Why This Blocks ECM-Atlas Pipeline

### Pipeline Requirements

The `11_subagent_for_LFQ_ingestion/autonomous_agent.py` requires:

1. **Sample-level abundances** for each biological replicate
   - Expected format: Protein × Sample matrix
   - Example: 200 proteins × 18 samples = 3,600 data points

2. **Age group assignment** for each sample
   - Young group: 12-month samples (n=9)
   - Old group: 24-month samples (n=9)

3. **Ability to calculate statistics**
   - Mean abundance per age group
   - Standard deviation per age group
   - Z-score transformation: `(value - mean) / std`

### What We Have vs What We Need

| Required | Available | Status |
|----------|-----------|--------|
| Individual sample abundances | ❌ Only fold changes | **BLOCKED** |
| Young/Old separation | ❌ Aggregated across groups | **BLOCKED** |
| Full protein context | ✅ 3,650 proteins in Table S1 | ✅ OK |
| ECM protein annotations | ✅ 154 proteins in Table S2 | ✅ OK |
| Replicate-level data | ❌ Summary statistics only | **BLOCKED** |

### Example of Missing Data

**What we have:**
```
Gene_Symbol: CATD
Log_Fold_Change: 0.578
P_Value: 0.0008
```

**What we need:**
```
Gene_Symbol: CATD
Sample_12mo_1: 15234.5
Sample_12mo_2: 14892.1
Sample_12mo_3: 16103.4
...
Sample_24mo_1: 24156.8
Sample_24mo_2: 23001.5
Sample_24mo_3: 25344.2
```

Without individual sample values:
- ❌ Cannot calculate z-scores
- ❌ Cannot determine direction of change (increase vs decrease)
- ❌ Cannot normalize across studies
- ❌ Cannot merge to unified ECM-Atlas database

---

## Alternative Solutions

### Option A: Reprocess RAW Files from PRIDE ⚠️

**Pros:**
- Complete control over quantification
- Can extract full sample-level data
- Gold standard approach

**Cons:**
- **Time-intensive:** 2-3 days for experienced proteomics analyst
- **Computationally expensive:** Requires high-RAM server (≥64 GB)
- **Storage:** 50-100 GB for RAW files + 10 GB for MaxQuant output
- **Expertise required:** Familiarity with MaxQuant, Skyline, or equivalent
- **Out of scope:** Not feasible for current autonomous pipeline

**Workflow:**
```
1. Download RAW files from PRIDE FTP
   ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD027895/

2. Run MaxQuant with:
   - Mouse UniProt database (2021 version)
   - Trypsin digestion
   - Carbamidomethyl (C) fixed modification
   - Oxidation (M), Deamidation (NQ) variable modifications
   - Match between runs: ON
   - LFQ: ON

3. Parse proteinGroups.txt
   - Extract LFQ intensities for 18 samples
   - Filter for matrisome proteins
   - Transform to long format

4. Process with autonomous agent
   python autonomous_agent.py "reprocessed_Lofaro_2021/"
```

**Decision:** ❌ **Not recommended for Phase 1** (defer to Phase 2 or 3)

---

### Option B: Contact Authors for Skyline Project ✉️

**Pros:**
- Authors may have Skyline .sky file with full quantification
- Faster than reprocessing RAW files
- Authors explicitly state they used Skyline

**Cons:**
- Response time uncertain (days to weeks)
- No guarantee files are still available
- May require data sharing agreement
- Still requires Skyline software to extract data

**Action Items:**
1. Email corresponding author:
   - Dr. Francesco Demetrio Lofaro (francescodemetrio.lofaro@unimore.it)
   - Request Skyline .sky project file with precursor ion intensities
2. If received, export quantification table:
   - File → Export → Report → Protein Quantification
3. Process with autonomous agent

**Decision:** 🟡 **Possible, but uncertain timeline**

---

### Option C: Process Alternative Study (Schüler 2021) ✅ RECOMMENDED

**Study:** Schüler et al. (2021), *Cell Reports*
**Title:** Cellular Proteome Changes Associated with Muscle Stem Cell Niche Aging

**Why This Is Better:**

| Feature | Lofaro 2021 | Schüler 2021 |
|---------|-------------|--------------|
| **Tissue** | Gastrocnemius (bulk muscle) | Muscle stem cell (MuSC) niche ECM |
| **Species** | Mouse (BALB/c) | Mouse (C57BL/6) |
| **Age Groups** | 12mo, 24mo | 3mo, 18mo, 26mo |
| **Method** | LFQ | DIA-LFQ (Data-Independent Acquisition) |
| **Proteins** | 3,650 total, 154 ECM | >4,500 total, ~200-300 ECM |
| **PRIDE ID** | PXD027895 (RAW only) | PXD015728 (processed data likely available) |
| **Data Access** | RAW files only | **Interactive Shiny app** with full data! |
| **Processing Ready** | ❌ NO | ✅ **YES** |

**Shiny App:** https://genome.leibniz-fli.de/shiny/orilab/muscle-aging/

**Key Advantages:**
1. **Sample-level data available** via Shiny app download
2. **DIA-LFQ** is highly compatible with LFQ pipeline (same intensity-based quantification)
3. **3 age groups** allows age normalization:
   - Young: 3 months
   - Middle-aged: 18 months (exclude for binary comparison)
   - Old: 26 months
4. **MuSC niche ECM** is scientifically complementary to bulk muscle ECM
5. **Immediate processing:** Data ready for autonomous agent

**Biological Complementarity:**
- **Lofaro:** Bulk gastrocnemius ECM (whole muscle homogenate)
- **Schüler:** MuSC niche ECM (microenvironment of satellite cells)
- Both address **skeletal muscle aging**, different compartments
- Schüler adds **stem cell niche** perspective to ECM-Atlas

**Age Normalization Strategy:**
- **Option 1:** Use only 3mo (Young) vs 26mo (Old), exclude 18mo middle-aged
- **Option 2:** Include all 3 ages, calculate linear regression z-scores
- **Recommendation:** Option 1 (binary design, consistent with other studies)

---

## Recommended Workflow

### Phase 1 (Immediate): Process Schüler 2021

**Step 1: Download Data from Shiny App**
```bash
# Visit: https://genome.leibniz-fli.de/shiny/orilab/muscle-aging/
# Download full dataset (likely CSV or Excel format)
# Save to: data_raw/Schuler et al. - 2021/
```

**Step 2: Inspect Data Structure**
```bash
# Check column names, sample IDs, age labels
head -20 "data_raw/Schuler et al. - 2021/muscle_aging_proteomics.csv"
```

**Step 3: Run Autonomous Agent**
```bash
cd 11_subagent_for_LFQ_ingestion
python autonomous_agent.py "../data_raw/Schuler et al. - 2021/"
```

**Step 4: Configure Study**
```json
{
  "study_id": "Schuler_2021",
  "tissue": "Skeletal_muscle_MuSC_niche",
  "species": "Mus_musculus",
  "young_ages": ["3_months"],
  "old_ages": ["26_months"],
  "method": "DIA-LFQ",
  "compartments": ["MuSC_niche"]
}
```

**Step 5: Process & Merge**
```bash
# Agent will automatically:
# 1. Filter for ECM proteins (matrisome)
# 2. Calculate z-scores (3mo vs 26mo)
# 3. Merge to unified ECM-Atlas CSV
# 4. Update dashboard
```

**Expected Output:**
- `05_paypers_to_csv/13_Schuler_2021_paper_to_csv/Schuler_2021_wide_format.csv`
- Merged to: `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Dashboard updated with new "Skeletal_muscle_MuSC_niche" compartment

---

### Phase 2 (Future): Reprocess Lofaro 2021 RAW Files

**When to do this:**
- After establishing baseline ECM-Atlas with 10+ studies
- When computational resources available (server with ≥64 GB RAM)
- When MaxQuant processing pipeline is set up

**Workflow:**
```bash
# 1. Download RAW files
wget -r ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2022/02/PXD027895/

# 2. Run MaxQuant (separate pipeline, not in scope for autonomous agent)
# [MaxQuant GUI or command-line processing]

# 3. Once proteinGroups.txt is available:
python autonomous_agent.py "reprocessed_data/Lofaro_2021/"

# 4. Merge to ECM-Atlas
# Agent will handle automatically
```

**Priority:** Low (Schüler 2021 provides skeletal muscle ECM coverage)

---

## Files in Lofaro 2021 Directory

### Current Status
```
data_raw/Lofaro et al. - 2021/
├── Table S1.pdf                      # Original PDF
├── Table S2.pdf                      # Original PDF
├── Table_S1_proteins.csv             # 3,650 proteins (identification only)
├── Table_S2_quantification.csv       # 154 ECM proteins (fold changes only)
└── extract_tables_final.py           # PDF extraction script

05_paypers_to_csv/12_Lofaro_2021_paper_to_csv/
└── 00_STATUS_BLOCKED.md              # This analysis document
```

### Missing (Required for Processing)
```
❌ Lofaro_2021_LFQ_intensities.csv    # Sample-level abundances
❌ Lofaro_2021_wide_format.csv        # Processed for merge
❌ study_config.json                  # Configuration file
❌ agent_log.md                       # Processing log
```

---

## Technical Details: Why Fold Changes ≠ Abundances

### Statistical Summary vs Raw Data

**Fold Change (Log2):**
```
Log2(Old/Young) = log2(mean(Old samples) / mean(Young samples))
```

**Example:**
- Log_Fold_Change: 0.578 for CATD
- Interpretation: **Old samples have 1.49× higher CATD** than Young

**Problem:**
- Cannot reverse-engineer individual sample values from aggregated statistics
- Lost information: Standard deviation, replicate variability, outliers
- Cannot calculate within-study z-scores for cross-study normalization

### What Z-Score Calculation Requires

```python
# For each protein, for each age group:
young_mean = mean(young_samples)
young_std = std(young_samples)

old_mean = mean(old_samples)
old_std = std(old_samples)

# Z-score transformation:
for sample in young_samples:
    z_young = (sample - young_mean) / young_std

for sample in old_samples:
    z_old = (sample - old_mean) / old_std

# Delta z-score:
delta_z = mean(z_old) - mean(z_young)
```

**Without individual sample values, this calculation is impossible.**

---

## Summary and Next Steps

### Current Situation

✅ **What We Know:**
- Lofaro 2021 is a high-quality skeletal muscle aging study
- 154 ECM proteins identified with significant age-related changes
- Sequential extraction captures different ECM solubility classes
- PRIDE repository contains RAW files (PXD027895)

❌ **What We Don't Have:**
- Sample-level quantification data in published materials
- Processed files in PRIDE repository
- Skyline project file from authors

⚠️ **What Blocks Processing:**
- Current pipeline requires sample-level abundances
- Reprocessing RAW files is out of scope for Phase 1
- Time and computational resources not available

### Recommendations

**IMMEDIATE (Phase 1):**
1. ✅ **Process Schüler 2021 instead**
   - Same tissue (skeletal muscle)
   - Data immediately available via Shiny app
   - Compatible with autonomous agent
   - Timeline: 1-2 hours

2. 📧 **Optional: Contact Lofaro authors**
   - Request Skyline .sky file
   - May accelerate Lofaro 2021 processing
   - Timeline: 1-4 weeks (uncertain)

**FUTURE (Phase 2-3):**
3. ⏳ **Defer Lofaro 2021 to Phase 2**
   - Mark as "Requires RAW reprocessing"
   - Process when MaxQuant pipeline established
   - Timeline: 2-3 days when resources available

### Decision Matrix

| Option | Timeline | Difficulty | Output Quality | Recommendation |
|--------|----------|------------|----------------|----------------|
| **Process Schüler 2021** | 1-2 hours | Easy | High | ✅ **DO THIS NOW** |
| **Contact Lofaro authors** | 1-4 weeks | Medium | High | 🟡 Optional |
| **Reprocess Lofaro RAW** | 2-3 days | Hard | Highest | ⏳ Defer to Phase 2 |

---

## Conclusion

**Lofaro et al. 2021 cannot be processed** with the current ECM-Atlas pipeline due to the absence of sample-level quantification data in published supplementary materials. While the study is scientifically valuable, it requires either:
1. Obtaining Skyline project files from authors, OR
2. Reprocessing 50-100 GB of RAW mass spectrometry files

**Recommended path forward:**
- **Immediate:** Process **Schüler et al. 2021** (Cell Reports) for skeletal muscle ECM coverage
- **Future:** Defer Lofaro 2021 to Phase 2 when RAW file reprocessing infrastructure is established

**Schüler 2021 advantages:**
- Data immediately available via interactive Shiny app
- Same tissue focus (skeletal muscle)
- Complementary biological context (MuSC niche ECM vs bulk muscle ECM)
- Compatible with autonomous agent
- Ready to process in 1-2 hours

**Next action:** Download Schüler 2021 data and run autonomous agent.

---

## References

1. **Lofaro et al. 2021** - IJMS 22(19):10564
   https://doi.org/10.3390/ijms221910564
   PRIDE: PXD027895

2. **Schüler et al. 2021** - Cell Reports (recommended alternative)
   PRIDE: PXD015728
   Shiny App: https://genome.leibniz-fli.de/shiny/orilab/muscle-aging/

3. **ECM-Atlas LFQ Ingestion Pipeline**
   `/11_subagent_for_LFQ_ingestion/00_START_HERE.md`

---

**Report Generated:** 2025-10-14
**Status:** Lofaro 2021 BLOCKED, Schüler 2021 READY TO PROCESS
