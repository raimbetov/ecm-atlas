# H13 Independent Dataset Validation - Claude Code Agent
**Status:** 🚧 Data Acquisition Phase
**Priority:** 🔴 CRITICAL - Prevents overfitting

## Quick Start

### Current Status
✅ **Completed:**
- Comprehensive dataset search (6 datasets identified)
- Validation framework designed
- All analysis scripts written and ready
- Results report template created

⏳ **Pending:**
- External dataset download (PXD011967, PXD015982)
- Data harmonization
- Transfer learning validation execution
- Meta-analysis execution

### Immediate Next Steps

1. **Download External Datasets**
   ```bash
   # PXD011967 - Skeletal Muscle
   # Download from: https://elifesciences.org/articles/49874
   # Files: Figure 1—source data 1.xlsx (protein quantification)

   # PXD015982 - Skin ECM
   # Download from: https://www.ebi.ac.uk/pride/archive/projects/PXD015982
   # Or: https://pmc.ncbi.nlm.nih.gov/articles/PMC7852213/
   ```

2. **Place Files in Workspace**
   ```bash
   cd /Users/Kravtsovd/projects/ecm-atlas/.../claude_code
   mkdir -p external_datasets/PXD011967/raw
   mkdir -p external_datasets/PXD015982/raw
   # Copy downloaded files to respective folders
   ```

3. **Run Harmonization**
   ```bash
   python scripts/data_harmonization_claude_code.py
   ```

4. **Execute Validation**
   ```bash
   python scripts/transfer_learning_h08_claude_code.py
   python scripts/meta_analysis_claude_code.py
   ```

## Directory Structure

```
claude_code/
├── README.md                          # This file
├── DATA_ACQUISITION_PLAN.md           # Detailed data acquisition strategy
├── 90_results_claude_code.md          # Comprehensive results report
│
├── scripts/                           # Analysis scripts (ready to run)
│   ├── dataset_search_claude_code.py          # PRIDE/ProteomeXchange search
│   ├── data_harmonization_claude_code.py      # Process external data
│   ├── transfer_learning_h08_claude_code.py   # S100 model validation
│   └── meta_analysis_claude_code.py           # I² heterogeneity testing
│
├── external_datasets/                 # Downloaded external data
│   ├── PXD011967/                    # Skeletal muscle aging
│   │   ├── raw/                      # Downloaded files (TO BE ADDED)
│   │   ├── metadata.json
│   │   └── PXD011967_processed_zscore.csv
│   ├── PXD015982/                    # Skin ECM aging
│   │   ├── raw/
│   │   ├── metadata.json
│   │   └── PXD015982_processed_zscore.csv
│   └── merged_external_zscore.csv
│
├── models/                            # Saved models (from H08)
│   └── [Links to Iteration 03 models]
│
├── visualizations_claude_code/        # Output plots
│   ├── h08_transfer_scatter_claude_code.png
│   ├── meta_forest_plot_claude_code.png
│   └── [Other visualizations]
│
├── external_datasets_summary_claude_code.csv  # Dataset catalog
├── discovered_datasets_claude_code.csv        # Search results
├── dataset_search_summary.json                # Search metadata
├── h08_external_validation_claude_code.csv    # H08 results (TBD)
├── h06_external_validation_claude_code.csv    # H06 results (TBD)
├── h03_velocity_comparison_claude_code.csv    # H03 results (TBD)
├── meta_analysis_results_claude_code.csv      # Meta-analysis (TBD)
└── protein_stability_claude_code.csv          # Protein classification (TBD)
```

## Validation Framework

### Hypothesis Tests

| Test | Model Source | External Data | Metric | Target | Interpretation |
|------|--------------|---------------|--------|--------|----------------|
| **H08 S100→Stiffness** | `/iterations/iteration_03/hypothesis_08/` | PXD011967 (muscle) | R² | ≥ 0.60 | Model generalizes |
| **H06 Biomarkers** | `/iterations/iteration_02/hypothesis_06/` | PXD011967 (muscle) | AUC | ≥ 0.80 | Panel validated |
| **H03 Velocities** | `/iterations/iteration_01/hypothesis_03/` | PXD011967 (muscle) | ρ | > 0.70 | Tissue ranking universal |
| **Meta-analysis** | All top 20 proteins | PXD011967 + PXD015982 | I² | < 50% | Low heterogeneity |

### Success Scenarios

**✅ Scenario 1: Strong Validation**
- All targets met → Findings ROBUST
- **Action:** Publish, proceed to clinical translation

**⚠️ Scenario 2: Moderate Validation**
- Some targets met → Focus on stable proteins
- **Action:** Investigate context-dependence

**❌ Scenario 3: Poor Validation**
- Targets missed → Overfitting detected
- **Action:** Re-evaluate all hypotheses

## Identified External Datasets

### HIGH PRIORITY (Ready for Download)

**1. PXD011967 - Skeletal Muscle Aging**
- **Tissue:** Vastus lateralis muscle
- **Age groups:** 20-34, 35-49, 50-64, 65-79, 80+ (n=58)
- **Proteins:** 4,380 quantified
- **Method:** TMT 6-plex
- **Paper:** Ferri et al. (2019), eLife
- **Access:** https://elifesciences.org/articles/49874 (supplementary files)

**2. PXD015982 - Skin ECM Aging**
- **Tissue:** Skin (hip, underarm, forearm)
- **Age groups:** Young (26.7 yr), Aged (84.0 yr)
- **Proteins:** 229 matrisome proteins (115 core)
- **Method:** QconCAT absolute quantification
- **Paper:** Richter et al. (2021), Matrix Biol Plus
- **Access:** https://www.ebi.ac.uk/pride/archive/projects/PXD015982

### MEDIUM PRIORITY

- PXD007048: Bone marrow aging
- MSV000082958: Lung fibrosis (ECM/collagen focus)
- MSV000096508: Brain ECM and cognitive aging (mouse)

### PENDING

**Cell 2025 Multi-Tissue Study** (Ding et al., PMID: 40713952)
- 13 tissues, 516 samples, 5 decades
- **Status:** Accession not yet located (requires full paper access)

## Key Scripts

### 1. Dataset Search
```bash
python scripts/dataset_search_claude_code.py
```
**Purpose:** Query PRIDE/ProteomeXchange API for aging ECM datasets
**Output:** `discovered_datasets_claude_code.csv`
**Status:** ✅ Completed

### 2. Data Harmonization
```bash
python scripts/data_harmonization_claude_code.py
```
**Purpose:** Process external data to z-score format matching our merged dataset
**Requirements:** External data files in `external_datasets/*/raw/`
**Output:** `PXD011967_processed_zscore.csv`, `PXD015982_processed_zscore.csv`
**Status:** ⏳ Awaiting data download

### 3. H08 Transfer Learning
```bash
python scripts/transfer_learning_h08_claude_code.py
```
**Purpose:** Test S100→stiffness model on external muscle data WITHOUT retraining
**Requirements:**
- Processed external data
- Pre-trained model from H08 (`s100_stiffness_model_claude_code.pth`)
**Output:** `h08_external_validation_claude_code.csv`, scatter plot
**Status:** ⏳ Awaiting harmonized data

### 4. Meta-Analysis
```bash
python scripts/meta_analysis_claude_code.py
```
**Purpose:** Combine our + external data, calculate I² heterogeneity for top proteins
**Output:** `meta_analysis_results_claude_code.csv`, forest plot
**Status:** ⏳ Awaiting harmonized data

## Success Criteria

| Criterion | Target | Current Status |
|-----------|--------|----------------|
| Independent datasets found | ≥ 2 | ✅ 6 datasets |
| Gene overlap with our 648 ECM genes | ≥ 70% | ⏳ TBD |
| H08 S100 model R² (external) | ≥ 0.60 | ⏳ TBD |
| H06 biomarker panel AUC (external) | ≥ 0.80 | ⏳ TBD |
| H03 tissue velocity correlation | ρ > 0.70 | ⏳ TBD |
| Top 20 proteins I² (heterogeneity) | < 50% | ⏳ TBD |
| Stable proteins identified | ≥ 15/20 | ⏳ TBD |

## Critical Question

**Are Iterations 01-03 findings ROBUST or OVERFIT?**

- High R² and AUC on our dataset could be artifacts
- External validation is GOLD STANDARD
- If models fail on external data → overfitting confirmed
- If models succeed → findings are generalizable and clinically translatable

## References

- **Task Document:** `../01_task.md`
- **Data Acquisition Plan:** `DATA_ACQUISITION_PLAN.md`
- **Results Report:** `90_results_claude_code.md`
- **H08 S100 Model:** `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/`
- **H06 Biomarkers:** `/iterations/iteration_02/hypothesis_06_biomarker_discovery_panel/`
- **H03 Velocities:** `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/`

## Contact

**Agent:** claude_code
**Project:** ECM-Atlas Iteration 04
**Hypothesis:** H13 - Independent Dataset Validation
**Date:** 2025-10-21

---

**🚨 NEXT ACTION REQUIRED:**
Download supplementary data from eLife and PRIDE, then run harmonization pipeline.
