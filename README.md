# ECM-Atlas

**Unified database for cross-study analysis of age-related extracellular matrix changes**

ECM-Atlas aggregates 13 proteomic studies (128 files, 15 papers, 2017-2023) tracking age-related ECM protein changes across tissues and organisms, processed via autonomous agent pipeline into unified database with interactive dashboards for cross-tissue aging signature analysis.

---

## Quick Start

### View Interactive Dashboard
```bash
cd 10_unified_dashboard_2_tabs
./start_servers.sh
```
Open http://localhost:8083/dashboard.html

### Process New Dataset
```bash
source env/bin/activate
python 11_subagent_for_data_ingestion/autonomous_agent.py "data_raw/Author et al. - Year/"
```

The autonomous agent will:
- ✅ Identify data files and create configuration
- ✅ Process dataset through normalization pipeline
- ✅ Merge to unified database
- ✅ Calculate z-scores

**Track progress in real-time:**
```bash
tail -f XX_Author_Year_paper_to_csv/agent_log.md
```

### Access Main Database
```bash
# Main unified database with z-scores (all proteins)
08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

# Alternative versions:
# - merged_ecm_aging_zscore_ECM_ONLY.csv (ECM proteins only)
# - merged_ecm_aging_zscore_enriched.csv (with UniProt metadata)
```

---

## Project Status

**Current:** 10/15 papers processed into unified database
**Database:** 1.1 MB main file, includes z-scores for cross-study comparison

### Data Coverage
- **Studies:** 15 raw datasets, 10 processed into unified database
- **Tissues:** Lung, kidney, skin, heart, pancreas, adipose, ovary, brain, muscle, intervertebral disc
- **Species:** Human, mouse
- **Methods:** LFQ, TMT, SILAC, iTRAQ, DiLeu
- **Proteins:** ~2,000+ ECM proteins tracked across studies

---

## Repository Structure

```
ecm-atlas/
├── 00_data_raw/                   # 15 study directories with raw proteomics data
├── 01_references/                 # Matrisome reference databases
│   ├── human_matrisome_v2.csv
│   └── mouse_matrisome_v2.csv
├── 02_documentation/              # Project documentation and research notes
├── 04_compilation_of_papers/      # Paper compilation and processing status
├── 05_papers_to_csv/              # 10 processed studies (CSV format)
├── 08_merged_ecm_dataset/         # Unified database (main output)
│   ├── merged_ecm_aging_zscore.csv           # Main database
│   ├── merged_ecm_aging_zscore_enriched.csv  # With UniProt metadata
│   ├── merged_ecm_aging_zscore_ECM_ONLY.csv  # ECM proteins only
│   └── backups/                              # Automatic backups
├── 10_unified_dashboard_2_tabs/   # Interactive web dashboard
│   ├── api_server.py              # Flask API backend
│   ├── dashboard.html             # Main dashboard
│   ├── datasets/                  # Individual dataset pages
│   └── static/                    # Frontend assets
├── 11_subagent_for_data_ingestion/ # Autonomous processing pipeline
│   ├── autonomous_agent.py        # Main orchestrator (Phase 0-3)
│   ├── merge_to_unified.py        # Phase 2: Merge script
│   ├── universal_zscore_function.py # Phase 3: Z-score calculation
│   └── *.md                       # Detailed documentation
├── 13_1_meta_insights/            # Multi-agent analysis results
├── CLAUDE.md                      # AI agent project guide
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## Key Features

### Autonomous Processing Pipeline
The pipeline processes raw proteomic datasets through 4 phases:

**Phase 0: Reconnaissance**
- Identifies data files in paper folder
- Generates configuration template
- Extracts study metadata

**Phase 1: Data Normalization**
- Converts Excel/TSV to long format
- Assigns age groups (young vs. old)
- Enriches protein metadata via UniProt API
- Annotates ECM proteins using matrisome reference
- Converts to wide format (ECM proteins only)
- Validates quality and completeness

**Phase 2: Merge to Unified Database**
- Adds processed study to `merged_ecm_aging_zscore.csv`
- Creates automatic backups
- Updates metadata registry

**Phase 3: Z-Score Calculation**
- Calculates within-study z-scores per tissue compartment
- Handles missing data appropriately (50-80% NaN is normal)
- Preserves zero values (detected absence)
- Enables cross-study comparison via standardized effect sizes

**See [11_subagent_for_data_ingestion/00_START_HERE.md](11_subagent_for_data_ingestion/00_START_HERE.md) for detailed workflow**

### Interactive Dashboard
- Heatmaps for protein abundance patterns
- Volcano plots for differential expression
- Scatter plots for correlation analysis
- Cross-dataset comparison tools
- Individual study views and multi-study analysis

### Unified Database Schema
```csv
Protein_ID, Gene_Symbol, Tissue, Species, Age, Abundance, Z_score, Study_ID,
Tissue_Compartment, Match_Confidence, Matrisome_Category, Protein_Name
```

---

## Scientific Context

**Problem:** The extracellular matrix (ECM) composition changes with age, but there's no unified understanding of these changes across tissues and organisms.

**Solution:** Aggregate published proteomic datasets with tissue-level granularity, normalize using z-scores, and enable direct comparison of matrisomal protein abundances.

**Goal:** Identify universal ECM aging signatures and tissue-specific patterns.

### Data Handling Notes
- **NaN (missing data):** 50-80% missingness is normal in label-free quantification (LFQ). We preserve NaN values without imputation.
- **Zero values (detected absence):** 0.0 abundance means protein was measured as undetectable/absent. These are included in statistical calculations.
- **Normalization:** Within-study z-scores calculated per compartment/tissue.
- **Batch correction:** Z-score normalization is applied within each study separately to account for systematic differences between datasets (different labs, instruments, protocols). This approach preserves biological signal while removing technical batch effects. Cross-study comparisons are enabled through standardized effect sizes rather than raw abundances.

---

## Development Setup

### Environment
```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Common Tasks

```bash
# Start dashboard
cd 10_unified_dashboard_2_tabs && ./start_servers.sh

# Process new study (fully automated)
python 11_subagent_for_data_ingestion/autonomous_agent.py "data_raw/Author et al. - Year/"

# Monitor processing in real-time
tail -f XX_Author_Year_paper_to_csv/agent_log.md

# Manual workflow (if needed)
# Phase 2: Merge to unified database
python 11_subagent_for_data_ingestion/merge_to_unified.py 05_Study_paper_to_csv/Study_wide_format.csv

# Phase 3: Calculate z-scores
python 11_subagent_for_data_ingestion/universal_zscore_function.py Study_ID Tissue
```

---

## Documentation

- **Getting Started:** [11_subagent_for_data_ingestion/00_START_HERE.md](11_subagent_for_data_ingestion/00_START_HERE.md)
- **Pipeline Flowchart:** [11_subagent_for_data_ingestion/00_PIPELINE_FLOWCHART.md](11_subagent_for_data_ingestion/00_PIPELINE_FLOWCHART.md)
- **Agent Guide:** [11_subagent_for_data_ingestion/AUTONOMOUS_AGENT_GUIDE.md](11_subagent_for_data_ingestion/AUTONOMOUS_AGENT_GUIDE.md)
- **Data Normalization:** [11_subagent_for_data_ingestion/01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md](11_subagent_for_data_ingestion/01_LFQ_DATASET_NORMALIZATION_AND_MERGE.md)
- **Z-score Calculation:** [11_subagent_for_data_ingestion/02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md](11_subagent_for_data_ingestion/02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md)
- **Processing Status:** [04_compilation_of_papers/00_README_compilation.md](04_compilation_of_papers/00_README_compilation.md)
- **Project Guide for AI Agents:** [CLAUDE.md](CLAUDE.md)

---

## Data Sources

### Processed Studies (10/15)
Studies converted to standardized CSV format in [05_papers_to_csv/](05_papers_to_csv/):
- Caldeira 2017, Randles 2021, Tam 2020, Ouni 2022, Angelidis 2019
- Dipali 2023, Li (Dermis) 2021, Tsumagari 2023, Lofaro 2021, Schuler 2021

### Raw Studies (15 total)
All raw proteomic datasets available in [00_data_raw/](00_data_raw/)

### Reference Databases
- Human matrisome v2 (core + associated ECM proteins)
- Mouse matrisome v2
- Located in [01_references/](01_references/)

---

## Workflow: Adding a New Dataset

### Option A: Fully Automated (Recommended)

```bash
# 1. Run autonomous agent
python 11_subagent_for_data_ingestion/autonomous_agent.py "data_raw/Author et al. - Year/"

# 2. If prompted, edit configuration
nano XX_Author_Year_paper_to_csv/study_config.json

# 3. Re-run agent
python 11_subagent_for_data_ingestion/autonomous_agent.py "data_raw/Author et al. - Year/"

# Done! All phases completed automatically.
```

**Output:**
```
XX_Author_Year_paper_to_csv/
├── agent_log.md                      # Complete execution log
├── agent_state.json                  # Current state (for debugging)
├── study_config.json                 # Configuration
└── Author_Year_wide_format.csv       # Processed dataset

08_merged_ecm_dataset/
├── merged_ecm_aging_zscore.csv       # UPDATED with new study + z-scores
└── backups/                          # Automatic backups
```

### Option B: Manual Step-by-Step

See [11_subagent_for_data_ingestion/00_START_HERE.md](11_subagent_for_data_ingestion/00_START_HERE.md) for manual processing instructions.

**After adding dataset:** Update [04_compilation_of_papers/00_README_compilation.md](04_compilation_of_papers/00_README_compilation.md) with processing status

---

**Note:** This is an active research project. Database schema and analysis methods are subject to refinement as new studies are processed.
