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
python 11_subagent_for_LFQ_ingestion/autonomous_agent.py "data_raw/Study_Name/"
```

### Access Main Database
```bash
# Main unified database with z-scores
08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

# Enriched version with UniProt metadata
08_merged_ecm_dataset/merged_ecm_aging_zscore_enriched.csv
```

---

## Project Status

**Current:** 7/15 papers processed into unified database
**Database:** 1.1 MB, includes z-scores for cross-study comparison
**Last updated:** 2025-10-14

### Data Coverage
- **Tissues:** Lung, kidney, skin, heart, pancreas, adipose, ovary, brain
- **Species:** Human, mouse
- **Methods:** LFQ, TMT, SILAC, iTRAQ, DiLeu
- **Proteins:** ~2,000+ ECM proteins tracked

---

## Repository Structure

```
ecm-atlas/
├── data_raw/                    # 19 study directories with raw proteomics data
├── 05_papers_to_csv/           # 7 processed studies (CSV format)
├── 08_merged_ecm_dataset/      # Unified database (main output)
├── 10_unified_dashboard_2_tabs/ # Interactive web dashboard
├── 11_subagent_for_LFQ_ingestion/ # Autonomous processing pipeline
├── 12_priority_research_questions/ # Analysis results
├── knowledge_base/             # Documentation and guides
└── reports/                    # Analysis reports

```

---

## Key Features

### Autonomous Processing Pipeline
- **Phase 0:** Reconnaissance - identify data files
- **Phase 1:** Normalization - Excel/TSV → standardized CSV
- **Phase 2:** Merge - combine into unified database
- **Phase 3:** Z-score calculation - enable cross-study comparison

### Interactive Dashboard
- Heatmaps for protein abundance patterns
- Volcano plots for differential expression
- Scatter plots for correlation analysis
- Cross-dataset comparison tools

### Unified Database Schema
```csv
Protein_ID, Gene_Symbol, Tissue, Species, Age, Abundance, Z_score, Study_ID
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

# Process new study
python 11_subagent_for_LFQ_ingestion/autonomous_agent.py "data_raw/Study/"

# Run analysis scripts
python analyze_aging_signatures.py
python find_common_signatures.py
```

---

## Documentation

- **Getting Started:** `11_subagent_for_LFQ_ingestion/00_START_HERE.md`
- **Agent Guide:** `11_subagent_for_LFQ_ingestion/AUTONOMOUS_AGENT_GUIDE.md`
- **Z-score Calculation:** `11_subagent_for_LFQ_ingestion/02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md`
- **Processing Status:** `04_compilation_of_papers/00_README_compilation.md`
- **Project Guide:** `CLAUDE.md`

---

## Data Sources

### Processed Studies (7/15)
Studies converted to standardized CSV format in `05_papers_to_csv/`

### Raw Studies (19 total)
All raw proteomic datasets available in `data_raw/` with corresponding PDFs

### Publications
15 full-text papers (2017-2023) providing methodological context

---

## Contributing

When adding a new dataset:
1. Run autonomous agent: `python 11_subagent_for_LFQ_ingestion/autonomous_agent.py "data_raw/New_Study/"`
2. Verify output in `05_papers_to_csv/`
3. Update `04_compilation_of_papers/00_README_compilation.md` with processing status
4. Re-merge into unified database if needed

---

## License

[To be specified]

---

**Note:** This is an active research project. Database schema and analysis methods are subject to refinement as new studies are processed.
