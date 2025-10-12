# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECM-Atlas is a database and web application for aggregating and analyzing proteomic datasets tracking age-related extracellular matrix (ECM) changes across tissues. The project consolidates 13 published studies (2017-2023) with 128 files of proteomic data from various tissues, organisms, and methodologies.

**Core scientific goal:** Enable cross-study comparison of matrisomal protein abundances to identify ECM aging signatures across different tissues and organisms.

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Run Streamlit web app
streamlit run app.py

# Streamlit will typically start at http://localhost:8501
```

### Running the Randles 2021 Z-Score Dashboard

The `06_Randles_z_score_by_tissue_compartment/` directory contains an interactive visualization dashboard for kidney aging data.

**Requirements:**
```bash
pip install flask flask-cors pandas numpy plotly playwright
```

**Starting the servers:**

1. Start the API server (port 5001):
```bash
cd 06_Randles_z_score_by_tissue_compartment
python3 api_server.py
```

2. In another terminal, start the HTTP server (port 8080):
```bash
cd 06_Randles_z_score_by_tissue_compartment
python3 00_start_server.py
```

3. Open the dashboard in your browser:
```bash
open http://localhost:8080/dashboard.html
```

Or visit: http://localhost:8080/dashboard.html

**Available visualizations:**
- Heatmaps: Top 100 aging-associated proteins with color gradients
- Volcano plots: Differential expression analysis
- Scatter plots: Young vs Old comparison with ECM protein highlighting
- Bar charts: Top 20 aging markers
- Histograms: Distribution of z-score changes
- Compartment comparison: Glomerular vs Tubulointerstitial

**Screenshot validation:**
```bash
python3 01_capture_screenshots.py
```
This captures headless browser screenshots to validate all visualizations render correctly.

### Working with Jupyter Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# Notebooks are located in notebooks/ directory
```

## Repository Structure

### Key Directories

- **data_raw/**: 13 study directories containing original proteomic datasets
  - Each directory follows pattern: `[Author] et al. - [Year]/`
  - Files include Excel (.xlsx, .xls), TSV, PDF methods, and DOCX protocols
  - Total: 128 files, ~500MB

- **data_processed/**: Intended for standardized/harmonized datasets (currently empty)
  - Target format: CSV with unified schema across all studies

- **notebooks/**: Jupyter notebooks for exploratory analysis

- **documentation/**: Additional documentation files (currently empty)

- **pdf/**: Full-text publications for the 13 studies

### Key Files

- **app.py**: Streamlit application entry point (basic navigation structure)
- **data.csv**: Study metadata registry with DOIs, organisms, methods, and PRIDE IDs
- **requirements.txt**: Python dependencies (Streamlit, pandas, numpy, LangGraph, etc.)
- **00_REPO_OVERVIEW.md**: Comprehensive repository documentation
- **01_TASK_DATA_STANDARDIZATION.md**: Detailed task specification for data harmonization

## Architecture & Data Pipeline

### Current Implementation Status

The project is in early development with basic infrastructure:

1. **Data Collection** (Complete): 13 published studies with proteomic datasets collected
2. **Metadata Registry** (Complete): data.csv contains study-level metadata
3. **Web Interface** (Basic): Streamlit app with navigation placeholder
4. **Data Processing** (In Progress): Standardization pipeline not yet implemented

### Planned Data Processing Pipeline

```
Raw Data (128 files)
  → Parse Excel/TSV (custom parsers per journal format)
  → Extract protein abundances, IDs, age groups
  → Standardize schema (12-column unified format)
  → Normalize abundances (z-scores, percentiles)
  → Write to data_processed/ as harmonized CSVs
  → Load into Streamlit for interactive queries
```

### Target Unified Schema

When processing datasets, aim for this standardized format:

```
Protein_ID        - UniProt ID (primary identifier)
Protein_Name      - Full protein name
Gene_Symbol       - Gene nomenclature
Tissue            - Organ/tissue type (lung, skin, kidney, etc.)
Species           - Organism (Mus musculus, Homo sapiens, Bos taurus)
Age               - Numeric age value
Age_Unit          - Time unit (years, months, weeks)
Abundance         - Quantitative protein measurement
Abundance_Unit    - Measurement unit (LFQ intensity, ppm, spectral counts)
Method            - Proteomic technique (LFQ, TMT, SILAC, iTRAQ, DiLeu)
Study_ID          - Publication identifier (DOI or PMID)
Sample_ID         - Biological/technical replicate ID
```

## Data Standardization Challenges

### Heterogeneity Issues

1. **Protein identifiers**: Studies use different databases (UniProt vs Gene symbols vs Ensembl)
   - Solution: Use UniProt Mapping API to harmonize to UniProt IDs

2. **Abundance metrics**: Different units across studies (raw intensities, LFQ, spectral counts)
   - Solution: Apply within-study z-score normalization, then cross-study percentile ranking

3. **File formats**: Varies by publisher
   - Nature journals: `MOESM[N]_ESM.xlsx`
   - eLife: `fig[N]-data[N].xlsx`
   - Frontiers: `Data Sheet [N].xlsx`

4. **Age definitions**: Different age bins and species-specific lifespans
   - Solution: Normalize as percentage of species maximum lifespan

### Parsing Strategy

Each study requires custom parsing logic due to varying Excel structures:
- Identify header rows (often multiple rows for column names)
- Locate protein identifier columns (can be labeled as "Protein IDs", "Gene Symbol", "Accession")
- Extract abundance columns (look for "Intensity", "LFQ", "Abundance", age-related labels)
- Handle missing values appropriately (median imputation or exclusion)

## Development Workflow

### Adding New Study Data

1. Add study directory to data_raw/ with format: `[Author] et al. - [Year]/`
2. Update data.csv with metadata (DOI, organ, organism, method, PRIDE ID)
3. Create custom parser for the study's Excel/TSV format
4. Run standardization pipeline to generate data_processed/ output
5. Update Streamlit app to include new study in queries

### Testing Data Processing

When implementing data processing scripts:
- Start with 1-2 representative studies (e.g., Angelidis 2019, Dipali 2023)
- Validate output schema matches 12-column standard
- Check protein ID coverage (>70% should have valid UniProt IDs)
- Verify abundance normalization preserves biological signal

## Key Dependencies

### Core Stack
- **Streamlit**: Web application framework
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical operations
- **openpyxl**: Reading .xlsx files
- **xlrd**: Reading older .xls files

### Analysis Tools (Planned)
- **scipy**: Statistical functions for normalization
- **scikit-learn**: Z-score standardization, clustering
- **Matrisome AnalyzeR**: ECM protein classification (R package, external)

### LLM/Agent Framework (Experimental)
The repository includes LangGraph and related dependencies, suggesting potential for:
- Natural language queries to the database
- Automated literature extraction
- Chatbot interface for protein aging signatures

## Scientific Context

### The ECM Aging Problem

The extracellular matrix (ECM) provides mechanical support and biochemical signaling to cells. Its composition changes with age and disease, but there's no unified understanding of these changes at the tissue level. This project addresses that by:

1. Aggregating scattered proteomic datasets from multiple publications
2. Standardizing different measurement methodologies
3. Enabling direct comparison of protein abundances across tissues/ages
4. Identifying universal vs tissue-specific aging signatures

### Matrisome Classification

The "matrisome" is the complete set of ECM proteins and ECM-associated proteins. Use Matrisome AnalyzeR to classify proteins into categories:
- Core matrisome: Collagens, glycoproteins, proteoglycans
- Matrisome-associated: ECM regulators, secreted factors, ECM-affiliated proteins

### Study Characteristics (from data.csv)

The 13 studies cover:
- **Organisms**: Mouse (majority), Human, Cow
- **Tissues**: Lung, skin, kidney, pancreas, ovary, brain, intervertebral disc, skeletal muscle
- **Methods**: LFQ (most common), TMT, SILAC, iTRAQ, DiLeu, QconCAT
- **Raw data repositories**: PRIDE (European), MassIVE (US), jPOST (Japan)

## Known Limitations

1. **No automated testing**: No test suite currently exists
2. **Manual file paths**: Hard-coded paths may need adjustment for different environments
3. **No database backend**: Currently file-based, may need SQLite/PostgreSQL for production
4. **Incomplete documentation**: Processing scripts not yet implemented
5. **No protein ID validation**: Need to verify UniProt IDs are current and correct

## Future Enhancements

From 01_TASK_DATA_STANDARDIZATION.md, planned features include:
- Automated Excel/TSV parsing for all 13 studies
- Within-study and cross-study normalization
- Interactive Streamlit filters (tissue, age, protein family)
- Protein abundance heatmaps and clustering
- Natural language chatbot interface for queries
- Integration with CELLxGENE or AnVIL metadata standards
