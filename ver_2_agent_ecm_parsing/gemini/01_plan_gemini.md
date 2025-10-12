# Plan for ECM Atlas Dataset Parsing - Version 2

This plan outlines the strategy to parse 11 proteomic datasets, adhering to the "Knowledge-First" approach mandated by the task.

## Phase 0: Knowledge Base Creation (COMPLETE FIRST)

This is the most critical phase. No parsing code will be written until this is complete.

1.  **Create Directory Structure:**
    *   Create the `knowledge_base` directory.
    *   Create the `knowledge_base/01_paper_analysis` directory.

2.  **Create `00_dataset_inventory.md`:**
    *   Use `glob` to find all relevant Excel and TSV files in the `data_raw` directory.
    *   For each file, I will read it to determine the number of rows and sheets.
    *   Create a markdown table with the required information.

3.  **Create Paper Analysis Files:**
    *   For each of the 11 studies, I will create an analysis file in `knowledge_base/01_paper_analysis/`.
    *   I will read the corresponding paper from the `pdf/` directory to understand the experimental design, age groups, and abundance calculation methods.
    *   I will read the data files to understand their structure and map the columns to the unified schema.
    *   I will fill out the analysis template for each study, including the column mapping, abundance calculation, and any ambiguities.

4.  **Create `02_column_mapping_strategy.md`:**
    *   Based on the paper analysis, I will create a document that summarizes the column mapping strategy for all studies.

5.  **Create `03_normalization_strategy.md`:**
    *   Based on the paper analysis, I will create a document that outlines the strategy for handling different abundance units and normalization methods.

6.  **Create `04_implementation_plan.md`:**
    *   This will be the final step of the knowledge base creation.
    *   I will create a detailed plan for the implementation of the `parse_datasets.py` script, including the order of parsing, reusable functions, and error handling.

## Phase 1: Implementation

1.  **Create `parse_datasets.py`:**
    *   Create the Python script for parsing the datasets.
    *   The script will be config-driven, using the information from the knowledge base.
    *   It will contain functions for reading the different file formats, mapping the columns, and writing the output CSV files.

2.  **Implement Parsing Logic:**
    *   For each study, I will implement the parsing logic based on the `04_implementation_plan.md`.
    *   I will pay close attention to the `Parsing_Notes` column, ensuring that it contains detailed information about the data provenance.

3.  **Generate Output Files:**
    *   The script will generate 11 individual CSV files in the `data_processed/` directory.
    *   It will also generate the `ecm_atlas_unified.csv` file, which will be a concatenation of all the individual CSV files.

## Phase 2: Validation and Documentation

1.  **Create `validation_report.md`:**
    *   The script will also generate a validation report with statistics about the parsed data, including row counts, null value analysis, and protein ID format compliance.

2.  **Create `metadata.json`:**
    *   The script will generate a `metadata.json` file with metadata about the studies.

3.  **Create `90_results_gemini.md`:**
    *   I will complete the self-evaluation file, providing evidence for each of the 25 success criteria.

By following this plan, I will ensure that all the requirements of the task are met, with a strong emphasis on the "Knowledge-First" approach, traceability, and data quality.
