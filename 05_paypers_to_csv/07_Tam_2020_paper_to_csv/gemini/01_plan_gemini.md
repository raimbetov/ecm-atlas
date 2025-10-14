# Plan for Tam 2020 Dataset Conversion

This plan outlines the steps to convert the Tam et al. 2020 dataset into a standardized CSV format, following the instructions in `00_TASK_TAM_2020_CSV_CONVERSION.md`.

## 1.0 File Reconnaissance

1.  **Verify File Existence:** Check that the source Excel file `data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx` exists.
2.  **Inspect Excel Structure:**
    *   Load the Excel file using `pandas`.
    *   List the sheet names to confirm the presence of "Raw data" and "Sample information".
    *   Read both sheets into pandas DataFrames.
    *   Verify the dimensions of the DataFrames.
3.  **Validate Metadata Join:**
    *   Extract profile names from the column headers of the "Raw data" sheet.
    *   Compare them with the profile names in the "Sample information" sheet to ensure they match.
4.  **Perform Data Quality Checks:**
    *   Check for null Protein IDs and Gene names.
    *   Check for empty LFQ intensity columns.
    *   Check for missing compartment labels in the metadata.

## 2.0 Data Parsing

1.  **Load and Prepare Data:**
    *   Load the "Raw data" and "Sample information" sheets.
    *   Clean up column names.
2.  **Reshape to Long Format:**
    *   Use `pandas.melt` to transform the wide-format LFQ intensity data into a long format.
3.  **Join with Metadata:**
    *   Merge the long-format data with the metadata DataFrame.
4.  **Add Metadata Columns:**
    *   Create `Age`, `Age_Bin`, `Tissue_Compartment`, and `Sample_ID` columns.

## 3.0 Schema Standardization

1.  **Create Standardized DataFrame:**
    *   Create a new DataFrame that adheres to the 15-column wide-format schema specified in the task file. This will be done after the long-format processing is complete.
2.  **Clean and Validate:**
    *   Remove rows with missing `Protein_ID` or `Abundance`.
    *   Ensure all required columns are present and have no null values.
    *   Verify that the three tissue compartments are correctly represented.

## 4.0 Protein Annotation

1.  **Load Reference Data:**
    *   Load the `human_matrisome_v2.csv` file.
2.  **Create Lookup Dictionaries:**
    *   Build dictionaries for matching by gene symbol, UniProt ID, and synonyms.
3.  **Apply Annotation:**
    *   Define and apply a function to annotate each protein based on the hierarchical matching strategy.
4.  **Validate Annotation:**
    *   Calculate the annotation coverage and check if it meets the >=90% target.
    *   Validate the annotation of known marker proteins.

## 5.0 Z-Score Normalization

1.  **Convert to Wide Format:**
    *   Group the data by protein and compartment, creating `Abundance_Young` and `Abundance_Old` columns.
2.  **Split by Compartment:**
    *   Create separate DataFrames for NP, IAF, and OAF.
3.  **Calculate Z-Scores:**
    *   For each compartment DataFrame:
        *   Check data skewness and apply a log2 transformation if necessary.
        *   Calculate z-scores for `Abundance_Young` and `Abundance_Old`.
        *   Validate that the resulting z-scores have a mean of ~0 and a standard deviation of ~1.
4.  **Export Z-Score Files:**
    *   Save the z-score normalized data for each compartment into separate CSV files.

## 6.0 Quality Validation and Final Export

1.  **Run Final Validation:**
    *   Perform all the quantitative and biological checks listed in the task file.
2.  **Generate Metadata:**
    *   Create a JSON file containing metadata about the dataset and the parsing process.
3.  **Create Final Report:**
    *   Generate a `90_results_gemini.md` file summarizing the entire process, including validation results and paths to the output files.
4.  **Export All Artifacts:**
    *   Ensure all specified output files (`.csv`, `.json`, `.md`) are created in the correct location.
