# Plan for Randles 2021 Dataset Conversion

This plan outlines the steps to convert the Randles et al. 2021 dataset into a standardized CSV format.

## Phase 1: Reconnaissance
1.  **Verify File Existence:** Check if the file `data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx` exists.
2.  **Inspect Excel Structure:** Read the Excel file and check the sheet names, dimensions, and columns.

## Phase 2: Data Parsing
1.  **Load and Filter:** Load the specified sheet and filter for the required columns.
2.  **Reshape Data:** Convert the data from wide to long format.
3.  **Parse Metadata:** Extract compartment and age from the column names.
4.  **Add Age Bins:** Add 'Young' and 'Old' age bins.

## Phase 3: Schema Standardization
1.  **Map to Schema:** Create a new DataFrame with the 17-column standardized schema.
2.  **Clean Data:** Remove rows with null `Protein_ID` or `Abundance`.
3.  **Validate Schema:** Check for null values in required columns.

## Phase 4: Protein Annotation
1.  **Load Reference:** Load the human matrisome reference file.
2.  **Create Lookups:** Create dictionaries for matching by gene symbol, UniProt ID, and synonyms.
3.  **Annotate Data:** Apply the multi-level annotation strategy.
4.  **Validate Annotation:** Check the annotation coverage and validate known markers.

## Phase 5: Quality Validation and Export
1.  **Run Final Validation:** Perform all validation checks.
2.  **Export Data:** Export the final DataFrame to a CSV file.
3.  **Generate Metadata:** Create and export a JSON metadata file.
4.  **Create Reports:** Generate reports for annotation and unmatched proteins.
5.  **Final Results:** Collate all results and summaries into the final results file.
