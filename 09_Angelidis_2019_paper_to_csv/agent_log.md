# Angelidis 2019 Processing Log

**Study ID:** Angelidis_2019
**Start Time:** 2025-10-13 15:25:00
**Input:** data_raw/Angelidis et al. - 2019/
**Comprehensive Analysis:** 04_compilation_of_papers/01_Angelidis_2019_comprehensive_analysis.md

---

## PHASE 0: Reconnaissance

### Step 0.1: Read Comprehensive Analysis

Reading comprehensive analysis to extract:
- Data file path
- Sheet name
- Column mappings
- Age groups
- Expected output

**Extracted information:**
- **Data file:** `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`
- **Sheet:** `Proteome`
- **Species:** Mus musculus
- **Tissue:** Lung
- **Method:** Label-free LC-MS/MS (MaxQuant LFQ)
- **Young ages:** 3 months (4 replicates: young_1, young_2, young_3, young_4)
- **Old ages:** 24 months (4 replicates: old_1, old_2, old_3, old_4)
- **Expected rows:** 5,214 proteins × 8 samples = 41,712 rows

✅ Comprehensive analysis loaded

---

## PHASE 1: Data Normalization

### Step 1.1: Verify data file exists

[2025-10-13 15:28:09] ### Step 1.1: Load Excel file

[2025-10-13 15:28:09] Project root: /Users/Kravtsovd/projects/ecm-atlas

[2025-10-13 15:28:09] Loading: /Users/Kravtsovd/projects/ecm-atlas/data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx

[2025-10-13 15:28:09] Sheet: Proteome

[2025-10-13 15:28:11] ✅ Loaded: 5213 rows, 36 columns

[2025-10-13 15:28:11] 
Columns found:

[2025-10-13 15:28:11]   0: Differential protein abundance in total lung tissue proteomes of young (3 months) and old mice (24 months).

[2025-10-13 15:28:11]   1: Majority protein IDs

[2025-10-13 15:28:11]   2: Protein names

[2025-10-13 15:28:11]   3: Gene names

[2025-10-13 15:28:11]   4: Student's T-test Difference_old/young [log2]

[2025-10-13 15:28:11]   5: old_1

[2025-10-13 15:28:11]   6: old_2

[2025-10-13 15:28:11]   7: old_3

[2025-10-13 15:28:11]   8: old_4

[2025-10-13 15:28:11]   9: young_1

[2025-10-13 15:28:11]   ... and 26 more

[2025-10-13 15:28:11] 
### Step 1.2: Verify sample columns exist

[2025-10-13 15:28:11] ✅ All 8 sample columns found

[2025-10-13 15:28:11] 
### Step 1.3: Extract and process protein IDs

[2025-10-13 15:28:11] Sample Protein IDs (first 5):

[2025-10-13 15:28:11]   Q9JLC8

[2025-10-13 15:28:11]   Q00898

[2025-10-13 15:28:11]   P06683

[2025-10-13 15:28:11]   Q3UWB6

[2025-10-13 15:28:11]   Q8BLX7

[2025-10-13 15:28:11] 
### Step 1.4: Transform to long format

[2025-10-13 15:28:11] ✅ Long format created: 41704 rows

[2025-10-13 15:28:11]    Expected: 5213 proteins × 8 samples = 41704 rows

[2025-10-13 15:28:11]    Match: ✅

[2025-10-13 15:28:11] 
### Step 1.5: Validate schema

[2025-10-13 15:28:11] Schema columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Tissue', 'Species', 'Age', 'Age_Unit', 'Abundance', 'Abundance_Unit', 'Method', 'Study_ID', 'Sample_ID', 'Parsing_Notes']

[2025-10-13 15:28:11] All expected columns present: ✅

[2025-10-13 15:28:11] 
### Step 1.6: Basic statistics

[2025-10-13 15:28:11] Total rows: 41704

[2025-10-13 15:28:11] Unique proteins: 5213

[2025-10-13 15:28:11] Young samples: 4

[2025-10-13 15:28:11] Old samples: 4

[2025-10-13 15:28:11] Missing abundances: 3647 (8.7%)

[2025-10-13 15:28:11] 
### Step 1.7: Save long format CSV

[2025-10-13 15:28:11] ✅ Saved: Angelidis_2019_long_format.csv

[2025-10-13 15:28:11] 
✅ PHASE 1 COMPLETE - Long format created

[2025-10-13 15:29:02] 
---


[2025-10-13 15:29:02] ## PHASE 1 (continued): ECM Annotation

[2025-10-13 15:29:02] 
### Step 2.1: Load long format and matrisome reference

[2025-10-13 15:29:02] Loaded long format: 41704 rows

[2025-10-13 15:29:02] Unique proteins: 5213

[2025-10-13 15:29:02] Loaded mouse matrisome: 1110 ECM proteins

[2025-10-13 15:29:02] 
### Step 2.2: 4-level annotation hierarchy

[2025-10-13 15:29:02] 
Level 1: Gene Symbol match

[2025-10-13 15:29:02] 
❌ ERROR: "['Category', 'Division'] not in index"

[2025-10-13 15:29:02] 
```
Traceback (most recent call last):
  File "/Users/Kravtsovd/projects/ecm-atlas/09_Angelidis_2019_paper_to_csv/annotate_ecm.py", line 133, in <module>
    df_result = main()
                ^^^^^^
  File "/Users/Kravtsovd/projects/ecm-atlas/09_Angelidis_2019_paper_to_csv/annotate_ecm.py", line 46, in main
    df_matrisome[['Gene Symbol', 'Category', 'Division']],
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Kravtsovd/projects/chrome-extension-tcs/venv/lib/python3.11/site-packages/pandas/core/frame.py", line 4119, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/Kravtsovd/projects/chrome-extension-tcs/venv/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 6212, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "/Users/Kravtsovd/projects/chrome-extension-tcs/venv/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 6264, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['Category', 'Division'] not in index"

```

[2025-10-13 15:30:02] 
---


[2025-10-13 15:30:02] ## PHASE 1 (continued): ECM Annotation

[2025-10-13 15:30:02] 
### Step 2.1: Load long format and matrisome reference

[2025-10-13 15:30:02] Loaded long format: 41704 rows

[2025-10-13 15:30:02] Unique proteins: 5213

[2025-10-13 15:30:02] Loaded mouse matrisome: 1110 ECM proteins

[2025-10-13 15:30:02] 
### Step 2.2: 4-level annotation hierarchy

[2025-10-13 15:30:02] 
Level 1: Gene Symbol match

[2025-10-13 15:30:02]   Matrisome columns: ['Matrisome Division', 'Matrisome Category', 'Gene Symbol', 'Gene Name', 'Synonyms']...

[2025-10-13 15:30:02]   Matched: 2256 / 41704 rows (5.4%)

[2025-10-13 15:30:02] 
Level 2: UniProt ID match (for unmatched)

[2025-10-13 15:30:02]   Additional matches: 72 rows

[2025-10-13 15:30:02] 
Level 3: Synonym match (skipped - would need synonym mapping)

[2025-10-13 15:30:02] 
Level 4: Unmatched (non-ECM proteins)

[2025-10-13 15:30:02]   Unmatched: 39376 rows (94.4%)

[2025-10-13 15:30:02] 
### Step 2.3: Add annotation columns

[2025-10-13 15:30:02] ✅ Annotation complete

[2025-10-13 15:30:02]    ECM proteins: 2328 rows

[2025-10-13 15:30:02]    Non-ECM: 39376 rows

[2025-10-13 15:30:02] 
### Step 2.4: ECM coverage statistics

[2025-10-13 15:30:02] ECM protein coverage:

[2025-10-13 15:30:02]   Unique ECM proteins: 291

[2025-10-13 15:30:02]   Total unique proteins: 5213

[2025-10-13 15:30:02]   ECM percentage: 5.6%

[2025-10-13 15:30:02] 
### Step 2.5: Save annotated long format

[2025-10-13 15:30:03] ✅ Saved: Angelidis_2019_long_annotated.csv

[2025-10-13 15:30:03] 
✅ ECM Annotation complete

[2025-10-13 15:30:42] 
---


[2025-10-13 15:30:42] ## PHASE 1 (continued): Wide Format Conversion

[2025-10-13 15:30:42] 
### Step 3.1: Load annotated data and filter ECM

[2025-10-13 15:30:42] Loaded annotated data: 41704 rows

[2025-10-13 15:30:42] Filtered to ECM proteins: 2328 rows

[2025-10-13 15:30:42] Unique ECM proteins: 291

[2025-10-13 15:30:42] 
### Step 3.2: Aggregate by Age group

[2025-10-13 15:30:42] Young aggregation: 291 proteins

[2025-10-13 15:30:42] Old aggregation: 291 proteins

[2025-10-13 15:30:42] 
### Step 3.3: Merge Young and Old

[2025-10-13 15:30:42] Merged wide format: 291 proteins

[2025-10-13 15:30:42] Missing Abundance_Young: 9 (3.1%)

[2025-10-13 15:30:42] Missing Abundance_Old: 15 (5.2%)

[2025-10-13 15:30:42] 
### Step 3.4: Add Tissue_Compartment column

[2025-10-13 15:30:42] Tissue_Compartment set to: ['Lung']

[2025-10-13 15:30:42] 
### Step 3.5: Reorder columns to match expected schema

[2025-10-13 15:30:42] 
### Step 3.6: Validation

[2025-10-13 15:30:42] Final schema check:

[2025-10-13 15:30:42]   Columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Canonical_Gene_Symbol', 'Matrisome Category', 'Matrisome Division', 'Tissue', 'Tissue_Compartment', 'Species', 'Abundance_Young', 'Abundance_Old', 'Method', 'Study_ID', 'Match_Level', 'Match_Confidence']

[2025-10-13 15:30:42]   Rows: 291

[2025-10-13 15:30:42]   Unique proteins: 291

[2025-10-13 15:30:42] 
Sample rows (first 3):

[2025-10-13 15:30:42] 
❌ ERROR: Invalid format specifier '.2f if pd.notna(row['Abundance_Young']) else 'NaN'' for object of type 'float'

[2025-10-13 15:30:42] 
```
Traceback (most recent call last):
  File "/Users/Kravtsovd/projects/ecm-atlas/09_Angelidis_2019_paper_to_csv/convert_to_wide.py", line 115, in <module>
    df_result = main()
                ^^^^^^
  File "/Users/Kravtsovd/projects/ecm-atlas/09_Angelidis_2019_paper_to_csv/convert_to_wide.py", line 101, in main
    log(f"  {row['Protein_ID']}: {row['Gene_Symbol']} - Young={row['Abundance_Young']:.2f if pd.notna(row['Abundance_Young']) else 'NaN'}, Old={row['Abundance_Old']:.2f if pd.notna(row['Abundance_Old']) else 'NaN'}")
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Invalid format specifier '.2f if pd.notna(row['Abundance_Young']) else 'NaN'' for object of type 'float'

```

[2025-10-13 15:30:59] 
---


[2025-10-13 15:30:59] ## PHASE 1 (continued): Wide Format Conversion

[2025-10-13 15:30:59] 
### Step 3.1: Load annotated data and filter ECM

[2025-10-13 15:30:59] Loaded annotated data: 41704 rows

[2025-10-13 15:30:59] Filtered to ECM proteins: 2328 rows

[2025-10-13 15:30:59] Unique ECM proteins: 291

[2025-10-13 15:30:59] 
### Step 3.2: Aggregate by Age group

[2025-10-13 15:30:59] Young aggregation: 291 proteins

[2025-10-13 15:30:59] Old aggregation: 291 proteins

[2025-10-13 15:30:59] 
### Step 3.3: Merge Young and Old

[2025-10-13 15:30:59] Merged wide format: 291 proteins

[2025-10-13 15:30:59] Missing Abundance_Young: 9 (3.1%)

[2025-10-13 15:30:59] Missing Abundance_Old: 15 (5.2%)

[2025-10-13 15:30:59] 
### Step 3.4: Add Tissue_Compartment column

[2025-10-13 15:30:59] Tissue_Compartment set to: ['Lung']

[2025-10-13 15:30:59] 
### Step 3.5: Reorder columns to match expected schema

[2025-10-13 15:30:59] 
### Step 3.6: Validation

[2025-10-13 15:30:59] Final schema check:

[2025-10-13 15:30:59]   Columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Canonical_Gene_Symbol', 'Matrisome Category', 'Matrisome Division', 'Tissue', 'Tissue_Compartment', 'Species', 'Abundance_Young', 'Abundance_Old', 'Method', 'Study_ID', 'Match_Level', 'Match_Confidence']

[2025-10-13 15:30:59]   Rows: 291

[2025-10-13 15:30:59]   Unique proteins: 291

[2025-10-13 15:30:59] 
Sample rows (first 3):

[2025-10-13 15:30:59]   A0A087WR50: Fn1 - Young=35.29, Old=35.14

[2025-10-13 15:30:59]   A0A087WSN6: Fn1 - Young=27.32, Old=27.54

[2025-10-13 15:30:59]   A0A0R4J039: Hrg - Young=27.60, Old=28.72

[2025-10-13 15:30:59] 
### Step 3.7: Save wide format CSV

[2025-10-13 15:30:59] ✅ Saved: Angelidis_2019_wide_format.csv

[2025-10-13 15:30:59] 
✅ PHASE 1 COMPLETE - Wide format ready for merge
