# Angelidis 2019 Processing Log (LEGACY Format)

**Study ID:** Angelidis_2019
**Start Time:** 2025-10-13 15:50:00
**Input:** data_raw/Angelidis et al. - 2019/
**Output Format:** LEGACY (merged_ecm_aging_zscore.csv compatible)

---

## PHASE 0: Reconnaissance

Study parameters extracted from comprehensive analysis:
- **Data file:** `data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx`
- **Sheet:** `Proteome`
- **Species:** Mus musculus
- **Tissue:** Lung
- **Method:** Label-free LC-MS/MS (MaxQuant LFQ)
- **Young ages:** 3 months (4 replicates)
- **Old ages:** 24 months (4 replicates)

✅ Configuration ready

---

## PHASE 1: Data Normalization (LEGACY Format)

[2025-10-13 16:00:03] ### Step 1.1: Load Excel file

[2025-10-13 16:00:03] Project root: /Users/Kravtsovd/projects/ecm-atlas

[2025-10-13 16:00:03] Loading: /Users/Kravtsovd/projects/ecm-atlas/data_raw/Angelidis et al. - 2019/41467_2019_8831_MOESM5_ESM.xlsx

[2025-10-13 16:00:03] Sheet: Proteome

[2025-10-13 16:00:04] ✅ Loaded: 5213 rows, 36 columns

[2025-10-13 16:00:04] 
Columns found:

[2025-10-13 16:00:04]   0: Differential protein abundance in total lung tissue proteomes of young (3 months) and old mice (24 months).

[2025-10-13 16:00:04]   1: Majority protein IDs

[2025-10-13 16:00:04]   2: Protein names

[2025-10-13 16:00:04]   3: Gene names

[2025-10-13 16:00:04]   4: Student's T-test Difference_old/young [log2]

[2025-10-13 16:00:04]   5: old_1

[2025-10-13 16:00:04]   6: old_2

[2025-10-13 16:00:04]   7: old_3

[2025-10-13 16:00:04]   8: old_4

[2025-10-13 16:00:04]   9: young_1

[2025-10-13 16:00:04]   ... and 26 more

[2025-10-13 16:00:04] 
### Step 1.2: Verify sample columns exist

[2025-10-13 16:00:04] ✅ All 8 sample columns found

[2025-10-13 16:00:04] 
### Step 1.3: Extract and process protein IDs

[2025-10-13 16:00:04] Sample Protein IDs (first 5):

[2025-10-13 16:00:04]   Q9JLC8

[2025-10-13 16:00:04]   Q00898

[2025-10-13 16:00:04]   P06683

[2025-10-13 16:00:04]   Q3UWB6

[2025-10-13 16:00:04]   Q8BLX7

[2025-10-13 16:00:04] 
### Step 1.4: Transform to long format

[2025-10-13 16:00:04] ✅ Long format created: 41704 rows

[2025-10-13 16:00:04]    Expected: 5213 proteins × 8 samples = 41704 rows

[2025-10-13 16:00:04]    Match: ✅

[2025-10-13 16:00:04] 
### Step 1.5: Validate schema

[2025-10-13 16:00:04] Schema columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Tissue', 'Species', 'Age', 'Age_Unit', 'Abundance', 'Abundance_Unit', 'Method', 'Study_ID', 'Sample_ID', 'Parsing_Notes']

[2025-10-13 16:00:04] All expected columns present: ✅

[2025-10-13 16:00:04] 
### Step 1.6: Basic statistics

[2025-10-13 16:00:04] Total rows: 41704

[2025-10-13 16:00:04] Unique proteins: 5213

[2025-10-13 16:00:04] Young samples: 4

[2025-10-13 16:00:04] Old samples: 4

[2025-10-13 16:00:04] Missing abundances: 3647 (8.7%)

[2025-10-13 16:00:04] 
### Step 1.7: Save long format CSV

[2025-10-13 16:00:04] ✅ Saved: Angelidis_2019_long_format.csv

[2025-10-13 16:00:04] 
✅ PHASE 1 COMPLETE - Long format created

[2025-10-13 16:00:14] 
---


[2025-10-13 16:00:14] ## PHASE 1 (continued): ECM Annotation

[2025-10-13 16:00:14] 
### Step 2.1: Load long format and matrisome reference

[2025-10-13 16:00:14] Loaded long format: 41704 rows

[2025-10-13 16:00:14] Unique proteins: 5213

[2025-10-13 16:00:14] Loaded mouse matrisome: 1110 ECM proteins

[2025-10-13 16:00:14] 
### Step 2.2: 4-level annotation hierarchy

[2025-10-13 16:00:14] 
Level 1: Gene Symbol match

[2025-10-13 16:00:14]   Matrisome columns: ['Matrisome Division', 'Matrisome Category', 'Gene Symbol', 'Gene Name', 'Synonyms']...

[2025-10-13 16:00:14]   Matched: 2256 / 41704 rows (5.4%)

[2025-10-13 16:00:14] 
Level 2: UniProt ID match (for unmatched)

[2025-10-13 16:00:14]   Additional matches: 72 rows

[2025-10-13 16:00:14] 
Level 3: Synonym match (skipped - would need synonym mapping)

[2025-10-13 16:00:14] 
Level 4: Unmatched (non-ECM proteins)

[2025-10-13 16:00:14]   Unmatched: 39376 rows (94.4%)

[2025-10-13 16:00:14] 
### Step 2.3: Add annotation columns

[2025-10-13 16:00:14] ✅ Annotation complete

[2025-10-13 16:00:14]    ECM proteins: 2328 rows

[2025-10-13 16:00:14]    Non-ECM: 39376 rows

[2025-10-13 16:00:14] 
### Step 2.4: ECM coverage statistics

[2025-10-13 16:00:14] ECM protein coverage:

[2025-10-13 16:00:14]   Unique ECM proteins: 291

[2025-10-13 16:00:14]   Total unique proteins: 5213

[2025-10-13 16:00:14]   ECM percentage: 5.6%

[2025-10-13 16:00:14] 
### Step 2.5: Save annotated long format

[2025-10-13 16:00:15] ✅ Saved: Angelidis_2019_long_annotated.csv

[2025-10-13 16:00:15] 
✅ ECM Annotation complete

[2025-10-13 16:00:25] 
---


[2025-10-13 16:00:25] ## PHASE 1 (continued): Convert to LEGACY Format

[2025-10-13 16:00:25] 
### Step 3.1: Load annotated data and filter ECM

[2025-10-13 16:00:25] Loaded annotated data: 41704 rows

[2025-10-13 16:00:25] Filtered to ECM proteins: 2328 rows

[2025-10-13 16:00:25] Unique ECM proteins: 291

[2025-10-13 16:00:25] 
### Step 3.2: Aggregate by Age group WITH N_Profiles count

[2025-10-13 16:00:25] Young aggregation: 291 proteins

[2025-10-13 16:00:25]   N_Profiles_Young range: 0-4

[2025-10-13 16:00:25] Old aggregation: 291 proteins

[2025-10-13 16:00:25]   N_Profiles_Old range: 0-4

[2025-10-13 16:00:25] 
### Step 3.3: Merge Young and Old

[2025-10-13 16:00:25] Merged wide format: 291 proteins

[2025-10-13 16:00:25] Missing Abundance_Young: 9 (3.1%)

[2025-10-13 16:00:25] Missing Abundance_Old: 15 (5.2%)

[2025-10-13 16:00:25] 
### Step 3.4: Add LEGACY format columns

[2025-10-13 16:00:25] Added Dataset_Name, Organ, Compartment, Tissue_Compartment

[2025-10-13 16:00:25] 
### Step 3.5: Add log2 transformed abundances

[2025-10-13 16:00:25] Added log2-transformed abundances

[2025-10-13 16:00:25] 
Sample log2 transform:

[2025-10-13 16:00:25]   35.29 → 5.14

[2025-10-13 16:00:25]   27.32 → 4.77

[2025-10-13 16:00:25]   27.60 → 4.79

[2025-10-13 16:00:25] 
### Step 3.6: Rename columns to LEGACY format (underscore)

[2025-10-13 16:00:25] Renamed Matrisome columns (space → underscore)

[2025-10-13 16:00:25] 
### Step 3.7: Add placeholder z-score columns

[2025-10-13 16:00:25] Added placeholder z-score columns (will be filled by universal_zscore_function.py)

[2025-10-13 16:00:25] 
### Step 3.8: Reorder columns to match LEGACY format

[2025-10-13 16:00:25] Reordered columns to match legacy format

[2025-10-13 16:00:25] 
### Step 3.9: Validation

[2025-10-13 16:00:25] Final schema check:

[2025-10-13 16:00:25]   Columns: 25 (expected: 25)

[2025-10-13 16:00:25]   Rows: 291

[2025-10-13 16:00:25]   Unique proteins: 291

[2025-10-13 16:00:25] 
Sample rows (first 3):

[2025-10-13 16:00:25]   A0A087WR50: Fn1 - Young=35.29, Old=35.14, N_Profiles=4/4

[2025-10-13 16:00:25]   A0A087WSN6: Fn1 - Young=27.32, Old=27.54, N_Profiles=4/4

[2025-10-13 16:00:25]   A0A0R4J039: Hrg - Young=27.60, Old=28.72, N_Profiles=4/4

[2025-10-13 16:00:25] 
### Step 3.10: Save LEGACY format CSV

[2025-10-13 16:00:25] ✅ Saved: Angelidis_2019_LEGACY_format.csv

[2025-10-13 16:00:25] 
✅ PHASE 1 COMPLETE - LEGACY format ready

[2025-10-13 16:00:25] 
Next steps:

[2025-10-13 16:00:25]   1. Review output file

[2025-10-13 16:00:25]   2. Append to merged_ecm_aging_zscore.csv manually OR use merge script

[2025-10-13 16:00:25]   3. Calculate z-scores using universal_zscore_function.py
