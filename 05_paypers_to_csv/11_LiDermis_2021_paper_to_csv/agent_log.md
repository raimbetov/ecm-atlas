
[2025-10-13 21:04:38] # LiDermis 2021 Processing Log (LEGACY Format)

[2025-10-13 21:04:38] 

[2025-10-13 21:04:38] **Study ID:** LiDermis_2021

[2025-10-13 21:04:38] **Start Time:** 2025-10-13 21:04:38

[2025-10-13 21:04:38] **Input:** data_raw/Li et al. - 2021 | dermis/

[2025-10-13 21:04:38] **Output Format:** LEGACY (merged_ecm_aging_zscore.csv compatible)

[2025-10-13 21:04:38] 

[2025-10-13 21:04:38] ---

[2025-10-13 21:04:38] 

[2025-10-13 21:04:38] ## PHASE 1: Data Normalization (LEGACY Format)

[2025-10-13 21:04:38] 

[2025-10-13 21:04:38] ### Step 1.1: Load Excel file with complex headers

[2025-10-13 21:04:38] Project root: /Users/Kravtsovd/projects/ecm-atlas

[2025-10-13 21:04:38] Loading: /Users/Kravtsovd/projects/ecm-atlas/data_raw/Li et al. - 2021 | dermis/Table 2.xlsx

[2025-10-13 21:04:38] Sheet: Table S2

[2025-10-13 21:04:38] 
=== Sample names from row 3 ===

[2025-10-13 21:04:38]   Column 4: Toddler-Sample1 

[2025-10-13 21:04:38]   Column 5: Toddler-Sample2

[2025-10-13 21:04:38]   Column 6: Teenager-Sample1

[2025-10-13 21:04:38]   Column 7: Teenager-Sample2

[2025-10-13 21:04:38]   Column 8: Teenager-Sample3

[2025-10-13 21:04:38]   Column 9: Adult-Sample1

[2025-10-13 21:04:38]   Column 10: Adult-Sample2

[2025-10-13 21:04:38]   Column 11: Elderly-Sample1 

[2025-10-13 21:04:38]   Column 12: Elderly-Sample2

[2025-10-13 21:04:38]   Column 13: Elderly-Sample3

[2025-10-13 21:04:38]   Column 14: Ave_Toddler

[2025-10-13 21:04:38]   Column 15: Ave_Teenager

[2025-10-13 21:04:38]   Column 16: Ave_Adult

[2025-10-13 21:04:38]   Column 17: Ave_Elderly

[2025-10-13 21:04:38] 
✅ Found 14 sample columns

[2025-10-13 21:04:38] ✅ Loaded: 262 rows, 22 columns

[2025-10-13 21:04:38] 
Columns (first 10):

[2025-10-13 21:04:38]   0: Q5KU26

[2025-10-13 21:04:38]   1: COLEC12

[2025-10-13 21:04:38]   2: Matrisome-associated

[2025-10-13 21:04:38]   3: ECM-affiliated Proteins

[2025-10-13 21:04:38]   4: Toddler-Sample1

[2025-10-13 21:04:38]   5: Toddler-Sample2

[2025-10-13 21:04:38]   6: Teenager-Sample1

[2025-10-13 21:04:38]   7: Teenager-Sample2

[2025-10-13 21:04:38]   8: Teenager-Sample3

[2025-10-13 21:04:38]   9: Adult-Sample1

[2025-10-13 21:04:38] 
### Step 1.2: Filter samples (exclude Adult 40yr)

[2025-10-13 21:04:38] Young columns (Toddler + Teenager): 5

[2025-10-13 21:04:38]   - Toddler-Sample1

[2025-10-13 21:04:38]   - Toddler-Sample2

[2025-10-13 21:04:38]   - Teenager-Sample1

[2025-10-13 21:04:38]   - Teenager-Sample2

[2025-10-13 21:04:38]   - Teenager-Sample3

[2025-10-13 21:04:38] 
Old columns (Elderly): 3

[2025-10-13 21:04:38]   - Elderly-Sample1

[2025-10-13 21:04:38]   - Elderly-Sample2

[2025-10-13 21:04:38]   - Elderly-Sample3

[2025-10-13 21:04:38] 
❌ Excluded Adult columns (40yr, middle-aged): 2

[2025-10-13 21:04:38]   - Adult-Sample1

[2025-10-13 21:04:38]   - Adult-Sample2

[2025-10-13 21:04:38] 
✅ Retention: 8/14 = 57.1%

[2025-10-13 21:04:38] 
### Step 1.3: Extract protein identifiers

[2025-10-13 21:04:38] Protein ID column: 'Q5KU26'

[2025-10-13 21:04:38] Gene Symbol column: 'COLEC12'

[2025-10-13 21:04:38] 
Sample Protein IDs (first 5):

[2025-10-13 21:04:38]   P16112

[2025-10-13 21:04:38]   O75487

[2025-10-13 21:04:38]   P13928

[2025-10-13 21:04:38]   P29508

[2025-10-13 21:04:38]   Q8N6G6

[2025-10-13 21:04:38] 
### Step 1.4: Protein_Name lookup via UniProt API

[2025-10-13 21:04:38] ⚠️  Warning: This may take 5-10 minutes for ~260 proteins

[2025-10-13 21:04:38] Fetching names for 262 unique proteins...

[2025-10-13 21:05:10]   Progress: 50/262 (19.1%)

[2025-10-13 21:05:41]   Progress: 100/262 (38.2%)

[2025-10-13 21:06:13]   Progress: 150/262 (57.3%)

[2025-10-13 21:06:46]   Progress: 200/262 (76.3%)

[2025-10-13 21:07:18]   Progress: 250/262 (95.4%)

[2025-10-13 21:07:26] ✅ Fetched 262 protein names

[2025-10-13 21:07:26] ⚠️  0 proteins without names

[2025-10-13 21:07:26] 
### Step 1.5: Transform to long format

[2025-10-13 21:07:26] ✅ Long format created: 2096 rows

[2025-10-13 21:07:26]    Expected: 262 proteins × 8 samples = 2096 rows

[2025-10-13 21:07:26]    Match: ✅

[2025-10-13 21:07:26] 
### Step 1.6: Validate schema

[2025-10-13 21:07:26] Schema columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Tissue', 'Species', 'Age', 'Age_Unit', 'Abundance', 'Abundance_Unit', 'Method', 'Study_ID', 'Sample_ID', 'Parsing_Notes']

[2025-10-13 21:07:26] All expected columns present: ✅

[2025-10-13 21:07:26] 
### Step 1.7: Basic statistics

[2025-10-13 21:07:26] Total rows: 2096

[2025-10-13 21:07:26] Unique proteins: 262

[2025-10-13 21:07:26] Young samples: 5 (Toddler: 2yr × 2, Teenager: 14yr × 3)

[2025-10-13 21:07:26] Old samples: 3 (Elderly: 65yr × 3)

[2025-10-13 21:07:26] Missing abundances: 0 (0.0%)

[2025-10-13 21:07:26] 
Age distribution:

[2025-10-13 21:07:26]   2yr: 524 rows

[2025-10-13 21:07:26]   14yr: 786 rows

[2025-10-13 21:07:26]   65yr: 786 rows

[2025-10-13 21:07:26] 
### Step 1.8: Save long format CSV

[2025-10-13 21:07:26] ✅ Saved: LiDermis_2021_long_format.csv

[2025-10-13 21:07:26] 
✅ PHASE 1 COMPLETE - Long format created

[2025-10-13 21:08:41] 
---


[2025-10-13 21:08:41] ## PHASE 1 (continued): ECM Annotation

[2025-10-13 21:08:41] 
### Step 2.1: Load long format and matrisome reference

[2025-10-13 21:08:41] Loaded long format: 2096 rows

[2025-10-13 21:08:41] Unique proteins: 262

[2025-10-13 21:08:41] Loaded human matrisome: 1027 ECM proteins

[2025-10-13 21:08:41] 
### Step 2.2: 4-level annotation hierarchy

[2025-10-13 21:08:41] 
Level 1: Gene Symbol match

[2025-10-13 21:08:41]   Matrisome columns: ['Matrisome Division', 'Matrisome Category', 'Gene Symbol', 'Gene Name', 'Synonyms']...

[2025-10-13 21:08:41]   Matched: 2072 / 2096 rows (98.9%)

[2025-10-13 21:08:41] 
Level 2: UniProt ID match (for unmatched)

[2025-10-13 21:08:41]   Additional matches: 24 rows

[2025-10-13 21:08:41] 
Level 3: Synonym match (skipped - would need synonym mapping)

[2025-10-13 21:08:41] 
Level 4: Unmatched (non-ECM proteins)

[2025-10-13 21:08:41]   Unmatched: 0 rows (0.0%)

[2025-10-13 21:08:41] 
### Step 2.3: Add annotation columns

[2025-10-13 21:08:41] ✅ Annotation complete

[2025-10-13 21:08:41]    ECM proteins: 2096 rows

[2025-10-13 21:08:41]    Non-ECM: 0 rows

[2025-10-13 21:08:41] 
### Step 2.4: ECM coverage statistics

[2025-10-13 21:08:41] ECM protein coverage:

[2025-10-13 21:08:41]   Unique ECM proteins: 262

[2025-10-13 21:08:41]   Total unique proteins: 262

[2025-10-13 21:08:41]   ECM percentage: 100.0%

[2025-10-13 21:08:41] 
### Step 2.5: Save annotated long format

[2025-10-13 21:08:41] ✅ Saved: LiDermis_2021_long_annotated.csv

[2025-10-13 21:08:41] 
✅ ECM Annotation complete

[2025-10-13 21:10:10] 
---


[2025-10-13 21:10:10] ## PHASE 1 (continued): Convert to LEGACY Format

[2025-10-13 21:10:10] 
### Step 3.1: Load annotated data and filter ECM

[2025-10-13 21:10:10] Loaded annotated data: 2096 rows

[2025-10-13 21:10:10] Filtered to ECM proteins: 2096 rows

[2025-10-13 21:10:10] Unique ECM proteins: 262

[2025-10-13 21:10:10] 
### Step 3.2: Aggregate by Age group WITH N_Profiles count

[2025-10-13 21:10:10] Young aggregation: 262 proteins

[2025-10-13 21:10:10]   N_Profiles_Young range: 5-5

[2025-10-13 21:10:10] Old aggregation: 262 proteins

[2025-10-13 21:10:10]   N_Profiles_Old range: 3-3

[2025-10-13 21:10:10] 
### Step 3.3: Merge Young and Old

[2025-10-13 21:10:10] Merged wide format: 262 proteins

[2025-10-13 21:10:10] Missing Abundance_Young: 0 (0.0%)

[2025-10-13 21:10:10] Missing Abundance_Old: 0 (0.0%)

[2025-10-13 21:10:10] 
### Step 3.4: Add LEGACY format columns

[2025-10-13 21:10:10] Added Dataset_Name, Organ, Compartment, Tissue_Compartment

[2025-10-13 21:10:10] 
### Step 3.5: Keep abundances as-is (ALREADY log2-transformed)

[2025-10-13 21:10:10] Added log2-transformed abundances

[2025-10-13 21:10:10] 
Sample log2 transform:

[2025-10-13 21:10:10]   4.65 → 4.65

[2025-10-13 21:10:10]   11.95 → 11.95

[2025-10-13 21:10:10]   7.74 → 7.74

[2025-10-13 21:10:10] 
### Step 3.6: Rename columns to LEGACY format (underscore)

[2025-10-13 21:10:10] Renamed Matrisome columns (space → underscore)

[2025-10-13 21:10:10] 
### Step 3.7: Add placeholder z-score columns

[2025-10-13 21:10:10] Added placeholder z-score columns (will be filled by universal_zscore_function.py)

[2025-10-13 21:10:10] 
### Step 3.8: Reorder columns to match LEGACY format

[2025-10-13 21:10:10] Reordered columns to match legacy format

[2025-10-13 21:10:10] 
### Step 3.9: Validation

[2025-10-13 21:10:10] Final schema check:

[2025-10-13 21:10:10]   Columns: 25 (expected: 25)

[2025-10-13 21:10:10]   Rows: 262

[2025-10-13 21:10:10]   Unique proteins: 262

[2025-10-13 21:10:10] 
Sample rows (first 3):

[2025-10-13 21:10:10]   A4D0S4: LAMB4 - Young=4.65, Old=0.00, N_Profiles=5/3

[2025-10-13 21:10:10]   A6NMZ7: COL6A6 - Young=11.95, Old=10.81, N_Profiles=5/3

[2025-10-13 21:10:10]   A8K2U0: A2ML1 - Young=7.74, Old=0.00, N_Profiles=5/3

[2025-10-13 21:10:10] 
### Step 3.10: Save LEGACY format CSV

[2025-10-13 21:10:10] ✅ Saved: LiDermis_2021_LEGACY_format.csv

[2025-10-13 21:10:10] 
✅ PHASE 1 COMPLETE - LEGACY format ready

[2025-10-13 21:10:10] 
Next steps:

[2025-10-13 21:10:10]   1. Review output file

[2025-10-13 21:10:10]   2. Append to merged_ecm_aging_zscore.csv manually OR use merge script

[2025-10-13 21:10:10]   3. Calculate z-scores using universal_zscore_function.py
