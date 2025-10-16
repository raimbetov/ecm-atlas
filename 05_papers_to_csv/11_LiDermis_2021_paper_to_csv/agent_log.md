
[2025-10-15 22:14:34] # LiDermis 2021 Processing Log (LEGACY Format)

[2025-10-15 22:14:34] 

[2025-10-15 22:14:34] **Study ID:** LiDermis_2021

[2025-10-15 22:14:34] **Start Time:** 2025-10-15 22:14:34

[2025-10-15 22:14:34] **Input:** data_raw/Li et al. - 2021 | dermis/

[2025-10-15 22:14:34] **Output Format:** LEGACY (merged_ecm_aging_zscore.csv compatible)

[2025-10-15 22:14:34] 

[2025-10-15 22:14:34] ---

[2025-10-15 22:14:34] 

[2025-10-15 22:14:34] ## PHASE 1: Data Normalization (LEGACY Format)

[2025-10-15 22:14:34] 

[2025-10-15 22:14:34] ### Step 1.1: Load Excel file with complex headers

[2025-10-15 22:14:34] Project root: /Users/Kravtsovd/projects/ecm-atlas

[2025-10-15 22:14:34] Loading: /Users/Kravtsovd/projects/ecm-atlas/data_raw/Li et al. - 2021 | dermis/Table 2.xlsx

[2025-10-15 22:14:34] Sheet: Table S2

[2025-10-15 22:14:35] 
=== Sample names from row 3 ===

[2025-10-15 22:14:35]   Column 4: Toddler-Sample1 

[2025-10-15 22:14:35]   Column 5: Toddler-Sample2

[2025-10-15 22:14:35]   Column 6: Teenager-Sample1

[2025-10-15 22:14:35]   Column 7: Teenager-Sample2

[2025-10-15 22:14:35]   Column 8: Teenager-Sample3

[2025-10-15 22:14:35]   Column 9: Adult-Sample1

[2025-10-15 22:14:35]   Column 10: Adult-Sample2

[2025-10-15 22:14:35]   Column 11: Elderly-Sample1 

[2025-10-15 22:14:35]   Column 12: Elderly-Sample2

[2025-10-15 22:14:35]   Column 13: Elderly-Sample3

[2025-10-15 22:14:35]   Column 14: Ave_Toddler

[2025-10-15 22:14:35]   Column 15: Ave_Teenager

[2025-10-15 22:14:35]   Column 16: Ave_Adult

[2025-10-15 22:14:35]   Column 17: Ave_Elderly

[2025-10-15 22:14:35] 
✅ Found 14 sample columns

[2025-10-15 22:14:35] ✅ Loaded: 262 rows, 22 columns

[2025-10-15 22:14:35] 
### Step 1.1b: Convert zeros to NaN (proteomics not-detected convention)

[2025-10-15 22:14:35] Abundance columns identified: 10

[2025-10-15 22:14:35] Zeros before conversion: 785/2620 (30.0%)

[2025-10-15 22:14:35] Zeros after conversion: 0/2620 (0.0%)

[2025-10-15 22:14:35] NaN values now: 785/2620 (30.0%)

[2025-10-15 22:14:35] ✅ Converted 785 zeros to NaN

[2025-10-15 22:14:35] 
Columns (first 10):

[2025-10-15 22:14:35]   0: Q5KU26

[2025-10-15 22:14:35]   1: COLEC12

[2025-10-15 22:14:35]   2: Matrisome-associated

[2025-10-15 22:14:35]   3: ECM-affiliated Proteins

[2025-10-15 22:14:35]   4: Toddler-Sample1

[2025-10-15 22:14:35]   5: Toddler-Sample2

[2025-10-15 22:14:35]   6: Teenager-Sample1

[2025-10-15 22:14:35]   7: Teenager-Sample2

[2025-10-15 22:14:35]   8: Teenager-Sample3

[2025-10-15 22:14:35]   9: Adult-Sample1

[2025-10-15 22:14:35] 
### Step 1.2: Filter samples (exclude Adult 40yr)

[2025-10-15 22:14:35] Young columns (Toddler + Teenager): 5

[2025-10-15 22:14:35]   - Toddler-Sample1

[2025-10-15 22:14:35]   - Toddler-Sample2

[2025-10-15 22:14:35]   - Teenager-Sample1

[2025-10-15 22:14:35]   - Teenager-Sample2

[2025-10-15 22:14:35]   - Teenager-Sample3

[2025-10-15 22:14:35] 
Old columns (Elderly): 3

[2025-10-15 22:14:35]   - Elderly-Sample1

[2025-10-15 22:14:35]   - Elderly-Sample2

[2025-10-15 22:14:35]   - Elderly-Sample3

[2025-10-15 22:14:35] 
❌ Excluded Adult columns (40yr, middle-aged): 2

[2025-10-15 22:14:35]   - Adult-Sample1

[2025-10-15 22:14:35]   - Adult-Sample2

[2025-10-15 22:14:35] 
✅ Retention: 8/14 = 57.1%

[2025-10-15 22:14:35] 
### Step 1.3: Extract protein identifiers

[2025-10-15 22:14:35] Protein ID column: 'Q5KU26'

[2025-10-15 22:14:35] Gene Symbol column: 'COLEC12'

[2025-10-15 22:14:35] 
Sample Protein IDs (first 5):

[2025-10-15 22:14:35]   P16112

[2025-10-15 22:14:35]   O75487

[2025-10-15 22:14:35]   P13928

[2025-10-15 22:14:35]   P29508

[2025-10-15 22:14:35]   Q8N6G6

[2025-10-15 22:14:35] 
### Step 1.4: Protein_Name lookup via UniProt API

[2025-10-15 22:14:35] ⚠️  Warning: This may take 5-10 minutes for ~260 proteins

[2025-10-15 22:14:35] Fetching names for 262 unique proteins...

[2025-10-15 22:15:07]   Progress: 50/262 (19.1%)

[2025-10-15 22:15:39]   Progress: 100/262 (38.2%)

[2025-10-15 22:16:12]   Progress: 150/262 (57.3%)

[2025-10-15 22:16:45]   Progress: 200/262 (76.3%)

[2025-10-15 22:17:17]   Progress: 250/262 (95.4%)

[2025-10-15 22:17:25] ✅ Fetched 262 protein names

[2025-10-15 22:17:25] ⚠️  0 proteins without names

[2025-10-15 22:17:25] 
### Step 1.5: Transform to long format

[2025-10-15 22:17:25] ✅ Long format created: 2096 rows

[2025-10-15 22:17:25]    Expected: 262 proteins × 8 samples = 2096 rows

[2025-10-15 22:17:25]    Match: ✅

[2025-10-15 22:17:25] 
### Step 1.6: Validate schema

[2025-10-15 22:17:25] Schema columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Tissue', 'Species', 'Age', 'Age_Unit', 'Abundance', 'Abundance_Unit', 'Method', 'Study_ID', 'Sample_ID', 'Parsing_Notes']

[2025-10-15 22:17:25] All expected columns present: ✅

[2025-10-15 22:17:25] 
### Step 1.7: Basic statistics

[2025-10-15 22:17:25] Total rows: 2096

[2025-10-15 22:17:25] Unique proteins: 262

[2025-10-15 22:17:25] Young samples: 5 (Toddler: 2yr × 2, Teenager: 14yr × 3)

[2025-10-15 22:17:25] Old samples: 3 (Elderly: 65yr × 3)

[2025-10-15 22:17:25] Missing abundances: 694 (33.1%)

[2025-10-15 22:17:25] 
Age distribution:

[2025-10-15 22:17:25]   2yr: 524 rows

[2025-10-15 22:17:25]   14yr: 786 rows

[2025-10-15 22:17:25]   65yr: 786 rows

[2025-10-15 22:17:25] 
### Step 1.8: Save long format CSV

[2025-10-15 22:17:25] ✅ Saved: LiDermis_2021_long_format.csv

[2025-10-15 22:17:25] 
✅ PHASE 1 COMPLETE - Long format created

[2025-10-15 22:17:41] 
---


[2025-10-15 22:17:41] ## PHASE 1 (continued): ECM Annotation

[2025-10-15 22:17:41] 
### Step 2.1: Load long format and matrisome reference

[2025-10-15 22:17:41] Loaded long format: 2096 rows

[2025-10-15 22:17:41] Unique proteins: 262

[2025-10-15 22:17:41] Loaded human matrisome: 1027 ECM proteins

[2025-10-15 22:17:41] 
### Step 2.2: 4-level annotation hierarchy

[2025-10-15 22:17:41] 
Level 1: Gene Symbol match

[2025-10-15 22:17:41]   Matrisome columns: ['Matrisome Division', 'Matrisome Category', 'Gene Symbol', 'Gene Name', 'Synonyms']...

[2025-10-15 22:17:41]   Matched: 2072 / 2096 rows (98.9%)

[2025-10-15 22:17:41] 
Level 2: UniProt ID match (for unmatched)

[2025-10-15 22:17:41]   Additional matches: 24 rows

[2025-10-15 22:17:41] 
Level 3: Synonym match (skipped - would need synonym mapping)

[2025-10-15 22:17:41] 
Level 4: Unmatched (non-ECM proteins)

[2025-10-15 22:17:41]   Unmatched: 0 rows (0.0%)

[2025-10-15 22:17:41] 
### Step 2.3: Add annotation columns

[2025-10-15 22:17:41] ✅ Annotation complete

[2025-10-15 22:17:41]    ECM proteins: 2096 rows

[2025-10-15 22:17:41]    Non-ECM: 0 rows

[2025-10-15 22:17:41] 
### Step 2.4: ECM coverage statistics

[2025-10-15 22:17:41] ECM protein coverage:

[2025-10-15 22:17:41]   Unique ECM proteins: 262

[2025-10-15 22:17:41]   Total unique proteins: 262

[2025-10-15 22:17:41]   ECM percentage: 100.0%

[2025-10-15 22:17:41] 
### Step 2.5: Save annotated long format

[2025-10-15 22:17:41] ✅ Saved: LiDermis_2021_long_annotated.csv

[2025-10-15 22:17:41] 
✅ ECM Annotation complete

[2025-10-15 22:17:50] 
---


[2025-10-15 22:17:50] ## PHASE 1 (continued): Convert to LEGACY Format

[2025-10-15 22:17:50] 
### Step 3.1: Load annotated data and filter ECM

[2025-10-15 22:17:50] Loaded annotated data: 2096 rows

[2025-10-15 22:17:50] Filtered to ECM proteins: 2096 rows

[2025-10-15 22:17:50] Unique ECM proteins: 262

[2025-10-15 22:17:50] 
### Step 3.2: Aggregate by Age group WITH N_Profiles count

[2025-10-15 22:17:50] Young aggregation: 262 proteins

[2025-10-15 22:17:50]   N_Profiles_Young range: 0-5

[2025-10-15 22:17:50] Old aggregation: 262 proteins

[2025-10-15 22:17:50]   N_Profiles_Old range: 0-3

[2025-10-15 22:17:50] 
### Step 3.3: Merge Young and Old

[2025-10-15 22:17:50] Merged wide format: 262 proteins

[2025-10-15 22:17:50] Missing Abundance_Young: 22 (8.4%)

[2025-10-15 22:17:50] Missing Abundance_Old: 82 (31.3%)

[2025-10-15 22:17:50] 
### Step 3.4: Add LEGACY format columns

[2025-10-15 22:17:50] Added Dataset_Name, Organ, Compartment, Tissue_Compartment

[2025-10-15 22:17:50] 
### Step 3.5: Keep abundances as-is (ALREADY log2-transformed)

[2025-10-15 22:17:50] Added log2-transformed abundances

[2025-10-15 22:17:50] 
Sample log2 transform:

[2025-10-15 22:17:50]   7.76 → 7.76

[2025-10-15 22:17:50]   11.95 → 11.95

[2025-10-15 22:17:50]   7.74 → 7.74

[2025-10-15 22:17:50] 
### Step 3.6: Rename columns to LEGACY format (underscore)

[2025-10-15 22:17:50] Renamed Matrisome columns (space → underscore)

[2025-10-15 22:17:50] 
### Step 3.7: Add placeholder z-score columns

[2025-10-15 22:17:50] Added placeholder z-score columns (will be filled by universal_zscore_function.py)

[2025-10-15 22:17:50] 
### Step 3.8: Reorder columns to match LEGACY format

[2025-10-15 22:17:50] Reordered columns to match legacy format

[2025-10-15 22:17:50] 
### Step 3.9: Validation

[2025-10-15 22:17:50] Final schema check:

[2025-10-15 22:17:50]   Columns: 25 (expected: 25)

[2025-10-15 22:17:50]   Rows: 262

[2025-10-15 22:17:50]   Unique proteins: 262

[2025-10-15 22:17:50] 
Sample rows (first 3):

[2025-10-15 22:17:50]   A4D0S4: LAMB4 - Young=7.76, Old=NaN, N_Profiles=3/0

[2025-10-15 22:17:50]   A6NMZ7: COL6A6 - Young=11.95, Old=10.81, N_Profiles=5/3

[2025-10-15 22:17:50]   A8K2U0: A2ML1 - Young=7.74, Old=NaN, N_Profiles=5/0

[2025-10-15 22:17:50] 
### Step 3.10: Save LEGACY format CSV

[2025-10-15 22:17:50] ✅ Saved: LiDermis_2021_LEGACY_format.csv

[2025-10-15 22:17:50] 
✅ PHASE 1 COMPLETE - LEGACY format ready

[2025-10-15 22:17:50] 
Next steps:

[2025-10-15 22:17:50]   1. Review output file

[2025-10-15 22:17:50]   2. Append to merged_ecm_aging_zscore.csv manually OR use merge script

[2025-10-15 22:17:50]   3. Calculate z-scores using universal_zscore_function.py
