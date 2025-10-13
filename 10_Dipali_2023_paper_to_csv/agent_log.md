
[2025-10-13 16:12:49] # Dipali 2023 Processing Log (LEGACY Format)

[2025-10-13 16:12:49] 

[2025-10-13 16:12:49] **Study ID:** Dipali_2023

[2025-10-13 16:12:49] **Start Time:** 2025-10-13 16:12:49

[2025-10-13 16:12:49] **Input:** data_raw/Dipali et al. - 2023/

[2025-10-13 16:12:49] **Output Format:** LEGACY (merged_ecm_aging_zscore.csv compatible)

[2025-10-13 16:12:49] 

[2025-10-13 16:12:49] ---

[2025-10-13 16:12:49] 

[2025-10-13 16:12:49] ## PHASE 1: Data Normalization (LEGACY Format)

[2025-10-13 16:12:49] 

[2025-10-13 16:12:49] ### Step 1.1: Load TSV file (despite .xls extension)

[2025-10-13 16:12:49] Project root: /Users/Kravtsovd/projects/ecm-atlas

[2025-10-13 16:12:49] Loading: /Users/Kravtsovd/projects/ecm-atlas/data_raw/Dipali et al. - 2023/Report_Birgit_Protein+Quant_Pivot+(Pivot).xls

[2025-10-13 16:12:49] ✅ Loaded: 3903 rows, 38 columns

[2025-10-13 16:12:49] 
Columns found:

[2025-10-13 16:12:49]   0: PG.Qvalue

[2025-10-13 16:12:49]   1: PG.Genes

[2025-10-13 16:12:49]   2: PG.ProteinDescriptions

[2025-10-13 16:12:49]   3: PG.UniProtIds

[2025-10-13 16:12:49]   4: PG.ProteinNames

[2025-10-13 16:12:49]   5: PG.CellularComponent

[2025-10-13 16:12:49]   6: PG.BiologicalProcess

[2025-10-13 16:12:49]   7: PG.MolecularFunction

[2025-10-13 16:12:49]   8: [1] 210910_0013_SD7_11_Y1L_S6_800ng_180min_DIA.raw.PG.NrOfPrecursorsIdentified

[2025-10-13 16:12:49]   9: [2] 210910_0014_SD7_16_O1L_S16_800ng_180min_DIA.raw.PG.NrOfPrecursorsIdentified

[2025-10-13 16:12:49]   ... and 28 more

[2025-10-13 16:12:49] 
### Step 1.2: Find and verify .PG.Quantity columns

[2025-10-13 16:12:49] Found 10 quantity columns

[2025-10-13 16:12:49]   Y1L: 2.25 months (Young)

[2025-10-13 16:12:49]   O1L: 11 months (Old)

[2025-10-13 16:12:49]   Y2L: 2.25 months (Young)

[2025-10-13 16:12:49]   O2L: 11 months (Old)

[2025-10-13 16:12:49]   Y3L: 2.25 months (Young)

[2025-10-13 16:12:49]   O3L: 11 months (Old)

[2025-10-13 16:12:49]   Y4L: 2.25 months (Young)

[2025-10-13 16:12:49]   O4L: 11 months (Old)

[2025-10-13 16:12:49]   Y5L: 2.25 months (Young)

[2025-10-13 16:12:49]   O5L: 11 months (Old)

[2025-10-13 16:12:49] 
✅ Sample mapping complete:

[2025-10-13 16:12:49]   Young: 5 samples

[2025-10-13 16:12:49]   Old: 5 samples

[2025-10-13 16:12:49] 
### Step 1.3: Extract and process protein IDs

[2025-10-13 16:12:49] Sample Protein IDs (first 5):

[2025-10-13 16:12:49]   A0A023T672

[2025-10-13 16:12:49]   A0A067XG53

[2025-10-13 16:12:49]   A0A075B5K8

[2025-10-13 16:12:49]   A0A075B5P2

[2025-10-13 16:12:49]   A0A075B5P3

[2025-10-13 16:12:49] 
### Step 1.4: Transform to long format

[2025-10-13 16:12:49] ✅ Long format created: 39030 rows

[2025-10-13 16:12:49]    Expected: 3903 proteins × 10 samples = 39030 rows

[2025-10-13 16:12:49]    Match: ✅

[2025-10-13 16:12:49] 
### Step 1.5: Validate schema

[2025-10-13 16:12:49] Schema columns: ['Protein_ID', 'Protein_Name', 'Gene_Symbol', 'Tissue', 'Species', 'Age', 'Age_Unit', 'Abundance', 'Abundance_Unit', 'Method', 'Study_ID', 'Sample_ID', 'Parsing_Notes']

[2025-10-13 16:12:49] All expected columns present: ✅

[2025-10-13 16:12:49] 
### Step 1.6: Basic statistics

[2025-10-13 16:12:49] Total rows: 39030

[2025-10-13 16:12:49] Unique proteins: 3903

[2025-10-13 16:12:49] Young samples: 5

[2025-10-13 16:12:49] Old samples: 5

[2025-10-13 16:12:49] Missing abundances: 0 (0.0%)

[2025-10-13 16:12:49] Abundance range: 1.00e+00 to 8.35e+08

[2025-10-13 16:12:49] 
### Step 1.7: Save long format CSV

[2025-10-13 16:12:49] ✅ Saved: Dipali_2023_long_format.csv

[2025-10-13 16:12:49] 
✅ PHASE 1 COMPLETE - Long format created

[2025-10-13 16:13:25] 
---


[2025-10-13 16:13:25] ## PHASE 1 (continued): ECM Annotation

[2025-10-13 16:13:25] 
### Step 2.1: Load long format and matrisome reference

[2025-10-13 16:13:25] Loaded long format: 39030 rows

[2025-10-13 16:13:26] Unique proteins: 3903

[2025-10-13 16:13:26] Loaded mouse matrisome: 1110 ECM proteins

[2025-10-13 16:13:26] 
### Step 2.2: 4-level annotation hierarchy

[2025-10-13 16:13:26] 
Level 1: Gene Symbol match

[2025-10-13 16:13:26]   Matrisome columns: ['Matrisome Division', 'Matrisome Category', 'Gene Symbol', 'Gene Name', 'Synonyms']...

[2025-10-13 16:13:26]   Matched: 1710 / 39030 rows (4.4%)

[2025-10-13 16:13:26] 
Level 2: UniProt ID match (for unmatched)

[2025-10-13 16:13:26]   Additional matches: 20 rows

[2025-10-13 16:13:26] 
Level 3: Synonym match (skipped - would need synonym mapping)

[2025-10-13 16:13:26] 
Level 4: Unmatched (non-ECM proteins)

[2025-10-13 16:13:26]   Unmatched: 37300 rows (95.6%)

[2025-10-13 16:13:26] 
### Step 2.3: Add annotation columns

[2025-10-13 16:13:26] ✅ Annotation complete

[2025-10-13 16:13:26]    ECM proteins: 1730 rows

[2025-10-13 16:13:26]    Non-ECM: 37300 rows

[2025-10-13 16:13:26] 
### Step 2.4: ECM coverage statistics

[2025-10-13 16:13:26] ECM protein coverage:

[2025-10-13 16:13:26]   Unique ECM proteins: 173

[2025-10-13 16:13:26]   Total unique proteins: 3903

[2025-10-13 16:13:26]   ECM percentage: 4.4%

[2025-10-13 16:13:26] 
### Step 2.5: Save annotated long format

[2025-10-13 16:13:26] ✅ Saved: Dipali_2023_long_annotated.csv

[2025-10-13 16:13:26] 
✅ ECM Annotation complete

[2025-10-13 16:14:26] 
---


[2025-10-13 16:14:26] ## PHASE 1 (continued): Convert to LEGACY Format

[2025-10-13 16:14:26] 
### Step 3.1: Load annotated data and filter ECM

[2025-10-13 16:14:26] Loaded annotated data: 39030 rows

[2025-10-13 16:14:26] Filtered to ECM proteins: 1730 rows

[2025-10-13 16:14:26] Unique ECM proteins: 173

[2025-10-13 16:14:26] 
### Step 3.2: Aggregate by Age group WITH N_Profiles count

[2025-10-13 16:14:26] Young aggregation: 173 proteins

[2025-10-13 16:14:26]   N_Profiles_Young range: 5-5

[2025-10-13 16:14:26] Old aggregation: 173 proteins

[2025-10-13 16:14:26]   N_Profiles_Old range: 5-5

[2025-10-13 16:14:26] 
### Step 3.3: Merge Young and Old

[2025-10-13 16:14:26] Merged wide format: 173 proteins

[2025-10-13 16:14:26] Missing Abundance_Young: 0 (0.0%)

[2025-10-13 16:14:26] Missing Abundance_Old: 0 (0.0%)

[2025-10-13 16:14:26] 
### Step 3.4: Add LEGACY format columns

[2025-10-13 16:14:26] Added Dataset_Name, Organ, Compartment, Tissue_Compartment

[2025-10-13 16:14:26] 
### Step 3.5: Add log2 transformed abundances

[2025-10-13 16:14:26] Added log2-transformed abundances

[2025-10-13 16:14:26] 
Sample log2 transform:

[2025-10-13 16:14:26]   36812.72 → 15.17

[2025-10-13 16:14:26]   4736457.15 → 22.18

[2025-10-13 16:14:26]   547019.88 → 19.06

[2025-10-13 16:14:26] 
### Step 3.6: Rename columns to LEGACY format (underscore)

[2025-10-13 16:14:26] Renamed Matrisome columns (space → underscore)

[2025-10-13 16:14:26] 
### Step 3.7: Add placeholder z-score columns

[2025-10-13 16:14:26] Added placeholder z-score columns (will be filled by universal_zscore_function.py)

[2025-10-13 16:14:26] 
### Step 3.8: Reorder columns to match LEGACY format

[2025-10-13 16:14:26] Reordered columns to match legacy format

[2025-10-13 16:14:26] 
### Step 3.9: Validation

[2025-10-13 16:14:26] Final schema check:

[2025-10-13 16:14:26]   Columns: 25 (expected: 25)

[2025-10-13 16:14:26]   Rows: 173

[2025-10-13 16:14:26]   Unique proteins: 173

[2025-10-13 16:14:26] 
Sample rows (first 3):

[2025-10-13 16:14:26]   A0A087WQ70: Serpine2 - Young=36812.72, Old=14351.44, N_Profiles=5/5

[2025-10-13 16:14:26]   A0A087WR50: Fn1 - Young=4736457.15, Old=4680555.65, N_Profiles=5/5

[2025-10-13 16:14:26]   A0A0A0MQ90: S100a13 - Young=547019.88, Old=879133.35, N_Profiles=5/5

[2025-10-13 16:14:26] 
### Step 3.10: Save LEGACY format CSV

[2025-10-13 16:14:26] ✅ Saved: Dipali_2023_LEGACY_format.csv

[2025-10-13 16:14:26] 
✅ PHASE 1 COMPLETE - LEGACY format ready

[2025-10-13 16:14:26] 
Next steps:

[2025-10-13 16:14:26]   1. Review output file

[2025-10-13 16:14:26]   2. Append to merged_ecm_aging_zscore.csv manually OR use merge script

[2025-10-13 16:14:26]   3. Calculate z-scores using universal_zscore_function.py
