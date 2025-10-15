# Caldeira et al. 2017 - iTRAQ Processing

## Study Information
- **Title:** Matrisome Profiling During Intervertebral Disc Development And Ageing
- **PMID:** 28900233
- **DOI:** 10.1038/s41598-017-11960-0
- **Tissue:** Bovine caudal intervertebral disc (nucleus pulposus)
- **Species:** Bos taurus (bovine)
- **Method:** iTRAQ 8-plex LC-MS/MS (2 technical batches)

## Age Groups
- **Young:** Young adult bovine (n=6 replicates from 2 batches)
- **Old:** Aged bovine (n=3 replicates)
- **Foetus:** Developmental (EXCLUDED - not relevant for aging comparison)
- **Age gap:** Young adult to aged transition

## Processing Summary
This dataset was processed using the **iTRAQ adapter pipeline** (lightweight transformation):
- iTRAQ data is PRE-NORMALIZED (ratios already calculated by authors)
- Processing is LIGHTWEIGHT (mainly column mapping, not full LFQ parsing)
- Empty rows filtered (none found)
- ECM proteins identified using keyword-based classification (bovine matrisome not available)
- Processing time: ~10 minutes

## Files

### Input
- `../../data_raw/Caldeira et al. - 2017/41598_2017_11960_MOESM3_ESM.xls`
  - Sheet: `1. Proteins` (104 rows × 25 columns)
  - Contains iTRAQ 8-plex ratios (pre-normalized by authors)
  - Technical duplicates for Young group (batch 1 and batch 2)
  - Note: MOESM2 has only 81 proteins (subset), MOESM3 is complete

### Output
- `Caldeira_2017_wide_format.csv` - **43 ECM proteins** in unified schema
  - 18 columns: Protein_ID, Gene_Symbol, Tissue, Abundance_Young, Abundance_Old, etc.
  - Ready for Phase 2 merge and Phase 3 z-score normalization

### Scripts
- `itraq_adapter_caldeira2017.py` - iTRAQ adapter script
  - Loads iTRAQ data from MOESM3
  - Identifies age groups (Young/Old, excludes Foetus)
  - Annotates ECM proteins using keyword matching
  - Calculates mean abundances per age group
  - Maps to unified schema
  - Exports wide-format CSV

### Documentation
- `00_TASK_CALDEIRA_2017_iTRAQ_PROCESSING.md` - Complete task specification
- `README.md` - This file

### Metadata
- `zscore_metadata_Caldeira_2017.json` - Z-score normalization metadata
  - Normalization parameters (mean=1.47, std=0.71 for Young; mean=1.76, std=1.05 for Old)
  - Log2 transformation applied (skewness > 1)
  - Validation results (mean=0, std=1)
  - Outlier statistics (0 outliers detected)
  - Generated: 2025-10-14

## Data Quality
- **ECM proteins:** 43/104 (41.3% of total proteome)
- **Missing values:** Young=4 (9.3%), Old=4 (9.3%) - typical for iTRAQ
- **Matrisome categories:**
  - Collagens: 14
  - ECM Glycoproteins: 14
  - Proteoglycans: 12
  - ECM-affiliated Proteins: 3
- **Match confidence:**
  - 100% (direct matrisome match): 4 proteins
  - 70% (keyword-based classification): 39 proteins

## Processing Pipeline (Completed)

### Phase 1: iTRAQ Adapter
```bash
cd 05_papers_to_csv/03_Caldeira_2017_paper_to_csv
python itraq_adapter_caldeira2017.py
```
- Processed 43 ECM proteins from 104 total proteins
- Created `Caldeira_2017_wide_format.csv`

### Phase 2: Merge to Unified Database
```bash
cd ../../
python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \
    05_papers_to_csv/03_Caldeira_2017_paper_to_csv/Caldeira_2017_wide_format.csv
```
- Merged to `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Added Dataset_Name, Organ, Compartment metadata columns
- Study now represents 9th dataset in unified database

### Phase 3: Calculate Z-scores
```bash
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Caldeira_2017' 'Tissue'
```
- Grouped by: Tissue (Intervertebral_disc_Nucleus_pulposus)
- Log2 transformation applied (skewness: Young=2.43, Old=1.38)
- Validation passed: mean=0, std=1
- Outliers: Young=0 (0.0%), Old=0 (0.0%)
- Results saved to `zscore_metadata_Caldeira_2017.json`

## Validation Checklist
- [x] File loaded successfully (104 proteins)
- [x] ECM proteins identified (43 proteins, 41.3%)
- [x] Age groups mapped correctly (Young vs Old, Foetus excluded)
- [x] Missing values acceptable (9.3% for both groups)
- [x] Schema compliance (18 columns)
- [x] No NaN in critical fields (Dataset_Name, Organ, Compartment)
- [x] Method field correct ("iTRAQ 8-plex LC-MS/MS", not "TMT")
- [x] Species consistency (Bos taurus)
- [x] Abundance validation passed (mean differences observed)
- [x] Wide-format CSV created
- [x] Merged to unified database (43 rows added)
- [x] Z-scores calculated (log2 transform applied, validation passed)
- [x] Metadata files created

## Technical Notes

### iTRAQ vs TMT
- **iTRAQ** is NOT TMT, but both are isobaric labeling methods
- This study used iTRAQ 8-plex (8 samples per experiment)
- Data appears to be PRE-NORMALIZED ratios (not raw intensities)
- Processing follows TMT-like adapter approach (lightweight transformation)
- See `11_subagent_for_LFQ_ingestion/03_TMT_vs_LFQ_PROCESSING_GUIDE.md` for methodology

### Cross-Species Annotation
- Primary species: Bos taurus (bovine)
- Bovine-specific matrisome reference not available
- Used keyword-based ECM classification instead of direct UniProt matching
- 4 proteins matched human matrisome reference (100% confidence)
- 39 proteins classified by ECM keywords (70% confidence)

### Intervertebral Disc Biology
- Nucleus pulposus (NP) is the gelatinous core of intervertebral discs
- NP ECM is rich in proteoglycans (especially aggrecan) and type II collagen
- Age-related changes include loss of proteoglycans and increased fibrous tissue
- This dataset represents one of the few ECM-focused proteomics studies of disc aging

### Sample Structure
- Young group: 6 replicates (3 from batch 1 + 3 from batch 2)
- Old group: 3 replicates (single batch)
- Foetus group: 3 replicates (EXCLUDED - developmental, not aging)
- Pool samples (Pool Foetus, Pool Old) EXCLUDED - not independent replicates

---

**Created:** 2025-10-14
**Updated:** 2025-10-14
**Status:** Complete - All Phases Done (Phase 1-3)
