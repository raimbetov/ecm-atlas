# Ouni et al. 2022 - TMT Processing

## Study Information
- **Title:** Proteome-wide and matrisome-specific atlas of the human ovary
- **PMID:** 35341935
- **Tissue:** Human ovarian cortex
- **Species:** Homo sapiens
- **Method:** DC-MaP + TMTpro 16-plex LC-MS/MS

## Age Groups
- **Young:** Reproductive age (26±5 years, n=5)
- **Old:** Menopausal (59±8 years, n=5)
- **Age gap:** 33 years

## Processing Summary
This dataset was processed using the **TMT adapter pipeline** (lightweight transformation):
- ✅ Pre-normalized data (quantile normalization by authors)
- ✅ ECM classification pre-annotated (Category/Division columns)
- ✅ Simple column mapping (no complex parsing required)
- ✅ Empty rows filtered (4 blank rows at end of Excel table)
- ✅ Processing time: ~15 minutes

## Files

### Input
- `../../data_raw/Ouni et al. - 2022/Supp Table 3.xlsx`
  - Sheet: `Matrisome Proteins` (102 rows × 33 columns)
  - Contains quantile-normalized TMT reporter intensities
  - Note: 4 empty rows at end filtered out during processing

### Output
- `Ouni_2022_wide_format.csv` - **98 ECM proteins** in unified schema
  - 15 columns: Protein_ID, Gene_Symbol, Tissue, Abundance_Young, Abundance_Old, etc.
  - Ready for Phase 2 merge and Phase 3 z-score normalization

### Scripts
- `tmt_adapter_ouni2022.py` - TMT adapter script
  - Loads TMT data
  - Calculates mean abundances per age group
  - Maps to unified schema
  - Exports wide-format CSV

### Documentation
- `00_TASK_OUNI_2022_TMT_PROCESSING.md` - Complete task specification
- `README.md` - This file

### Metadata
- `zscore_metadata_Ouni_2022.json` - Z-score normalization metadata
  - Normalization parameters (mean, std)
  - Validation results
  - Outlier statistics
  - Generated: 2025-10-14

## Data Quality
- **ECM proteins:** 98 (all pre-annotated by authors)
- **Missing values:** Young=0, Old=0 (100% data completeness)
- **Matrisome categories:**
  - ECM Glycoproteins: 27
  - ECM Regulators: 22
  - ECM-affiliated Proteins: 16
  - Collagens: 16
  - Secreted Factors: 9
  - Proteoglycans: 8
- **Match confidence:** 100% (ECM classification by authors)

## Processing Pipeline (Completed)

### Phase 1: TMT Adapter ✅
```bash
python tmt_adapter_ouni2022.py
```
- Processed 98 ECM proteins
- Created `Ouni_2022_wide_format.csv`

### Phase 2: Merge to Unified Database ✅
```bash
cd ../../
python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \
    05_papers_to_csv/08_Ouni_2022_paper_to_csv/Ouni_2022_wide_format.csv
```
- Merged to `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Added Organ, Compartment, Dataset_Name metadata columns

### Phase 3: Calculate Z-scores ✅
```bash
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Ouni_2022' 'Tissue'
```
- Grouped by: Tissue (Ovary_Cortex)
- No log-transformation needed (skewness < 1)
- Validation passed: μ=0, σ=1
- Outliers: Young=2 (2.0%), Old=1 (1.0%)
- Results saved to `zscore_metadata_Ouni_2022.json`

## Validation
All 13 success criteria met:
- ✅ File loaded successfully
- ✅ Row count correct (102 proteins)
- ✅ No null critical fields
- ✅ Age mapping correct
- ✅ Schema compliance (15 columns)
- ✅ ECM classification preserved
- ✅ Match confidence 100%
- ✅ Species consistency
- ✅ Abundance validation passed
- ✅ Missing values acceptable
- ✅ Wide-format CSV created
- ✅ Adapter script documented
- ✅ Task documentation complete

## Technical Notes
- TMT data requires different processing than LFQ datasets
- See `11_subagent_for_LFQ_ingestion/03_TMT_vs_LFQ_PROCESSING_GUIDE.md` for methodology differences
- This dataset represents human reproductive aging in ECM-rich ovarian tissue

---

**Created:** 2025-10-14
**Updated:** 2025-10-14
**Status:** ✅ Complete - All Phases Done (Phase 1-3)
**Dashboard:** Available at http://localhost:8083/dashboard.html
