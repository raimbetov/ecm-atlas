# Tsumagari et al. 2023 - TMT Processing

## Study Information
- **Title:** Age-related proteomic changes in mouse brain (Cortex and Hippocampus)
- **PMID:** 38086838
- **DOI:** 10.1038/s41598-023-45570-8
- **Tissue:** Mouse brain (Cortex and Hippocampus)
- **Species:** Mus musculus
- **Method:** TMT 6-plex LC-MS/MS

## Age Groups
- **Young:** 3 months (adult, n=6 per region)
- **Old:** 24 months (aged, n=6 per region)
- **Age gap:** 21 months
- **Excluded:** 15 months (middle age) - not included for binary Young/Old comparison

## Processing Summary
This dataset was processed using the **TMT adapter pipeline** (lightweight transformation):
- Pre-normalized data (TMT reporter ion intensities)
- Two brain regions processed separately: Cortex and Hippocampus
- Simple column mapping and averaging across replicates
- ECM annotation using mouse matrisome reference (v2)
- Processing time: ~5 minutes

## Files

### Input
- `../../data_raw/Tsumagari et al. - 2023/41598_2023_45570_MOESM3_ESM.xlsx`
  - Sheet: `expression` (6,821 proteins)
  - Cortex samples: Cx_3mo_1-6, Cx_15mo_1-6, Cx_24mo_1-6
- `../../data_raw/Tsumagari et al. - 2023/41598_2023_45570_MOESM4_ESM.xlsx`
  - Sheet: `expression` (6,910 proteins)
  - Hippocampus samples: Hipp_3mo_1-6, Hipp_15mo_1-6, Hipp_24mo_1-6

### Output
- `Tsumagari_2023_wide_format.csv` - **423 ECM proteins** in unified schema
  - 18 columns: Protein_ID, Gene_Symbol, Tissue, Abundance_Young, Abundance_Old, etc.
  - 209 proteins in Cortex, 214 proteins in Hippocampus
  - Ready for Phase 2 merge and Phase 3 z-score normalization

### Scripts
- `tmt_adapter_tsumagari2023.py` - TMT adapter script
  - Loads TMT data from both brain regions
  - Calculates mean abundances per age group (3mo, 15mo, 24mo)
  - Maps to Young (3mo) and Old (24mo) binary groups
  - Annotates with mouse matrisome reference
  - Filters to ECM proteins only
  - Exports wide-format CSV

### Documentation
- `00_TASK_TSUMAGARI_2023_TMT_PROCESSING.md` - Complete task specification (to be created)
- `README.md` - This file

### Metadata
- `zscore_metadata_Tsumagari_2023.json` - Z-score normalization metadata
  - Normalization parameters (mean, std) per brain region
  - Validation results (mean ≈ 0, std ≈ 1)
  - Outlier statistics (|z| > 3)
  - Generated: 2025-10-14

## Data Quality
- **ECM proteins:** 423 total (209 Cortex + 214 Hippocampus)
  - Unique proteins: 224 (some proteins appear in both regions)
- **Missing values:** Young=0%, Old=0% (100% data completeness - typical for TMT)
- **Annotation coverage:** 3.1% (ECM proteins out of whole proteome)
  - This is CORRECT - brain tissue contains mostly non-ECM proteins
- **Matrisome categories:**
  - ECM Regulators: 105
  - ECM Glycoproteins: 96
  - ECM-affiliated Proteins: 94
  - Secreted Factors: 65
  - Proteoglycans: 40
  - Collagens: 23
- **Match confidence:** 100% (all ECM proteins matched via matrisome reference)

## Processing Pipeline (Completed)

### Phase 1: TMT Adapter
```bash
cd /Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/11_Tsumagari_2023_paper_to_csv
python tmt_adapter_tsumagari2023.py
```

**Results:**
- Processed 6,821 proteins (Cortex) and 6,910 proteins (Hippocampus)
- Annotated 209 ECM proteins (Cortex) and 214 ECM proteins (Hippocampus)
- Combined to 423 total rows in wide format
- Created `Tsumagari_2023_wide_format.csv`
- Validation: ALL 6 checks PASSED

### Phase 2: Merge to Unified Database
```bash
cd ../../
python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \
    05_papers_to_csv/11_Tsumagari_2023_paper_to_csv/Tsumagari_2023_wide_format.csv
```

**Results:**
- Merged to `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Added 423 rows to unified database
- Added metadata columns: Dataset_Name, Organ, Compartment
- Backup created: `backups/merged_ecm_aging_zscore_2025-10-14_23-51-06.csv`
- Total studies in database: 9 (including Tsumagari_2023)

### Phase 3: Calculate Z-scores
```bash
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Tsumagari_2023' 'Tissue'
```

**Results:**
- Grouped by: Tissue (Brain_Cortex and Brain_Hippocampus)
- No log-transformation needed (skewness < 1 for both regions)
- Normalization parameters:
  - **Brain_Cortex:**
    - Young: μ=27.89, σ=2.88
    - Old: μ=28.01, σ=2.90
  - **Brain_Hippocampus:**
    - Young: μ=27.67, σ=2.95
    - Old: μ=27.82, σ=2.98
- Validation: PASSED (μ ≈ 0, σ ≈ 1) for both regions
- Outliers:
  - Cortex: Young=1 (0.5%), Old=1 (0.5%)
  - Hippocampus: Young=1 (0.5%), Old=1 (0.5%)
- Results saved to `zscore_metadata_Tsumagari_2023.json`

## Validation

All quality checks passed:

Phase 1 (TMT Adapter):
- [x] No null Protein_ID
- [x] No null Organ
- [x] No null Compartment
- [x] No null Dataset_Name
- [x] All rows are ECM proteins (Match_Confidence > 0)
- [x] Both brain regions present (Cortex and Hippocampus)

Phase 2 (Merge):
- [x] 423 rows added to unified database
- [x] No duplicate rows
- [x] Schema compliance maintained
- [x] Backup created successfully

Phase 3 (Z-scores):
- [x] Z-score calculation completed for both regions
- [x] Validation passed (mean ≈ 0, std ≈ 1)
- [x] Low outlier rate (<1%)
- [x] Metadata JSON saved

## Technical Notes

### TMT vs LFQ Processing
- TMT data is pre-normalized by authors (reporter ion intensities)
- No complex parsing required - simple column mapping
- 100% data completeness (no missing values)
- Processing is much faster than LFQ datasets (~5 min vs 2-4 hours)
- See `11_subagent_for_LFQ_ingestion/03_TMT_vs_LFQ_PROCESSING_GUIDE.md` for methodology differences

### Age Group Mapping Decision
The study included 3 age groups (3mo, 15mo, 24mo). For binary Young/Old comparison:
- **Young:** 3 months (adult mice, sexually mature)
- **Old:** 24 months (aged mice, ~80% of maximum lifespan)
- **Excluded:** 15 months (middle age)

This mapping ensures:
1. Clear biological separation (21-month gap)
2. Comparability with other aging studies in ECM Atlas
3. Adult vs aged comparison (not developmental changes)

### Brain Region Compartments
Both regions are kept **separate** in the database:
- `Brain_Cortex` - 209 ECM proteins
- `Brain_Hippocampus` - 214 ECM proteins

This preserves region-specific ECM aging signatures and enables:
- Regional comparison of ECM changes
- Independent z-score normalization per region
- No data loss from averaging different compartments

### ECM Annotation
Used **mouse matrisome reference v2** (1,110 proteins):
- Level 1 match (Gene Symbol): Primary method
- Level 2 match (UniProt ID): Fallback
- Level 3 match (Synonyms): Alternative gene names
- 3.1% annotation rate is EXPECTED - brain is not ECM-rich tissue

## Scientific Context

### Why Brain ECM Aging Matters
The brain extracellular matrix (ECM) undergoes significant changes with aging:
- Perineuronal nets (PNNs) degrade, affecting synaptic plasticity
- ECM remodeling enzymes (MMPs, ADAMTs) alter activity
- Proteoglycan composition shifts
- Changes contribute to cognitive decline and neurodegeneration

### Key ECM Components in Brain
- **Collagens:** Structural support, basement membrane
- **Proteoglycans:** Aggrecan, versican, brevican (PNN components)
- **Glycoproteins:** Tenascins, laminins
- **ECM Regulators:** Matrix metalloproteinases (MMPs), TIMPs

### Study Significance
- First whole-proteome TMT analysis of brain ECM aging
- Covers two distinct regions (cortex vs hippocampus)
- Enables regional comparison of ECM aging signatures
- Contributes mouse brain data to ECM Atlas (previously human-dominated)

---

**Created:** 2025-10-14
**Updated:** 2025-10-14
**Status:** Complete - All Phases Done (Phase 1-3)
**Processing Time:** ~10 minutes total
**Next Steps:** Data ready for dashboard visualization and cross-study comparison
