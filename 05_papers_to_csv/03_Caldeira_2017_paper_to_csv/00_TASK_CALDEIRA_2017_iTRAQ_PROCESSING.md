# TASK: Caldeira et al. 2017 iTRAQ Dataset Processing

## Task Summary
Successfully processed Caldeira et al. 2017 iTRAQ proteomics dataset and integrated it into the ECM Atlas unified database.

**Status:** COMPLETE

**Date Completed:** 2025-10-14

---

## Study Overview

### Publication Details
- **Title:** Matrisome Profiling During Intervertebral Disc Development And Ageing
- **Authors:** Caldeira J, Santa C, Osório H, Molinos M, Manadas B, Gonçalves R, Barbosa M
- **Journal:** Scientific Reports
- **Published:** September 14, 2017
- **PMID:** 28900233
- **DOI:** 10.1038/s41598-017-11960-0

### Experimental Design
- **Tissue:** Bovine caudal intervertebral disc (nucleus pulposus)
- **Species:** Bos taurus (cattle)
- **Method:** iTRAQ 8-plex LC-MS/MS
- **Technical replicates:** 2 batches for iTRAQ labeling
- **Age groups:**
  - Foetus (n=3) - EXCLUDED (developmental)
  - Young adult (n=6, 3 per batch) - USED
  - Aged (n=3) - USED

### Data Location
- **Raw data file:** `data_raw/Caldeira et al. - 2017/41598_2017_11960_MOESM3_ESM.xls`
- **Sheet:** `1. Proteins`
- **Total proteins:** 104
- **ECM proteins identified:** 43 (41.3%)

---

## Processing Pipeline Execution

### Phase 0: Reconnaissance
**Duration:** ~15 minutes

**Actions taken:**
1. Examined all XLS files (MOESM2, MOESM3, MOESM6, MOESM7)
2. Identified MOESM3 as the complete dataset (104 proteins vs 81 in MOESM2)
3. Analyzed data structure and identified iTRAQ sample columns
4. Searched for paper online to understand experimental design
5. Confirmed iTRAQ method (NOT TMT, but similar processing approach)

**Key findings:**
- Data contains PRE-NORMALIZED iTRAQ ratios (not raw channel intensities)
- Young group has 6 replicates (2 technical batches: Young 1-3, Young 1-3 (2))
- Old group has 3 replicates
- Foetus group excluded (not relevant for aging comparison)
- Pool samples excluded (not independent biological replicates)
- Values are ratios centered around 1.0, typical for iTRAQ normalized data

### Phase 1: iTRAQ Adapter Script
**Duration:** ~20 minutes

**Script created:** `itraq_adapter_caldeira2017.py`

**Processing steps:**
1. Load iTRAQ data from MOESM3 (104 proteins)
2. Identify sample columns:
   - Young: Young 1, Young 2, Young 3, Young 1 (2), Young 2 (2), Young 3 (2)
   - Old: Old 1, Old 2, Old 3
   - Excluded: Foetus 1-3, Pool Foetus, Pool Old
3. Annotate ECM proteins:
   - Attempted to load bovine matrisome reference (not available)
   - Used multi-species strategy with human/mouse references
   - Applied keyword-based classification for cross-species matches
   - Categories: Collagens, ECM Glycoproteins, Proteoglycans, ECM-affiliated Proteins
4. Calculate mean abundances per age group (skipna=True)
5. Map to unified 18-column schema
6. Export `Caldeira_2017_wide_format.csv`

**Results:**
- Input: 104 proteins
- Output: 43 ECM proteins (41.3% coverage)
- Missing values: Young=4 (9.3%), Old=4 (9.3%)
- Match confidence: 4 proteins at 100%, 39 proteins at 70%

### Phase 2: Merge to Unified Database
**Duration:** <1 minute

**Command:**
```bash
python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \
    05_papers_to_csv/03_Caldeira_2017_paper_to_csv/Caldeira_2017_wide_format.csv
```

**Results:**
- Backup created: `merged_ecm_aging_zscore_2025-10-14_23-51-34.csv`
- Added 43 rows to unified CSV
- Total studies: 9 (Caldeira_2017 is the newest)
- Total rows: 4031
- No duplicate rows detected
- Metadata updated: `unified_metadata.json`

### Phase 3: Z-Score Calculation
**Duration:** <1 minute

**Command:**
```bash
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Caldeira_2017' 'Tissue'
```

**Normalization details:**
- Grouping: By Tissue (1 group: Intervertebral_disc_Nucleus_pulposus)
- Skewness: Young=2.426, Old=1.384 (both > 1)
- **Log2 transformation applied:** Yes
- Transformed mean: Young=1.47, Old=1.76
- Transformed std: Young=0.71, Old=1.05
- Z-score validation: PASSED (mean=0, std=1)
- Outliers detected: 0 (0.0% for both groups)

**Results:**
- Backup created: `merged_ecm_aging_zscore_2025-10-14_23-52-40.csv`
- Updated 43 rows with z-scores
- Metadata saved: `zscore_metadata_Caldeira_2017.json`

---

## Data Quality Assessment

### ECM Coverage
- **Total proteins:** 104
- **ECM proteins:** 43 (41.3%)
- **Expected coverage:** 30-50% for whole-proteome studies
- **Assessment:** GOOD - Within expected range

### Matrisome Distribution
| Category | Count | % of ECM |
|----------|-------|----------|
| Collagens | 14 | 32.6% |
| ECM Glycoproteins | 14 | 32.6% |
| Proteoglycans | 12 | 27.9% |
| ECM-affiliated Proteins | 3 | 7.0% |

### Missing Values
- **Young:** 4/43 (9.3%)
- **Old:** 4/43 (9.3%)
- **Assessment:** EXCELLENT - iTRAQ typically has <10% missing values

### Z-Score Quality
- **Log2 transformation:** Applied (skewness > 1)
- **Validation:** PASSED (mean=0, std=1)
- **Outliers:** 0 (0.0%)
- **Assessment:** EXCELLENT - Clean normalization, no outliers

---

## Key Biological Findings (from dataset)

### Notable ECM Proteins Identified
1. **Aggrecan (P13608)** - Major proteoglycan in nucleus pulposus
2. **Collagen II (P02459)** - Primary structural collagen
3. **Biglycan (P21809)** - Small leucine-rich proteoglycan
4. **Versican (P81282)** - Large aggregating proteoglycan
5. **Decorin (P21793)** - Collagen fibril assembly regulator

### Age-Related Changes (Raw Data)
- Biglycan: Young=10.76, Old=13.18 (increased with age)
- Decorin: Young=1.22, Old=11.42 (dramatically increased)
- Collagen II: Young=5.14, Old=0.43 (decreased with age)
- Aggrecan: Young=1.13, Old=1.34 (slightly increased)

These patterns are consistent with known disc aging biology:
- Loss of type II collagen (cartilage marker)
- Increase in fibrous ECM components
- Altered proteoglycan composition

---

## Technical Notes

### iTRAQ vs TMT Processing
- **iTRAQ** (Isobaric Tags for Relative and Absolute Quantification) is NOT TMT
- Both use isobaric labeling for quantitative proteomics
- iTRAQ 8-plex uses different mass tags than TMT
- Processing pipeline is SIMILAR to TMT (lightweight adapter)
- Data is PRE-NORMALIZED by authors (ratios, not raw intensities)

### Cross-Species Annotation Challenge
- Primary species: Bos taurus (bovine)
- Bovine matrisome reference not available in repository
- Solution: Keyword-based ECM classification
- Match levels:
  - Level 1 (100%): Direct UniProt match to human/mouse matrisome
  - Level 3 (70%): Keyword-based classification (e.g., "collagen" → Collagens)
- Future improvement: Add bovine matrisome reference

### Sample Pooling Decision
- Excluded "Pool Foetus" and "Pool Old" samples
- Rationale: Pooled samples are not independent biological replicates
- Including them would artificially inflate sample size
- Only independent biological replicates included

---

## Files Created

### Paper Folder Structure
```
05_papers_to_csv/03_Caldeira_2017_paper_to_csv/
├── itraq_adapter_caldeira2017.py          (Processing script)
├── Caldeira_2017_wide_format.csv          (Wide-format output, 43 rows)
├── zscore_metadata_Caldeira_2017.json     (Z-score normalization metadata)
├── README.md                               (Study documentation)
└── 00_TASK_CALDEIRA_2017_iTRAQ_PROCESSING.md  (This file)
```

### Updated Unified Database
- `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` (4031 rows, 9 studies)
- `08_merged_ecm_dataset/unified_metadata.json` (Study metadata)
- Backups in `08_merged_ecm_dataset/backups/`

---

## Validation Summary

All success criteria met:

- [x] **File loaded:** MOESM3 with 104 proteins
- [x] **ECM identification:** 43 proteins (41.3% coverage)
- [x] **Age mapping:** Young vs Old (Foetus excluded)
- [x] **Missing values:** 9.3% (acceptable for iTRAQ)
- [x] **Schema compliance:** 18 columns, all required fields present
- [x] **No NaN in critical columns:** Dataset_Name, Organ, Compartment all filled
- [x] **Method field correct:** "iTRAQ 8-plex LC-MS/MS" (not "TMT")
- [x] **Species consistency:** Bos taurus
- [x] **Abundance validation:** Mean differences between Young/Old observed
- [x] **Wide-format CSV created:** 43 rows
- [x] **Merged to unified database:** 4031 total rows
- [x] **Z-scores calculated:** Log2 transform applied, validation passed
- [x] **Documentation complete:** README.md and task file created

---

## Comparison with Other Studies

### Similar Studies in Database
1. **Tam 2020** - Human intervertebral disc (LFQ)
   - Same tissue type (nucleus pulposus)
   - Different method (LFQ vs iTRAQ)
   - Higher ECM coverage (426 proteins vs 43)
2. **Schuler 2021** - Human skeletal muscle (LFQ)
   - Different tissue (muscle vs disc)
   - Both are collagen-rich tissues
   - Similar missing value rates

### Unique Contribution
- **Only bovine dataset** in database (all others are human/mouse)
- **First iTRAQ dataset** processed (others are LFQ, TMT, or SILAC)
- **Intervertebral disc focus** - only Tam 2020 shares tissue type
- **Developmental comparison** (foetus data available but excluded)

---

## Issues Encountered and Solutions

### Issue 1: MOESM2 vs MOESM3
- **Problem:** Two similar Excel files with different protein counts
- **Solution:** Compared both, chose MOESM3 (104 proteins vs 81)
- **Rationale:** MOESM3 is more complete and has Species column

### Issue 2: No Bovine Matrisome Reference
- **Problem:** Repository only has human and mouse matrisome references
- **Solution:** Used keyword-based ECM classification
- **Impact:** Lower match confidence (70% vs 100%) but acceptable coverage

### Issue 3: Pool Samples
- **Problem:** Dataset includes pooled samples (Pool Foetus, Pool Old)
- **Solution:** Excluded from analysis (not independent replicates)
- **Rationale:** Pooled samples would artificially inflate sample size

### Issue 4: High Skewness
- **Problem:** Skewness > 1 for both age groups (Young=2.43, Old=1.38)
- **Solution:** Applied log2(x+1) transformation before z-score calculation
- **Result:** Validation passed (mean=0, std=1)

---

## Lessons Learned

1. **iTRAQ processing is similar to TMT** - Use TMT adapter approach
2. **Pre-normalized data is faster** - No complex parsing needed
3. **Cross-species annotation works** - Keyword matching is effective
4. **Pooled samples should be excluded** - Only independent replicates
5. **Documentation is critical** - Future users need method details

---

## Future Improvements

1. **Add bovine matrisome reference** to improve annotation confidence
2. **Cross-validate with human disc studies** (Tam 2020)
3. **Analyze developmental patterns** using excluded foetus data
4. **Compare iTRAQ vs LFQ** for same tissue type
5. **Add tissue-specific ECM markers** for disc biology

---

## References

1. Caldeira J, et al. (2017) Matrisome Profiling During Intervertebral Disc Development And Ageing. Scientific Reports. DOI: 10.1038/s41598-017-11960-0
2. ECM Atlas Pipeline Documentation: `11_subagent_for_LFQ_ingestion/03_TMT_vs_LFQ_PROCESSING_GUIDE.md`
3. Universal Z-Score Function: `11_subagent_for_LFQ_ingestion/02_ZSCORE_CALCULATION_UNIVERSAL_FUNCTION.md`

---

**Task completed by:** Claude Code Agent
**Date:** 2025-10-14
**Total processing time:** ~45 minutes
**Status:** COMPLETE - All phases done, all files created, all validation passed
