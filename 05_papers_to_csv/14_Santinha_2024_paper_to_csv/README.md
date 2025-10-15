# Santinha et al. 2024 - Dual-Species TMT Cardiac Aging Proteomics

## Study Information
- **Title:** Age-related extracellular matrix changes in mouse and human cardiac ventricles
- **Species:** **DUAL-SPECIES** - Mus musculus (Mouse) AND Homo sapiens (Human)
- **Tissue:** Left Ventricle (LV) cardiac tissue
- **Method:** TMT-10plex LC-MS/MS
- **Special Features:**
  - Dual-species comparative study
  - HGPS (Hutchinson-Gilford Progeria Syndrome) model included
  - Native tissue AND decellularized tissue (mouse only)

## Age Groups

### Mouse (Mus musculus)
- **Young:** 3 months (n=5, C57BL/6 wild-type males)
- **Old:** 20 months (n=5, C57BL/6 wild-type males)
- **Age gap:** 17 months
- **% of lifespan:** Young=12.5%, Old=83% (24-month lifespan)

### Human (Homo sapiens)
- **Young:** Age information not specified in supplementary tables (reproductive age, estimated ~30-40 years)
- **Old:** Age information not specified in supplementary tables (advanced age, estimated ~60-75 years)
- **Note:** Exact ages to be confirmed from main manuscript

## Processing Summary
This dataset was processed using the **TMT adapter pipeline** (lightweight transformation):
- ✅ Dual-species processing with separate Study_IDs
- ✅ Pre-normalized TMT data (log2-transformed abundances back-calculated from logFC/AveExpr)
- ✅ Three separate datasets: Mouse_NT, Mouse_DT, Human
- ✅ No missing values (100% data completeness for TMT)
- ✅ Species-specific matrisome reference matching
- ✅ Processing time: ~5 minutes per dataset

## Files

### Input Data
Located in `../../data_raw/Santinha et al. - 2024/`:
- `mmc1.xlsx` - Mouse age groups and sample metadata
- `mmc2.xlsx` - Mouse Native Tissue (NT) old vs young (4,827 proteins)
- `mmc5.xlsx` - Human old vs young (3,922 proteins)
- `mmc6.xlsx` - Mouse Decellularized Tissue (DT) old vs young (4,089 proteins)
- `mmc8.xlsx` - Mouse progeria vs young (not processed - HGPS model)
- `mmc16.docx` - Detailed methods document

### Output Files
- `Santinha_2024_wide_format.csv` - **553 ECM proteins** in unified schema
  - 18 columns following ECM Atlas standard
  - Combined: 191 Mouse_NT + 155 Mouse_DT + 207 Human
  - Ready for downstream analysis

### Scripts
- `tmt_adapter_santinha2024.py` - TMT adapter script
  - Loads TMT differential expression results
  - Back-calculates Young/Old abundances from logFC and AveExpr
  - Annotates with species-specific matrisome references
  - Maps to unified schema
  - Exports combined wide-format CSV

### Metadata
- `zscore_metadata_Santinha_2024_Mouse_NT.json` - Z-score normalization metadata
- `zscore_metadata_Santinha_2024_Mouse_DT.json` - Z-score normalization metadata
- `zscore_metadata_Santinha_2024_Human.json` - Z-score normalization metadata
  - Normalization parameters (mean, std)
  - Validation results (μ≈0, σ≈1)
  - Outlier statistics
  - Log-transformation status

### Documentation
- `README.md` - This file
- `00_TASK_SANTINHA_2024_TMT_PROCESSING.md` - Complete task specification (to be created)

## Data Quality

### Mouse Native Tissue (NT)
- **ECM proteins:** 191/4,827 total proteins (4.0%)
- **Missing values:** Young=0, Old=0 (100% completeness)
- **Matrisome categories:**
  - ECM Regulators: 68 (35.6%)
  - ECM Glycoproteins: 54 (28.3%)
  - ECM-affiliated Proteins: 34 (17.8%)
  - Secreted Factors: 14 (7.3%)
  - Collagens: 13 (6.8%)
  - Proteoglycans: 8 (4.2%)
- **Annotation quality:**
  - Level 1 (Gene Symbol): 186 (97.4%)
  - Level 2 (UniProt ID): 0
  - Level 3 (Synonym): 5 (2.6%)
- **Match confidence:** 100% (all annotated proteins)

### Mouse Decellularized Tissue (DT)
- **ECM proteins:** 155/4,089 total proteins (3.8%)
- **Missing values:** Young=0, Old=0 (100% completeness)
- **Matrisome categories:**
  - ECM Glycoproteins: 61 (39.4%)
  - ECM Regulators: 37 (23.9%)
  - ECM-affiliated Proteins: 19 (12.3%)
  - Collagens: 18 (11.6%)
  - Secreted Factors: 12 (7.7%)
  - Proteoglycans: 8 (5.2%)
- **Annotation quality:**
  - Level 1 (Gene Symbol): 151 (97.4%)
  - Level 2 (UniProt ID): 0
  - Level 3 (Synonym): 4 (2.6%)
- **Match confidence:** 100% (all annotated proteins)
- **Note:** Decellularized tissue enriched for structural ECM (Glycoproteins, Collagens)

### Human Native Tissue
- **ECM proteins:** 207/3,922 total proteins (5.3%)
- **Missing values:** Young=0, Old=0 (100% completeness)
- **Matrisome categories:**
  - ECM Regulators: 68 (32.9%)
  - ECM Glycoproteins: 63 (30.4%)
  - ECM-affiliated Proteins: 25 (12.1%)
  - Secreted Factors: 23 (11.1%)
  - Collagens: 19 (9.2%)
  - Proteoglycans: 9 (4.3%)
- **Annotation quality:**
  - Level 1 (Gene Symbol): 194 (93.7%)
  - Level 2 (UniProt ID): 1 (0.5%)
  - Level 3 (Synonym): 12 (5.8%)
- **Match confidence:** 100% (all annotated proteins)
- **Note:** Higher ECM protein percentage reflects cardiac ECM-rich tissue

### Combined Dataset
- **Total ECM proteins:** 553 rows
- **Unique proteins:** 429 (some proteins appear in multiple datasets)
- **Species breakdown:**
  - Mouse: 346 rows (62.6%) across 2 compartments
  - Human: 207 rows (37.4%) in 1 compartment
- **Data completeness:** 100% for both Young and Old across all datasets

## Processing Pipeline

### Phase 1: TMT Adapter ✅
**Script:** `tmt_adapter_santinha2024.py`

**Steps:**
1. Load differential expression results (logFC, AveExpr format)
2. Back-calculate Young/Old abundances:
   - `log2(Young) = AveExpr - (logFC / 2)`
   - `log2(Old) = AveExpr + (logFC / 2)`
3. Annotate with species-specific matrisome references:
   - Mouse: `references/mouse_matrisome_v2.csv` (1,110 genes)
   - Human: `references/human_matrisome_v2.csv` (1,026 genes)
4. Filter to ECM proteins only (Match_Confidence > 0)
5. Map to unified 18-column schema
6. Combine all three datasets into single CSV

**Command:**
```bash
cd /Users/Kravtsovd/projects/ecm-atlas
python 05_papers_to_csv/14_Santinha_2024_paper_to_csv/tmt_adapter_santinha2024.py
```

**Output:** `Santinha_2024_wide_format.csv`

### Phase 2: Merge to Unified Database ✅
**Script:** `11_subagent_for_LFQ_ingestion/merge_to_unified.py`

**Steps:**
1. Backup existing unified CSV
2. Load new study CSV
3. Validate schema compatibility
4. Concatenate to unified database
5. Update metadata

**Command:**
```bash
python 11_subagent_for_LFQ_ingestion/merge_to_unified.py \
    05_papers_to_csv/14_Santinha_2024_paper_to_csv/Santinha_2024_wide_format.csv
```

**Output:** Updated `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` (+553 rows)

### Phase 3: Calculate Z-scores ✅
**Script:** `11_subagent_for_LFQ_ingestion/universal_zscore_function.py`

Calculated separately for each dataset to maintain biological compartment specificity:

**Commands:**
```bash
# Mouse Native Tissue
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Santinha_2024_Mouse_NT' 'Tissue'

# Mouse Decellularized Tissue
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Santinha_2024_Mouse_DT' 'Tissue'

# Human
python 11_subagent_for_LFQ_ingestion/universal_zscore_function.py \
    'Santinha_2024_Human' 'Tissue'
```

**Results:**
- **Mouse_NT:** 191 proteins, no log-transform, μ=0 σ=1 (validated)
- **Mouse_DT:** 155 proteins, no log-transform, μ=0 σ=1 (validated)
- **Human:** 207 proteins, no log-transform, μ=0 σ=1 (validated)

**Grouping:**
- All datasets grouped by `Tissue` (single compartment per dataset)
- Mouse_NT: Heart_Native_Tissue
- Mouse_DT: Heart_Decellularized_Tissue
- Human: Heart_Native_Tissue

**Normalization:**
- No log2-transformation applied (data already log2-scaled, skewness < 1)
- Z-score: (Abundance - μ) / σ per compartment
- Validation: mean ≈ 0, std ≈ 1 (all passed)
- Outliers (|z| > 3): <1% across all datasets

**Output:** Updated unified CSV with z-score columns, metadata JSONs

## Validation

### ✅ All Success Criteria Met

#### Phase 1 Validation
- [x] Three datasets loaded successfully
- [x] Row counts correct (191 + 155 + 207 = 553)
- [x] No null critical fields (Protein_ID, Species, Organ, Compartment, Dataset_Name)
- [x] Species correctly assigned (Mouse vs Human)
- [x] Abundances back-calculated correctly (no NaN)
- [x] Schema compliance (18 columns)
- [x] ECM classification preserved (6 categories)
- [x] Match confidence 100% for all ECM proteins
- [x] Species consistency per dataset
- [x] Wide-format CSV created
- [x] Adapter script documented

#### Phase 2 Validation
- [x] Merged to unified CSV successfully
- [x] 553 rows added (191 + 155 + 207)
- [x] No duplicates detected
- [x] Backup created
- [x] Metadata updated
- [x] Three Study_IDs correctly separated

#### Phase 3 Validation
- [x] Z-scores calculated for all three datasets
- [x] Grouping by Tissue correct
- [x] No log-transformation (data already log2-scaled)
- [x] Mean ≈ 0, Std ≈ 1 validation passed
- [x] Outliers < 1% for all datasets
- [x] Metadata JSONs created for each dataset
- [x] NaN handling correct (no missing values in TMT data)

## Technical Notes

### Dual-Species Handling
This is a **dual-species dataset** - special considerations:
1. **Separate Study_IDs:** Each dataset has unique identifier to prevent mixing
   - `Santinha_2024_Mouse_NT`
   - `Santinha_2024_Mouse_DT`
   - `Santinha_2024_Human`
2. **Species-specific matrisome:** Different reference files used for annotation
3. **Separate z-score normalization:** Cannot normalize across species
4. **Compartment separation:** Native vs Decellularized kept distinct

### TMT Data Characteristics
- **Pre-normalized:** TMT reporter ion intensities already quantile-normalized
- **Log2-scaled:** Data in log2 space (AveExpr format)
- **Differential expression:** Provided as logFC (log2 fold-change)
- **Back-calculation:** Young/Old abundances derived from logFC + AveExpr
- **No missing values:** 100% data completeness typical for TMT

### Decellularization Effect
Mouse_DT (Decellularized Tissue) shows:
- **Enrichment for structural ECM:** Higher % of Glycoproteins and Collagens
- **Depletion of cellular proteins:** Lower total protein count vs NT
- **Different ECM profile:** Some proteins unique to DT vs NT
- **Biological insight:** Reveals stable ECM components surviving decellularization

### HGPS Model
The study includes progeria data (mmc8.xlsx) but this was **NOT processed** in current pipeline:
- HGPS = Hutchinson-Gilford Progeria Syndrome (accelerated aging model)
- Data available but not included in ECM Atlas aging comparison
- Could be processed separately if needed for progeria-specific analysis

## Comparison with Other Studies

### Coverage
- **Higher ECM %:** 4-5% ECM proteins is typical for cardiac tissue
- **Comparable to:** Ouni 2022 (ovary), Angelidis 2019 (lung)
- **Lower than:** Tam 2020 (intervertebral disc, highly ECM-rich)

### Methodology
- **TMT advantages:** No missing values, precise quantification, multiplexing
- **Similar to:** Ouni 2022 (also TMT)
- **Different from:** Randles 2021, Tam 2020 (LFQ with 50-80% missing values)

### Tissue Type
- **First cardiac study** in ECM Atlas
- **Complements:** Lung (Angelidis), Kidney (Randles), Disc (Tam), Ovary (Ouni)
- **Cross-tissue comparison:** Now possible with heart added

## Known Limitations

1. **Age information incomplete:** Human age groups not specified in supplementary tables
2. **Single compartment (human):** Only native tissue for human (no decellularized)
3. **HGPS data not processed:** Progeria model data available but excluded
4. **Sample size:** n=5 per group (typical for TMT but smaller than some LFQ studies)
5. **Species-specific analysis required:** Cannot directly compare mouse vs human ECM

## Future Enhancements

Potential extensions of this dataset:
1. **Add HGPS data:** Process mmc8.xlsx progeria vs young comparison
2. **Confirm human ages:** Extract from main manuscript for accurate metadata
3. **Cross-species analysis:** Compare orthologous ECM proteins mouse vs human
4. **Cardiac-specific insights:** Focus on heart ECM aging signatures
5. **Decellularization analysis:** Compare NT vs DT to identify stable ECM core

## References

- **Data source:** Santinha et al. 2024 supplementary materials
- **Matrisome database:** Naba Lab matrisome reference v2
- **TMT methodology:** TMT-10plex LC-MS/MS (Thermo Fisher)
- **Processing pipeline:** ECM Atlas universal normalization framework

---

**Created:** 2025-10-14
**Updated:** 2025-10-14
**Status:** ✅ Complete - All 3 Phases Done
**Datasets added to ECM Atlas:** 3 (Mouse_NT, Mouse_DT, Human)
**Total ECM proteins:** 553 rows (429 unique proteins)
