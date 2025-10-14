# Santinha et al. 2024 - Comprehensive Analysis

**Compilation Date:** 2025-10-13
**Study Type:** Non-LFQ (TMT Isobaric Labeling - Deferred to Phase 3)
**Sources:** User-provided study metadata

---

## 1. Paper Overview

- **Title:** Remodeling of the Cardiac Extracellular Matrix Proteome During Chronological and Pathological Aging
- **Journal:** Molecular & Cellular Proteomics (MCP), 2024
- **DOI:** 10.1016/j.mcpro.2023.100706
- **PMID:** 38141925
- **Tissue:** Cardiac left ventricle
- **Species:** Dual-species study
  - **Mouse:** Mus musculus (C57BL/6J)
  - **Human:** Homo sapiens
- **Age groups:**
  - **Mouse:** 3 months (young), 20 months (old), 10 weeks HGPS (progeria model)
  - **Human:** 20-31 years (young), 65-70 years (old), 10-14 years HGPS (Hutchinson-Gilford Progeria Syndrome)
- **Design:** Multi-species aging study with progeria disease model
- **PRIDE repositories:**
  - Mouse data: PXD039548
  - Human data: PXD040234

---

## 2. Method Classification & Phase Assignment

### Quantification Method
- **Method:** TMT (Tandem Mass Tags) - Isobaric labeling LC-MS/MS
- **Labeling chemistry:** TMT isobaric tags enable multiplexed quantification
- **Workflow:** Labeled peptide quantification via MS2 reporter ions
- **Reference:** Methods section in paper (DOI: 10.1016/j.mcpro.2023.100706)

### Non-LFQ Classification
- **Label-free?** ❌ NO - Uses TMT isobaric labeling
- **Isobaric tags:** TMT enables simultaneous quantification of multiple samples
- **Normalization:** TMT-specific normalization to account for labeling efficiency and channel bias
- **Reason for exclusion:** Labeled method incompatible with Phase 1 LFQ normalization pipeline

### Phase Assignment
- **Phase 1 (LFQ):** ❌ EXCLUDED - TMT is labeled method, not label-free
- **Phase 2 (Age bins):** ⏸️ SKIPPED - Not applicable to non-LFQ studies
- **Phase 3 (Labeled methods):** ✅ DEFERRED - TMT studies processed in Phase 3
- **Status:** Awaiting Phase 3 implementation with isobaric normalization

---

## 3. Age Bin Normalization Strategy (Phase 3 Preview)

### Mouse Age Design
- **Young cohort:** 3 months
  - Biological context: Young adult mice, sexually mature
  - Lifespan percentage: ~10-12% of maximum lifespan (~24-30 months)
- **Old cohort:** 20 months
  - Biological context: Geriatric age, near end of typical lifespan
  - Lifespan percentage: ~67-83% of maximum lifespan
- **HGPS model:** 10 weeks
  - Biological context: Accelerated aging progeria model
  - Not used for chronological aging bins (pathological aging model)

### Human Age Design
- **Young cohort:** 20-31 years
  - Biological context: Young adults, post-development
  - Lifespan percentage: ~25-38% of maximum lifespan (~80 years)
- **Old cohort:** 65-70 years
  - Biological context: Elderly adults, retirement age
  - Lifespan percentage: ~81-87% of maximum lifespan
- **HGPS patients:** 10-14 years
  - Biological context: Pediatric progeria patients with accelerated aging
  - Not used for chronological aging bins (pathological aging model)

### Normalization Assessment (Phase 3)
- **Current design:** Already binary for chronological aging (young vs old)
- **Mouse groups:** 3mo (young) vs 20mo (old) - clear separation
- **Human groups:** 20-31yr (young) vs 65-70yr (old) - clear separation
- **HGPS exclusion:** Progeria samples should be analyzed separately as pathological aging model
- **Data retention:** 100% for chronological aging analysis
- **Conclusion:** NO NORMALIZATION REQUIRED for young/old comparison; HGPS requires separate pathological aging analysis

---

## 4. Column Mapping to 13-Column Schema (Phase 3 Preview)

### Expected Data Structure
Based on typical TMT workflows and MCP journal format:

| Schema Column | Expected Source | Status | Notes |
|---------------|----------------|--------|-------|
| **Protein_ID** | UniProt accession column | ⏸️ PENDING | Likely primary protein identifier; extract canonical ID if semicolon-separated |
| **Protein_Name** | Protein names column | ⏸️ PENDING | Full protein names from UniProt annotation |
| **Gene_Symbol** | Gene names/symbols column | ⏸️ PENDING | Gene nomenclature (HGNC for human, MGI for mouse) |
| **Tissue** | Constant `Cardiac left ventricle` | ✅ MAPPED | Study focused exclusively on left ventricle cardiac tissue |
| **Species** | Parse from sample metadata | ⏸️ PENDING | `Mus musculus` for mouse samples, `Homo sapiens` for human samples |
| **Age** | Extract from sample columns | ⏸️ PENDING | Mouse: 3, 20 (months) or 10 (weeks HGPS); Human: 20-31, 65-70 (years) or 10-14 (HGPS) |
| **Age_Unit** | Derived from species/cohort | ⏸️ PENDING | `months` for mouse, `years` for human, `weeks` for mouse HGPS |
| **Abundance** | TMT reporter intensity columns | ⏸️ PENDING | TMT channel intensities per sample |
| **Abundance_Unit** | Constant `TMT_normalized_intensity` | ✅ MAPPED | TMT reporter ion intensities with normalization |
| **Method** | Constant `TMT isobaric labeling LC-MS/MS` | ✅ MAPPED | Phase 3 labeled method |
| **Study_ID** | Constant `Santinha_2024` | ✅ MAPPED | Unique identifier for downstream joins |
| **Sample_ID** | Derive from column headers | ⏸️ PENDING | Format: `{species}_{age}_{condition}_{replicate}` |
| **Parsing_Notes** | Template (see below) | ✅ MAPPED | Capture TMT-specific context, dual-species design, HGPS model notes |

### Mapping Status
- ⏸️ **Awaiting data files** - Supplementary data not yet obtained from journal/PRIDE
- ⏸️ **Species separation required** - Mouse and human data likely in separate files or sheets
- ⏸️ **HGPS handling** - Progeria samples need special annotation for pathological aging analysis

### TMT-Specific Considerations
1. **Multiplexing design:** TMT allows multiple samples per LC-MS run; identify TMT plex size (6-plex, 10-plex, 11-plex, 16-plex)
2. **Batch effects:** If multiple TMT batches used, check for batch bridging channels
3. **Normalization:** TMT data typically pre-normalized for loading and channel bias
4. **Sample pooling:** Identify if pooled reference channels used for normalization
5. **Missing values:** TMT can have channel-specific missing values; document imputation strategy

---

## 5. Parsing Implementation (Deferred to Phase 3)

### Data Acquisition Requirements
**Primary sources:**
1. **Journal supplementary files:** Download from MCP journal website (DOI: 10.1016/j.mcpro.2023.100706)
2. **PRIDE repositories:**
   - Mouse data: https://www.ebi.ac.uk/pride/archive/projects/PXD039548
   - Human data: https://www.ebi.ac.uk/pride/archive/projects/PXD040234

**Expected file types:**
- Excel (.xlsx) supplementary tables with TMT intensities
- Possibly separate files for mouse vs human data
- May include MaxQuant or Proteome Discoverer output tables

### Parsing Steps (Phase 3 Implementation)

**Step 1: File Download & Inspection**
```python
# Download supplementary files from MCP journal
# Check for multiple sheets (mouse vs human, raw vs normalized)
# Identify TMT plex size and sample column naming convention
```

**Step 2: Species Separation**
```python
# Parse mouse data
mouse_samples = [col for col in df.columns if 'mouse' in col.lower() or '3mo' in col or '20mo' in col]

# Parse human data
human_samples = [col for col in df.columns if 'human' in col.lower() or 'yr' in col]

# HGPS samples (separate analysis)
hgps_samples = [col for col in df.columns if 'HGPS' in col or 'progeria' in col]
```

**Step 3: Age Mapping**
```python
# Mouse chronological aging
if '3mo' in column_name or '3_mo' in column_name:
    age = 3
    age_unit = 'months'
    condition = 'chronological_young'
elif '20mo' in column_name or '20_mo' in column_name:
    age = 20
    age_unit = 'months'
    condition = 'chronological_old'
elif '10wk' in column_name and 'HGPS' in column_name:
    age = 10
    age_unit = 'weeks'
    condition = 'pathological_HGPS'

# Human chronological aging
if 'young' in column_name and 'human' in column_name:
    age = 25  # Representative midpoint of 20-31yr range
    age_unit = 'years'
    condition = 'chronological_young'
elif 'old' in column_name and 'human' in column_name:
    age = 67  # Representative midpoint of 65-70yr range
    age_unit = 'years'
    condition = 'chronological_old'
elif 'HGPS' in column_name and 'human' in column_name:
    age = 12  # Representative midpoint of 10-14yr range
    age_unit = 'years'
    condition = 'pathological_HGPS'
```

**Step 4: Sample_ID Format**
```python
# Template: {species}_{age}{unit}_{condition}_{replicate}
# Examples:
#   - Mouse: "mouse_3mo_young_rep1", "mouse_20mo_old_rep2", "mouse_10wk_HGPS_rep1"
#   - Human: "human_25yr_young_rep1", "human_67yr_old_rep2", "human_12yr_HGPS_rep1"
```

**Step 5: Parsing_Notes Template**
```python
parsing_notes = (
    f"Species={species}; Age={age}{age_unit}; Condition={condition}; "
    f"TMT reporter intensity from {tmt_plex} experiment; "
    f"Cardiac left ventricle tissue; "
    f"PRIDE={pride_id}; "
    f"Replicate={replicate_num}"
)
```

### Expected Output (Phase 3)
- **Format:** Long-format CSV with 13 columns
- **Species separation:** Separate CSV files for mouse vs human, or Species column for combined file
- **HGPS annotation:** Pathological aging marked in Parsing_Notes or separate Condition column
- **Expected rows:** ~300 ECM proteins × (multiple samples per species) = estimated 3,000-6,000 rows depending on replicate count

### Preprocessing Requirements
- ⏸️ **Data not yet available** - Requires downloading supplementary files
- ⚠️ **TMT normalization verification** - Confirm if data is pre-normalized or requires normalization
- ⚠️ **Batch correction** - Check for batch effects across TMT runs
- ⚠️ **HGPS filtering** - Decide whether to include progeria samples in main atlas or separate pathological aging database

---

## 6. Quality Assurance & Biological Context

### Study Design Strengths
- ✅ **Dual-species design:** Mouse and human cardiac aging enables cross-species comparison
- ✅ **Progeria model:** HGPS samples provide accelerated aging comparison
- ✅ **Tissue specificity:** Cardiac left ventricle focus enables comparison with other cardiac aging studies
- ✅ **Binary age design:** Young vs old already defined for chronological aging
- ✅ **ECM enrichment:** ~300 ECM proteins quantified - excellent matrisome coverage
- ✅ **PRIDE availability:** Raw data available for reanalysis (PXD039548, PXD040234)

### Known Limitations & Considerations
1. **TMT multiplexing constraints:**
   - Limited dynamic range compared to LFQ
   - Potential ratio compression from co-isolated peptides
   - Channel-specific missing values possible
2. **Age range heterogeneity:**
   - Human young cohort spans 11 years (20-31yr) - larger variance than typical mouse cohorts
   - Human old cohort relatively narrow (65-70yr) - smaller variance
3. **HGPS interpretation:**
   - Progeria is accelerated aging model but not equivalent to chronological aging
   - HGPS data should be analyzed separately from young/old comparisons
   - Pathological vs chronological aging mechanisms may differ
4. **Species lifespan normalization:**
   - Mouse 20mo ≈ 80% lifespan; Human 67yr ≈ 84% lifespan - roughly equivalent
   - Mouse 3mo ≈ 12% lifespan; Human 25yr ≈ 31% lifespan - less equivalent (mouse younger relative to lifespan)
5. **Replicate count unknown:**
   - Supplementary data inspection needed to determine statistical power
   - Typical TMT studies use 3-6 biological replicates per condition

### Key Biological Findings (From Abstract/Title)
- **ECM remodeling:** Study identifies ECM changes during cardiac aging
- **13 upregulated proteins:** Aging-associated proteins identified
- **Lactadherin marker:** Identified as aging biomarker in cardiac tissue
- **Chronological vs pathological:** Comparison of natural aging vs progeria-driven changes

### Cross-Study Comparisons
- **Similar TMT studies in atlas:**
  - Tsumagari 2023 (mouse brain, TMTpro 11-plex)
  - Ouni 2022 (human pancreas, TMT 10-plex)
  - Lofaro 2021 (human skeletal muscle, TMT)
- **Cardiac aging context:** First cardiac-focused study in ECM-Atlas
- **Dual-species value:** Enables human-mouse cardiac ECM aging comparison
- **Progeria comparison:** Complements chronological aging studies with pathological model

### Integration Opportunities (Phase 3)
1. **Cross-species cardiac aging:** Compare mouse vs human cardiac ECM changes
2. **Matrisome analysis:** Classify ~300 ECM proteins using Matrisome AnalyzeR
3. **Progeria validation:** Use HGPS samples to identify accelerated aging signatures
4. **Lactadherin follow-up:** Cross-reference with other studies for universal aging marker
5. **Tissue comparison:** Compare cardiac ECM aging with lung (Angelidis), kidney (Randles), skin (Li)

---

## 7. Phase 3 Deferral Status

### Why Excluded from Phase 1
**Primary reason:** TMT isobaric labeling incompatible with label-free quantification normalization

| Aspect | LFQ (Phase 1) | TMT (This Study) |
|--------|---------------|-------------------|
| **Labeling** | None | TMT isobaric tags |
| **Quantification** | MS1 precursor intensity | MS2 reporter ion intensity |
| **Normalization** | Between-sample median normalization | Within-TMT-plex channel normalization |
| **Batch effects** | Sample-to-sample variation | TMT-plex batch effects |
| **Dynamic range** | Higher (MS1) | Lower (MS2 ratio compression) |
| **Integration risk** | Cannot mix with labeled methods | Requires Phase 3 separate pipeline |

### Phase 3 Readiness Checklist
- ⏸️ **Data acquisition:** Supplementary files not yet downloaded
- ⏸️ **File structure inspection:** TMT channel naming unknown
- ⏸️ **Replicate count:** Unknown until data inspection
- ⏸️ **Species file separation:** Determine if mouse/human in separate files
- ⏸️ **TMT plex size:** Unknown (6-plex, 10-plex, 11-plex, 16-plex?)
- ⏸️ **Normalization status:** Confirm if data is pre-normalized
- ✅ **Biological relevance:** High priority - cardiac aging with dual-species design
- ✅ **PRIDE availability:** Raw data accessible for reanalysis

### Phase 3 Parsing Priority
- **Priority:** HIGH (Tier 1)
- **Rationale:**
  1. First cardiac-focused study in ECM-Atlas
  2. Dual-species (mouse + human) enables cross-species comparison
  3. ~300 ECM proteins - excellent matrisome coverage
  4. HGPS progeria model adds pathological aging dimension
  5. Published in high-impact journal (MCP) with rigorous methods
  6. PRIDE data availability enables reanalysis and validation
- **Recommendation:** Parse immediately when Phase 3 begins; use as pilot for dual-species TMT integration

### Next Steps for Phase 3
1. **Download supplementary data** from MCP journal (DOI: 10.1016/j.mcpro.2023.100706)
2. **Inspect file structure** to identify TMT channels, species separation, replicate design
3. **Download PRIDE raw data** if supplementary tables insufficient
4. **Implement species-specific parsing** for mouse (PXD039548) and human (PXD040234)
5. **Validate HGPS handling** - decide on inclusion in main atlas or separate pathological aging analysis
6. **Cross-reference lactadherin** with other studies to validate as universal aging marker
7. **Generate cardiac aging signatures** for comparison with other tissues

---

## 8. Data Availability

### Published Resources
- **Journal:** Molecular & Cellular Proteomics (2024)
- **DOI:** 10.1016/j.mcpro.2023.100706
- **PMID:** 38141925
- **Open access:** Likely (check journal policy)

### Raw Data Repositories
- **Mouse proteomics:** PRIDE PXD039548
  - URL: https://www.ebi.ac.uk/pride/archive/projects/PXD039548
  - Data type: Raw MS files + MaxQuant/Proteome Discoverer outputs
- **Human proteomics:** PRIDE PXD040234
  - URL: https://www.ebi.ac.uk/pride/archive/projects/PXD040234
  - Data type: Raw MS files + MaxQuant/Proteome Discoverer outputs

### Supplementary Files (Expected)
- **Table S1:** Likely contains mouse TMT proteomics data
- **Table S2:** Likely contains human TMT proteomics data
- **Table S3:** Possibly HGPS-specific comparisons
- **Table S4:** Possibly differential expression analysis (young vs old)
- **Figure data:** May include processed data for figures

---

**Compilation Notes:**
- **Source:** User-provided metadata (title, DOI, PMID, PRIDE, method, findings)
- **Data files:** Not yet acquired - analysis based on typical TMT study structure and MCP journal format
- **Template:** Follows Tsumagari 2023 structure for Phase 3 non-LFQ studies
- **Status:** Ready for Phase 3 data acquisition and parsing implementation

**Phase 3 Recommendations:**
- 🔴 **HIGH PRIORITY:** First cardiac study + dual-species + excellent ECM coverage
- 📊 **Pilot study candidate:** Use for testing dual-species TMT integration pipeline
- 🧬 **Cross-species comparison:** Enables mouse-human cardiac aging signature identification
- 🔬 **Progeria model:** HGPS samples add pathological aging dimension (handle separately)
- 📈 **Lactadherin validation:** Cross-reference aging marker with other tissues/studies
