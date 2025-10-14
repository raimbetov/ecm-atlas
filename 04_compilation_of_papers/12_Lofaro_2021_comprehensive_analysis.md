# Lofaro et al. 2021 - Comprehensive Analysis

**Study ID:** 12_Lofaro_2021
**Status:** LFQ Study - Phase 1 Ready
**Last Updated:** 2025-10-13

---

## 1. Paper Overview

### Publication Details
- **Full Title:** Age-Related Changes in the Matrisome of the Mouse Skeletal Muscle
- **Authors:** Francesco Demetrio Lofaro, Andrea Smirnov Sartori, Cecilia Catoni, Alberto Caretti, Manuela Marcoli, Guido Maura, Ute Moll, Katia Cortese, Silvia Ravera
- **Journal:** International Journal of Molecular Sciences (IJMS)
- **Year:** 2021
- **DOI:** 10.3390/ijms221910564
- **PMID:** 34686327
- **PRIDE Accession:** PXD027895

### Biological Context
- **Species:** Mouse (Mus musculus, male BALB/c strain)
- **Tissue:** Gastrocnemius skeletal muscle
- **Age Groups:**
  - Adult: 12 months
  - Old: 24 months
- **Sample Size:** 9 mice per age group (n=9)

### Scientific Rationale
The study investigates age-related changes in the extracellular matrix (ECM) composition of skeletal muscle. The matrisome undergoes significant remodeling during aging, which contributes to muscle dysfunction and sarcopenia. This study provides quantitative proteomic characterization of these ECM changes in the gastrocnemius muscle.

---

## 2. Method Classification & Quality Control

### Proteomic Method
- **Primary Method:** Label-Free Quantification (LFQ)
- **Specific Technique:** Liquid chromatography-tandem mass spectrometry (LC-MS/MS)
- **Instrumentation:** Not specified in abstract
- **Quantification:** Label-free intensity-based quantification

### Sample Preparation
- **Sequential Protein Extraction:**
  1. PBS (physiological buffer - soluble proteins)
  2. Urea/thiourea (cytoplasmic/nuclear proteins)
  3. Guanidinium-HCl (ECM/structural proteins)
- **Rationale:** Sequential extraction enriches for matrisomal proteins in the guanidinium-HCl fraction

### LFQ Compatibility: YES ✅
- **Verdict:** Fully compatible with unified ECM-Atlas schema
- **Reason:** Label-free quantification allows direct comparison with other LFQ studies (Angelidis, Randles, Tam, Dipali, LiDermis)
- **Quality:** High - uses appropriate extraction protocol for ECM enrichment

---

## 3. Age Bin Normalization Strategy

### Original Age Groups
| Age Bin | Age Value | Species Lifespan Context |
|---------|-----------|--------------------------|
| Adult   | 12 months | ~40-50% of max lifespan (~24-30mo) |
| Old     | 24 months | ~80-100% of max lifespan |

### Age Bin Normalization
**Status:** Already Binary - No normalization required ✅

**Rationale:**
- Study design already uses binary comparison (Adult vs Old)
- 12 months = middle age (approaching maturity)
- 24 months = aged/senescent (near end of lifespan)
- Clear biological separation (~12 months apart, equivalent to ~40 human years)

### Data Retention
- **Samples Retained:** 100% (18/18 mice)
- **Groups:** 2 age bins × 9 biological replicates each

### Final Age Mapping
```
Adult (12mo) → Young
Old (24mo)   → Old
```

---

## 4. Column Mapping to 13-Column Schema

### Source Files
- **Primary Data File:** Table S1 (PDF format in data_raw/)
- **Quantification File:** Table S2 (PDF format in data_raw/)
- **Location:** `data_raw/Lofaro et al. - 2021/`

**Note:** PDF tables will need to be converted to machine-readable format (CSV/Excel) for parsing.

### Column Mapping Table

| Schema Column | Source Column/Value | Data Type | Notes |
|---------------|---------------------|-----------|-------|
| **Protein_ID** | Extract from Table S1 | UniProt ID | May need UniProt mapping if Gene Symbols provided |
| **Protein_Name** | From Table S1 or UniProt API | String | If missing, enrich via UniProt API |
| **Gene_Symbol** | From Table S1 | String | Primary identifier in table |
| **Tissue** | Constant: "Skeletal muscle - Gastrocnemius" | String | Specific muscle type |
| **Species** | Constant: "Mus musculus" | String | BALB/c strain |
| **Age** | Constant per sample: 12 or 24 | Integer | In months |
| **Age_Unit** | Constant: "months" | String | |
| **Abundance** | From Table S2 quantification | Float | Label-free intensity values |
| **Abundance_Unit** | Constant: "LFQ intensity" | String | Or "precursor intensity" |
| **Method** | Constant: "Label-free LC-MS/MS" | String | |
| **Study_ID** | Constant: "Lofaro_2021" or DOI | String | |
| **Sample_ID** | Format: `Lofaro2021_M{mouse_num}_A{age}mo` | String | Example: Lofaro2021_M1_A12mo |
| **Parsing_Notes** | Document extraction method, matrisome classification | String | Note: "Sequential extraction - Guanidinium-HCl fraction" |

### Known Data Gaps
1. **PDF Extraction Required:**
   - Tables S1 and S2 are in PDF format
   - Need extraction pipeline: PDF → CSV (using tabula-py or manual export)

2. **Protein ID Format:**
   - Verify if Table S1 contains UniProt IDs or Gene Symbols only
   - Use UniProt Mapping API if only Gene Symbols available

3. **Sample-Level Quantification:**
   - Confirm if Table S2 provides individual replicate values (n=9 per group)
   - Or if only aggregated values (mean ± SD) are provided

---

## 5. Parsing Implementation Guide

### Step 1: Extract Data from PDF
```python
# Option A: Using tabula-py
import tabula
df_s1 = tabula.read_pdf("data_raw/Lofaro et al. - 2021/Table S1.pdf", pages='all')
df_s2 = tabula.read_pdf("data_raw/Lofaro et al. - 2021/Table S2.pdf", pages='all')

# Option B: Manual export from PDF to Excel/CSV
# Open PDF, select all, copy to Excel, save as CSV
```

### Step 2: Protein ID Mapping
```python
# If only Gene Symbols available
from bioservices import UniProt
u = UniProt(verbose=False)

def map_gene_to_uniprot(gene_symbol, organism='Mus musculus'):
    query = f'gene:{gene_symbol} AND organism:"{organism}"'
    result = u.search(query, columns='id,protein_names,genes')
    # Parse and return UniProt ID + Protein_Name
```

### Step 3: Generate Sample_IDs
```python
# For each biological replicate
sample_ids = []
for age_group in ['12mo', '24mo']:
    for mouse_num in range(1, 10):  # 9 mice per group
        sample_id = f"Lofaro2021_M{mouse_num}_A{age_group}"
        sample_ids.append(sample_id)
# Total: 18 samples
```

### Expected Output
- **Row Count:** Depends on number of matrisomal proteins identified
  - Estimate: 100-300 ECM proteins × 18 samples = 1,800-5,400 rows
- **Format:** Long format (one row per protein per sample)

### Data Quality Checks
1. Verify all 9 replicates per age group are present
2. Check for missing values (should be minimal for LFQ)
3. Validate UniProt IDs against current database
4. Confirm matrisome classification (use Matrisome AnalyzeR if needed)

### Ready for Parsing: PARTIAL ⚠️
**Blockers:**
- PDF tables need conversion to CSV/Excel first
- Protein ID format needs verification (Gene Symbol vs UniProt ID)

**Once resolved:** Ready for Phase 1 processing

---

## 6. Quality Assurance Notes

### Strengths
1. **Appropriate ECM Enrichment:** Sequential extraction with guanidinium-HCl specifically targets matrisomal proteins
2. **Binary Age Design:** No middle-aged groups to exclude
3. **Good Sample Size:** n=9 per group provides statistical power
4. **Focused Tissue:** Gastrocnemius is clinically relevant for sarcopenia studies
5. **Public Data:** PRIDE accession PXD027895 available

### Limitations
1. **Single Tissue:** Only gastrocnemius muscle (no other muscle types)
2. **Male-Only:** Results may not generalize to female mice
3. **Single Strain:** BALB/c may have strain-specific aging phenotypes
4. **PDF Supplementary Tables:** Requires extra extraction step vs Excel files

### Biological Considerations
- **Gastrocnemius Muscle:** Primarily glycolytic (fast-twitch) muscle
  - May have different matrisome aging signature than oxidative (slow-twitch) muscles
- **12 vs 24 months:** Captures middle-age to old-age transition
  - Comparable to human transition from ~45yr to ~75yr

### Cross-Study Comparisons
- **vs Angelidis 2019 (Lung):** Both use binary design, but different tissues
- **vs Dipali 2023 (Ovary):** Similar age range (10-12mo vs 24mo), different tissue
- **Potential Pairing:** Could combine with Schüler 2021 (also skeletal muscle MuSC niche) for muscle-specific ECM aging signatures

---

## 7. Integration into ECM-Atlas

### Phase Assignment
**Phase 1 (LFQ Studies)** - 6th study in processing queue

### Processing Priority
**Priority Level:** Medium-High
- **Rationale:**
  1. Skeletal muscle is underrepresented in current atlas (Lung, Kidney, Spine, Ovary, Dermis)
  2. Pairs well with Schüler 2021 for muscle-specific analysis
  3. Relatively clean binary design
- **Recommended Order:** Process after completing current 5 LFQ studies

### Expected Contributions
- Adds **skeletal muscle matrisome** to tissue panel
- Expands **mouse aging proteomics** (along with Angelidis, Dipali)
- Complements **MuSC niche study** (Schüler 2021) with bulk muscle ECM

---

## 8. Data Availability

### Raw Data
- **PRIDE:** PXD027895
- **URL:** http://www.ebi.ac.uk/pride/archive/projects/PXD027895

### Processed Data (in this repository)
- **PDF Tables:** `data_raw/Lofaro et al. - 2021/Table S1.pdf`, `Table S2.pdf`
- **CSV (to be generated):** `data_processed/Lofaro_2021/Lofaro_2021_matrisome_raw.csv`
- **Standardized Output:** `data_processed/Lofaro_2021/Lofaro_2021_standardized.csv`

### Publication PDF
- **To be downloaded to:** `pdf/Lofaro_2021_IJMS.pdf`

---

## 9. Parsing Roadmap

### Phase 1: Data Extraction (Manual/Script)
- [ ] Convert Table S1 to CSV (protein list)
- [ ] Convert Table S2 to CSV (quantification data)
- [ ] Verify column headers and structure

### Phase 2: ID Mapping
- [ ] Check if UniProt IDs present in Table S1
- [ ] If not, map Gene Symbols → UniProt IDs
- [ ] Fetch Protein_Name via UniProt API

### Phase 3: Standardization
- [ ] Generate Sample_IDs for 18 samples
- [ ] Reshape to long format (protein × sample)
- [ ] Add constant columns (Species, Tissue, Method, etc.)
- [ ] Calculate z-scores within each age group

### Phase 4: Quality Control
- [ ] Check for missing values
- [ ] Validate UniProt IDs (current vs deprecated)
- [ ] Plot abundance distributions
- [ ] Compare adult vs old (sanity check)

### Phase 5: Integration
- [ ] Merge with existing 5 LFQ studies
- [ ] Update unified dataset
- [ ] Regenerate cross-study visualizations

---

**Analysis Complete:** 2025-10-13
**Agent:** Claude Code (Sonnet 4.5)
**Next Steps:** Extract PDF tables, verify protein IDs, begin parsing
