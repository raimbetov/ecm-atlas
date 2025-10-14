# Sun et al. 2025 - Comprehensive Analysis

**Study ID:** 15_Sun_2025
**Status:** Hybrid ECM Study - Phase 3 (Non-LFQ Method)
**Last Updated:** 2025-10-13

---

## 1. Paper Overview

### Publication Details
- **Full Title:** Hybrid hydrogel–extracellular matrix scaffolds identify biochemical and mechanical signatures of cardiac ageing
- **Authors:** Avery Rui Sun, Jennifer L. Young, et al.
- **Journal:** Nature Materials
- **Year:** 2025
- **Publication Date:** June 12, 2025
- **DOI:** 10.1038/s41563-025-02234-6
- **PMID:** 40506498
- **PRIDE Accession:** PXD060864

### Biological Context
- **Species:** Mouse (Mus musculus)
- **Tissue:** Cardiac decellularized extracellular matrix (dECM)
- **Age Groups:** Young vs Aged mice (specific ages to be confirmed)
- **Experimental Design:** Hybrid hydrogel-dECM scaffolds

### Scientific Rationale
This study introduces **DECIPHER** (Decellularized ECM–synthetic hydrogel hybrid), a novel biomaterial platform that independently controls:
1. **ECM ligand presentation** (young vs aged cardiac ECM composition)
2. **Mechanical stiffness** (young vs aged tissue stiffness)

**Key Finding:** The ligand presentation of young ECM can override the profibrotic stiffness cues typically present in aged ECM, maintaining cardiac fibroblast quiescence. This demonstrates that **biochemical ECM composition is more influential than mechanical properties** in cardiac aging.

**Novelty:** First systematic dissection of age-dependent ECM biochemical vs mechanical contributions to cardiac fibroblast activation and senescence.

---

## 2. Method Classification & Quality Control

### Proteomic Method
- **Primary Method:** NOT Label-Free Quantification
- **Likely Method:** Isobaric labeling (TMT or iTRAQ) or other labeled method
- **Rationale:** User explicitly stated "не LFQ; изобарическое" (not LFQ; isobaric)
- **Purpose:** Characterization of decellularized ECM (dECM) composition from young vs aged hearts

### Experimental Approach
**Hybrid Scaffold System:**
1. **Decellularization:** Remove cells from young/aged cardiac tissue
2. **dECM Integration:** Incorporate dECM into synthetic hydrogel
3. **Mechanical Tuning:** Adjust hydrogel stiffness independently of ECM composition
4. **Proteomics:** Quantify ECM protein composition in young vs aged dECM

**Unique Features:**
- Orthogonal control of ligand (ECM composition) and mechanics (stiffness)
- In vitro cardiac fibroblast assays on hybrid scaffolds
- Not traditional bulk tissue proteomics

### LFQ Compatibility: NO ❌ - Labeled Method
- **Verdict:** **NOT compatible with Phase 1 LFQ pipeline**
- **Reason:**
  1. User confirmed "не LFQ" (not LFQ)
  2. Likely uses isobaric labeling (TMT/iTRAQ)
  3. dECM-focused (not bulk tissue) - different biological context
  4. Primary focus is biomaterial engineering, not aging atlas
- **Phase Assignment:** **Phase 3** (Non-LFQ Studies)

### Study Type Considerations
**Not a Traditional Aging Proteomics Study:**
- Focus: ECM ligand vs stiffness effects on fibroblasts
- dECM is a **biomaterial substrate**, not physiological tissue
- Proteomic data may be **descriptive** (characterizing dECM) rather than quantitative aging comparison
- Functional readouts (fibroblast activation, senescence) are primary outputs

**Potential Value for ECM-Atlas:**
- ✅ Identifies age-associated ECM proteins in cardiac tissue
- ✅ Validates which ECM components drive aging phenotypes
- ⚠️ dECM composition may differ from native ECM (decellularization artifacts)
- ⚠️ Proteomic data may be in supplementary tables, not main dataset

---

## 3. Age Bin Normalization Strategy

### Original Age Groups
**Status:** TO BE CONFIRMED from supplementary materials

**Expected Design (typical for cardiac aging studies):**
- Young: 2-4 months (young adult mice)
- Aged: 20-24 months (aged mice, near end of lifespan)

**Mouse Lifespan Context:**
- Mouse lifespan: ~24-30 months
- Young (2-4mo) ≈ Human ~20-30 years
- Aged (20-24mo) ≈ Human ~65-75 years

### Age Bin Normalization
**Status:** Likely Already Binary - No normalization required ✅

**Rationale:**
- Biomaterial studies typically use binary design (young vs aged ECM)
- Maximizes differences in ECM properties for functional testing
- No intermediate age groups expected

### Data Retention (Estimated)
- **Samples Retained:** Expected 100%
- **Groups:** 2 age bins (Young ECM, Aged ECM) × multiple biological replicates

### Final Age Mapping (Expected)
```
Young dECM (2-4mo) → Young
Aged dECM (20-24mo) → Old
```

**Note:** Exact ages and sample sizes need verification from Nature Materials supplementary materials.

---

## 4. Column Mapping to 13-Column Schema

### Source Files (To Be Downloaded)
- **Primary Data File:** Nature Materials Supplementary Tables + Source Data
- **Expected Files:**
  - Supplementary Data (Excel/CSV with dECM proteomics)
  - Source Data figures (if proteomics shown in figures)
- **Source URL:** https://www.nature.com/articles/s41563-025-02234-6
- **Location (after download):** `data_raw/Sun et al. - 2025/`

**Note:** Nature journals provide Source Data files (per-figure Excel files) and Supplementary Data (comprehensive datasets).

### Expected Column Mapping Table

| Schema Column | Expected Source | Data Type | Notes |
|---------------|-----------------|-----------|-------|
| **Protein_ID** | UniProt IDs from supplementary table | UniProt ID | Verify if full IDs or Gene Symbols |
| **Protein_Name** | Protein names column or UniProt API | String | Enrich via API if missing |
| **Gene_Symbol** | Gene names column | String | Mouse gene symbols |
| **Tissue** | Constant: "Cardiac - Decellularized ECM" | String | Note: dECM, not native tissue |
| **Species** | Constant: "Mus musculus" | String | |
| **Age** | Extract from sample metadata | Integer | In months (e.g., 3, 24) |
| **Age_Unit** | Constant: "months" | String | |
| **Abundance** | TMT/iTRAQ intensity or ratio | Float | Relative abundance |
| **Abundance_Unit** | Constant: "TMT intensity" or "iTRAQ ratio" | String | Method-dependent |
| **Method** | Constant: "TMT-labeled" or "Isobaric labeling" | String | Confirm from methods |
| **Study_ID** | Constant: "Sun_2025" or DOI | String | |
| **Sample_ID** | Format: `Sun2025_M{mouse_num}_A{age}mo_dECM` | String | Example: Sun2025_M1_A3mo_dECM |
| **Parsing_Notes** | "Decellularized ECM, hybrid scaffold study" | String | Flag as non-physiological |

### Unique Considerations
**Decellularized ECM Caveats:**
1. **Not Bulk Tissue:** dECM represents ECM-enriched fraction, not whole heart
2. **Decellularization Artifacts:** Harsh chemicals may remove some ECM components
3. **Biomaterial Context:** ECM is substrate for cell culture, not in vivo tissue

**Parsing Strategy:**
- **For ECM-Atlas:** Include with clear `Parsing_Notes` flag
- **Interpretation:** Use as "ECM-enriched aging signature" but note non-physiological context
- **Comparison:** May differ from Santinha 2024 (bulk cardiac tissue TMT)

### Known Data Gaps (To Be Resolved)
1. **Proteomic Method:** Confirm if TMT, iTRAQ, or other labeling
2. **Age Details:** Exact ages (young: ?mo, aged: ?mo)
3. **Sample Size:** Number of biological replicates per group
4. **Data Availability:** Check if full proteomics in supplementary tables or only selected proteins
5. **Decellularization Protocol:** Which method (SDS, Triton X-100, etc.) affects ECM retention

---

## 5. Parsing Implementation Guide

### Step 0: Download Supplementary Materials
```bash
# Nature Materials supplementary files URL
# https://www.nature.com/articles/s41563-025-02234-6#Sec20

# Download:
# 1. Supplementary Information PDF (methods details)
# 2. Supplementary Data files (Excel with proteomics)
# 3. Source Data files (figure-specific data)
```

### Step 1: Identify Proteomics Dataset
```python
import pandas as pd

# Check Supplementary Data structure
# Nature typically provides Excel with multiple sheets
xls = pd.ExcelFile("data_raw/Sun et al. - 2025/Supplementary_Data.xlsx")
print(xls.sheet_names)

# Expected: ['dECM_Proteomics_Young', 'dECM_Proteomics_Aged', 'Metadata']
# Or: Single sheet with "Age" column

# Load dECM proteomics
df_decm = pd.read_excel(xls, sheet_name='dECM_Proteomics')
```

### Step 2: Verify ECM Enrichment
```python
# Check if dataset is ECM-enriched (should have high proportion of matrisome proteins)
# Load matrisome gene list
matrisome_genes = set([...])  # From Matrisome DB

# Calculate enrichment
total_proteins = len(df_decm)
ecm_proteins = df_decm['Gene_Symbol'].isin(matrisome_genes).sum()
ecm_percentage = (ecm_proteins / total_proteins) * 100

print(f"ECM enrichment: {ecm_percentage:.1f}%")
# Expect: >50% for good decellularization
```

### Step 3: Extract Age Metadata
```python
# Sample columns might be: "Young_Rep1", "Young_Rep2", "Aged_Rep1", etc.
# Or metadata sheet with sample IDs mapped to ages

def parse_age_from_column(col_name):
    if 'Young' in col_name or 'Y_' in col_name:
        return 3  # Placeholder, update with actual age
    elif 'Aged' in col_name or 'Old' in col_name or 'A_' in col_name:
        return 24  # Placeholder
```

### Step 4: Generate Sample_IDs
```python
# Example format
sample_ids = []
for age_group in ['Young', 'Aged']:
    for replicate in range(1, N+1):  # N = number of replicates
        sample_id = f"Sun2025_M{replicate}_A{age_dict[age_group]}mo_dECM"
        sample_ids.append(sample_id)
```

### Expected Output
- **Row Count:** Estimated 200-400 ECM proteins × ~8-12 samples = 1,600-4,800 rows
- **Format:** Long format (one row per protein per sample)

### Data Quality Checks
1. Verify ECM enrichment >50% (matrisome proteins)
2. Check for collagen/laminin/fibronectin (should be abundant)
3. Low abundance of intracellular proteins (validates decellularization)
4. Compare young vs aged composition (expect more crosslinking proteins in aged)

### Ready for Parsing: NO ⚠️
**Blockers:**
1. Supplementary materials not yet downloaded
2. Proteomic method not confirmed (TMT? iTRAQ?)
3. Age metadata not confirmed
4. Unclear if full proteomics dataset or selected proteins only

**Next Steps:**
1. Download Nature Materials supplementary files
2. Inspect data structure (Supplementary Data + Source Data)
3. Extract age metadata from methods section
4. Verify PRIDE accession PXD060864 for raw data

---

## 6. Quality Assurance Notes

### Strengths
1. **Orthogonal Control:** Independently manipulates ECM composition and stiffness
2. **Mechanistic Insights:** Identifies which ECM property (ligand vs stiffness) drives aging phenotypes
3. **Functional Validation:** Links dECM proteomics to fibroblast activation/senescence
4. **High-Impact Journal:** Nature Materials ensures rigorous peer review
5. **Therapeutic Relevance:** Suggests young ECM could rejuvenate aged cardiac tissue

### Limitations
1. **Non-Physiological Context:** dECM is biomaterial, not native tissue
2. **Decellularization Artifacts:** May lose soluble ECM components (e.g., growth factors)
3. **In Vitro System:** Fibroblast assays on 2D/3D scaffolds ≠ in vivo cardiac aging
4. **Proteomic Data Scope:** May only characterize dECM composition, not comprehensive aging atlas
5. **Limited to Structural ECM:** May miss ECM-associated proteins (secreted factors, regulators)

### Biological Considerations
**dECM vs Native ECM:**
- **Sun 2025 (dECM):** ECM-enriched fraction, post-decellularization
- **Santinha 2024 (Native):** Bulk left ventricle tissue with cellular components removed enzymatically
- **Implication:** Both cardiac, but different preparation methods may yield different ECM profiles

**Key Biological Insight:**
- Young ECM ligands (e.g., specific collagens, laminins?) suppress fibroblast activation
- Aged ECM stiffness alone is insufficient to drive fibrosis
- Suggests **biochemical rejuvenation** may be more effective than mechanical softening

### Cross-Study Comparisons
- **vs Santinha 2024 (Cardiac LV):** Same organ, different method (bulk TMT vs dECM)
- **vs Schüler 2021 (MuSC niche):** Both use functional assays to validate ECM effects
- **vs Lofaro 2021 (Muscle dECM?):** If Lofaro also used ECM enrichment, could compare muscle vs heart

---

## 7. Integration into ECM-Atlas

### Phase Assignment
**Phase 3 (Non-LFQ Studies)** - 8th study overall, 3rd cardiac study

### Processing Priority
**Priority Level:** Medium
- **Rationale:**
  1. **Unique biomaterial approach** - different from traditional tissue proteomics
  2. **Mechanistic insights** - identifies functional ECM components
  3. **Complements Santinha 2024** - both cardiac, different methods
  4. **dECM caveat** - less representative of in vivo aging than bulk tissue
- **Recommended Order:** Process **after** Santinha 2024 (to compare bulk vs dECM)

### Processing Order Recommendation
```
Phase 1 LFQ studies (complete)
  ↓
Lofaro 2021 + Schüler 2021 (skeletal muscle pair)
  ↓
Santinha 2024 (bulk cardiac TMT) ← Process first
  ↓
Sun 2025 (cardiac dECM) ← Then compare to Santinha
```

### Expected Contributions
- Adds **cardiac dECM proteomics** to atlas
- Provides **biomaterial-derived ECM aging signature**
- Identifies **functional ECM components** (validated by fibroblast assays)
- Complements **Santinha 2024** for cardiac aging comparison

### Comparison Opportunity: Bulk vs dECM
```python
# After processing both Santinha and Sun datasets
bulk_cardiac = df[df['Study_ID'] == 'Santinha_2024']
decm_cardiac = df[df['Study_ID'] == 'Sun_2025']

# Compare ECM protein overlap
bulk_proteins = set(bulk_cardiac['Gene_Symbol'].unique())
decm_proteins = set(decm_cardiac['Gene_Symbol'].unique())

shared = bulk_proteins & decm_proteins
bulk_only = bulk_proteins - decm_proteins
decm_only = decm_proteins - bulk_proteins

# Interpretation:
# - Shared: Core cardiac ECM aging signature
# - Bulk_only: Cellular/intracellular contaminants OR soluble ECM lost in decellularization
# - dECM_only: Structural ECM retained post-decellularization
```

---

## 8. Data Availability

### Raw Data
- **PRIDE:** PXD060864
- **URL:** http://www.ebi.ac.uk/pride/archive/projects/PXD060864

### Processed Data (in this repository)
- **Supplementary Files (to be downloaded):**
  - `data_raw/Sun et al. - 2025/Supplementary_Information.pdf`
  - `data_raw/Sun et al. - 2025/Supplementary_Data.xlsx`
  - `data_raw/Sun et al. - 2025/Source_Data_Fig*.xlsx` (if proteomics in figures)
- **CSV (to be generated):** `data_processed/Sun_2025/Sun_2025_dECM_proteomics_raw.csv`
- **Standardized Output:** `data_processed/Sun_2025/Sun_2025_standardized.csv`

### Publication PDF
- **To be downloaded to:** `pdf/Sun_2025_NatureMaterials.pdf`

---

## 9. Parsing Roadmap

### Phase 1: Data Acquisition
- [ ] Download Nature Materials supplementary materials
- [ ] Download PDF for full methods section
- [ ] Verify PRIDE accession PXD060864 accessibility
- [ ] Identify which supplementary file contains proteomics

### Phase 2: Metadata Extraction
- [ ] Extract exact ages (young: ?mo, aged: ?mo) from methods
- [ ] Determine sample size (n=? per group)
- [ ] Identify proteomic method (TMT? iTRAQ?)
- [ ] Document decellularization protocol (SDS? Triton?)

### Phase 3: ECM Validation
- [ ] Calculate matrisome enrichment percentage
- [ ] Verify presence of structural ECM (collagens, laminins, fibronectin)
- [ ] Check for intracellular contaminants (should be minimal)
- [ ] Compare to Santinha 2024 cardiac ECM composition

### Phase 4: Protein Annotation
- [ ] Check if UniProt IDs provided or Gene Symbols only
- [ ] Map Gene Symbols → UniProt IDs if needed
- [ ] Fetch Protein_Name via UniProt API
- [ ] Classify proteins using Matrisome AnalyzeR

### Phase 5: Standardization
- [ ] Generate Sample_IDs for all replicates
- [ ] Reshape to long format (protein × sample)
- [ ] Add constant columns + Parsing_Notes flag ("dECM, non-physiological")
- [ ] Normalize using method-appropriate strategy (TMT ratio normalization)

### Phase 6: Quality Control
- [ ] PCA: young vs aged dECM separation
- [ ] Differential expression: age-associated ECM proteins
- [ ] Cross-reference with paper's functional validation (which proteins affect fibroblasts?)
- [ ] Compare to Santinha 2024: bulk vs dECM overlap

### Phase 7: Integration
- [ ] Merge into unified ECM-Atlas dataset (with dECM flag)
- [ ] Create cardiac-specific aging signature (Santinha + Sun)
- [ ] Highlight proteins validated in functional assays
- [ ] Document biomaterial context in metadata

---

## 10. Unique Analysis Opportunities

### Functional ECM Components
```python
# Identify ECM proteins that functionally affect fibroblasts
# (from paper's validation experiments)
functional_ecm = ['COL1A1', 'COL3A1', 'FN1', ...]  # Example

# Check if these show age-related changes in other tissues
cross_tissue_validation = unified_df[unified_df['Gene_Symbol'].isin(functional_ecm)]
# Tissues: Lung (Angelidis), Kidney (Randles), Muscle (Lofaro), etc.
```

### Bulk vs dECM Cardiac Comparison
```python
# After processing both Santinha and Sun
cardiac_comparison = {
    'Santinha_2024': {
        'Method': 'TMT bulk tissue',
        'ECM_enrichment': 'Enzymatic + TMT',
        'Context': 'Physiological'
    },
    'Sun_2025': {
        'Method': 'Isobaric dECM',
        'ECM_enrichment': 'Decellularization',
        'Context': 'Biomaterial'
    }
}

# Overlap analysis
# Q: Do both identify same cardiac aging markers (e.g., lactadherin, collagens)?
```

### Ligand vs Stiffness Aging Signature
```python
# Use paper's functional data to separate:
# 1. Proteins driving aging via ligand effects (composition)
# 2. Proteins driving aging via stiffness effects (crosslinking)

ligand_driven = [...]  # Proteins that change with young vs aged ECM
stiffness_driven = [...]  # Proteins that change with soft vs stiff hydrogel

# Check if these patterns generalize to other tissues
```

---

## 11. Notes for Phase 3 Processing

### Biomaterial Context Flagging
**Important:** When integrating Sun 2025 data into ECM-Atlas, add metadata flag:
```python
df_sun['Tissue_Type'] = 'Decellularized ECM (biomaterial)'
df_sun['Physiological'] = False  # Non-physiological context
```

This allows filtering:
- **In vivo aging studies:** Exclude Sun 2025
- **ECM composition studies:** Include Sun 2025

### Integration Decision
**Recommendation:**
- ✅ **Include in Phase 3** for comprehensive cardiac ECM characterization
- ⚠️ **Flag as non-physiological** in metadata
- ✅ **Use for functional validation** of age-associated ECM proteins identified in other studies
- ⚠️ **Do not use for cross-tissue aging signature** (biomaterial artifact risk)

---

**Analysis Complete:** 2025-10-13
**Agent:** Claude Code (Sonnet 4.5)
**Next Steps:**
1. Download Nature Materials supplementary materials
2. Verify proteomic method and data structure
3. Process after Santinha 2024 (cardiac bulk TMT)
4. Compare bulk vs dECM cardiac aging signatures
