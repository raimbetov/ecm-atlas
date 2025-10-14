# McCabe et al. 2020 - Comprehensive Analysis

**Study ID:** 16_McCabe_2020
**Status:** QconCAT Study - Phase 3 (Targeted Quantitative)
**Last Updated:** 2025-10-13

---

## 1. Paper Overview

### Publication Details
- **Full Title:** Alterations in extracellular matrix composition during aging and photoaging of the skin
- **Authors:** Maxwell C. McCabe, Ryan C. Hill, Kenneth Calderone, Yilei Cui, Yan Yan, Taihao Quan, Gary J. Fisher, Kirk C. Hansen
- **Journal:** Matrix Biology Plus
- **Year:** 2020
- **Publication Date:** June 17, 2020
- **DOI:** 10.1016/j.mbplus.2020.100041
- **PMID:** 33543036
- **PMC:** PMC7852213
- **PRIDE Accession:** PXD015982

### Biological Context
- **Species:** Human (Homo sapiens)
- **Tissue:** Skin dermis (two sites)
  - **Sun-protected:** Buttock (intrinsic aging only)
  - **Sun-exposed:** Forearm (intrinsic + photoaging)
- **Age Groups:** Young (mid-20s) and aged (80+ years)
- **Study Design:** Paired comparison - intrinsic vs photoaging

### Scientific Rationale
This study provides **absolute quantification** of ECM proteins in human skin during:
1. **Intrinsic aging** (chronological, sun-protected buttock)
2. **Photoaging** (extrinsic, UV-exposed forearm)

Using **QconCAT** (Quantification conCATenation) with stable isotope-labeled (SIL) concatemeric peptide standards, the study achieves:
- **Absolute protein quantification** (not relative abundance)
- **Direct comparison** of intrinsic vs extrinsic aging mechanisms
- **Early detection** of photoaging signatures in young individuals (mid-20s)

**Key Finding:** Photoaging signatures are detectable decades before clinical signs appear, with distinct ECM remodeling patterns compared to intrinsic aging.

---

## 2. Method Classification & Quality Control

### Proteomic Method
- **Primary Method:** **QconCAT (Quantification conCATenation)** - **Targeted Quantitative Proteomics**
- **Quantification Strategy:** Stable Isotope Labeled (SIL) concatemeric peptide standards
- **Instrumentation:** LC-MS/MS with targeted acquisition
- **Output:** **Absolute protein abundance** (mol/mol or fmol/μg)

### QconCAT Method Details
**What is QconCAT?**
- **Artificial protein** containing concatenated tryptic peptides from target ECM proteins
- **Heavy isotope labeling** (^13C, ^15N) for mass shift
- **Spiked into samples** as internal standard
- **Absolute quantification** via:
  - Ratio of light (endogenous) to heavy (standard) peptide intensities
  - Converts to absolute molar amounts

**Advantages:**
- ✅ True absolute quantification (not relative like LFQ)
- ✅ High precision and reproducibility
- ✅ Focused on pre-selected ECM proteins (targeted approach)
- ✅ Less affected by matrix effects

**Limitations:**
- ⚠️ Limited proteome coverage (only pre-selected proteins)
- ⚠️ Requires prior knowledge of target proteins
- ⚠️ More complex sample preparation vs LFQ

### LFQ Compatibility: NO ❌ - Targeted Quantitative Method
- **Verdict:** **NOT compatible with Phase 1 LFQ pipeline**
- **Reason:**
  1. QconCAT is a **targeted method** (not discovery proteomics)
  2. Uses **stable isotope labeling** (not label-free)
  3. Provides **absolute quantification** (different scale than LFQ intensities)
  4. Limited to **pre-selected ECM proteins** (not全面proteome)
  5. Requires different normalization strategy
- **Phase Assignment:** **Phase 3** (Non-LFQ, Targeted Methods)

### Methodological Quality
**Strengths:**
- ✅ Absolute quantification (gold standard for ECM abundance)
- ✅ Paired comparison (buttock vs forearm, same individual)
- ✅ Focused on clinically relevant ECM proteins (collagens, proteoglycans)
- ✅ Early photoaging detection (mid-20s)
- ✅ Human tissue (directly translatable)

**Limitations:**
- ⚠️ Narrow proteome coverage (~20-40 targeted ECM proteins vs 200-400 in LFQ)
- ⚠️ No discovery of novel aging-associated proteins
- ⚠️ Small sample size (typical for human skin biopsies)
- ⚠️ Buttock biopsies may be difficult to obtain (ethical/practical)

---

## 3. Age Bin Normalization Strategy

### Original Age Groups
**Status:** Binary Design - Young vs Aged

| Age Bin | Age Range | Sample Site | Context |
|---------|-----------|-------------|---------|
| Young | Mid-20s (~25yr) | Buttock (sun-protected) | Intrinsic aging baseline |
| Young | Mid-20s (~25yr) | Forearm (sun-exposed) | Early photoaging |
| Aged | 80+ years | Buttock (sun-protected) | Intrinsic aging |
| Aged | 80+ years | Forearm (sun-exposed) | Intrinsic + photoaging |

**Human Lifespan Context:**
- Young (25yr) ≈ 30% of typical lifespan (~75-85yr)
- Aged (80+yr) ≈ 95-106% of typical lifespan

### Age Bin Normalization
**Status:** Already Binary - No normalization required ✅

**Unique Design:**
- Not traditional young vs old comparison
- **Paired within-individual:** Buttock vs Forearm (same person)
- **Aging Type Separation:**
  - **Intrinsic Aging:** Buttock_Young → Buttock_Aged
  - **Photoaging:** Forearm_Young → Forearm_Aged
  - **Photoaging Effect:** Buttock → Forearm (within age group)

### Data Retention
- **Samples Retained:** 100% (all young and aged samples)
- **Groups:** 2 age bins × 2 tissue sites = 4 conditions

### Final Age Mapping
```
Young_Buttock (25yr, sun-protected) → Young_Intrinsic
Young_Forearm (25yr, sun-exposed) → Young_Photoaged
Aged_Buttock (80+yr, sun-protected) → Old_Intrinsic
Aged_Forearm (80+yr, sun-exposed) → Old_Photoaged
```

**Special Consideration:**
- Add **`Aging_Type`** column: "Intrinsic" vs "Photoaging"
- Add **`Tissue_Site`** column: "Buttock" vs "Forearm"

---

## 4. Column Mapping to 13-Column Schema

### Source Files
- **Primary Data File:** Supplementary DOCX file
  - Located: `data_raw/McCabe et al. - 2020/1-s2.0-S2590028520300223-mmc1.docx`
- **Expected Content:** QconCAT absolute quantification data (fmol/μg or mol/mol ratios)
- **Format:** Word document tables (requires extraction)

### Expected Column Mapping Table

| Schema Column | Source/Value | Data Type | Notes |
|---------------|--------------|-----------|-------|
| **Protein_ID** | Extract from supplementary table or map Gene Symbol → UniProt | UniProt ID | May need UniProt API mapping |
| **Protein_Name** | From table or UniProt API | String | Enrich if missing |
| **Gene_Symbol** | From QconCAT target list | String | Human gene symbols (e.g., COL1A1, VCAN) |
| **Tissue** | Conditional: "Skin - Dermis (Buttock)" or "Skin - Dermis (Forearm)" | String | Distinguish sun-protected vs exposed |
| **Species** | Constant: "Homo sapiens" | String | |
| **Age** | Conditional: 25 (young) or 80 (aged) | Integer | Approximate, verify from methods |
| **Age_Unit** | Constant: "years" | String | |
| **Abundance** | QconCAT absolute abundance | Float | In fmol/μg or mol/mol |
| **Abundance_Unit** | Constant: "fmol/μg" or "mol/mol" | String | Verify from methods |
| **Method** | Constant: "QconCAT - Targeted MS" | String | |
| **Study_ID** | Constant: "McCabe_2020" or DOI | String | |
| **Sample_ID** | Format: `McCabe2020_S{subject}_A{age}yr_{site}` | String | Example: McCabe2020_S1_A25yr_Buttock |
| **Parsing_Notes** | "QconCAT absolute quantification; Intrinsic vs photoaging" | String | Include aging type |

### Additional Columns (Unique to this study)
| Custom Column | Value | Purpose |
|---------------|-------|---------|
| **Aging_Type** | "Intrinsic" or "Photoaging" | Distinguish aging mechanisms |
| **Tissue_Site** | "Buttock" or "Forearm" | Anatomical location |
| **UV_Exposure** | "Protected" or "Exposed" | Sun exposure status |

### Known Data Gaps (To Be Resolved)
1. **DOCX Extraction:** Tables in Word format (need conversion to CSV)
2. **Sample Size:** Number of individuals per age group (n=?)
3. **Exact Ages:** Confirm if "mid-20s" = 25yr, "80+" = 80yr or 85yr
4. **Target Protein List:** Which ECM proteins were in QconCAT panel?
5. **Abundance Units:** Confirm if fmol/μg or mol/mol (or both)

---

## 5. Parsing Implementation Guide

### Step 0: Extract Data from DOCX
```bash
# Option A: Convert DOCX to plain text or CSV using pandoc
pandoc "data_raw/McCabe et al. - 2020/1-s2.0-S2590028520300223-mmc1.docx" \
  -o "data_raw/McCabe et al. - 2020/supplementary_tables.csv"

# Option B: Open in Word/LibreOffice, copy tables, save as CSV
# Manual extraction may be more reliable for complex tables
```

### Step 1: Identify QconCAT Target Proteins
```python
import pandas as pd

# Load extracted supplementary data
df = pd.read_csv("data_raw/McCabe et al. - 2020/supplementary_tables.csv")

# Expected proteins (based on abstract):
# - Collagens (COL1A1, COL3A1, COL5A1, etc.)
# - Proteoglycans (VCAN, DCN, LUM, BGN)
# - Elastic fiber proteins (ELN, FBN1, FBLN5)
# - Crosslinking enzymes (LOX, LOXL1)
# - Proteases (MMPs, ADAMTSs)

print(df['Gene_Symbol'].unique())
```

### Step 2: Separate by Aging Type and Site
```python
# Create Aging_Type and Tissue_Site columns
df['Aging_Type'] = df['Tissue_Site'].apply(
    lambda x: 'Intrinsic' if x == 'Buttock' else 'Photoaging'
)

df['UV_Exposure'] = df['Tissue_Site'].apply(
    lambda x: 'Protected' if x == 'Buttock' else 'Exposed'
)
```

### Step 3: Map Gene Symbols to UniProt IDs
```python
from bioservices import UniProt
u = UniProt(verbose=False)

def map_gene_to_uniprot(gene_symbol, organism='Homo sapiens'):
    query = f'gene:{gene_symbol} AND organism:"{organism}" AND reviewed:true'
    result = u.search(query, columns='id,protein_names,genes')
    # Parse and return UniProt ID + Protein_Name
```

### Step 4: Generate Sample_IDs
```python
# Format: McCabe2020_S{subject}_A{age}yr_{site}
sample_ids = []
for subject_num in range(1, N+1):  # N = number of subjects
    for age_group in ['25yr', '80yr']:
        for site in ['Buttock', 'Forearm']:
            sample_id = f"McCabe2020_S{subject_num}_A{age_group}_{site}"
            sample_ids.append(sample_id)
# Total: N subjects × 2 ages × 2 sites = 4N samples
```

### Expected Output
- **Row Count:** Estimated 20-40 target ECM proteins × 4N samples = 80-160N rows (if N=5-10, ~400-1600 rows)
- **Format:** Long format (one row per protein per sample)

### Data Quality Checks
1. Verify QconCAT target proteins are ECM-relevant (collagens, proteoglycans, etc.)
2. Check abundance units (should be fmol/μg or mol/mol)
3. Confirm paired samples (same individual has all 4 conditions)
4. Validate intrinsic aging: Buttock_Aged vs Buttock_Young
5. Validate photoaging: Forearm vs Buttock (within age group)

### Ready for Parsing: PARTIAL ⚠️
**Blockers:**
1. DOCX file requires extraction to CSV
2. Sample size (n=?) not confirmed
3. Exact ages (25yr? 80yr? 85yr?) need verification
4. QconCAT target protein list not documented

**Next Steps:**
1. Extract tables from supplementary DOCX
2. Read full methods section to confirm ages and sample sizes
3. Identify QconCAT target protein panel
4. Begin parsing after extraction

---

## 6. Quality Assurance Notes

### Strengths
1. **Absolute Quantification:** Gold standard for ECM protein abundance (vs relative LFQ)
2. **Paired Design:** Buttock vs Forearm from same individual (controls for inter-individual variation)
3. **Clinical Relevance:** Human tissue, directly translatable
4. **Aging Type Separation:** Intrinsic vs photoaging mechanisms distinguished
5. **Early Detection:** Photoaging signatures in mid-20s (preventive medicine implications)
6. **High-Impact Proteins:** Collagens, proteoglycans, elastic fibers (key ECM components)

### Limitations
1. **Narrow Proteome Coverage:** ~20-40 targeted proteins (vs 200-400 in LFQ discovery)
2. **No Novel Discovery:** Pre-selected proteins, cannot identify new aging markers
3. **Small Sample Size:** Human skin biopsies are invasive, limiting N
4. **Single Time Point:** Only 2 age groups (young, aged) - no middle age
5. **Site-Specific:** Results may not generalize to other skin sites (face, hands)

### Biological Considerations
**Intrinsic Aging Signatures (Buttock):**
- ↓ Collagens (COL1A1, COL3A1) - structural ECM loss
- ↓ Proteoglycans (VCAN, DCN) - ECM hydration loss
- ↓ Elastic fibers (ELN, FBLN5) - skin elasticity decline
- ↓ Crosslinking enzymes (LOX) - ECM stabilization impaired

**Photoaging Signatures (Forearm vs Buttock):**
- ↑ Elastic fiber-associated proteins (even in young forearm)
- ↑ Pro-inflammatory proteases (MMPs, ADAMTSs)
- ↓ Structural collagens (accelerated vs intrinsic)
- ↑ Collagen fragmentation (UV-induced)

**Clinical Insight:**
- Forearm in mid-20s already shows photoaging signatures
- Decades before wrinkles/clinical signs appear
- Suggests early sunscreen use is critical

### Cross-Study Comparisons
- **vs LiDermis 2021 (Human Dermis, LFQ):** Same tissue, different method (discovery LFQ vs targeted QconCAT)
- **vs Santinha 2024 (Cardiac TMT):** Both human aging, different tissues
- **vs Sun 2025 (dECM):** McCabe uses native tissue (not decellularized)

**Complementary with LiDermis 2021:**
- LiDermis: Discovery proteomics, broader coverage
- McCabe: Targeted quantification, absolute abundance
- **Combined:** Discovery (LiDermis) → Validation (McCabe)

---

## 7. Integration into ECM-Atlas

### Phase Assignment
**Phase 3 (Targeted Methods)** - 9th non-LFQ study

### Processing Priority
**Priority Level:** Medium
- **Rationale:**
  1. **Unique method:** QconCAT provides absolute quantification (vs relative LFQ/TMT)
  2. **Human skin aging:** Complements LiDermis 2021 (LFQ dermis aging)
  3. **Photoaging model:** Unique intrinsic vs extrinsic aging comparison
  4. **Limited coverage:** Fewer proteins than LFQ studies
- **Recommended Order:** Process **after** LiDermis 2021 for skin aging comparison

### Processing Order Recommendation
```
Phase 1 LFQ studies (complete)
  ↓
Phase 1 LFQ remaining (Lofaro, Schüler)
  ↓
Phase 3 TMT studies (Santinha, Sun, Ouni, Tsumagari, LiPancreas)
  ↓
Phase 3 isotope (Ariosa SILAC, Caldeira iTRAQ)
  ↓
Phase 3 targeted (McCabe QconCAT) ← Process last for skin validation
```

### Expected Contributions
- Adds **absolute ECM quantification** to atlas (only QconCAT study)
- Provides **intrinsic vs photoaging** comparison (unique design)
- Complements **LiDermis 2021** for human skin aging
- Validates **collagen/proteoglycan decline** seen in LFQ studies
- Enables **cross-method comparison** (QconCAT vs LFQ vs TMT)

---

## 8. Data Availability

### Raw Data
- **PRIDE:** PXD015982
- **URL:** http://www.ebi.ac.uk/pride/archive/projects/PXD015982

### Processed Data (in this repository)
- **Supplementary File:** `data_raw/McCabe et al. - 2020/1-s2.0-S2590028520300223-mmc1.docx`
- **CSV (to be generated):** `data_processed/McCabe_2020/McCabe_2020_QconCAT_raw.csv`
- **Standardized Output:** `data_processed/McCabe_2020/McCabe_2020_standardized.csv`

### Publication PDF
- **To be downloaded to:** `pdf/McCabe_2020_MatrixBiolPlus.pdf`
- **PMC URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7852213/

---

## 9. Parsing Roadmap

### Phase 1: Data Extraction
- [ ] Extract tables from supplementary DOCX file
- [ ] Convert to CSV format
- [ ] Verify column headers (Gene_Symbol, Buttock_Young, Buttock_Aged, Forearm_Young, Forearm_Aged)

### Phase 2: Metadata Extraction
- [ ] Read full methods section for exact ages
- [ ] Determine sample size (n=? per age group)
- [ ] Identify QconCAT target protein list
- [ ] Confirm abundance units (fmol/μg or mol/mol)

### Phase 3: Protein Annotation
- [ ] Map Gene Symbols → UniProt IDs
- [ ] Fetch Protein_Name via UniProt API
- [ ] Classify proteins (collagens, proteoglycans, elastic fibers, enzymes, proteases)

### Phase 4: Standardization
- [ ] Reshape from wide to long format (protein × sample)
- [ ] Add Aging_Type, Tissue_Site, UV_Exposure columns
- [ ] Generate Sample_IDs
- [ ] Add constant columns (Species, Method, Study_ID, etc.)

### Phase 5: Normalization (Special for QconCAT)
- [ ] **Do NOT z-score normalize** (absolute values are meaningful!)
- [ ] Optional: Convert units if needed (fmol/μg ↔ mol/mol)
- [ ] Calculate fold-changes: Aged/Young, Forearm/Buttock

### Phase 6: Quality Control
- [ ] Verify paired samples (same individual)
- [ ] Check intrinsic aging: Buttock_Aged vs Buttock_Young
- [ ] Check photoaging: Forearm vs Buttock
- [ ] Validate early photoaging: Young_Forearm vs Young_Buttock

### Phase 7: Integration
- [ ] Compare with LiDermis 2021 (LFQ dermis aging)
- [ ] Cross-validate collagen/proteoglycan changes
- [ ] Highlight proteins unique to QconCAT quantification
- [ ] Add to unified ECM-Atlas with method flag

---

## 10. Unique Analysis Opportunities

### Absolute vs Relative Quantification Comparison
```python
# After processing both McCabe (QconCAT) and LiDermis (LFQ)
qconcat_data = df[df['Study_ID'] == 'McCabe_2020']
lfq_data = df[df['Study_ID'] == 'LiDermis_2021']

# Overlapping proteins
shared_proteins = set(qconcat_data['Gene_Symbol']) & set(lfq_data['Gene_Symbol'])

# Compare fold-changes (Aged/Young)
# - QconCAT: Absolute abundance ratios
# - LFQ: Z-score differences or intensity ratios
```

### Intrinsic vs Photoaging Signatures
```python
# Intrinsic aging (Buttock)
intrinsic_aging = qconcat_data[qconcat_data['Tissue_Site'] == 'Buttock']
intrinsic_fc = intrinsic_aging.groupby('Gene_Symbol')['Abundance'].apply(
    lambda x: x[aged] / x[young]
)

# Photoaging (Forearm vs Buttock)
photoaging_effect = qconcat_data.groupby('Gene_Symbol').apply(
    lambda x: x[x['Tissue_Site']=='Forearm']['Abundance'].mean() /
              x[x['Tissue_Site']=='Buttock']['Abundance'].mean()
)

# Venn diagram: Intrinsic-specific, Photoaging-specific, Shared
```

### Early Photoaging Biomarkers
```python
# Identify proteins elevated in Young_Forearm vs Young_Buttock
young_photoaging = qconcat_data[qconcat_data['Age'] == 25]
early_markers = young_photoaging[
    young_photoaging['Tissue_Site'] == 'Forearm'
]['Abundance'] / young_photoaging[
    young_photoaging['Tissue_Site'] == 'Buttock'
]['Abundance']

# Proteins with FC > 1.5 = early photoaging signatures
early_biomarkers = early_markers[early_markers > 1.5].index.tolist()
```

### Cross-Method Validation
```python
# Validate aging-associated proteins across methods
qconcat_aging = qconcat_data.groupby('Gene_Symbol')['Abundance'].apply(
    lambda x: aged/young
)
lfq_aging = lfq_data.groupby('Gene_Symbol')['Zscore_Change'].mean()

# Correlation plot: QconCAT fold-change vs LFQ z-score change
# Expect: Strong correlation for shared proteins
```

---

## 11. Notes for Phase 3 Processing

### Unique Considerations for QconCAT Data
1. **Do NOT normalize within study:**
   - Absolute quantification is the strength of QconCAT
   - Z-scores would destroy absolute scale
   - Keep original fmol/μg or mol/mol values

2. **Cross-study comparison challenges:**
   - Cannot directly compare with LFQ (relative) or TMT (relative within batch)
   - Use fold-changes (Aged/Young) for cross-study validation
   - Qualitative agreement (direction of change) more important than magnitude

3. **Aging Type Metadata:**
   - Add custom columns: Aging_Type, Tissue_Site, UV_Exposure
   - Enable filtering: "Show only intrinsic aging" or "Show only photoaging"

4. **Integration Decision:**
   - ✅ **Include in Phase 3** for comprehensive ECM aging atlas
   - ⚠️ **Flag as absolute quantification** in metadata
   - ✅ **Use for validation** of LFQ/TMT findings
   - ⚠️ **Do not merge abundance scales** with LFQ/TMT

---

**Analysis Complete:** 2025-10-13
**Agent:** Claude Code (Sonnet 4.5)
**Next Steps:**
1. Extract supplementary DOCX tables to CSV
2. Read full methods for ages and sample sizes
3. Identify QconCAT target protein panel
4. Begin parsing after LiDermis 2021 processing (for skin aging comparison)
