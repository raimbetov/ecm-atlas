# Dataset Structure Analysis for Parser Configuration

## 1. Ariosa-Morejon et al. - 2021

**File:** `elife-66635-fig2-data1-v1.xlsx` (1.05 MB)

**Sheets:** Plasma (173 rows), Cartilage (634 rows), Bone (712 rows), Skin (352 rows), % Stable&Fast turnover proteins

**Structure:**
- **Protein IDs:** `Majority protein IDs` column
- **Gene symbols:** `Gene names` column
- **Protein names:** `Protein names` column (note: may have leading space)
- **Abundance columns:** iBAQ ratios (H/L) and raw iBAQ values
  - Pattern: `Ratio H/L A1`, `Ratio H/L A2`, etc. (Light isotope = young)
  - Pattern: `iBAQ L A1`, `iBAQ H A1`, etc.
  - Contains: `Mean Ratio H/L A`, `Mean Ratio H/L B`, `Fold change`, `Log fold change`
- **Age representation:**
  - Group A vs Group B (encoded in column names)
  - Heavy/Light isotope labeling (H = old, L = young based on paper methods)
  - 21-30 columns per tissue

**Parser Strategy:** Multi-sheet reader, extract H/L ratios as age comparison, use iBAQ for abundance

---

## 2. Chmelova et al. - 2023

**File:** `Data Sheet 1.XLSX` (758 KB)

**Sheets:** protein.expression.imputed.new(

**Structure:**
- **Total:** 17 rows × 3,830 columns
- **Rows are samples, Columns are gene symbols** (TRANSPOSED format!)
- **Sample names in column 0:**
  - `X3m_ctrl_A`, `X3m_ctrl_B`, `X3m_ctrl_C` (3 month controls)
  - `X3m_MCAO_3d_A`, `X3m_MCAO_3d_B`, `X3m_MCAO_3d_C` (3 month, stroke, 3 days)
  - `X3m_MCAO_7d_A`, `X3m_MCAO_7d_B`, `X3m_MCAO_7d_C` (3 month, stroke, 7 days)
  - `X18m_ctrl_A`, `X18m_ctrl_B` (18 month controls)
  - `X18m_MCAO_3d_A`, `X18m_MCAO_3d_B`, `X18m_MCAO_3d_D` (18 month, stroke, 3 days)
  - `X18m_MCAO_7d_A`, `X18m_MCAO_7d_B`, `X18m_MCAO_7d_C` (18 month, stroke, 7 days)
- **Gene symbols:** Column headers (Ighv1.31, Igha, Ptprf, etc.)
- **Values:** Log2-transformed protein abundance
- **Age groups:** 3m vs 18m encoded in sample names

**Parser Strategy:** TRANSPOSE data! Extract age from sample names (3m vs 18m), filter for ctrl samples only

---

## 3. Li et al. - 2021 | dermis

**File:** `Table 2.xlsx` (71 KB)

**Sheet:** Table S2

**Structure:**
- **Skip rows:** 2 (title and description rows)
- **Total:** 264 rows × 22 columns
- **Protein IDs:** `Protein ID` (column 0)
- **Gene symbols:** `Gene symbol` (column 1)
- **ECM classification:** `ECM components` (column 2), with subcategory in column 3
- **Abundance columns:**
  - Header row is at skiprows=3: `Toddler-Sample1`, `Toddler-Sample2`, `Teenager-Sample1`, `Teenager-Sample2`, `Teenager-Sample3`, `Adult-Sample1`, `Adult-Sample2`, `Adult-Sample3`, `Elderly-Sample1`, `Elderly-Sample2`, `Elderly-Sample3`, `Elderly-Sample4`, `Elderly-Sample5`
  - Columns 4-18 (indices after skiprows=2, but first data row contains actual headers)
  - Values are log2-normalized FOT (Fraction of Total)
- **Age groups:** Toddler, Teenager, Adult, Elderly (in sample column names)
- **Statistics:** `Highly expressed age stage` (col 19), `Expression trend` (col 20), `P Value` (col 21)

**Parser Strategy:** Skip 3 rows to get headers, parse age groups from column names

---

## 4. Li et al. - 2021 | pancreas

**File:** `41467_2021_21261_MOESM6_ESM.xlsx` (1.11 MB)

**Sheet:** Data 3

**Structure:**
- **Skip rows:** 2 (title rows)
- **Total:** 2,064 rows × 53 columns
- **Protein IDs:** `Accession` (column 0)
- **Protein names:** `Description` (column 1)
- **Abundance columns:**
  - Sample codes: `F_7`, `F_8`, `F_9`, `F_10`, `F_11`, `F_12` (Fetal)
  - `J_8`, `J_22`, `J_46`, `J_59`, `J_67` (Juvenile - weeks old)
  - `Y_12`, `Y_31`, `Y_51`, `Y_57`, `Y_61`, `Y_64` (Young adult - years old)
  - `O_14`, `O_20`, `O_48`, `O_52`, `O_54`, `O_68` (Old - years old)
  - 24 abundance columns total (columns 2-25)
- **Metadata:** `Coverage`, `# Peptides`, `# PSMs`, `# Unique Peptides` (columns 26-29)
- **Age groups:** F=Fetal, J=Juvenile (weeks), Y=Young (years), O=Old (years)

**Parser Strategy:** Skip 2 rows, extract age group from column prefix (F/J/Y/O), numbers indicate age value

---

## 5. Lofaro et al. - 2021

**Files:** Table S1.pdf (707 KB), Table S2.pdf (56 KB)

**Status:** NO EXCEL FILES - Only PDFs

**Parser Strategy:** SKIP or require manual PDF extraction

---

## 6. McCabe et al. - 2020

**Files:** 1-s2.0-S2590028520300223-mmc1.docx (30 KB)

**Status:** NO EXCEL FILES - Only DOCX

**Parser Strategy:** SKIP or require manual extraction

---

## 7. Ouni et al. - 2022

**File:** `Supp Table 2.xlsx` (1.74 MB)

**Sheets:** 28 sheets (Hippo_TGFBI, Hippo_S100A6, etc.) - each sheet is pathway/protein specific

**Structure:**
- **NOT PROTEOMIC DATA** - This is literature mining/pathway data
- Contains: Relation Name, Authors, CellLineName, Journal, DOI, etc.
- Each sheet represents protein-pathway relationships from literature

**Parser Strategy:** SKIP - Not a proteomics dataset, this is metadata/literature analysis

---

## 8. Randles et al. - 2021

**File:** `ASN.2020101442-File027.xlsx` (1.35 MB)

**Sheets:** Summary of datasets, Col4a1 matrix fraction, Col4a3 early matrix fraction, Col4a3 late matrix fraction, Col4a5 early matrix fraction, Col4a5 late matrix fraction, Human data matrix fraction

**Main Sheet:** `Human data matrix fraction`

**Structure:**
- **Skip rows:** 1 (header row has "Hi-N normalised abundance" merged header)
- **Total rows:** 2,610
- **Protein IDs:** `Accession` (column 1)
- **Gene symbols:** `Gene name` (column 0)
- **Protein names:** `Description` (column 6)
- **Abundance columns:**
  - Glomerular (G) samples: `G15`, `G29`, `G37`, `G61`, `G67`, `G69` (ages)
  - Tubular (T) samples: `T15`, `T29`, `T37`, `T61`, `T67`, `T69` (ages)
  - 12 abundance columns total (columns 7-18)
  - Additional columns with .1 suffix appear to be presence/absence flags
- **Metadata:** `Peptide count`, `Unique peptides`, `Confidence score`, `P value` (columns 2-5)
- **Age groups:** Numbers indicate patient age (15, 29, 37, 61, 67, 69 years)
- **Tissue types:** G=Glomerular, T=Tubular (two compartments)

**Parser Strategy:** Skip 1 row, extract age from column names (G/T prefix + number), handle two tissue types

---

## 9. Tam et al. - 2020

**File:** `elife-64940-supp1-v3.xlsx` (1.42 MB)

**Sheets:** Raw data, Sample information

**Main Sheet:** `Raw data`

**Structure:**
- **Skip rows:** 1 (first row contains metadata labels like "T: Majority protein IDs")
- **Total rows:** 3,157
- **Total columns:** 80
- **Protein IDs:** Column 0 = `T: Majority protein IDs` → becomes `Majority protein IDs` after skip
- **Protein names:** Column 1 = `T: Protein names` → becomes `Protein names`
- **Gene symbols:** Column 2 = `T: Gene names` → becomes `Gene names`
- **Abundance columns (LFQ intensities):**
  - Pattern: `LFQ intensity L3/4 old L OAF`, `LFQ intensity L3/4 old A OAF`, etc.
  - Disc levels: L3/4, L4/5, L5/S1
  - Age groups: old, Young (note capitalization)
  - Disc regions: L (left), A (anterior), P (posterior), R (right)
  - Tissue types: OAF (outer annulus fibrosus), IAF (inner annulus fibrosus), IAF/NP, NP (nucleus pulposus)
  - 66 LFQ columns total (columns 3-68)
- **Metadata:** `N: Peptides`, `N: Razor + unique peptides`, etc. (columns 69-79)
- **Age groups:** "old" vs "Young" in column names

**Parser Strategy:** Skip 1 row, parse complex column structure (disc level + age + region + tissue type)

---

## 10. Tsumagari et al. - 2023

**File:** `41598_2023_45570_MOESM3_ESM.xlsx` (3.19 MB)

**Sheets:** expression, Welch's test

**Main Sheet:** `expression`

**Structure:**
- **Skip rows:** 0
- **Total rows:** 6,821
- **Total columns:** 32
- **Row index contains protein info**
- **Abundance columns:**
  - Pattern: `Cx_3mo_1`, `Cx_15mo_1`, `Cx_24mo_1`, etc.
  - Tissue: Cx (Cortex)
  - Age groups: 3mo, 15mo, 24mo (months)
  - Replicates: 1-6
  - 18 sample columns total (columns 0-17)
- **Metadata columns:**
  - `Peptides`, `Razor + unique peptides`, `Unique peptides`
  - `Sequence coverage [%]`, `Unique + razor sequence coverage [%]`, `Unique sequence coverage [%]`
  - `Mol. weight [kDa]`, `Q-value`, `Score`, `Intensity`, `MS/MS count`
  - `id`, `UniProt accession`, `Gene name`
  - (columns 18-31)
- **Protein IDs:** `UniProt accession` (column 30)
- **Gene symbols:** `Gene name` (column 31)
- **Age groups:** 3mo, 15mo, 24mo encoded in column names

**Parser Strategy:** No skip, extract age from column pattern (3mo/15mo/24mo), protein info in metadata columns

---

## Summary by Parsing Complexity

### Simple/Standard Format:
1. **Ariosa-Morejon et al.** - Multi-sheet, standard columns, ratio-based
2. **Randles et al.** - Skip 1 row, age in column numbers, dual tissue type
3. **Tsumagari et al.** - Standard format, age in column names, metadata at end

### Moderate Complexity:
4. **Li et al. dermis** - Skip 3 rows for headers, age groups in sample names
5. **Li et al. pancreas** - Skip 2 rows, age group prefixes (F/J/Y/O)

### Complex Format:
6. **Tam et al.** - Multi-level column structure (disc × age × region × tissue)
7. **Chmelova et al.** - TRANSPOSED (samples as rows, genes as columns)

### Skip/Manual:
8. **Lofaro et al.** - PDF only
9. **McCabe et al.** - DOCX only
10. **Ouni et al.** - Not proteomics data (literature mining)
