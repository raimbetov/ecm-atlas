# Proteomic Dataset Structures - Quick Reference

## 1. Ariosa-Morejon et al. - 2021
**File:** `elife-66635-fig2-data1-v1.xlsx`
**Sheets:** Plasma (173), Cartilage (634), Bone (712), Skin (352)

| Element | Column Name | Notes |
|---------|-------------|-------|
| Protein ID | `Majority protein IDs` | |
| Gene | `Gene names` | |
| Protein Name | `Protein names` | May have leading space |
| Abundance | `iBAQ L A1-4`, `iBAQ H A1-4`, `Ratio H/L A1-4`, `Ratio H/L B1-4` | L=young, H=old |

**Age Encoding:** Heavy/Light isotope ratios (H=old, L=young)

---

## 2. Chmelova et al. - 2023
**File:** `Data Sheet 1.XLSX`
**Sheet:** protein.expression.imputed.new(
**Rows:** 17 × **Columns:** 3,830

| Element | Location | Notes |
|---------|----------|-------|
| Format | **TRANSPOSED** | Samples as rows, genes as columns |
| Sample Names | Column 0 | `X3m_ctrl_A`, `X18m_ctrl_B`, etc. |
| Genes | Column headers | `Ighv1.31`, `Igha`, `Ptprf`, etc. |
| Abundance | Cell values | Log2-transformed |

**Age Encoding:** 3m vs 18m in sample names (`X3m_*` vs `X18m_*`)
**Filter:** Use only `ctrl` samples (exclude MCAO stroke samples)

---

## 3. Li et al. - 2021 | dermis
**File:** `Table 2.xlsx`
**Sheet:** Table S2
**Rows:** 263 × **Columns:** 22
**Skip rows:** 3 (then rename cols 0-1)

| Element | Column | Notes |
|---------|--------|-------|
| Protein ID | Column 0 (`Unnamed: 0`) | Rename to "Protein ID" |
| Gene | Column 1 (`Unnamed: 1`) | Rename to "Gene symbol" |
| ECM Category | `Division`, `Category` | Columns 2-3 |
| Abundance | Columns 4-14 | `Toddler-Sample1-2`, `Teenager-Sample1-3`, `Adult-Sample1-3`, `Elderly-Sample1-5` |

**Age Encoding:** Age group in sample name prefix (Toddler/Teenager/Adult/Elderly)
**Values:** Log2-normalized FOT (Fraction of Total)
**Special:** Header row embedded in data; skip 3 rows and manually rename first 2 columns

---

## 4. Li et al. - 2021 | pancreas
**File:** `41467_2021_21261_MOESM6_ESM.xlsx`
**Sheet:** Data 3
**Rows:** 2,064 × **Columns:** 53
**Skip rows:** 2

| Element | Column | Notes |
|---------|--------|-------|
| Protein ID | `Accession` | |
| Protein Name | `Description` | |
| Abundance | Columns 2-25 (24 samples) | `F_7-12`, `J_8,22,46,59,67`, `Y_12,31,51,57,61,64`, `O_14,20,48,52,54,68` |

**Age Encoding:**
- **F** = Fetal
- **J** = Juvenile (number = weeks old)
- **Y** = Young adult (number = years old)
- **O** = Old (number = years old)

---

## 5. Lofaro et al. - 2021
**Status:** ❌ PDF only - **SKIP**

---

## 6. McCabe et al. - 2020
**Status:** ❌ DOCX only - **SKIP**

---

## 7. Ouni et al. - 2022
**Status:** ❌ Literature mining data, not proteomics - **SKIP**

---

## 8. Randles et al. - 2021
**File:** `ASN.2020101442-File027.xlsx`
**Sheet:** Human data matrix fraction
**Rows:** 2,610 × **Columns:** 31
**Skip rows:** 1

| Element | Column | Notes |
|---------|--------|-------|
| Protein ID | `Accession` | |
| Gene | `Gene name` | |
| Protein Name | `Description` | |
| Abundance | Columns 7-18 | `G15,29,37,61,67,69` and `T15,29,37,61,67,69` |

**Age Encoding:** Number in column name = patient age in years (15-69)
**Tissue Types:** G=Glomerular, T=Tubular (two kidney compartments)

---

## 9. Tam et al. - 2020
**File:** `elife-64940-supp1-v3.xlsx`
**Sheet:** Raw data
**Rows:** 3,157 × **Columns:** 80
**Skip rows:** 1

| Element | Column | Notes |
|---------|--------|-------|
| Protein ID | `T: Majority protein IDs` | Column 0 (remove "T: " prefix) |
| Protein Name | `T: Protein names` | Column 1 (remove "T: " prefix) |
| Gene | `T: Gene names` | Column 2 (remove "T: " prefix) |
| Abundance | Columns 3-68 (66 LFQ) | `LFQ intensity {disc}_{age} {region} {tissue}` |

**Column Pattern:** `LFQ intensity L3/4 old L OAF`
- **Disc level:** L3/4, L4/5, L5/S1
- **Age:** "old" vs "Young" (note capitalization)
- **Region:** L (left), A (anterior), P (posterior), R (right)
- **Tissue:** OAF (outer annulus fibrosus), IAF (inner), IAF/NP (mixed), NP (nucleus pulposus)

**Age Encoding:** "old" vs "Young" in column names

---

## 10. Tsumagari et al. - 2023
**File:** `41598_2023_45570_MOESM3_ESM.xlsx`
**Sheet:** expression
**Rows:** 6,821 × **Columns:** 32
**Skip rows:** 0

| Element | Column | Notes |
|---------|--------|-------|
| Protein ID | `UniProt accession` | Column 30 |
| Gene | `Gene name` | Column 31 |
| Abundance | Columns 0-17 (18 samples) | `Cx_3mo_1-6`, `Cx_15mo_1-6`, `Cx_24mo_1-6` |

**Column Pattern:** `Cx_3mo_1`
- **Tissue:** Cx (Cortex)
- **Age:** 3mo, 15mo, 24mo (months)
- **Replicate:** 1-6

**Age Encoding:** 3mo/15mo/24mo in column names

---

## Summary Statistics

**Total datasets:** 10
**Parseable:** 7 (70%)
**Skip:** 3 (30% - 2 wrong format, 1 non-proteomic)

**Total proteins across all datasets:** ~16,500 rows
**Age range covered:** Fetal → 69 years
**Tissues:** Plasma, Cartilage, Bone, Skin, Brain, Dermis, Pancreas, Kidney, Intervertebral disc

## Parsing Strategy Key

1. **Standard format:** Ariosa-Morejon, Randles, Tsumagari
2. **Header skip required:** Li dermis (3), Li pancreas (2), Randles (1), Tam (1)
3. **Transpose required:** Chmelova
4. **Complex column parsing:** Tam (hierarchical structure)
5. **Multi-sheet:** Ariosa-Morejon (4 tissues)
