# Randles 2021 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: Progenesis Hi-N (top-3 peptide) label-free quantification
- LFQ compatible: YES - Label-free LC-MS/MS with Hi-N normalization (Methods p.5)
- Included in Phase 1 parsing: YES

## 2. Current Age Groups
- Young donors: 15, 29, 37 years (3 donors)
- Aged donors: 61, 67, 69 years (3 donors)
- **Total:** 6 human kidney donors
- Tissue compartments: Glomerular (G) and Tubulointerstitial (T) fractions per donor
- File/sheet: `ASN.2020101442-File027.xlsx`, sheet "Human data matrix fraction"

## 3. Species Context
- Species: Homo sapiens
- Lifespan reference: ~75-85 years
- Aging cutoffs applied:
  - Human: young ≤30 years, old ≥55 years
  - Study groups: 15-37yr (young) vs 61-69yr (aged)

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Already 2 Groups)
- **Young group:** 15, 29, 37 years
  - Ages: 15 years, 29 years, 37 years
  - Justification: 15yr and 29yr clearly ≤30yr cutoff (young adults); 37yr slightly above but grouped by authors as "young donors"
  - Sample count: 3 donors × 2 compartments = 6 samples (G15, G29, G37, T15, T29, T37)
  - Note: 37yr is marginally above 30yr cutoff but below middle-age range; authors classified as young cohort

- **Old group:** 61, 67, 69 years
  - Ages: 61 years, 67 years, 69 years
  - Justification: All ≥55yr cutoff, clearly geriatric/post-reproductive age
  - Sample count: 3 donors × 2 compartments = 6 samples (G61, G67, G69, T61, T67, T69)

- **EXCLUDED:** None
  - Study already has binary age design
  - 37yr donor included in young group per author classification (minor deviation from strict 30yr cutoff)

### Impact Assessment
- **Data retained:** 100% (12/12 samples; 6 donors × 2 compartments)
- **Data excluded:** 0%
- **Meets ≥66% threshold?** YES (100%)
- **Signal strength:** Excellent - 24-54 year age gap between young (15-37yr) and aged (61-69yr) cohorts; strong aging contrast for kidney ECM

**Note on 37yr donor:** While technically above the 30yr cutoff, the donor was classified by study authors as "young" and represents early adulthood rather than middle age. Excluding this would reduce sample size without biological justification.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `ASN.2020101442-File027.xlsx`
- Sheet/tab name: "Human data matrix fraction"
- File size: 2,611 rows × 31 columns
- Format: Excel (.xlsx)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "Accession" | ✅ MAPPED | UniProt accession from Mascot search |
| Protein_Name | "Description" | ✅ MAPPED | Canonical protein names |
| Gene_Symbol | "Gene name" | ✅ MAPPED | Required for matrisome classification |
| Tissue | Parse from column prefix | ✅ MAPPED | "G"=Glomerular, "T"=Tubulointerstitial |
| Species | Constant "Homo sapiens" | ✅ MAPPED | Human kidney donors |
| Age | Parse from column suffix | ✅ MAPPED | "G15"→15yr, "T67"→67yr, etc. |
| Age_Unit | Constant "years" | ✅ MAPPED | Donor ages in years |
| Abundance | Intensity columns (Gxx, Txx) | ✅ MAPPED | Hi-N normalized LFQ intensities |
| Abundance_Unit | Constant "HiN_LFQ_intensity" | ✅ MAPPED | Progenesis Hi-N top-3 peptide normalization |
| Method | Constant "Label-free LC-MS/MS (Progenesis)" | ✅ MAPPED | Progenesis + Mascot workflow (Methods p.5) |
| Study_ID | Constant "Randles_2021" | ✅ MAPPED | Unique identifier |
| Sample_ID | Template "{compartment}_{age}" | ✅ MAPPED | "G_15", "T_61", etc. |
| Parsing_Notes | Template | ✅ MAPPED | Compartment assignment, Hi-N normalization |

### Mapping Gaps (if any)

⚠️ **Minor note: Duplicate .1 suffix columns**
- Problem: Exported sheet contains duplicate columns with ".1" suffix (e.g., "G15.1")
- Proposed solution: Retain only primary intensity columns (G15, T15, etc.); ".1" columns appear to be binary detection flags per supplementary notes
- Impact: Filter out ".1" columns during parsing; use only primary intensity values
- Status: Not a gap, just requires column filtering logic

✅ **All 13 schema columns:** Successfully mapped

## 6. Implementation Notes
- Column name parsing: "G15" → Tissue="Glomerular", Age=15; "T67" → Tissue="Tubulointerstitial", Age=67
- Sample_ID format: `{compartment}_{age}` (e.g., "G_15", "T_61", "G_37", "T_69")
- Parsing_Notes template: "Age={age}yr donor from column '{col_name}'; Tissue={compartment} (G=Glomerular, T=Tubulointerstitial); Hi-N normalized LFQ intensity from Progenesis LC-MS (Methods p.5); Top-3 peptide quantification"
- Special handling:
  - Filter out duplicate ".1" suffix columns (binary detection, not quantitative)
  - Retain only primary intensity columns: G15, G29, G37, G61, G67, G69, T15, T29, T37, T61, T67, T69
  - Each protein will generate 12 rows (6 glomerular + 6 tubulointerstitial samples)
- Age group classification:
  - Young: G15, G29, G37, T15, T29, T37 (ages 15, 29, 37)
  - Old: G61, G67, G69, T61, T67, T69 (ages 61, 67, 69)
- Tissue-specific parsing: Generate separate rows for each compartment to preserve spatial resolution
