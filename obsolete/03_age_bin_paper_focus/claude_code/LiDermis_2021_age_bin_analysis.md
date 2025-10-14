# Li Dermis 2021 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: LC-MS/MS label-free quantification with log2 normalization
- LFQ compatible: YES - Mass spectrometry-based label-free quantification (Figure 1A workflow)
- Included in Phase 1 parsing: YES

## 2. Current Age Groups
- Toddler (1-3 years, midpoint 2yr): 2 samples
- Teenager (8-20 years, midpoint 14yr): 3 samples
- Adult (30-50 years, midpoint 40yr): 3 samples
- Elderly (>60 years, midpoint 65yr): 5 samples
- **Total:** 15 samples across 4 age groups
- File/sheet: `Table 2.xlsx`, sheet "Table S2"

## 3. Species Context
- Species: Homo sapiens
- Lifespan reference: ~75-85 years
- Aging cutoffs applied:
  - Human: young ≤30 years, old ≥55 years
  - Study groups: 2yr, 14yr, 40yr, 65yr

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** Toddler (2yr) + Teenager (14yr)
  - Ages: 2 years, 14 years
  - Justification: Both ≤30yr cutoff, pre-reproductive peak, high collagen synthesis, no aging markers
  - Sample count: 2 + 3 = 5 samples

- **Old group:** Elderly (65yr) ONLY
  - Ages: 65 years
  - Justification: ≥55yr cutoff, clearly post-reproductive, geriatric phenotype, ECM degradation markers
  - Sample count: 5 samples

- **EXCLUDED:** Adult (40yr) - middle-aged
  - Ages: 40 years (30-50 year range)
  - Rationale: Between 30-55yr cutoffs, ambiguous aging state (pre-senescent but not young)
  - Sample count: 3 samples (20% data loss)

### Impact Assessment
- **Data retained:** 10/15 samples = 67% ✅ (meets ≥66% threshold)
- **Data excluded:** 3/15 samples = 20%
- **Meets ≥66% threshold?** YES (67% retention)
- **Signal strength:** Excellent - Strong contrast between 2-14yr vs 65yr (43-51 year gap); captures pre-reproductive vs geriatric dermis biology

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `Table 2.xlsx`
- Sheet/tab name: "Table S2"
- File size: 266 rows × 22 columns
- Format: Excel (.xlsx)
- Skip rows: 3 (header rows with merged cells per Supplementary note)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "Protein ID" (col 0, after skip) | ✅ MAPPED | UniProt accessions |
| Protein_Name | Not in file, derive from UniProt | ⚠️ DERIVED | Requires external lookup; may use "Protein Description" from Table S3 |
| Gene_Symbol | "Gene symbol" (col 1, after skip) | ✅ MAPPED | Official gene symbols |
| Tissue | Constant "Skin dermis" | ✅ MAPPED | Decellularized dermal scaffolds |
| Species | Constant "Homo sapiens" | ✅ MAPPED | Human donors |
| Age | Parse from column group prefix | ✅ MAPPED | Toddler=2yr, Teenager=14yr, Adult=40yr, Elderly=65yr (midpoints from Table S1) |
| Age_Unit | Constant "years" | ✅ MAPPED | Age ranges in years |
| Abundance | Sample columns (cols 4-18) | ✅ MAPPED | "Toddler-Sample1-2", "Teenager-Sample1-3", etc. |
| Abundance_Unit | Constant "log2_normalized_intensity" | ✅ MAPPED | Per Figure 5B caption |
| Method | Constant "Label-free LC-MS/MS" | ✅ MAPPED | Figure 1A workflow, quantitative proteomics |
| Study_ID | Constant "LiDermis_2021" | ✅ MAPPED | Unique identifier |
| Sample_ID | Template "{age_group}_{sample_num}" | ✅ MAPPED | "Toddler_1", "Teenager_2", "Elderly_3" |
| Parsing_Notes | Template | ✅ MAPPED | Midpoint ages, decellularization protocol, log2 scaling |

### Mapping Gaps (if any)

⚠️ **Gap 1: Protein_Name**
- Problem: Source file "Table S2" contains only Protein IDs and Gene symbols, no protein name column
- Proposed solution:
  - Option A: Map Protein_ID (UniProt) → Protein_Name via UniProt API
  - Option B: Cross-reference with Table S3 which may have "Protein Description"
  - Recommendation: Option B first (check Table S3), fallback to Option A (UniProt lookup)
- Impact: Requires pre-processing step before final parsing

✅ **All other columns:** Direct mapping or hardcoded constants available

## 6. Implementation Notes
- Column name parsing: "Toddler-Sample1-2" → extract "Toddler" → lookup midpoint age 2yr from Table S1
- Sample_ID format: `Dermis_{age_group}_{sample_num}` (e.g., "Dermis_Toddler_1", "Dermis_Teenager_2", "Dermis_Elderly_3")
- Parsing_Notes template: "Age={age}yr (midpoint of {range}) from column '{col_name}' per Supplementary Table S1; Abundance from Table 2 sheet 'Table S2' col {col}; log2 normalized per Methods and Figure 5B; Decellularized dermal scaffold per Figure 1A"
- Special handling:
  - Skip first 3 rows before reading data (merged header cells)
  - Filter samples to EXCLUDE Adult group (40yr): columns "Adult-Sample1-3"
  - Retain only Toddler, Teenager, and Elderly sample columns
  - UniProt mapping: Pre-process Protein_ID list to fetch Protein_Name before final parsing
- Age group filtering for normalization:
  - INCLUDE: Toddler-Sample1, Toddler-Sample2 (2yr)
  - INCLUDE: Teenager-Sample1, Teenager-Sample2, Teenager-Sample3 (14yr)
  - EXCLUDE: Adult-Sample1, Adult-Sample2, Adult-Sample3 (40yr) ❌
  - INCLUDE: Elderly-Sample1, Elderly-Sample2, Elderly-Sample3, Elderly-Sample4, Elderly-Sample5 (65yr)
