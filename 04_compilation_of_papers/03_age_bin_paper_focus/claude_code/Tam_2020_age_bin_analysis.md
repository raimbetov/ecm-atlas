# Tam 2020 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: MaxQuant Label-Free Quantification (LFQ)
- LFQ compatible: YES - Label-free LC-MS/MS with MaxQuant LFQ (Methods p.29)
- Included in Phase 1 parsing: YES

## 2. Current Age Groups
- Young cadaveric spine: 16-year-old male (1 donor)
- Aged cadaveric spine: 59-year-old male (1 donor)
- Additional validation cohorts: Teens (14-20 years) and adults (47-68 years) from surgical samples (Table 1)
- **Primary LFQ profiles:** 66 spatial profiles from young (16yr) and old (59yr) cadavers
- File/sheet: `elife-64940-supp1-v3.xlsx`, sheet "Raw data" (3,158 rows × 80 columns)

## 3. Species Context
- Species: Homo sapiens
- Lifespan reference: ~75-85 years
- Aging cutoffs applied:
  - Human: young ≤30 years, old ≥55 years
  - Study groups: 16 years (young) vs 59 years (aged)

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Already 2 Groups)
- **Young group:** 16 years
  - Ages: 16 years (cadaveric donor)
  - Justification: Well below 30yr cutoff, adolescent/young adult spine
  - Sample count: ~33 spatial profiles from 1 donor (multiple disc compartments/coordinates)

- **Old group:** 59 years
  - Ages: 59 years (cadaveric donor)
  - Justification: Above 55yr cutoff, aged/degenerating spine phenotype
  - Sample count: ~33 spatial profiles from 1 donor (multiple disc compartments/coordinates)

- **EXCLUDED:** None
  - Primary LFQ dataset already has binary age design (young vs old cadavers)
  - Validation cohorts (Table 1) not part of main LFQ dataset

### Impact Assessment
- **Data retained:** 100% (all 66 spatial profiles from 2 donors)
- **Data excluded:** 0%
- **Meets ≥66% threshold?** YES (100%)
- **Signal strength:** Excellent - 43-year age gap (16yr vs 59yr); captures adolescent vs aging spine biology; spatially resolved across disc compartments

**Note:** Study design uses single donors per age group but multiple spatial sampling (66 profiles across nucleus pulposus, inner/outer annulus fibrosus, transition zones) providing rich biological context despite limited donor number.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `elife-64940-supp1-v3.xlsx`
- Sheet/tab name: "Raw data"
- File size: 3,158 rows × 80 columns
- Format: Excel (.xlsx)
- Metadata sheet: "Sample information" (profile name, disc level, age-group, direction, compartment)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "T: Majority protein IDs" (Raw data) | ✅ MAPPED | MaxQuant UniProt accessions |
| Protein_Name | "T: Protein names" | ✅ MAPPED | Standard protein annotations |
| Gene_Symbol | "T: Gene names" | ✅ MAPPED | Required for matrisome classification |
| Tissue | Parse from Sample information | ✅ MAPPED | Compartment: NP (nucleus pulposus), IAF (inner annulus), OAF (outer annulus), NP/IAF (transition) |
| Species | Constant "Homo sapiens" | ✅ MAPPED | Human cadaveric and surgical discs |
| Age | Map from Sample information | ✅ MAPPED | "young" profile → 16yr, "old" profile → 59yr (per Table 1) |
| Age_Unit | Constant "years" | ✅ MAPPED | Ages in years |
| Abundance | "LFQ intensity ..." columns | ✅ MAPPED | Each column = one spatial profile (e.g., "LFQ intensity L3/4_old_L_OAF") |
| Abundance_Unit | Constant "LFQ_intensity" | ✅ MAPPED | MaxQuant LFQ output (Methods p.29) |
| Method | Constant "Label-free LC-MS/MS (MaxQuant)" | ✅ MAPPED | MaxQuant LFQ workflow |
| Study_ID | Constant "Tam_2020" | ✅ MAPPED | Unique identifier |
| Sample_ID | Use profile name from column | ✅ MAPPED | e.g., "L3/4_old_L_OAF", "L4/5_young_R_NP" |
| Parsing_Notes | Template | ✅ MAPPED | Spatial profile details, disc level, direction, compartment |

### Mapping Gaps (if any)

⚠️ **Minor handling note: Profile name parsing**
- Challenge: Profile names encode multiple attributes (e.g., "L3/4_old_L_OAF" = disc level L3/4, old donor, left side, outer annulus)
- Proposed solution:
  - Parse profile name to extract: disc_level, age_group, direction, compartment
  - Cross-reference with "Sample information" sheet for metadata validation
  - Use profile name directly as Sample_ID for traceability
- Impact: Requires string parsing logic but all information is present
- Status: Mappable with parsing logic (not a true gap)

✅ **All 13 schema columns:** Successfully mapped

## 6. Implementation Notes
- Column name parsing:
  - "LFQ intensity L3/4_old_L_OAF" → extract "L3/4_old_L_OAF" as profile name
  - Parse profile: L3/4 (disc level), old (age=59yr), L (left side), OAF (outer annulus fibrosus)
- Sample_ID format: Use profile name directly (e.g., "L3/4_old_L_OAF", "L4/5_young_R_NP")
- Parsing_Notes template: "Age={age}yr from '{age_group}' cadaver per Table 1; Disc level={level}; Direction={L/R}; Compartment={compartment} (NP=nucleus pulposus, IAF=inner annulus, OAF=outer annulus, NP/IAF=transition); Spatial profile from {profile_name}; LFQ intensity from MaxQuant (Methods p.29)"
- Special handling:
  - Strip "T: " prefix from column names in Raw data sheet
  - Cross-reference Sample information sheet to validate profile metadata
  - Each protein generates 66 rows (one per spatial profile)
  - Tissue mapping: NP→"Nucleus pulposus", IAF→"Inner annulus fibrosus", OAF→"Outer annulus fibrosus", NP/IAF→"Transition zone"
- Age group mapping:
  - Profiles with "young" → Age=16 years
  - Profiles with "old" → Age=59 years
- Validation data: Supplementary workbooks (Supp2-Supp5) contain derived statistics; focus parsing on Supp1 Raw data for primary LFQ intensities
