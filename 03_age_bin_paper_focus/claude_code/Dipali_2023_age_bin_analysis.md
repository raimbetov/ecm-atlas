# Dipali 2023 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: DirectDIA label-free quantitative proteomics
- LFQ compatible: YES - Data-independent acquisition (DIA) is a label-free method
- Included in Phase 1 parsing: YES

## 2. Current Age Groups
- Reproductively young: 6-12 weeks
- Reproductively old: 10-12 months
- Total samples: Not specified in available analysis (data shows log2 ratios)
- File/sheet: `Candidates.tsv`

## 3. Species Context
- Species: Mus musculus
- Lifespan reference: ~24-30 months
- Aging cutoffs applied:
  - Mouse: young ≤4 months, old ≥18 months
  - Study groups: 6-12 weeks (~1.5-3 months, young) vs 10-12 months (old)

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Already 2 Groups)
- **Young group:** 6-12 weeks (~1.5-3 months)
  - Ages: 6-12 weeks (reproductively young)
  - Justification: Well below 4-month cutoff, pre-reproductive peak
  - Sample count: Unknown (data provided as ratios)
- **Old group:** 10-12 months
  - Ages: 10-12 months (reproductively old)
  - Justification: Below 18-month cutoff but biologically relevant aging for ovarian tissue
  - Sample count: Unknown (data provided as ratios)
- **EXCLUDED:** None
  - Study already has binary age groups

### Impact Assessment
- **Data retained:** 100% (study already has 2 groups)
- **Data excluded:** 0%
- **Meets ≥66% threshold?** YES (100%)
- **Signal strength:** Good - ~7-10.5 month age gap, relevant for reproductive aging in mice

**Note:** While the "old" group (10-12 months) is technically below the strict 18-month geriatric cutoff, it represents reproductively old mice and is biologically meaningful for ovarian aging studies. No normalization needed as study already has binary age design.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `Candidates.tsv`
- Sheet/tab name: N/A (TSV file)
- File size: 4,909 rows × unknown columns
- Format: Tab-separated values (.tsv)

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | "ProteinGroups" | ✅ MAPPED | UniProt IDs |
| Protein_Name | Derive from UniProt lookup | ⚠️ DERIVED | Requires external mapping |
| Gene_Symbol | "Genes" | ✅ MAPPED | Gene symbols provided |
| Tissue | Constant "Ovary" | ✅ MAPPED | Study focus on ovarian tissue |
| Species | Constant "Mus musculus" | ✅ MAPPED | Mouse study |
| Age | Parse from "Condition" columns | ✅ MAPPED | "Young_Native" vs "Old_Native" |
| Age_Unit | Mixed: "weeks" for young, "months" for old | ✅ MAPPED | Young=6-12wk, Old=10-12mo |
| Abundance | "AVG Log2 Ratio" | ⚠️ COMPLEX | Ratio data, not absolute intensities |
| Abundance_Unit | Constant "log2_ratio" | ✅ MAPPED | Differential expression metric |
| Method | Constant "DirectDIA label-free" | ✅ MAPPED | DIA-based quantification |
| Study_ID | Constant "Dipali_2023" | ✅ MAPPED | Unique identifier |
| Sample_ID | Template from conditions | ✅ MAPPED | Generate as "Young_Native_X", "Old_Native_X" |
| Parsing_Notes | Template | ✅ MAPPED | Document ratio-based data structure |

### Mapping Gaps (if any)

⚠️ **Gap 1: Protein_Name**
- Problem: Source file only has UniProt IDs in "ProteinGroups", no protein names
- Proposed solution: Map UniProt ID → Protein_Name via UniProt API or reference table
- Impact: Requires pre-processing step

⚠️ **Gap 2: Abundance representation**
- Problem: Data provided as log2 ratios (Old/Young), not absolute intensities per sample
- Proposed solution:
  - Option A: Represent as differential expression (ratio values)
  - Option B: Request raw intensity files from authors/repository
  - Recommendation: Document as ratio-based metric in Parsing_Notes
- Impact: Different interpretation than absolute LFQ intensities

## 6. Implementation Notes
- Column name mapping: "Condition Numerator" = Old_Native, "Condition Denominator" = Young_Native
- Sample_ID format: Generate as "{condition}_{replicate}" (e.g., "Young_Native_1", "Old_Native_1")
- Parsing_Notes template: "Age: Young=6-12 weeks, Old=10-12 months from Condition columns; Abundance is log2 ratio (Old/Young) from 'AVG Log2 Ratio'; DirectDIA label-free quantification; Native (non-enriched) ovary samples"
- Special handling: Data structure is differential (ratio-based) rather than per-sample intensities; may need separate parsing strategy or note as comparative dataset
- Age unit conversion: Young group in weeks (convert to ~1.5-3 months), Old group in months (10-12 months)
