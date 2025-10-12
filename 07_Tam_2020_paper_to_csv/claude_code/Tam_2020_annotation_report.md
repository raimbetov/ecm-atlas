# Tam 2020 Protein Annotation Report

**Generated:** 2025-10-12 16:28:22

## Summary

- **Total Proteins:** 48,961
- **Matched Proteins:** 14,099
- **Unmatched Proteins:** 34,862
- **Coverage Rate:** 28.80%
- **Target Coverage:** 90%
- **Status:** ⚠️ WARNING - Below target

## Match Level Distribution

| Match Level | Count | Percentage |
|-------------|-------|------------|
| Level 1: Gene Symbol | 13,650 | 27.9% |
| Level 2: UniProt ID | 351 | 0.7% |
| Level 3: Synonym | 98 | 0.2% |
| Level 4: Unmatched | 34,862 | 71.2% |

## Matrisome Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| Collagens | 1,277 | 9.1% |
| ECM Glycoproteins | 4,702 | 33.3% |
| ECM Regulators | 3,505 | 24.9% |
| ECM-affiliated Proteins | 1,439 | 10.2% |
| Proteoglycans | 1,197 | 8.5% |
| Secreted Factors | 1,979 | 14.0% |

## Matrisome Division Distribution

| Division | Count | Percentage |
|----------|-------|------------|
| Core matrisome | 7,176 | 50.9% |
| Matrisome-associated | 6,923 | 49.1% |

## Known Marker Validation

| Marker | Status | Category | Division | Match Level | Notes |
|--------|--------|----------|----------|-------------|-------|
| COL1A1 | ✅ PASS | Collagens | Core matrisome | Level 1: Gene Symbol | Correctly annotated |
| COL2A1 | ✅ PASS | Collagens | Core matrisome | Level 1: Gene Symbol | Correctly annotated |
| FN1 | ✅ PASS | ECM Glycoproteins | Core matrisome | Level 1: Gene Symbol | Correctly annotated |
| ACAN | ✅ PASS | Proteoglycans | Core matrisome | Level 1: Gene Symbol | Correctly annotated |
| MMP2 | ✅ PASS | ECM Regulators | Matrisome-associated | Level 1: Gene Symbol | Correctly annotated |

## Methodology

### Hierarchical Matching Strategy

1. **Level 1: Gene Symbol Match (100% confidence)**
   - Exact match on gene symbol (case-insensitive)

2. **Level 2: UniProt ID Match (95% confidence)**
   - Match on UniProt accession numbers
   - Handles multiple IDs (semicolon-separated in dataset, colon-separated in reference)

3. **Level 3: Synonym Match (80% confidence)**
   - Match on known gene synonyms
   - Pipe-separated in reference

4. **Level 4: Unmatched (0% confidence)**
   - No match found in reference

### Data Sources

- **Dataset:** Tam_2020_standardized.csv
- **Reference:** human_matrisome_v2.csv

### Quality Metrics

- **Target Coverage:** ≥90% of proteins annotated
- **Validation:** Known ECM markers correctly classified

## ⚠️ Coverage Warning

The annotation coverage (28.80%) is below the target threshold (90%).

**Unmatched Proteins:** 34,862

This may indicate:
- Novel ECM proteins not in the reference database
- Non-ECM proteins in the dataset
- Gene symbol mismatches requiring manual curation
- Dataset-specific naming conventions

**Recommendation:** Review unmatched proteins for manual curation.
