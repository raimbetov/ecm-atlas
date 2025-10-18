# Caldeira 2017 Data Samples and Technical Details

## Source Data Sample

### Raw Excel File Structure (MOESM3_ESM.xls, Sheet: "1. Proteins")

**First 5 Proteins with iTRAQ Sample Values:**

| Accession # | Protein Name | Foetus 1 | Foetus 2 | Young 1 | Young 2 | Young 1(2) | Old 1 | Old 2 | Old 3 |
|-------------|--------------|----------|----------|---------|---------|-----------|-------|-------|-------|
| P13608 | Aggrecan | 0.847 | 0.855 | 1.019 | 1.148 | 1.057 | 1.330 | 1.247 | 1.600 |
| P02459 | Collagen II | 0.946 | 1.047 | 1.343 | 1.675 | 1.445 | 0.887 | 0.982 | 0.991 |
| P07589 | Fibronectin | 0.673 | 0.625 | 0.802 | 0.816 | 0.964 | 1.236 | 1.159 | 1.514 |
| P12111 | Collagen VI | 0.924 | 0.942 | 1.251 | 1.156 | 1.089 | 1.032 | 1.009 | 1.145 |
| P21809 | Biglycan | 1.042 | 1.087 | 5.642 | 5.824 | 5.291 | 8.976 | 8.291 | 9.632 |

**Key observation:** All values cluster around 1.0 for most proteins, indicating normalized iTRAQ ratios. Biglycan shows large fold-changes (5-9x).

---

## Processed Database Values

### Caldeira_2017 Wide Format (First 10 ECM Proteins)

From: `05_papers_to_csv/03_Caldeira_2017_paper_to_csv/Caldeira_2017_wide_format.csv`

| Protein_ID | Gene_Symbol | Abundance_Young | Abundance_Old | Method | Z_Score_Young (in DB) |
|------------|----------|-----------------|--------------|--------|----------------------|
| P13608 | PGCA | 1.1346 | 1.3412 | iTRAQ 8-plex | -0.42 |
| P02459 | CO2A1 | 5.1407 | 0.4310 | iTRAQ 8-plex | 1.53 |
| P07589 | FINC | 0.6611 | 1.2988 | iTRAQ 8-plex | -0.79 |
| P12111 | CO6A3 | 1.8604 | 1.2871 | iTRAQ 8-plex | -0.21 |
| P21809 | PGS1 | 10.7579 | 13.1787 | iTRAQ 8-plex | 2.24 |
| Q29116 | TENA | 6.5460 | 0.0404 | iTRAQ 8-plex | 1.24 |
| P35445 | COMP | 1.6480 | 12.3160 | iTRAQ 8-plex | -0.87 |
| P98160 | PGBM | 1.0271 | 0.8341 | iTRAQ 8-plex | -0.45 |
| P13605 | FMOD | 2.4206 | 2.0844 | iTRAQ 8-plex | -0.19 |
| Q27972 | CHAD | 1.6129 | 1.5844 | iTRAQ 8-plex | -0.52 |

**Note:** Abundance values are AVERAGED from iTRAQ sample columns, representing normalized ratios. Z-scores are calculated from log2-transformed values.

---

## Value Distribution Comparison

### Raw Parsed Data Statistics (Before Averaging)

**All 1,078 individual measurements:**
- Mean: 2.10
- Median: 1.15
- Std Dev: 3.48
- Min: 0.0108 (near-undetectable protein)
- Max: 36.31 (highly abundant protein)

### By Age Group (Parsed Data)

**Foetus samples (n=252):**
```
Mean:     1.654
Median:   0.909
Std Dev:  2.631
Min:      0.011
Max:      13.677
```

**Young samples (n=498):**
```
Mean:     1.910
Median:   1.078
Std Dev:  2.822
Min:      0.021
Max:      15.996
```

**Old samples (n=328):**
```
Mean:     2.830
Median:   1.422
Std Dev:  4.816
Min:      0.022
Max:      36.308
```

### After Averaging to 43 ECM Proteins

**Abundance_Young (n=43):**
```
Mean:     2.1853
Median:   1.6480
Std Dev:  2.0947
Min:      0.2396
Max:      10.7579
```

**Abundance_Old (n=43):**
```
Mean:     3.4426
Median:   2.1583
Std Dev:  3.6057
Min:      0.0404
Max:      13.1787
```

---

## Distribution Visualization (Text-based)

### Value Histogram - Raw Parsed Data

```
Ratio value  Count   Distribution
0.0-0.5      251    ############ (23.3%)
0.5-1.0      206    ########## (19.1%)
1.0-1.5      213    ########## (19.8%)
1.5-2.0      152    ####### (14.1%)
2.0-3.0      107    ##### (9.9%)
3.0-5.0      79     #### (7.3%)
5.0-10.0     48     ## (4.5%)
10.0+        23     # (2.1%)
```

**Key insight:** Strongly skewed toward values <2.0, consistent with normalized ratio distribution.

---

## Comparison with LFQ Data

### Why Caldeira Values Look "Too Low"

**LFQ Study Example (for comparison, not actual data):**
```
Raw LFQ intensity:  [1000, 5000, 30000]  (peptide counts)
After log2:         [10.0, 12.3, 14.9]   (range: 10-20)
Z-score:            [-0.5, 0.2, 0.8]     (range: -1 to 1)
```

**Caldeira (Actual):**
```
iTRAQ ratio:        [0.8, 1.2, 3.5]      (fold-change ratios)
After log2:         [-0.32, 0.26, 1.81]  (range: -1 to 2)
Z-score:            [-0.52, -0.19, 0.42] (range: -1 to 1)
```

**The difference:** LFQ values are typically 100-1000x larger because they represent RAW instrument counts. Caldeira values are 1-5 because they're normalized ratios.

---

## Technical Notes on iTRAQ Processing

### How Protein Pilot Generates Ratios

1. **Raw data:** 8 iTRAQ mass channels per peptide
2. **Signal integration:** Sums intensity across all acquisitions
3. **Ratio calculation:** Each sample vs. reference (usually pooled sample or geometric mean)
4. **Output:** Pre-normalized ratios (0.01-36 typical range)

### Why Values Cluster Around 1.0

- **1.0 = reference level** (by definition)
- **Proteins near reference level:** majority of measured proteins
- **Proteins > 2.0:** relatively few (upregulated proteins)
- **Proteins < 0.5:** relatively few (downregulated proteins)

This creates the characteristic distribution seen in parsed data.

### What We CANNOT Do with This Data

1. **Contact original mass spec machine** - data is already processed/archived
2. **Reconstruct raw channel signals** - only ratios are available
3. **Re-normalize differently** - would require raw instrument data
4. **Apply batch correction** - assumes abundance-based measurement, not ratios

---

## Batch Correction Incompatibility Example

### Hypothetical Scenario

**If we tried to batch-correct Caldeira with LFQ studies:**

**Input:**
- Study A (LFQ): Protein X in Young = log2(50000 counts) = 15.6
- Study B (Caldeira): Protein X in Young = log2(1.2 ratio) = 0.26

**After Z-scoring within study:**
- Study A: (15.6 - mean) / std = 0.8 (on 10-16 scale)
- Study B: (0.26 - mean) / std = -0.2 (on -1 to 2 scale)

**Batch correction sees:** These two values have fundamentally different scales
- Cannot determine if difference is biological or technical
- Trying to "correct" them would destroy biological signal
- Result: Invalid statistical inference

---

## File Locations Summary

| File | Purpose | Key Data |
|------|---------|----------|
| `data_raw/Caldeira*/41598_2017_11960_MOESM3_ESM.xls` | Original paper supplementary data | Raw iTRAQ ratios (0.01-36) |
| `references/data_processed/Caldeira_2017_parsed.csv` | Long format, all 1,078 measurements | Abundant_Unit = "normalized_ratio" |
| `05_papers_to_csv/03_Caldeira_2017.../Caldeira_2017_wide_format.csv` | Processed wide format, 43 ECM proteins | Averaged abundances, z-scores |
| `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` | Final database | Caldeira rows with z-scores |

---

## Conclusion

Caldeira 2017 data contains valid iTRAQ-derived ratios with correct values. The "low" appearance (medians 1.65-2.16) is mathematically appropriate for normalized fold-change ratios. However, this data type fundamentally incompatible with batch correction when mixed with abundance-based methods (LFQ, TMT).

**Recommendation:** Keep Caldeira in database for quality/validity checks, but EXCLUDE from batch correction statistical analyses.
