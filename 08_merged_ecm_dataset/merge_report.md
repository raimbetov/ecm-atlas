# ECM Dataset Merge Report

**Generated:** 2025-10-12 18:05:38

## Summary

- **Total proteins:** 1451
- **Unique proteins:** 498
- **Datasets:** 5
- **Output file:** merged_ecm_aging_zscore.csv

## Datasets Included

### Randles_2021 - Glomerular
- **Organ:** Kidney
- **Compartment:** Glomerular
- **Proteins:** 458
- **ECM Filtered:** Yes

### Randles_2021 - Tubulointerstitial
- **Organ:** Kidney
- **Compartment:** Tubulointerstitial
- **Proteins:** 458
- **ECM Filtered:** Yes

### Tam_2020 - NP
- **Organ:** Intervertebral_disc
- **Compartment:** NP
- **Proteins:** 993
- **ECM Filtered:** No (pre-filtered)

### Tam_2020 - IAF
- **Organ:** Intervertebral_disc
- **Compartment:** IAF
- **Proteins:** 993
- **ECM Filtered:** No (pre-filtered)

### Tam_2020 - OAF
- **Organ:** Intervertebral_disc
- **Compartment:** OAF
- **Proteins:** 993
- **ECM Filtered:** No (pre-filtered)

## Statistics by Compartment

- **Glomerular:** 229 proteins
- **IAF:** 317 proteins
- **NP:** 300 proteins
- **OAF:** 376 proteins
- **Tubulointerstitial:** 229 proteins

## Matrisome Categories

- **ECM Glycoproteins:** 453 proteins
- **ECM Regulators:** 375 proteins
- **Secreted Factors:** 222 proteins
- **ECM-affiliated Proteins:** 162 proteins
- **Collagens:** 154 proteins
- **Proteoglycans:** 85 proteins

## Z-Score Statistics

### Young
- Mean: 0.131
- Std Dev: 1.096

### Old
- Mean: 0.145
- Std Dev: 1.103

### Delta (Old - Young)
- Mean: 0.116
- Std Dev: 0.595

## ✅ Data Quality

All validation checks passed successfully!

## Column Schema

### Metadata Columns (Added)
- `Dataset_Name` - Study identifier (Randles_2021 / Tam_2020)
- `Organ` - Tissue/organ (Kidney / Intervertebral_disc)
- `Compartment` - Tissue compartment (Glomerular, Tubulointerstitial, NP, IAF, OAF)

### Core Data Columns
- `Protein_ID` - UniProt ID
- `Protein_Name` - Full protein name
- `Gene_Symbol` - Gene symbol
- `Canonical_Gene_Symbol` - Canonical gene symbol
- `Species` - Organism (Homo sapiens)

### Abundance Columns
- `Abundance_Young` - Abundance in young samples
- `Abundance_Old` - Abundance in old samples
- `Abundance_Young_transformed` - log2-transformed (Randles only)
- `Abundance_Old_transformed` - log2-transformed (Randles only)

### Z-Score Columns
- `Zscore_Young` - Z-score for young samples
- `Zscore_Old` - Z-score for old samples
- `Zscore_Delta` - Change in z-score (Old - Young)

### Annotation Columns
- `Matrisome_Category` - ECM protein category
- `Matrisome_Division` - Core matrisome / Matrisome-associated
- `Match_Level` - Annotation match level
- `Match_Confidence` - Confidence score (0-100)

### Method & Study
- `Method` - Proteomics method
- `Study_ID` - Study identifier
- `Tissue` - Original tissue annotation
- `Tissue_Compartment` - Original compartment annotation

### Additional (Tam 2020 only)
- `N_Profiles_Young` - Number of spatial profiles (young)
- `N_Profiles_Old` - Number of spatial profiles (old)
