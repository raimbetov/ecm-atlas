# LFQ-Focused Age Bin Normalization Summary

## Executive Summary
- Total studies reviewed: 11
- LFQ-compatible studies: 6 (`Angelidis 2019`, `Chmelova 2023`, `Dipali 2023`, `Li Dermis 2021`, `Randles 2021`, `Tam 2020`)
- Non-LFQ studies (excluded in Phase 1): 5 (`Ariosa 2021`, `Caldeira 2017`, `Li Pancreas 2021`, `Ouni 2022`, `Tsumagari 2023`)
- Studies requiring new age-bin normalization: 1 (`Li Dermis 2021` — 4 groups → 2 bins, exclude adult)
- Column mapping complete for 4/6 LFQ studies (Angelidis, Dipali, Randles, Tam); 2 studies need external enrichment (`Chmelova`, `Li Dermis`)

## LFQ Study Classification

| Study | Method | LFQ? | Age Groups (original) | Normalization Needed? | Column Mapping Complete? |
|-------|--------|------|-----------------------|-----------------------|--------------------------|
| Angelidis 2019 | MaxQuant LFQ (Orbitrap) | ✅ | 3 mo, 24 mo | No (already binary) | ✅ Complete |
| Chmelova 2023 | MaxQuant LFQ (Orbitrap) | ✅ | 3 mo, 18 mo (ctrl + MCAO) | No (already binary) | ⚠️ Needs UniProt mapping |
| Dipali 2023 | directDIA label-free | ✅ | 6–12 wk, 10–12 mo | No (already binary) | ✅ Complete (age ranges noted) |
| Li Dermis 2021 | Label-free LC-MS/MS | ✅ | 2 yr, 14 yr, 40 yr, 65 yr | ✅ Yes (exclude 40 yr) | ⚠️ Protein names missing |
| Randles 2021 | Progenesis Hi-N LFQ | ✅ | 15/29/37 yr vs 61/67/69 yr | No (already binary) | ✅ Complete |
| Tam 2020 | MaxQuant LFQ | ✅ | 16 yr vs 59 yr (66 spatial profiles) | No (already binary) | ✅ Complete |

## Excluded Studies (Non-LFQ)

| Study | Method | Reason for Exclusion |
|-------|--------|---------------------|
| Ariosa 2021 | Pulsed SILAC (iBAQ) | Isotope labeling; heavy/light ratios not LFQ comparable |
| Caldeira 2017 | iTRAQ | Isobaric labeling; requires reporter workflow |
| Li Pancreas 2021 | DiLeu 12-plex | Isobaric labeling; needs reporter normalization |
| Ouni 2022 | TMTpro 16-plex | Isobaric labeling; reporter intensities |
| Tsumagari 2023 | TMTpro 11-plex | Isobaric labeling; reporter intensities |

## Age Bin Normalization Outcomes (LFQ Only)

### Studies Already Two-Group (No Changes)
- **Angelidis 2019:** 3 mo vs 24 mo (mouse lung) — 8 samples retained.
- **Chmelova 2023:** 3 mo vs 18 mo (mouse cortex) — 17 samples retained (ctrl + MCAO metadata preserved).
- **Dipali 2023:** 6–12 wk vs 10–12 mo (mouse ovary) — 10 samples retained; note reproductive “old” <18 mo.
- **Randles 2021:** 15/29/37 yr vs 61/67/69 yr (human kidney) — 12 samples retained.
- **Tam 2020:** 16 yr vs 59 yr (human disc, 66 spatial profiles) — 66 profiles retained.

### Study Requiring Normalization
1. **Li Dermis 2021:** 4 groups → 2 bins (exclude adult 30–50 yr).
   - Young: Toddler (2 yr) + Teenager (14 yr) — 5 samples.
   - Old: Elderly (>60 yr, midpoint 65) — 3 samples.
   - Excluded: Adult (40 yr midpoint) — 2 samples (20% loss).
   - Retention: 80% ✅

## Column Mapping Verification Results

### Complete Mappings (✅)
- Angelidis 2019 — All 13 schema fields present.
- Dipali 2023 — All 13 fields mapped via directDIA export + condition metadata.
- Randles 2021 — All 13 fields mapped (compartment captured in notes).
- Tam 2020 — All 13 fields mapped with sample metadata join.

### Incomplete / Needs Enrichment (⚠️)
- **Chmelova 2023:** Missing UniProt accessions and protein names; require gene symbol → UniProt lookup (e.g., reference mapping) before ingestion.
- **Li Dermis 2021:** Supplementary Table S2 lacks protein names; map accession → protein name via Table S3 or UniProt service.

## Data Retention Summary

| Study | Original Samples | Retained Samples | Retention % | Notes |
|-------|-----------------|------------------|-------------|-------|
| Angelidis 2019 | 8 | 8 | 100% | Already two groups |
| Chmelova 2023 | 17 | 17 | 100% | Include ctrl + MCAO metadata |
| Dipali 2023 | 10 | 10 | 100% | Old cohort <18 mo; document assumption |
| Li Dermis 2021 | 10 | 8 | 80% | Adult (40 yr) excluded |
| Randles 2021 | 12 | 12 | 100% | Balanced compartments |
| Tam 2020 | 66 | 66 | 100% | Paired spatial profiles |

## Recommendations for Phase 2 Parsing

1. **Parse immediately:** Angelidis 2019, Randles 2021, Tam 2020, Dipali 2023 (after confirming age metadata export).
2. **Parse after minor enrichment:**
   - Chmelova 2023 — add UniProt mapping file for protein IDs/names.
   - Li Dermis 2021 — enrich protein names via accession lookup; ensure Adult group excluded during load.
3. **Defer (non-LFQ):** Ariosa 2021, Caldeira 2017, Li Pancreas 2021, Ouni 2022, Tsumagari 2023.

## Outstanding Risks / Follow-Ups
- Document deviation between reproductive “old” (10–12 mo) and standard ≥18 mo cutoff for Dipali 2023; track in parsing notes for transparency.
- Build UniProt mapping cache to resolve protein name gaps for Chmelova and Li Dermis before ingestion.
- Maintain metadata flags for Chmelova ischemia status so age-only analyses can filter to control subsets if desired.
