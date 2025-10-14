# Tam 2020 - Age Bin Normalization Analysis (LFQ)

## 1. Method Verification
- Quantification method: MaxQuant LFQ intensities from LC-MS/MS profiling of intervertebral disc regions (Methods p.29, DIPPER study).
- LFQ compatible: ✅ YES — Label-free workflow with MaxQuant LFQ; no labeling chemistry used in atlas backbone.

## 2. Current Age Groups
- Young cadaveric disc (16-year male) — 33 spatial profiles (`LFQ intensity … young …`) covering nucleus pulposus, inner/outer annulus fibrosus, transition zones, and directional gradients.
- Old cadaveric disc (59-year male) — 33 matched profiles (`LFQ intensity … old …`) across identical spatial coordinates.
- Sample metadata stored in `Sample information` sheet describing profile name, disc level, compartment, and direction.

## 3. Species Context
- Species: Homo sapiens.
- Lifespan reference: ~80-year lifespan; disc degeneration accelerates post-50 years.
- Aging cutoffs applied: young ≤30 years, old ≥55 years.

## 4. Age Bin Mapping (Conservative - USER-APPROVED)

### Binary Split (Exclude Middle-Aged)
- **Young group:** All columns containing `young`
  - Ages: 16 years (single donor).
  - Justification: ≤30-year cutoff; healthy adolescent disc baseline.
  - Sample count: 33 spatial profiles.
- **Old group:** All columns containing `old`
  - Ages: 59 years (single donor).
  - Justification: ≥55-year cutoff; captures degenerative disc phenotype described in paper.
  - Sample count: 33 spatial profiles.
- **EXCLUDED:** None — dataset only provides young vs old cadaver donors.

### Impact Assessment
- **Data retained:** 66 / 66 profiles = 100% ✅
- **Data excluded:** 0%.
- **Meets ≥66% threshold?** ✅ YES.
- **Signal strength:** High — same disc level coverage across two age extremes enables spatial differential analysis.

## 5. Column Mapping Verification

### Source File Identification
- Primary proteomic file: `data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx`.
- Sheet/tab name: `Raw data` (LFQ intensities) plus `Sample information` for metadata.
- File size: 3,157 rows × 80 columns (after promoting row 2 as header).
- Format: Excel (.xlsx) from eLife supplement.

### 13-Column Schema Mapping

| Schema Column | Source | Status | Notes |
|---------------|--------|--------|-------|
| Protein_ID | `T: Majority protein IDs` | ✅ | MaxQuant protein group accession (UniProt IDs).
| Protein_Name | `T: Protein names` | ✅ | Canonical protein names.
| Gene_Symbol | `T: Gene names` | ✅ | Gene symbols from MaxQuant export.
| Tissue | From `Sample information` (compartment column) | ✅ | Distinguish NP, IAF, OAF, transition zones.
| Species | Constant `Homo sapiens` | ✅ | Human cadaver donors.
| Age | Map `young` → 16 years, `old` → 59 years | ✅ | Age assigned per donor; replicate profile inherits donor age.
| Age_Unit | Constant `years` | ✅ | Ages documented in years.
| Abundance | Columns `LFQ intensity ...` | ✅ | MaxQuant LFQ intensities per spatial profile.
| Abundance_Unit | Constant `LFQ_intensity` | ✅ | Linear LFQ intensity values.
| Method | Constant `Label-free LC-MS/MS (MaxQuant LFQ)` | ✅ | Documented workflow.
| Study_ID | Constant `Tam_2020` | ✅ | Parser identifier.
| Sample_ID | Use profile name from `Sample information` (e.g., `L3/4_old_L_OAF`) | ✅ | Unique identifier encodes disc level + orientation.
| Parsing_Notes | Template capturing disc level, compartment, direction, donor age | ✅ | Generated during ETL.

### Mapping Gaps (if any)
- ✅ All columns mapped — no outstanding gaps.

## 6. Implementation Notes
- Read raw sheet with `header=1`, then strip `T:` prefix from ID/name columns for clarity.
- Join long-format data with `Sample information` sheet to enrich compartment, orientation, and coordinate metadata.
- Ensure consistent ordering between young and old profiles (same disc coordinates) to support paired analyses.
- Keep log of additional validation cohorts (surgical samples) but exclude from primary LFQ matrix until Phase 2 expansion.
