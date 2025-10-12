# Normalization Strategy (Tier 0.2)

The 11 datasets use heterogeneous quantification schemes (LFQ, iBAQ, TMT/iTRAQ ratios, DIA intensities). This strategy defines how to harmonize abundances for cross-study comparison while preserving traceability.

## Guiding Principles
1. **Preserve native scale initially:** Store raw/native values with explicit `Abundance_Unit`. Transformations for cross-study analyses will be applied on derived tables, not raw ingestion.
2. **Document transforms:** Any log transform, imputation, or scaling must be recorded per-sample in metadata and appended to `Parsing_Notes`.
3. **Two-tier normalization: within-study and cross-study.**

## Within-Study Handling

| Study | Native Scale | Immediate Actions | Notes |
|-------|--------------|-------------------|-------|
| Angelidis 2019 | LFQ intensity (linear) | Optional log2 for analysis; no additional normalization (MaxQuant already applied). | Use sample-level median scaling only if QC indicates drift. |
| Ariosa-Morejon 2021 | iBAQ (heavy/light) & H/L ratios | Keep heavy and light channels separate. Convert ratios to log2 when building comparative metrics. Normalize by shared reference (mean of light channels per tissue) to compare incorporation. |
| Caldeira 2017 | iTRAQ ratios (dimensionless) | Ratios already normalized via ProteinPilot. No extra scaling; convert to log2 fold change for modeling. |
| Chmelova 2023 | Log2 LFQ intensities (imputed) | Leave as-is; no back transformation. Track imputation flags when available. |
| Dipali 2023 | directDIA intensities (linear) | Apply log2 transform after minimal offset (e.g., +1) for statistical models. Within-run normalization via sample median recommended (Spectronaut already applies intensity normalization). |
| Li dermis 2021 | Log2 normalized intensity | Retain provided log2 values. Use midpoint age mapping for group comparisons. |
| Li pancreas 2021 | Log10 DiLeu reporter intensity | Confirm log base (values 6–10). Convert to linear abundance if integrated with other TMT datasets; otherwise work in log10. Adjust for batch using shared reference channel metadata. |
| Ouni 2022 | Normalized TMTpro intensities | Use provided `Q. Norm.` values. If combining soluble/insoluble fractions, apply fraction-specific scaling (e.g., subtract fraction median) before concatenation. |
| Randles 2021 | Hi-N LFQ intensity (linear) | Evaluate need for log2; use same approach as Angelidis/Tam. |
| Tam 2020 | LFQ intensity (linear) | Apply log2 after ingestion for analyses; maintain raw values in storage. |
| Dipali Condition TSVs | Ratios/log2 in supplementary | Convert to numeric features as needed; keep consistent with primary intensities. |

## Cross-Study Harmonization
1. **Log Transformation:** For downstream comparative models, convert all linear intensities to log2 scale (after adding small offset where zeros present).
2. **Z-score within study:** Compute sample-wise z-scores or quantile normalize within each study to remove run-specific effects.
3. **Batch Scaling:** Use per-study median centering to align scales before concatenation. For isobaric datasets (Caldeira, Li pancreas, Ouni) ensure reference channel alignment prior to cross-study scaling.
4. **Missing Data:** Retain NA where proteins absent; no imputation during ingestion. For modeling, consider study-specific imputation (e.g., MinDet for LFQ) with provenance in analysis scripts.
5. **Unit Metadata:** Build translation layer mapping `Abundance_Unit` to transformation pipeline (e.g., `'LFQ_intensity' -> log2`, `'iTRAQ_ratio' -> log2`, `'iBAQ_intensity' -> log2`, `'log2_normalized_intensity' -> identity`).

## Quality Control
- **Sample QC metrics:** Compute total ion intensity, number of quantified proteins, and coefficient of variation per sample. Flag outliers prior to normalization.
- **Protein overlap assessment:** Track intersection of UniProt IDs across studies to evaluate necessity of imputation or mapping to matrisome subset.
- **Versioning:** Store normalized matrices separately with naming convention `study_normalized_v1`. Each transformation step must be reproducible via config.

## Future Enhancements
- Integrate reference ECM marker scaling (e.g., anchor COL1A1, FN1) to align across label-free and isobaric datasets.
- Explore variance-stabilizing normalization (VSN) for DIA datasets (Dipali) if heteroscedasticity persists after log transformation.
- Implement multi-omics integration with transcriptomic datasets (Angelidis, Chmelova) once proteomic normalization stabilized.
