# Results: Randles 2021 Z-Score Normalization

## Workflow
- Implemented `zscore_normalization.py` to split the Randles 2021 wide-format dataset by kidney compartment, apply conditional log2(x+1) transforms, and compute per-compartment z-scores for Young and Old abundance values.
- Generated compartment-specific outputs with original metadata plus derived `Abundance_*_log2`, `Zscore_*`, and `Zscore_Delta` columns.
- Captured normalization parameters in `zscore_metadata.json` and detailed validation metrics in `zscore_validation_report.md`.

## Validation Highlights
- Row preservation: 2 × 2,610 proteins exported; totals match the 5,220-row source file.
- Log2 transforms triggered for both compartments (raw skewness 11–17) before z-scoring.
- Z-score quality (means within ±0.0002, std within ±0.0002 of 1.0) and skewness reduced to |0.15| or less.
- Outlier rates remain <1% of values (`|Z| > 3` counts: Glomerular 39, Tubulointerstitial 45).
- Marker check satisfied: COL1A1, COL1A2, and FN1 present in both outputs.
- No nulls detected in `Zscore_Young` or `Zscore_Old` columns.

## Artifacts
- `06_Randles_z_score_by_tissue_compartment/codex_cli/Randles_2021_Glomerular_zscore.csv`
- `06_Randles_z_score_by_tissue_compartment/codex_cli/Randles_2021_Tubulointerstitial_zscore.csv`
- `06_Randles_z_score_by_tissue_compartment/codex_cli/zscore_metadata.json`
- `06_Randles_z_score_by_tissue_compartment/codex_cli/zscore_validation_report.md`
- `06_Randles_z_score_by_tissue_compartment/codex_cli/zscore_normalization.py`

## Next Steps
1. Review distribution plots or additional QC visualizations if needed for publication-ready figures.
2. Integrate these normalized outputs into downstream cross-study aggregation workflows.
