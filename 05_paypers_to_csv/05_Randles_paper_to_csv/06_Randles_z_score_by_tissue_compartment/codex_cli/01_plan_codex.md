# Plan: Z-Score Normalization by Tissue Compartment

1. **Review inputs & context**  
   - Inspect `Randles_2021_wide_format.csv` structure, column names, null counts, compartment balance.  
   - Confirm instructions/quality gates from task doc and any prior metadata.

2. **Design normalization workflow**  
   - Decide column subset to propagate to outputs and naming for derived metrics (e.g., `Zscore_Young`, `Zscore_Old`, `Zscore_Delta`).  
   - Define log2 transformation rule: evaluate skewness per compartment/age column and flag if |skew| > 1 before z-scoring.  
   - Outline validation metrics (means, std, skewness, outlier share, marker presence) and metadata schema.

3. **Implement processing script**  
   - Build reusable pandas script/notebook that: loads data, splits by compartment, applies conditional log2(x+1), computes per-compartment z-scores for Young/Old, derives delta, and tracks summary stats.  
   - Capture normalization parameters (means, stds, skewness, log flags) for export; ensure no NA propagation.

4. **Execute & generate artifacts**  
   - Run script, produce `Randles_2021_Glomerular_zscore.csv`, `Randles_2021_Tubulointerstitial_zscore.csv`, `zscore_metadata.json`.  
   - Assemble `zscore_validation_report.md` with tabulated metrics, outlier counts, and marker checks.  
   - Spot-check outputs (row counts, distributions) against success criteria.

5. **Document findings & next steps**  
   - Summarize workflow, key statistics, and validation outcomes in `90_results_codex.md`.  
   - Note any assumptions, open questions, or recommended follow-up actions.
