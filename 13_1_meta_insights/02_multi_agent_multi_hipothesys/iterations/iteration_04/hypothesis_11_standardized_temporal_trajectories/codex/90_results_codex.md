# H11 – Standardized Temporal Trajectories (Codex)

## 1. Executive Summary
- Recomputed pseudo-time orderings for 17 ECM tissues using four methods (velocity, PCA, diffusion maps, Slingshot-inspired MST).  
- All orderings stored in `pseudotime_orderings_codex.csv`; Slingshot required a fallback MST principal-graph approximation because Bioconductor dependencies (nloptr) could not be compiled in the current environment.
- LSTM forecasting on each pseudo-time yielded modest R² values (≤0.03). None of the methods reproduced Claude’s H09 >0.8 R²; velocity performed marginally best but still near-random, confirming that pseudo-time choice alone does not recover high predictability.
- Literature review highlights consensus best practices: diffusion kernels + graph smoothing, ensemble trajectory checks, and uncertainty quantification.  `literature_pseudotime.md` captures the synthesis; `literature_recommendations_codex.csv` logs five cornerstone references.
- Longitudinal dataset reconnaissance identified two actionable resources for external validation: PRIDE PXD056694 (human calorie restriction trial, two timepoints) and ENA PRJNA1185831 (controlled-access multi-omic immune aging cohort with annual sampling). These are catalogued in `longitudinal_dataset_candidates_codex.csv` with suitability notes.

## 2. Pseudo-time Method Comparison
- **Velocity (H03)** reproduces the expected lung→fast muscle ordering; alias mapping was required for tissues with different naming conventions (`Kidney_Glomerular`→`Glomerular`, etc.).  
- **PCA (PC1)** pushes the three intervertebral disc regions to the latest positions, diverging sharply from velocity-based rankings.  
- **Diffusion Map** emphasises disc tissues as earliest, skeletal muscle mid-trajectory, lung/skin late. Implementation uses `pydiffmap` Gaussian kernel with automatic bandwidth.  
- **Slingshot (fallback)** constructs an MST over PCA coordinates; ordering largely mirrors diffusion early branching but with broader spacing. `fallback_used=True` flags the approximation.
- Visualization: `visualizations_codex/pseudotime_comparison_codex.png` overlays normalized pseudo-time curves per method.

## 3. LSTM Benchmarking
- Uniform seq2seq LSTM (2-layer, hidden=128, input window=4, forecast horizon=2) trained per method using 60/20/20 sliding windows across tissue orderings.  
- Results (`lstm_performance_codex.csv`): R² ranges from -0.02 to 0.03, MAE ≈0.17–0.27 across horizons.  Diffusion and Slingshot underperform velocity slightly.  
- `visualizations_codex/lstm_performance_codex.png` summarises R² by method/horizon.  
- Conclusion: pseudo-time differences alone do not explain Claude’s strong H09 performance; additional preprocessing, architecture tweaks, or leakage must have contributed previously.

## 4. Literature Highlights (see `literature_pseudotime.md`)
- Saelens et al. 2019 and Haghverdi et al. 2016 emphasise diffusion kernels + ensemble validation.  
- Street et al. 2018 recommends MST+principal curves with bootstrapped lineage confidence.  
- Bergen et al. 2020 advocates combining velocity fields with diffusion embeddings to stabilise temporal direction.  
- Campbell & Yau 2018 motivates probabilistic uncertainty reporting for pseudo-time.

## 5. Longitudinal Dataset Reconnaissance (`longitudinal_dataset_candidates_codex.csv`)
- **PXD056694 (PRIDE)**: Plasma DIA proteomics pre/post 8-week calorie restriction in 44 prediabetic adults; provides ground-truth intervention timeline.  
- **PRJNA1185831 (ENA/dbGaP)**: Annual multi-omic immune profiling of 96 adults over two years plus cross-sectional cohort; proteomics accessible via controlled access.  
- **PXD009160 (PRIDE)**: Age-graded BXD mouse livers (cross-sectional). Useful as quasi-time-course but lacks repeated measures.

## 6. Root Cause Assessment for H09 Disagreement
- Reproducing Codex PCA-based pseudo-time confirmed strong divergence from velocity ordering; however, even Claude’s velocity sequence fails to deliver high R² under the stricter training pipeline used here.  
- Likely contributing factors for the earlier 0.81 R²: (i) smaller model evaluating on validation rather than held-out tissues; (ii) potential leakage from using future windows in training; (iii) aggressive smoothing or feature selection differences not captured in current code.  
- Standardisation recommendation: adopt diffusion-map or MST-based ordering only if validated against true longitudinal cohorts (e.g., PXD056694). Without external validation, pseudo-time should be treated as exploratory.

## 7. Gaps & Next Steps
- Sensitivity analyses (tissue/protein dropout, noise injection) and attention-based transition comparisons remain outstanding.  
- R installation hurdles prevented execution of the official Slingshot and DESTINY pipelines; revisit after resolving `nloptr` shared library issues.  
- Integrate longitudinal datasets once access is approved to correlate pseudo-time with real months/years and retrain LSTMs under matched conditions.

## Artifacts
- Pseudo-time results: `pseudotime_orderings_codex.csv`
- LSTM metrics & samples: `lstm_performance_codex.csv`, `intermediate/lstm_samples_codex.json`
- Visuals: `visualizations_codex/pseudotime_comparison_codex.png`, `visualizations_codex/lstm_performance_codex.png`
- Literature: `literature_pseudotime.md`, `literature_recommendations_codex.csv`
- Dataset scouting: `longitudinal_dataset_candidates_codex.csv`
- Scripts: `analysis_pseudotime_comparison_codex.py`, `lstm_benchmark_codex.py`, `literature_search_pseudotime_codex.py`, `diffusion_pseudotime_codex.R`, `slingshot_trajectory_codex.R`
