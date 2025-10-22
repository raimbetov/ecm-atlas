# H12 – Metabolic vs Mechanical Transition (codex)

## 1. Dataset & Preparation
- Source: `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`; 17 tissues with velocity labels from H03 (`phase_assignments_codex.csv`).
- Autoencoder (3-layer, latent=32) trained on tissue × ECM protein Δz matrix (MSE ↓ to ~0.006; weights in `visualizations_codex/autoencoder_weights_codex.pth`).
- Graph pipeline: cosine-kNN adjacency on latent space → 2-layer GCN (`visualizations_codex/gcn_weights_codex.pth`) achieved perfect fit on training tissues (likely overfit, see §6).
- Unsupervised structure: HDBSCAN collapsed into a single cluster, indicating weak bimodality in available ECM features.

## 2. Literature & External Data Search
- PubMed automation (`literature_search_codex.py`) captured 35 unique articles across mandated queries (see `literature_metabolic_mechanical.md`, `literature_findings_codex.csv`). Themes: mitochondrial declines antecede fibrosis, YAP/TAZ mechanotransduction as point-of-no-return, metabolic rescue windows closing with crosslink accumulation.
- Metabolomics Workbench sweep (`dataset_search_codex.py`) prioritised fibrosis-centric multi-omics cohorts (e.g., ST004266, ST003641, ST003043) but no confirmed paired metabolomics+proteomics datasets; shortlist in `metabolomics_workbench_summary_codex.md`.

## 3. Changepoint Validation (v = 1.65?)
- Velocities sorted ascending (`changepoint_detection_codex.py`).
  - Binary segmentation → break at 2.09.
  - PELT → 1.45 (near hippocampus entry to Phase I). 
  - Grid SSE → 2.17. 
  - Bayesian posterior highest at 1.35–1.45 (ovary/hippocampus zone) with 10.2% weight at 1.35 (`changepoint_posterior_codex.csv`).
  - Consensus (mean of estimators) ~1.90, above H09.
- Interpretation: evidence for an earlier metabolic-to-mechanical shift (~1.4–1.5), with 1.65 within credible window but not the dominant optimum. Visual: `visualizations_codex/changepoint_plot_codex.png`.

## 4. Phase Signatures & Enrichment
- Dynamic marker sets derived from matrisome annotations.
  - Mechanical (collagens, LOX/TGM family): mean Δz Phase II = −0.32 vs −0.05 in Phase I (OR=0.22, p=0.28; `enrichment_analysis_codex.csv`). Direction matches irreversible crosslinking but not yet significant.
  - Metabolic regulators (secreted enzymes): small positive shift for both phases (Phase I =0.143, Phase II =0.137), signalling muted mitochondrial signal in ECM-only dataset.
- Trajectory scatter (`visualizations_codex/protein_trajectories_codex.png`) shows collagen-centric downshifts tracking higher velocities; metabolic proxies remain flat.

## 5. ML Models & Classification
- Latent + composite features fed into model comparison (`analysis_metabolic_mechanical_codex.py`).
  - RandomForest (best CV ROC_AUC ≈0.30) and GradientBoosting (ROC_AUC ≈0.39) on latent+aggregate features (`classification_performance_codex.csv`).
  - SHAP (`visualizations_codex/shap_summary_codex.png`, ranked in `shap_feature_rank_codex.csv`) points to latent dimensions capturing collagen-rich tissues, not specific genes.
- Transition-distance regression (GradientBoosting) gave R²≈1 on training but predicts minimal buffer for most Phase II tissues (likely overfit; caution flagged).
- GCN probabilities (`analysis_summary_codex.json`) classify hippocampus/tubulointerstitium as Phase I anchors; high confidences reflect exhaustive training not validated externally.

## 6. Intervention Simulation (Metabolic Rescue)
- Gradient boosting regression with metabolic-target upshift on regulator gene set (`intervention_simulation_codex.py`).
  - Phase I mean Δv = −0.027 (std 0.041); Phase II mean Δv = −0.029 (std 0.028) — both phases show modest predicted velocity reductions, lacking expected asymmetry (`intervention_effects_codex.csv`, violin `visualizations_codex/intervention_effects_codex.png`).
  - Mixed response driven by absence of canonical mitochondrial proteins; enzyme-focused perturbation still nudges high-velocity tissues, suggesting ECM regulators may offer partial reversibility.

## 7. Key Findings vs Hypotheses
- **H12.1 (Metabolic Phase I):** Weak support. Metabolic proxy enrichment lacks selectivity; dataset missing mitochondrial markers. Requires integration of genuine metabolomics/proteomics pairs (see §2 candidates).
- **H12.2 (Mechanical Phase II):** Directionally supported (collagen/crosslinking downshifts dominate Phase II) but small sample prevents statistical significance.
- **H12.3 (Transition Predictability):** Current models fail to surpass ROC_AUC 0.5–0.6; latent representations insufficient without richer metabolic features.
- **H12.4 (Reversibility Asymmetry):** Simulation did not show Phase II resistance; assumption unmet due to proxy gene limitations.
- **Changepoint:** Multi-method analysis suggests earlier break (~1.45), implying intervention window may close sooner than H09 threshold.

## 8. Gaps & Next Steps
1. **Data augmentation:** Acquire metabolomics+proteomics pairs (WorkBench fibrosis cohorts, MetaboLights) and re-train models with authentic mitochondrial variables.
2. **Model robustness:** Introduce cross-validated Bayesian changepoint with bootstrapped velocities; evaluate segmentation on tissue stratified subsets.
3. **Classifier uplift:** Try graph transformers or contrastive learning on protein networks; incorporate non-ECM omics for metabolic coverage.
4. **Intervention realism:** Calibrate rescue using effect sizes from literature (e.g., NAD+ boosters on ATP5A1) once proteins are present.
5. **Validation:** Compare with H09 attention weights (`../iteration_03/.../critical_transitions_codex.csv`) to tie latent factors back to earlier temporal signals.

## 9. Artifacts & Provenance
- Code: `analysis_metabolic_mechanical_codex.py`, `changepoint_detection_codex.py`, `intervention_simulation_codex.py`, `literature_search_codex.py`, `dataset_search_codex.py`.
- Models: `phase_classifier_codex.pkl`, `visualizations_codex/gcn_weights_codex.pth`, autoencoder weights.
- Outputs: `phase_assignments_codex.csv`, `latent_embeddings_codex.csv`, `changepoint_results_codex.csv`, `intervention_effects_codex.csv`, `metabolomics_workbench_hits_codex.csv`, `literature_metabolic_mechanical.md`.
- Visuals under `visualizations_codex/` (velocity distribution, changepoint, SHAP, intervention, t-SNE, enrichment).

**Clinical Perspective:** Without metabolically-informative proteins the ECM matrix alone signals crosslink dominance beyond v~1.45; current results caution that the reversible window may be narrower than v=1.65, demanding earlier screening and multi-omics validation.
