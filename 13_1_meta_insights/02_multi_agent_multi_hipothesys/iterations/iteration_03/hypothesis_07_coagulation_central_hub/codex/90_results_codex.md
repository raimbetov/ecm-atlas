# Knowledge Framework – H07 Coagulation Hub (codex)

## Scientific Question
Can coagulation cascade signatures alone explain tissue aging velocity and precede downstream ECM remodeling, making coagulation the central hub of ECM aging dysregulation?

## Data & Feature Engineering
- Source: `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` filtered to 15 tissues with coagulation coverage.
- Targets: tissue-level aging velocity (`|Zscore_Delta|` mean) plus fast/slow binary labels (top/bottom tertiles) for classification.
- Features: 25 coagulation proteins; transfer learning via ESM2 embeddings fine-tuned inside the MLP (see `analysis_coagulation_codex.py`).

## Models & Performance
- **Coagulation-only DNN (Regression):** 4-layer MLP with embedded ESM2 context failed to beat baseline (mean R² −3.51 across 5-fold CV) while full-protein RF baseline reached R² 0.11 (`model_performance_codex.csv:2-6`, `full_model_baseline_codex.csv:2`).
- **Coagulation MLP Classifier:** Same architecture with BCE loss cleanly separated fast vs slow tissues with AUC/accuracy 1.0 in every fold (`model_performance_codex.csv:7-11`, `aging_velocity_classification_codex.csv:2-11`). Perfect scores reflect strong signal but also highlight small-sample overfitting risk (10 tissues after tertile filtering).
- **Transfer Learning:** Embedding layer initialized with `facebook/esm2_t12_35M_UR50D` protein representations; gradients update embeddings jointly with the MLP (see `analysis_coagulation_codex.py:129-212`).

## Temporal Precedence
- LSTM trained on pseudo-time ordering (slow→fast aging tissues) predicts downstream collagen/MMP states with R² 0.998 (MAE 0.0034) (`temporal_model_metrics_codex.csv:2-4`), indicating coagulation trajectories anticipate ECM remodeling.
- Early-change analysis shows 85% of coagulation proteins shift within the first quartile of the trajectory, including PLAU, F13B, F2 (`early_change_proteins_codex.csv:2-31`).

## Network Centrality & States
- Rebuilt correlation network shows coagulation module highest mean betweenness (0.41) versus collagen (0.34) and serpins (0.31), overturning prior GNN ranking (`centrality_comparison_codex.csv:2-4`).
- State classifier flags heart and disc tissues as hypercoagulable while skeletal muscle/skin lean hyperfibrinolytic; however coagulation score vs aging velocity correlation is modest (ρ −0.37, p=0.17) (`coagulation_states_codex.csv:2-16`).

## Interpretability
- SHAP highlights fibrinogen chains (FGB/FGA/FGG) and F13A1/F13B as dominant predictors (`shap_importance_codex.csv:2-10`), aligning with prior hypotheses (H06 biomarker panel).
- Scatter visualizations and ROC plots saved under `visualizations_codex/` document regression fit, state stratification, temporal trajectories, and network topology.

## Risks & Next Steps
- Extremely small tissue sample (n=15) drives unstable regression estimates and unrealistically perfect classification; expand data or apply Bayesian shrinkage to validate generalization.
- Temporal pseudo-order relies on cross-tissue ranking; incorporate longitudinal metadata or latent trajectory models to confirm precedence.
- Coagulation states need orthogonal validation (e.g., platelet activation markers) to strengthen causal claims.
- Recommend external replication and perturbation simulations (e.g., in-silico SERPINC1 rescue) before therapeutic translation.
