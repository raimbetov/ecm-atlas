# Temporal Trajectory Prediction Results (Agent: codex)

## Data & Pseudo-time Construction
- Dataset: merged_ecm_aging_zscore.csv (3715 rows, 17 tissues).
- Pseudo-temporal ordering derived from PCA (PC1) on tissue-level ΔZ profiles.
- Ordered tissues (early→late): Skeletal_muscle_Gastrocnemius → ... → Intervertebral_disc_IAF (see analysis_metadata_codex.json).
- Protein × pseudo-time matrix built with tissue-median imputation for missing ΔZ entries.

## Sequence Forecasting Performance
- Trained encoder-decoder LSTM (2×128 hidden, dropout 0.2) and 2-layer Transformer (d_model 96, 4 heads, dropout 0.2) on sliding windows (input=4, forecast=3) with leave-future-out splits (train≤t10, val≤t13, test>t13).
- Test-set metrics (forecast horizon 1/2/3):
  - LSTM MSE: 0.381 / 0.400 / 0.370; R²: 0.011 / −0.001 / −0.001.
  - Transformer MSE: 0.406 / 0.405 / 0.381; R²: −0.053 / −0.015 / −0.031.
- Target of MSE<0.3 / R²>0.6 unmet, likely due to short sequences (17 pseudo-steps) and high tissue heterogeneity.
- Top 10% predictable proteins (lowest mean MSE): SEMA3G, LAMA3, CCL14, PDGFC, COL4A5, S100B, FBN2, COL16A1, SFRP4, … (top_predictable_proteins_codex.csv).

## Early vs Late Change Proteins
- Gradient-based transition index divided proteins into quartiles (early n=228, late n=228).
- Regression (early mean → late mean) across lags 1–3 produced poor future-fit (best test R² ≈ −3.42, p≈0.35), falling short of ≥0.70 requirement.
- Enrichment:
  - Coagulation proteins depleted in early set (odds=0, p=0.047).
  - Collagens enriched among late changers (odds=3.94, p=4.0×10⁻⁵).

## Transformer Attention & Critical Transitions
- Attention concentrated at Ovary_Cortex and Heart_Native_Tissue (top 10% weights; critical_transitions_codex.csv).
- Paired gradient comparison around these steps shows significant slope shift (t=−11.49, p=1.3×10⁻²⁸), indicating acceleration post-transition.

## Granger Causality
- Tested 100 early→late pairs (lags 1–3). No edges achieved p<0.05; causal_network_codex.png notes absence of significant causal arrows.

## Artifacts Produced
- Models: lstm_seq2seq_model_codex.pth, transformer_model_codex.pth.
- Tables: prediction_performance_codex.csv, early/late protein lists, regression & enrichment stats, attention_weights_per_timestep_codex.csv, granger_causality_codex.csv, top_predictable_proteins_codex.csv, critical_transitions_codex.csv, analysis_metadata_codex.json.
- Visualizations (visualizations_codex/): prediction_performance_codex.png, attention_heatmap_codex.png, trajectory_plot_codex.png, causal_network_codex.png.

## Limitations & Next Steps
- Performance goals unmet; consider richer pseudo-time (longer observable trajectory) or multi-task forecasting to stabilise training.
- Explore dimensionality reduction (e.g., protein modules) prior to RNN to reduce noise.
- Investigate transfer learning (ProtBERT/ESM embeddings) and regularisation (dropout scheduling, teacher forcing decay) to enhance generalisation.
- Revisit causal analysis with smoothed/aggregated signals or VAR models to boost statistical power.
