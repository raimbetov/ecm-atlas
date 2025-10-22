# Plan for Temporal Trajectory Prediction (Agent: codex)

## Objectives
- Model ECM protein dynamics across pseudo-temporal tissue ordering.
- Train LSTM seq2seq and Transformer models to forecast future states and detect transitions.
- Quantify early vs late-changing proteins and evaluate causal influence via Granger tests.
- Deliver required artifacts, metrics, and visualizations per hypothesis instructions.

## Key Steps
1. **Data Audit & Pseudo-Time Construction**
   - Inspect merged dataset schema and completeness.
   - Define pseudo-temporal ordering using tissue velocity proxy (H03) and evaluate alternative metadata if available.
   - Reshape data into protein × time matrices and handle missing values/imputation strategy.
2. **Feature Engineering & Splits**
   - Compute temporal gradients, normalize sequences, and build train/validation/test splits respecting future holdout.
   - Identify tissue subsets suitable for sequence length ≥4 to support multi-step forecasting.
3. **Modeling Pipeline Implementation**
   - Implement modular script (`analysis_temporal_codex.py`) covering dataset loading, preprocessing, and reproducible configuration (seeds, logging).
   - Develop LSTM encoder-decoder with teacher forcing, scheduled sampling, and metrics (MSE, MAE, R²) for 1-step and multi-step predictions.
   - Design Transformer-based time series model (Positional encoding + multi-head attention) to learn transition points; capture attention weights.
4. **Analysis & Interpretability**
   - Rank proteins by gradient magnitude to define early/late quartiles; evaluate regression (R²) predicting late from early proteins.
   - Run Granger causality tests across early→late pairs; summarize significant lags and build causal graph visualization.
   - Perform enrichment tests for coagulation vs structural proteins using Fisher's exact; document methodology.
5. **Evaluation & Visualization**
   - Generate performance tables/CSVs, save model checkpoints, and produce required plots (prediction trajectories, attention heatmaps, causal network).
   - Validate criteria (MSE/R² thresholds, attention-derived transitions) and perform sensitivity checks if metrics fall short.
6. **Documentation & Results Packaging**
   - Compile findings, interpretation, and limitations into `90_results_codex.md`.
   - Ensure all artifacts named per specification and stored in workspace structure (including `visualizations_codex/`).
   - Verify reproducibility: record hyperparameters, random seeds, dependency versions as needed.

## Contingencies
- If pseudo-time ordering fails to produce adequate sequence length, derive alternative ordering via PCA latent trajectory or learned embeddings.
- For sparse proteins, apply dimensionality reduction (e.g., top variance features) or masking to maintain model stability.
- If Transformer training unstable, fall back to temporal convolution + attention hybrid while still extracting attention weights.

## Next Actions
- Implement data audit and pseudo-time generation utilities.
- Scaffold analysis script with configuration and logging placeholders before model development.
