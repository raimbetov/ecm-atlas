# Plan – Hypothesis 07 Coagulation Hub (Agent: codex)

## Objectives
- Quantify whether coagulation proteins alone explain tissue aging velocity.
- Demonstrate temporal precedence of coagulation dysregulation relative to ECM remodeling.
- Re-run network centrality analysis to test coagulation dominance.
- Satisfy advanced ML requirements (deep NN, LSTM, transfer learning, SHAP, network analytics).

## Workstreams
1. **Data Engineering & Targets**
   - Load merged ECM dataset; curate coagulation, serpin, collagen panels.
   - Derive tissue-level aging velocity, coagulation state labels, and pseudo-temporal ordering (slow→fast aging).
   - Cache metadata (protein modules, embeddings, sequences) for reuse.

2. **Deep NN + Transfer Learning Regression**
   - Build PyTorch MLP (128-64-32-16) for coagulation-only features.
   - Inject pre-trained protein embeddings (ESM/ProtBERT) as learnable context → fine-tune during training.
   - Perform 5-fold tissue-level CV, log R²/MAE/RMSE, save best checkpoint and predictions.
   - Benchmark against full-protein baseline to compute relative performance.

3. **Interpretability & State Analysis**
   - Compute SHAP values for the trained coagulation MLP; export summary plot.
   - Quantify hypercoagulable vs hyperfibrinolytic states; correlate with aging velocity; scatter visualization.

4. **Temporal Precedence (LSTM)**
   - Construct pseudo-time sequences via aging-velocity ordering.
   - Train LSTM to forecast downstream ECM (collagen/MMP) signatures from coagulation trajectories.
   - Extract early-change proteins (top quartile lead times) and visualize temporal trajectories.

5. **Network Centrality Re-Analysis**
   - Build protein correlation network (coagulation, serpins, collagens).
   - Compute betweenness centrality, community structure, and compare module statistics.
   - Generate network visualization highlighting coagulation hubs.

6. **Results Assembly**
   - Export required CSVs, model checkpoints, and plots under specified filenames.
   - Summarize findings in `90_results_codex.md` following Knowledge Framework template.

## Status Update
- Data engineering, models, interpretability, temporal analysis, and network re-analysis completed via `analysis_coagulation_codex.py` (all deliverables generated in `codex/`).
