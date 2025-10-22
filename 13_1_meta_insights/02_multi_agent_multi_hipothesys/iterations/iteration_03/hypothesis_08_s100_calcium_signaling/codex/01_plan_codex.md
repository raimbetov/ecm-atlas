# Plan (Agent: codex)

## Objective
Resolve the S100 paradox by demonstrating a calcium signaling → ECM crosslinking pathway (independent of inflammation) using deep learning, attention, network analysis, and structural transfer learning on the merged ECM aging dataset.

## Key Tasks
1. **Data Assembly** – Load merged ECM dataset, pivot to tissue-level matrices for S100, crosslinking, mechanotransduction, and inflammation panels; define stiffness proxies.
2. **Stiffness Modeling** – Train and evaluate an S100-based deep neural network predicting stiffness score; record metrics, export predictions, correlations vs inflammation.
3. **Network Discovery** – Compute S100 ↔ crosslinking Spearman network, compare with inflammation, train attention model to extract relationship weights, render heatmap.
4. **Mechanotransduction Enrichment** – Quantify S100 co-expression with mechanotransduction genes, perform Fisher tests vs inflammation, visualize cascade network.
5. **Structural Transfer Learning** – Download AlphaFold/ESM resources for S100A8/A9/B, derive embeddings, fine-tune model against aging data, benchmark vs expression-only baseline.
6. **Reporting** – Save required artifacts (models, CSVs, visuals) and synthesize findings into `90_results_codex.md` with interpretation and validation notes.

## Status
- Data assembly & preprocessing ✅
- Deep stiffness model + diagnostics ✅
- Correlation & attention networks ✅
- Mechanotransduction enrichment + visualization ✅
- AlphaFold/ESM transfer learning workflow ✅ (structural model underperformed vs expression baseline)
- Results synthesis 🚧 (pending `90_results_codex.md`)

## Tooling & Checks
- PyTorch + attention modules, NetworkX for network graph, seaborn/matplotlib for plots.
- Spearman correlations, paired tests, Fisher exact via SciPy.
- Maintain reproducibility (random seeds, checkpoints) and document hyperparameters in results summary.
