# H05 Master Regulator Discovery — Execution Plan (Agent: codex)

## Objectives
- Build ECM protein interaction graph from merged aging z-score dataset.
- Train attention-enabled GNN (GAT) for protein aging state classification and embedding generation.
- Derive master regulator rankings via attention, gradients, and PageRank; quantify perturbation cascades.
- Compare GNN-derived communities against Louvain baseline and document insights per Advanced ML requirements.

## Work Packages
1. **Data Engineering & Graph Construction**
   - Aggregate protein-level statistics (Δz mean/std, tissue breadth, matrisome encodings).
   - Compute Spearman correlation network (|ρ| > 0.5); export `network_graph_codex.graphml`.
   - Prepare node labels (up/down/stable) and train/val/test masks.
2. **GNN Training Pipeline**
   - Implement multi-layer GAT with dropout and weight decay.
   - Run stratified training loop with early stopping; log loss, accuracy, F1 per epoch.
   - Save checkpoints (`gnn_weights_codex.pth`), embeddings (`protein_embeddings_gnn_codex.csv`), and metrics (`gnn_training_metrics_codex.csv`).
3. **Master Regulator Analytics**
   - Summarize incoming attention weights, gradient saliency, and embedding PageRank.
   - Produce `master_regulators_codex.csv` (top 10) and `perturbation_analysis_codex.csv` (cascade magnitude).
   - Generate attention heatmap visualization.
4. **Community & Embedding Evaluation**
   - Louvain baseline communities vs. HDBSCAN on embeddings; compute ARI, silhouette, matrisome purity.
   - Create UMAP, community comparison plots; store under `visualizations_codex/` (UMAP, attention heatmap, community comparison, network snapshot).
5. **Knowledge Synthesis**
   - Summarize methodology, metrics, key discoveries in `90_results_codex.md`.
   - Highlight compliance with ML requirements and future directions.

## Dependencies & Tooling
- Libraries: `pandas`, `numpy`, `scipy`, `networkx`, `torch`, `torch_geometric`, `scikit-learn`, `umap-learn`, `hdbscan`, `matplotlib`, `seaborn`.
- Reproducibility: set random seeds; persist intermediate tables for reuse.

## Risks & Mitigations
- **Sparse correlations:** relax threshold or add k-NN backup if graph becomes disconnected.
- **Class imbalance:** apply class weights / focal loss; monitor per-class F1.
- **Attention extraction:** capture α coefficients from PyG GAT layers; validate shapes before aggregation.

## Deliverable Checklist
- [ ] Graph: `network_graph_codex.graphml`
- [ ] Script: `analysis_gnn_codex.py`
- [ ] Model: `gnn_weights_codex.pth`
- [ ] Embeddings: `protein_embeddings_gnn_codex.csv`
- [ ] Training metrics: `gnn_training_metrics_codex.csv`
- [ ] Master regulators: `master_regulators_codex.csv`
- [ ] Perturbation analysis: `perturbation_analysis_codex.csv`
- [ ] Visuals: `visualizations_codex/*.png`
- [ ] Summary: `90_results_codex.md`
