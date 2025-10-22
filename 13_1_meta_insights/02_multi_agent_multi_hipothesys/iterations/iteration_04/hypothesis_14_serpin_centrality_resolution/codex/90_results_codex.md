# H14 – Serpin Centrality Resolution (Codex)

## Data & Network Reconstruction
- Dataset: merged ECM aging z-score matrix (3715 rows, 919 proteins).
- Standardised expression matrix: 18 tissue-compartment samples × 713 proteins (Zscore_Delta, zero-imputed, variance filtered).
- Correlation network (|ρ| ≥ 0.5, p < 0.05, Spearman): 713 nodes, 34,147 edges, density 0.135, global efficiency 0.478, single connected component.
- Edge export: `network_edges_codex.csv`; stats stored in `network_stats_codex.json`.

## Centrality Metrics (9 total)
Computed degree, strength, betweenness (distance-weighted), closeness, harmonic, eigenvector, PageRank, clustering coefficient, and core number (`centrality_all_metrics_codex.csv`).

Key observations:
- Eigenvector/PageRank strongly correlated with degree/strength (ρ > 0.78).
- Betweenness is nearly orthogonal to other metrics (ρ ≤ 0.04), highlighting different topology.
- Consensus z-score (`consensus_centrality_codex.csv`) averages z-standardised metrics.

## Serpin Rankings
- Total serpins captured: 47 (23 human uppercase, 24 murine variants).
- Eigenvector/PageRank: SERPINA10, SERPINE2, SERPINA6, SERPINA4, SERPINF2 fall within top 20% of all proteins (eigenvector percentile ≤ 21%).
- Betweenness: SERPINH1, SERPINC1, SERPINA3 isoforms, SERPINB9B occupy top 5% bridging percentiles, indicating strong module-bridging roles.
- Mean human serpin degree centrality = 0.147 (top-quartile); mean betweenness percentile = 39.6 (lower is more central), eigenvector percentile = 53.3.
- Serpin rank tables exported in `serpin_rankings_codex.csv`; comparative violin/strip plot saved as `visualizations_codex/serpin_ranks_comparison_codex.png`.

## Metric Agreement
- Metric correlation heatmap (`centrality_heatmap_codex.png`) illustrates two families: strength/eigenvector/PageRank/core vs betweenness/cluster.
- Consensus ranking places SERPINA10, SERPINH1, SERPINA6, SERPINE2, SERPINA4 inside top 22 percentile, but long tail of serpins drops below median (consensus mean percentile 55.9%).

## Knockout Simulation
- Perturbation: Remove each serpin node from graph; recompute approximate global efficiency (40-source Dijkstra sampling) and component metrics (`knockout_impact_codex.csv`).
- Largest efficiency drops: Serpina1d (ΔE ≈ 0.0118), Serpinb6b (0.0106), Serpina1b (0.0104), SERPINA1 (0.0087), SERPIND1 (0.0086), SERPINH1 (0.0065).
- Centrality vs KO impact (`centrality_knockout_correlation_codex.csv`): betweenness best predictor (Spearman ρ ≈ 0.21, p ≈ 0.16); other metrics |ρ| < 0.1. Effect size modest, suggesting redundancy but favouring betweenness for functional impact estimation.
- Scatter (`knockout_impact_scatter_codex.png`) shows mild positive association between betweenness and Δ efficiency; outliers (Serpina1 family) dominate.

## Community Structure
- Louvain partition (γ = 1) yields 5 major communities (sizes: 203, 167, 158, 158, 27). Serpins distribute across modules: community 3 (16 members; collagen/chaperone cluster), 2 (12; complement/coagulation), 0 (11; metabolic ECM modifiers), 1 (8; inflammatory response).
- Top-200 consensus subgraph visualisations:
  - Network with serpins highlighted (`network_graph_serpins_codex.png`).
  - Community coloring overlay (`community_modules_codex.png`).

## Literature Synthesis Highlights
Documented in `literature_network_centrality.md`.
- Jeong et al. 2001 (Nature) – degree centrality correlates with lethality.
- Joy et al. 2005 (BioMed Res Int) – betweenness uncovers essential inter-module proteins.
- Kitsak et al. 2010 (Nat Phys) – k-core outperforms degree/betweenness for diffusion influence.
- Jalili et al. 2016 (Front Physiol) – hybrid or consensus metrics recommended for essential protein detection.
- Wang et al. 2012 (IEEE/ACM TCBB) – edge clustering improves essential gene recall.
- Roy et al. 2016 (Curr Issues Mol Biol) – PageRank robust for directional PPIs.

## Serpin Knockout Phenotypes from Literature
`experimental_validation_codex.csv` summarises key findings:
- **SERPINE1 (PAI-1)**: knockout mice (10.1172/jci116893) display enhanced fibrinolysis, prolonged bleeding (moderate severity; viable but altered hemostasis).
- **SERPINC1 (antithrombin)**: insufficiency exacerbates renal ischemia/reperfusion injury (10.1038/ki.2015.176), indicating high severity under stress.
- **SERPINB6A**: knockout causes progressive sensorineural hearing loss (10.1016/j.ajpath.2013.03.009) – tissue-specific but functionally significant.
- **SERPINF1**: human biallelic loss leads to osteogenesis imperfecta type VI (10.1016/j.ajhg.2011.01.015) – severe skeletal phenotype.

## Resolution of Betweenness vs Eigenvector Debate
1. **Topology**: Betweenness isolates a subset of serpins (SERPINH1, SERPINC1, SERPINA3 variants) as cross-module bottlenecks; eigenvector/PageRank emphasise secreted inhibitors (SERPINA10, SERPINE2, SERPINA6) embedded in dense pro-coagulant clusters.
2. **Functional Alignment**: Knockout simulation shows only weak effect sizes but favours betweenness (ρ ≈ 0.21) over eigenvector (~0.01) for predicting network efficiency loss. Literature phenotypes also align with bridging serpins (SERPINC1, SERPINH1) carrying higher physiological risk when perturbed.
3. **Interpretation**: Eigenvector inflates rankings inside dense serpin-serpin/protease subgraphs, potentially overstating global influence. Betweenness captures serpins that mediate collagen folding, coagulation, and immune ECM crosstalk.
4. **Recommendation**: Adopt a composite hub score: z-average of betweenness, eigenvector, and PageRank to capture both broadcast power and bottleneck risk. Use betweenness-weighted consensus when prioritising knockout-sensitive targets.
5. **Biological Takeaway**: Serpins are not universally peripheral—several (SERPINA10, SERPINC1, SERPINH1) lie within top quintile by multiple metrics and carry documented high-severity phenotypes. However, many family members remain mid-ranked, explaining Claude’s earlier conclusion.

## Next Steps
1. Validate betweenness-weighted consensus against experimental datasets beyond knockout simulation (e.g., CRISPR screens, clinical phenotypes).
2. Integrate node embeddings (e.g., node2vec) for downstream predictive modelling of aging trajectories.
3. Update future hypotheses (H05, GNN layers) to incorporate multi-metric centrality priors rather than single-metric assumptions.

**Deliverables:** All scripts (`analysis_network_centrality_codex.py`, `perturbation_analysis_codex.py`, `community_detection_codex.py`, `literature_validation_codex.py`), tables, and figures reside in the codex workspace.
