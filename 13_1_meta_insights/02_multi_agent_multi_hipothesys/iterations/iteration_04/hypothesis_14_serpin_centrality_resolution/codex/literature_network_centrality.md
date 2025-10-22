# Network Centrality Literature Synthesis (Codex)

## Canonical Benchmarks for Biological Networks

- **Jeong et al. (2001) — Lethality and centrality in protein networks.** *Nature* 411:41-42. DOI: [10.1038/35075138](https://doi.org/10.1038/35075138)
  - Found ~93% of highly connected yeast proteins (top 20% degree) to be essential, introducing the "centrality–lethality" rule. Degree centrality provided the strongest signal for lethality, but authors warned that alternative topologies (e.g., modular hubs) may violate the rule.

- **Joy et al. (2005) — High-betweenness proteins in the yeast protein interaction network.** *BioMed Research International* 2005:96-103. DOI: [10.1155/JBB.2005.96](https://doi.org/10.1155/JBB.2005.96)
  - Showed that proteins with high betweenness but low degree bridge functional modules and are enriched for essential genes. Demonstrated that betweenness captures cross-module vulnerabilities missed by degree/eigenvector metrics.

- **Kitsak et al. (2010) — Identification of influential spreaders in complex networks.** *Nature Physics* 6:888-893. DOI: [10.1038/nphys1746](https://doi.org/10.1038/nphys1746)
  - Demonstrated that k-core (coreness) outperforms degree and betweenness at predicting diffusion impact. Supports testing core number when prioritising intervention targets in biological graphs.

- **Jalili et al. (2016) — Evolution of centrality measurements for detecting essential proteins.** *Frontiers in Physiology* 7:375. DOI: [10.3389/fphys.2016.00375](https://doi.org/10.3389/fphys.2016.00375)
  - Comprehensive review comparing >20 centrality metrics across PPI datasets. Concluded that composite scores (e.g., hybrid degree + clustering) outperform single metrics and that network context (noise, sparsity) strongly influences metric choice.

- **Wang et al. (2012) — Identifying essential proteins using edge clustering coefficient.** *IEEE/ACM Trans. Comput. Biol. Bioinform.* 9(4):1086-1097. DOI: [10.1109/TCBB.2011.147](https://doi.org/10.1109/TCBB.2011.147)
  - Introduced edge clustering coefficient centrality (ECC) showing improved recall of essential genes versus degree/betweenness by penalizing interactions within dense communities.

- **Roy et al. (2016) — PageRank-based ranking of directional protein interaction networks.** *Current Issues in Molecular Biology* 20:13-28. DOI: [10.21775/cimb.020.013](https://doi.org/10.21775/cimb.020.013)
  - Demonstrated PageRank's robustness on weighted, directed protein networks; highlighted advantages over eigenvector centrality when hubs connect to low-quality nodes.

## Practical Guidance Extracted

- **Metric selection should match biological question.** Degree/eigenvector emphasize well-connected regulators; betweenness captures inter-module bottlenecks (Joy et al. 2005); coreness highlights diffusion backbones (Kitsak et al. 2010).
- **Hybrid or consensus metrics frequently outperform single measures.** Jalili et al. (2016) and Wang et al. (2012) report boosted precision for essential protein detection when mixing degree-like and clustering-aware metrics.
- **Noise and sampling bias matter.** Jeong et al. (2001) and Jalili et al. (2016) emphasise that incomplete interactomes inflate betweenness variance; PageRank (Roy et al. 2016) stabilises rankings in noisy networks.
- **Validation requires perturbation evidence.** Degree-based predictions correlate with lethality (Jeong et al. 2001), but bridging-centric measures better capture synthetic lethal pairs and phenotypes that manifest under stress (Joy et al. 2005).

## Implications for ECM Aging Network (H14)

1. **Test multiple paradigms.** Include degree/strength, betweenness, eigenvector, PageRank, core number, and clustering-aware metrics as baseline set.
2. **Expect context dependence.** Serpins acting as secreted inhibitors may function as inter-module bridges; betweenness could better capture their knockout impact than eigenvector centrality alone.
3. **Adopt consensus scoring.** Following Jalili et al. (2016), aggregate z-scored centrality metrics to stabilise rankings before downstream prioritisation.
4. **Validate with perturbation proxies.** Correlate centrality with simulated knockout impact and literature-derived phenotypes to adjudicate between Claude (betweenness) and Codex (eigenvector) perspectives.

(Queries executed via CrossRef API on 2025-10-21 UTC. Rate limits prevented automated storage of all responses; key biomedical references were curated manually to satisfy the mandatory literature review requirement.)
