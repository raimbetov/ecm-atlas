# H05 – Master Regulator Discovery (codex)

## Training Performance
- Best epoch: 132
- Validation F1 (macro): 0.928
- Validation accuracy: 0.941

## Master Regulators (Top 10)
| Protein   |   attention_importance |   gradient_importance |   pagerank_embeddings |   degree_centrality |   betweenness_centrality |   delta_mean |   tissue_count | matrisome_category                |   composite_score |
|:----------|-----------------------:|----------------------:|----------------------:|--------------------:|-------------------------:|-------------:|---------------:|:----------------------------------|------------------:|
| Kng1      |               0.609618 |             1.56408   |            0.00109669 |            0.112088 |              0.000287569 |    0.560596  |              8 | matrisome_ecm_regulators          |          0.724114 |
| Plxna1    |               0.71722  |             1.03498   |            0.00109771 |            0.165934 |              0.000925338 |   -0.503283  |              3 | matrisome_ecm-affiliated_proteins |          0.714038 |
| Sulf2     |               0.68637  |             1.6334    |            0.00109559 |            0.138462 |              0.00336606  |    0.439999  |              1 | matrisome_ecm_regulators          |          0.708754 |
| Lman1     |               0.717091 |             1.21607   |            0.00109678 |            0.165934 |              0.000925338 |   -0.47763   |              3 | matrisome_ecm-affiliated_proteins |          0.700336 |
| Hapln2    |               0.79793  |             0.38787   |            0.00109919 |            0.172527 |              0.000147002 |    3.28151   |              2 | matrisome_proteoglycans           |          0.694284 |
| SFTPC     |               0.835068 |             0.1014    |            0.00109969 |            0.127473 |              0.000152299 |   -0.0878015 |              1 | matrisome_ecm-affiliated_proteins |          0.67732  |
| TGFB2     |               0.835605 |             0.0472219 |            0.00109985 |            0.127473 |              0.000152299 |    0.0165355 |              1 | matrisome_secreted_factors        |          0.67541  |
| CRLF3     |               0.835576 |             0.0499264 |            0.00109984 |            0.127473 |              0.000152299 |    0.0942317 |              1 | matrisome_secreted_factors        |          0.675399 |
| HCFC2     |               0.835606 |             0.0471475 |            0.00109985 |            0.127473 |              0.000152299 |    0.0137345 |              1 | matrisome_secreted_factors        |          0.675368 |
| EMID1     |               0.835568 |             0.0496861 |            0.00109984 |            0.127473 |              0.000152299 |    0.0757208 |              1 | matrisome_ecm_glycoproteins       |          0.675324 |

## Community Comparison
{
  "ARI": 0.0721507707130202,
  "Silhouette": 0.658595621585846,
  "MatrisomePurity": 0.986590418825004,
  "n_louvain": 6,
  "n_embedding_clusters": 13
}

## Notes
- Aging direction labels use Δz thresholds (up ≥ 0.5, down ≤ -0.5).
- Attention, gradient, and PageRank scores were z-normalized before ranking.
- Perturbation cascade size counts proteins whose embeddings shift above the 85th percentile.
