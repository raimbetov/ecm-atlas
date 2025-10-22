# H14 – Serpin Network Centrality Resolution: Methodological Deep Dive

## Scientific Question
Are serpins (SERPIN family protease inhibitors) central hubs in ECM aging networks, and which network centrality metric (betweenness, eigenvector, closeness, PageRank) best captures functional importance in proteomics networks?

## Background & Rationale

**DISAGREEMENT from Iteration 01 (H02):**
- **Claude Code:** Serpins are NOT central hubs
  - Used betweenness centrality
  - Top hubs: MMP14, ADAMTS, collagens (NOT serpins)
- **Codex:** Serpins ARE central hubs
  - Used eigenvector centrality (or degree)
  - Top hubs: SERPINC1, SERPINF2, SERPINE1

**Why This Disagreement Matters:**
- Network centrality informs TARGET PRIORITIZATION for drug development
- If serpins are central → anticoagulants (e.g., heparin for SERPINC1) could be anti-aging
- If serpins are peripheral → focus on MMPs, ADAMTS instead
- **Methodological lesson:** Choice of centrality metric changes conclusions!

**Biological Context:**
- Serpins (SERine Protease INhibitors) regulate coagulation, inflammation, ECM degradation
- Known aging-related serpins: SERPINC1 (antithrombin), SERPINE1 (PAI-1), SERPINF1 (PEDF)
- But: Are they CAUSAL regulators or DOWNSTREAM effectors?

## Objectives

### Primary Objective
Systematically compare 6+ network centrality metrics on the SAME ECM aging network, identify which metric best predicts functional importance (validated by perturbation experiments or literature), and resolve the H02 serpin centrality dispute.

### Secondary Objectives
1. **Literature validation:** Find network centrality best practices for proteomics/systems biology
2. **Perturbation analysis:** Simulate serpin knockouts, measure network impact
3. **Experimental validation:** Cross-reference with published knockout/overexpression studies
4. **Recommendation:** Standardize centrality metric for all future network hypotheses (H05, future GNN)

## Hypotheses to Test

### H14.1: Betweenness Centrality (Claude Method)
Betweenness identifies serpin's as non-central (mean rank >50th percentile).

### H14.2: Eigenvector Centrality (Codex Method)
Eigenvector identifies serpins as central hubs (mean rank <20th percentile).

### H14.3: PageRank Centrality (NEW)
PageRank (Google's algorithm) identifies serpins as central, matching eigenvector.

### H14.4: Closeness Centrality (NEW)
Closeness identifies serpins as peripheral (matches betweenness).

### H14.5: Perturbation-Based Validation
Metric that best predicts knockout impact (measured as Δ network connectivity) is the "true" functional centrality.

### H14.6: Consensus Centrality
Ensemble of all metrics produces most robust hub ranking.

## Required Analyses

### 1. LITERATURE SEARCH (MANDATORY)

**Search queries:**
```
1. "network centrality metrics comparison proteomics"
2. "betweenness vs eigenvector centrality biological networks"
3. "PageRank protein interaction networks"
4. "centrality lethality rule validation"
5. "serpin protease inhibitor networks aging"
6. "graph theory systems biology best practices"
7. "knockout experiments network topology"
```

**Tasks:**
- Search Nature Methods, PLOS Comp Bio, Network Science, Bioinformatics
- Download methodological papers (centrality metric benchmarks)
- Extract: Which metric predicts lethality, disease genes, drug targets best?
- Save: `literature_network_centrality.md`

### 2. NETWORK RECONSTRUCTION (STANDARDIZED)

**Data source:**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Edge definition (match H02/H05 methods):**
```python
from scipy.stats import spearmanr

# Correlation network
correlations = {}
for protein_i in proteins:
    for protein_j in proteins:
        if i < j:  # avoid duplicates
            rho, p = spearmanr(data[protein_i], data[protein_j])
            if abs(rho) > 0.5 and p < 0.05:  # threshold
                correlations[(protein_i, protein_j)] = rho

# Create graph
import networkx as nx
G = nx.Graph()
for (pi, pj), rho in correlations.items():
    G.add_edge(pi, pj, weight=abs(rho))
```

**Network stats:**
```python
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G)}")
print(f"Avg clustering: {nx.average_clustering(G)}")
```

**Save:**
- `network_edges_{agent}.csv` (all edges with weights)
- `network_stats_{agent}.json`

### 3. CENTRALITY METRICS COMPUTATION (6+ METRICS)

#### Metric 1: Betweenness Centrality (Claude H02 Method)

**Definition:** Fraction of shortest paths passing through node.

**Interpretation:** Nodes connecting disparate network modules (bridging).

```python
betweenness = nx.betweenness_centrality(G, weight='weight')
```

#### Metric 2: Eigenvector Centrality (Codex H02 Method)

**Definition:** Influence based on connections to high-degree neighbors.

**Interpretation:** Nodes connected to other important nodes (prestige).

```python
eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
```

#### Metric 3: Degree Centrality

**Definition:** Number of direct connections.

**Interpretation:** Local connectivity (hub-ness).

```python
degree = nx.degree_centrality(G)
```

#### Metric 4: Closeness Centrality

**Definition:** Average shortest path to all other nodes.

**Interpretation:** Nodes quickly reachable from anywhere (efficiency).

```python
closeness = nx.closeness_centrality(G, distance='weight')
```

#### Metric 5: PageRank

**Definition:** Google's algorithm (eigenvector + random walk damping).

**Interpretation:** Probability of landing on node in random walk.

```python
pagerank = nx.pagerank(G, weight='weight')
```

#### Metric 6: Katz Centrality

**Definition:** Weighted sum of all path lengths (eigenvector + paths).

**Interpretation:** Global reachability with decay.

```python
katz = nx.katz_centrality(G, weight='weight')
```

#### Metric 7: Subgraph Centrality (BONUS)

**Definition:** Weighted count of closed walks starting/ending at node.

**Interpretation:** Participation in network motifs.

```python
subgraph = nx.subgraph_centrality(G)
```

**Output table:**
```csv
Protein, Betweenness, Eigenvector, Degree, Closeness, PageRank, Katz, Subgraph
COL1A1, 0.12, 0.08, 0.15, 0.42, 0.003, 0.05, 124.3
SERPINC1, 0.03, 0.21, 0.10, 0.38, 0.005, 0.18, 89.2
...
```

### 4. SERPIN FAMILY RANKING

**Serpin proteins (from dataset):**
```
SERPINA1, SERPINA3, SERPINA5, SERPINB1, SERPINB6, SERPINB8,
SERPINC1, SERPINE1, SERPINE2, SERPINF1, SERPINF2, SERPING1, SERPINH1
```

**For each metric:**
```python
# Rank all proteins
ranked_proteins = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

# Serpin ranks
serpin_ranks = [rank for rank, (protein, score) in enumerate(ranked_proteins) if 'SERPIN' in protein]
serpin_mean_rank = np.mean(serpin_ranks)
serpin_median_rank = np.median(serpin_ranks)

# Percentile
serpin_percentile = serpin_mean_rank / len(ranked_proteins) * 100
```

**Output:**
| Metric | Serpin Mean Rank | Serpin Percentile | Top Serpin | Top Serpin Rank |
|--------|------------------|-------------------|------------|-----------------|
| Betweenness | 180 / 648 | 28% | SERPINE1 | 45 |
| Eigenvector | 50 / 648 | 8% | SERPINC1 | 12 |
| PageRank | 45 / 648 | 7% | SERPINC1 | 10 |
| ... | ... | ... | ... | ... |

**Classification:**
- **Central:** Percentile <20% (top quintile)
- **Peripheral:** Percentile >50% (bottom half)

### 5. CORRELATION BETWEEN METRICS

**Pairwise Spearman correlations:**
```python
import pandas as pd
centrality_df = pd.DataFrame({
    'Betweenness': list(betweenness.values()),
    'Eigenvector': list(eigenvector.values()),
    ...
})

correlation_matrix = centrality_df.corr(method='spearman')
```

**Interpretation:**
- High correlation (ρ>0.80) → metrics agree, robust consensus
- Low correlation (ρ<0.50) → metrics measure different aspects

**Visualize:**
- Heatmap: metric × metric correlation matrix
- Scatter plots: Betweenness vs Eigenvector (highlight serpins)

### 6. PERTURBATION ANALYSIS (IN SILICO KNOCKOUT)

**Goal:** Which centrality metric best predicts network impact when serpin is removed?

**Method:**
```python
# For each serpin
for serpin in serpins:
    # Remove node
    G_knockout = G.copy()
    G_knockout.remove_node(serpin)

    # Measure network impact
    connectivity_before = nx.average_node_connectivity(G)
    connectivity_after = nx.average_node_connectivity(G_knockout)
    delta_connectivity = connectivity_before - connectivity_after

    # Also: largest component size, avg shortest path, clustering
    metrics = {
        'delta_connectivity': delta_connectivity,
        'delta_components': nx.number_connected_components(G_knockout) - nx.number_connected_components(G),
        'delta_clustering': nx.average_clustering(G) - nx.average_clustering(G_knockout)
    }

    knockouts.append({
        'Protein': serpin,
        'Impact': delta_connectivity,  # composite score
        **metrics
    })
```

**Correlation with centrality:**
```python
# For each metric
from scipy.stats import spearmanr

for metric in ['Betweenness', 'Eigenvector', ...]:
    rho, p = spearmanr(
        [knockouts_impact[s] for s in serpins],
        [centrality_scores[metric][s] for s in serpins]
    )
    print(f"{metric}: ρ={rho:.3f}, p={p:.4f}")
```

**Best metric:** Highest ρ (centrality → knockout impact correlation)

**Hypothesis:**
- If Eigenvector has highest ρ → Eigenvector is "true" functional centrality
- If Betweenness has highest ρ → Betweenness wins

### 7. EXPERIMENTAL VALIDATION (LITERATURE CROSS-REFERENCE)

**Search published knockout/overexpression studies:**

**Query PubMed:**
```
"SERPINC1 knockout" OR "SERPINE1 knockout" OR "SERPINF1 knockout"
AND ("network" OR "interactome" OR "pleiotropy")
```

**Extract:**
- Phenotype severity (mild, moderate, lethal)
- Number of downstream affected proteins/pathways
- Tissue-specific effects

**Rank serpins by experimental impact:**
```
Severe knockout phenotype → High functional importance
Mild/no phenotype → Low functional importance
```

**Compare to centrality rankings:**
```python
from scipy.stats import spearmanr

experimental_importance = [3, 2, 1, ...]  # hand-curated from literature
centrality_betweenness_ranks = [45, 120, 78, ...]

rho, p = spearmanr(experimental_importance, centrality_betweenness_ranks)
```

**Best metric:** Highest ρ with experimental data

### 8. CONSENSUS CENTRALITY (ENSEMBLE)

**Combine all metrics:**
```python
# Z-score normalize each metric
from scipy.stats import zscore

centrality_zscore = {}
for metric in metrics:
    centrality_zscore[metric] = zscore(list(centrality_scores[metric].values()))

# Consensus: mean z-score
consensus = np.mean([centrality_zscore[m] for m in metrics], axis=0)

# Rank by consensus
ranked_consensus = sorted(zip(proteins, consensus), key=lambda x: x[1], reverse=True)
```

**Serpin ranking in consensus:**
- Does ensemble resolve disagreement?
- Or does it average out to "moderate" centrality?

### 9. COMMUNITY DETECTION & SERPIN MODULE

**Goal:** Are serpins in a distinct module, or dispersed across network?

**Method:**
```python
from community import community_louvain

# Detect communities
communities = community_louvain.best_partition(G, weight='weight')

# Serpin community assignments
serpin_communities = {s: communities[s] for s in serpins if s in G.nodes}
serpin_modularity = len(set(serpin_communities.values()))  # number of distinct modules

# Are serpins in ONE module?
if serpin_modularity == 1:
    print("Serpins form cohesive module")
else:
    print(f"Serpins dispersed across {serpin_modularity} modules")
```

**Intra-module centrality:**
```python
# Centrality within serpin module only
serpin_subgraph = G.subgraph([n for n, c in communities.items() if c == serpin_module_id])
intra_centrality = nx.betweenness_centrality(serpin_subgraph)
```

**Interpretation:**
- If serpins cluster → use intra-module centrality (within serpin network)
- If dispersed → global centrality applies

## Deliverables

### Code & Models
- `analysis_network_centrality_{agent}.py` — main script
- `perturbation_analysis_{agent}.py` — knockout simulations
- `literature_validation_{agent}.py` — scrape PubMed for knockout studies
- `community_detection_{agent}.py` — modularity analysis

### Data Tables
- `network_edges_{agent}.csv` — All edges (protein pairs, weights)
- `centrality_all_metrics_{agent}.csv` — All proteins × 7 metrics
- `serpin_rankings_{agent}.csv` — Serpin ranks per metric
- `metric_correlation_matrix_{agent}.csv` — Metric × metric Spearman ρ
- `knockout_impact_{agent}.csv` — Serpin × knockout Δ connectivity
- `centrality_knockout_correlation_{agent}.csv` — Which metric predicts KO impact best
- `experimental_validation_{agent}.csv` — Literature-derived serpin importance
- `consensus_centrality_{agent}.csv` — Ensemble scores
- `community_assignments_{agent}.csv` — Protein → module mapping

### Visualizations
- `visualizations_{agent}/centrality_heatmap_{agent}.png` — Metric correlation heatmap
- `visualizations_{agent}/serpin_ranks_comparison_{agent}.png` — Violin plot: serpin ranks per metric
- `visualizations_{agent}/betweenness_vs_eigenvector_{agent}.png` — Scatter, serpins highlighted
- `visualizations_{agent}/knockout_impact_scatter_{agent}.png` — Centrality vs KO Δ connectivity
- `visualizations_{agent}/network_graph_serpins_{agent}.png` — Network with serpins colored
- `visualizations_{agent}/community_modules_{agent}.png` — Network colored by module

### Report
- `90_results_{agent}.md` — comprehensive findings with:
  - Literature synthesis (centrality metric best practices)
  - Network reconstruction (stats, topology)
  - Centrality comparison (which metrics agree/disagree?)
  - Serpin ranking resolution (Claude betweenness vs Codex eigenvector)
  - Perturbation validation (which metric predicts KO impact?)
  - Experimental validation (literature knockout data)
  - Consensus ranking (ensemble approach)
  - **RESOLUTION:** Are serpins central or not? Which metric is correct?
  - **RECOMMENDATION:** Standardized metric for Iterations 05-07

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Centrality metrics computed | ≥6 | NetworkX algorithms |
| Metric correlation (ρ) | Report all | Spearman test |
| Knockout simulations | All serpins (≥10) | In silico perturbation |
| Centrality-KO correlation | ρ>0.60 for best metric | Validation |
| Literature papers | ≥5 relevant | PubMed knockout studies |
| Experimental validation | ≥3 serpins with data | Cross-reference |
| Agent consensus | Same top metric | Both agents agree |

## Expected Outcomes

### Scenario 1: Betweenness Wins (Claude Correct)
- Betweenness best predicts knockout impact (ρ>0.70)
- Serpins are peripheral (mean percentile >40%)
- **Conclusion:** Serpins are DOWNSTREAM effectors, not central hubs
- **Action:** Focus drug targets on MMPs, ADAMTS, collagens

### Scenario 2: Eigenvector Wins (Codex Correct)
- Eigenvector best predicts knockout impact (ρ>0.70)
- Serpins are central (mean percentile <20%)
- **Conclusion:** Serpins are REGULATORY hubs
- **Action:** Anticoagulants (SERPINC1), PAI-1 inhibitors (SERPINE1) are anti-aging targets

### Scenario 3: PageRank Wins (NEW Discovery)
- PageRank outperforms both (ρ>0.75)
- Serpins moderately central (percentile 20-40%)
- **Conclusion:** Neither agent used optimal metric, adopt PageRank

### Scenario 4: Consensus Ensemble Best
- No single metric wins, ensemble ρ>0.80
- **Conclusion:** Use composite centrality score (average of top 3 metrics)

### Scenario 5: Metric Depends on Context
- Betweenness predicts KO impact for structural proteins (collagens)
- Eigenvector predicts KO impact for regulatory proteins (serpins, MMPs)
- **Conclusion:** Use protein class-specific centrality metrics

## Clinical Translation

**If serpins ARE central:**
- **Drug targets:** SERPINE1 (PAI-1) inhibitors already in trials (TM5441, SK-216)
- **Biomarkers:** SERPINC1 (antithrombin) plasma levels
- **Caution:** Anticoagulation has bleeding risk → careful dosing

**If serpins are NOT central:**
- **Redirect focus:** MMPs (MMP inhibitors failed in trials), ADAMTS, LOX
- **Avoid:** Overinvestment in serpin-targeted therapies

**Methodological impact:**
- Standardize centrality metric for ALL future network analyses
- Update H05 GNN centrality layer with validated metric
- Publish methods paper: "Network centrality metrics in aging proteomics"

## Dataset

**Primary:**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Network:**
- Nodes: 648 ECM-centric proteins
- Edges: Spearman |ρ|>0.5, p<0.05 (from H02/H05)

## References

1. H02 Results: `/iterations/iteration_01/hypothesis_02_serpin_dysregulation/{claude_code,codex}/90_results_{agent}.md`
2. H05 Results: `/iterations/iteration_02/hypothesis_05_gnn_hidden_connections/{claude_code}/90_results_claude_code.md`
3. ADVANCED_ML_REQUIREMENTS.md
4. Jeong et al. (2001). "Lethality and centrality in protein networks." Nature. (Centrality-lethality rule)
5. Zitnik et al. (2019). "Machine learning for integrating data in biology and medicine: Principles, practice, and opportunities." Nature Methods.
6. Kitsak et al. (2010). "Identification of influential spreaders in complex networks." Nature Physics. (k-core vs degree)
