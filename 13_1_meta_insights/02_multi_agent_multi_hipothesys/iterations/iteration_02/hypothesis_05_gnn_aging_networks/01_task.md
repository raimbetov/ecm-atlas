# Hypothesis 05: Graph Neural Networks Identify Master Regulator Proteins

## Scientific Question

Can Graph Neural Networks (GNNs) trained on protein interaction networks identify "master regulator" proteins whose perturbation cascades through the aging network, and do GNN-learned embeddings reveal hidden protein communities invisible to traditional network analysis?

## Background Context

**Traditional Network Analysis Limitations:**
- Degree centrality, betweenness = simple topological metrics
- Community detection (Louvain) uses modularity optimization, may miss functional modules
- Ignores node features (protein properties)

**GNN Advantage:** GNNs learn protein embeddings by aggregating information from neighbors, capturing both network topology AND node features. Attention mechanisms highlight influential edges.

**Expected Discovery:** GNN identifies 5-10 "master regulators" with high learned attention weights, whose aging dysregulation predicts downstream cascade effects. GNN communities have stronger biological coherence than Louvain.

## Data Source

**Primary Dataset:**
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Network Data:**
- Build correlation network from protein z-scores
- Optional: STRING database protein-protein interactions (API or download)

## ML Requirements (MANDATORY)

### Must Use At Least 3 of These:

1. **Graph Convolutional Network (GCN) - Required:**
   - Build protein network: nodes=proteins, edges=correlation (|ρ| > 0.5)
   - Node features: Δz, tissue breadth, Matrisome category (one-hot)
   - Train GCN to predict: Aging class (up/down/stable)
   - Extract learned node embeddings

2. **Graph Attention Network (GAT):**
   - Multi-head attention to learn edge importance
   - Attention weights reveal influential protein-protein relationships
   - Identify proteins with high incoming attention = master regulators

3. **Graph Community Detection (Advanced):**
   - Louvain, Leiden (baseline)
   - Compare with GNN-based clustering:
     - Cluster GNN embeddings (HDBSCAN)
     - Evaluate biological coherence (GO enrichment, Matrisome purity)

4. **Temporal Graph Networks:**
   - If age metadata available: Model aging as temporal graph evolution
   - Predict future aging state from current network structure

5. **Graph Autoencoders:**
   - Unsupervised learning of protein embeddings
   - Reconstruct edges from learned embeddings
   - Identify "link prediction" pairs = likely aging co-dysregulation

6. **Message Passing Neural Networks (MPNN):**
   - Custom aggregation functions for protein neighborhoods
   - Learn aging propagation rules

## Success Criteria

### Criterion 1: GNN Training and Embeddings (40 pts)

**Required:**
1. Build protein interaction network:
   - Nodes: All ECM proteins
   - Edges: Spearman |ρ| > 0.5 across tissues
   - Node features: [Δz_mean, Δz_std, Tissue_Count, Matrisome_OneHot]
2. Train GCN or GAT:
   - Task: Node classification (upregulated / downregulated / stable)
   - Architecture: ≥2 GNN layers
   - Metrics: Accuracy, F1-score, confusion matrix
3. Extract node embeddings from final GNN layer
4. Validate embeddings:
   - Do embeddings cluster by Matrisome category?
   - UMAP visualization: Color by aging direction

**Deliverables:**
- `gnn_weights_[agent].pth` - Trained GNN model
- `protein_embeddings_gnn_[agent].csv` - Node embeddings (proteins × embedding_dim)
- `gnn_training_metrics_[agent].csv` - Accuracy, F1, loss per epoch
- `network_graph_[agent].graphml` - Network structure (for Gephi visualization)

### Criterion 2: Master Regulator Identification (30 pts)

**Required:**
1. Rank proteins by GNN-learned importance:
   - **Attention weights** (if GAT): Sum incoming attention across all edges
   - **Gradient-based importance:** ∂(output) / ∂(node_features)
   - **PageRank on GNN embeddings:** Run PageRank on GNN similarity graph
2. Identify top 10 "master regulators"
3. Validate master regulators:
   - Are they serpins, collagens, or known hubs?
   - Compare with degree centrality, betweenness from traditional network
   - Do GNN-identified hubs differ from topology-based hubs?
4. Perturbation analysis:
   - Simulate: Remove master regulator → Measure effect on network embeddings
   - Quantify cascade: How many downstream proteins affected?

**Deliverables:**
- `master_regulators_[agent].csv` - Top 10 proteins with importance scores
- `attention_heatmap_[agent].png` (if GAT) - Attention weights visualization
- `perturbation_analysis_[agent].csv` - Cascade effects per master regulator

### Criterion 3: Community Detection Comparison (20 pts)

**Required:**
1. **Baseline:** Run Louvain algorithm on correlation network
2. **GNN-based:** Cluster GNN embeddings (HDBSCAN, k-means)
3. Compare communities:
   - **Biological coherence:** GO enrichment, Matrisome category purity
   - **Adjusted Rand Index (ARI):** GNN vs Louvain
   - **Silhouette score:** GNN embeddings vs raw features
4. Visualize communities:
   - Network graph colored by GNN communities
   - Compare with Louvain communities side-by-side

**Deliverables:**
- `community_comparison_[agent].csv` - ARI, silhouette scores
- `gnn_communities_[agent].png` - Network visualization
- `louvain_communities_[agent].png` - Baseline visualization

### Criterion 4: Novel Insights (10 pts)

**Required:**
1. Identify non-obvious connections:
   - Protein pairs with LOW correlation but HIGH GNN similarity
   - Suggests indirect relationship via network paths
2. Predict aging co-dysregulation:
   - Use link prediction: Proteins not directly correlated but GNN predicts will co-vary
3. Therapeutic targets:
   - Which master regulators are druggable?
   - Cascade analysis: Which perturbation has widest protective effect?

**Deliverables:**
- `hidden_connections_[agent].csv` - Non-obvious protein pairs
- `link_prediction_[agent].csv` - Predicted future interactions
- `therapeutic_ranking_[agent].csv` - Master regulators by impact × druggability

## Required Artifacts

All in `claude_code/` or `codex/`:

1. **01_plan_[agent].md**
2. **analysis_gnn_[agent].py** - Full GNN pipeline
3. **gnn_weights_[agent].pth** - Trained model
4. **protein_embeddings_gnn_[agent].csv**
5. **master_regulators_[agent].csv**
6. **visualizations_[agent]/**:
   - network_graph_[agent].png (interactive HTML optional)
   - attention_heatmap_[agent].png
   - gnn_umap_embeddings_[agent].png
   - community_comparison_[agent].png
7. **90_results_[agent].md** - Knowledge Framework format

## ML Implementation Template

```python
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import networkx as nx

# 1. Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# 2. Build correlation network
from scipy.stats import spearmanr
pivot = df.pivot_table(values='Zscore_Delta', index='Gene_Symbol', columns='Tissue')
X = pivot.fillna(0).values
proteins = pivot.index.tolist()

# Correlation matrix
corr_matrix, _ = spearmanr(X, axis=1)
threshold = 0.5

# Edge index
edges = np.argwhere(np.abs(corr_matrix) > threshold)
edge_index = torch.tensor(edges.T, dtype=torch.long)

# 3. Node features
node_features = torch.tensor(X, dtype=torch.float32)  # Or add more features

# 4. Node labels (example: up/down/stable)
labels = (X.mean(axis=1) > 0).astype(int)  # Binary: upregulated vs not
labels = torch.tensor(labels, dtype=torch.long)

# 5. PyG Data object
data = Data(x=node_features, edge_index=edge_index, y=labels)

# 6. Define GCN
class ProteinGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        embeddings = x  # Save for later
        x = self.fc(x)
        return F.log_softmax(x, dim=1), embeddings

# 7. Training
model = ProteinGCN(num_features=node_features.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")

# 8. Extract embeddings
model.eval()
with torch.no_grad():
    _, embeddings = model(data.x, data.edge_index)
    embeddings_np = embeddings.numpy()

# Save
pd.DataFrame(embeddings_np, index=proteins).to_csv('protein_embeddings_gnn_[agent].csv')
torch.save(model.state_dict(), 'gnn_weights_[agent].pth')

# 9. (Optional) GAT for attention
class ProteinGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2, heads=4):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## Documentation Standards

Follow Knowledge Framework.

Reference: `/Users/Kravtsovd/projects/ecm-atlas/ADVANCED_ML_REQUIREMENTS.md`

## Expected Results

- GNN accuracy > 80% on node classification
- 5-10 master regulators identified (serpins, collagens expected)
- GNN communities have higher biological coherence than Louvain (GO enrichment p < 0.01)
- Attention mechanism reveals serpin-collagen cross-talk

---

**Hypothesis ID:** H05
**Iteration:** 02
**Predicted Scores:** Novelty 10/10, Impact 9/10
**ML Focus:** ✅ GNNs, GAT, Network Analysis
