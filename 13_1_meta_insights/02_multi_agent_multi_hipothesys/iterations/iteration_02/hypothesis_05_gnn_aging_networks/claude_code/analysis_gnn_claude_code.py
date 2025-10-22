#!/usr/bin/env python3
"""
Graph Neural Network Analysis for ECM Aging Master Regulator Discovery
Agent: claude_code
Hypothesis H05: GNN identifies master regulator proteins via attention mechanisms
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import from_networkx
import networkx as nx
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import hdbscan
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Constants
DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_02/hypothesis_05_gnn_aging_networks/claude_code")
VIZ_DIR = OUTPUT_DIR / "visualizations_claude_code"
VIZ_DIR.mkdir(exist_ok=True)

CORRELATION_THRESHOLD = 0.5
HIDDEN_DIM = 128
EMBEDDING_DIM = 32
LEARNING_RATE = 0.005
EPOCHS = 300
PATIENCE = 20
GAT_HEADS = 8

print("=" * 80)
print("GNN MASTER REGULATOR DISCOVERY PIPELINE")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n[1/10] Loading ECM aging dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Filter for proteins with sufficient data
print("\n[2/10] Filtering and preparing protein matrix...")
# Create protein × tissue matrix of Zscore_Delta
pivot = df.pivot_table(
    values='Zscore_Delta',
    index='Gene_Symbol',
    columns='Tissue',
    aggfunc='mean'
)
print(f"Pivot shape (before filtering): {pivot.shape}")

# Filter proteins present in at least 3 tissues
min_tissues = 3
protein_tissue_count = (~pivot.isnull()).sum(axis=1)
valid_proteins = protein_tissue_count[protein_tissue_count >= min_tissues].index
pivot = pivot.loc[valid_proteins]
pivot_filled = pivot.fillna(0)  # Fill NaN with 0 for correlation calculation
print(f"Pivot shape (after filtering ≥{min_tissues} tissues): {pivot_filled.shape}")

proteins = pivot_filled.index.tolist()
X_raw = pivot_filled.values  # Shape: (n_proteins, n_tissues)

# ============================================================================
# 2. NODE FEATURE ENGINEERING
# ============================================================================
print("\n[3/10] Engineering node features...")
# Feature 1: Δz mean (average aging response)
delta_z_mean = X_raw.mean(axis=1)

# Feature 2: Δz std (variability of aging response)
delta_z_std = X_raw.std(axis=1)

# Feature 3: Tissue count (breadth metric)
tissue_count = (~pivot.isnull()).sum(axis=1).values

# Feature 4: Matrisome category (one-hot encoding)
# Map proteins to Matrisome categories
protein_to_matrisome = df.groupby('Gene_Symbol')['Matrisome_Category_Simplified'].first()
protein_matrisome = [protein_to_matrisome.get(p, 'Unknown') for p in proteins]

matrisome_categories = ['ECM Glycoproteins', 'Collagens', 'Proteoglycans', 'ECM Regulators']
matrisome_onehot = np.zeros((len(proteins), len(matrisome_categories)))
for i, cat in enumerate(protein_matrisome):
    if cat in matrisome_categories:
        idx = matrisome_categories.index(cat)
        matrisome_onehot[i, idx] = 1

# Combine features
node_features = np.column_stack([
    delta_z_mean,
    delta_z_std,
    tissue_count,
    matrisome_onehot
])

# Standardize features
scaler = StandardScaler()
node_features_scaled = scaler.fit_transform(node_features)
print(f"Node features shape: {node_features_scaled.shape}")
print(f"Feature names: Δz_mean, Δz_std, Tissue_Count, Matrisome_OneHot(4D)")

# ============================================================================
# 3. NODE LABELS (3-class classification)
# ============================================================================
print("\n[4/10] Creating node labels...")
# 3-class: Upregulated (Δz > 0.5), Downregulated (Δz < -0.5), Stable
labels_raw = np.zeros(len(proteins), dtype=int)
labels_raw[delta_z_mean > 0.5] = 0  # Upregulated
labels_raw[delta_z_mean < -0.5] = 1  # Downregulated
labels_raw[(delta_z_mean >= -0.5) & (delta_z_mean <= 0.5)] = 2  # Stable

label_names = ['Upregulated', 'Downregulated', 'Stable']
label_counts = np.bincount(labels_raw)
print(f"Label distribution: Upregulated={label_counts[0]}, Downregulated={label_counts[1]}, Stable={label_counts[2]}")

# ============================================================================
# 4. BUILD CORRELATION NETWORK
# ============================================================================
print("\n[5/10] Building protein correlation network...")
# Compute Spearman correlation matrix across tissues
corr_matrix, _ = spearmanr(X_raw, axis=1, nan_policy='omit')
corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)  # Replace NaN with 0

# Create edges where |correlation| > threshold
abs_corr = np.abs(corr_matrix)
edge_mask = abs_corr > CORRELATION_THRESHOLD
np.fill_diagonal(edge_mask, False)  # Remove self-loops

edge_list = np.argwhere(edge_mask)
edge_weights = abs_corr[edge_mask]

print(f"Network stats:")
print(f"  Nodes: {len(proteins)}")
print(f"  Edges: {len(edge_list)} (|ρ| > {CORRELATION_THRESHOLD})")
print(f"  Density: {len(edge_list) / (len(proteins) * (len(proteins) - 1)):.4f}")

# Convert to PyTorch tensors
edge_index = torch.tensor(edge_list.T, dtype=torch.long)
edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
x = torch.tensor(node_features_scaled, dtype=torch.float32)
y = torch.tensor(labels_raw, dtype=torch.long)

# Create PyG Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
print(f"PyG Data object created: {data}")

# Train/Val/Test split
num_nodes = len(proteins)
indices = np.arange(num_nodes)
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=SEED, stratify=labels_raw)
train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=SEED, stratify=labels_raw[train_idx])  # 0.176 * 0.85 ≈ 0.15

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print(f"Split sizes: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

# ============================================================================
# 5. DEFINE GNN MODELS
# ============================================================================
class ProteinGCN(nn.Module):
    """Graph Convolutional Network for protein classification"""
    def __init__(self, num_features, hidden_dim=128, embedding_dim=32, num_classes=3):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, return_embeddings=False):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3 (embeddings)
        embeddings = self.conv3(x, edge_index)
        embeddings = F.relu(embeddings)

        if return_embeddings:
            return embeddings

        # Classification head
        out = self.fc(embeddings)
        return F.log_softmax(out, dim=1), embeddings


class ProteinGAT(nn.Module):
    """Graph Attention Network with multi-head attention"""
    def __init__(self, num_features, hidden_dim=128, num_classes=3, heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim // 2, heads=heads, dropout=0.3)
        self.conv3 = GATConv(hidden_dim // 2 * heads, num_classes, heads=1, concat=False, dropout=0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, return_attention=False):
        # Layer 1
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 3
        x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)

        if return_attention:
            return F.log_softmax(x, dim=1), (attn1, attn2, attn3)

        return F.log_softmax(x, dim=1)


# ============================================================================
# 6. TRAINING FUNCTIONS
# ============================================================================
def train_model(model, data, optimizer, criterion):
    """Single training epoch"""
    model.train()
    optimizer.zero_grad()

    if isinstance(model, ProteinGCN):
        out, _ = model(data.x, data.edge_index)
    else:  # GAT
        out = model(data.x, data.edge_index)

    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_model(model, data, mask):
    """Evaluate model on given mask"""
    model.eval()
    with torch.no_grad():
        if isinstance(model, ProteinGCN):
            out, _ = model(data.x, data.edge_index)
        else:  # GAT
            out = model(data.x, data.edge_index)

        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total

        # F1 score
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        f1 = f1_score(y_true, y_pred, average='macro')

    return acc, f1


# ============================================================================
# 7. TRAIN GCN MODEL
# ============================================================================
print("\n[6/10] Training Graph Convolutional Network (GCN)...")
gcn_model = ProteinGCN(
    num_features=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    embedding_dim=EMBEDDING_DIM,
    num_classes=len(label_names)
)
optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
criterion = nn.NLLLoss()

gcn_metrics = []
best_val_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    loss = train_model(gcn_model, data, optimizer_gcn, criterion)
    train_acc, train_f1 = evaluate_model(gcn_model, data, data.train_mask)
    val_acc, val_f1 = evaluate_model(gcn_model, data, data.val_mask)

    gcn_metrics.append({
        'epoch': epoch,
        'train_loss': loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_acc': val_acc,
        'val_f1': val_f1
    })

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(gcn_model.state_dict(), OUTPUT_DIR / 'gcn_best.pth')
    else:
        patience_counter += 1

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model and evaluate on test set
gcn_model.load_state_dict(torch.load(OUTPUT_DIR / 'gcn_best.pth'))
test_acc, test_f1 = evaluate_model(gcn_model, data, data.test_mask)
print(f"\nGCN Test Performance: Accuracy={test_acc:.4f}, F1={test_f1:.4f}")

# Save metrics
gcn_metrics_df = pd.DataFrame(gcn_metrics)
gcn_metrics_df.to_csv(OUTPUT_DIR / 'gnn_training_metrics_gcn_claude_code.csv', index=False)

# ============================================================================
# 8. TRAIN GAT MODEL
# ============================================================================
print("\n[7/10] Training Graph Attention Network (GAT)...")
gat_model = ProteinGAT(
    num_features=data.x.shape[1],
    hidden_dim=HIDDEN_DIM,
    num_classes=len(label_names),
    heads=GAT_HEADS
)
optimizer_gat = torch.optim.Adam(gat_model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

gat_metrics = []
best_val_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    loss = train_model(gat_model, data, optimizer_gat, criterion)
    train_acc, train_f1 = evaluate_model(gat_model, data, data.train_mask)
    val_acc, val_f1 = evaluate_model(gat_model, data, data.val_mask)

    gat_metrics.append({
        'epoch': epoch,
        'train_loss': loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_acc': val_acc,
        'val_f1': val_f1
    })

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(gat_model.state_dict(), OUTPUT_DIR / 'gat_best.pth')
    else:
        patience_counter += 1

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model and evaluate on test set
gat_model.load_state_dict(torch.load(OUTPUT_DIR / 'gat_best.pth'))
test_acc_gat, test_f1_gat = evaluate_model(gat_model, data, data.test_mask)
print(f"\nGAT Test Performance: Accuracy={test_acc_gat:.4f}, F1={test_f1_gat:.4f}")

# Save metrics
gat_metrics_df = pd.DataFrame(gat_metrics)
gat_metrics_df.to_csv(OUTPUT_DIR / 'gnn_training_metrics_gat_claude_code.csv', index=False)

# ============================================================================
# 9. EXTRACT EMBEDDINGS
# ============================================================================
print("\n[8/10] Extracting GNN embeddings...")
gcn_model.eval()
with torch.no_grad():
    embeddings = gcn_model(data.x, data.edge_index, return_embeddings=True)
    embeddings_np = embeddings.cpu().numpy()

# Save embeddings
embeddings_df = pd.DataFrame(embeddings_np, index=proteins)
embeddings_df.to_csv(OUTPUT_DIR / 'protein_embeddings_gnn_claude_code.csv')
print(f"Saved embeddings: {embeddings_df.shape}")

# ============================================================================
# 10. MASTER REGULATOR IDENTIFICATION
# ============================================================================
print("\n[9/10] Identifying master regulators...")

# Method 1: Attention-based (GAT)
print("  Method 1: Attention-based (GAT)...")
gat_model.eval()
with torch.no_grad():
    _, attention_weights = gat_model(data.x, data.edge_index, return_attention=True)

# Use layer 1 attention weights (most interpretable)
attn_edge_index, attn_values = attention_weights[0]
attn_edge_index = attn_edge_index.cpu().numpy()
attn_values = attn_values.cpu().numpy()

# Sum incoming attention for each node
incoming_attention = np.zeros(len(proteins))
for i in range(attn_edge_index.shape[1]):
    target_node = attn_edge_index[1, i]
    attention = attn_values[i].mean()  # Average over heads
    incoming_attention[target_node] += attention

# Method 2: Gradient-based importance
print("  Method 2: Gradient-based importance...")
gcn_model.eval()
data.x.requires_grad = True
out, _ = gcn_model(data.x, data.edge_index)

# Compute gradient for upregulated class (class 0)
target_class = 0
gradients = torch.autograd.grad(
    outputs=out[:, target_class].sum(),
    inputs=data.x,
    create_graph=False
)[0]
gradient_importance = gradients.abs().sum(dim=1).detach().cpu().numpy()

# Method 3: PageRank on embeddings
print("  Method 3: PageRank on embeddings...")
# Build similarity graph from embeddings
embedding_sim_threshold = 0.7
embedding_edges = []
for i in range(len(proteins)):
    for j in range(i+1, len(proteins)):
        sim = 1 - cosine(embeddings_np[i], embeddings_np[j])
        if sim > embedding_sim_threshold:
            embedding_edges.append((i, j, sim))

G_embeddings = nx.Graph()
G_embeddings.add_nodes_from(range(len(proteins)))
G_embeddings.add_weighted_edges_from(embedding_edges)
pagerank_scores = nx.pagerank(G_embeddings, weight='weight')
pagerank_array = np.array([pagerank_scores.get(i, 0) for i in range(len(proteins))])

# Combine scores (normalized)
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

combined_score = (
    normalize(incoming_attention) +
    normalize(gradient_importance) +
    normalize(pagerank_array)
) / 3

# Rank proteins
master_regulators_df = pd.DataFrame({
    'Gene_Symbol': proteins,
    'Attention_Score': incoming_attention,
    'Gradient_Importance': gradient_importance,
    'PageRank_Score': pagerank_array,
    'Combined_Score': combined_score,
    'Delta_Z_Mean': delta_z_mean,
    'Matrisome_Category': protein_matrisome
})
master_regulators_df = master_regulators_df.sort_values('Combined_Score', ascending=False)
master_regulators_df.to_csv(OUTPUT_DIR / 'master_regulators_claude_code.csv', index=False)

print("\nTop 10 Master Regulators:")
print(master_regulators_df.head(10)[['Gene_Symbol', 'Combined_Score', 'Matrisome_Category', 'Delta_Z_Mean']])

# ============================================================================
# 11. PERTURBATION ANALYSIS
# ============================================================================
print("\n[10/10] Perturbation analysis...")
top_10_mr = master_regulators_df.head(10)['Gene_Symbol'].tolist()
perturbation_results = []

for mr_gene in top_10_mr:
    mr_idx = proteins.index(mr_gene)

    # Create perturbed data (mask out master regulator)
    data_perturbed = data.clone()
    data_perturbed.x[mr_idx] = 0  # Zero out features

    # Get embeddings without master regulator
    gcn_model.eval()
    with torch.no_grad():
        emb_original = gcn_model(data.x, data.edge_index, return_embeddings=True).cpu().numpy()
        emb_perturbed = gcn_model(data_perturbed.x, data_perturbed.edge_index, return_embeddings=True).cpu().numpy()

    # Compute embedding shift
    delta_embeddings = np.linalg.norm(emb_original - emb_perturbed, axis=1)

    # Count affected proteins
    affected_threshold = 0.5
    affected_count = (delta_embeddings > affected_threshold).sum()
    affected_pct = affected_count / len(proteins) * 100

    perturbation_results.append({
        'Master_Regulator': mr_gene,
        'Affected_Count': affected_count,
        'Affected_Percentage': affected_pct,
        'Max_Embedding_Shift': delta_embeddings.max(),
        'Mean_Embedding_Shift': delta_embeddings.mean()
    })

perturbation_df = pd.DataFrame(perturbation_results)
perturbation_df.to_csv(OUTPUT_DIR / 'perturbation_analysis_claude_code.csv', index=False)
print("\nPerturbation Analysis (Top 5):")
print(perturbation_df.head())

# ============================================================================
# 12. COMMUNITY DETECTION COMPARISON
# ============================================================================
print("\n[11/10] Community detection comparison...")

# Baseline: Louvain on correlation network
print("  Running Louvain algorithm...")
G_corr = nx.Graph()
G_corr.add_nodes_from(range(len(proteins)))
for i in range(edge_index.shape[1]):
    src, dst = edge_index[:, i].tolist()
    G_corr.add_edge(src, dst, weight=edge_attr[i].item())

import community as community_louvain
louvain_communities = community_louvain.best_partition(G_corr)
louvain_labels = np.array([louvain_communities[i] for i in range(len(proteins))])

# GNN-based: HDBSCAN on embeddings
print("  Running HDBSCAN on GNN embeddings...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
gnn_labels = clusterer.fit_predict(embeddings_np)

# Compute metrics
from sklearn.metrics import adjusted_rand_score, silhouette_score

ari = adjusted_rand_score(louvain_labels, gnn_labels)

# Silhouette scores (only for clustered points, exclude noise -1)
valid_gnn = gnn_labels >= 0
if valid_gnn.sum() > 1:
    silhouette_gnn = silhouette_score(embeddings_np[valid_gnn], gnn_labels[valid_gnn])
else:
    silhouette_gnn = -1

silhouette_raw = silhouette_score(node_features_scaled, louvain_labels)

# Matrisome purity
def compute_purity(labels, categories):
    unique_labels = np.unique(labels)
    purities = []
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        cluster_mask = labels == label
        cluster_cats = [categories[i] for i in range(len(categories)) if cluster_mask[i]]
        if len(cluster_cats) == 0:
            continue
        dominant_cat_count = max([cluster_cats.count(cat) for cat in set(cluster_cats)])
        purity = dominant_cat_count / len(cluster_cats)
        purities.append(purity)
    return np.mean(purities) if purities else 0

louvain_purity = compute_purity(louvain_labels, protein_matrisome)
gnn_purity = compute_purity(gnn_labels, protein_matrisome)

community_comparison = {
    'Method': ['Louvain', 'GNN_HDBSCAN'],
    'Num_Communities': [len(np.unique(louvain_labels)), len(np.unique(gnn_labels[gnn_labels >= 0]))],
    'Silhouette_Score': [silhouette_raw, silhouette_gnn],
    'Matrisome_Purity': [louvain_purity, gnn_purity],
    'ARI_vs_Louvain': [1.0, ari]
}
community_df = pd.DataFrame(community_comparison)
community_df.to_csv(OUTPUT_DIR / 'community_comparison_claude_code.csv', index=False)
print("\nCommunity Comparison:")
print(community_df)

# ============================================================================
# 13. VISUALIZATIONS
# ============================================================================
print("\n[12/10] Creating visualizations...")

# 1. Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(gcn_metrics_df['epoch'], gcn_metrics_df['train_loss'], label='Train Loss')
axes[0].plot(gcn_metrics_df['epoch'], gcn_metrics_df['val_acc'], label='Val Acc')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Metric')
axes[0].set_title('GCN Training Curves')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(gat_metrics_df['epoch'], gat_metrics_df['train_loss'], label='Train Loss')
axes[1].plot(gat_metrics_df['epoch'], gat_metrics_df['val_acc'], label='Val Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Metric')
axes[1].set_title('GAT Training Curves')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'training_curves_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. UMAP of embeddings
print("  Creating UMAP visualization...")
reducer = umap.UMAP(n_components=2, random_state=SEED)
embeddings_2d = reducer.fit_transform(embeddings_np)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Color by aging direction
colors_aging = ['red' if dz > 0.5 else 'blue' if dz < -0.5 else 'gray' for dz in delta_z_mean]
axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_aging, alpha=0.6, s=20)
axes[0].set_title('UMAP of GNN Embeddings (by Aging Direction)')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')

# Color by Matrisome category
category_to_color = {cat: plt.cm.Set1(i) for i, cat in enumerate(matrisome_categories)}
colors_matrisome = [category_to_color.get(cat, (0.5, 0.5, 0.5)) for cat in protein_matrisome]
axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_matrisome, alpha=0.6, s=20)
axes[1].set_title('UMAP of GNN Embeddings (by Matrisome Category)')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'gnn_umap_embeddings_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Attention heatmap (top 50 proteins)
print("  Creating attention heatmap...")
top_50_idx = master_regulators_df.head(50).index.tolist()
top_50_proteins = [proteins[i] for i in range(len(proteins)) if i in top_50_idx]

# Build attention matrix for top 50
attention_matrix = np.zeros((len(top_50_idx), len(top_50_idx)))
for i in range(attn_edge_index.shape[1]):
    src, dst = attn_edge_index[:, i]
    if src in top_50_idx and dst in top_50_idx:
        src_pos = top_50_idx.index(src)
        dst_pos = top_50_idx.index(dst)
        attention_matrix[dst_pos, src_pos] = attn_values[i].mean()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(attention_matrix, cmap='Reds', xticklabels=top_50_proteins, yticklabels=top_50_proteins,
            cbar_kws={'label': 'Attention Weight'}, ax=ax)
ax.set_title('Attention Weights Heatmap (Top 50 Master Regulators)')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'attention_heatmap_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Network graph with master regulators highlighted
print("  Creating network graph...")
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.spring_layout(G_corr, k=0.5, iterations=50, seed=SEED)

# Draw edges
nx.draw_networkx_edges(G_corr, pos, alpha=0.1, width=0.5, ax=ax)

# Draw nodes
node_colors = []
node_sizes = []
for i, protein in enumerate(proteins):
    if protein in top_10_mr[:5]:  # Top 5 master regulators
        node_colors.append('red')
        node_sizes.append(300)
    elif protein in top_10_mr[5:]:  # Next 5 master regulators
        node_colors.append('orange')
        node_sizes.append(150)
    else:
        node_colors.append('lightgray')
        node_sizes.append(20)

nx.draw_networkx_nodes(G_corr, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7, ax=ax)

# Labels for top 10
top_10_pos = {i: pos[i] for i in range(len(proteins)) if proteins[i] in top_10_mr}
top_10_labels = {i: proteins[i] for i in range(len(proteins)) if proteins[i] in top_10_mr}
nx.draw_networkx_labels(G_corr, top_10_pos, top_10_labels, font_size=8, ax=ax)

# Legend
red_patch = mpatches.Patch(color='red', label='Top 5 Master Regulators')
orange_patch = mpatches.Patch(color='orange', label='Top 6-10 Master Regulators')
gray_patch = mpatches.Patch(color='lightgray', label='Other Proteins')
ax.legend(handles=[red_patch, orange_patch, gray_patch], loc='upper right')

ax.set_title('Protein Correlation Network with Master Regulators Highlighted')
ax.axis('off')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'network_graph_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Community comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Louvain communities
colors_louvain = [plt.cm.tab20(louvain_labels[i] % 20) for i in range(len(proteins))]
axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_louvain, alpha=0.6, s=20)
axes[0].set_title(f'Louvain Communities (n={len(np.unique(louvain_labels))})')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')

# GNN communities
colors_gnn = [plt.cm.tab20(gnn_labels[i] % 20) if gnn_labels[i] >= 0 else (0.5, 0.5, 0.5) for i in range(len(proteins))]
axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors_gnn, alpha=0.6, s=20)
axes[1].set_title(f'GNN Communities (n={len(np.unique(gnn_labels[gnn_labels >= 0]))})')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'community_comparison_claude_code.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"Visualizations saved to: {VIZ_DIR}")
print("\nKey files:")
print(f"  - master_regulators_claude_code.csv")
print(f"  - protein_embeddings_gnn_claude_code.csv")
print(f"  - perturbation_analysis_claude_code.csv")
print(f"  - community_comparison_claude_code.csv")
print(f"  - visualizations_claude_code/")
