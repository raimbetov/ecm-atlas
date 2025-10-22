import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
METADATA_PATH = Path('../../../iteration_03/hypothesis_09_temporal_rnn_trajectories/codex/analysis_metadata_codex.json')
OUTPUT_DIR = Path('outputs_codex')
VIS_DIR = Path('visualizations_codex')

OUTPUT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

with open(METADATA_PATH) as f:
    metadata = json.load(f)
order = metadata['pseudo_time_order']
order_scores = {k: metadata['pseudo_time_scores'][k] for k in order}

df = pd.read_csv(DATA_PATH)
# Average duplicate protein measurements per tissue
matrix = df.pivot_table(index='Tissue', columns='Gene_Symbol', values='Zscore_Delta', aggfunc='mean')
matrix = matrix.loc[order]

score_vec = np.array([order_scores[t] for t in matrix.index])
vals = matrix.to_numpy()

gradients = np.full_like(vals, np.nan)
for i in range(1, len(order) - 1):
    prev_vals = vals[i - 1]
    next_vals = vals[i + 1]
    dt = score_vec[i + 1] - score_vec[i - 1]
    gradients[i] = (next_vals - prev_vals) / dt

grad_df = pd.DataFrame(gradients, index=matrix.index, columns=matrix.columns)
grad_df.to_csv(OUTPUT_DIR / 'gradient_matrix_codex.csv')

stacked = grad_df.stack(dropna=True).rename('gradient').reset_index()
stacked['abs_gradient'] = stacked['gradient'].abs()
idx = stacked.groupby('Gene_Symbol')['abs_gradient'].idxmax()
max_gradients = stacked.loc[idx].copy()
max_gradients = max_gradients.sort_values('abs_gradient', ascending=False)
max_gradients.to_csv(OUTPUT_DIR / 'max_gradient_assignments_codex.csv', index=False)

ovary_hits = max_gradients[max_gradients['Tissue'] == 'Ovary_Cortex']
ovary_hits = ovary_hits.assign(zscore=matrix.loc['Ovary_Cortex'].reindex(ovary_hits['Gene_Symbol']).values)
ovary_hits.to_csv(OUTPUT_DIR / 'ovary_specific_gradients_codex.csv', index=False)

heart_hits = max_gradients[max_gradients['Tissue'] == 'Heart_Native_Tissue']
heart_hits = heart_hits.assign(zscore=matrix.loc['Heart_Native_Tissue'].reindex(heart_hits['Gene_Symbol']).values)
heart_hits.to_csv(OUTPUT_DIR / 'heart_specific_gradients_codex.csv', index=False)

estrogen_targets = {
    'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'COL6A1', 'COL6A2', 'COL6A3', 'FN1', 'FBN1',
    'LAMA2', 'LAMB1', 'LAMB2', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4', 'MMP2', 'MMP9', 'PLOD1',
    'PLOD2', 'PLOD3', 'SERPINE1', 'TIMP1', 'TIMP3', 'TNC', 'VCAN', 'POSTN', 'THBS1'
}
yap_taz_targets = {
    'CTGF', 'CYR61', 'COL1A1', 'COL1A2', 'COL3A1', 'ITGA5', 'ITGB3', 'FN1', 'LAMC1', 'LAMB1',
    'TAGLN', 'SERPINE1', 'VCAN', 'TNC', 'THBS1', 'COL5A1', 'COL5A2', 'COL6A3', 'POSTN'
}
metabolic_ecm = {
    'LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4', 'TGM1', 'TGM2', 'PLOD1', 'PLOD2', 'PLOD3',
    'P4HA1', 'P4HA2', 'ANGPTL2', 'ANGPTL4', 'COL6A1', 'COL6A2', 'COL6A3'
}

def summarize_pathway(genes, tissue):
    present = sorted(genes.intersection(matrix.columns))
    sub = grad_df.loc[tissue, present]
    zvals = matrix.loc[tissue, present]
    return pd.DataFrame({'Gene_Symbol': present, 'gradient': sub.values, 'zscore': zvals.values})

estrogen_summary = summarize_pathway(estrogen_targets, 'Ovary_Cortex')
estrogen_summary.to_csv(OUTPUT_DIR / 'estrogen_gradient_profile_codex.csv', index=False)

yap_summary = summarize_pathway(yap_taz_targets, 'Heart_Native_Tissue')
yap_summary.to_csv(OUTPUT_DIR / 'yap_taz_gradient_profile_codex.csv', index=False)

metabolic_ovary = summarize_pathway(metabolic_ecm, 'Ovary_Cortex')
metabolic_heart = summarize_pathway(metabolic_ecm, 'Heart_Native_Tissue')
merged_metabolic = metabolic_ovary.merge(metabolic_heart, on='Gene_Symbol', suffixes=('_ovary', '_heart'))
merged_metabolic['gradient_mean'] = merged_metabolic[['gradient_ovary', 'gradient_heart']].mean(axis=1)
merged_metabolic.to_csv(OUTPUT_DIR / 'metabolic_overlap_codex.csv', index=False)

grad_pairs = merged_metabolic[['gradient_ovary', 'gradient_heart']].dropna()
zscore_pairs = merged_metabolic[['zscore_ovary', 'zscore_heart']].dropna()
metabolic_corr = float(grad_pairs.corr(method='spearman').iloc[0, 1]) if len(grad_pairs) > 1 else None
metabolic_z_corr = float(zscore_pairs.corr(method='spearman').iloc[0, 1]) if len(zscore_pairs) > 1 else None

# Autoencoder on gene-by-tissue matrix (genes as samples)
gene_matrix = matrix.transpose().rename_axis('Gene_Symbol').reset_index()
features = gene_matrix.drop(columns=['Gene_Symbol']).to_numpy(dtype=np.float32)
features = np.nan_to_num(features, nan=0.0)
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

torch.manual_seed(42)
inputs = torch.tensor(features_std, dtype=torch.float32)
input_dim = inputs.shape[1]
latent_dim = min(8, max(2, input_dim // 2))

class Autoencoder(torch.nn.Module):
    def __init__(self, in_dim, latent):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, max(16, in_dim // 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(16, in_dim // 2), latent)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent, max(16, in_dim // 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(16, in_dim // 2), in_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

autoencoder = Autoencoder(input_dim, latent_dim)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    recon, latent = autoencoder(inputs)
    loss = criterion(recon, inputs)
    loss.backward()
    optimizer.step()

autoencoder.eval()
with torch.no_grad():
    reconstructed, latent = autoencoder(inputs)
latent_np = latent.numpy()
recon_error = torch.mean((inputs - reconstructed) ** 2, dim=1).numpy()

latent_df = pd.DataFrame(latent_np, columns=[f'latent_{i+1}' for i in range(latent_np.shape[1])])
latent_df.insert(0, 'Gene_Symbol', gene_matrix['Gene_Symbol'])
latent_df['reconstruction_error'] = recon_error
latent_df.to_csv(OUTPUT_DIR / 'autoencoder_gene_latents_codex.csv', index=False)

# Spectral clustering on gradient patterns for genes
valid_grad = grad_df.dropna(axis=1, how='all').transpose()
valid_grad = valid_grad.fillna(0.0)
scaler_grad = StandardScaler()
features_grad = scaler_grad.fit_transform(valid_grad)
cluster_model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
clusters = cluster_model.fit_predict(features_grad)
cluster_df = pd.DataFrame({'Gene_Symbol': valid_grad.index, 'cluster': clusters})
cluster_df.to_csv(OUTPUT_DIR / 'spectral_clusters_codex.csv', index=False)

# Build correlation network among key pathways
target_genes = sorted((estrogen_targets | yap_taz_targets | metabolic_ecm).intersection(matrix.columns))
submatrix = matrix[target_genes].copy()
submatrix = submatrix.fillna(0.0)
corr = submatrix.corr(method='pearson')
threshold = 0.6
G = nx.Graph()
for gene in target_genes:
    G.add_node(gene)
for i, gene_i in enumerate(target_genes):
    for j in range(i + 1, len(target_genes)):
        gene_j = target_genes[j]
        val = corr.at[gene_i, gene_j]
        if np.abs(val) >= threshold:
            G.add_edge(gene_i, gene_j, weight=val)

degree = dict(G.degree())
betweenness = nx.betweenness_centrality(G, weight='weight')
closeness = nx.closeness_centrality(G)
network_df = pd.DataFrame({
    'Gene_Symbol': list(G.nodes()),
    'degree': [degree[g] for g in G.nodes()],
    'betweenness': [betweenness[g] for g in G.nodes()],
    'closeness': [closeness[g] for g in G.nodes()]
})
network_df = network_df.sort_values('degree', ascending=False)
network_df.to_csv(OUTPUT_DIR / 'pathway_network_metrics_codex.csv', index=False)

# Heatmap to visualize gradients of key genes
heatmap_genes = estrogen_targets.union(yap_taz_targets).union(metabolic_ecm)
heatmap_genes = [g for g in heatmap_genes if g in grad_df.columns]
heatmap = grad_df[heatmap_genes]
plt.figure(figsize=(12, 6))
plt.title('Gradient signatures for ovary-heart pathways')
plt.imshow(heatmap.transpose(), aspect='auto', cmap='coolwarm', interpolation='nearest')
plt.yticks(range(len(heatmap_genes)), heatmap_genes)
plt.xticks(range(len(heatmap.index)), heatmap.index, rotation=90)
plt.colorbar(label='Gradient')
plt.tight_layout()
plt.savefig(VIS_DIR / 'gradient_heatmap_codex.png', dpi=300)
plt.close()

summary = {
    'num_ovary_specific': int(len(ovary_hits)),
    'num_heart_specific': int(len(heart_hits)),
    'estrogen_present': len(estrogen_summary),
    'yap_present': len(yap_summary),
    'metabolic_overlap_genes': len(merged_metabolic),
    'metabolic_gradient_pairs': len(grad_pairs),
    'metabolic_gradient_spearman': metabolic_corr,
    'metabolic_zscore_pairs': len(zscore_pairs),
    'metabolic_zscore_spearman': metabolic_z_corr,
    'autoencoder_latent_dim': latent_np.shape[1],
    'mean_recon_error': float(recon_error.mean()),
    'network_nodes': G.number_of_nodes(),
    'network_edges': G.number_of_edges()
}
pd.Series(summary).to_json(OUTPUT_DIR / 'analysis_summary_codex.json', indent=2)
