import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
import shap
import hdbscan
import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns

WORKSPACE = Path(__file__).resolve().parent
DATA_PATH = Path('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
VELOCITY_PATH = Path('../../../iteration_01/hypothesis_03_tissue_aging_clocks/codex/tissue_aging_velocity_codex.csv').resolve()
OUTPUT_DIR = WORKSPACE
VIS_DIR = WORKSPACE / 'visualizations_codex'
VIS_DIR.mkdir(exist_ok=True)

PHASE_ASSIGNMENTS = OUTPUT_DIR / 'phase_assignments_codex.csv'
CLASSIFICATION_PERF = OUTPUT_DIR / 'classification_performance_codex.csv'
TRANSITION_PRED = OUTPUT_DIR / 'transition_prediction_codex.csv'
ENRICHMENT_FILE = OUTPUT_DIR / 'enrichment_analysis_codex.csv'
RF_MODEL_PATH = OUTPUT_DIR / 'phase_classifier_codex.pkl'
AUTOENCODER_WEIGHTS = VIS_DIR / 'autoencoder_weights_codex.pth'
GCN_WEIGHTS = VIS_DIR / 'gcn_weights_codex.pth'
LATENT_EMBEDDINGS_FILE = OUTPUT_DIR / 'latent_embeddings_codex.csv'
HDBSCAN_FILE = OUTPUT_DIR / 'hdbscan_clusters_codex.csv'
TSNE_FILE = VIS_DIR / 'tsne_latent_codex.png'
SHAP_FILE = VIS_DIR / 'shap_summary_codex.png'
VELOCITY_DISTRIBUTION = VIS_DIR / 'velocity_distribution_codex.png'
ENRICHMENT_HEATMAP = VIS_DIR / 'enrichment_heatmap_codex.png'
PROTEIN_TRAJ_PLOT = VIS_DIR / 'protein_trajectories_codex.png'

THRESHOLD = 1.65
STATIC_METABOLIC_GENES = {
    'ATP5A1', 'ATP5B', 'ATP5C1', 'ATP5D', 'ATP5F1', 'COX4I1', 'COX5A', 'COX6A1',
    'NDUFA9', 'NDUFS1', 'NDUFS2', 'SDHB', 'SDHC', 'UQCRC1', 'UQCRC2', 'CS',
    'IDH3A', 'PDHA1', 'PDHB', 'HK1', 'HK2', 'PFKM', 'PFKP', 'ALDOA', 'GAPDH',
    'PGK1', 'ENO1', 'PKM', 'LDHA', 'LDHB', 'SOD2', 'PRDX3', 'ACLY', 'ACAT2'
}
STATIC_MECHANICAL_GENES = {
    'LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'TGM1', 'TGM2', 'TGM3', 'COL1A1', 'COL1A2',
    'COL3A1', 'COL5A1', 'COL5A2', 'COL6A1', 'COL6A2', 'COL6A3', 'COL14A1',
    'COL15A1', 'FN1', 'SPP1', 'POSTN', 'TGFB1', 'TGFB2', 'TGFB3', 'ITGB1',
    'ITGA5', 'ITGAV', 'ACTA2', 'YAP1', 'WWTR1', 'ROCK1', 'ROCK2', 'TEAD1',
    'TEAD2', 'TEAD3', 'TEAD4', 'CTGF', 'PLOD1', 'PLOD2'
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


class SimpleGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def gcn_layer(x, adj):
        return adj @ x

    def forward(self, x, adj):
        x = self.gcn_layer(x, adj)
        x = torch.relu(self.fc1(x))
        x = self.gcn_layer(x, adj)
        x = self.fc2(x)
        return x


def load_data():
    df = pd.read_csv(DATA_PATH)
    velocity_df = pd.read_csv(VELOCITY_PATH)
    velocity_df['Phase'] = np.where(velocity_df['Velocity'] < THRESHOLD, 'Phase_I', 'Phase_II')
    velocity_df.to_csv(PHASE_ASSIGNMENTS, index=False)
    tissue_map = velocity_df.set_index('Tissue')['Velocity'].to_dict()
    phase_map = velocity_df.set_index('Tissue')['Phase'].to_dict()
    pivot = (df.pivot_table(index='Tissue', columns='Gene_Symbol', values='Zscore_Delta', aggfunc='mean'))
    pivot = pivot.reindex(tissue_map.keys())
    imputer = SimpleImputer(strategy='constant', fill_value=0.0)
    features = imputer.fit_transform(pivot)
    tissues = pivot.index.tolist()
    velocities = np.array([tissue_map.get(t, np.nan) for t in tissues])
    phases = np.array([phase_map.get(t, None) for t in tissues])
    valid_mask = ~np.isnan(velocities)
    features = features[valid_mask]
    tissues = [tissues[i] for i, keep in enumerate(valid_mask) if keep]
    velocities = velocities[valid_mask]
    phases = phases[valid_mask]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    metadata = {
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_var': scaler.var_.tolist(),
        'tissues': tissues
    }
    with open(OUTPUT_DIR / 'analysis_metadata_codex.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    return features_scaled.astype(np.float32), tissues, velocities.astype(np.float32), phases, pivot.columns.tolist()


def train_autoencoder(features: np.ndarray, epochs: int = 200, latent_dim: int = 32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_data = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = Autoencoder(features.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs} MSE {epoch_loss / len(dataset):.4f}')
    model.eval()
    with torch.no_grad():
        _, latent = model(tensor_data.to(device))
    torch.save(model.state_dict(), AUTOENCODER_WEIGHTS)
    return latent.cpu().numpy(), model


def build_adjacency(latent: np.ndarray, k: int = 3):
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(latent)
    np.fill_diagonal(sim, 0.0)
    adj = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        idx = np.argsort(sim[i])[-k:]
        adj[i, idx] = sim[i, idx]
    # Symmetrize
    adj = np.maximum(adj, adj.T)
    degree = np.sum(adj, axis=1)
    degree[degree == 0] = 1.0
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt
    return torch.tensor(adj_norm, dtype=torch.float32)


def train_gcn(latent: np.ndarray, labels: np.ndarray, tissues: list):
    label_map = {'Phase_I': 0, 'Phase_II': 1}
    y = torch.tensor([label_map[l] for l in labels], dtype=torch.long)
    x = torch.tensor(latent, dtype=torch.float32)
    adj = build_adjacency(latent)
    model = SimpleGCN(latent.shape[1], hidden_dim=64, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(500):
        optimizer.zero_grad()
        logits = model(x, adj)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
            print(f'GCN epoch {epoch+1}/500 loss={loss.item():.4f} acc={acc:.3f}')
    torch.save(model.state_dict(), GCN_WEIGHTS)
    with torch.no_grad():
        logits = model(x, adj)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return probs


def run_hdbscan(latent: np.ndarray, tissues: list):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
    labels = clusterer.fit_predict(latent)
    pd.DataFrame({'Tissue': tissues, 'HDBSCAN_Cluster': labels}).to_csv(HDBSCAN_FILE, index=False)
    return labels


def random_forest_classification(features: np.ndarray, labels: np.ndarray, tissues: list, feature_names: list):
    label_map = {'Phase_I': 0, 'Phase_II': 1}
    y = np.array([label_map[l] for l in labels])
    skf = StratifiedKFold(n_splits=min(5, sum(y) if sum(y) > 1 else 2), shuffle=True, random_state=42)
    candidate_models = {
        'RandomForest': RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    performance_rows = []
    best_model_name = None
    best_auc = -np.inf
    best_model = None

    for name, model in candidate_models.items():
        fold_metrics = []
        for train_idx, test_idx in skf.split(features, y):
            model.fit(features[train_idx], y[train_idx])
            prob = model.predict_proba(features[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_metrics.append({
                'Model': name,
                'Accuracy': accuracy_score(y[test_idx], pred),
                'F1': f1_score(y[test_idx], pred),
                'ROC_AUC': roc_auc_score(y[test_idx], prob)
            })
        model_df = pd.DataFrame(fold_metrics)
        performance_rows.append({
            'Model': name,
            'Accuracy': model_df['Accuracy'].mean(),
            'F1': model_df['F1'].mean(),
            'ROC_AUC': model_df['ROC_AUC'].mean()
        })
        model_auc = performance_rows[-1]['ROC_AUC']
        if model_auc > best_auc:
            best_auc = model_auc
            best_model_name = name
            best_model = model.__class__(**model.get_params())

    pd.DataFrame(performance_rows).to_csv(CLASSIFICATION_PERF, index=False)

    if best_model is None:
        best_model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
        best_model_name = 'RandomForest'

    final_clf = best_model
    final_clf.fit(features, y)
    with open(RF_MODEL_PATH, 'wb') as f:
        pickle.dump({'model': final_clf, 'feature_names': feature_names}, f)

    feature_df = pd.DataFrame(features, columns=feature_names)
    explainer = shap.TreeExplainer(final_clf)
    shap_values = explainer.shap_values(feature_df)
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    elif getattr(shap_values, 'ndim', 0) == 3:
        shap_matrix = shap_values[:, :, 1]
    else:
        shap_matrix = shap_values
    shap.summary_plot(shap_matrix, feature_df, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_FILE, dpi=300)
    plt.close()

    shap_importance = np.abs(shap_matrix).mean(axis=0)
    top_idx = np.argsort(shap_importance)[::-1][:30]
    pd.DataFrame({
        'Feature': np.array(feature_names)[top_idx],
        'Mean_ABS_SHAP': shap_importance[top_idx]
    }).to_csv(OUTPUT_DIR / 'shap_feature_rank_codex.csv', index=False)

    prob_all = final_clf.predict_proba(features)[:, 1]
    pred_all = (prob_all >= 0.5).astype(int)
    pd.DataFrame({
        'Tissue': tissues,
        'True_Label': y,
        'Predicted_Label': pred_all,
        'Predicted_Prob_PhaseII': prob_all,
        'Model': best_model_name
    }).to_csv(OUTPUT_DIR / 'rf_predictions_codex.csv', index=False)


def compute_transition_prediction(latent: np.ndarray, velocities: np.ndarray, tissues: list):
    from sklearn.ensemble import GradientBoostingRegressor
    y = np.maximum(0.0, THRESHOLD - velocities)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(latent, y)
    preds = model.predict(latent)
    r2 = model.score(latent, y)
    pd.DataFrame({
        'Tissue': tissues,
        'Distance_To_Threshold': y,
        'Predicted_Distance': preds
    }).to_csv(TRANSITION_PRED, index=False)
    return r2


def derive_marker_genes(df: pd.DataFrame):
    names = df[['Gene_Symbol', 'Protein_Name', 'Matrisome_Category_Simplified']].dropna().copy()
    names['Protein_Name'] = names['Protein_Name'].str.lower()
    names['Matrisome_Category_Simplified'] = names['Matrisome_Category_Simplified'].str.lower()

    metabolic_genes = {g for g in STATIC_METABOLIC_GENES if g in names['Gene_Symbol'].values}
    mechanical_genes = {g for g in STATIC_MECHANICAL_GENES if g in names['Gene_Symbol'].values}

    if len(metabolic_genes) < 5:
        metabolic_categories = names[names['Matrisome_Category_Simplified'].str.contains('regulator|secreted', na=False)]
        metabolic_keywords = ('peroxid', 'dehydrogenase', 'oxidase', 'glycol', 'nad', 'succinate', 'mitochond')
        metabolic_kw = names[names['Protein_Name'].str.contains('|'.join(metabolic_keywords), na=False)]
        metabolic_genes = set(metabolic_categories['Gene_Symbol']).union(set(metabolic_kw['Gene_Symbol']))

    if len(mechanical_genes) < 5:
        mechanical_categories = names[names['Matrisome_Category_Simplified'].str.contains('collagen|glycoprotein|proteoglycan', na=False)]
        mechanical_keywords = ('collagen', 'lysyl oxidase', 'transglutaminase', 'fibronectin', 'integrin', 'crosslink', 'elastic')
        mechanical_kw = names[names['Protein_Name'].str.contains('|'.join(mechanical_keywords), na=False)]
        mechanical_genes = set(mechanical_categories['Gene_Symbol']).union(set(mechanical_kw['Gene_Symbol']))
    return metabolic_genes, mechanical_genes


def enrichment_analysis(df: pd.DataFrame, tissues: list, phases: np.ndarray):
    metabolic_genes, mechanical_genes = derive_marker_genes(df)
    results = []
    pivot = df.pivot_table(index='Tissue', columns='Gene_Symbol', values='Zscore_Delta', aggfunc='mean')
    pivot = pivot.loc[tissues]
    for label, genes in [('Metabolic', metabolic_genes), ('Mechanical', mechanical_genes)]:
        common_genes = [g for g in genes if g in pivot.columns]
        if not common_genes:
            continue
        mean_values = pivot[common_genes].mean(axis=1)
        table = pd.DataFrame({
            'Tissue': tissues,
            'Mean_Score': mean_values,
            'Phase': phases
        })
        phase_groups = table.groupby('Phase')['Mean_Score'].mean()
        high_phase = phase_groups.idxmax()
        from scipy.stats import fisher_exact
        threshold = table['Mean_Score'].median()
        table['High'] = table['Mean_Score'] >= threshold
        contingency = pd.crosstab(table['Phase'], table['High'])
        if contingency.shape != (2, 2):
            odds_ratio, p_value = np.nan, np.nan
        else:
            odds_ratio, p_value = fisher_exact(contingency)
        results.append({
            'Marker_Set': label,
            'Preferred_Phase': high_phase,
            'Mean_Phase_I': phase_groups.get('Phase_I', np.nan),
            'Mean_Phase_II': phase_groups.get('Phase_II', np.nan),
            'Odds_Ratio': odds_ratio,
            'P_Value': p_value,
            'Genes_Considered': len(common_genes)
        })
    pd.DataFrame(results).to_csv(ENRICHMENT_FILE, index=False)
    if results:
        plot_df = pd.DataFrame(results).set_index('Marker_Set')[['Mean_Phase_I', 'Mean_Phase_II']]
        plot_df.plot(kind='bar', figsize=(6, 4))
        plt.ylabel('Mean ΔZ score')
        plt.title('Phase-Specific Marker Enrichment')
        plt.tight_layout()
        plt.savefig(ENRICHMENT_HEATMAP, dpi=300)
        plt.close()

    # Save per-gene trajectories
    selected_genes = list(metabolic_genes.union(mechanical_genes))
    selected_genes = [g for g in selected_genes if g in pivot.columns]
    melted = pivot[selected_genes].reset_index().melt(id_vars='Tissue', var_name='Gene', value_name='DeltaZ')
    meta = pd.read_csv(PHASE_ASSIGNMENTS)
    merged = melted.merge(meta[['Tissue', 'Velocity', 'Phase']], on='Tissue', how='left')
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=merged, x='Velocity', y='DeltaZ', hue='Phase', style=merged['Gene'].isin(metabolic_genes))
    plt.axvline(THRESHOLD, color='red', linestyle='--', label='v=1.65')
    plt.title('Marker Trajectories Across Velocity')
    plt.tight_layout()
    plt.savefig(PROTEIN_TRAJ_PLOT, dpi=300)
    plt.close()



def plot_velocity_distribution(velocities: np.ndarray):
    plt.figure(figsize=(6, 4))
    sns.histplot(velocities, bins=20, kde=True)
    plt.axvline(THRESHOLD, color='red', linestyle='--')
    plt.xlabel('Velocity')
    plt.ylabel('Count of tissues')
    plt.tight_layout()
    plt.savefig(VELOCITY_DISTRIBUTION, dpi=300)
    plt.close()


def run_tsne(latent: np.ndarray, phases: np.ndarray):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, latent.shape[0] - 1))
    coords = tsne.fit_transform(latent)
    df = pd.DataFrame({'Dim1': coords[:, 0], 'Dim2': coords[:, 1], 'Phase': phases})
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x='Dim1', y='Dim2', hue='Phase')
    plt.title('t-SNE of Autoencoder Latent Space')
    plt.tight_layout()
    plt.savefig(TSNE_FILE, dpi=300)
    plt.close()


def main():
    features, tissues, velocities, phases, feature_names = load_data()
    plot_velocity_distribution(velocities)
    latent, ae_model = train_autoencoder(features)
    pd.DataFrame(latent, index=tissues).reset_index().rename(columns={'index': 'Tissue'}).to_csv(
        LATENT_EMBEDDINGS_FILE, index=False
    )
    gcn_probs = train_gcn(latent, phases, tissues)
    hdbscan_labels = run_hdbscan(latent, tissues)
    df = pd.read_csv(DATA_PATH)
    metabolic_genes, mechanical_genes = derive_marker_genes(df)
    pivot = df.pivot_table(index='Tissue', columns='Gene_Symbol', values='Zscore_Delta', aggfunc='mean').reindex(tissues).fillna(0.0)
    metabolic_cols = [g for g in metabolic_genes if g in pivot.columns]
    mechanical_cols = [g for g in mechanical_genes if g in pivot.columns]
    metabolic_scores = pivot[metabolic_cols].mean(axis=1).fillna(0.0).values if metabolic_cols else np.zeros(len(tissues))
    mechanical_scores = pivot[mechanical_cols].mean(axis=1).fillna(0.0).values if mechanical_cols else np.zeros(len(tissues))
    hybrid_features = np.column_stack([
        latent,
        metabolic_scores,
        mechanical_scores,
        metabolic_scores - mechanical_scores
    ])
    hybrid_feature_names = [f'Latent_{i}' for i in range(latent.shape[1])] + [
        'Metabolic_Score', 'Mechanical_Score', 'Metabolic_Minus_Mechanical'
    ]
    random_forest_classification(hybrid_features, phases, tissues, hybrid_feature_names)
    r2 = compute_transition_prediction(latent, velocities, tissues)
    run_tsne(latent, phases)
    enrichment_analysis(df, tissues, phases)
    outputs = {
        'num_tissues': len(tissues),
        'autoencoder_latent_dim': latent.shape[1],
        'gcn_phaseII_probs': dict(zip(tissues, gcn_probs.tolist())),
        'hdbscan_unique_clusters': int(len(set(hdbscan_labels))),
        'transition_prediction_r2': float(r2)
    }
    with open(OUTPUT_DIR / 'analysis_summary_codex.json', 'w') as f:
        json.dump(outputs, f, indent=2)


if __name__ == '__main__':
    main()
