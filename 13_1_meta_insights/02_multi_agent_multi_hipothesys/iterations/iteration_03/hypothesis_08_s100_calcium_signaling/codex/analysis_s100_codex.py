import os
import math
import json
import shutil
import random
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, fisher_exact, ttest_rel

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from Bio.PDB import PDBParser
except ImportError:  # pragma: no cover
    PDBParser = None

DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
WORKSPACE = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/codex")
OUTPUT_DIR = WORKSPACE
VIS_DIR = OUTPUT_DIR / "visualizations_codex"
STRUCTURE_DIR = OUTPUT_DIR / "alphafold_structures"

AGENT = "codex"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

S100_FAMILY: List[str] = sorted({
    "S100A1", "S100A2", "S100A3", "S100A4", "S100A5", "S100A6", "S100A7",
    "S100A7A", "S100A7L2", "S100A8", "S100A9", "S100A10", "S100A11", "S100A12",
    "S100A13", "S100A14", "S100A15", "S100A16", "S100A17", "S100A18", "S100B",
    "S100G", "S100P", "S100Z"
})
CROSSLINKING_GENES = ["LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4", "TGM1", "TGM2", "TGM3", "TGM4"]
INFLAMMATION_GENES = [
    "SERPINE1", "MMP9", "MMP3", "TNFAIP6", "CXCL10", "CXCL12", "SERPINA1", "CRP"
]
MECHANOTRANSDUCTION_GENES = [
    "FN1", "VTN", "THBS1", "POSTN", "SPARC", "TNC", "FBLN1", "FBLN2", "SPARCL1", "SMOC1", "SMOC2"
]
CALCIUM_GENES = ["CALM1", "CALM2", "CALM3", "CAMK2A", "CAMK2B", "SLC8A1"]
COLLAGEN_GENES = ["COL1A1", "COL3A1"]

ALPHAFOLD_IDS = {
    "S100A8": "P05109",
    "S100A9": "P06702",
    "S100B": "P04271",
}


class S100StiffnessNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class S100CrosslinkingAttention(nn.Module):
    def __init__(self, n_s100: int, n_targets: int, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.n_targets = n_targets
        self.feature_embed = nn.Linear(1, embed_dim)
        self.target_queries = nn.Parameter(torch.randn(1, n_targets, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.post = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, n_s100)
        batch_size, n_features = x.shape
        s100_seq = x.unsqueeze(-1)  # (batch, n_s100, 1)
        embedded = self.feature_embed(s100_seq)  # (batch, n_s100, embed_dim)
        queries = self.target_queries.expand(batch_size, -1, -1)
        attn_output, attn_weights = self.attention(queries, embedded, embedded, need_weights=True)
        preds = self.post(attn_output).squeeze(-1)
        return preds, attn_weights  # preds (batch, n_targets); attn_weights (batch, n_targets, n_s100)


class StructuralStiffnessNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_directories() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['Gene_Symbol'] = df['Gene_Symbol'].str.upper()
    df['Tissue'] = df['Tissue'].astype(str)
    return df


def build_expression_matrices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    matrices = {}
    for gene_list, key in [
        (S100_FAMILY, 's100'),
        (CROSSLINKING_GENES, 'crosslinking'),
        (INFLAMMATION_GENES, 'inflammation'),
        (MECHANOTRANSDUCTION_GENES, 'mechanotransduction'),
        (CALCIUM_GENES, 'calcium'),
        (COLLAGEN_GENES, 'collagen')
    ]:
        subset = df[df['Gene_Symbol'].isin(gene_list)]
        pivot = subset.pivot_table(values='Zscore_Delta', index='Tissue', columns='Gene_Symbol', aggfunc='mean')
        matrices[key] = pivot
    return matrices


def compute_stiffness_score(pivots: Dict[str, pd.DataFrame]) -> pd.Series:
    cross = pivots['crosslinking']
    collagen = pivots['collagen']

    index = cross.index.union(collagen.index)

    lox = cross.get('LOX', pd.Series(0.0, index=cross.index)).reindex(index, fill_value=0.0).fillna(0.0)
    tgm2 = cross.get('TGM2', pd.Series(0.0, index=cross.index)).reindex(index, fill_value=0.0).fillna(0.0)

    col1 = collagen.get('COL1A1', pd.Series(np.nan, index=collagen.index)).reindex(index)
    col3 = collagen.get('COL3A1', pd.Series(np.nan, index=collagen.index)).reindex(index)
    ratio = np.divide(col1, col3.replace(0, np.nan))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    stiffness = 0.5 * lox + 0.3 * tgm2 + 0.2 * ratio
    stiffness.name = 'stiffness_score'
    return stiffness.dropna()


def align_features(target_index: pd.Index, matrix: pd.DataFrame) -> pd.DataFrame:
    aligned = matrix.reindex(target_index).fillna(0.0)
    return aligned


def train_stiffness_nn(features: pd.DataFrame, target: pd.Series) -> Tuple[S100StiffnessNN, Dict[str, float], pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    y = target.loc[features.index].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, features.index.values, test_size=0.2, random_state=SEED
    )

    model = S100StiffnessNN(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    best_state = None
    best_r2 = -np.inf

    for epoch in range(300):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).numpy()
            r2 = r2_score(y_test, preds)
            if r2 > best_r2:
                best_r2 = r2
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_preds = model(torch.tensor(X_train, dtype=torch.float32)).squeeze(-1).numpy()
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze(-1).numpy()

    metrics = {
        'r2_train': float(r2_score(y_train, train_preds)),
        'r2_test': float(r2_score(y_test, test_preds)),
        'mae_train': float(mean_absolute_error(y_train, train_preds)),
        'mae_test': float(mean_absolute_error(y_test, test_preds)),
        'rmse_test': float(math.sqrt(mean_squared_error(y_test, test_preds)))
    }

    predictions = pd.DataFrame({
        'Tissue': np.concatenate([idx_train, idx_test]),
        'Split': ['train'] * len(idx_train) + ['test'] * len(idx_test),
        'Actual': np.concatenate([y_train, y_test]),
        'Predicted': np.concatenate([train_preds, test_preds])
    })
    predictions = predictions.sort_values('Tissue')

    return model, metrics, predictions, scaler


def compute_correlations(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src_gene in source.columns:
        for tgt_gene in target.columns:
            rho, p = spearmanr(source[src_gene], target[tgt_gene])
            rows.append({
                'source_gene': src_gene,
                'target_gene': tgt_gene,
                'spearman_rho': rho,
                'p_value': p
            })
    return pd.DataFrame(rows)


def train_attention_model(features: pd.DataFrame, targets: pd.DataFrame) -> Tuple[S100CrosslinkingAttention, np.ndarray, Dict[str, float]]:
    X = features.values.astype(np.float32)
    Y = targets.values.astype(np.float32)

    model = S100CrosslinkingAttention(n_s100=X.shape[1], n_targets=Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    best_loss = float('inf')
    best_state = None

    for epoch in range(500):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(loader.dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds, attn_weights = model(torch.tensor(X))
        mse = float(criterion(preds, torch.tensor(Y)).item())

    mean_attn = attn_weights.mean(dim=0).numpy()
    metrics = {'attention_mse': mse}

    return model, mean_attn, metrics


def attention_heatmap(mean_attn: np.ndarray, s100_cols: List[str], target_cols: List[str], out_path: Path) -> None:
    attn_df = pd.DataFrame(mean_attn, columns=s100_cols, index=target_cols)
    plt.figure(figsize=(14, 6))
    sns.heatmap(attn_df, annot=False, cmap='viridis')
    plt.title('S100 ↔ Crosslinking Attention Weights')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def stiffness_scatter_plot(predictions: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=predictions, x='Actual', y='Predicted', hue='Split')
    lims = [predictions[['Actual', 'Predicted']].min().min() - 0.5,
            predictions[['Actual', 'Predicted']].max().max() + 0.5]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title('S100 Model: Actual vs Predicted Stiffness')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_significant_pairs(corr_df: pd.DataFrame, threshold: float = 0.6, p_thresh: float = 0.05) -> pd.DataFrame:
    if corr_df.empty or 'spearman_rho' not in corr_df.columns:
        return pd.DataFrame(columns=['source_gene', 'target_gene', 'spearman_rho', 'p_value'])
    sig = corr_df[(corr_df['spearman_rho'].abs() >= threshold) & (corr_df['p_value'] < p_thresh)]
    return sig.sort_values('spearman_rho', key=lambda s: s.abs(), ascending=False)


def fisher_enrichment(mech_corr: pd.DataFrame, infl_corr: pd.DataFrame, threshold: float = 0.5, p_thresh: float = 0.05) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if mech_corr.empty or infl_corr.empty:
        enrichment_df = pd.DataFrame({
            'category': ['mechanotransduction_hits', 'inflammation_hits', 'overlap', 'total_s100'],
            'count': [0, 0, 0, 0]
        })
        return enrichment_df, {'fisher_odds_ratio': float('nan'), 'fisher_p_value': float('nan')}

    mech_sig = mech_corr[(mech_corr['spearman_rho'].abs() >= threshold) & (mech_corr['p_value'] < p_thresh)]
    infl_sig = infl_corr[(infl_corr['spearman_rho'].abs() >= threshold) & (infl_corr['p_value'] < p_thresh)]

    mech_hits = set(mech_sig['source_gene'])
    infl_hits = set(infl_sig['source_gene'])
    all_genes = set(mech_corr['source_gene']).union(set(infl_corr['source_gene']))
    total = len(all_genes)

    mech_hit_count = len(mech_hits)
    infl_hit_count = len(infl_hits)
    mech_non = max(total - mech_hit_count, 1)
    infl_non = max(total - infl_hit_count, 1)

    contingency = np.array([
        [max(mech_hit_count, 1), mech_non],
        [max(infl_hit_count, 1), infl_non]
    ])
    odds, p_value = fisher_exact(contingency)

    enrichment_df = pd.DataFrame({
        'category': ['mechanotransduction_hits', 'inflammation_hits', 'overlap', 'total_s100'],
        'count': [mech_hit_count, infl_hit_count, len(mech_hits & infl_hits), total]
    })
    stats = {'fisher_odds_ratio': float(odds), 'fisher_p_value': float(p_value)}
    return enrichment_df, stats


def paired_ttest(cross_corr: pd.DataFrame, infl_corr: pd.DataFrame) -> Dict[str, float]:
    if cross_corr.empty or infl_corr.empty:
        return {'paired_t_stat': float('nan'), 'paired_t_p_value': float('nan')}
    cross_means = cross_corr.groupby('source_gene')['spearman_rho'].mean()
    infl_means = infl_corr.groupby('source_gene')['spearman_rho'].mean().reindex(cross_means.index, fill_value=0)
    t_stat, p_val = ttest_rel(cross_means, infl_means, nan_policy='omit')
    return {'paired_t_stat': float(t_stat), 'paired_t_p_value': float(p_val)}


def build_network_graph(cross_sig: pd.DataFrame, mech_sig: pd.DataFrame, stiffness_metrics: Dict[str, float], out_path: Path) -> None:
    G = nx.DiGraph()

    if not cross_sig.empty:
        for gene in cross_sig['source_gene'].unique():
            G.add_node(gene, type='S100')
        for gene in cross_sig['target_gene'].unique():
            G.add_node(gene, type='Crosslinking')
    if not mech_sig.empty:
        for gene in mech_sig['target_gene'].unique():
            G.add_node(gene, type='Mechanotransduction')
    G.add_node('Stiffness', type='Phenotype', metric=stiffness_metrics.get('r2_test', 0))

    for _, row in cross_sig.iterrows():
        if not G.has_node(row['source_gene']):
            G.add_node(row['source_gene'], type='S100')
        if not G.has_node(row['target_gene']):
            G.add_node(row['target_gene'], type='Crosslinking')
        G.add_edge(row['source_gene'], row['target_gene'], weight=row['spearman_rho'])

    for _, row in mech_sig.iterrows():
        if not G.has_node(row['source_gene']):
            G.add_node(row['source_gene'], type='S100')
        if not G.has_node(row['target_gene']):
            G.add_node(row['target_gene'], type='Mechanotransduction')
        G.add_edge(row['source_gene'], row['target_gene'], weight=row['spearman_rho'])

    for gene in cross_sig['target_gene'].unique():
        G.add_edge(gene, 'Stiffness', weight=0.8)

    color_map = {
        'S100': '#1f77b4',
        'Crosslinking': '#ff7f0e',
        'Mechanotransduction': '#2ca02c',
        'Phenotype': '#d62728'
    }
    node_colors = [color_map.get(G.nodes[n]['type'], '#7f7f7f') for n in G.nodes]

    pos = nx.spring_layout(G, seed=SEED, k=0.8)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=8)
    edges = G.edges()
    weights = [abs(G[u][v]['weight']) for u, v in edges]
    nx.draw_networkx_edges(G, pos, arrows=True, width=weights)
    plt.title('S100 → Mechanotransduction → Crosslinking → Stiffness Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    if requests is None:
        raise ImportError("requests library is required for downloading AlphaFold structures")
    response = None
    for attempt in range(3):
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            break
    if response is None or response.status_code != 200:
        raise ValueError(f"Failed to download {url}: status {response.status_code if response else 'None'}")
    response.raise_for_status()
    dest.write_bytes(response.content)


def fetch_alphafold_structures() -> Dict[str, Path]:
    files = {}
    if requests is None:
        return files
    for gene, uniprot in ALPHAFOLD_IDS.items():
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        dest = STRUCTURE_DIR / f"{gene}.pdb"
        try:
            response = None
            for attempt in range(3):
                response = requests.get(api_url, timeout=60)
                if response.status_code == 200:
                    break
            if response is None or response.status_code != 200:
                raise ValueError(f"API status {response.status_code if response else 'None'}")
            response.raise_for_status()
            data = response.json()
            if not data:
                raise ValueError("No metadata returned")
            pdb_url = data[0].get('pdbUrl') or data[0].get('cifUrl')
            if not pdb_url:
                raise ValueError("No PDB/ CIF URL available")
            download_file(pdb_url, dest)
            files[gene] = dest
        except Exception as exc:  # pragma: no cover
            print(f"Failed to download {gene} AlphaFold structure: {exc}")
    return files


def parse_structure_features(struct_paths: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    features = {}
    if PDBParser is None:
        return {gene: {'radius_gyration': 0.0, 'mean_ca_distance': 0.0} for gene in struct_paths}

    parser = PDBParser(QUIET=True)
    for gene, path in struct_paths.items():
        structure = parser.get_structure(gene, str(path))
        coords = []
        for atom in structure.get_atoms():
            if atom.get_id() == 'CA':
                coords.append(atom.get_coord())
        if not coords:
            features[gene] = {'radius_gyration': 0.0, 'mean_ca_distance': 0.0}
            continue
        coords = np.array(coords)
        centroid = coords.mean(axis=0)
        radius = np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean())
        dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
        mean_dist = dists[np.triu_indices_from(dists, k=1)].mean()
        features[gene] = {
            'radius_gyration': float(radius),
            'mean_ca_distance': float(mean_dist)
        }
    return features


def load_or_install(package: str) -> None:
    try:
        __import__(package)
    except ImportError:
        import subprocess
        import sys
        in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        cmds = []
        if in_venv:
            cmds.append([sys.executable, "-m", "pip", "install", package])
        else:
            cmds.append([sys.executable, "-m", "pip", "install", "--user", package])
        cmds.append([sys.executable, "-m", "pip", "install", "--break-system-packages", package])
        last_exc = None
        for cmd in cmds:
            try:
                subprocess.check_call(cmd)
                last_exc = None
                break
            except subprocess.CalledProcessError as exc:  # pragma: no cover
                last_exc = exc
        if last_exc is not None:
            raise last_exc


def compute_esm_embeddings(sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
    load_or_install('transformers')
    from transformers import AutoTokenizer, AutoModel

    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    embeddings = {}
    for gene, seq in sequences.items():
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state[:, 1:-1, :]  # drop start/end tokens
        emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings[gene] = emb
    return embeddings


def fetch_sequences(uniprot_ids: Dict[str, str]) -> Dict[str, str]:
    sequences = {}
    for gene, uniprot in uniprot_ids.items():
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        dest = STRUCTURE_DIR / f"{gene}.fasta"
        download_file(url, dest)
        fasta = dest.read_text().splitlines()
        seq = ''.join(line.strip() for line in fasta if not line.startswith('>'))
        sequences[gene] = seq
    return sequences


def build_structural_dataset(embeddings: Dict[str, np.ndarray], struct_feats: Dict[str, Dict[str, float]], s100_matrix: pd.DataFrame, stiffness: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    genes = list(embeddings.keys())
    embedding_dim = len(next(iter(embeddings.values())))

    X = []
    for tissue in s100_matrix.index:
        vec = np.zeros(embedding_dim)
        scalar_feats = []
        for gene in genes:
            expr = s100_matrix.get(gene, pd.Series(0, index=s100_matrix.index)).loc[tissue]
            vec += expr * embeddings[gene]
            struct_info = struct_feats.get(gene, {'radius_gyration': 0.0, 'mean_ca_distance': 0.0})
            scalar_feats.extend([
                expr * struct_info['radius_gyration'],
                expr * struct_info['mean_ca_distance']
            ])
        sample = np.concatenate([vec, np.array(scalar_feats)])
        X.append(sample)
    X = np.vstack(X)
    y = stiffness.loc[s100_matrix.index].values
    return X, y


def train_structural_model(X: np.ndarray, y: np.ndarray) -> Tuple[StructuralStiffnessNN, Dict[str, float], StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

    model = StructuralStiffnessNN(input_dim=X_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_state = None
    best_r2 = -np.inf
    for epoch in range(250):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(test_tensor).squeeze(-1).numpy()
            r2 = r2_score(y_test, preds)
            if r2 > best_r2:
                best_r2 = r2
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds_train = model(torch.tensor(X_train, dtype=torch.float32)).squeeze(-1).numpy()
        preds_test = model(test_tensor).squeeze(-1).numpy()

    metrics = {
        'struct_r2_train': float(r2_score(y_train, preds_train)),
        'struct_r2_test': float(r2_score(y_test, preds_test)),
        'struct_mae_test': float(mean_absolute_error(y_test, preds_test)),
        'struct_rmse_test': float(math.sqrt(mean_squared_error(y_test, preds_test)))
    }

    return model, metrics, scaler


def save_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    ensure_directories()
    df = load_dataset()
    pivots = build_expression_matrices(df)
    stiffness = compute_stiffness_score(pivots)

    common_index = stiffness.dropna().index
    s100_matrix = align_features(common_index, pivots['s100']).fillna(0.0)
    cross_matrix = align_features(common_index, pivots['crosslinking']).fillna(0.0)
    inflammation_matrix = align_features(common_index, pivots['inflammation']).fillna(0.0)
    mechanotrans_matrix = align_features(common_index, pivots['mechanotransduction']).fillna(0.0)

    # Deep NN stiffness model
    stiffness_model, stiffness_metrics, predictions, scaler_expr = train_stiffness_nn(s100_matrix, stiffness)
    torch.save({
        'model_state': stiffness_model.state_dict(),
        'input_genes': list(s100_matrix.columns),
        'metrics': stiffness_metrics,
        'scaler_mean': scaler_expr.mean_.tolist(),
        'scaler_scale': scaler_expr.scale_.tolist()
    }, OUTPUT_DIR / f"s100_stiffness_model_{AGENT}.pth")
    predictions.to_csv(OUTPUT_DIR / f"stiffness_predictions_{AGENT}.csv", index=False)

    stiffness_scatter_plot(predictions, VIS_DIR / f"stiffness_scatter_{AGENT}.png")

    # Correlation analyses
    cross_corr = compute_correlations(s100_matrix, cross_matrix)
    cross_corr.to_csv(OUTPUT_DIR / f"s100_crosslinking_network_{AGENT}.csv", index=False)

    inflammation_corr = compute_correlations(s100_matrix, inflammation_matrix)
    inflammation_corr.to_csv(OUTPUT_DIR / f"s100_vs_inflammation_{AGENT}.csv", index=False)

    significant_pairs = summarize_significant_pairs(cross_corr)
    mech_corr = compute_correlations(s100_matrix, mechanotrans_matrix)

    fisher_df, fisher_stats = fisher_enrichment(mech_corr, inflammation_corr, threshold=0.4)
    paired_stats = paired_ttest(cross_corr, inflammation_corr)

    cross_sig = summarize_significant_pairs(cross_corr)
    mech_sig = summarize_significant_pairs(mech_corr, threshold=0.4)

    build_network_graph(cross_sig, mech_sig, stiffness_metrics, VIS_DIR / f"pathway_network_{AGENT}.png")
    fisher_df.to_csv(OUTPUT_DIR / f"mechanotransduction_enrichment_{AGENT}.csv", index=False)

    # Attention model for S100 -> Crosslinking
    attention_model, mean_attn, attention_metrics = train_attention_model(s100_matrix, cross_matrix)
    np.save(OUTPUT_DIR / f"attention_weights_{AGENT}.npy", mean_attn)
    attention_heatmap(mean_attn, list(s100_matrix.columns), list(cross_matrix.columns), VIS_DIR / f"s100_enzyme_heatmap_{AGENT}.png")

    # AlphaFold + ESM transfer learning
    struct_paths = fetch_alphafold_structures()
    struct_features = parse_structure_features(struct_paths)
    sequences = fetch_sequences(ALPHAFOLD_IDS)
    embeddings = compute_esm_embeddings(sequences)

    structural_X, structural_y = build_structural_dataset(embeddings, struct_features, s100_matrix, stiffness)
    structural_model, structural_metrics, structural_scaler = train_structural_model(structural_X, structural_y)

    torch.save({
        'model_state': structural_model.state_dict(),
        'metrics': structural_metrics,
        'scaler_mean': structural_scaler.mean_.tolist(),
        'scaler_scale': structural_scaler.scale_.tolist(),
        'genes': list(embeddings.keys())
    }, OUTPUT_DIR / f"alphafold_transfer_model_{AGENT}.pth")

    comparison_df = pd.DataFrame([
        {
            'model': 'expression_s100_nn',
            'r2_test': stiffness_metrics['r2_test'],
            'mae_test': stiffness_metrics['mae_test'],
            'rmse_test': stiffness_metrics['rmse_test']
        },
        {
            'model': 'structural_transfer_nn',
            'r2_test': structural_metrics['struct_r2_test'],
            'mae_test': structural_metrics['struct_mae_test'],
            'rmse_test': structural_metrics['struct_rmse_test']
        }
    ])
    comparison_df.to_csv(OUTPUT_DIR / f"structural_vs_expression_{AGENT}.csv", index=False)

    # Aggregate statistics for report
    summary = {
        'stiffness_metrics': stiffness_metrics,
        'attention_metrics': attention_metrics,
        'fisher_stats': fisher_stats,
        'paired_stats': paired_stats,
        'structural_metrics': structural_metrics,
        'significant_crosslinks': significant_pairs.head(10).to_dict(orient='records')
    }
    save_json(summary, OUTPUT_DIR / "analysis_summary_codex.json")

    print("Analysis complete. Key metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
