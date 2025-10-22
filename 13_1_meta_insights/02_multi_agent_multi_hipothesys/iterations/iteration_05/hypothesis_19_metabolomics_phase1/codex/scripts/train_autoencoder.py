#!/usr/bin/env python3
"""Autoencoder for multi-omics feature learning."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_19_metabolomics_phase1/codex')
ANALYSES_DIR = ROOT / 'analyses_codex'
MODELS_DIR = ROOT / 'models_codex'
VIS_DIR = ROOT / 'visualizations_codex'

PROTEIN_GENES = ['COL1A1', 'COL3A1', 'COL5A1', 'FN1', 'ELN', 'LOX', 'TGM2', 'MMP2', 'MMP9', 'PLOD1', 'COL4A1', 'LAMC1']
METABOLITES = ['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate', 'Lactate/Pyruvate']

LATENT_DIM = 6
BATCH_SIZE = 16
EPOCHS = 800
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data() -> tuple[torch.Tensor, list[str], StandardScaler]:
    df = pd.read_csv(ANALYSES_DIR / 'multiomics_samples_codex.csv')
    df = df[df['is_control'] == False].reset_index(drop=True)
    features = PROTEIN_GENES + METABOLITES
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(0.0))
    tensor = torch.tensor(X, dtype=torch.float32)
    return tensor, features, scaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def train(model: Autoencoder, loader: DataLoader) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[float] = []
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch[0].to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f}')
    return history


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    tensor, features, scaler = load_data()
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Autoencoder(input_dim=tensor.shape[1], latent_dim=LATENT_DIM).to(DEVICE)
    history = train(model, loader)

    # Save training history
    history_path = ANALYSES_DIR / 'autoencoder_loss_history_codex.csv'
    pd.DataFrame({'epoch': np.arange(1, EPOCHS + 1), 'loss': history}).to_csv(history_path, index=False)

    # Save latent embeddings
    model.eval()
    with torch.no_grad():
        _, latent = model(tensor.to(DEVICE))
    latent_path = ANALYSES_DIR / 'autoencoder_latent_codex.csv'
    pd.DataFrame(latent.cpu().numpy(), columns=[f'latent_{i+1}' for i in range(LATENT_DIM)]).to_csv(latent_path, index=False)

    # Save model state
    model_path = MODELS_DIR / 'multiomics_autoencoder_codex.pt'
    torch.save(model.state_dict(), model_path)

    # Plot loss curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(history, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Autoencoder Training Loss')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'autoencoder_loss_curve_codex.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
