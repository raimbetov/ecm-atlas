#!/usr/bin/env python3
"""Generate key visualizations for metabolomics analyses."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_19_metabolomics_phase1/codex')
ANALYSES_DIR = ROOT / 'analyses_codex'
VIS_DIR = ROOT / 'visualizations_codex'

FOCUS_METABOLITES = ['ATP', 'NADH']
ALL_METABOLITES = ['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate', 'Lactate/Pyruvate']


def scatter_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 4))
    for idx, metabolite in enumerate(FOCUS_METABOLITES, start=1):
        sub = df[(df['canonical_metabolite'] == metabolite) & (~df['is_control'])]
        ax = plt.subplot(1, len(FOCUS_METABOLITES), idx)
        sns.regplot(data=sub, x='velocity', y='percent_change', scatter=True, ax=ax, ci=None, color='#1f77b4')
        ax.set_title(f'{metabolite} vs Velocity')
        ax.set_xlabel('Velocity')
        ax.set_ylabel('% change vs control')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'metabolite_velocity_scatter_codex.png', dpi=300)
    plt.close()


def phase_boxplot(df: pd.DataFrame) -> None:
    sub = df[~df['is_control'] & df['canonical_metabolite'].isin(ALL_METABOLITES)]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=sub, x='canonical_metabolite', y='percent_change', hue='phase')
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.ylabel('% change vs control')
    plt.xlabel('Metabolite')
    plt.title('Phase I vs Phase II metabolite shifts')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'phase1_vs_phase2_boxplot_codex.png', dpi=300)
    plt.close()


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ANALYSES_DIR / 'metabolomics_combined_codex.csv')
    scatter_plots(df)
    phase_boxplot(df)


if __name__ == '__main__':
    main()
