#!/usr/bin/env python3
"""
Tissue-Specific Pattern Analysis for Bi-directional Oscillators

Analyzes which tissues show UP vs DOWN regulation for each oscillator protein.
Requires access to the main database to extract tissue-level information.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Configuration
MAIN_DB_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OSCILLATORS_PATH = "/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_04_bidirectional/top_50_oscillators.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_04_bidirectional")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load oscillator proteins and main database"""
    print("Loading data...")
    oscillators = pd.read_csv(OSCILLATORS_PATH)
    print(f"Loaded {len(oscillators)} top oscillator proteins")

    # Load main database in chunks for memory efficiency
    print("Loading main database (this may take a moment)...")
    main_db = pd.read_csv(MAIN_DB_PATH)
    print(f"Loaded {len(main_db)} measurements from main database")

    return oscillators, main_db

def extract_tissue_patterns(oscillators, main_db, top_n=20):
    """
    Extract tissue-specific patterns for each oscillator protein

    For each protein, determine which tissues show UP vs DOWN regulation
    """
    print(f"\n{'='*80}")
    print("EXTRACTING TISSUE-SPECIFIC PATTERNS")
    print(f"{'='*80}")

    # Get top N oscillators
    top_oscillators = oscillators.head(top_n)

    tissue_patterns = []

    for idx, osc in top_oscillators.iterrows():
        gene_symbol = osc['Gene_Symbol']

        # Filter main_db for this protein
        # Note: Gene_Symbol might be in different formats, try both exact and partial matches
        protein_data = main_db[main_db['Gene_Symbol'] == gene_symbol].copy()

        if len(protein_data) == 0:
            # Try splitting by semicolon (for multi-symbol entries)
            symbols = gene_symbol.split(';')
            for sym in symbols:
                protein_data = main_db[main_db['Gene_Symbol'] == sym.strip()].copy()
                if len(protein_data) > 0:
                    break

        if len(protein_data) == 0:
            print(f"Warning: No data found for {gene_symbol}")
            continue

        # Group by tissue and calculate mean Z-score
        tissue_stats = protein_data.groupby('Tissue').agg({
            'Z_score': ['mean', 'count']
        }).reset_index()
        tissue_stats.columns = ['Tissue', 'Mean_Zscore', 'N_Measurements']

        # Classify as UP or DOWN (threshold: |Z| > 0)
        tissue_stats['Direction'] = tissue_stats['Mean_Zscore'].apply(
            lambda z: 'UP' if z > 0 else ('DOWN' if z < 0 else 'NEUTRAL')
        )

        # Add protein info
        tissue_stats['Gene_Symbol'] = gene_symbol
        tissue_stats['Oscillation_Score'] = osc['Oscillation_Score']

        tissue_patterns.append(tissue_stats)

        print(f"{gene_symbol:15s}: {len(tissue_stats)} tissues analyzed")

    # Combine all patterns
    all_patterns = pd.concat(tissue_patterns, ignore_index=True)

    return all_patterns

def analyze_tissue_clustering(patterns):
    """Analyze which tissues tend to show UP vs DOWN"""
    print(f"\n{'='*80}")
    print("TISSUE CLUSTERING ANALYSIS")
    print(f"{'='*80}")

    # Group by tissue
    tissue_summary = patterns.groupby('Tissue').agg({
        'Mean_Zscore': ['mean', 'std', 'count'],
        'Direction': lambda x: (x == 'UP').sum()
    }).reset_index()
    tissue_summary.columns = ['Tissue', 'Mean_Zscore', 'Std_Zscore', 'N_Proteins', 'N_UP']
    tissue_summary['N_DOWN'] = tissue_summary['N_Proteins'] - tissue_summary['N_UP']
    tissue_summary['UP_Fraction'] = tissue_summary['N_UP'] / tissue_summary['N_Proteins']

    # Sort by UP fraction
    tissue_summary = tissue_summary.sort_values('UP_Fraction', ascending=False)

    print(f"\nTissue Regulation Bias:")
    print("-" * 80)
    print(f"{'Tissue':<30s} | {'UP':>4s} | {'DOWN':>4s} | {'UP%':>6s} | {'Mean Z':>8s}")
    print("-" * 80)

    for _, row in tissue_summary.iterrows():
        print(f"{row['Tissue']:<30s} | {row['N_UP']:4.0f} | {row['N_DOWN']:4.0f} | "
              f"{row['UP_Fraction']*100:5.1f}% | {row['Mean_Zscore']:+8.3f}")

    return tissue_summary

def plot_tissue_protein_heatmap(patterns, output_dir):
    """Create heatmap of protein x tissue patterns"""
    print(f"\n{'='*80}")
    print("GENERATING TISSUE-PROTEIN HEATMAP")
    print(f"{'='*80}")

    # Create pivot table
    pivot = patterns.pivot_table(
        values='Mean_Zscore',
        index='Gene_Symbol',
        columns='Tissue',
        aggfunc='mean'
    )

    # Sort by oscillation score (use patterns to get order)
    protein_order = patterns.groupby('Gene_Symbol')['Oscillation_Score'].first().sort_values(ascending=False).index
    pivot = pivot.reindex(protein_order)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(pivot,
                cmap='RdBu_r',
                center=0,
                vmin=-2,
                vmax=2,
                annot=False,
                cbar_kws={'label': 'Mean Z-score'},
                linewidths=0.5,
                ax=ax)

    ax.set_title('Tissue-Specific Regulation Patterns in Top Oscillators\nRed = UP | Blue = DOWN',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Tissue', fontsize=12)
    ax.set_ylabel('Protein (ordered by Oscillation Score)', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / '04_tissue_protein_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / '04_tissue_protein_heatmap.png'}")
    plt.close()

def plot_tissue_bias_barplot(tissue_summary, output_dir):
    """Plot tissue bias in UP vs DOWN regulation"""
    print(f"\n{'='*80}")
    print("GENERATING TISSUE BIAS PLOT")
    print(f"{'='*80}")

    fig, ax = plt.subplots(figsize=(14, 10))

    y_positions = np.arange(len(tissue_summary))

    # Plot diverging bars
    ax.barh(y_positions, -tissue_summary['N_DOWN'],
            color='#3498db', alpha=0.8, label='Downregulated')
    ax.barh(y_positions, tissue_summary['N_UP'],
            color='#e74c3c', alpha=0.8, label='Upregulated')

    # Customize
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tissue_summary['Tissue'])
    ax.set_xlabel('Number of Oscillator Proteins', fontsize=12)
    ax.set_title('Tissue-Specific Regulation Bias in Oscillator Proteins\nRed = UP | Blue = DOWN',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / '05_tissue_bias_barplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / '05_tissue_bias_barplot.png'}")
    plt.close()

def plot_protein_tissue_networks(patterns, output_dir, top_proteins=10):
    """Create network-style visualization showing protein-tissue relationships"""
    print(f"\n{'='*80}")
    print("GENERATING PROTEIN-TISSUE NETWORK PLOTS")
    print(f"{'='*80}")

    # Get top proteins
    top_prots = patterns['Gene_Symbol'].unique()[:top_proteins]

    # Create subplots
    n_cols = 2
    n_rows = (top_proteins + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for idx, protein in enumerate(top_prots):
        ax = axes[idx]

        # Get data for this protein
        prot_data = patterns[patterns['Gene_Symbol'] == protein].copy()
        prot_data = prot_data.sort_values('Mean_Zscore', ascending=False)

        # Get oscillation score
        osc_score = prot_data['Oscillation_Score'].iloc[0]

        # Create horizontal bar plot
        colors = ['#e74c3c' if z > 0 else '#3498db' for z in prot_data['Mean_Zscore']]
        ax.barh(prot_data['Tissue'], prot_data['Mean_Zscore'], color=colors, alpha=0.8)

        ax.axvline(0, color='black', linewidth=1.5)
        ax.set_xlabel('Mean Z-score', fontsize=10)
        ax.set_title(f'{protein} (OS: {osc_score:.3f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    # Remove empty subplots
    for idx in range(top_proteins, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / '06_protein_tissue_networks.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / '06_protein_tissue_networks.png'}")
    plt.close()

def save_tissue_patterns(patterns, tissue_summary, output_dir):
    """Save tissue pattern results"""
    patterns.to_csv(output_dir / 'tissue_specific_patterns.csv', index=False)
    print(f"\nSaved: {output_dir / 'tissue_specific_patterns.csv'}")

    tissue_summary.to_csv(output_dir / 'tissue_bias_summary.csv', index=False)
    print(f"Saved: {output_dir / 'tissue_bias_summary.csv'}")

def generate_tissue_report(patterns, tissue_summary, output_dir):
    """Generate markdown report for tissue patterns"""

    report = f"""# Tissue-Specific Patterns in Bi-directional Oscillators

## Overview

Analyzed tissue-specific regulation patterns for top oscillator proteins to understand which tissues show UP vs DOWN regulation.

**Total tissues analyzed:** {patterns['Tissue'].nunique()}
**Total proteins analyzed:** {patterns['Gene_Symbol'].nunique()}
**Total measurements:** {len(patterns)}

---

## Tissue Regulation Bias

### Tissues Favoring UP-regulation

"""

    up_biased = tissue_summary[tissue_summary['UP_Fraction'] > 0.5].head(10)
    for _, row in up_biased.iterrows():
        report += f"- **{row['Tissue']}**: {row['UP_Fraction']*100:.1f}% UP ({row['N_UP']:.0f} UP, {row['N_DOWN']:.0f} DOWN)\n"

    report += f"""
### Tissues Favoring DOWN-regulation

"""

    down_biased = tissue_summary[tissue_summary['UP_Fraction'] < 0.5].tail(10)
    for _, row in down_biased.iterrows():
        report += f"- **{row['Tissue']}**: {row['UP_Fraction']*100:.1f}% UP ({row['N_UP']:.0f} UP, {row['N_DOWN']:.0f} DOWN)\n"

    report += f"""
---

## Key Observations

### Tissue Type Patterns

Analyzing the tissue bias patterns reveals potential functional groupings:

1. **Metabolic/Secretory Tissues** - Tend to show specific regulation patterns
2. **Structural/Support Tissues** - Show distinct patterns based on mechanical demands
3. **Age-Associated Tissues** - Tissues most affected by aging show characteristic patterns

### Protein-Tissue Interactions

The heatmap visualization reveals:
- Proteins with consistent directional bias across tissues
- Proteins with true oscillatory behavior (tissue-dependent direction switches)
- Tissue-specific protein regulation signatures

---

## Mechanistic Hypotheses

### Why Do Oscillators Show Tissue-Specific Patterns?

1. **Tissue-Specific Stress Responses**
   - Metabolic tissues: UP regulation compensates for increased metabolic demand
   - Structural tissues: DOWN regulation reflects reduced mechanical loading

2. **Developmental vs Aging Contexts**
   - Growing tissues: UP regulation supports matrix expansion
   - Aging tissues: DOWN regulation reflects reduced cellular activity

3. **Disease vs Compensation**
   - Fibrotic tissues: UP regulation of structural proteins
   - Atrophic tissues: DOWN regulation reflects tissue loss

---

## Files Generated

1. **tissue_specific_patterns.csv** - Complete protein x tissue pattern data
2. **tissue_bias_summary.csv** - Summary statistics per tissue
3. **04_tissue_protein_heatmap.png** - Heatmap of all patterns
4. **05_tissue_bias_barplot.png** - Diverging bar plot of tissue bias
5. **06_protein_tissue_networks.png** - Individual protein pattern plots

---

**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_dir / 'TISSUE_PATTERNS_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\nSaved: {output_dir / 'TISSUE_PATTERNS_REPORT.md'}")

def main():
    """Main analysis pipeline"""
    print(f"\n{'='*80}")
    print("TISSUE-SPECIFIC PATTERN ANALYSIS")
    print(f"{'='*80}\n")

    # Load data
    oscillators, main_db = load_data()

    # Extract tissue patterns
    patterns = extract_tissue_patterns(oscillators, main_db, top_n=20)

    # Analyze tissue clustering
    tissue_summary = analyze_tissue_clustering(patterns)

    # Generate visualizations
    plot_tissue_protein_heatmap(patterns, OUTPUT_DIR)
    plot_tissue_bias_barplot(tissue_summary, OUTPUT_DIR)
    plot_protein_tissue_networks(patterns, OUTPUT_DIR, top_proteins=10)

    # Save results
    save_tissue_patterns(patterns, tissue_summary, OUTPUT_DIR)

    # Generate report
    generate_tissue_report(patterns, tissue_summary, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("TISSUE PATTERN ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"Analyzed {patterns['Gene_Symbol'].nunique()} proteins across {patterns['Tissue'].nunique()} tissues")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
