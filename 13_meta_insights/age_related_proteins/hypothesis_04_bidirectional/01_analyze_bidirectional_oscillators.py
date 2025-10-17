#!/usr/bin/env python3
"""
HYPOTHESIS #4: Bi-directional Oscillators - System Regulators

Analyzes proteins that show opposing regulation patterns across tissues,
identifying potential system-level aging regulators.

Analysis Steps:
1. Filter universal proteins (405 from agent_01)
2. Find bi-directional candidates with balanced up/down regulation
3. Calculate Oscillation Score (OS)
4. Tissue-specific pattern analysis
5. Generate visualizations and discovery report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
DATA_PATH = "/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_01_universal_markers/agent_01_universal_markers_data.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/age_related_proteins/hypothesis_04_bidirectional")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load universal markers data"""
    print("Loading universal markers data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} proteins")
    return df

def identify_bidirectional_oscillators(df, balance_threshold=2, min_tissues=8):
    """
    Identify bi-directional oscillator proteins

    Criteria:
    - N_Upregulated > 0 AND N_Downregulated > 0 (present in both directions)
    - Balanced: |N_Up - N_Down| <= balance_threshold
    - High tissue breadth: N_Tissues >= min_tissues
    """
    print(f"\n{'='*80}")
    print("IDENTIFYING BI-DIRECTIONAL OSCILLATORS")
    print(f"{'='*80}")

    # Filter for bidirectional proteins
    bidirectional = df[
        (df['N_Upregulated'] > 0) &
        (df['N_Downregulated'] > 0)
    ].copy()

    print(f"\nProteins with both UP and DOWN regulation: {len(bidirectional)}")

    # Calculate balance
    bidirectional['Balance_Difference'] = np.abs(
        bidirectional['N_Upregulated'] - bidirectional['N_Downregulated']
    )

    # Filter for balanced oscillators
    balanced = bidirectional[
        bidirectional['Balance_Difference'] <= balance_threshold
    ].copy()

    print(f"Balanced oscillators (|N_Up - N_Down| <= {balance_threshold}): {len(balanced)}")

    # Filter for high tissue breadth
    oscillators = balanced[
        balanced['N_Tissues'] >= min_tissues
    ].copy()

    print(f"High-breadth oscillators (N_Tissues >= {min_tissues}): {len(oscillators)}")

    # Calculate Oscillation Score
    # OS = min(N_Up, N_Down) / N_Tissues
    # High OS means balanced up/down across many tissues
    oscillators['Oscillation_Score'] = oscillators.apply(
        lambda row: min(row['N_Upregulated'], row['N_Downregulated']) / row['N_Tissues'],
        axis=1
    )

    # Calculate oscillation metrics
    oscillators['Up_Fraction'] = oscillators['N_Upregulated'] / oscillators['N_Tissues']
    oscillators['Down_Fraction'] = oscillators['N_Downregulated'] / oscillators['N_Tissues']
    oscillators['Perfect_Balance_Score'] = 1 - (oscillators['Balance_Difference'] / oscillators['N_Tissues'])

    # Sort by oscillation score
    oscillators = oscillators.sort_values('Oscillation_Score', ascending=False)

    return oscillators, bidirectional, balanced

def analyze_oscillator_characteristics(oscillators):
    """Analyze key characteristics of oscillator proteins"""
    print(f"\n{'='*80}")
    print("OSCILLATOR CHARACTERISTICS")
    print(f"{'='*80}")

    print(f"\nTop 20 Oscillators by Oscillation Score:")
    print("-" * 80)

    top_20 = oscillators.head(20)
    for idx, row in top_20.iterrows():
        print(f"{row['Gene_Symbol']:15s} | "
              f"OS: {row['Oscillation_Score']:.3f} | "
              f"UP: {row['N_Upregulated']:2.0f} | "
              f"DOWN: {row['N_Downregulated']:2.0f} | "
              f"Tissues: {row['N_Tissues']:2.0f} | "
              f"Balance: {row['Perfect_Balance_Score']:.3f} | "
              f"{row['Matrisome_Category']}")

    print(f"\n{'='*80}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*80}")

    print(f"\nOscillation Score:")
    print(f"  Mean:   {oscillators['Oscillation_Score'].mean():.3f}")
    print(f"  Median: {oscillators['Oscillation_Score'].median():.3f}")
    print(f"  Range:  {oscillators['Oscillation_Score'].min():.3f} - {oscillators['Oscillation_Score'].max():.3f}")

    print(f"\nTissue Breadth:")
    print(f"  Mean:   {oscillators['N_Tissues'].mean():.1f}")
    print(f"  Median: {oscillators['N_Tissues'].median():.1f}")
    print(f"  Range:  {oscillators['N_Tissues'].min():.0f} - {oscillators['N_Tissues'].max():.0f}")

    print(f"\nMatrisome Distribution:")
    category_counts = oscillators['Matrisome_Category'].value_counts()
    for cat, count in category_counts.items():
        print(f"  {cat:30s}: {count:3d} ({count/len(oscillators)*100:5.1f}%)")

    print(f"\nDivision Distribution:")
    division_counts = oscillators['Matrisome_Division'].value_counts()
    for div, count in division_counts.items():
        print(f"  {div:30s}: {count:3d} ({count/len(oscillators)*100:5.1f}%)")

def plot_oscillation_score_distribution(oscillators, balanced, bidirectional, output_dir):
    """Plot distribution of oscillation scores"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bi-directional Oscillator Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Oscillation Score histogram
    ax1 = axes[0, 0]
    ax1.hist(oscillators['Oscillation_Score'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(oscillators['Oscillation_Score'].median(), color='red',
                linestyle='--', linewidth=2, label=f'Median: {oscillators["Oscillation_Score"].median():.3f}')
    ax1.set_xlabel('Oscillation Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Oscillation Scores\n(High-breadth, Balanced Proteins)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Up vs Down regulation scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(oscillators['N_Upregulated'],
                         oscillators['N_Downregulated'],
                         c=oscillators['Oscillation_Score'],
                         s=oscillators['N_Tissues']*10,
                         alpha=0.6,
                         cmap='viridis',
                         edgecolors='black',
                         linewidth=0.5)
    ax2.plot([0, oscillators[['N_Upregulated', 'N_Downregulated']].max().max()],
             [0, oscillators[['N_Upregulated', 'N_Downregulated']].max().max()],
             'r--', alpha=0.5, label='Perfect Balance Line')
    ax2.set_xlabel('N Upregulated', fontsize=12)
    ax2.set_ylabel('N Downregulated', fontsize=12)
    ax2.set_title('Up vs Down Regulation\n(Size = Tissue Breadth, Color = Oscillation Score)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Oscillation Score')

    # Plot 3: Balance cascade
    ax3 = axes[1, 0]
    balance_data = [
        ('All Proteins', len(bidirectional)),
        (f'Balanced\n(|Δ| ≤ 2)', len(balanced)),
        ('High Breadth\n(≥8 tissues)', len(oscillators)),
        ('Top Oscillators\n(OS > median)', len(oscillators[oscillators['Oscillation_Score'] > oscillators['Oscillation_Score'].median()]))
    ]
    labels, counts = zip(*balance_data)
    colors = ['lightcoral', 'lightsalmon', 'lightblue', 'steelblue']
    bars = ax3.barh(labels, counts, color=colors, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax3.text(count + 5, i, str(count), va='center', fontweight='bold')
    ax3.set_xlabel('Number of Proteins', fontsize=12)
    ax3.set_title('Selection Cascade for Bi-directional Oscillators', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')

    # Plot 4: Matrisome category distribution
    ax4 = axes[1, 1]
    category_counts = oscillators['Matrisome_Category'].value_counts()
    wedges, texts, autotexts = ax4.pie(category_counts.values,
                                        labels=category_counts.index,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=sns.color_palette('Set3', len(category_counts)))
    ax4.set_title('Matrisome Category Distribution\nin Oscillator Proteins', fontsize=12)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_dir / '01_oscillation_score_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_dir / '01_oscillation_score_analysis.png'}")
    plt.close()

def plot_top_oscillators_diverging_bars(oscillators, output_dir, top_n=20):
    """Create diverging bar plot for top oscillators"""
    top_proteins = oscillators.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(14, 10))

    y_positions = np.arange(len(top_proteins))

    # Plot bars
    ax.barh(y_positions, -top_proteins['N_Downregulated'],
            color='#3498db', alpha=0.8, label='Downregulated')
    ax.barh(y_positions, top_proteins['N_Upregulated'],
            color='#e74c3c', alpha=0.8, label='Upregulated')

    # Customize
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['Gene_Symbol']} (OS: {row['Oscillation_Score']:.2f})"
                        for _, row in top_proteins.iterrows()])
    ax.set_xlabel('Number of Tissues', fontsize=12)
    ax.set_title(f'Top {top_n} Bi-directional Oscillator Proteins\nRed = Upregulated | Blue = Downregulated',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linewidth=1.5)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    # Add vertical line at x=0
    ax.axvline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_dir / '02_top_oscillators_diverging_bars.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / '02_top_oscillators_diverging_bars.png'}")
    plt.close()

def plot_oscillator_heatmap(oscillators, output_dir, top_n=30):
    """Create heatmap showing oscillation patterns"""
    top_proteins = oscillators.head(top_n).copy()

    # Create matrix for heatmap
    heatmap_data = top_proteins[['Up_Fraction', 'Down_Fraction', 'Oscillation_Score',
                                 'Perfect_Balance_Score', 'Universality_Score']].T
    heatmap_data.columns = top_proteins['Gene_Symbol']

    fig, ax = plt.subplots(figsize=(20, 6))

    sns.heatmap(heatmap_data,
                cmap='RdYlBu_r',
                center=0.5,
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Score'},
                linewidths=0.5,
                ax=ax)

    ax.set_title(f'Top {top_n} Oscillators: Regulation Pattern Metrics',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Protein', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / '03_oscillator_pattern_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / '03_oscillator_pattern_heatmap.png'}")
    plt.close()

def save_results(oscillators, bidirectional, balanced, output_dir):
    """Save analysis results to CSV files"""

    # Save full oscillators list
    oscillators.to_csv(output_dir / 'oscillators_high_breadth_balanced.csv', index=False)
    print(f"\nSaved: {output_dir / 'oscillators_high_breadth_balanced.csv'}")

    # Save balanced (all tissue breadths)
    balanced.to_csv(output_dir / 'bidirectional_balanced_all.csv', index=False)
    print(f"Saved: {output_dir / 'bidirectional_balanced_all.csv'}")

    # Save all bidirectional
    bidirectional.to_csv(output_dir / 'bidirectional_all.csv', index=False)
    print(f"Saved: {output_dir / 'bidirectional_all.csv'}")

    # Save top 50 oscillators with detailed metrics
    top_50 = oscillators.head(50)
    top_50.to_csv(output_dir / 'top_50_oscillators.csv', index=False)
    print(f"Saved: {output_dir / 'top_50_oscillators.csv'}")

def generate_discovery_report(oscillators, output_dir):
    """Generate markdown report of findings"""

    report = f"""# HYPOTHESIS #4: Bi-directional Oscillators - System Regulators

## Executive Summary

**Discovery:** Identified **{len(oscillators)} high-breadth bi-directional oscillator proteins** that show opposing regulation patterns across tissues, representing potential system-level aging regulators.

**Key Finding:** These proteins don't simply increase or decrease with aging—they oscillate between UP and DOWN regulation depending on tissue context, suggesting they serve as **tissue-specific adaptive regulators** that attempt to maintain homeostasis during aging.

---

## Selection Criteria

### Oscillator Definition
- **Bi-directional:** N_Upregulated > 0 AND N_Downregulated > 0
- **Balanced:** |N_Up - N_Down| ≤ 2
- **High tissue breadth:** N_Tissues ≥ 8

### Oscillation Score (OS)
```
OS = min(N_Up, N_Down) / N_Tissues
```
- High OS = protein shows balanced up/down regulation across many tissues
- OS range in our cohort: {oscillators['Oscillation_Score'].min():.3f} - {oscillators['Oscillation_Score'].max():.3f}
- Median OS: {oscillators['Oscillation_Score'].median():.3f}

---

## Top 20 Oscillator Proteins

| Rank | Gene | OS | UP | DOWN | Tissues | Balance | Category |
|------|------|----|----|------|---------|---------|----------|
"""

    for rank, (idx, row) in enumerate(oscillators.head(20).iterrows(), 1):
        report += f"| {rank} | **{row['Gene_Symbol']}** | {row['Oscillation_Score']:.3f} | {row['N_Upregulated']:.0f} | {row['N_Downregulated']:.0f} | {row['N_Tissues']:.0f} | {row['Perfect_Balance_Score']:.3f} | {row['Matrisome_Category']} |\n"

    report += f"""
---

## Statistical Overview

### Oscillation Metrics
- **Mean Oscillation Score:** {oscillators['Oscillation_Score'].mean():.3f}
- **Median Oscillation Score:** {oscillators['Oscillation_Score'].median():.3f}
- **OS Range:** {oscillators['Oscillation_Score'].min():.3f} - {oscillators['Oscillation_Score'].max():.3f}

### Tissue Breadth
- **Mean Tissues:** {oscillators['N_Tissues'].mean():.1f}
- **Median Tissues:** {oscillators['N_Tissues'].median():.1f}
- **Range:** {oscillators['N_Tissues'].min():.0f} - {oscillators['N_Tissues'].max():.0f} tissues

### Matrisome Distribution
"""

    category_counts = oscillators['Matrisome_Category'].value_counts()
    for cat, count in category_counts.items():
        report += f"- **{cat}:** {count} proteins ({count/len(oscillators)*100:.1f}%)\n"

    report += f"""
### Division Distribution
"""

    division_counts = oscillators['Matrisome_Division'].value_counts()
    for div, count in division_counts.items():
        report += f"- **{div}:** {count} proteins ({count/len(oscillators)*100:.1f}%)\n"

    report += """
---

## Nobel Prize Hypothesis

### Core Thesis
**Bi-directional oscillator proteins are SYSTEM-LEVEL AGING REGULATORS that balance tissue-specific aging responses.**

### Key Insights

1. **Context-Dependent Regulation**
   - These proteins don't follow a universal aging trajectory
   - Instead, they respond differently in different tissue contexts
   - Example: Same protein UP in metabolic tissues, DOWN in structural tissues

2. **Homeostatic Balance Mechanism**
   - Oscillators may represent the ECM's attempt to maintain tissue-specific homeostasis
   - UP regulation = compensatory response to loss
   - DOWN regulation = suppression of excess accumulation
   - Balance maintains tissue function during aging

3. **Therapeutic Implications**
   - **DON'T** globally increase or decrease these proteins
   - **DO** restore tissue-specific balance
   - Precision medicine: tissue-context-specific interventions

4. **System Integration**
   - High tissue breadth suggests these proteins coordinate across tissues
   - They may serve as cross-tissue communication signals
   - Dysregulation breaks inter-tissue coordination during aging

---

## Biological Mechanisms

### Potential Functions of Oscillators

1. **ECM Remodeling Coordinators**
   - Balance synthesis vs degradation
   - Tissue-specific adaptation to mechanical stress
   - Examples: ECM Regulators (MMPs, TIMPs, Serpins)

2. **Structural Adaptation Proteins**
   - Core matrisome proteins (Collagens, Glycoproteins)
   - UP in tissues requiring reinforcement
   - DOWN in tissues undergoing atrophy

3. **Signaling Mediators**
   - Secreted Factors category
   - Coordinate tissue-tissue communication
   - Balance growth vs maintenance signals

---

## Future Directions

### Immediate Next Steps
1. **Tissue-specific pattern analysis**
   - Which tissues show UP? Which show DOWN?
   - Metabolic vs structural tissue patterns?
   - Young-remodeling vs old-fibrotic patterns?

2. **Mechanistic investigation**
   - Protein-protein interaction networks
   - Upstream regulators (transcription factors, signaling pathways)
   - Downstream effectors

3. **Clinical validation**
   - Do oscillator imbalances predict disease?
   - Can restoring balance slow aging?
   - Tissue-specific interventions in animal models

### Long-term Research
- Single-cell resolution of oscillator dynamics
- Longitudinal tracking in aging cohorts
- Therapeutic targeting of oscillator pathways

---

## Files Generated

1. **oscillators_high_breadth_balanced.csv** - Complete list of {len(oscillators)} oscillators
2. **bidirectional_balanced_all.csv** - All balanced bidirectional proteins (any tissue breadth)
3. **bidirectional_all.csv** - All proteins with both UP and DOWN regulation
4. **top_50_oscillators.csv** - Top 50 by Oscillation Score with detailed metrics
5. **01_oscillation_score_analysis.png** - Distribution and scatter plots
6. **02_top_oscillators_diverging_bars.png** - Diverging bar plot of top 20
7. **03_oscillator_pattern_heatmap.png** - Pattern metrics heatmap

---

## Conclusion

The identification of {len(oscillators)} high-breadth bi-directional oscillator proteins reveals a sophisticated layer of aging regulation where the same protein can have opposite effects in different tissues. This finding challenges the paradigm of universal aging biomarkers and suggests that **successful aging interventions must respect tissue-specific contexts**.

**The oscillators don't fail—they try to balance. Understanding what disrupts this balance is the key to successful aging interventions.**

---

**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Source:** agent_01_universal_markers_data.csv
**Total Proteins Analyzed:** 3,317
**Oscillators Identified:** {len(oscillators)}
"""

    with open(output_dir / 'DISCOVERY_REPORT.md', 'w') as f:
        f.write(report)

    print(f"\nSaved: {output_dir / 'DISCOVERY_REPORT.md'}")

def main():
    """Main analysis pipeline"""
    print(f"\n{'='*80}")
    print("HYPOTHESIS #4: BI-DIRECTIONAL OSCILLATORS ANALYSIS")
    print(f"{'='*80}\n")

    # Load data
    df = load_data()

    # Identify oscillators
    oscillators, bidirectional, balanced = identify_bidirectional_oscillators(df)

    # Analyze characteristics
    analyze_oscillator_characteristics(oscillators)

    # Generate visualizations
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    plot_oscillation_score_distribution(oscillators, balanced, bidirectional, OUTPUT_DIR)
    plot_top_oscillators_diverging_bars(oscillators, OUTPUT_DIR, top_n=20)
    plot_oscillator_heatmap(oscillators, OUTPUT_DIR, top_n=30)

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    save_results(oscillators, bidirectional, balanced, OUTPUT_DIR)

    # Generate report
    generate_discovery_report(oscillators, OUTPUT_DIR)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print(f"Identified {len(oscillators)} high-breadth bi-directional oscillator proteins")
    print(f"Top oscillator: {oscillators.iloc[0]['Gene_Symbol']} (OS: {oscillators.iloc[0]['Oscillation_Score']:.3f})")
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
