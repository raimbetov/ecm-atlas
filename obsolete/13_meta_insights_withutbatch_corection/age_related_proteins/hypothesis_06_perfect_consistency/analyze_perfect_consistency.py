#!/usr/bin/env python3
"""
HYPOTHESIS #6: Perfect Consistency Club - The Absolute Laws of Aging

Analyzes proteins with 100% directional consistency across tissues.
These are the "absolute aging laws" - why are they perfect?

Author: Claude Code Agent
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path('/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights')
DATA_FILE = BASE_DIR / 'agent_01_universal_markers/agent_01_universal_markers_data.csv'
OUTPUT_DIR = BASE_DIR / 'age_related_proteins/hypothesis_06_perfect_consistency'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load universal markers data"""
    print("Loading universal markers data...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} universal proteins")
    return df

def identify_perfect_proteins(df, min_tissues=4):
    """
    Identify proteins with perfect directional consistency

    Perfect = Direction_Consistency == 1.0 AND N_Tissues >= min_tissues
    """
    print(f"\n{'='*80}")
    print("IDENTIFYING PERFECT CONSISTENCY PROTEINS")
    print(f"{'='*80}")

    # Filter for perfect proteins
    perfect_df = df[
        (df['Direction_Consistency'] == 1.0) &
        (df['N_Tissues'] >= min_tissues)
    ].copy()

    print(f"\nPerfect Proteins (100% consistency, ≥{min_tissues} tissues): {len(perfect_df)}")

    # Split by direction
    perfect_up = perfect_df[perfect_df['Predominant_Direction'] == 'UP']
    perfect_down = perfect_df[perfect_df['Predominant_Direction'] == 'DOWN']

    print(f"  - Perfect UP-regulated: {len(perfect_up)}")
    print(f"  - Perfect DOWN-regulated: {len(perfect_down)}")

    # Imperfect proteins for comparison
    imperfect_df = df[
        (df['Direction_Consistency'] < 1.0) &
        (df['N_Tissues'] >= min_tissues)
    ].copy()

    print(f"\nImperfect Proteins (< 100% consistency, ≥{min_tissues} tissues): {len(imperfect_df)}")

    return perfect_df, imperfect_df, perfect_up, perfect_down

def analyze_category_enrichment(perfect_df, imperfect_df):
    """
    Test if Perfect proteins are enriched in specific matrisome categories
    """
    print(f"\n{'='*80}")
    print("CATEGORY ENRICHMENT ANALYSIS")
    print(f"{'='*80}")

    results = []

    # Test each category
    for category in perfect_df['Matrisome_Category'].unique():
        # Perfect proteins in this category
        perfect_cat = len(perfect_df[perfect_df['Matrisome_Category'] == category])
        perfect_other = len(perfect_df[perfect_df['Matrisome_Category'] != category])

        # Imperfect proteins in this category
        imperfect_cat = len(imperfect_df[imperfect_df['Matrisome_Category'] == category])
        imperfect_other = len(imperfect_df[imperfect_df['Matrisome_Category'] != category])

        # Fisher's exact test
        contingency_table = [
            [perfect_cat, perfect_other],
            [imperfect_cat, imperfect_other]
        ]
        odds_ratio, p_value = stats.fisher_exact(contingency_table)

        # Calculate percentages
        perfect_pct = perfect_cat / len(perfect_df) * 100
        imperfect_pct = imperfect_cat / len(imperfect_df) * 100
        enrichment = perfect_pct / imperfect_pct if imperfect_pct > 0 else np.inf

        results.append({
            'Category': category,
            'Perfect_Count': perfect_cat,
            'Perfect_Pct': perfect_pct,
            'Imperfect_Count': imperfect_cat,
            'Imperfect_Pct': imperfect_pct,
            'Enrichment': enrichment,
            'Odds_Ratio': odds_ratio,
            'P_Value': p_value
        })

    results_df = pd.DataFrame(results).sort_values('P_Value')

    print("\nCategory Enrichment Results:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'category_enrichment.csv', index=False)

    return results_df

def analyze_division_enrichment(perfect_df, imperfect_df):
    """
    Test if Perfect proteins are enriched in Core vs Associated matrisome
    """
    print(f"\n{'='*80}")
    print("DIVISION ENRICHMENT ANALYSIS (Core vs Associated)")
    print(f"{'='*80}")

    results = []

    for division in perfect_df['Matrisome_Division'].unique():
        # Perfect proteins in this division
        perfect_div = len(perfect_df[perfect_df['Matrisome_Division'] == division])
        perfect_other = len(perfect_df[perfect_df['Matrisome_Division'] != division])

        # Imperfect proteins in this division
        imperfect_div = len(imperfect_df[imperfect_df['Matrisome_Division'] == division])
        imperfect_other = len(imperfect_df[imperfect_df['Matrisome_Division'] != division])

        # Fisher's exact test
        contingency_table = [
            [perfect_div, perfect_other],
            [imperfect_div, imperfect_other]
        ]
        odds_ratio, p_value = stats.fisher_exact(contingency_table)

        # Calculate percentages
        perfect_pct = perfect_div / len(perfect_df) * 100
        imperfect_pct = imperfect_div / len(imperfect_df) * 100
        enrichment = perfect_pct / imperfect_pct if imperfect_pct > 0 else np.inf

        results.append({
            'Division': division,
            'Perfect_Count': perfect_div,
            'Perfect_Pct': perfect_pct,
            'Imperfect_Count': imperfect_div,
            'Imperfect_Pct': imperfect_pct,
            'Enrichment': enrichment,
            'Odds_Ratio': odds_ratio,
            'P_Value': p_value
        })

    results_df = pd.DataFrame(results).sort_values('P_Value')

    print("\nDivision Enrichment Results:")
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'division_enrichment.csv', index=False)

    return results_df

def compare_effect_sizes(perfect_df, imperfect_df):
    """
    Compare effect sizes between Perfect and Imperfect proteins
    """
    print(f"\n{'='*80}")
    print("EFFECT SIZE COMPARISON")
    print(f"{'='*80}")

    # Statistical tests
    # Use absolute mean z-score delta as effect size
    perfect_effects = perfect_df['Abs_Mean_Zscore_Delta'].values
    imperfect_effects = imperfect_df['Abs_Mean_Zscore_Delta'].values

    # Mann-Whitney U test (non-parametric)
    statistic, p_value = stats.mannwhitneyu(perfect_effects, imperfect_effects, alternative='two-sided')

    print(f"\nEffect Size Statistics:")
    print(f"  Perfect proteins:")
    print(f"    Mean: {np.mean(perfect_effects):.4f}")
    print(f"    Median: {np.median(perfect_effects):.4f}")
    print(f"    Std: {np.std(perfect_effects):.4f}")
    print(f"  Imperfect proteins:")
    print(f"    Mean: {np.mean(imperfect_effects):.4f}")
    print(f"    Median: {np.median(imperfect_effects):.4f}")
    print(f"    Std: {np.std(imperfect_effects):.4f}")
    print(f"\nMann-Whitney U test:")
    print(f"  Statistic: {statistic}")
    print(f"  P-value: {p_value:.4e}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(perfect_effects)**2 + np.std(imperfect_effects)**2) / 2)
    cohens_d = (np.mean(perfect_effects) - np.mean(imperfect_effects)) / pooled_std
    print(f"  Cohen's d: {cohens_d:.4f}")

    return {
        'perfect_mean': np.mean(perfect_effects),
        'imperfect_mean': np.mean(imperfect_effects),
        'p_value': p_value,
        'cohens_d': cohens_d
    }

def analyze_strong_effects(perfect_df, imperfect_df):
    """
    Compare strong effect rates between Perfect and Imperfect proteins
    """
    print(f"\n{'='*80}")
    print("STRONG EFFECT RATE COMPARISON")
    print(f"{'='*80}")

    perfect_strong = perfect_df['Strong_Effect_Rate'].values
    imperfect_strong = imperfect_df['Strong_Effect_Rate'].values

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(perfect_strong, imperfect_strong, alternative='two-sided')

    print(f"\nStrong Effect Rate Statistics:")
    print(f"  Perfect proteins:")
    print(f"    Mean: {np.mean(perfect_strong):.4f}")
    print(f"    Median: {np.median(perfect_strong):.4f}")
    print(f"  Imperfect proteins:")
    print(f"    Mean: {np.mean(imperfect_strong):.4f}")
    print(f"    Median: {np.median(imperfect_strong):.4f}")
    print(f"\nMann-Whitney U test:")
    print(f"  P-value: {p_value:.4e}")

    return {
        'perfect_mean': np.mean(perfect_strong),
        'imperfect_mean': np.mean(imperfect_strong),
        'p_value': p_value
    }

def create_venn_diagram(perfect_df, imperfect_df):
    """
    Create Venn-style comparison of Perfect vs Imperfect proteins
    """
    print(f"\nCreating Venn-style comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Category distribution
    ax1 = axes[0]

    categories = sorted(perfect_df['Matrisome_Category'].unique())
    perfect_counts = [len(perfect_df[perfect_df['Matrisome_Category'] == cat]) for cat in categories]
    imperfect_counts = [len(imperfect_df[imperfect_df['Matrisome_Category'] == cat]) for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, perfect_counts, width, label='Perfect (100%)', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, imperfect_counts, width, label='Imperfect (<100%)', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Matrisome Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax1.set_title('Perfect vs Imperfect: Category Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Division distribution
    ax2 = axes[1]

    divisions = ['Core matrisome', 'Matrisome-associated']
    perfect_div_counts = [
        len(perfect_df[perfect_df['Matrisome_Division'] == 'Core matrisome']),
        len(perfect_df[perfect_df['Matrisome_Division'] == 'Matrisome-associated'])
    ]
    imperfect_div_counts = [
        len(imperfect_df[imperfect_df['Matrisome_Division'] == 'Core matrisome']),
        len(imperfect_df[imperfect_df['Matrisome_Division'] == 'Matrisome-associated'])
    ]

    x = np.arange(len(divisions))

    ax2.bar(x - width/2, perfect_div_counts, width, label='Perfect (100%)', color='#2ecc71', alpha=0.8)
    ax2.bar(x + width/2, imperfect_div_counts, width, label='Imperfect (<100%)', color='#e74c3c', alpha=0.8)

    ax2.set_xlabel('Matrisome Division', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax2.set_title('Perfect vs Imperfect: Division Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(divisions)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'venn_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'venn_comparison.png'}")
    plt.close()

def create_effect_size_boxplot(perfect_df, imperfect_df):
    """
    Create boxplot comparing effect sizes
    """
    print(f"\nCreating effect size comparison boxplot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prepare data
    data = []
    labels = []

    # Plot 1: Absolute Mean Z-score Delta
    ax1 = axes[0]

    perfect_effects = perfect_df['Abs_Mean_Zscore_Delta'].values
    imperfect_effects = imperfect_df['Abs_Mean_Zscore_Delta'].values

    bp1 = ax1.boxplot([perfect_effects, imperfect_effects],
                       labels=['Perfect\n(100%)', 'Imperfect\n(<100%)'],
                       patch_artist=True,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    bp1['boxes'][0].set_facecolor('#2ecc71')
    bp1['boxes'][1].set_facecolor('#e74c3c')

    ax1.set_ylabel('Absolute Mean Z-score Delta', fontsize=12, fontweight='bold')
    ax1.set_title('Effect Size Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add sample sizes
    ax1.text(1, ax1.get_ylim()[1]*0.95, f'n={len(perfect_effects)}', ha='center', fontsize=10)
    ax1.text(2, ax1.get_ylim()[1]*0.95, f'n={len(imperfect_effects)}', ha='center', fontsize=10)

    # Plot 2: Strong Effect Rate
    ax2 = axes[1]

    perfect_strong = perfect_df['Strong_Effect_Rate'].values
    imperfect_strong = imperfect_df['Strong_Effect_Rate'].values

    bp2 = ax2.boxplot([perfect_strong, imperfect_strong],
                       labels=['Perfect\n(100%)', 'Imperfect\n(<100%)'],
                       patch_artist=True,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    bp2['boxes'][0].set_facecolor('#2ecc71')
    bp2['boxes'][1].set_facecolor('#e74c3c')

    ax2.set_ylabel('Strong Effect Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Strong Effect Rate Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add sample sizes
    ax2.text(1, ax2.get_ylim()[1]*0.95, f'n={len(perfect_strong)}', ha='center', fontsize=10)
    ax2.text(2, ax2.get_ylim()[1]*0.95, f'n={len(imperfect_strong)}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'effect_size_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'effect_size_boxplot.png'}")
    plt.close()

def create_tissue_distribution(perfect_df, imperfect_df):
    """
    Compare tissue distribution between Perfect and Imperfect
    """
    print(f"\nCreating tissue distribution plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Count proteins by N_Tissues
    tissue_bins = range(4, perfect_df['N_Tissues'].max() + 2)

    perfect_hist = [len(perfect_df[perfect_df['N_Tissues'] == t]) for t in tissue_bins[:-1]]
    imperfect_hist = [len(imperfect_df[imperfect_df['N_Tissues'] == t]) for t in tissue_bins[:-1]]

    x = np.arange(len(tissue_bins) - 1)
    width = 0.35

    ax.bar(x - width/2, perfect_hist, width, label='Perfect (100%)', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, imperfect_hist, width, label='Imperfect (<100%)', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Number of Tissues', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax.set_title('Tissue Breadth: Perfect vs Imperfect Proteins', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tissue_bins[:-1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tissue_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'tissue_distribution.png'}")
    plt.close()

def create_direction_heatmap(perfect_up, perfect_down):
    """
    Create heatmap showing Perfect UP vs DOWN by category
    """
    print(f"\nCreating direction heatmap...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Count by category and direction
    categories = sorted(set(list(perfect_up['Matrisome_Category'].unique()) +
                            list(perfect_down['Matrisome_Category'].unique())))

    data = []
    for cat in categories:
        up_count = len(perfect_up[perfect_up['Matrisome_Category'] == cat])
        down_count = len(perfect_down[perfect_down['Matrisome_Category'] == cat])
        data.append([up_count, down_count])

    data = np.array(data)

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

    # Set ticks
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['UP-regulated', 'DOWN-regulated'], fontsize=11)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Proteins', fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(len(categories)):
        for j in range(2):
            text = ax.text(j, i, int(data[i, j]),
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax.set_title('Perfect Proteins: Direction by Category', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'direction_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'direction_heatmap.png'}")
    plt.close()

def export_perfect_proteins(perfect_df, perfect_up, perfect_down):
    """
    Export lists of perfect proteins for pathway analysis
    """
    print(f"\nExporting perfect protein lists...")

    # All perfect proteins
    perfect_df[['Gene_Symbol', 'Protein_Name', 'Matrisome_Category', 'Matrisome_Division',
                'N_Tissues', 'Predominant_Direction', 'Abs_Mean_Zscore_Delta',
                'Strong_Effect_Rate', 'Universality_Score']].to_csv(
        OUTPUT_DIR / 'perfect_proteins_all.csv', index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'perfect_proteins_all.csv'}")

    # Perfect UP
    perfect_up[['Gene_Symbol', 'Protein_Name', 'Matrisome_Category', 'Matrisome_Division',
                'N_Tissues', 'Abs_Mean_Zscore_Delta', 'Strong_Effect_Rate',
                'Universality_Score']].to_csv(
        OUTPUT_DIR / 'perfect_proteins_UP.csv', index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'perfect_proteins_UP.csv'}")

    # Perfect DOWN
    perfect_down[['Gene_Symbol', 'Protein_Name', 'Matrisome_Category', 'Matrisome_Division',
                  'N_Tissues', 'Abs_Mean_Zscore_Delta', 'Strong_Effect_Rate',
                  'Universality_Score']].to_csv(
        OUTPUT_DIR / 'perfect_proteins_DOWN.csv', index=False
    )
    print(f"Saved: {OUTPUT_DIR / 'perfect_proteins_DOWN.csv'}")

    # Gene symbols only for pathway tools
    with open(OUTPUT_DIR / 'perfect_genes_all.txt', 'w') as f:
        f.write('\n'.join(perfect_df['Gene_Symbol'].values))

    with open(OUTPUT_DIR / 'perfect_genes_UP.txt', 'w') as f:
        f.write('\n'.join(perfect_up['Gene_Symbol'].values))

    with open(OUTPUT_DIR / 'perfect_genes_DOWN.txt', 'w') as f:
        f.write('\n'.join(perfect_down['Gene_Symbol'].values))

    print(f"Saved gene symbol lists for pathway analysis")

def main():
    """Main analysis pipeline"""
    print(f"\n{'='*80}")
    print("HYPOTHESIS #6: PERFECT CONSISTENCY CLUB")
    print("The Absolute Laws of Aging")
    print(f"{'='*80}\n")

    # Load data
    df = load_data()

    # Identify perfect proteins
    perfect_df, imperfect_df, perfect_up, perfect_down = identify_perfect_proteins(df, min_tissues=4)

    # Analyze category enrichment
    category_results = analyze_category_enrichment(perfect_df, imperfect_df)

    # Analyze division enrichment
    division_results = analyze_division_enrichment(perfect_df, imperfect_df)

    # Compare effect sizes
    effect_results = compare_effect_sizes(perfect_df, imperfect_df)

    # Compare strong effect rates
    strong_results = analyze_strong_effects(perfect_df, imperfect_df)

    # Create visualizations
    create_venn_diagram(perfect_df, imperfect_df)
    create_effect_size_boxplot(perfect_df, imperfect_df)
    create_tissue_distribution(perfect_df, imperfect_df)
    create_direction_heatmap(perfect_up, perfect_down)

    # Export protein lists
    export_perfect_proteins(perfect_df, perfect_up, perfect_down)

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nPerfect Proteins (100% consistency, ≥4 tissues):")
    print(f"  Total: {len(perfect_df)}")
    print(f"  UP-regulated: {len(perfect_up)} ({len(perfect_up)/len(perfect_df)*100:.1f}%)")
    print(f"  DOWN-regulated: {len(perfect_down)} ({len(perfect_down)/len(perfect_df)*100:.1f}%)")
    print(f"\nMean effect size:")
    print(f"  Perfect: {effect_results['perfect_mean']:.4f}")
    print(f"  Imperfect: {effect_results['imperfect_mean']:.4f}")
    print(f"  Cohen's d: {effect_results['cohens_d']:.4f}")
    print(f"  P-value: {effect_results['p_value']:.4e}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
