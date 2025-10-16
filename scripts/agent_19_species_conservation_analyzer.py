#!/usr/bin/env python3
"""
Agent 19: Species Conservation Analyzer
Analyzes conservation of ECM aging patterns between Human and Mouse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/10_insights")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and prepare dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Basic statistics
    print(f"\nTotal records: {len(df):,}")
    print(f"\nSpecies distribution:")
    print(df['Species'].value_counts())
    print(f"\nTissue distribution:")
    print(df['Tissue'].value_counts())

    return df

def identify_orthologous_proteins(df):
    """Identify proteins present in both human and mouse"""
    print("\n" + "="*80)
    print("IDENTIFYING ORTHOLOGOUS PROTEINS")
    print("="*80)

    # Get proteins by species - normalize to uppercase for comparison
    human_proteins_raw = df[df['Species'] == 'Homo sapiens']['Canonical_Gene_Symbol'].dropna()
    mouse_proteins_raw = df[df['Species'] == 'Mus musculus']['Canonical_Gene_Symbol'].dropna()

    human_proteins_upper = set(human_proteins_raw.str.upper())
    mouse_proteins_upper = set(mouse_proteins_raw.str.upper())

    # Find overlaps using uppercase
    orthologous = human_proteins_upper & mouse_proteins_upper
    human_specific = human_proteins_upper - mouse_proteins_upper
    mouse_specific = mouse_proteins_upper - human_proteins_upper

    print(f"\nTotal Human proteins: {len(human_proteins_upper)}")
    print(f"Total Mouse proteins: {len(mouse_proteins_upper)}")
    print(f"Orthologous proteins (in both): {len(orthologous)}")
    print(f"Human-specific proteins: {len(human_specific)}")
    print(f"Mouse-specific proteins: {len(mouse_specific)}")

    # Save lists
    pd.DataFrame({'Protein': sorted(orthologous)}).to_csv(
        OUTPUT_DIR / 'orthologous_proteins.csv', index=False
    )
    pd.DataFrame({'Protein': sorted(human_specific)}).to_csv(
        OUTPUT_DIR / 'human_specific_proteins.csv', index=False
    )
    pd.DataFrame({'Protein': sorted(mouse_specific)}).to_csv(
        OUTPUT_DIR / 'mouse_specific_proteins.csv', index=False
    )

    # Show examples of orthologous proteins
    print(f"\nExample orthologous proteins (first 20):")
    for i, protein in enumerate(sorted(orthologous)[:20], 1):
        print(f"  {i}. {protein}")

    return orthologous, human_specific, mouse_specific

def analyze_tissue_pairs(df, orthologous):
    """Analyze conservation across tissue pairs"""
    print("\n" + "="*80)
    print("ANALYZING TISSUE PAIR CONSERVATION")
    print("="*80)

    # Add uppercase gene symbol column for matching
    df['Gene_Upper'] = df['Canonical_Gene_Symbol'].str.upper()

    # Get available tissues for each species
    human_tissues = set(df[df['Species'] == 'Homo sapiens']['Tissue'].dropna())
    mouse_tissues = set(df[df['Species'] == 'Mus musculus']['Tissue'].dropna())

    print(f"\nHuman tissues: {sorted(human_tissues)}")
    print(f"\nMouse tissues: {sorted(mouse_tissues)}")

    # Find potential tissue matches (similar names)
    tissue_pairs = []
    for ht in human_tissues:
        for mt in mouse_tissues:
            # Simple name matching (can be improved)
            if ht.lower() in mt.lower() or mt.lower() in ht.lower():
                tissue_pairs.append((ht, mt))

    print(f"\nPotential tissue pairs: {tissue_pairs}")

    # Analyze each tissue pair
    results = []

    for human_tissue, mouse_tissue in tissue_pairs:
        print(f"\n--- Analyzing: {human_tissue} (Human) vs {mouse_tissue} (Mouse) ---")

        # Filter data using uppercase matching
        human_data = df[
            (df['Species'] == 'Homo sapiens') &
            (df['Tissue'] == human_tissue) &
            (df['Gene_Upper'].isin(orthologous))
        ].copy()

        mouse_data = df[
            (df['Species'] == 'Mus musculus') &
            (df['Tissue'] == mouse_tissue) &
            (df['Gene_Upper'].isin(orthologous))
        ].copy()

        # Merge on gene symbol (using uppercase)
        merged = pd.merge(
            human_data[['Gene_Upper', 'Zscore_Delta', 'Zscore_Old', 'Zscore_Young']],
            mouse_data[['Gene_Upper', 'Zscore_Delta', 'Zscore_Old', 'Zscore_Young']],
            on='Gene_Upper',
            suffixes=('_human', '_mouse')
        )

        # Remove NaN values
        merged_clean = merged.dropna(subset=['Zscore_Delta_human', 'Zscore_Delta_mouse'])

        if len(merged_clean) < 5:
            print(f"  Insufficient data: only {len(merged_clean)} overlapping proteins")
            continue

        # Calculate conservation score (correlation)
        corr, p_value = stats.pearsonr(
            merged_clean['Zscore_Delta_human'],
            merged_clean['Zscore_Delta_mouse']
        )

        # Calculate Spearman (rank correlation)
        spearman_corr, spearman_p = stats.spearmanr(
            merged_clean['Zscore_Delta_human'],
            merged_clean['Zscore_Delta_mouse']
        )

        print(f"  Overlapping proteins: {len(merged_clean)}")
        print(f"  Pearson correlation: {corr:.3f} (p={p_value:.2e})")
        print(f"  Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.2e})")

        results.append({
            'Human_Tissue': human_tissue,
            'Mouse_Tissue': mouse_tissue,
            'N_Proteins': len(merged_clean),
            'Pearson_Correlation': corr,
            'Pearson_P_Value': p_value,
            'Spearman_Correlation': spearman_corr,
            'Spearman_P_Value': spearman_p,
            'Conservation_Score': corr  # Use Pearson as main score
        })

        # Save protein-level comparison
        merged_clean.to_csv(
            OUTPUT_DIR / f'tissue_comparison_{human_tissue}_{mouse_tissue}.csv',
            index=False
        )

    # Create summary dataframe
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Conservation_Score', ascending=False)
        results_df.to_csv(OUTPUT_DIR / 'tissue_conservation_scores.csv', index=False)

        print("\n" + "="*80)
        print("CONSERVATION SCORE RANKING")
        print("="*80)
        print(results_df.to_string(index=False))

        return results_df
    else:
        print("\nNo valid tissue pairs found for comparison")
        return None

def analyze_conserved_aging_proteins(df, orthologous):
    """Identify proteins with conserved aging patterns"""
    print("\n" + "="*80)
    print("IDENTIFYING CONSERVED AGING PROTEINS")
    print("="*80)

    # Add uppercase gene symbol column if not already present
    if 'Gene_Upper' not in df.columns:
        df['Gene_Upper'] = df['Canonical_Gene_Symbol'].str.upper()

    # Get all data for orthologous proteins
    human_data = df[
        (df['Species'] == 'Homo sapiens') &
        (df['Gene_Upper'].isin(orthologous))
    ][['Gene_Upper', 'Tissue', 'Zscore_Delta']].copy()

    mouse_data = df[
        (df['Species'] == 'Mus musculus') &
        (df['Gene_Upper'].isin(orthologous))
    ][['Gene_Upper', 'Tissue', 'Zscore_Delta']].copy()

    # Calculate average delta z-score per protein across all tissues
    human_avg = human_data.groupby('Gene_Upper')['Zscore_Delta'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    human_avg.columns = ['Protein', 'Human_Avg_Delta', 'Human_Std', 'Human_Count']

    mouse_avg = mouse_data.groupby('Gene_Upper')['Zscore_Delta'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    mouse_avg.columns = ['Protein', 'Mouse_Avg_Delta', 'Mouse_Std', 'Mouse_Count']

    # Merge
    comparison = pd.merge(human_avg, mouse_avg, on='Protein')
    comparison = comparison.dropna()

    # Calculate agreement
    comparison['Direction_Human'] = comparison['Human_Avg_Delta'].apply(
        lambda x: 'Up' if x > 0.5 else ('Down' if x < -0.5 else 'Stable')
    )
    comparison['Direction_Mouse'] = comparison['Mouse_Avg_Delta'].apply(
        lambda x: 'Up' if x > 0.5 else ('Down' if x < -0.5 else 'Stable')
    )
    comparison['Direction_Match'] = comparison['Direction_Human'] == comparison['Direction_Mouse']

    # Define conserved proteins (same direction, substantial change)
    conserved = comparison[
        (comparison['Direction_Match']) &
        (comparison['Direction_Human'].isin(['Up', 'Down']))
    ].copy()

    # Calculate conservation strength
    conserved['Conservation_Strength'] = np.abs(
        conserved['Human_Avg_Delta'] + conserved['Mouse_Avg_Delta']
    ) / 2

    conserved = conserved.sort_values('Conservation_Strength', ascending=False)

    print(f"\nProteins with conserved aging patterns: {len(conserved)}")
    print(f"  Up-regulated in both: {len(conserved[conserved['Direction_Human'] == 'Up'])}")
    print(f"  Down-regulated in both: {len(conserved[conserved['Direction_Human'] == 'Down'])}")

    print("\nTop 20 conserved aging proteins:")
    print(conserved[['Protein', 'Human_Avg_Delta', 'Mouse_Avg_Delta',
                     'Direction_Human', 'Conservation_Strength']].head(20).to_string(index=False))

    # Save results
    conserved.to_csv(OUTPUT_DIR / 'conserved_aging_proteins.csv', index=False)
    comparison.to_csv(OUTPUT_DIR / 'all_ortholog_comparison.csv', index=False)

    return conserved, comparison

def analyze_therapeutic_targets(conserved, comparison):
    """Analyze conservation of key therapeutic targets"""
    print("\n" + "="*80)
    print("THERAPEUTIC TARGET TRANSLATABILITY")
    print("="*80)

    # Key therapeutic targets from prior analyses (uppercase for matching)
    key_targets = [
        'TIMP3', 'FRZB', 'VCAN', 'FN1', 'COL1A1', 'COL3A1',
        'COMP', 'CILP', 'ACAN', 'BGN', 'DCN', 'LUM',
        'MMP2', 'MMP3', 'MMP9', 'ADAMTS4', 'ADAMTS5'
    ]

    target_analysis = []

    for target in key_targets:
        if target.upper() in comparison['Protein'].values:
            row = comparison[comparison['Protein'] == target.upper()].iloc[0]

            # Calculate translatability index
            if row['Direction_Match']:
                # Agreement bonus
                translatability = 0.8
                # Add magnitude similarity
                mag_similarity = 1 - abs(
                    abs(row['Human_Avg_Delta']) - abs(row['Mouse_Avg_Delta'])
                ) / (abs(row['Human_Avg_Delta']) + abs(row['Mouse_Avg_Delta']) + 0.1)
                translatability += mag_similarity * 0.2
            else:
                # Disagreement penalty
                translatability = 0.3

            target_analysis.append({
                'Protein': target,
                'Human_Delta': row['Human_Avg_Delta'],
                'Mouse_Delta': row['Mouse_Avg_Delta'],
                'Direction_Human': row['Direction_Human'],
                'Direction_Mouse': row['Direction_Mouse'],
                'Conserved': row['Direction_Match'],
                'Translatability_Index': translatability
            })
        else:
            target_analysis.append({
                'Protein': target,
                'Human_Delta': np.nan,
                'Mouse_Delta': np.nan,
                'Direction_Human': 'Not found',
                'Direction_Mouse': 'Not found',
                'Conserved': False,
                'Translatability_Index': 0.0
            })

    target_df = pd.DataFrame(target_analysis)
    target_df = target_df.sort_values('Translatability_Index', ascending=False)

    print("\nTherapeutic Target Assessment:")
    print(target_df.to_string(index=False))

    target_df.to_csv(OUTPUT_DIR / 'therapeutic_target_translatability.csv', index=False)

    return target_df

def create_visualizations(orthologous, human_specific, mouse_specific,
                         conservation_scores, conserved, comparison, target_df):
    """Generate all visualization plots"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. Venn diagram (as bar chart since we can't easily draw circles)
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Human-specific', 'Conserved\n(Orthologous)', 'Mouse-specific']
    counts = [len(human_specific), len(orthologous), len(mouse_specific)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Proteins', fontsize=12, fontweight='bold')
    ax.set_title('Species Distribution of ECM Proteins', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(counts) * 1.2)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'species_distribution_venn.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: species_distribution_venn.png")
    plt.close()

    # 2. Conservation score heatmap (if we have tissue pairs)
    if conservation_scores is not None and len(conservation_scores) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        scores_df = conservation_scores.sort_values('Conservation_Score', ascending=True)

        y_labels = [f"{row['Human_Tissue']} vs\n{row['Mouse_Tissue']}"
                   for _, row in scores_df.iterrows()]

        bars = ax.barh(range(len(scores_df)), scores_df['Conservation_Score'])

        # Color bars by score
        for i, (bar, score) in enumerate(zip(bars, scores_df['Conservation_Score'])):
            if score > 0.5:
                bar.set_color('#2ECC71')  # Green for high conservation
            elif score > 0:
                bar.set_color('#F39C12')  # Orange for moderate
            else:
                bar.set_color('#E74C3C')  # Red for low/negative

        ax.set_yticks(range(len(scores_df)))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('Conservation Score (Pearson r)', fontsize=12, fontweight='bold')
        ax.set_title('Tissue-Level Conservation of ECM Aging Patterns',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_xlim(-1, 1)

        # Add sample size annotations
        for i, (_, row) in enumerate(scores_df.iterrows()):
            ax.text(row['Conservation_Score'] + 0.05, i,
                   f"n={row['N_Proteins']}",
                   va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'tissue_conservation_scores.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: tissue_conservation_scores.png")
        plt.close()

    # 3. Scatter plot: Human vs Mouse delta z-scores
    fig, ax = plt.subplots(figsize=(10, 10))

    comp_clean = comparison.dropna()

    # Plot all proteins
    ax.scatter(comp_clean['Human_Avg_Delta'], comp_clean['Mouse_Avg_Delta'],
              alpha=0.3, s=30, c='gray', label='All orthologs')

    # Highlight conserved proteins
    if len(conserved) > 0:
        conserved_data = comp_clean[comp_clean['Protein'].isin(conserved['Protein'])]
        ax.scatter(conserved_data['Human_Avg_Delta'], conserved_data['Mouse_Avg_Delta'],
                  alpha=0.7, s=60, c='#E74C3C', edgecolors='black', linewidths=1,
                  label='Conserved aging proteins')

    # Highlight therapeutic targets
    targets_in_data = target_df[target_df['Human_Delta'].notna()]
    for _, row in targets_in_data.iterrows():
        ax.scatter(row['Human_Delta'], row['Mouse_Delta'],
                  s=150, marker='*', c='#FFD700', edgecolors='black', linewidths=2,
                  zorder=10)
        ax.annotate(row['Protein'],
                   (row['Human_Delta'], row['Mouse_Delta']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold')

    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.plot([-3, 3], [-3, 3], 'k--', linewidth=1, alpha=0.5, label='Perfect conservation')

    # Calculate and show correlation
    if len(comp_clean) > 0:
        corr, p_val = stats.pearsonr(comp_clean['Human_Avg_Delta'],
                                     comp_clean['Mouse_Avg_Delta'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.2e}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Human Δ Z-score (Old - Young)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mouse Δ Z-score (Old - Young)', fontsize=12, fontweight='bold')
    ax.set_title('Species Conservation of ECM Aging Patterns',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'human_vs_mouse_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: human_vs_mouse_scatter.png")
    plt.close()

    # 4. Translatability index bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    target_sorted = target_df.sort_values('Translatability_Index', ascending=True)

    colors_map = {
        True: '#2ECC71',   # Green for conserved
        False: '#E74C3C'   # Red for not conserved
    }
    colors = [colors_map.get(c, '#95A5A6') for c in target_sorted['Conserved']]

    bars = ax.barh(range(len(target_sorted)), target_sorted['Translatability_Index'],
                   color=colors, edgecolor='black', linewidth=1)

    ax.set_yticks(range(len(target_sorted)))
    ax.set_yticklabels(target_sorted['Protein'], fontsize=10)
    ax.set_xlabel('Translatability Index (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Therapeutic Target Translatability: Mouse Model Validity',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', edgecolor='black', label='Conserved pattern'),
        Patch(facecolor='#E74C3C', edgecolor='black', label='Non-conserved pattern'),
        Patch(facecolor='#95A5A6', edgecolor='black', label='No data')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'therapeutic_translatability.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: therapeutic_translatability.png")
    plt.close()

    # 5. Direction agreement pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Overall direction agreement
    direction_counts = comparison['Direction_Match'].value_counts()
    colors = ['#2ECC71', '#E74C3C']
    explode = (0.05, 0)

    ax1.pie(direction_counts.values, labels=['Conserved', 'Divergent'],
           autopct='%1.1f%%', startangle=90, colors=colors, explode=explode,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Overall Direction Agreement\n(All Orthologous Proteins)',
                 fontsize=12, fontweight='bold', pad=20)

    # Direction breakdown
    direction_matrix = pd.crosstab(comparison['Direction_Human'],
                                   comparison['Direction_Mouse'])

    # Create heatmap
    sns.heatmap(direction_matrix, annot=True, fmt='d', cmap='YlOrRd',
               ax=ax2, cbar_kws={'label': 'Count'},
               linewidths=1, linecolor='black')
    ax2.set_xlabel('Mouse Direction', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Human Direction', fontsize=12, fontweight='bold')
    ax2.set_title('Direction Agreement Matrix', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'direction_agreement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: direction_agreement.png")
    plt.close()

    print("\n✓ All visualizations generated successfully!")

def main():
    """Main execution function"""
    print("="*80)
    print("AGENT 19: SPECIES CONSERVATION ANALYZER")
    print("Human vs Mouse ECM Aging Pattern Conservation")
    print("="*80)

    # Load data
    df = load_data()

    # Identify orthologous proteins
    orthologous, human_specific, mouse_specific = identify_orthologous_proteins(df)

    # Analyze tissue pairs
    conservation_scores = analyze_tissue_pairs(df, orthologous)

    # Analyze conserved aging proteins
    conserved, comparison = analyze_conserved_aging_proteins(df, orthologous)

    # Analyze therapeutic targets
    target_df = analyze_therapeutic_targets(conserved, comparison)

    # Create visualizations
    create_visualizations(
        orthologous, human_specific, mouse_specific,
        conservation_scores, conserved, comparison, target_df
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - orthologous_proteins.csv")
    print("  - human_specific_proteins.csv")
    print("  - mouse_specific_proteins.csv")
    print("  - tissue_conservation_scores.csv")
    print("  - conserved_aging_proteins.csv")
    print("  - all_ortholog_comparison.csv")
    print("  - therapeutic_target_translatability.csv")
    print("  - species_distribution_venn.png")
    print("  - tissue_conservation_scores.png")
    print("  - human_vs_mouse_scatter.png")
    print("  - therapeutic_translatability.png")
    print("  - direction_agreement.png")

if __name__ == "__main__":
    main()
