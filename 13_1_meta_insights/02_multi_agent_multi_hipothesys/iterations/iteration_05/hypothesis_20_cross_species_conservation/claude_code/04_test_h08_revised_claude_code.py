"""
H20 - Cross-Species Conservation: Test H08 S100→TGM/LOX Pathway (REVISED)
Uses tissue-level aggregation for sparse data
Agent: claude_code
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("H08 S100 PATHWAY CONSERVATION TEST (REVISED)")
print("="*80)

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Normalize gene symbols (handle case variations)
df['Gene_Symbol_Upper'] = df['Gene_Symbol'].str.upper()

# S100 genes and their crosslinking targets
s100_genes = ['S100A9', 'S100A10', 'S100B', 'S100A4', 'S100A6', 'S100A8', 'S100A11']
tgm_genes = ['TGM2', 'TGM3', 'TGM1']
lox_genes = ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4']

def aggregate_by_tissue_species(df_species, gene_list, species_name):
    """
    Aggregate protein abundance by tissue (mean z-score)
    """
    tissue_means = {}

    for gene in gene_list:
        gene_data = df_species[df_species['Gene_Symbol_Upper'] == gene.upper()]

        if len(gene_data) == 0:
            continue

        # Group by tissue, compute mean z-score
        tissue_agg = gene_data.groupby('Tissue')['Zscore_Delta'].mean()

        tissue_means[gene] = tissue_agg.to_dict()

    # Convert to DataFrame (tissues × genes)
    df_agg = pd.DataFrame(tissue_means)
    df_agg.index.name = 'Tissue'

    return df_agg

def compute_correlations_from_aggregates(df_s100, df_target, target_type):
    """
    Compute correlations between S100 genes and target genes
    """
    results = []

    for s100 in df_s100.columns:
        for target in df_target.columns:
            # Find common tissues
            common_tissues = df_s100.index.intersection(df_target.index)

            # Need data for both genes in same tissues
            s100_values = df_s100.loc[common_tissues, s100].dropna()
            target_values = df_target.loc[common_tissues, target].dropna()

            # Align
            common_idx = s100_values.index.intersection(target_values.index)

            if len(common_idx) >= 3:  # Need at least 3 tissues
                s100_vec = s100_values.loc[common_idx]
                target_vec = target_values.loc[common_idx]

                rho, p_rho = spearmanr(s100_vec, target_vec)
                r, p_r = pearsonr(s100_vec, target_vec)

                results.append({
                    'S100_Gene': s100,
                    'Target_Gene': target,
                    'Target_Type': target_type,
                    'Spearman_Rho': rho,
                    'Spearman_P': p_rho,
                    'Pearson_R': r,
                    'Pearson_P': p_r,
                    'N_Tissues': len(common_idx),
                    'Significant_Spearman': p_rho < 0.05,
                    'Significant_Pearson': p_r < 0.05
                })

    return pd.DataFrame(results)

# HUMAN ANALYSIS
print("\n[1/2] Analyzing HUMAN data...")
df_human = df[df['Species'] == 'Homo sapiens']

human_s100 = aggregate_by_tissue_species(df_human, s100_genes, 'Human')
human_tgm = aggregate_by_tissue_species(df_human, tgm_genes, 'Human')
human_lox = aggregate_by_tissue_species(df_human, lox_genes, 'Human')

print(f"  Human S100 genes: {human_s100.shape[1]} genes × {human_s100.shape[0]} tissues")
print(f"  Human TGM genes: {human_tgm.shape[1]} genes × {human_tgm.shape[0]} tissues")
print(f"  Human LOX genes: {human_lox.shape[1]} genes × {human_lox.shape[0]} tissues")

# Compute correlations
human_s100_tgm = compute_correlations_from_aggregates(human_s100, human_tgm, 'TGM')
human_s100_lox = compute_correlations_from_aggregates(human_s100, human_lox, 'LOX')
results_human = pd.concat([human_s100_tgm, human_s100_lox], ignore_index=True)
results_human['Species'] = 'Homo sapiens'

print(f"  ✓ Found {len(results_human)} gene pairs")

# MOUSE ANALYSIS
print("\n[2/2] Analyzing MOUSE data...")
df_mouse = df[df['Species'] == 'Mus musculus']

mouse_s100 = aggregate_by_tissue_species(df_mouse, s100_genes, 'Mouse')
mouse_tgm = aggregate_by_tissue_species(df_mouse, tgm_genes, 'Mouse')
mouse_lox = aggregate_by_tissue_species(df_mouse, lox_genes, 'Mouse')

print(f"  Mouse S100 genes: {mouse_s100.shape[1]} genes × {mouse_s100.shape[0]} tissues")
print(f"  Mouse TGM genes: {mouse_tgm.shape[1]} genes × {mouse_tgm.shape[0]} tissues")
print(f"  Mouse LOX genes: {mouse_lox.shape[1]} genes × {mouse_lox.shape[0]} tissues")

# Compute correlations
mouse_s100_tgm = compute_correlations_from_aggregates(mouse_s100, mouse_tgm, 'TGM')
mouse_s100_lox = compute_correlations_from_aggregates(mouse_s100, mouse_lox, 'LOX')
results_mouse = pd.concat([mouse_s100_tgm, mouse_s100_lox], ignore_index=True)
results_mouse['Species'] = 'Mus musculus'

print(f"  ✓ Found {len(results_mouse)} gene pairs")

# Combine
df_results = pd.concat([results_human, results_mouse], ignore_index=True)

# Save
output_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/s100_pathway_conservation_claude_code.csv'
df_results.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed results to: {output_file}")

# SUMMARY
print("\n" + "="*80)
print("SUMMARY: S100 PATHWAY CONSERVATION")
print("="*80)

for species in ['Homo sapiens', 'Mus musculus']:
    sp_results = df_results[df_results['Species'] == species]

    if len(sp_results) == 0:
        print(f"\n{species}: NO DATA")
        continue

    sp_label = 'HUMAN' if species == 'Homo sapiens' else 'MOUSE'
    print(f"\n{sp_label}:")
    print(f"  Total pairs tested: {len(sp_results)}")
    print(f"  Significant pairs (Spearman p<0.05): {sp_results['Significant_Spearman'].sum()}")
    print(f"  Mean Spearman ρ: {sp_results['Spearman_Rho'].mean():.3f}")
    print(f"  Median Spearman ρ: {sp_results['Spearman_Rho'].median():.3f}")
    print(f"  Strong correlations (|ρ|>0.6): {(sp_results['Spearman_Rho'].abs() > 0.6).sum()}")

    # Top 5
    top5 = sp_results.nlargest(5, 'Spearman_Rho')
    print(f"\n  Top 5 S100→Target correlations:")
    for _, row in top5.iterrows():
        sig = '*' if row['Significant_Spearman'] else ''
        print(f"    {row['S100_Gene']:10} → {row['Target_Gene']:8} ({row['Target_Type']:3}) | "
              f"ρ={row['Spearman_Rho']:6.3f} (p={row['Spearman_P']:.4f}) | n={row['N_Tissues']} tissues{sig}")

# CROSS-SPECIES COMPARISON
print("\n" + "="*80)
print("CROSS-SPECIES COMPARISON")
print("="*80)

comparison = []

for _, h_row in results_human.iterrows():
    s100 = h_row['S100_Gene']
    target = h_row['Target_Gene']

    # Find matching pair in mouse
    m_matches = results_mouse[(results_mouse['S100_Gene'] == s100) & (results_mouse['Target_Gene'] == target)]

    if len(m_matches) > 0:
        m_row = m_matches.iloc[0]

        comparison.append({
            'S100_Gene': s100,
            'Target_Gene': target,
            'Target_Type': h_row['Target_Type'],
            'Human_Rho': h_row['Spearman_Rho'],
            'Human_P': h_row['Spearman_P'],
            'Human_N': h_row['N_Tissues'],
            'Mouse_Rho': m_row['Spearman_Rho'],
            'Mouse_P': m_row['Spearman_P'],
            'Mouse_N': m_row['N_Tissues'],
            'Rho_Diff': abs(h_row['Spearman_Rho'] - m_row['Spearman_Rho']),
            'Both_Positive': (h_row['Spearman_Rho'] > 0) and (m_row['Spearman_Rho'] > 0),
            'Both_Significant': h_row['Significant_Spearman'] and m_row['Significant_Spearman'],
            'Conserved_Strong': (h_row['Spearman_Rho'] > 0.6) and (m_row['Spearman_Rho'] > 0.6)
        })

df_comparison = pd.DataFrame(comparison)

if len(df_comparison) > 0:
    comp_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/s100_pathway_cross_species_comparison_claude_code.csv'
    df_comparison.to_csv(comp_file, index=False)
    print(f"✓ Saved comparison to: {comp_file}")

    print(f"\nPairs tested in both species: {len(df_comparison)}")
    print(f"Both positive correlation: {df_comparison['Both_Positive'].sum()}")
    print(f"Both statistically significant: {df_comparison['Both_Significant'].sum()}")
    print(f"Conserved (ρ>0.6 in both): {df_comparison['Conserved_Strong'].sum()}")

    print(f"\n{'S100':<12} {'Target':<10} {'Type':<5} {'Human ρ':<10} {'Mouse ρ':<10} {'Δρ':<8} {'Status'}")
    print("-"*80)
    for _, row in df_comparison.nlargest(15, 'Human_Rho').iterrows():
        status = '✓ CONSERVED' if row['Conserved_Strong'] else ('± Partial' if row['Both_Positive'] else '✗ Different')
        print(f"{row['S100_Gene']:<12} {row['Target_Gene']:<10} {row['Target_Type']:<5} "
              f"{row['Human_Rho']:>9.3f} {row['Mouse_Rho']:>9.3f} {row['Rho_Diff']:>7.3f}  {status}")

# VISUALIZATIONS
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Distribution comparison
ax1 = fig.add_subplot(gs[0, 0])
human_rho = results_human['Spearman_Rho'].dropna()
mouse_rho = results_mouse['Spearman_Rho'].dropna()
ax1.hist([human_rho, mouse_rho], bins=20, label=['Human', 'Mouse'], alpha=0.7, edgecolor='black')
ax1.axvline(0.6, color='red', linestyle='--', linewidth=2, label='Threshold (ρ=0.6)')
ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax1.set_xlabel('Spearman ρ (S100→TGM/LOX)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Correlation Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Human vs Mouse scatter
if len(df_comparison) > 0:
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['green' if c else ('orange' if p else 'gray') for c, p in zip(df_comparison['Conserved_Strong'], df_comparison['Both_Positive'])]
    ax2.scatter(df_comparison['Human_Rho'], df_comparison['Mouse_Rho'],
                c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax2.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='Perfect conservation')
    ax2.axhline(0.6, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(0.6, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Human Spearman ρ', fontsize=11)
    ax2.set_ylabel('Mouse Spearman ρ', fontsize=11)
    ax2.set_title('Cross-Species Correlation', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()

# 3. Conservation rate by target type
ax3 = fig.add_subplot(gs[0, 2])
if len(df_comparison) > 0:
    cons_by_type = df_comparison.groupby('Target_Type').agg({
        'Conserved_Strong': 'sum',
        'S100_Gene': 'count'
    })
    cons_by_type['Rate'] = cons_by_type['Conserved_Strong'] / cons_by_type['S100_Gene'] * 100
    ax3.bar(cons_by_type.index, cons_by_type['Rate'], color=['steelblue', 'coral'], edgecolor='black', linewidth=2)
    ax3.set_ylabel('Conservation Rate (%)', fontsize=11)
    ax3.set_xlabel('Target Type', fontsize=11)
    ax3.set_title('Conservation Rate by Target', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_ylim([0, 100])

# 4. Top conserved pairs heatmap
if len(df_comparison) > 0:
    ax4 = fig.add_subplot(gs[1, :2])
    top10 = df_comparison.nlargest(10, 'Human_Rho')
    heatmap_data = top10[['Human_Rho', 'Mouse_Rho']].T
    heatmap_data.columns = [f"{row['S100_Gene']}→{row['Target_Gene']}" for _, row in top10.iterrows()]

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, ax=ax4, cbar_kws={'label': 'Spearman ρ'}, linewidths=1)
    ax4.set_title('Top 10 S100 Pathway Pairs', fontsize=12, fontweight='bold')
    ax4.set_yticklabels(['Human', 'Mouse'], rotation=0)

# 5. Bar plot: Mean by species and type
ax5 = fig.add_subplot(gs[1, 2])
summary = df_results.groupby(['Species', 'Target_Type'])['Spearman_Rho'].mean().reset_index()
summary['Label'] = summary['Species'].map({'Homo sapiens': 'Human', 'Mus musculus': 'Mouse'}) + '\n' + summary['Target_Type']
x = np.arange(len(summary))
colors = ['steelblue' if 'Human' in lbl else 'coral' for lbl in summary['Label']]
ax5.bar(x, summary['Spearman_Rho'], color=colors, edgecolor='black', linewidth=1.5)
ax5.set_xticks(x)
ax5.set_xticklabels(summary['Label'], fontsize=9)
ax5.set_ylabel('Mean Spearman ρ', fontsize=11)
ax5.set_title('Mean Correlation by Type', fontsize=12, fontweight='bold')
ax5.axhline(0.6, color='red', linestyle='--', label='Threshold')
ax5.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax5.grid(alpha=0.3, axis='y')
ax5.legend()

# 6. Sample size comparison
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist([results_human['N_Tissues'], results_mouse['N_Tissues']], bins=10, label=['Human', 'Mouse'], alpha=0.7, edgecolor='black')
ax6.set_xlabel('Number of Tissues', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Sample Size Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 7. Significance comparison
ax7 = fig.add_subplot(gs[2, 1])
sig_data = [
    [results_human['Significant_Spearman'].sum(), len(results_human) - results_human['Significant_Spearman'].sum()],
    [results_mouse['Significant_Spearman'].sum(), len(results_mouse) - results_mouse['Significant_Spearman'].sum()]
]
ax7.bar(['Human', 'Mouse'], [s[0] for s in sig_data], label='Significant (p<0.05)', color='green', alpha=0.7, edgecolor='black')
ax7.bar(['Human', 'Mouse'], [s[1] for s in sig_data], bottom=[s[0] for s in sig_data], label='Not significant', color='gray', alpha=0.5, edgecolor='black')
ax7.set_ylabel('Number of Pairs', fontsize=11)
ax7.set_title('Statistical Significance', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3, axis='y')

# 8. Text summary
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

if len(df_comparison) > 0:
    conserved_count = df_comparison['Conserved_Strong'].sum()
    partial_count = df_comparison['Both_Positive'].sum() - conserved_count
    total_count = len(df_comparison)

    summary_text = f"""
H08 CONSERVATION SUMMARY

Tested pairs: {total_count}
✓ Conserved (ρ>0.6): {conserved_count}
± Partial (both +): {partial_count}
✗ Different: {total_count - conserved_count - partial_count}

Conservation rate: {100*conserved_count/total_count:.1f}%

VERDICT:
{'STRONG conservation' if conserved_count/total_count > 0.5 else 'PARTIAL conservation' if conserved_count/total_count > 0.2 else 'WEAK conservation'}
"""
else:
    summary_text = "Insufficient data for comparison"

ax8.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('H08: S100→TGM/LOX Pathway Cross-Species Conservation Analysis', fontsize=14, fontweight='bold', y=0.995)
viz_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/visualizations_claude_code/h08_s100_pathway_conservation_claude_code.png'
plt.savefig(viz_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {viz_file}")
plt.close()

print("\n" + "="*80)
print("H08 CONSERVATION TEST COMPLETED")
print("="*80)