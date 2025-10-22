"""
H20 - Cross-Species Conservation: Test H08 S100→TGM/LOX Pathway
Tests if S100→crosslinking pathway is conserved in mouse
Agent: claude_code
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("H08 S100 PATHWAY CONSERVATION TEST")
print("="*80)

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# S100 genes and their crosslinking targets
s100_genes = ['S100A9', 'S100A10', 'S100B', 'S100A4', 'S100A6']
tgm_genes = ['TGM2', 'TGM3', 'TGM1']
lox_genes = ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4']

def compute_pathway_correlations(df_species, species_name):
    """
    Compute Spearman correlations between S100 and crosslinking enzymes
    """
    results = []

    # Get protein IDs for each gene
    for s100 in s100_genes:
        s100_data = df_species[df_species['Gene_Symbol'] == s100]

        if len(s100_data) == 0:
            continue

        # Test S100 → TGM correlations
        for tgm in tgm_genes:
            tgm_data = df_species[df_species['Gene_Symbol'] == tgm]

            if len(tgm_data) == 0:
                continue

            # Merge by Sample_ID (construct from metadata)
            # Use Study_ID + Tissue + Age as sample identifier
            s100_df = s100_data[['Study_ID', 'Tissue', 'Zscore_Delta', 'Protein_ID']].copy()
            s100_df.columns = ['Study_ID', 'Tissue', 'S100_Zscore', 'S100_Protein']

            tgm_df = tgm_data[['Study_ID', 'Tissue', 'Zscore_Delta', 'Protein_ID']].copy()
            tgm_df.columns = ['Study_ID', 'Tissue', 'TGM_Zscore', 'TGM_Protein']

            merged = pd.merge(s100_df, tgm_df, on=['Study_ID', 'Tissue'])

            if len(merged) >= 3:  # Need at least 3 samples
                rho, p = spearmanr(merged['S100_Zscore'], merged['TGM_Zscore'])

                results.append({
                    'Species': species_name,
                    'S100_Gene': s100,
                    'Target_Gene': tgm,
                    'Target_Type': 'TGM',
                    'Spearman_Rho': rho,
                    'P_Value': p,
                    'N_Samples': len(merged),
                    'Significant': p < 0.05
                })

        # Test S100 → LOX correlations
        for lox in lox_genes:
            lox_data = df_species[df_species['Gene_Symbol'] == lox]

            if len(lox_data) == 0:
                continue

            lox_df = lox_data[['Study_ID', 'Tissue', 'Zscore_Delta', 'Protein_ID']].copy()
            lox_df.columns = ['Study_ID', 'Tissue', 'LOX_Zscore', 'LOX_Protein']

            merged = pd.merge(s100_df, lox_df, on=['Study_ID', 'Tissue'])

            if len(merged) >= 3:
                rho, p = spearmanr(merged['S100_Zscore'], merged['LOX_Zscore'])

                results.append({
                    'Species': species_name,
                    'S100_Gene': s100,
                    'Target_Gene': lox,
                    'Target_Type': 'LOX',
                    'Spearman_Rho': rho,
                    'P_Value': p,
                    'N_Samples': len(merged),
                    'Significant': p < 0.05
                })

    return pd.DataFrame(results)

# Test in human
print("\n[1/2] Testing S100 pathway in HUMAN...")
df_human = df[df['Species'] == 'Homo sapiens']
results_human = compute_pathway_correlations(df_human, 'Homo sapiens')
print(f"  ✓ Found {len(results_human)} gene pairs")

# Test in mouse
print("\n[2/2] Testing S100 pathway in MOUSE...")
df_mouse = df[df['Species'] == 'Mus musculus']
results_mouse = compute_pathway_correlations(df_mouse, 'Mus musculus')
print(f"  ✓ Found {len(results_mouse)} gene pairs")

# Combine results
df_results = pd.concat([results_human, results_mouse], ignore_index=True)

# Save detailed results
output_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/s100_pathway_conservation_detailed_claude_code.csv'
df_results.to_csv(output_file, index=False)
print(f"\n✓ Saved detailed results to: {output_file}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY: S100 PATHWAY CONSERVATION")
print("="*80)

for species in ['Homo sapiens', 'Mus musculus']:
    sp_results = df_results[df_results['Species'] == species]

    if len(sp_results) == 0:
        print(f"\n{species}: NO DATA")
        continue

    print(f"\n{species}:")
    print(f"  Total pairs tested: {len(sp_results)}")
    print(f"  Significant pairs (p<0.05): {sp_results['Significant'].sum()}")

    # Mean correlation
    mean_rho = sp_results['Spearman_Rho'].mean()
    median_rho = sp_results['Spearman_Rho'].median()
    print(f"  Mean ρ: {mean_rho:.3f}")
    print(f"  Median ρ: {median_rho:.3f}")

    # Strongest correlations
    top5 = sp_results.nlargest(5, 'Spearman_Rho')
    print(f"\n  Top 5 correlations:")
    for _, row in top5.iterrows():
        sig_marker = '*' if row['Significant'] else ''
        print(f"    {row['S100_Gene']:10} → {row['Target_Gene']:10} | ρ={row['Spearman_Rho']:6.3f} (p={row['P_Value']:.4f}){sig_marker}")

# Comparison: Human vs Mouse
print("\n" + "="*80)
print("CROSS-SPECIES COMPARISON")
print("="*80)

# For each pair present in both species, compare correlations
human_pairs = set(zip(results_human['S100_Gene'], results_human['Target_Gene']))
mouse_pairs = set(zip(results_mouse['S100_Gene'], results_mouse['Target_Gene']))
common_pairs = human_pairs & mouse_pairs

print(f"\nGene pairs present in both species: {len(common_pairs)}")

if len(common_pairs) > 0:
    comparison = []

    for s100, target in common_pairs:
        h_row = results_human[(results_human['S100_Gene'] == s100) & (results_human['Target_Gene'] == target)].iloc[0]
        m_row = results_mouse[(results_mouse['S100_Gene'] == s100) & (results_mouse['Target_Gene'] == target)].iloc[0]

        comparison.append({
            'S100_Gene': s100,
            'Target_Gene': target,
            'Target_Type': h_row['Target_Type'],
            'Human_Rho': h_row['Spearman_Rho'],
            'Human_P': h_row['P_Value'],
            'Mouse_Rho': m_row['Spearman_Rho'],
            'Mouse_P': m_row['P_Value'],
            'Rho_Difference': abs(h_row['Spearman_Rho'] - m_row['Spearman_Rho']),
            'Both_Significant': (h_row['P_Value'] < 0.05) and (m_row['P_Value'] < 0.05),
            'Conserved': (h_row['Spearman_Rho'] > 0.5) and (m_row['Spearman_Rho'] > 0.5)
        })

    df_comparison = pd.DataFrame(comparison)

    # Save comparison
    comp_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/s100_pathway_comparison_claude_code.csv'
    df_comparison.to_csv(comp_file, index=False)
    print(f"✓ Saved comparison to: {comp_file}")

    # Statistics
    conserved_count = df_comparison['Conserved'].sum()
    print(f"\nConserved pairs (ρ>0.5 in both): {conserved_count}/{len(df_comparison)}")

    # Best conserved
    print(f"\nBest conserved correlations:")
    print(f"{'S100':<10} {'Target':<10} {'Type':<5} {'Human ρ':<10} {'Mouse ρ':<10} {'Δρ':<10}")
    print("-"*80)
    best = df_comparison.nsmallest(10, 'Rho_Difference')
    for _, row in best.iterrows():
        print(f"{row['S100_Gene']:<10} {row['Target_Gene']:<10} {row['Target_Type']:<5} "
              f"{row['Human_Rho']:>9.3f} {row['Mouse_Rho']:>9.3f} {row['Rho_Difference']:>9.3f}")

# Visualization
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Correlation distribution by species
ax1 = axes[0, 0]
for species in ['Homo sapiens', 'Mus musculus']:
    sp_data = df_results[df_results['Species'] == species]['Spearman_Rho']
    label = 'Human' if species == 'Homo sapiens' else 'Mouse'
    ax1.hist(sp_data, bins=20, alpha=0.6, label=label, edgecolor='black')
ax1.axvline(0.6, color='red', linestyle='--', label='Conservation threshold (ρ=0.6)')
ax1.set_xlabel('Spearman ρ')
ax1.set_ylabel('Frequency')
ax1.set_title('S100→TGM/LOX Correlation Distribution')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Human vs Mouse scatter
if len(comparison) > 0:
    ax2 = axes[0, 1]
    colors = ['green' if c else 'gray' for c in df_comparison['Conserved']]
    ax2.scatter(df_comparison['Human_Rho'], df_comparison['Mouse_Rho'],
                c=colors, alpha=0.6, s=100, edgecolors='black')
    ax2.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='Perfect conservation')
    ax2.axhline(0.5, color='red', linestyle=':', alpha=0.5)
    ax2.axvline(0.5, color='red', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Human Spearman ρ')
    ax2.set_ylabel('Mouse Spearman ρ')
    ax2.set_title('Cross-Species Correlation Comparison')
    ax2.grid(alpha=0.3)
    ax2.legend()

# 3. Bar plot: Mean correlation by target type
ax3 = axes[1, 0]
summary = df_results.groupby(['Species', 'Target_Type'])['Spearman_Rho'].mean().reset_index()
species_labels = summary['Species'].map({'Homo sapiens': 'Human', 'Mus musculus': 'Mouse'})
x = np.arange(len(summary))
bars = ax3.bar(x, summary['Spearman_Rho'], color=['steelblue' if 'Human' in lbl else 'coral' for lbl in species_labels])
ax3.set_xticks(x)
ax3.set_xticklabels([f"{lbl}\n{tt}" for lbl, tt in zip(species_labels, summary['Target_Type'])], rotation=0)
ax3.set_ylabel('Mean Spearman ρ')
ax3.set_title('Mean S100 Pathway Correlation by Target Type')
ax3.axhline(0.6, color='red', linestyle='--', label='Conservation threshold')
ax3.grid(alpha=0.3, axis='y')
ax3.legend()

# 4. Heatmap: Key pairs
if len(comparison) > 0:
    ax4 = axes[1, 1]
    # Select top conserved pairs
    top_pairs = df_comparison.nsmallest(10, 'Rho_Difference')
    heatmap_data = top_pairs[['Human_Rho', 'Mouse_Rho']].T
    heatmap_data.columns = [f"{row['S100_Gene']}→{row['Target_Gene']}" for _, row in top_pairs.iterrows()]

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, ax=ax4, cbar_kws={'label': 'Spearman ρ'})
    ax4.set_title('Top 10 Conserved S100 Pathway Correlations')
    ax4.set_ylabel('Species')
    ax4.set_yticklabels(['Human', 'Mouse'], rotation=0)

plt.tight_layout()
viz_file = '13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_20_cross_species_conservation/claude_code/visualizations_claude_code/s100_pathway_conservation_claude_code.png'
plt.savefig(viz_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {viz_file}")
plt.close()

print("\n" + "="*80)
print("H08 CONSERVATION TEST COMPLETED")
print("="*80)
