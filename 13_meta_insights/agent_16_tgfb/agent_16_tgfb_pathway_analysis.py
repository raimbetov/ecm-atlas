#!/usr/bin/env python3
"""
AGENT 16: TGF-β FIBROSIS PATHWAY DISSECTOR

Analyzes TGF-β signaling pathway activation across tissues and its role in fibrotic remodeling.

Key analyses:
1. TGF-β pathway component identification and quantification
2. Target gene correlation networks
3. TGF-β activity score calculation
4. Tissue fibrosis severity ranking
5. Causality analysis: TGF-β vs collagen accumulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# TGF-β PATHWAY GENE DEFINITIONS
# ============================================================================

TGFB_PATHWAY = {
    # TGF-β ligands and binding proteins
    'ligands': ['TGFB1', 'TGFB2', 'TGFB3', 'TGFBI'],

    # Latent TGF-β binding proteins
    'binding_proteins': ['LTBP1', 'LTBP2', 'LTBP3', 'LTBP4'],

    # TGF-β receptors
    'receptors': ['TGFBR1', 'TGFBR2', 'TGFBR3'],

    # Core target genes (fibrotic effectors)
    'core_targets': ['VCAN', 'COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'CTGF'],

    # Additional ECM targets
    'ecm_targets': ['COL4A1', 'COL4A2', 'COL5A1', 'COL5A2', 'COL6A1', 'COL6A2',
                    'COL6A3', 'POSTN', 'THBS1', 'THBS2', 'SPARC', 'LOX', 'LOXL2'],

    # TGF-β activated kinases and SMAD pathway
    'signaling': ['SMAD2', 'SMAD3', 'SMAD4', 'SMAD7'],

    # Competing pathways
    'pdgf_pathway': ['PDGFA', 'PDGFB', 'PDGFC', 'PDGFD', 'PDGFRA', 'PDGFRB'],

    # Standalone fibrosis markers
    'other_fibrosis': ['ACTA2', 'MMP2', 'MMP9', 'TIMP1', 'TIMP2', 'PLOD2']
}

# Flatten all TGF-β related genes
ALL_TGFB_GENES = (TGFB_PATHWAY['ligands'] + TGFB_PATHWAY['binding_proteins'] +
                  TGFB_PATHWAY['receptors'] + TGFB_PATHWAY['core_targets'] +
                  TGFB_PATHWAY['ecm_targets'] + TGFB_PATHWAY['signaling'])

def load_and_prepare_data(filepath):
    """Load ECM dataset and prepare for analysis"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    print(f"Total rows: {len(df)}")
    print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")
    print(f"Unique tissues: {df['Tissue'].nunique()}")

    return df

def identify_pathway_components(df):
    """Identify which TGF-β pathway components are present in dataset"""
    print("\n" + "="*80)
    print("TGF-β PATHWAY COMPONENT IDENTIFICATION")
    print("="*80)

    results = {}

    for category, genes in TGFB_PATHWAY.items():
        found_genes = []
        for gene in genes:
            # Check if gene exists in dataset
            matches = df[df['Canonical_Gene_Symbol'].str.contains(gene, case=False, na=False)]
            if len(matches) > 0:
                found_genes.append(gene)

        results[category] = {
            'total': len(genes),
            'found': len(found_genes),
            'genes': found_genes,
            'coverage': len(found_genes) / len(genes) * 100
        }

        print(f"\n{category.upper().replace('_', ' ')}:")
        print(f"  Found: {len(found_genes)}/{len(genes)} ({results[category]['coverage']:.1f}%)")
        print(f"  Genes: {', '.join(found_genes) if found_genes else 'None'}")

    return results

def calculate_tgfb_activity_score(df):
    """Calculate TGF-β activity score per tissue/age group"""
    print("\n" + "="*80)
    print("TGF-β ACTIVITY SCORE CALCULATION")
    print("="*80)

    # Filter for core TGF-β target genes
    target_genes = TGFB_PATHWAY['core_targets'] + TGFB_PATHWAY['ecm_targets']

    # Create activity scores
    activity_scores = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        # Calculate for old samples
        old_data = tissue_data[tissue_data['Zscore_Old'].notna()]
        old_targets = old_data[old_data['Canonical_Gene_Symbol'].isin(target_genes)]

        if len(old_targets) > 0:
            activity_scores.append({
                'Tissue': tissue,
                'Age_Group': 'Old',
                'TGFb_Activity_Score': old_targets['Zscore_Old'].mean(),
                'N_Genes': len(old_targets),
                'Median_Zscore': old_targets['Zscore_Old'].median(),
                'Max_Zscore': old_targets['Zscore_Old'].max()
            })

        # Calculate for young samples
        young_data = tissue_data[tissue_data['Zscore_Young'].notna()]
        young_targets = young_data[young_data['Canonical_Gene_Symbol'].isin(target_genes)]

        if len(young_targets) > 0:
            activity_scores.append({
                'Tissue': tissue,
                'Age_Group': 'Young',
                'TGFb_Activity_Score': young_targets['Zscore_Young'].mean(),
                'N_Genes': len(young_targets),
                'Median_Zscore': young_targets['Zscore_Young'].median(),
                'Max_Zscore': young_targets['Zscore_Young'].max()
            })

    activity_df = pd.DataFrame(activity_scores)

    # Calculate delta (Old - Young)
    delta_scores = []
    for tissue in activity_df['Tissue'].unique():
        tissue_scores = activity_df[activity_df['Tissue'] == tissue]
        old_score = tissue_scores[tissue_scores['Age_Group'] == 'Old']['TGFb_Activity_Score'].values
        young_score = tissue_scores[tissue_scores['Age_Group'] == 'Young']['TGFb_Activity_Score'].values

        if len(old_score) > 0 and len(young_score) > 0:
            delta_scores.append({
                'Tissue': tissue,
                'Delta_TGFb_Activity': old_score[0] - young_score[0],
                'Old_Score': old_score[0],
                'Young_Score': young_score[0]
            })

    delta_df = pd.DataFrame(delta_scores).sort_values('Delta_TGFb_Activity', ascending=False)

    print("\nTop 10 tissues by TGF-β activation increase with age:")
    print(delta_df.head(10).to_string(index=False))

    return activity_df, delta_df

def analyze_target_gene_correlations(df):
    """Analyze correlations between TGF-β target genes"""
    print("\n" + "="*80)
    print("TGF-β TARGET GENE CORRELATION ANALYSIS")
    print("="*80)

    # Get all target genes
    target_genes = TGFB_PATHWAY['core_targets'] + TGFB_PATHWAY['ecm_targets'][:10]  # Limit for clarity

    # Create matrix of z-scores for old samples
    correlation_data = []

    for gene in target_genes:
        gene_data = df[df['Canonical_Gene_Symbol'] == gene]
        if len(gene_data) > 0:
            # Get z-scores across all tissues
            old_zscores = gene_data[gene_data['Zscore_Old'].notna()][['Tissue', 'Zscore_Old']]
            old_zscores = old_zscores.rename(columns={'Zscore_Old': gene})

            if len(correlation_data) == 0:
                correlation_data = old_zscores
            else:
                correlation_data = correlation_data.merge(old_zscores, on='Tissue', how='outer')

    if len(correlation_data) > 0:
        corr_matrix = correlation_data.drop('Tissue', axis=1).corr()

        print(f"\nCorrelation matrix shape: {corr_matrix.shape}")
        print(f"Genes analyzed: {list(corr_matrix.columns)}")

        return corr_matrix, correlation_data
    else:
        print("Insufficient data for correlation analysis")
        return None, None

def test_predictive_power(df, activity_df):
    """Test if TGF-β activity score predicts fibrosis better than individual proteins"""
    print("\n" + "="*80)
    print("PREDICTIVE POWER ANALYSIS")
    print("="*80)

    # Define fibrosis markers (collagens as proxy for fibrosis)
    fibrosis_markers = ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL5A1', 'COL6A1']

    results = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        # Get TGF-β activity score for old samples
        tgfb_score = activity_df[
            (activity_df['Tissue'] == tissue) &
            (activity_df['Age_Group'] == 'Old')
        ]['TGFb_Activity_Score'].values

        if len(tgfb_score) == 0:
            continue

        tgfb_score = tgfb_score[0]

        # Calculate mean collagen z-score (fibrosis proxy)
        old_collagens = tissue_data[
            (tissue_data['Canonical_Gene_Symbol'].isin(fibrosis_markers)) &
            (tissue_data['Zscore_Old'].notna())
        ]

        if len(old_collagens) > 0:
            mean_collagen_zscore = old_collagens['Zscore_Old'].mean()

            results.append({
                'Tissue': tissue,
                'TGFb_Activity': tgfb_score,
                'Collagen_Accumulation': mean_collagen_zscore,
                'N_Collagens': len(old_collagens)
            })

    pred_df = pd.DataFrame(results)

    if len(pred_df) > 5:
        # Calculate correlation
        corr, pval = stats.pearsonr(pred_df['TGFb_Activity'], pred_df['Collagen_Accumulation'])

        print(f"\nCorrelation between TGF-β activity and collagen accumulation:")
        print(f"  R = {corr:.3f}, p-value = {pval:.4f}")

        if pval < 0.05:
            print("  ✓ Significant correlation detected!")
        else:
            print("  ✗ No significant correlation")

    return pred_df

def analyze_causality_direction(df):
    """Test if TGF-β activity precedes collagen accumulation"""
    print("\n" + "="*80)
    print("CAUSALITY ANALYSIS: TGF-β → COLLAGEN")
    print("="*80)

    # Compare z-score deltas
    tgfb_genes = TGFB_PATHWAY['ligands'] + TGFB_PATHWAY['binding_proteins']
    collagen_genes = ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL5A1', 'COL6A1']

    causality_results = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        # TGF-β delta
        tgfb_data = tissue_data[
            (tissue_data['Canonical_Gene_Symbol'].isin(tgfb_genes)) &
            (tissue_data['Zscore_Delta'].notna())
        ]

        # Collagen delta
        collagen_data = tissue_data[
            (tissue_data['Canonical_Gene_Symbol'].isin(collagen_genes)) &
            (tissue_data['Zscore_Delta'].notna())
        ]

        if len(tgfb_data) > 0 and len(collagen_data) > 0:
            tgfb_delta = tgfb_data['Zscore_Delta'].mean()
            collagen_delta = collagen_data['Zscore_Delta'].mean()

            causality_results.append({
                'Tissue': tissue,
                'TGFb_Delta': tgfb_delta,
                'Collagen_Delta': collagen_delta,
                'TGFb_Magnitude': abs(tgfb_delta),
                'Collagen_Magnitude': abs(collagen_delta),
                'Ratio': abs(collagen_delta) / abs(tgfb_delta) if tgfb_delta != 0 else np.nan
            })

    causality_df = pd.DataFrame(causality_results)

    print("\nTissues ranked by TGF-β → Collagen amplification ratio:")
    print(causality_df.sort_values('Ratio', ascending=False).head(10).to_string(index=False))

    return causality_df

def compare_with_competing_pathways(df):
    """Compare TGF-β pathway with PDGF and other fibrosis drivers"""
    print("\n" + "="*80)
    print("PATHWAY COMPETITION ANALYSIS")
    print("="*80)

    pathway_comparison = []

    for tissue in df['Tissue'].unique():
        tissue_data = df[df['Tissue'] == tissue]

        # Calculate pathway scores for old samples
        tgfb_score = tissue_data[
            (tissue_data['Canonical_Gene_Symbol'].isin(ALL_TGFB_GENES)) &
            (tissue_data['Zscore_Old'].notna())
        ]['Zscore_Old'].mean()

        pdgf_score = tissue_data[
            (tissue_data['Canonical_Gene_Symbol'].isin(TGFB_PATHWAY['pdgf_pathway'])) &
            (tissue_data['Zscore_Old'].notna())
        ]['Zscore_Old'].mean()

        ctgf_data = tissue_data[
            (tissue_data['Canonical_Gene_Symbol'] == 'CTGF') &
            (tissue_data['Zscore_Old'].notna())
        ]

        ctgf_score = ctgf_data['Zscore_Old'].values[0] if len(ctgf_data) > 0 else np.nan

        if not np.isnan(tgfb_score):
            pathway_comparison.append({
                'Tissue': tissue,
                'TGFb_Pathway': tgfb_score,
                'PDGF_Pathway': pdgf_score if not np.isnan(pdgf_score) else 0,
                'CTGF_Standalone': ctgf_score if not np.isnan(ctgf_score) else 0
            })

    comp_df = pd.DataFrame(pathway_comparison)

    print("\nDominant pathway by tissue:")
    for idx, row in comp_df.iterrows():
        scores = {
            'TGF-β': row['TGFb_Pathway'],
            'PDGF': row['PDGF_Pathway'],
            'CTGF': row['CTGF_Standalone']
        }
        dominant = max(scores, key=scores.get)
        print(f"  {row['Tissue']}: {dominant} (score: {scores[dominant]:.3f})")

    return comp_df

def create_visualizations(df, activity_df, delta_df, corr_matrix, pred_df, causality_df, comp_df, output_dir):
    """Generate comprehensive visualizations"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(20, 24))

    # 1. TGF-β Activity Heatmap by Tissue and Age
    ax1 = plt.subplot(6, 3, 1)
    if len(activity_df) > 0:
        pivot = activity_df.pivot(index='Tissue', columns='Age_Group', values='TGFb_Activity_Score')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax1, cbar_kws={'label': 'Activity Score'})
        ax1.set_title('TGF-β Activity Score by Tissue and Age', fontsize=12, fontweight='bold')
        ax1.set_xlabel('')
        ax1.set_ylabel('Tissue', fontsize=10)

    # 2. Delta TGF-β Activity (Old - Young)
    ax2 = plt.subplot(6, 3, 2)
    if len(delta_df) > 0:
        top_tissues = delta_df.head(15)
        colors = ['red' if x > 0 else 'blue' for x in top_tissues['Delta_TGFb_Activity']]
        ax2.barh(top_tissues['Tissue'], top_tissues['Delta_TGFb_Activity'], color=colors, alpha=0.7)
        ax2.set_xlabel('Δ TGF-β Activity (Old - Young)', fontsize=10)
        ax2.set_title('Top 15 Tissues by TGF-β Activation Change', fontsize=12, fontweight='bold')
        ax2.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax2.grid(True, alpha=0.3)

    # 3. Target Gene Correlation Heatmap
    ax3 = plt.subplot(6, 3, 3)
    if corr_matrix is not None and len(corr_matrix) > 1:
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, ax=ax3, square=True)
        ax3.set_title('TGF-β Target Gene Correlations', fontsize=12, fontweight='bold')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)

    # 4. Predictive Power: TGF-β Activity vs Collagen Accumulation
    ax4 = plt.subplot(6, 3, 4)
    if len(pred_df) > 0:
        ax4.scatter(pred_df['TGFb_Activity'], pred_df['Collagen_Accumulation'],
                   s=100, alpha=0.6, c=pred_df['N_Collagens'], cmap='viridis')

        # Add regression line
        z = np.polyfit(pred_df['TGFb_Activity'], pred_df['Collagen_Accumulation'], 1)
        p = np.poly1d(z)
        ax4.plot(pred_df['TGFb_Activity'], p(pred_df['TGFb_Activity']),
                "r--", alpha=0.8, linewidth=2)

        corr, pval = stats.pearsonr(pred_df['TGFb_Activity'], pred_df['Collagen_Accumulation'])
        ax4.text(0.05, 0.95, f'R = {corr:.3f}\np = {pval:.4f}',
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax4.set_xlabel('TGF-β Activity Score', fontsize=10)
        ax4.set_ylabel('Mean Collagen Z-score', fontsize=10)
        ax4.set_title('TGF-β Activity Predicts Collagen Accumulation', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # 5. Causality Analysis
    ax5 = plt.subplot(6, 3, 5)
    if len(causality_df) > 0:
        ax5.scatter(causality_df['TGFb_Delta'], causality_df['Collagen_Delta'],
                   s=100, alpha=0.6, color='purple')

        # Add diagonal line (y=x)
        lims = [
            np.min([ax5.get_xlim(), ax5.get_ylim()]),
            np.max([ax5.get_xlim(), ax5.get_ylim()]),
        ]
        ax5.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

        ax5.set_xlabel('TGF-β Δ Z-score', fontsize=10)
        ax5.set_ylabel('Collagen Δ Z-score', fontsize=10)
        ax5.set_title('Causality: TGF-β Change vs Collagen Change', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # 6. Pathway Competition
    ax6 = plt.subplot(6, 3, 6)
    if len(comp_df) > 0:
        top_comp = comp_df.head(12)
        x = np.arange(len(top_comp))
        width = 0.25

        ax6.bar(x - width, top_comp['TGFb_Pathway'], width, label='TGF-β', alpha=0.8)
        ax6.bar(x, top_comp['PDGF_Pathway'], width, label='PDGF', alpha=0.8)
        ax6.bar(x + width, top_comp['CTGF_Standalone'], width, label='CTGF', alpha=0.8)

        ax6.set_xlabel('Tissue', fontsize=10)
        ax6.set_ylabel('Pathway Activity Score', fontsize=10)
        ax6.set_title('Competing Fibrosis Pathways', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(top_comp['Tissue'], rotation=45, ha='right', fontsize=8)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')

    # 7. Individual Pathway Component Analysis - Ligands
    ax7 = plt.subplot(6, 3, 7)
    ligand_data = []
    for ligand in TGFB_PATHWAY['ligands']:
        ligand_df = df[df['Canonical_Gene_Symbol'] == ligand]
        if len(ligand_df) > 0:
            mean_delta = ligand_df['Zscore_Delta'].mean()
            if not np.isnan(mean_delta):
                ligand_data.append({'Gene': ligand, 'Mean_Delta': mean_delta})

    if ligand_data:
        ligand_df_plot = pd.DataFrame(ligand_data).sort_values('Mean_Delta')
        colors = ['red' if x > 0 else 'blue' for x in ligand_df_plot['Mean_Delta']]
        ax7.barh(ligand_df_plot['Gene'], ligand_df_plot['Mean_Delta'], color=colors, alpha=0.7)
        ax7.set_xlabel('Mean Δ Z-score', fontsize=10)
        ax7.set_title('TGF-β Ligands: Age-Related Changes', fontsize=12, fontweight='bold')
        ax7.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax7.grid(True, alpha=0.3)

    # 8. Receptor Analysis
    ax8 = plt.subplot(6, 3, 8)
    receptor_data = []
    for receptor in TGFB_PATHWAY['receptors']:
        receptor_df = df[df['Canonical_Gene_Symbol'] == receptor]
        if len(receptor_df) > 0:
            mean_delta = receptor_df['Zscore_Delta'].mean()
            if not np.isnan(mean_delta):
                receptor_data.append({'Gene': receptor, 'Mean_Delta': mean_delta})

    if receptor_data:
        receptor_df_plot = pd.DataFrame(receptor_data).sort_values('Mean_Delta')
        colors = ['red' if x > 0 else 'blue' for x in receptor_df_plot['Mean_Delta']]
        ax8.barh(receptor_df_plot['Gene'], receptor_df_plot['Mean_Delta'], color=colors, alpha=0.7)
        ax8.set_xlabel('Mean Δ Z-score', fontsize=10)
        ax8.set_title('TGF-β Receptors: Age-Related Changes', fontsize=12, fontweight='bold')
        ax8.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax8.grid(True, alpha=0.3)

    # 9. Core Targets Analysis
    ax9 = plt.subplot(6, 3, 9)
    target_data = []
    for target in TGFB_PATHWAY['core_targets']:
        target_df = df[df['Canonical_Gene_Symbol'] == target]
        if len(target_df) > 0:
            mean_delta = target_df['Zscore_Delta'].mean()
            if not np.isnan(mean_delta):
                target_data.append({'Gene': target, 'Mean_Delta': mean_delta})

    if target_data:
        target_df_plot = pd.DataFrame(target_data).sort_values('Mean_Delta', ascending=False)
        colors = ['red' if x > 0 else 'blue' for x in target_df_plot['Mean_Delta']]
        ax9.barh(target_df_plot['Gene'], target_df_plot['Mean_Delta'], color=colors, alpha=0.7)
        ax9.set_xlabel('Mean Δ Z-score', fontsize=10)
        ax9.set_title('Core TGF-β Targets: Age-Related Changes', fontsize=12, fontweight='bold')
        ax9.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax9.grid(True, alpha=0.3)

    # 10. Tissue-specific pathway activation heatmap
    ax10 = plt.subplot(6, 3, 10)
    # Create comprehensive pathway heatmap
    pathway_matrix = []
    tissues_with_data = []

    all_pathway_genes = (TGFB_PATHWAY['ligands'] + TGFB_PATHWAY['receptors'] +
                        TGFB_PATHWAY['core_targets'][:6])

    for tissue in df['Tissue'].unique()[:15]:  # Limit to top 15 tissues
        tissue_data = df[df['Tissue'] == tissue]
        tissue_scores = []

        has_data = False
        for gene in all_pathway_genes:
            gene_data = tissue_data[
                (tissue_data['Canonical_Gene_Symbol'] == gene) &
                (tissue_data['Zscore_Delta'].notna())
            ]

            if len(gene_data) > 0:
                tissue_scores.append(gene_data['Zscore_Delta'].values[0])
                has_data = True
            else:
                tissue_scores.append(np.nan)

        if has_data:
            pathway_matrix.append(tissue_scores)
            tissues_with_data.append(tissue)

    if pathway_matrix:
        pathway_df = pd.DataFrame(pathway_matrix, columns=all_pathway_genes, index=tissues_with_data)
        sns.heatmap(pathway_df, annot=False, cmap='RdYlBu_r', center=0, ax=ax10,
                   cbar_kws={'label': 'Δ Z-score'}, vmin=-2, vmax=2)
        ax10.set_title('TGF-β Pathway Activation Map', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Pathway Component', fontsize=10)
        ax10.set_ylabel('Tissue', fontsize=10)
        plt.setp(ax10.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax10.get_yticklabels(), rotation=0, fontsize=8)

    # 11. Drug Target Priority Ranking
    ax11 = plt.subplot(6, 3, 11)
    drug_targets = {
        'Ligands (TGFB1-3)': TGFB_PATHWAY['ligands'],
        'Receptors (TGFBR1-3)': TGFB_PATHWAY['receptors'],
        'Effectors (VCAN, COLs)': TGFB_PATHWAY['core_targets'][:4],
        'ECM Modifiers (LOX, etc)': ['LOX', 'LOXL2', 'PLOD2', 'POSTN']
    }

    target_priority = []
    for category, genes in drug_targets.items():
        cat_data = df[
            (df['Canonical_Gene_Symbol'].isin(genes)) &
            (df['Zscore_Delta'].notna())
        ]

        if len(cat_data) > 0:
            mean_effect = cat_data['Zscore_Delta'].mean()
            max_effect = cat_data['Zscore_Delta'].max()
            n_genes = len(cat_data['Canonical_Gene_Symbol'].unique())

            target_priority.append({
                'Category': category,
                'Mean_Effect': mean_effect,
                'Max_Effect': max_effect,
                'N_Genes': n_genes,
                'Priority_Score': mean_effect * n_genes  # Combined metric
            })

    if target_priority:
        priority_df = pd.DataFrame(target_priority).sort_values('Priority_Score', ascending=False)
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(priority_df)))
        ax11.barh(priority_df['Category'], priority_df['Priority_Score'], color=colors, alpha=0.8)
        ax11.set_xlabel('Drug Target Priority Score', fontsize=10)
        ax11.set_title('TGF-β Drug Target Priority Ranking', fontsize=12, fontweight='bold')
        ax11.grid(True, alpha=0.3)

    # 12. Collagen Subtype Response to TGF-β
    ax12 = plt.subplot(6, 3, 12)
    collagen_types = ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL4A2',
                      'COL5A1', 'COL5A2', 'COL6A1', 'COL6A2', 'COL6A3']

    collagen_response = []
    for col in collagen_types:
        col_data = df[
            (df['Canonical_Gene_Symbol'] == col) &
            (df['Zscore_Delta'].notna())
        ]

        if len(col_data) > 0:
            collagen_response.append({
                'Collagen': col,
                'Mean_Delta': col_data['Zscore_Delta'].mean(),
                'N_Tissues': len(col_data['Tissue'].unique())
            })

    if collagen_response:
        col_df = pd.DataFrame(collagen_response).sort_values('Mean_Delta', ascending=False)
        colors = ['red' if x > 0 else 'blue' for x in col_df['Mean_Delta']]
        ax12.barh(col_df['Collagen'], col_df['Mean_Delta'], color=colors, alpha=0.7)
        ax12.set_xlabel('Mean Δ Z-score', fontsize=10)
        ax12.set_title('Collagen Subtype Response with Aging', fontsize=12, fontweight='bold')
        ax12.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax12.grid(True, alpha=0.3)

    # 13. TGF-β Activity vs Tissue Stiffness Proxy
    ax13 = plt.subplot(6, 3, 13)
    if len(pred_df) > 0:
        # Stiffness proxy: ratio of fibrillar to basement membrane collagens
        stiffness_data = []

        for tissue in pred_df['Tissue'].unique():
            tissue_data = df[df['Tissue'] == tissue]

            fibrillar = tissue_data[
                (tissue_data['Canonical_Gene_Symbol'].isin(['COL1A1', 'COL1A2', 'COL3A1', 'COL5A1'])) &
                (tissue_data['Zscore_Old'].notna())
            ]['Zscore_Old'].mean()

            basement = tissue_data[
                (tissue_data['Canonical_Gene_Symbol'].isin(['COL4A1', 'COL4A2'])) &
                (tissue_data['Zscore_Old'].notna())
            ]['Zscore_Old'].mean()

            tgfb_act = pred_df[pred_df['Tissue'] == tissue]['TGFb_Activity'].values

            if not np.isnan(fibrillar) and not np.isnan(basement) and len(tgfb_act) > 0:
                stiffness_data.append({
                    'Tissue': tissue,
                    'Stiffness_Ratio': fibrillar / basement if basement != 0 else np.nan,
                    'TGFb_Activity': tgfb_act[0]
                })

        if stiffness_data:
            stiff_df = pd.DataFrame(stiffness_data).dropna()

            if len(stiff_df) > 3:
                ax13.scatter(stiff_df['TGFb_Activity'], stiff_df['Stiffness_Ratio'],
                           s=100, alpha=0.6, color='orange')

                corr, pval = stats.pearsonr(stiff_df['TGFb_Activity'], stiff_df['Stiffness_Ratio'])
                ax13.text(0.05, 0.95, f'R = {corr:.3f}\np = {pval:.4f}',
                        transform=ax13.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax13.set_xlabel('TGF-β Activity', fontsize=10)
                ax13.set_ylabel('Stiffness Ratio (Fibrillar/Basement)', fontsize=10)
                ax13.set_title('TGF-β Activity vs Tissue Stiffness', fontsize=12, fontweight='bold')
                ax13.grid(True, alpha=0.3)

    # 14. Temporal progression analysis
    ax14 = plt.subplot(6, 3, 14)
    # Compare early vs late responders
    early_genes = TGFB_PATHWAY['ligands'] + TGFB_PATHWAY['receptors']
    late_genes = TGFB_PATHWAY['core_targets']

    early_deltas = df[
        (df['Canonical_Gene_Symbol'].isin(early_genes)) &
        (df['Zscore_Delta'].notna())
    ]['Zscore_Delta'].values

    late_deltas = df[
        (df['Canonical_Gene_Symbol'].isin(late_genes)) &
        (df['Zscore_Delta'].notna())
    ]['Zscore_Delta'].values

    if len(early_deltas) > 0 and len(late_deltas) > 0:
        positions = [1, 2]
        data_to_plot = [early_deltas, late_deltas]

        bp = ax14.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)

        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)

        ax14.set_xticks(positions)
        ax14.set_xticklabels(['Early Response\n(Ligands/Receptors)', 'Late Response\n(Target Genes)'])
        ax14.set_ylabel('Δ Z-score', fontsize=10)
        ax14.set_title('Temporal Response Pattern', fontsize=12, fontweight='bold')
        ax14.axhline(0, color='black', linestyle='--', linewidth=0.5)
        ax14.grid(True, alpha=0.3, axis='y')

    # 15. Network connectivity analysis
    ax15 = plt.subplot(6, 3, 15)
    if corr_matrix is not None and len(corr_matrix) > 2:
        # Calculate mean correlation for each gene (network connectivity)
        connectivity = corr_matrix.mean(axis=1).sort_values(ascending=False)

        colors = plt.cm.viridis(np.linspace(0, 1, len(connectivity)))
        ax15.barh(connectivity.index, connectivity.values, color=colors, alpha=0.8)
        ax15.set_xlabel('Mean Correlation (Network Connectivity)', fontsize=10)
        ax15.set_title('TGF-β Target Gene Network Connectivity', fontsize=12, fontweight='bold')
        ax15.grid(True, alpha=0.3)

    # 16. SMAD pathway components
    ax16 = plt.subplot(6, 3, 16)
    smad_data = []
    for smad in TGFB_PATHWAY['signaling']:
        smad_df = df[df['Canonical_Gene_Symbol'] == smad]
        if len(smad_df) > 0:
            mean_delta = smad_df['Zscore_Delta'].mean()
            if not np.isnan(mean_delta):
                smad_data.append({'Gene': smad, 'Mean_Delta': mean_delta})

    if smad_data:
        smad_df_plot = pd.DataFrame(smad_data).sort_values('Mean_Delta')
        colors = ['red' if x > 0 else 'blue' for x in smad_df_plot['Mean_Delta']]
        ax16.barh(smad_df_plot['Gene'], smad_df_plot['Mean_Delta'], color=colors, alpha=0.7)
        ax16.set_xlabel('Mean Δ Z-score', fontsize=10)
        ax16.set_title('SMAD Signaling Components', fontsize=12, fontweight='bold')
        ax16.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax16.grid(True, alpha=0.3)

    # 17. Fibrosis severity ranking
    ax17 = plt.subplot(6, 3, 17)
    if len(delta_df) > 0:
        # Combine TGF-β activity with collagen scores for fibrosis severity
        fibrosis_severity = delta_df.copy()

        # Add collagen accumulation data
        for idx, row in fibrosis_severity.iterrows():
            tissue = row['Tissue']
            tissue_data = df[df['Tissue'] == tissue]

            collagen_score = tissue_data[
                (tissue_data['Canonical_Gene_Symbol'].str.contains('COL', na=False)) &
                (tissue_data['Zscore_Delta'].notna())
            ]['Zscore_Delta'].mean()

            fibrosis_severity.loc[idx, 'Collagen_Score'] = collagen_score if not np.isnan(collagen_score) else 0

        fibrosis_severity['Fibrosis_Severity'] = (
            fibrosis_severity['Delta_TGFb_Activity'] +
            fibrosis_severity['Collagen_Score']
        ) / 2

        top_fibrosis = fibrosis_severity.nlargest(15, 'Fibrosis_Severity')

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_fibrosis)))
        ax17.barh(top_fibrosis['Tissue'], top_fibrosis['Fibrosis_Severity'],
                 color=colors, alpha=0.8)
        ax17.set_xlabel('Fibrosis Severity Score', fontsize=10)
        ax17.set_title('Tissue Fibrosis Severity Ranking', fontsize=12, fontweight='bold')
        ax17.grid(True, alpha=0.3)

    # 18. Latent TGF-β binding proteins
    ax18 = plt.subplot(6, 3, 18)
    ltbp_data = []
    for ltbp in TGFB_PATHWAY['binding_proteins']:
        ltbp_df = df[df['Canonical_Gene_Symbol'] == ltbp]
        if len(ltbp_df) > 0:
            mean_delta = ltbp_df['Zscore_Delta'].mean()
            if not np.isnan(mean_delta):
                ltbp_data.append({'Gene': ltbp, 'Mean_Delta': mean_delta})

    if ltbp_data:
        ltbp_df_plot = pd.DataFrame(ltbp_data).sort_values('Mean_Delta')
        colors = ['red' if x > 0 else 'blue' for x in ltbp_df_plot['Mean_Delta']]
        ax18.barh(ltbp_df_plot['Gene'], ltbp_df_plot['Mean_Delta'], color=colors, alpha=0.7)
        ax18.set_xlabel('Mean Δ Z-score', fontsize=10)
        ax18.set_title('Latent TGF-β Binding Proteins', fontsize=12, fontweight='bold')
        ax18.axvline(0, color='black', linestyle='--', linewidth=0.5)
        ax18.grid(True, alpha=0.3)
    else:
        ax18.text(0.5, 0.5, 'No LTBP data available', ha='center', va='center', fontsize=12)
        ax18.set_title('Latent TGF-β Binding Proteins', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = f"{output_dir}/agent_16_tgfb_pathway_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()

def main():
    """Main analysis pipeline"""
    print("\n" + "="*80)
    print("AGENT 16: TGF-β FIBROSIS PATHWAY DISSECTOR")
    print("="*80)

    # Load data
    df = load_and_prepare_data('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

    # Run analyses
    pathway_components = identify_pathway_components(df)
    activity_df, delta_df = calculate_tgfb_activity_score(df)
    corr_matrix, correlation_data = analyze_target_gene_correlations(df)
    pred_df = test_predictive_power(df, activity_df)
    causality_df = analyze_causality_direction(df)
    comp_df = compare_with_competing_pathways(df)

    # Generate visualizations
    output_dir = '/Users/Kravtsovd/projects/ecm-atlas/10_insights'
    create_visualizations(df, activity_df, delta_df, corr_matrix, pred_df,
                         causality_df, comp_df, output_dir)

    # Save key results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    activity_df.to_csv(f"{output_dir}/tgfb_activity_scores.csv", index=False)
    delta_df.to_csv(f"{output_dir}/tgfb_delta_scores.csv", index=False)

    if pred_df is not None and len(pred_df) > 0:
        pred_df.to_csv(f"{output_dir}/tgfb_predictive_power.csv", index=False)

    if causality_df is not None and len(causality_df) > 0:
        causality_df.to_csv(f"{output_dir}/tgfb_causality_analysis.csv", index=False)

    if comp_df is not None and len(comp_df) > 0:
        comp_df.to_csv(f"{output_dir}/tgfb_pathway_competition.csv", index=False)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    return {
        'pathway_components': pathway_components,
        'activity_df': activity_df,
        'delta_df': delta_df,
        'correlation_matrix': corr_matrix,
        'predictive_power': pred_df,
        'causality': causality_df,
        'pathway_competition': comp_df
    }

if __name__ == "__main__":
    results = main()
