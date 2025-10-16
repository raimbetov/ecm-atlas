#!/usr/bin/env python3
"""
AGENT 14: FRZB/WNT Signaling Dysregulation Analysis

Validates hypothesis: FRZB loss → unleashed Wnt → cartilage breakdown → aging
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Configuration
DATA_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
OUTPUT_PATH = Path("/Users/Kravtsovd/projects/ecm-atlas/10_insights")
OUTPUT_PATH.mkdir(exist_ok=True)

# WNT Pathway Components (comprehensive)
WNT_PATHWAY = {
    'Wnt_Inhibitors': [
        'FRZB',      # Frizzled-related protein B (main target)
        'DKK1', 'DKK2', 'DKK3', 'DKK4',  # Dickkopf family
        'SFRP1', 'SFRP2', 'SFRP3', 'SFRP4', 'SFRP5',  # Secreted frizzled-related proteins
        'WIF1',      # WNT inhibitory factor
        'APCDD1',    # APC down-regulated 1
    ],
    'Wnt_Activators': [
        'WNT1', 'WNT2', 'WNT2B', 'WNT3', 'WNT3A', 'WNT4', 'WNT5A', 'WNT5B',
        'WNT6', 'WNT7A', 'WNT7B', 'WNT8A', 'WNT8B', 'WNT9A', 'WNT9B',
        'WNT10A', 'WNT10B', 'WNT11', 'WNT16',
        'RSPO1', 'RSPO2', 'RSPO3', 'RSPO4',  # R-spondins (Wnt enhancers)
        'WISP1', 'WISP2', 'WISP3',  # WNT1-inducible signaling proteins (CCN family)
    ],
    'Cartilage_Structural': [
        'COL2A1',    # Type II collagen (main cartilage collagen)
        'ACAN',      # Aggrecan (main proteoglycan)
        'MATN3',     # Matrilin-3 (cartilage matrix protein)
        'MATN1', 'MATN2', 'MATN4',  # Other matrilins
        'COMP',      # Cartilage oligomeric matrix protein
        'CILP',      # Cartilage intermediate layer protein
        'CHAD',      # Chondroadherin
        'PRG4',      # Proteoglycan 4 (lubricin)
    ]
}

def load_data():
    """Load and prepare ECM aging dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total records: {len(df):,}")
    print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique():,}")
    print(f"Tissues: {df['Tissue'].nunique()}")
    return df

def extract_wnt_proteins(df):
    """Extract all WNT pathway proteins from dataset"""
    print("\n=== WNT PATHWAY PROTEIN DISCOVERY ===")

    results = {}
    for category, genes in WNT_PATHWAY.items():
        found = df[df['Canonical_Gene_Symbol'].isin(genes)]
        results[category] = found

        print(f"\n{category}:")
        print(f"  Searched: {len(genes)} genes")
        print(f"  Found: {found['Canonical_Gene_Symbol'].nunique()} unique genes")
        if len(found) > 0:
            print(f"  Data points: {len(found):,}")
            print(f"  Genes: {sorted(found['Canonical_Gene_Symbol'].unique())}")

    return results

def frzb_tissue_analysis(df):
    """Analyze FRZB loss across tissues - is it disc-specific or universal?"""
    print("\n=== FRZB TISSUE-SPECIFIC ANALYSIS ===")

    frzb = df[df['Canonical_Gene_Symbol'] == 'FRZB'].copy()

    if len(frzb) == 0:
        print("WARNING: FRZB not found in dataset!")
        return None

    print(f"Total FRZB measurements: {len(frzb)}")
    print(f"Tissues with FRZB: {frzb['Tissue'].unique()}")

    # Calculate tissue-specific trajectories
    tissue_stats = []
    for tissue in frzb['Tissue'].unique():
        tissue_data = frzb[frzb['Tissue'] == tissue]

        # Z-score delta (old - young)
        zscore_delta = tissue_data['Zscore_Delta'].mean()

        tissue_stats.append({
            'Tissue': tissue,
            'N_measurements': len(tissue_data),
            'Zscore_Delta_mean': zscore_delta,
            'Zscore_Old_mean': tissue_data['Zscore_Old'].mean(),
            'Zscore_Young_mean': tissue_data['Zscore_Young'].mean(),
            'FRZB_Loss': zscore_delta < 0  # True if FRZB decreases with age
        })

    frzb_summary = pd.DataFrame(tissue_stats).sort_values('Zscore_Delta_mean')
    print("\nFRZB Loss by Tissue (most negative = greatest loss):")
    print(frzb_summary.to_string(index=False))

    return frzb_summary

def calculate_wnt_balance(df, wnt_proteins):
    """Calculate Wnt Unleashing Index = Activators / Inhibitors"""
    print("\n=== WNT BALANCE CALCULATION ===")

    inhibitors = wnt_proteins['Wnt_Inhibitors']
    activators = wnt_proteins['Wnt_Activators']

    # Group by tissue and age condition
    balance_results = []

    for tissue in df['Tissue'].unique():
        tissue_df = df[df['Tissue'] == tissue]

        # Young samples
        young_inhib = inhibitors[(inhibitors['Tissue'] == tissue)]['Zscore_Young'].dropna()
        young_activ = activators[(activators['Tissue'] == tissue)]['Zscore_Young'].dropna()

        # Old samples
        old_inhib = inhibitors[(inhibitors['Tissue'] == tissue)]['Zscore_Old'].dropna()
        old_activ = activators[(activators['Tissue'] == tissue)]['Zscore_Old'].dropna()

        if len(young_inhib) > 0 and len(old_inhib) > 0:
            balance_results.append({
                'Tissue': tissue,
                'Young_Inhibitor_Zscore': young_inhib.mean(),
                'Old_Inhibitor_Zscore': old_inhib.mean(),
                'Young_Activator_Zscore': young_activ.mean() if len(young_activ) > 0 else 0,
                'Old_Activator_Zscore': old_activ.mean() if len(old_activ) > 0 else 0,
                'Young_Balance': (young_activ.mean() / abs(young_inhib.mean())) if len(young_activ) > 0 and len(young_inhib) > 0 and young_inhib.mean() != 0 else np.nan,
                'Old_Balance': (old_activ.mean() / abs(old_inhib.mean())) if len(old_activ) > 0 and len(old_inhib) > 0 and old_inhib.mean() != 0 else np.nan,
                'N_Inhibitors_Young': len(young_inhib),
                'N_Inhibitors_Old': len(old_inhib),
                'N_Activators_Young': len(young_activ),
                'N_Activators_Old': len(old_activ),
            })

    balance_df = pd.DataFrame(balance_results)
    balance_df['Wnt_Unleashing_Index'] = balance_df['Old_Balance'] - balance_df['Young_Balance']

    print("\nWnt Balance by Tissue:")
    print(balance_df[['Tissue', 'Young_Balance', 'Old_Balance', 'Wnt_Unleashing_Index']].to_string(index=False))

    return balance_df

def cartilage_protein_correlation(df, wnt_proteins):
    """Test correlation between FRZB and cartilage structural proteins"""
    print("\n=== FRZB vs CARTILAGE PROTEIN CORRELATION ===")

    frzb = df[df['Canonical_Gene_Symbol'] == 'FRZB']
    cartilage = wnt_proteins['Cartilage_Structural']

    if len(frzb) == 0:
        print("WARNING: FRZB not found - cannot calculate correlations")
        return None

    # Merge on common tissues
    correlations = []

    for cart_gene in cartilage['Canonical_Gene_Symbol'].unique():
        cart_data = df[df['Canonical_Gene_Symbol'] == cart_gene]

        # Find overlapping tissues
        common_tissues = set(frzb['Tissue'].unique()) & set(cart_data['Tissue'].unique())

        if len(common_tissues) > 0:
            frzb_values = []
            cart_values = []

            for tissue in common_tissues:
                frzb_tissue = frzb[frzb['Tissue'] == tissue]['Zscore_Delta'].values
                cart_tissue = cart_data[cart_data['Tissue'] == tissue]['Zscore_Delta'].values

                if len(frzb_tissue) > 0 and len(cart_tissue) > 0:
                    frzb_values.append(frzb_tissue[0])
                    cart_values.append(cart_tissue[0])

            if len(frzb_values) >= 2:  # Need at least 2 points for correlation
                corr, pval = stats.pearsonr(frzb_values, cart_values)
                correlations.append({
                    'Cartilage_Protein': cart_gene,
                    'N_Tissues': len(frzb_values),
                    'Pearson_R': corr,
                    'P_value': pval,
                    'Significant': pval < 0.05
                })

    if len(correlations) > 0:
        corr_df = pd.DataFrame(correlations).sort_values('Pearson_R')
        print("\nFRZB-Cartilage Protein Correlations:")
        print(corr_df.to_string(index=False))
        return corr_df
    else:
        print("No overlapping tissues found for correlation analysis")
        return None

def gene_therapy_priority(wnt_proteins, frzb_summary):
    """Rank WNT antagonists by therapeutic potential"""
    print("\n=== GENE THERAPY PRIORITY RANKING ===")

    inhibitors = wnt_proteins['Wnt_Inhibitors']

    # Calculate metrics for each inhibitor
    priority_scores = []

    for gene in inhibitors['Canonical_Gene_Symbol'].unique():
        gene_data = inhibitors[inhibitors['Canonical_Gene_Symbol'] == gene]

        # Metrics
        mean_loss = gene_data['Zscore_Delta'].mean()  # More negative = more lost
        n_tissues = gene_data['Tissue'].nunique()
        consistency = gene_data['Zscore_Delta'].std()  # Lower = more consistent

        # Priority score (higher = better target)
        # Prioritize: large loss, present in many tissues, consistent
        priority = abs(mean_loss) * n_tissues / (consistency + 1)

        priority_scores.append({
            'Gene': gene,
            'Mean_Zscore_Delta': mean_loss,
            'N_Tissues': n_tissues,
            'Consistency_SD': consistency,
            'Priority_Score': priority,
            'Therapeutic_Rationale': 'High loss, broad expression' if priority > 1 else 'Limited potential'
        })

    priority_df = pd.DataFrame(priority_scores).sort_values('Priority_Score', ascending=False)

    print("\nGene Therapy Target Ranking:")
    print(priority_df.to_string(index=False))

    return priority_df

def temporal_precedence_test(df):
    """Test if FRZB loss precedes or follows cartilage degradation"""
    print("\n=== TEMPORAL PRECEDENCE TEST ===")
    print("Note: Limited by cross-sectional data - need longitudinal studies for definitive causality")

    # Proxy test: Check if FRZB changes are larger/earlier than cartilage proteins
    # If FRZB loss is a CAUSE, we expect:
    # 1. FRZB delta to be more negative than cartilage proteins
    # 2. FRZB loss to be more consistent across tissues

    frzb = df[df['Canonical_Gene_Symbol'] == 'FRZB']
    cartilage_genes = WNT_PATHWAY['Cartilage_Structural']
    cartilage = df[df['Canonical_Gene_Symbol'].isin(cartilage_genes)]

    if len(frzb) == 0:
        print("FRZB not found - cannot perform test")
        return None

    frzb_delta = frzb['Zscore_Delta'].mean()
    cart_delta = cartilage.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean()

    print(f"\nFRZB mean Z-score delta: {frzb_delta:.3f}")
    print(f"Cartilage proteins mean Z-score delta: {cart_delta.mean():.3f}")
    print(f"\nInterpretation:")
    if frzb_delta < cart_delta.mean():
        print("  ✓ FRZB loss is GREATER than cartilage protein loss")
        print("  ✓ Consistent with FRZB as upstream driver")
    else:
        print("  ✗ FRZB loss is SMALLER than cartilage protein loss")
        print("  ✗ May be downstream consequence, not driver")

    return {'FRZB_delta': frzb_delta, 'Cartilage_delta_mean': cart_delta.mean()}

def create_visualizations(df, wnt_proteins, output_path):
    """Generate comprehensive visualization suite"""
    print("\n=== GENERATING VISUALIZATIONS ===")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FRZB/WNT Signaling Dysregulation in ECM Aging', fontsize=16, fontweight='bold')

    # Plot 1: FRZB trajectory across tissues
    ax1 = axes[0, 0]
    frzb = df[df['Canonical_Gene_Symbol'] == 'FRZB']
    if len(frzb) > 0:
        tissue_order = frzb.groupby('Tissue')['Zscore_Delta'].mean().sort_values().index
        sns.boxplot(data=frzb, x='Tissue', y='Zscore_Delta', order=tissue_order, ax=ax1, palette='coolwarm')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('FRZB Loss by Tissue', fontweight='bold')
        ax1.set_ylabel('Z-score Delta (Old - Young)')
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, 'FRZB not found in dataset', ha='center', va='center')

    # Plot 2: WNT pathway balance
    ax2 = axes[0, 1]
    inhibitors = wnt_proteins['Wnt_Inhibitors']
    activators = wnt_proteins['Wnt_Activators']

    pathway_deltas = []
    for category, data in [('Inhibitors', inhibitors), ('Activators', activators)]:
        if len(data) > 0:
            for tissue in data['Tissue'].unique():
                tissue_data = data[data['Tissue'] == tissue]
                pathway_deltas.append({
                    'Tissue': tissue,
                    'Category': category,
                    'Zscore_Delta': tissue_data['Zscore_Delta'].mean()
                })

    if pathway_deltas:
        pathway_df = pd.DataFrame(pathway_deltas)
        sns.barplot(data=pathway_df, x='Tissue', y='Zscore_Delta', hue='Category', ax=ax2, palette=['#1f77b4', '#ff7f0e'])
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('WNT Pathway Balance Change', fontweight='bold')
        ax2.set_ylabel('Mean Z-score Delta')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Component')

    # Plot 3: Cartilage protein changes
    ax3 = axes[1, 0]
    cartilage = wnt_proteins['Cartilage_Structural']
    if len(cartilage) > 0:
        cart_summary = cartilage.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean().sort_values()
        cart_summary.plot(kind='barh', ax=ax3, color='steelblue')
        ax3.axvline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_title('Cartilage Structural Protein Changes', fontweight='bold')
        ax3.set_xlabel('Mean Z-score Delta (Old - Young)')

    # Plot 4: FRZB vs Cartilage correlation heatmap
    ax4 = axes[1, 1]
    frzb = df[df['Canonical_Gene_Symbol'] == 'FRZB']
    cartilage_full = df[df['Canonical_Gene_Symbol'].isin(WNT_PATHWAY['Cartilage_Structural'])]

    if len(frzb) > 0 and len(cartilage_full) > 0:
        # Build correlation matrix
        corr_matrix = []
        tissues = sorted(set(frzb['Tissue'].unique()) & set(cartilage_full['Tissue'].unique()))
        cart_genes = sorted(cartilage_full['Canonical_Gene_Symbol'].unique())

        matrix_data = np.full((len(cart_genes), len(tissues)), np.nan)

        for i, gene in enumerate(cart_genes):
            for j, tissue in enumerate(tissues):
                frzb_val = frzb[frzb['Tissue'] == tissue]['Zscore_Delta'].values
                cart_val = cartilage_full[(cartilage_full['Canonical_Gene_Symbol'] == gene) &
                                          (cartilage_full['Tissue'] == tissue)]['Zscore_Delta'].values
                if len(frzb_val) > 0 and len(cart_val) > 0:
                    matrix_data[i, j] = frzb_val[0] * cart_val[0]  # Product indicates co-directional change

        if not np.all(np.isnan(matrix_data)):
            sns.heatmap(matrix_data, xticklabels=tissues, yticklabels=cart_genes,
                       cmap='RdBu_r', center=0, ax=ax4, cbar_kws={'label': 'FRZB × Cartilage Product'})
            ax4.set_title('FRZB-Cartilage Co-regulation', fontweight='bold')

    plt.tight_layout()

    viz_path = output_path / "agent_14_frzb_wnt_visualizations.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {viz_path}")
    plt.close()

def main():
    """Execute FRZB/WNT analysis pipeline"""
    print("=" * 80)
    print("AGENT 14: FRZB/WNT SIGNALING DYSREGULATION SPECIALIST")
    print("=" * 80)

    # Load data
    df = load_data()

    # Extract WNT pathway proteins
    wnt_proteins = extract_wnt_proteins(df)

    # Analysis modules
    frzb_summary = frzb_tissue_analysis(df)
    wnt_balance = calculate_wnt_balance(df, wnt_proteins)
    correlations = cartilage_protein_correlation(df, wnt_proteins)
    priority_ranking = gene_therapy_priority(wnt_proteins, frzb_summary)
    temporal_test = temporal_precedence_test(df)

    # Visualizations
    create_visualizations(df, wnt_proteins, OUTPUT_PATH)

    # Save results
    results = {
        'frzb_summary': frzb_summary,
        'wnt_balance': wnt_balance,
        'correlations': correlations,
        'priority_ranking': priority_ranking,
        'temporal_test': temporal_test
    }

    # Export to CSV
    for name, data in results.items():
        if data is not None and isinstance(data, pd.DataFrame):
            output_file = OUTPUT_PATH / f"agent_14_{name}.csv"
            data.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - Results saved to 10_insights/")
    print("=" * 80)

    return results

if __name__ == "__main__":
    main()
