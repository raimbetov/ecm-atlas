#!/usr/bin/env python3
"""
Tissue-Specific Aging Velocity Clocks Analysis
Agent: claude_code
Hypothesis: Different tissues age at different velocities (rates), not just patterns
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
DATASET_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_01/hypothesis_03_tissue_aging_clocks/claude_code')
VIZ_DIR = OUTPUT_DIR / 'visualizations_claude_code'
VIZ_DIR.mkdir(exist_ok=True, parents=True)

TSI_THRESHOLD = 3.0  # Tissue specificity threshold
N_BOOTSTRAP = 1000   # Bootstrap iterations for CI
N_TOP_MARKERS = 10   # Top markers per tissue

# Inflammatory protein keywords
INFLAMMATORY_KEYWORDS = [
    'IL', 'TNF', 'NFKB', 'CCL', 'CXCL', 'IFNG', 'IFNA', 'IFNB',
    'CSF', 'TGF', 'CRP', 'SAA', 'PTGS', 'COX', 'ALOX', 'LTB',
    'C3', 'C4', 'C5', 'CFB', 'CFD'  # Complement
]

# Functional categories
FUNCTIONAL_CATEGORIES = {
    'Structural': ['COL', 'LAMA', 'LAMB', 'LAMC', 'FBN', 'ELN'],
    'Regulatory': ['MMP', 'TIMP', 'ADAM', 'PLOD', 'P4H', 'LOX'],
    'Signaling': ['ITGA', 'ITGB', 'THBS', 'SPARC', 'TNC', 'FN1'],
    'Proteoglycan': ['VCAN', 'DCN', 'BGN', 'LUM', 'FMOD', 'ASPN', 'CHAD', 'KERA']
}

def load_dataset():
    """Load and validate dataset"""
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Validate required columns
    required_cols = ['Gene_Symbol', 'Zscore_Delta', 'Tissue_Compartment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Try alternative column names
        if 'Tissue' in df.columns:
            df['Tissue_Compartment'] = df['Tissue']

    # Use Tissue_Compartment for tissue identity
    df['Tissue'] = df['Tissue_Compartment']

    # Filter valid data
    df = df[df['Zscore_Delta'].notna()].copy()
    df = df[df['Gene_Symbol'].notna()].copy()

    tissues = df['Tissue'].unique()
    print(f"✓ Tissues identified: {len(tissues)}")
    print(f"  Sample tissues: {', '.join(list(tissues)[:5])}")

    # Sanity checks
    assert len(tissues) >= 5, f"Expected ≥5 tissues, found {len(tissues)}"
    assert 'Zscore_Delta' in df.columns, "Missing Zscore_Delta column!"

    print(f"✓ Data quality: {len(df)} valid protein-tissue observations")
    print()

    return df

def calculate_tsi(df):
    """
    Calculate Tissue Specificity Index (TSI) for all proteins
    TSI = (Max_tissue - Mean_others) / SD_others
    """
    print("=" * 80)
    print("CALCULATING TISSUE SPECIFICITY INDEX (TSI)")
    print("=" * 80)

    # Calculate mean |Zscore_Delta| per protein per tissue
    tissue_protein_summary = df.groupby(['Gene_Symbol', 'Tissue']).agg({
        'Zscore_Delta': lambda x: np.abs(x).mean()
    }).reset_index()
    tissue_protein_summary.columns = ['Gene_Symbol', 'Tissue', 'Mean_AbsZ']

    # Calculate TSI for each protein-tissue combination
    tsi_results = []

    for gene in tissue_protein_summary['Gene_Symbol'].unique():
        gene_data = tissue_protein_summary[tissue_protein_summary['Gene_Symbol'] == gene]

        for tissue in gene_data['Tissue'].unique():
            max_tissue_z = gene_data[gene_data['Tissue'] == tissue]['Mean_AbsZ'].values[0]
            other_tissues_z = gene_data[gene_data['Tissue'] != tissue]['Mean_AbsZ'].values

            if len(other_tissues_z) > 0:
                mean_others = np.mean(other_tissues_z)
                std_others = np.std(other_tissues_z)

                if std_others > 0:
                    tsi = (max_tissue_z - mean_others) / std_others
                else:
                    tsi = 0.0
            else:
                tsi = 0.0

            tsi_results.append({
                'Gene_Symbol': gene,
                'Tissue': tissue,
                'TSI': tsi,
                'Mean_AbsZ': max_tissue_z
            })

    tsi_df = pd.DataFrame(tsi_results)

    # Filter tissue-specific markers
    tissue_specific = tsi_df[tsi_df['TSI'] > TSI_THRESHOLD].copy()

    print(f"✓ Calculated TSI for {len(tsi_df)} protein-tissue combinations")
    print(f"✓ Tissue-specific markers (TSI > {TSI_THRESHOLD}): {len(tissue_specific)}")
    print(f"  TSI range: {tsi_df['TSI'].min():.2f} to {tsi_df['TSI'].max():.2f}")

    # Validate S4 markers
    s4_markers = ['S100A5', 'COL6A4', 'PLOD1']
    for marker in s4_markers:
        marker_data = tsi_df[tsi_df['Gene_Symbol'].str.upper().str.contains(marker, na=False)]
        if len(marker_data) > 0:
            max_tsi_row = marker_data.loc[marker_data['TSI'].idxmax()]
            print(f"  ✓ {marker}: TSI={max_tsi_row['TSI']:.2f} in {max_tsi_row['Tissue']}")
        else:
            print(f"  ✗ {marker}: Not found in dataset")

    print()
    return tsi_df, tissue_specific

def calculate_tissue_velocities(df, tissue_specific):
    """
    Calculate aging velocity for each tissue
    Velocity = Mean |Zscore_Delta| across tissue-specific markers
    """
    print("=" * 80)
    print("CALCULATING TISSUE AGING VELOCITIES")
    print("=" * 80)

    velocities = []

    for tissue in df['Tissue'].unique():
        # Get tissue-specific markers for this tissue
        tissue_markers = tissue_specific[tissue_specific['Tissue'] == tissue]['Gene_Symbol'].unique()

        if len(tissue_markers) == 0:
            # Fallback: use all proteins with high |Zscore_Delta| in this tissue
            tissue_data = df[df['Tissue'] == tissue].copy()
            tissue_data['AbsZ'] = tissue_data['Zscore_Delta'].abs()
            top_proteins = tissue_data.nlargest(20, 'AbsZ')['Gene_Symbol'].unique()
            tissue_markers = top_proteins

        # Get z-score data for these markers in this tissue
        marker_data = df[(df['Tissue'] == tissue) & (df['Gene_Symbol'].isin(tissue_markers))].copy()

        if len(marker_data) == 0:
            continue

        # Calculate metrics
        mean_abs_z = marker_data['Zscore_Delta'].abs().mean()
        upregulated = (marker_data['Zscore_Delta'] > 0).sum()
        downregulated = (marker_data['Zscore_Delta'] < 0).sum()
        total = len(marker_data)

        # Bootstrap confidence intervals
        bootstrap_means = []
        for _ in range(N_BOOTSTRAP):
            sample = marker_data['Zscore_Delta'].sample(n=len(marker_data), replace=True)
            bootstrap_means.append(sample.abs().mean())

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        velocities.append({
            'Tissue': tissue,
            'Mean_AbsZ': mean_abs_z,
            'Velocity': mean_abs_z,  # Proxy for velocity
            'N_Markers': len(tissue_markers),
            'N_Observations': total,
            'Upregulated_Pct': (upregulated / total * 100) if total > 0 else 0,
            'Downregulated_Pct': (downregulated / total * 100) if total > 0 else 0,
            'Bootstrap_CI_Lower': ci_lower,
            'Bootstrap_CI_Upper': ci_upper
        })

    velocity_df = pd.DataFrame(velocities)
    velocity_df = velocity_df.sort_values('Velocity', ascending=False)

    print(f"✓ Calculated velocities for {len(velocity_df)} tissues")
    print(f"\nTop 5 fastest-aging tissues:")
    for i, row in velocity_df.head().iterrows():
        print(f"  {i+1}. {row['Tissue']}: Velocity={row['Velocity']:.3f} "
              f"(95% CI: [{row['Bootstrap_CI_Lower']:.3f}, {row['Bootstrap_CI_Upper']:.3f}])")

    print()
    return velocity_df

def identify_top_markers(df, tsi_df):
    """Identify top tissue-specific markers per tissue"""
    print("=" * 80)
    print("IDENTIFYING TOP TISSUE-SPECIFIC MARKERS")
    print("=" * 80)

    top_markers_all = []

    for tissue in df['Tissue'].unique():
        tissue_tsi = tsi_df[tsi_df['Tissue'] == tissue].copy()
        top_markers = tissue_tsi.nlargest(N_TOP_MARKERS, 'TSI')

        # Add functional category
        for idx, row in top_markers.iterrows():
            gene = row['Gene_Symbol']
            category = 'Other'

            for cat, prefixes in FUNCTIONAL_CATEGORIES.items():
                if any(gene.upper().startswith(prefix) for prefix in prefixes):
                    category = cat
                    break

            # Get mean Zscore_Delta for this protein in this tissue
            protein_tissue_data = df[(df['Tissue'] == tissue) & (df['Gene_Symbol'] == gene)]
            mean_delta = protein_tissue_data['Zscore_Delta'].mean() if len(protein_tissue_data) > 0 else 0

            top_markers_all.append({
                'Gene_Symbol': gene,
                'Tissue': tissue,
                'TSI': row['TSI'],
                'Mean_Zscore_Delta': mean_delta,
                'Function_Category': category
            })

    markers_df = pd.DataFrame(top_markers_all)

    print(f"✓ Identified {len(markers_df)} top markers across tissues")

    # Show example for one tissue
    if len(markers_df) > 0:
        example_tissue = markers_df.iloc[0]['Tissue']
        example_markers = markers_df[markers_df['Tissue'] == example_tissue]
        print(f"\nExample - Top markers for {example_tissue}:")
        for i, row in example_markers.head(5).iterrows():
            print(f"  {row['Gene_Symbol']}: TSI={row['TSI']:.2f}, Δz={row['Mean_Zscore_Delta']:.3f}, {row['Function_Category']}")

    print()
    return markers_df

def analyze_fast_aging_mechanisms(df, velocity_df):
    """Analyze shared mechanisms in fast-aging tissues"""
    print("=" * 80)
    print("ANALYZING FAST-AGING TISSUE MECHANISMS")
    print("=" * 80)

    # Define fast-aging tissues (top 33%)
    n_fast = max(1, len(velocity_df) // 3)
    fast_tissues = velocity_df.head(n_fast)['Tissue'].values
    slow_tissues = velocity_df.tail(n_fast)['Tissue'].values

    print(f"Fast-aging tissues (top {n_fast}):")
    for tissue in fast_tissues:
        velocity = velocity_df[velocity_df['Tissue'] == tissue]['Velocity'].values[0]
        print(f"  - {tissue}: {velocity:.3f}")

    print(f"\nSlow-aging tissues (bottom {n_fast}):")
    for tissue in slow_tissues:
        velocity = velocity_df[velocity_df['Tissue'] == tissue]['Velocity'].values[0]
        print(f"  - {tissue}: {velocity:.3f}")

    # Find shared proteins in fast-aging tissues
    fast_tissue_proteins = {}
    for tissue in fast_tissues:
        tissue_data = df[df['Tissue'] == tissue].copy()
        # Get proteins with high |Zscore_Delta|
        tissue_data['AbsZ'] = tissue_data['Zscore_Delta'].abs()
        top_proteins = tissue_data.nlargest(50, 'AbsZ')
        fast_tissue_proteins[tissue] = set(top_proteins['Gene_Symbol'])

    # Count occurrences
    protein_counts = {}
    for tissue, proteins in fast_tissue_proteins.items():
        for protein in proteins:
            protein_counts[protein] = protein_counts.get(protein, 0) + 1

    # Shared proteins (appear in ≥2 fast tissues)
    shared_proteins = [p for p, count in protein_counts.items() if count >= 2]

    print(f"\n✓ Shared proteins in fast-aging tissues: {len(shared_proteins)}")

    # Classify by pathway
    mechanisms = []
    for protein in shared_proteins:
        # Get mean Zscore_Delta across fast tissues
        fast_data = df[(df['Gene_Symbol'] == protein) & (df['Tissue'].isin(fast_tissues))]
        mean_delta = fast_data['Zscore_Delta'].mean()
        n_tissues = len(fast_data['Tissue'].unique())

        # Classify pathway
        is_inflammatory = any(kw in protein.upper() for kw in INFLAMMATORY_KEYWORDS)

        pathway = 'Other'
        if 'COL' in protein.upper() or 'LAMA' in protein.upper():
            pathway = 'ECM_Structural'
        elif 'MMP' in protein.upper() or 'TIMP' in protein.upper():
            pathway = 'Matrix_Remodeling'
        elif is_inflammatory:
            pathway = 'Inflammation'
        elif any(protein.upper().startswith(kw) for kw in ['F2', 'F9', 'F10', 'F12', 'PLG', 'SERPIN']):
            pathway = 'Coagulation'

        mechanisms.append({
            'Gene_Symbol': protein,
            'N_Tissues_Present': n_tissues,
            'Mean_Zscore_Delta': mean_delta,
            'Pathway': pathway,
            'Is_Inflammatory': is_inflammatory
        })

    mechanisms_df = pd.DataFrame(mechanisms)

    # Pathway summary
    print("\nPathway distribution:")
    pathway_counts = mechanisms_df['Pathway'].value_counts()
    for pathway, count in pathway_counts.items():
        print(f"  {pathway}: {count}")

    # Test inflammation hypothesis
    print("\n" + "=" * 80)
    print("TESTING INFLAMMATION HYPOTHESIS")
    print("=" * 80)

    # Get inflammatory proteins
    df['Is_Inflammatory'] = df['Gene_Symbol'].apply(
        lambda x: any(kw in str(x).upper() for kw in INFLAMMATORY_KEYWORDS)
    )

    fast_inflam = df[(df['Tissue'].isin(fast_tissues)) & (df['Is_Inflammatory'])]['Zscore_Delta'].abs()
    slow_inflam = df[(df['Tissue'].isin(slow_tissues)) & (df['Is_Inflammatory'])]['Zscore_Delta'].abs()

    if len(fast_inflam) > 0 and len(slow_inflam) > 0:
        # Mann-Whitney U test
        statistic, pvalue = stats.mannwhitneyu(fast_inflam, slow_inflam, alternative='greater')

        print(f"Inflammatory signature comparison:")
        print(f"  Fast tissues: mean |Δz| = {fast_inflam.mean():.3f} (n={len(fast_inflam)})")
        print(f"  Slow tissues: mean |Δz| = {slow_inflam.mean():.3f} (n={len(slow_inflam)})")
        print(f"  Mann-Whitney U: p={pvalue:.4f} {'✓ SIGNIFICANT' if pvalue < 0.05 else '✗ not significant'}")

        # Effect size (rank-biserial correlation)
        effect_size = 1 - (2 * statistic) / (len(fast_inflam) * len(slow_inflam))
        print(f"  Effect size (rank-biserial): {effect_size:.3f}")
    else:
        print("✗ Insufficient inflammatory protein data for comparison")

    print()
    return mechanisms_df, fast_tissues, slow_tissues

def create_visualizations(velocity_df, markers_df, mechanisms_df, df, fast_tissues, slow_tissues):
    """Create all required visualizations"""
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    sns.set_style("whitegrid")

    # 1. Velocity Bar Chart
    fig, ax = plt.subplots(figsize=(12, 8))
    tissues = velocity_df['Tissue'].values
    velocities = velocity_df['Velocity'].values
    ci_lower = velocity_df['Bootstrap_CI_Lower'].values
    ci_upper = velocity_df['Bootstrap_CI_Upper'].values
    errors = [velocities - ci_lower, ci_upper - velocities]

    colors = ['#d62728' if t in fast_tissues else '#1f77b4' if t in slow_tissues else '#7f7f7f'
              for t in tissues]

    y_pos = np.arange(len(tissues))
    ax.barh(y_pos, velocities, xerr=errors, color=colors, alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tissues, fontsize=9)
    ax.set_xlabel('Aging Velocity (Mean |Zscore_Delta|)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tissue', fontsize=12, fontweight='bold')
    ax.set_title('Tissue Aging Velocity Ranking (Fastest → Slowest)\nwith 95% Bootstrap Confidence Intervals',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label='Fast-aging (top 33%)'),
        Patch(facecolor='#7f7f7f', alpha=0.7, label='Medium'),
        Patch(facecolor='#1f77b4', alpha=0.7, label='Slow-aging (bottom 33%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'velocity_ranking.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: velocity_ranking.png")
    plt.close()

    # 2. TSI Heatmap (Top markers per tissue)
    # Select top 5 markers per tissue for visualization
    heatmap_data = []
    tissues_for_heatmap = velocity_df.head(10)['Tissue'].values  # Top 10 tissues

    for tissue in tissues_for_heatmap:
        tissue_markers = markers_df[markers_df['Tissue'] == tissue].nlargest(5, 'TSI')
        for _, row in tissue_markers.iterrows():
            heatmap_data.append({
                'Tissue': tissue,
                'Gene': row['Gene_Symbol'],
                'Zscore_Delta': row['Mean_Zscore_Delta']
            })

    if len(heatmap_data) > 0:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='Tissue', columns='Gene', values='Zscore_Delta')

        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(heatmap_pivot, cmap='RdBu_r', center=0, cbar_kws={'label': 'Mean Zscore_Delta'},
                    linewidths=0.5, ax=ax, vmin=-2, vmax=2)
        ax.set_title('Tissue-Specific Markers Heatmap (Top 5 per Tissue)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Gene Symbol', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tissue', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'tsi_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: tsi_heatmap.png")
        plt.close()

    # 3. Inflammation Comparison
    df['Is_Inflammatory'] = df['Gene_Symbol'].apply(
        lambda x: any(kw in str(x).upper() for kw in INFLAMMATORY_KEYWORDS)
    )

    fast_inflam = df[(df['Tissue'].isin(fast_tissues)) & (df['Is_Inflammatory'])]['Zscore_Delta'].abs()
    slow_inflam = df[(df['Tissue'].isin(slow_tissues)) & (df['Is_Inflammatory'])]['Zscore_Delta'].abs()

    if len(fast_inflam) > 0 and len(slow_inflam) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        data_for_plot = pd.DataFrame({
            'Tissue_Group': ['Fast-aging'] * len(fast_inflam) + ['Slow-aging'] * len(slow_inflam),
            'Abs_Zscore_Delta': list(fast_inflam) + list(slow_inflam)
        })

        sns.boxplot(data=data_for_plot, x='Tissue_Group', y='Abs_Zscore_Delta',
                    palette=['#d62728', '#1f77b4'], ax=ax)
        sns.swarmplot(data=data_for_plot, x='Tissue_Group', y='Abs_Zscore_Delta',
                      color='black', alpha=0.3, size=3, ax=ax)

        ax.set_xlabel('Tissue Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('|Zscore_Delta| of Inflammatory Proteins', fontsize=12, fontweight='bold')
        ax.set_title('Inflammatory Signature: Fast vs Slow-Aging Tissues',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics
        statistic, pvalue = stats.mannwhitneyu(fast_inflam, slow_inflam, alternative='greater')
        ax.text(0.5, 0.95, f'Mann-Whitney U: p={pvalue:.4f}',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'inflammation_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: inflammation_comparison.png")
        plt.close()

    # 4. Pathway distribution in fast-aging tissues
    if len(mechanisms_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        pathway_counts = mechanisms_df['Pathway'].value_counts()

        colors_pathway = sns.color_palette('Set2', len(pathway_counts))
        ax.barh(pathway_counts.index, pathway_counts.values, color=colors_pathway, alpha=0.8)
        ax.set_xlabel('Number of Shared Proteins', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pathway', fontsize=12, fontweight='bold')
        ax.set_title('Shared Pathways in Fast-Aging Tissues', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'pathway_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: pathway_distribution.png")
        plt.close()

    print()

def save_outputs(velocity_df, markers_df, mechanisms_df):
    """Save all CSV outputs"""
    print("=" * 80)
    print("SAVING OUTPUT FILES")
    print("=" * 80)

    # Save velocity data
    velocity_output = OUTPUT_DIR / 'tissue_aging_velocity_claude_code.csv'
    velocity_df.to_csv(velocity_output, index=False)
    print(f"✓ Saved: tissue_aging_velocity_claude_code.csv ({len(velocity_df)} tissues)")

    # Save markers
    markers_output = OUTPUT_DIR / 'tissue_specific_markers_claude_code.csv'
    markers_df.to_csv(markers_output, index=False)
    print(f"✓ Saved: tissue_specific_markers_claude_code.csv ({len(markers_df)} markers)")

    # Save mechanisms
    mechanisms_output = OUTPUT_DIR / 'fast_aging_mechanisms_claude_code.csv'
    mechanisms_df.to_csv(mechanisms_output, index=False)
    print(f"✓ Saved: fast_aging_mechanisms_claude_code.csv ({len(mechanisms_df)} proteins)")

    print()

def main():
    """Main analysis pipeline"""
    print("\n" + "=" * 80)
    print("TISSUE-SPECIFIC AGING VELOCITY CLOCKS ANALYSIS")
    print("Agent: claude_code")
    print("=" * 80 + "\n")

    # Load data
    df = load_dataset()

    # Calculate TSI
    tsi_df, tissue_specific = calculate_tsi(df)

    # Calculate velocities
    velocity_df = calculate_tissue_velocities(df, tissue_specific)

    # Identify top markers
    markers_df = identify_top_markers(df, tsi_df)

    # Analyze mechanisms
    mechanisms_df, fast_tissues, slow_tissues = analyze_fast_aging_mechanisms(df, velocity_df)

    # Create visualizations
    create_visualizations(velocity_df, markers_df, mechanisms_df, df, fast_tissues, slow_tissues)

    # Save outputs
    save_outputs(velocity_df, markers_df, mechanisms_df)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VIZ_DIR}")
    print("\nNext steps:")
    print("1. Review visualizations in visualizations_claude_code/")
    print("2. Examine tissue_aging_velocity_claude_code.csv for velocity rankings")
    print("3. Check fast_aging_mechanisms_claude_code.csv for shared pathways")
    print("4. Generate final report: 90_results_claude_code.md")
    print()

if __name__ == '__main__':
    main()
