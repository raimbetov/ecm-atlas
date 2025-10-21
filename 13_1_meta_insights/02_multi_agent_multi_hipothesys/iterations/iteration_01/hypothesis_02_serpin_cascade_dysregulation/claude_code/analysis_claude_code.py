#!/usr/bin/env python3
"""
Serpin Cascade Dysregulation Analysis
Agent: claude_code
Hypothesis: Serpin family dysregulation is the central unifying mechanism of ECM aging
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
import networkx as nx
from matplotlib_venn import venn3
import warnings
import re
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_01/hypothesis_02_serpin_cascade_dysregulation/claude_code/'
VIZ_DIR = OUTPUT_DIR + 'visualizations_claude_code/'


def load_data():
    """Load and validate dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def identify_serpins(df):
    """Identify all serpin family members"""
    print("\nIdentifying serpin family members...")

    # Pattern: SERPIN* or A2M or PZP (case-insensitive)
    serpin_pattern = re.compile(r'SERPIN|A2M|PZP', re.IGNORECASE)

    # Check both Gene_Symbol and Canonical_Gene_Symbol
    gene_cols = ['Gene_Symbol', 'Canonical_Gene_Symbol']
    serpin_mask = pd.Series(False, index=df.index)

    for col in gene_cols:
        if col in df.columns:
            serpin_mask |= df[col].astype(str).str.contains(serpin_pattern, na=False)

    serpin_df = df[serpin_mask].copy()
    unique_serpins = serpin_df['Gene_Symbol'].unique()

    print(f"✓ Serpins identified: {len(unique_serpins)}")
    print(f"  Examples: {list(unique_serpins)[:10]}")

    # Sanity checks
    assert len(unique_serpins) >= 10, f"Expected ≥10 serpins, found {len(unique_serpins)}"

    serpin_list_lower = [s.lower() for s in unique_serpins]
    assert any('pzp' in s for s in serpin_list_lower), "Missing PZP!"
    assert any('serpinc1' in s for s in serpin_list_lower), "Missing SERPINC1!"

    print("✓ Sanity checks passed")

    return serpin_df, unique_serpins


def comprehensive_serpin_profiling(df, serpin_df, unique_serpins):
    """Criterion 1: Comprehensive serpin profiling (40 pts)"""
    print("\n" + "="*80)
    print("CRITERION 1: Comprehensive Serpin Profiling (40 pts)")
    print("="*80)

    profiles = []

    for serpin in unique_serpins:
        serpin_data = serpin_df[serpin_df['Gene_Symbol'] == serpin]

        # Mean Zscore_Delta
        mean_zscore = serpin_data['Zscore_Delta'].mean()

        # Tissue breadth
        tissue_breadth = serpin_data['Tissue_Compartment'].nunique()

        # Directional consistency
        sign_of_mean = np.sign(mean_zscore)
        if sign_of_mean != 0:
            same_sign = (np.sign(serpin_data['Zscore_Delta']) == sign_of_mean).sum()
            directional_consistency = (same_sign / len(serpin_data)) * 100
        else:
            directional_consistency = 0

        # Study count
        study_count = serpin_data['Study_ID'].nunique()

        # Functional classification (based on known biology)
        functional_category = classify_serpin(serpin)

        profiles.append({
            'Gene_Symbol': serpin,
            'Mean_Zscore_Delta': mean_zscore,
            'Abs_Mean_Zscore_Delta': abs(mean_zscore),
            'Tissue_Breadth': tissue_breadth,
            'Directional_Consistency_Pct': directional_consistency,
            'Study_Count': study_count,
            'Functional_Category': functional_category,
            'N_Measurements': len(serpin_data)
        })

    profile_df = pd.DataFrame(profiles)
    profile_df = profile_df.sort_values('Abs_Mean_Zscore_Delta', ascending=False)

    # Save
    output_path = OUTPUT_DIR + 'serpin_comprehensive_profile_claude_code.csv'
    profile_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")

    # Statistical test: Serpins vs non-serpins
    print("\nStatistical Comparison: Serpins vs Non-Serpins")

    # Get absolute zscore for all proteins
    all_proteins = df.groupby('Gene_Symbol')['Zscore_Delta'].apply(
        lambda x: abs(x.mean())
    ).reset_index()
    all_proteins.columns = ['Gene_Symbol', 'Abs_Mean_Zscore_Delta']

    serpin_pattern = re.compile(r'SERPIN|A2M|PZP', re.IGNORECASE)
    all_proteins['Is_Serpin'] = all_proteins['Gene_Symbol'].str.contains(
        serpin_pattern, na=False
    )

    serpin_scores = all_proteins[all_proteins['Is_Serpin']]['Abs_Mean_Zscore_Delta']
    non_serpin_scores = all_proteins[~all_proteins['Is_Serpin']]['Abs_Mean_Zscore_Delta']

    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(serpin_scores, non_serpin_scores, alternative='greater')

    # Effect size (rank-biserial correlation)
    n1, n2 = len(serpin_scores), len(non_serpin_scores)
    rank_biserial = 1 - (2*u_stat) / (n1 * n2)

    print(f"  Serpin median |Δz|: {serpin_scores.median():.4f}")
    print(f"  Non-serpin median |Δz|: {non_serpin_scores.median():.4f}")
    print(f"  Mann-Whitney U: {u_stat:.1f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Effect size (rank-biserial): {rank_biserial:.4f}")

    # Create boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    data_for_plot = pd.DataFrame({
        'Group': ['Serpins']*len(serpin_scores) + ['Non-Serpins']*len(non_serpin_scores),
        'Abs_Zscore_Delta': list(serpin_scores) + list(non_serpin_scores)
    })

    sns.boxplot(data=data_for_plot, x='Group', y='Abs_Zscore_Delta', ax=ax, palette=['red', 'gray'])
    ax.set_ylabel('|Mean Zscore Delta|', fontsize=12)
    ax.set_xlabel('')
    ax.set_title('Serpin vs Non-Serpin Dysregulation\n' +
                 f'Mann-Whitney U p={p_value:.2e}, effect size={rank_biserial:.3f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR + 'serpin_vs_nonserpin_boxplot_claude_code.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {VIZ_DIR}serpin_vs_nonserpin_boxplot_claude_code.png")

    return profile_df, {'u_stat': u_stat, 'p_value': p_value, 'effect_size': rank_biserial}


def classify_serpin(serpin_name):
    """Classify serpin by functional category"""
    name_lower = serpin_name.lower()

    if any(x in name_lower for x in ['serpinc1', 'serpinf2']):
        return 'Coagulation'
    elif any(x in name_lower for x in ['serpinb2', 'serpine1', 'serpine2']):
        return 'Fibrinolysis'
    elif 'serping1' in name_lower:
        return 'Complement'
    elif 'serpinh1' in name_lower:
        return 'ECM Assembly'
    elif any(x in name_lower for x in ['serpina3', 'serpina1']):
        return 'Inflammation'
    elif any(x in name_lower for x in ['a2m', 'pzp']):
        return 'Broad Inhibitor'
    else:
        return 'Other'


def build_correlation_network(df, unique_serpins):
    """Criterion 2: Network centrality analysis (30 pts)"""
    print("\n" + "="*80)
    print("CRITERION 2: Network Centrality Analysis (30 pts)")
    print("="*80)

    # Create protein-tissue matrix for correlation
    print("Building protein-tissue expression matrix...")

    # Pivot: proteins x tissues, values = Zscore_Delta
    pivot_df = df.pivot_table(
        index='Gene_Symbol',
        columns='Tissue_Compartment',
        values='Zscore_Delta',
        aggfunc='mean'
    )

    # Filter proteins present in ≥3 tissues
    min_tissues = 3
    protein_counts = pivot_df.notna().sum(axis=1)
    pivot_df_filtered = pivot_df[protein_counts >= min_tissues]

    print(f"  Proteins with ≥{min_tissues} tissues: {len(pivot_df_filtered)}")

    # Calculate pairwise Spearman correlations
    print("Calculating pairwise correlations (this may take a few minutes)...")

    proteins = pivot_df_filtered.index.tolist()
    n_proteins = len(proteins)

    # Initialize network
    G = nx.Graph()
    G.add_nodes_from(proteins)

    # Correlation threshold
    corr_threshold = 0.5
    p_threshold = 0.05

    edge_count = 0

    for i in range(n_proteins):
        for j in range(i+1, n_proteins):
            protein1 = proteins[i]
            protein2 = proteins[j]

            # Get expression vectors
            vec1 = pivot_df_filtered.loc[protein1].values
            vec2 = pivot_df_filtered.loc[protein2].values

            # Find common tissues (non-NaN in both)
            valid_mask = ~np.isnan(vec1) & ~np.isnan(vec2)

            if valid_mask.sum() >= 3:  # Need ≥3 common tissues
                vec1_clean = vec1[valid_mask]
                vec2_clean = vec2[valid_mask]

                # Spearman correlation
                rho, p_val = spearmanr(vec1_clean, vec2_clean)

                if abs(rho) >= corr_threshold and p_val < p_threshold:
                    G.add_edge(protein1, protein2, weight=abs(rho), rho=rho, pval=p_val)
                    edge_count += 1

    print(f"✓ Network built: {len(G.nodes())} nodes, {edge_count} edges")

    # Calculate centrality metrics
    print("Calculating centrality metrics...")

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Eigenvector centrality (may fail if graph is not connected)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("  Warning: Eigenvector centrality failed (disconnected graph), using PageRank instead")
        eigenvector_centrality = nx.pagerank(G)

    # Create centrality dataframe
    centrality_data = []
    serpin_pattern = re.compile(r'SERPIN|A2M|PZP', re.IGNORECASE)

    for protein in G.nodes():
        is_serpin = bool(serpin_pattern.search(protein))

        centrality_data.append({
            'Protein': protein,
            'Degree_Centrality': degree_centrality[protein],
            'Betweenness_Centrality': betweenness_centrality[protein],
            'Eigenvector_Centrality': eigenvector_centrality[protein],
            'Is_Serpin': is_serpin,
            'Degree': G.degree(protein)
        })

    centrality_df = pd.DataFrame(centrality_data)

    # Calculate average centrality
    centrality_df['Avg_Centrality'] = centrality_df[[
        'Degree_Centrality', 'Betweenness_Centrality', 'Eigenvector_Centrality'
    ]].mean(axis=1)

    # Identify hubs (top 10% in ANY metric)
    hub_threshold = 0.90
    centrality_df['Is_Hub'] = (
        (centrality_df['Degree_Centrality'] >= centrality_df['Degree_Centrality'].quantile(hub_threshold)) |
        (centrality_df['Betweenness_Centrality'] >= centrality_df['Betweenness_Centrality'].quantile(hub_threshold)) |
        (centrality_df['Eigenvector_Centrality'] >= centrality_df['Eigenvector_Centrality'].quantile(hub_threshold))
    )

    # Save centrality data
    output_path = OUTPUT_DIR + 'network_centrality_claude_code.csv'
    centrality_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")

    # Statistical comparison: Serpins vs non-serpins
    print("\nCentrality Comparison: Serpins vs Non-Serpins")

    serpin_centrality = centrality_df[centrality_df['Is_Serpin']]
    nonserpin_centrality = centrality_df[~centrality_df['Is_Serpin']]

    stats_results = {}

    for metric in ['Degree_Centrality', 'Betweenness_Centrality', 'Eigenvector_Centrality']:
        serpin_vals = serpin_centrality[metric]
        nonserpin_vals = nonserpin_centrality[metric]

        u_stat, p_val = mannwhitneyu(serpin_vals, nonserpin_vals, alternative='greater')

        # Effect size
        n1, n2 = len(serpin_vals), len(nonserpin_vals)
        rank_biserial = 1 - (2*u_stat) / (n1 * n2)

        print(f"\n  {metric}:")
        print(f"    Serpin median: {serpin_vals.median():.4f}")
        print(f"    Non-serpin median: {nonserpin_vals.median():.4f}")
        print(f"    Mann-Whitney p: {p_val:.2e}")
        print(f"    Effect size: {rank_biserial:.4f}")

        stats_results[metric] = {
            'p_value': p_val,
            'effect_size': rank_biserial,
            'serpin_median': serpin_vals.median(),
            'nonserpin_median': nonserpin_vals.median()
        }

    # Hub analysis
    total_hubs = centrality_df['Is_Hub'].sum()
    serpin_hubs = centrality_df[centrality_df['Is_Serpin']]['Is_Hub'].sum()
    serpin_fraction = centrality_df['Is_Serpin'].mean()
    serpin_hub_fraction = serpin_hubs / total_hubs if total_hubs > 0 else 0

    print(f"\nHub Analysis:")
    print(f"  Total hubs (top 10%): {total_hubs}")
    print(f"  Serpin hubs: {serpin_hubs}")
    print(f"  % serpins in dataset: {serpin_fraction*100:.1f}%")
    print(f"  % serpins among hubs: {serpin_hub_fraction*100:.1f}%")
    print(f"  Enrichment: {serpin_hub_fraction/serpin_fraction:.2f}x")

    # Visualize network
    visualize_network(G, centrality_df, serpin_pattern)

    return centrality_df, G, stats_results


def visualize_network(G, centrality_df, serpin_pattern):
    """Create network visualization with serpins highlighted"""
    print("\nCreating network visualization...")

    # Get largest connected component for better visualization
    if nx.is_connected(G):
        G_vis = G
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_vis = G.subgraph(largest_cc).copy()
        print(f"  Using largest connected component: {len(G_vis.nodes())} nodes")

    # Prepare node attributes
    node_colors = []
    node_sizes = []

    for node in G_vis.nodes():
        is_serpin = bool(serpin_pattern.search(node))
        node_colors.append('red' if is_serpin else 'lightgray')

        # Size by average centrality
        node_data = centrality_df[centrality_df['Protein'] == node]
        if len(node_data) > 0:
            avg_cent = node_data['Avg_Centrality'].values[0]
            node_sizes.append(50 + avg_cent * 500)
        else:
            node_sizes.append(50)

    # Layout
    print("  Computing layout (this may take a moment)...")
    pos = nx.spring_layout(G_vis, k=0.5, iterations=50, seed=42)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw edges
    nx.draw_networkx_edges(G_vis, pos, alpha=0.2, width=0.5, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax, linewidths=0.5, edgecolors='black')

    # Label top serpins
    serpin_nodes = [n for n in G_vis.nodes() if serpin_pattern.search(n)]
    serpin_centrality = centrality_df[centrality_df['Protein'].isin(serpin_nodes)]
    top_serpins = serpin_centrality.nlargest(10, 'Avg_Centrality')['Protein'].tolist()

    labels = {node: node for node in top_serpins if node in G_vis.nodes()}
    nx.draw_networkx_labels(G_vis, pos, labels, font_size=8, font_weight='bold', ax=ax)

    ax.set_title('Protein Correlation Network\nRed = Serpins (size by centrality), Gray = Other proteins',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(VIZ_DIR + 'network_graph_claude_code.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {VIZ_DIR}network_graph_claude_code.png")


def pathway_analysis(df, serpin_df, centrality_df):
    """Criterion 3: Multi-pathway involvement (20 pts)"""
    print("\n" + "="*80)
    print("CRITERION 3: Multi-Pathway Involvement (20 pts)")
    print("="*80)

    # Define pathways
    pathways = {
        'Coagulation': ['F2', 'F9', 'F10', 'F12', 'F13A1', 'SERPINC1', 'SERPINF2'],
        'Fibrinolysis': ['PLG', 'PLAT', 'PLAU', 'SERPINB2', 'SERPINE1', 'SERPINE2'],
        'Complement': ['C1QA', 'C1QB', 'C1QC', 'C3', 'C4A', 'C4B', 'SERPING1'],
        'ECM_Assembly': ['SERPINH1', 'P4HA1', 'P4HA2', 'PLOD1', 'PLOD2', 'PLOD3'],
        'Inflammation': ['SERPINA3', 'SERPINA1', 'IL6', 'IL1B', 'CCL2', 'CCL5']
    }

    # Get all serpins
    serpin_pattern = re.compile(r'SERPIN|A2M|PZP', re.IGNORECASE)
    all_serpins = serpin_df['Gene_Symbol'].unique()

    # Create serpin-pathway matrix
    print("Mapping serpins to pathways...")

    serpin_pathway_matrix = pd.DataFrame(0, index=all_serpins, columns=pathways.keys())

    # Assign based on pathway definitions
    for pathway, genes in pathways.items():
        for serpin in all_serpins:
            # Direct membership
            if serpin in genes:
                serpin_pathway_matrix.loc[serpin, pathway] = 1
            else:
                # Network connectivity: correlated with ≥2 pathway members
                if serpin in centrality_df['Protein'].values:
                    # Count correlations with pathway members
                    # (simplified: check if serpin is in top correlated proteins)
                    pass  # Would need correlation matrix - simplified here

    # Additional pathway assignments based on functional classification
    for serpin in all_serpins:
        category = classify_serpin(serpin)
        if category == 'Coagulation':
            serpin_pathway_matrix.loc[serpin, 'Coagulation'] = 1
        elif category == 'Fibrinolysis':
            serpin_pathway_matrix.loc[serpin, 'Fibrinolysis'] = 1
        elif category == 'Complement':
            serpin_pathway_matrix.loc[serpin, 'Complement'] = 1
        elif category == 'ECM Assembly':
            serpin_pathway_matrix.loc[serpin, 'ECM_Assembly'] = 1
        elif category == 'Inflammation':
            serpin_pathway_matrix.loc[serpin, 'Inflammation'] = 1
        elif category == 'Broad Inhibitor':
            # A2M, PZP participate in multiple pathways
            serpin_pathway_matrix.loc[serpin, ['Coagulation', 'Fibrinolysis', 'Inflammation']] = 1

    # Calculate pathway dysregulation scores
    print("Calculating pathway dysregulation scores...")

    pathway_stats = []

    for pathway, genes in pathways.items():
        # Get serpins in pathway
        serpins_in_pathway = serpin_pathway_matrix[
            serpin_pathway_matrix[pathway] == 1
        ].index.tolist()

        # Get non-serpin pathway members
        nonserpins_in_pathway = [g for g in genes if not serpin_pattern.search(g)]

        # Calculate mean |Zscore_Delta| for serpins
        serpin_pathway_data = df[df['Gene_Symbol'].isin(serpins_in_pathway)]
        mean_serpin_zscore = serpin_pathway_data['Zscore_Delta'].abs().mean() if len(serpin_pathway_data) > 0 else 0

        # Calculate mean |Zscore_Delta| for non-serpins
        nonserpin_pathway_data = df[df['Gene_Symbol'].isin(nonserpins_in_pathway)]
        mean_nonserpin_zscore = nonserpin_pathway_data['Zscore_Delta'].abs().mean() if len(nonserpin_pathway_data) > 0 else 0

        pathway_stats.append({
            'Pathway': pathway,
            'Serpin_Count': len(serpins_in_pathway),
            'Serpins': ', '.join(serpins_in_pathway),
            'Mean_Serpin_Zscore_Delta': mean_serpin_zscore,
            'Mean_NonSerpin_Zscore_Delta': mean_nonserpin_zscore,
            'Serpin_Dominance': mean_serpin_zscore - mean_nonserpin_zscore
        })

    pathway_df = pd.DataFrame(pathway_stats)
    pathway_df = pathway_df.sort_values('Mean_Serpin_Zscore_Delta', ascending=False)
    pathway_df['Dysregulation_Rank'] = range(1, len(pathway_df) + 1)

    # Save
    output_path = OUTPUT_DIR + 'pathway_dysregulation_claude_code.csv'
    pathway_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")

    print("\nPathway Dysregulation Ranking:")
    for _, row in pathway_df.iterrows():
        print(f"  {row['Dysregulation_Rank']}. {row['Pathway']}: "
              f"{row['Serpin_Count']} serpins, "
              f"mean |Δz|={row['Mean_Serpin_Zscore_Delta']:.3f}")

    # Visualize serpin-pathway heatmap
    visualize_pathway_heatmap(serpin_pathway_matrix)

    # Venn diagram for top 3 pathways
    visualize_pathway_venn(serpin_pathway_matrix, pathway_df)

    return pathway_df, serpin_pathway_matrix


def visualize_pathway_heatmap(matrix):
    """Create serpin-pathway heatmap"""
    print("\nCreating pathway heatmap...")

    # Filter serpins that participate in at least one pathway
    active_serpins = matrix[matrix.sum(axis=1) > 0]

    if len(active_serpins) == 0:
        print("  Warning: No serpins with pathway assignments")
        return

    fig, ax = plt.subplots(figsize=(8, max(6, len(active_serpins) * 0.3)))

    sns.heatmap(active_serpins, cmap='Reds', cbar_kws={'label': 'Pathway Participation'},
               linewidths=0.5, linecolor='gray', ax=ax, vmin=0, vmax=1)

    ax.set_title('Serpin-Pathway Participation Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Biological Pathway', fontsize=10)
    ax.set_ylabel('Serpin', fontsize=10)

    plt.tight_layout()
    plt.savefig(VIZ_DIR + 'pathway_heatmap_claude_code.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {VIZ_DIR}pathway_heatmap_claude_code.png")


def visualize_pathway_venn(matrix, pathway_df):
    """Create Venn diagram of pathway overlaps"""
    print("Creating pathway Venn diagram...")

    # Get top 3 pathways by dysregulation
    top3_pathways = pathway_df.head(3)['Pathway'].tolist()

    if len(top3_pathways) < 3:
        print("  Warning: Less than 3 pathways available")
        return

    # Get serpins in each pathway
    sets = []
    labels = []

    for pathway in top3_pathways:
        serpins_in_pathway = set(matrix[matrix[pathway] == 1].index)
        sets.append(serpins_in_pathway)
        labels.append(pathway.replace('_', ' '))

    fig, ax = plt.subplots(figsize=(10, 8))

    v = venn3(sets, set_labels=labels, ax=ax)

    # Color circles
    if v.get_patch_by_id('100'): v.get_patch_by_id('100').set_color('lightcoral')
    if v.get_patch_by_id('010'): v.get_patch_by_id('010').set_color('lightblue')
    if v.get_patch_by_id('001'): v.get_patch_by_id('001').set_color('lightgreen')

    ax.set_title('Serpin Overlap Across Top 3 Pathways', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR + 'pathway_venn_claude_code.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {VIZ_DIR}pathway_venn_claude_code.png")


def therapeutic_analysis(profile_df, centrality_df, pathway_matrix):
    """Criterion 4: Temporal and therapeutic implications (10 pts)"""
    print("\n" + "="*80)
    print("CRITERION 4: Therapeutic Analysis (10 pts)")
    print("="*80)

    # Merge serpin data
    serpin_pattern = re.compile(r'SERPIN|A2M|PZP', re.IGNORECASE)

    # Get serpins that are in both datasets
    serpins_in_network = centrality_df[centrality_df['Is_Serpin']]['Protein'].tolist()

    therapeutic_data = []

    for serpin in serpins_in_network:
        # Get profile data
        profile_row = profile_df[profile_df['Gene_Symbol'] == serpin]
        if len(profile_row) == 0:
            continue

        # Get centrality data
        cent_row = centrality_df[centrality_df['Protein'] == serpin]

        # Get pathway data
        if serpin in pathway_matrix.index:
            pathway_count = pathway_matrix.loc[serpin].sum()
        else:
            pathway_count = 0

        # Druggability (simplified)
        druggable = serpin.upper() in ['SERPINC1', 'SERPINE1', 'SERPINA1', 'SERPINH1']

        # Calculate composite score
        # Normalize centrality (0-1)
        norm_centrality = cent_row['Avg_Centrality'].values[0] if len(cent_row) > 0 else 0

        # Normalize pathway count (0-1, max 5 pathways)
        norm_pathway = min(pathway_count / 5, 1.0)

        # Normalize dysregulation (0-1, using abs)
        abs_zscore = profile_row['Abs_Mean_Zscore_Delta'].values[0]
        norm_zscore = min(abs_zscore / 3.0, 1.0)  # Assuming max ~3

        # Druggability bonus
        drug_bonus = 0.5 if druggable else 0

        # Final score
        final_score = (norm_centrality + norm_pathway + norm_zscore) * (1 + drug_bonus)

        therapeutic_data.append({
            'Serpin': serpin,
            'Centrality_Score': norm_centrality,
            'Pathway_Count': pathway_count,
            'Dysregulation': abs_zscore,
            'Druggable': druggable,
            'Final_Score': final_score,
            'Mean_Zscore_Delta': profile_row['Mean_Zscore_Delta'].values[0]
        })

    therapeutic_df = pd.DataFrame(therapeutic_data)
    therapeutic_df = therapeutic_df.sort_values('Final_Score', ascending=False)

    print("\nTop 10 Therapeutic Targets:")
    for i, row in therapeutic_df.head(10).iterrows():
        print(f"  {row['Serpin']}: score={row['Final_Score']:.3f}, "
              f"centrality={row['Centrality_Score']:.3f}, "
              f"pathways={row['Pathway_Count']}, "
              f"Δz={row['Mean_Zscore_Delta']:.3f}, "
              f"druggable={row['Druggable']}")

    return therapeutic_df


def generate_summary_statistics():
    """Generate summary for final report"""
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - Summary Statistics")
    print("="*80)

    # Load generated files
    profile_df = pd.read_csv(OUTPUT_DIR + 'serpin_comprehensive_profile_claude_code.csv')
    centrality_df = pd.read_csv(OUTPUT_DIR + 'network_centrality_claude_code.csv')
    pathway_df = pd.read_csv(OUTPUT_DIR + 'pathway_dysregulation_claude_code.csv')

    print(f"\n✓ Total serpins profiled: {len(profile_df)}")
    print(f"✓ Serpins in network: {centrality_df['Is_Serpin'].sum()}")
    print(f"✓ Network size: {len(centrality_df)} proteins, "
          f"{centrality_df['Is_Hub'].sum()} hubs")
    print(f"✓ Pathways analyzed: {len(pathway_df)}")

    print("\nTop 5 Most Dysregulated Serpins:")
    for i, row in profile_df.head(5).iterrows():
        print(f"  {row['Gene_Symbol']}: Δz={row['Mean_Zscore_Delta']:.3f}, "
              f"tissues={row['Tissue_Breadth']}, "
              f"category={row['Functional_Category']}")

    print("\nAll outputs saved to:")
    print(f"  {OUTPUT_DIR}")


def main():
    """Main analysis pipeline"""
    print("="*80)
    print("SERPIN CASCADE DYSREGULATION ANALYSIS")
    print("Agent: claude_code")
    print("="*80)

    # Load data
    df = load_data()

    # Identify serpins
    serpin_df, unique_serpins = identify_serpins(df)

    # Criterion 1: Comprehensive profiling
    profile_df, serpin_stats = comprehensive_serpin_profiling(df, serpin_df, unique_serpins)

    # Criterion 2: Network centrality
    centrality_df, network, network_stats = build_correlation_network(df, unique_serpins)

    # Criterion 3: Pathway analysis
    pathway_df, pathway_matrix = pathway_analysis(df, serpin_df, centrality_df)

    # Criterion 4: Therapeutic analysis
    therapeutic_df = therapeutic_analysis(profile_df, centrality_df, pathway_matrix)

    # Summary
    generate_summary_statistics()

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext step: Review outputs and create 90_results_claude_code.md")


if __name__ == '__main__':
    main()
