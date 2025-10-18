#!/usr/bin/env python3
"""
DEATh Lemma 3 Validation Analysis
Author: Daniel Kravtsov (daniel@improvado.io)
Date: 2025-10-18

Validates DEATh Lemma 3 (entropy expulsion via ECM remodeling) using batch-corrected
proteomics data. Tests whether transition proteins represent mechanosensory-driven
entropy export from cells to ECM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1.0 DATA LOADING
# ============================================================================

def load_entropy_data():
    """Load V2 entropy metrics from claude_code_agent_01"""
    data_path = Path("../claude_code_agent_01/entropy_metrics_v2.csv")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} proteins from entropy_metrics_v2.csv")
    return df

# ============================================================================
# 2.0 PROTEIN CLASSIFICATION BY LEMMA 3 ROLES
# ============================================================================

def classify_lemma3_roles(df):
    """
    Classify proteins by Lemma 3 functional roles:
    - Mechanosensors: YAP/TAZ targets, integrins, focal adhesion
    - Entropy_Exporters: MMPs, serpins, LOX family (execute entropy expulsion)
    - Structural_Overflow: Collagens, laminins with high entropy (aberrant deposition)
    - Pathology_Markers: Inflammatory cytokines, fibrosis markers
    - Other: Remaining proteins
    """

    # YAP/TAZ targets (from Panciera et al. 2017, Nature Rev Mol Cell Biol)
    yap_targets = [
        'CTGF', 'CCN2',  # CTGF same as CCN2
        'CYR61', 'CCN1',  # CYR61 same as CCN1
        'ANKRD1',
        'SERPINE1',  # PAI-1
        'COL1A1', 'COL3A1', 'COL5A1',
        'FN1', 'LAMC2', 'ITGB2',
        'MMP7', 'MMP10'
    ]

    # ECM remodeling enzymes (entropy exporters)
    exporters = [
        # MMPs (matrix metalloproteinases)
        'MMP1', 'MMP2', 'MMP3', 'MMP7', 'MMP8', 'MMP9', 'MMP10', 'MMP11',
        'MMP12', 'MMP13', 'MMP14', 'MMP15', 'MMP16', 'MMP17',
        # Serpins (serine protease inhibitors - regulate MMPs)
        'SERPINA1', 'SERPINA3', 'SERPINB2', 'SERPINB3', 'SERPINB9',
        'SERPINC1', 'SERPINE1', 'SERPINF1', 'SERPING1', 'SERPINH1',
        # LOX family (lysyl oxidases - crosslinking enzymes)
        'LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4',
        # Other proteases
        'ADAM9', 'ADAM10', 'ADAM12', 'ADAM15', 'ADAM17',
        'ADAMTS1', 'ADAMTS2', 'ADAMTS4', 'ADAMTS5',
        # Protease regulators
        'PZP',  # Pregnancy zone protein (protease inhibitor like A2M)
        'A2M',  # Alpha-2-macroglobulin
        'TIMP1', 'TIMP2', 'TIMP3', 'TIMP4'
    ]

    # Structural proteins (potential aberrant deposition)
    structural = [
        # Collagens
        'COL1A1', 'COL1A2', 'COL2A1', 'COL3A1', 'COL4A1', 'COL4A2',
        'COL5A1', 'COL5A2', 'COL6A1', 'COL6A2', 'COL6A3',
        'COL10A1', 'COL11A1', 'COL12A1', 'COL14A1', 'COL15A1', 'COL16A1',
        # Laminins
        'LAMA1', 'LAMA2', 'LAMA4', 'LAMA5',
        'LAMB1', 'LAMB2', 'LAMC1', 'LAMC2',
        # Fibronectin
        'FN1',
        # Elastin
        'ELN'
    ]

    # Pathology markers (inflammation, fibrosis)
    pathology = [
        'TNFSF13', 'TNFSF13B',  # Inflammatory cytokines
        'CTGF', 'CCN2',  # Fibrosis marker
        'IL6', 'IL1B', 'IL8',
        'TGFB1', 'TGFB2', 'TGFB3',
        'CCL2', 'CCL21', 'CXCL10', 'CXCL14'
    ]

    # Mechanosensors (integrins, focal adhesion)
    mechanosensors = [
        'ITGA1', 'ITGA2', 'ITGA3', 'ITGA5', 'ITGA6', 'ITGAV',
        'ITGB1', 'ITGB2', 'ITGB3', 'ITGB4', 'ITGB5',
        'VCL',  # Vinculin
        'TLN1', 'TLN2',  # Talin
        'FLNA', 'FLNB', 'FLNC'  # Filamin
    ]

    # Classify each protein
    roles = []
    for protein in df['Protein']:
        assigned_roles = []

        if protein in yap_targets:
            assigned_roles.append('YAP_Target')
        if protein in mechanosensors:
            assigned_roles.append('Mechanosensor')
        if protein in exporters:
            assigned_roles.append('Entropy_Exporter')
        if protein in structural:
            assigned_roles.append('Structural')
        if protein in pathology:
            assigned_roles.append('Pathology')

        if not assigned_roles:
            assigned_roles.append('Other')

        # Primary role (priority: Exporter > Mechanosensor > Structural > Pathology > YAP > Other)
        if 'Entropy_Exporter' in assigned_roles:
            primary = 'Entropy_Exporter'
        elif 'Mechanosensor' in assigned_roles:
            primary = 'Mechanosensor'
        elif 'Structural' in assigned_roles:
            primary = 'Structural'
        elif 'Pathology' in assigned_roles:
            primary = 'Pathology'
        elif 'YAP_Target' in assigned_roles:
            primary = 'YAP_Target'
        else:
            primary = 'Other'

        roles.append({
            'Protein': protein,
            'Lemma3_Role': primary,
            'All_Roles': ';'.join(assigned_roles),
            'Is_YAP_Target': protein in yap_targets
        })

    roles_df = pd.DataFrame(roles)
    return roles_df

# ============================================================================
# 3.0 ENRICHMENT TESTS
# ============================================================================

def test_enrichment(df, roles_df, category, top_n=50):
    """
    Test enrichment of a category in top transition proteins using Fisher's exact test
    """
    # Merge roles
    df_merged = df.merge(roles_df, on='Protein')

    # Get top transition proteins
    df_sorted = df_merged.sort_values('Entropy_Transition', ascending=False)
    top_proteins = set(df_sorted.head(top_n)['Protein'])

    # Count category members in top vs background
    category_members = set(df_merged[df_merged['Lemma3_Role'] == category]['Protein'])

    # 2x2 contingency table
    a = len(top_proteins & category_members)  # Category in top
    b = len(top_proteins - category_members)  # Non-category in top
    c = len(category_members - top_proteins)  # Category in background
    d = len(df_merged) - len(category_members) - b  # Non-category in background

    # Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')

    result = {
        'Category': category,
        'Top_N': top_n,
        'In_Top': a,
        'In_Background': c,
        'Total_Category': a + c,
        'Enrichment': a / top_n if top_n > 0 else 0,
        'Background_Rate': (a + c) / len(df_merged),
        'Odds_Ratio': odds_ratio,
        'P_Value': p_value
    }

    return result

def test_yap_enrichment(df, roles_df, metric='Entropy_Transition', top_n=50):
    """Test YAP/TAZ target enrichment in high-metric proteins"""
    df_merged = df.merge(roles_df, on='Protein')

    # Get top proteins by metric
    df_sorted = df_merged.sort_values(metric, ascending=False)
    top_proteins = set(df_sorted.head(top_n)['Protein'])

    # YAP targets
    yap_targets = set(df_merged[df_merged['Is_YAP_Target']]['Protein'])

    # Contingency table
    a = len(top_proteins & yap_targets)
    b = len(top_proteins - yap_targets)
    c = len(yap_targets - top_proteins)
    d = len(df_merged) - len(yap_targets) - b

    odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')

    result = {
        'Metric': metric,
        'Top_N': top_n,
        'YAP_in_Top': a,
        'YAP_in_Background': c,
        'Total_YAP': a + c,
        'Enrichment': a / top_n if top_n > 0 else 0,
        'Background_Rate': (a + c) / len(df_merged),
        'Odds_Ratio': odds_ratio,
        'P_Value': p_value
    }

    return result

# ============================================================================
# 4.0 STRUCTURAL ENTROPY ANALYSIS (Prediction 3)
# ============================================================================

def analyze_structural_entropy(df):
    """
    Test Lemma 3 Prediction 3: Structural protein Shannon entropy increase
    reflects aberrant ECM deposition
    """
    # Compare Core matrisome vs Matrisome-associated
    core = df[df['Matrisome_Division'] == 'Core matrisome']['Shannon_Entropy']
    associated = df[df['Matrisome_Division'] == 'Matrisome-associated']['Shannon_Entropy']

    # Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(core, associated, alternative='greater')

    result = {
        'Core_Mean': core.mean(),
        'Core_Std': core.std(),
        'Core_N': len(core),
        'Associated_Mean': associated.mean(),
        'Associated_Std': associated.std(),
        'Associated_N': len(associated),
        'U_Statistic': statistic,
        'P_Value': p_value,
        'Effect_Size': (core.mean() - associated.mean()) / core.std()
    }

    return result

# ============================================================================
# 5.0 VISUALIZATION
# ============================================================================

def create_lemma3_pathway_diagram(enrichment_results, output_dir):
    """
    Create pathway diagram showing Lemma 3 flow with protein examples
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Define pathway stages
    stages = [
        ("ECM\nCrosslinking", 0.1, 0.5, 'lightcoral'),
        ("Matrix\nStiffening", 0.25, 0.5, 'salmon'),
        ("YAP/TAZ\nActivation", 0.4, 0.5, 'gold'),
        ("ECM Remodeling\nEnzymes", 0.55, 0.5, 'lightgreen'),
        ("Entropy\nExpulsion", 0.7, 0.5, 'lightblue'),
        ("Tissue\nPathology", 0.85, 0.5, 'plum')
    ]

    # Draw boxes
    for stage, x, y, color in stages:
        ax.add_patch(plt.Rectangle((x-0.05, y-0.08), 0.1, 0.16,
                                   facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x, y, stage, ha='center', va='center', fontsize=10,
               fontweight='bold', wrap=True)

    # Draw arrows
    for i in range(len(stages)-1):
        x1, y1 = stages[i][1] + 0.05, stages[i][2]
        x2, y2 = stages[i+1][1] - 0.05, stages[i+1][2]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add protein examples below each stage
    examples = [
        "AGEs\nLOX",
        "↑Stiffness\n↓E",
        "CTGF\nCYR61",
        "MMPs\nSerpins\nPZP",
        "↑E_ECM\n↓C_cell",
        "Fibrosis\nInflammation"
    ]

    for i, (stage, x, y, color) in enumerate(stages):
        ax.text(x, y - 0.15, examples[i], ha='center', va='top',
               fontsize=8, style='italic', color='darkblue')

    # Add title
    ax.text(0.5, 0.9, 'DEATh Lemma 3: Entropy Expulsion Pathway',
           ha='center', va='center', fontsize=14, fontweight='bold')

    # Add equations
    ax.text(0.5, 0.75, r'$\frac{dC}{dt} = f(C,E), \quad \frac{dE}{dt} = -g(C,E)$',
           ha='center', va='center', fontsize=12, style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / '01_lemma3_pathway_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created Lemma 3 pathway diagram")

def create_transition_pie_chart(df, roles_df, output_dir):
    """
    Pie chart showing proportion of top transition proteins by Lemma 3 role
    """
    # Merge and get top 50
    df_merged = df.merge(roles_df, on='Protein')
    top50 = df_merged.nlargest(50, 'Entropy_Transition')

    # Count by role
    role_counts = top50['Lemma3_Role'].value_counts()

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {
        'Entropy_Exporter': '#2ecc71',
        'Mechanosensor': '#f39c12',
        'Structural': '#3498db',
        'Pathology': '#e74c3c',
        'YAP_Target': '#9b59b6',
        'Other': '#95a5a6'
    }

    wedge_colors = [colors.get(role, '#95a5a6') for role in role_counts.index]

    wedges, texts, autotexts = ax.pie(role_counts.values,
                                       labels=role_counts.index,
                                       autopct='%1.1f%%',
                                       colors=wedge_colors,
                                       startangle=90,
                                       textprops={'fontsize': 12})

    # Bold percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax.set_title('Top 50 Transition Proteins by Lemma 3 Role',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / '02_transition_proteins_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created transition proteins pie chart")

def create_yap_target_heatmap(df, roles_df, output_dir):
    """
    Heatmap of entropy metrics for YAP/TAZ targets
    """
    # Get YAP targets
    df_merged = df.merge(roles_df, on='Protein')
    yap_df = df_merged[df_merged['Is_YAP_Target']].copy()

    if len(yap_df) == 0:
        print("⚠️  No YAP targets found in dataset")
        return

    # Select metrics for heatmap
    metrics = ['Shannon_Entropy', 'Entropy_Transition', 'Predictability_Score', 'Variance_Entropy_CV']
    heatmap_data = yap_df.set_index('Protein')[metrics]

    # Normalize each metric to 0-1 for visualization
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(yap_df) * 0.4)))

    sns.heatmap(heatmap_norm, annot=heatmap_data.round(3), fmt='',
               cmap='RdYlGn_r', linewidths=0.5, cbar_kws={'label': 'Normalized Value'},
               ax=ax, vmin=0, vmax=1)

    ax.set_title('YAP/TAZ Target Entropy Metrics', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Entropy Metrics', fontsize=12)
    ax.set_ylabel('YAP/TAZ Target Proteins', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / '03_yap_target_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Created YAP target heatmap ({len(yap_df)} proteins)")

def create_entropy_flow_schematic(structural_result, output_dir):
    """
    Entropy flow schematic showing C and E dynamics
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Young state (t0)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Young (t₀)', fontsize=14, fontweight='bold')

    # Cell
    cell = plt.Circle((5, 5), 2, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(cell)
    ax.text(5, 5, 'C = low', ha='center', va='center', fontsize=12, fontweight='bold')

    # Ordered ECM (parallel lines)
    for i in range(5):
        ax.plot([0.5, 2.5], [1+i*1.5, 1+i*1.5], 'k-', linewidth=2)
        ax.plot([7.5, 9.5], [1+i*1.5, 1+i*1.5], 'k-', linewidth=2)
    ax.text(1.5, 0.2, 'E = low', ha='center', fontsize=10, style='italic')
    ax.text(8.5, 0.2, '(ordered)', ha='center', fontsize=10, style='italic')

    # Equation
    ax.text(5, 9, 'ϕ(C, E) = constant', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Old state (Lemma 2)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Old: Lemma 2', fontsize=14, fontweight='bold')

    # Cell with high entropy
    cell = plt.Circle((5, 5), 2, color='salmon', ec='black', linewidth=2)
    ax.add_patch(cell)
    ax.text(5, 5, 'C = HIGH', ha='center', va='center', fontsize=12, fontweight='bold')

    # Add entropy symbols in cell
    for i in range(8):
        x, y = 5 + np.random.uniform(-1.5, 1.5), 5 + np.random.uniform(-1.5, 1.5)
        if (x-5)**2 + (y-5)**2 < 3:
            ax.plot(x, y, 'r^', markersize=6)

    # Crosslinked ECM (crosshatch pattern)
    for i in range(5):
        ax.plot([0.5, 2.5], [1+i*1.5, 1+i*1.5], 'k-', linewidth=2)
        ax.plot([1.5, 1.5], [1, 7.5], 'k-', linewidth=2)
    ax.text(1.5, 0.2, 'E = low', ha='center', fontsize=10, style='italic')
    ax.text(1.5, 9, '(crosslinked)', ha='center', fontsize=8, style='italic')

    # Equation
    ax.text(5, 9.5, 'dC/dt × dE/dt < 0', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Pathological state (Lemma 3)
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Pathological: Lemma 3', fontsize=14, fontweight='bold')

    # Cell with reduced entropy
    cell = plt.Circle((5, 5), 2, color='lightgreen', ec='black', linewidth=2)
    ax.add_patch(cell)
    ax.text(5, 5, 'C = reduced', ha='center', va='center', fontsize=11, fontweight='bold')

    # Fewer entropy symbols
    for i in range(3):
        x, y = 5 + np.random.uniform(-1.3, 1.3), 5 + np.random.uniform(-1.3, 1.3)
        if (x-5)**2 + (y-5)**2 < 2.5:
            ax.plot(x, y, 'r^', markersize=5)

    # Fragmented, disordered ECM
    for i in range(15):
        x1, x2 = np.random.uniform(0.5, 2.5), np.random.uniform(0.5, 2.5)
        y1, y2 = np.random.uniform(1, 8), np.random.uniform(1, 8)
        ax.plot([x1, x2], [y1, y2], 'purple', linewidth=1.5, alpha=0.6)

    # MMPs (scissors symbol)
    ax.text(2, 8.5, '✂️ MMPs', ha='center', fontsize=10, fontweight='bold')

    ax.text(1.5, 0.2, 'E = HIGH', ha='center', fontsize=10, style='italic', color='purple')
    ax.text(1.5, 9.5, '(fragmented)', ha='center', fontsize=8, style='italic')

    # Equation
    ax.text(5, 9.5, 'dE/dt = -g(C,E) → E↑', ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('DEATh Entropy Delocalization: Young → Old → Pathological',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_dir / '04_entropy_flow_schematic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Created entropy flow schematic")

# ============================================================================
# 6.0 MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("DEATh Lemma 3 Validation Analysis")
    print("=" * 80)

    # Create output directory
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)

    # 1. Load data
    print("\n1.0 LOADING DATA...")
    df = load_entropy_data()

    # 2. Classify proteins by Lemma 3 roles
    print("\n2.0 CLASSIFYING PROTEINS BY LEMMA 3 ROLES...")
    roles_df = classify_lemma3_roles(df)
    print(f"✅ Classified {len(roles_df)} proteins")
    print("\nRole Distribution:")
    print(roles_df['Lemma3_Role'].value_counts())

    # Save classification
    roles_df.to_csv(output_dir / 'protein_lemma3_roles.csv', index=False)
    print(f"\n✅ Saved protein_lemma3_roles.csv")

    # 3. Test Prediction 1: ECM Regulators in top transitions
    print("\n3.0 TESTING PREDICTION 1: ECM REGULATOR ENRICHMENT...")
    categories = ['Entropy_Exporter', 'Mechanosensor', 'Structural', 'Pathology']
    enrichment_results = []

    for category in categories:
        result = test_enrichment(df, roles_df, category, top_n=50)
        enrichment_results.append(result)
        print(f"\n{category}:")
        print(f"  In top 50: {result['In_Top']}/{result['Top_N']}")
        print(f"  Enrichment: {result['Enrichment']:.1%} (background: {result['Background_Rate']:.1%})")
        print(f"  Odds Ratio: {result['Odds_Ratio']:.2f}")
        print(f"  P-value: {result['P_Value']:.2e}")
        if result['P_Value'] < 0.05:
            print(f"  ✅ SIGNIFICANT")
        else:
            print(f"  ❌ Not significant")

    enrichment_df = pd.DataFrame(enrichment_results)
    enrichment_df.to_csv(output_dir / 'enrichment_results.csv', index=False)

    # 4. Test Prediction 2: YAP/TAZ target enrichment
    print("\n4.0 TESTING PREDICTION 2: YAP/TAZ TARGET ENRICHMENT...")
    yap_results = []

    for metric in ['Entropy_Transition', 'Shannon_Entropy']:
        result = test_yap_enrichment(df, roles_df, metric=metric, top_n=50)
        yap_results.append(result)
        print(f"\n{metric}:")
        print(f"  YAP targets in top 50: {result['YAP_in_Top']}/{result['Total_YAP']}")
        print(f"  Enrichment: {result['Enrichment']:.1%} (background: {result['Background_Rate']:.1%})")
        print(f"  Odds Ratio: {result['Odds_Ratio']:.2f}")
        print(f"  P-value: {result['P_Value']:.2e}")
        if result['P_Value'] < 0.05:
            print(f"  ✅ SIGNIFICANT")
        else:
            print(f"  ❌ Not significant")

    yap_df = pd.DataFrame(yap_results)
    yap_df.to_csv(output_dir / 'yap_enrichment_results.csv', index=False)

    # 5. Test Prediction 3: Structural entropy analysis
    print("\n5.0 TESTING PREDICTION 3: STRUCTURAL ENTROPY INCREASE...")
    structural_result = analyze_structural_entropy(df)
    print(f"\nCore matrisome: {structural_result['Core_Mean']:.3f} ± {structural_result['Core_Std']:.3f} (n={structural_result['Core_N']})")
    print(f"Associated: {structural_result['Associated_Mean']:.3f} ± {structural_result['Associated_Std']:.3f} (n={structural_result['Associated_N']})")
    print(f"Difference: {structural_result['Core_Mean'] - structural_result['Associated_Mean']:.3f}")
    print(f"Effect size: {structural_result['Effect_Size']:.3f}")
    print(f"P-value: {structural_result['P_Value']:.2e}")
    if structural_result['P_Value'] < 0.05:
        print(f"✅ SIGNIFICANT: Core matrisome has HIGHER entropy (Lemma 3 supported)")
    else:
        print(f"❌ Not significant")

    # 6. Create visualizations
    print("\n6.0 CREATING VISUALIZATIONS...")
    create_lemma3_pathway_diagram(enrichment_results, output_dir)
    create_transition_pie_chart(df, roles_df, output_dir)
    create_yap_target_heatmap(df, roles_df, output_dir)
    create_entropy_flow_schematic(structural_result, output_dir)

    # 7. Summary
    print("\n" + "=" * 80)
    print("LEMMA 3 VALIDATION SUMMARY")
    print("=" * 80)

    print("\n✅ PREDICTIONS SUPPORTED:")
    if enrichment_df[enrichment_df['P_Value'] < 0.05].shape[0] > 0:
        print(f"  - Prediction 1: {enrichment_df[enrichment_df['P_Value'] < 0.05]['Category'].tolist()}")
    if yap_df[yap_df['P_Value'] < 0.05].shape[0] > 0:
        print(f"  - Prediction 2: YAP targets enriched in {yap_df[yap_df['P_Value'] < 0.05]['Metric'].tolist()}")
    if structural_result['P_Value'] < 0.05:
        print(f"  - Prediction 3: Structural entropy increase (p={structural_result['P_Value']:.2e})")

    print("\n❌ PREDICTIONS NOT SUPPORTED:")
    if enrichment_df[enrichment_df['P_Value'] >= 0.05].shape[0] > 0:
        print(f"  - Prediction 1: {enrichment_df[enrichment_df['P_Value'] >= 0.05]['Category'].tolist()}")
    if yap_df[yap_df['P_Value'] >= 0.05].shape[0] > 0:
        print(f"  - Prediction 2: YAP targets NOT enriched in {yap_df[yap_df['P_Value'] >= 0.05]['Metric'].tolist()}")
    if structural_result['P_Value'] >= 0.05:
        print(f"  - Prediction 3: Structural entropy difference not significant")

    print("\n✅ Analysis complete! Outputs saved to current directory.")
    print("=" * 80)

if __name__ == "__main__":
    main()
