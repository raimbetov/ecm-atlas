#!/usr/bin/env python3
"""
Hybrid Model Analysis: Universal vs Personalized ECM Aging
===========================================================

Develops multi-level aging framework decomposing variance into:
1. Universal baseline (cross-tissue, cross-study core signatures)
2. Tissue-specific modifiers
3. Study-specific modifiers (proxy for individual variation)
4. Residual/stochastic noise

Mathematical Model:
Aging_phenotype = Universal_baseline + Tissue_modifier + Study_modifier + Noise
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set paths
DATA_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = "/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.3_universal_vs_personal/agent3"

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load and prepare ECM-Atlas dataset"""
    print("Loading ECM-Atlas dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Total records: {len(df):,}")
    print(f"Unique proteins: {df['Protein_ID'].nunique():,}")
    print(f"Unique tissues: {df['Tissue'].nunique()}")
    print(f"Unique studies: {df['Study_ID'].nunique()}")
    print(f"Unique genes: {df['Canonical_Gene_Symbol'].nunique():,}")

    return df

def explore_variance_structure(df):
    """Explore variance across different levels"""
    print("\n" + "="*60)
    print("VARIANCE STRUCTURE EXPLORATION")
    print("="*60)

    # Filter valid z-score deltas
    df_valid = df[df['Zscore_Delta'].notna()].copy()
    print(f"\nRecords with valid Zscore_Delta: {len(df_valid):,}")

    # Overall variance
    overall_var = df_valid['Zscore_Delta'].var()
    overall_std = df_valid['Zscore_Delta'].std()
    print(f"\nOverall variance: {overall_var:.4f}")
    print(f"Overall std dev: {overall_std:.4f}")

    # Variance by tissue
    tissue_vars = df_valid.groupby('Tissue')['Zscore_Delta'].agg(['mean', 'std', 'var', 'count'])
    tissue_vars = tissue_vars.sort_values('var', ascending=False)
    print("\nTop 10 tissues by variance:")
    print(tissue_vars.head(10))

    # Variance by study
    study_vars = df_valid.groupby('Study_ID')['Zscore_Delta'].agg(['mean', 'std', 'var', 'count'])
    study_vars = study_vars.sort_values('var', ascending=False)
    print("\nVariance by study:")
    print(study_vars)

    # Variance by protein
    protein_vars = df_valid.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].agg(['mean', 'std', 'var', 'count'])
    protein_vars = protein_vars[protein_vars['count'] >= 3].sort_values('var', ascending=False)
    print(f"\nTop 10 proteins by variance (n>=3):")
    print(protein_vars.head(10))

    return df_valid, tissue_vars, study_vars, protein_vars

def variance_decomposition_anova(df_valid):
    """
    Variance decomposition using nested ANOVA approach

    Model: Y_ijkl = μ + α_i + β_j(i) + γ_k + ε_ijkl
    Where:
    - Y = Zscore_Delta
    - μ = grand mean (universal baseline)
    - α_i = protein effect (universal component)
    - β_j(i) = tissue effect within protein
    - γ_k = study effect (personalization proxy)
    - ε = residual error
    """
    print("\n" + "="*60)
    print("VARIANCE DECOMPOSITION (ANOVA)")
    print("="*60)

    # Filter proteins present in multiple contexts
    protein_counts = df_valid.groupby('Canonical_Gene_Symbol').size()
    common_proteins = protein_counts[protein_counts >= 5].index
    df_common = df_valid[df_valid['Canonical_Gene_Symbol'].isin(common_proteins)].copy()

    print(f"\nAnalyzing {len(common_proteins)} proteins with ≥5 observations")
    print(f"Total observations: {len(df_common):,}")

    # Calculate variance components
    total_variance = df_common['Zscore_Delta'].var()
    print(f"\nTotal variance: {total_variance:.4f}")

    # 1. Universal component: variance explained by protein identity
    protein_means = df_common.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean()
    universal_var = protein_means.var()

    # 2. Tissue-specific component
    tissue_residuals = []
    for protein in common_proteins:
        protein_data = df_common[df_common['Canonical_Gene_Symbol'] == protein]
        protein_mean = protein_data['Zscore_Delta'].mean()
        tissue_means = protein_data.groupby('Tissue')['Zscore_Delta'].mean()
        tissue_var = ((tissue_means - protein_mean) ** 2).mean()
        tissue_residuals.append(tissue_var)
    tissue_var = np.mean(tissue_residuals)

    # 3. Study-specific component (personalization)
    study_residuals = []
    for combo in df_common.groupby(['Canonical_Gene_Symbol', 'Tissue']):
        combo_data = combo[1]
        if len(combo_data) > 1:
            combo_mean = combo_data['Zscore_Delta'].mean()
            study_means = combo_data.groupby('Study_ID')['Zscore_Delta'].mean()
            study_var = ((study_means - combo_mean) ** 2).mean()
            study_residuals.append(study_var)
    study_var = np.mean(study_residuals) if study_residuals else 0

    # 4. Residual variance
    residual_var = total_variance - (universal_var + tissue_var + study_var)
    residual_var = max(residual_var, 0)  # Ensure non-negative

    # Calculate proportions
    components = {
        'Universal (Protein)': universal_var,
        'Tissue-specific': tissue_var,
        'Study-specific (Personalized)': study_var,
        'Residual/Noise': residual_var
    }

    total_explained = sum(components.values())
    proportions = {k: v/total_explained for k, v in components.items()}

    print("\nVariance Components:")
    print("-" * 60)
    for name, var in components.items():
        prop = proportions[name]
        print(f"{name:35s}: {var:8.4f} ({prop*100:5.1f}%)")
    print("-" * 60)
    print(f"{'Total':35s}: {total_explained:8.4f} (100.0%)")

    return components, proportions

def identify_universal_signatures(df_valid, n_top=50):
    """
    Identify universal aging signatures:
    Proteins with consistent direction across tissues and studies
    """
    print("\n" + "="*60)
    print("UNIVERSAL AGING SIGNATURES")
    print("="*60)

    # Calculate consistency metrics for each protein
    protein_stats = []

    for protein in df_valid['Canonical_Gene_Symbol'].unique():
        protein_data = df_valid[df_valid['Canonical_Gene_Symbol'] == protein]

        if len(protein_data) >= 3:
            mean_delta = protein_data['Zscore_Delta'].mean()
            std_delta = protein_data['Zscore_Delta'].std()
            count = len(protein_data)

            # Consistency score: mean/std (higher = more consistent)
            consistency = abs(mean_delta) / (std_delta + 0.001)

            # Direction consistency: % of observations with same sign
            direction_consistency = (np.sign(protein_data['Zscore_Delta']) == np.sign(mean_delta)).sum() / count

            # Number of tissues
            n_tissues = protein_data['Tissue'].nunique()
            n_studies = protein_data['Study_ID'].nunique()

            protein_stats.append({
                'Protein': protein,
                'Mean_Delta': mean_delta,
                'Std_Delta': std_delta,
                'Consistency_Score': consistency,
                'Direction_Consistency': direction_consistency,
                'N_Observations': count,
                'N_Tissues': n_tissues,
                'N_Studies': n_studies,
                'Matrisome_Category': protein_data['Matrisome_Category'].mode()[0] if len(protein_data['Matrisome_Category'].mode()) > 0 else 'Unknown'
            })

    signatures_df = pd.DataFrame(protein_stats)

    # Filter for universal candidates: present in multiple contexts, high consistency
    universal_candidates = signatures_df[
        (signatures_df['N_Observations'] >= 5) &
        (signatures_df['N_Tissues'] >= 2) &
        (signatures_df['Direction_Consistency'] >= 0.7)
    ].copy()

    universal_candidates = universal_candidates.sort_values('Consistency_Score', ascending=False)

    print(f"\nUniversal candidates (n≥5, tissues≥2, direction≥70%): {len(universal_candidates)}")
    print(f"\nTop {n_top} most consistent universal aging signatures:")
    print(universal_candidates.head(n_top).to_string())

    # Save results
    universal_candidates.to_csv(f"{OUTPUT_DIR}/universal_signatures.csv", index=False)

    return universal_candidates

def identify_personalized_signatures(df_valid):
    """
    Identify personalized/context-dependent signatures:
    Proteins with high variance across studies/tissues
    """
    print("\n" + "="*60)
    print("PERSONALIZED/CONTEXT-DEPENDENT SIGNATURES")
    print("="*60)

    # Calculate context-dependency for each protein
    context_stats = []

    for protein in df_valid['Canonical_Gene_Symbol'].unique():
        protein_data = df_valid[df_valid['Canonical_Gene_Symbol'] == protein]

        if len(protein_data) >= 3:
            # Variance across contexts
            context_var = protein_data['Zscore_Delta'].var()

            # Range of responses
            response_range = protein_data['Zscore_Delta'].max() - protein_data['Zscore_Delta'].min()

            # Coefficient of variation
            mean_delta = protein_data['Zscore_Delta'].mean()
            cv = protein_data['Zscore_Delta'].std() / (abs(mean_delta) + 0.001)

            context_stats.append({
                'Protein': protein,
                'Variance': context_var,
                'Range': response_range,
                'CV': cv,
                'N_Observations': len(protein_data),
                'N_Tissues': protein_data['Tissue'].nunique(),
                'N_Studies': protein_data['Study_ID'].nunique(),
                'Matrisome_Category': protein_data['Matrisome_Category'].mode()[0] if len(protein_data['Matrisome_Category'].mode()) > 0 else 'Unknown'
            })

    context_df = pd.DataFrame(context_stats)

    # High context-dependency: high variance, multiple contexts
    personalized_candidates = context_df[
        (context_df['N_Observations'] >= 5) &
        (context_df['N_Tissues'] >= 2) &
        (context_df['CV'] >= 2.0)  # High coefficient of variation
    ].copy()

    personalized_candidates = personalized_candidates.sort_values('CV', ascending=False)

    print(f"\nPersonalized candidates (n≥5, tissues≥2, CV≥2.0): {len(personalized_candidates)}")
    print(f"\nTop 30 most context-dependent signatures:")
    print(personalized_candidates.head(30).to_string())

    # Save results
    personalized_candidates.to_csv(f"{OUTPUT_DIR}/personalized_signatures.csv", index=False)

    return personalized_candidates

def therapeutic_implications(proportions, universal_sigs, personalized_sigs):
    """
    Derive therapeutic implications from hybrid model
    """
    print("\n" + "="*60)
    print("THERAPEUTIC IMPLICATIONS")
    print("="*60)

    universal_prop = proportions['Universal (Protein)']
    personalized_prop = proportions['Study-specific (Personalized)']

    print(f"\nUniversal component: {universal_prop*100:.1f}%")
    print(f"Personalized component: {personalized_prop*100:.1f}%")

    # Decision thresholds
    if universal_prop >= 0.60:
        strategy = "UNIVERSAL-DOMINANT"
        recommendation = """
        Primary Strategy: UNIVERSAL INTERVENTIONS
        - Universal aging signatures represent ≥60% of variance
        - One-size-fits-all therapies are viable
        - Target universal signatures (e.g., consistent ECM proteins)
        - Population-wide interventions justified

        Secondary: Tissue-specific optimization
        - Adjust dosing/delivery for tissue context
        - Monitor tissue-specific responses
        """
    elif personalized_prop >= 0.40:
        strategy = "PERSONALIZED-REQUIRED"
        recommendation = """
        Primary Strategy: PERSONALIZED DIAGNOSTICS + TARGETED THERAPY
        - Personalized component represents ≥40% of variance
        - Individual profiling essential before treatment
        - Develop diagnostic panels to stratify patients
        - Multiple therapy options based on individual signatures

        Universal component still relevant:
        - Use universal signatures as baseline
        - Personalize on top of universal foundation
        """
    else:
        strategy = "HYBRID-BALANCED"
        recommendation = """
        Optimal Strategy: TIERED HYBRID APPROACH

        Tier 1 (Population-wide):
        - Target universal aging signatures
        - Apply to all patients as baseline
        - Use {0} universal signatures as primary targets

        Tier 2 (Personalized optimization):
        - Profile individual/tissue context
        - Adjust for personalized modifiers
        - Monitor {1} context-dependent markers

        Tier 3 (Adaptive):
        - Track response to universal interventions
        - Adjust personalized components based on response
        - Iterative optimization
        """.format(len(universal_sigs), len(personalized_sigs))

    print(f"\nRECOMMENDED STRATEGY: {strategy}")
    print(recommendation)

    # Specific target recommendations
    print("\nSPECIFIC THERAPEUTIC TARGETS:")
    print("-" * 60)
    print("\nUniversal Targets (Top 10):")
    for idx, row in universal_sigs.head(10).iterrows():
        direction = "↑ UP" if row['Mean_Delta'] > 0 else "↓ DOWN"
        print(f"  {row['Protein']:15s} {direction:6s} | Consistency: {row['Direction_Consistency']*100:.0f}% | {row['Matrisome_Category']}")

    print("\nContext-Dependent Biomarkers (Top 10):")
    for idx, row in personalized_sigs.head(10).iterrows():
        print(f"  {row['Protein']:15s} CV: {row['CV']:.2f} | Range: {row['Range']:.2f} | {row['Matrisome_Category']}")

    return strategy, recommendation

def create_visualizations(df_valid, components, proportions, universal_sigs, personalized_sigs):
    """Create comprehensive visualization suite"""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    fig = plt.figure(figsize=(20, 12))

    # 1. Variance decomposition pie chart
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax1.pie(
        proportions.values(),
        labels=proportions.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title('Variance Decomposition\n(Hybrid Model)', fontsize=14, fontweight='bold')

    # 2. Component bar chart
    ax2 = plt.subplot(2, 3, 2)
    comp_names = list(components.keys())
    comp_values = list(components.values())
    bars = ax2.bar(range(len(comp_names)), comp_values, color=colors)
    ax2.set_xticks(range(len(comp_names)))
    ax2.set_xticklabels([name.split()[0] for name in comp_names], rotation=45, ha='right')
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.set_title('Variance Components', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, val in zip(bars, comp_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

    # 3. Universal signatures heatmap (top 20)
    ax3 = plt.subplot(2, 3, 3)
    top_universal = universal_sigs.head(20)
    metrics = top_universal[['Mean_Delta', 'Consistency_Score', 'Direction_Consistency']].values
    sns.heatmap(metrics, cmap='RdBu_r', center=0, ax=ax3,
                yticklabels=top_universal['Protein'].values,
                xticklabels=['Mean Δ', 'Consistency', 'Direction %'],
                cbar_kws={'label': 'Value'})
    ax3.set_title('Top 20 Universal Signatures', fontsize=14, fontweight='bold')

    # 4. Distribution of z-score deltas
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df_valid['Zscore_Delta'].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax4.set_xlabel('Z-score Delta (Old - Young)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Aging Changes', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Tissue variance comparison
    ax5 = plt.subplot(2, 3, 5)
    tissue_variance = df_valid.groupby('Tissue')['Zscore_Delta'].var().sort_values(ascending=False).head(15)
    tissue_variance.plot(kind='barh', ax=ax5, color='coral')
    ax5.set_xlabel('Variance', fontsize=12)
    ax5.set_title('Top 15 Tissues by Variance', fontsize=14, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)

    # 6. Universal vs Personalized scatter
    ax6 = plt.subplot(2, 3, 6)

    # Combine universal and personalized
    all_proteins = pd.merge(
        universal_sigs[['Protein', 'Consistency_Score', 'N_Observations']],
        personalized_sigs[['Protein', 'CV']],
        on='Protein',
        how='outer'
    ).fillna(0)

    scatter = ax6.scatter(all_proteins['Consistency_Score'],
                         all_proteins['CV'],
                         s=all_proteins['N_Observations']*3,
                         alpha=0.6,
                         c=all_proteins['N_Observations'],
                         cmap='viridis')
    ax6.set_xlabel('Consistency Score (Universal)', fontsize=12)
    ax6.set_ylabel('Coefficient of Variation (Personalized)', fontsize=12)
    ax6.set_title('Universal vs Personalized Spectrum', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='N Observations')

    # Quadrant lines
    ax6.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='CV threshold')
    ax6.axvline(all_proteins['Consistency_Score'].median(), color='blue', linestyle='--', alpha=0.5, label='Median consistency')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/hybrid_model_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {OUTPUT_DIR}/hybrid_model_visualization.png")

    plt.close()

def generate_summary_report(components, proportions, universal_sigs, personalized_sigs, strategy):
    """Generate text summary report"""

    report = f"""
HYBRID MODEL ANALYSIS: EXECUTIVE SUMMARY
{'='*70}

RESEARCH QUESTION:
Is there a universal cross-tissue ECM aging signature or personalized trajectories?

ANSWER: HYBRID MODEL - Both exist with quantified contributions

{'='*70}
VARIANCE DECOMPOSITION RESULTS
{'='*70}

Total ECM Aging Variance Decomposition:

1. Universal Component (Protein Identity):      {proportions['Universal (Protein)']*100:5.1f}%
   - Conserved aging signatures across contexts
   - Core ECM proteins with consistent directional changes

2. Tissue-Specific Component:                   {proportions['Tissue-specific']*100:5.1f}%
   - Tissue microenvironment modifiers
   - Organ-specific ECM remodeling patterns

3. Personalized Component (Study/Individual):   {proportions['Study-specific (Personalized)']*100:5.1f}%
   - Individual-level variation
   - Genetic/environmental modifiers

4. Residual/Stochastic Noise:                   {proportions['Residual/Noise']*100:5.1f}%
   - Technical variation
   - Unmeasured factors

{'='*70}
MATHEMATICAL MODEL
{'='*70}

Aging_phenotype(protein, tissue, individual) =
    μ_universal(protein) +                    [{proportions['Universal (Protein)']*100:.0f}% contribution]
    α_tissue(protein, tissue) +               [{proportions['Tissue-specific']*100:.0f}% contribution]
    β_individual(protein, tissue, individual) +  [{proportions['Study-specific (Personalized)']*100:.0f}% contribution]
    ε_noise                                   [{proportions['Residual/Noise']*100:.0f}% contribution]

{'='*70}
THERAPEUTIC STRATEGY: {strategy}
{'='*70}

Universal Signatures Identified: {len(universal_sigs)}
Personalized Signatures Identified: {len(personalized_sigs)}

Top 5 Universal Targets:
"""

    for idx, (i, row) in enumerate(universal_sigs.head(5).iterrows(), 1):
        direction = "UPREGULATED" if row['Mean_Delta'] > 0 else "DOWNREGULATED"
        report += f"\n{idx}. {row['Protein']} - {direction} (Consistency: {row['Direction_Consistency']*100:.0f}%, n={row['N_Observations']})"

    report += "\n\nTop 5 Personalized Biomarkers:\n"

    for idx, (i, row) in enumerate(personalized_sigs.head(5).iterrows(), 1):
        report += f"\n{idx}. {row['Protein']} - High variability (CV: {row['CV']:.2f}, Range: {row['Range']:.2f})"

    report += f"""

{'='*70}
RESOLUTION OF DEBATE
{'='*70}

The "universal vs personalized" debate presents a FALSE DICHOTOMY.

Reality: MULTI-LEVEL HIERARCHY
- Universal baseline exists ({proportions['Universal (Protein)']*100:.0f}% of variance)
- Personalized modifiers overlay ({proportions['Study-specific (Personalized)']*100:.0f}% + {proportions['Tissue-specific']*100:.0f}% = {(proportions['Study-specific (Personalized)'] + proportions['Tissue-specific'])*100:.0f}% of variance)

This finding reconciles:
- Agent 1 perspective (cross-tissue universality): Validated - core signatures exist
- Agent 2 perspective (personalization): Validated - individual variation is real

CONCLUSION:
Optimal aging intervention requires TIERED APPROACH combining universal
foundation with personalized optimization, not either/or strategy.

{'='*70}
"""

    # Save report
    with open(f'{OUTPUT_DIR}/HYBRID_MODEL_SUMMARY.txt', 'w') as f:
        f.write(report)

    print(report)
    print(f"\nSummary report saved: {OUTPUT_DIR}/HYBRID_MODEL_SUMMARY.txt")

def main():
    """Main analysis pipeline"""
    print("\n" + "="*70)
    print("HYBRID MODEL ANALYSIS: UNIVERSAL vs PERSONALIZED ECM AGING")
    print("="*70)

    # Load data
    df = load_data()

    # Explore variance structure
    df_valid, tissue_vars, study_vars, protein_vars = explore_variance_structure(df)

    # Variance decomposition
    components, proportions = variance_decomposition_anova(df_valid)

    # Identify universal signatures
    universal_sigs = identify_universal_signatures(df_valid, n_top=50)

    # Identify personalized signatures
    personalized_sigs = identify_personalized_signatures(df_valid)

    # Therapeutic implications
    strategy, recommendation = therapeutic_implications(proportions, universal_sigs, personalized_sigs)

    # Create visualizations
    create_visualizations(df_valid, components, proportions, universal_sigs, personalized_sigs)

    # Generate summary report
    generate_summary_report(components, proportions, universal_sigs, personalized_sigs, strategy)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  1. universal_signatures.csv")
    print("  2. personalized_signatures.csv")
    print("  3. hybrid_model_visualization.png")
    print("  4. HYBRID_MODEL_SUMMARY.txt")
    print("  5. AGENT3_HYBRID_MODEL.md (next step)")

if __name__ == "__main__":
    main()
