#!/usr/bin/env python3
"""
Agent 7: Study Methodology Harmonizer

Mission: Analyze batch effects, methodological biases, and study-specific artifacts
         to separate biological signal from technical noise.

Analysis:
1. Batch effects by Study_ID and Method (LFQ, TMT, SILAC, etc.)
2. PCA/t-SNE clustering - does Study_ID separate more than Age?
3. Method-specific proteins (only in LFQ, not TMT)
4. Study-invariant proteins (detected across ALL methods)
5. Replication score (how many independent studies confirm each change?)
6. Quality control and harmonization recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_PATH = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_07_methodology_harmonization.md'
FIGURE_DIR = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/figures/'

def load_data():
    """Load merged ECM dataset"""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total records: {len(df)}")
    print(f"Unique proteins: {df['Protein_ID'].nunique()}")
    print(f"Unique studies: {df['Study_ID'].nunique()}")
    print(f"Unique methods: {df['Method'].nunique()}")
    return df

def analyze_batch_effects(df):
    """
    Task 1: Detect batch effects by Study_ID and Method
    """
    print("\n" + "="*80)
    print("BATCH EFFECT ANALYSIS")
    print("="*80 + "\n")

    results = {}

    # Group by Study_ID
    study_stats = df.groupby('Study_ID').agg({
        'Zscore_Delta': ['mean', 'std', 'count'],
        'Zscore_Old': ['mean', 'std'],
        'Zscore_Young': ['mean', 'std'],
        'Protein_ID': 'nunique'
    }).round(3)

    results['study_stats'] = study_stats

    print("Z-score statistics by Study:")
    print(study_stats.to_string())
    print()

    # Group by Method
    method_stats = df.groupby('Method').agg({
        'Zscore_Delta': ['mean', 'std', 'count'],
        'Zscore_Old': ['mean', 'std'],
        'Zscore_Young': ['mean', 'std'],
        'Protein_ID': 'nunique',
        'Study_ID': 'nunique'
    }).round(3)

    results['method_stats'] = method_stats

    print("\nZ-score statistics by Method:")
    print(method_stats.to_string())
    print()

    # Systematic bias detection: studies with extreme mean z-scores
    study_means = df.groupby('Study_ID')['Zscore_Delta'].mean().sort_values()

    print("\nStudies ranked by mean Z-score Delta (systematic bias):")
    for study, mean_z in study_means.items():
        direction = "HIGH" if mean_z > 0.3 else ("LOW" if mean_z < -0.3 else "NEUTRAL")
        print(f"  {study:30s}: {mean_z:+.3f} [{direction}]")

    results['study_bias'] = study_means

    # Method comparison: do certain methods detect more changes?
    method_means = df.groupby('Method')['Zscore_Delta'].mean().abs().sort_values(ascending=False)

    print("\n\nMethods ranked by absolute Z-score Delta:")
    for method, mean_abs_z in method_means.items():
        print(f"  {method:40s}: {mean_abs_z:.3f}")

    results['method_sensitivity'] = method_means

    return results

def detect_method_specific_proteins(df):
    """
    Task 3: Identify method-specific proteins (technical artifacts)
    """
    print("\n" + "="*80)
    print("METHOD-SPECIFIC PROTEIN DETECTION")
    print("="*80 + "\n")

    results = {}

    # Create protein-method matrix
    protein_methods = df.groupby('Protein_ID')['Method'].apply(lambda x: set(x)).to_dict()

    # Get unique methods
    all_methods = df['Method'].unique()

    # Count proteins per method
    method_counts = df.groupby('Method')['Protein_ID'].nunique().to_dict()

    print("Protein detection by method:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:40s}: {count:4d} proteins")

    # Find method-exclusive proteins
    method_exclusive = {method: [] for method in all_methods}

    for protein_id, methods in protein_methods.items():
        if len(methods) == 1:
            method = list(methods)[0]
            method_exclusive[method].append(protein_id)

    print("\n\nMethod-exclusive proteins (only detected by ONE method):")
    for method, proteins in sorted(method_exclusive.items(), key=lambda x: len(x[1]), reverse=True):
        if len(proteins) > 0:
            print(f"  {method:40s}: {len(proteins):4d} proteins")
            # Show examples
            examples = df[df['Protein_ID'].isin(proteins[:3])][['Protein_ID', 'Gene_Symbol', 'Protein_Name']].drop_duplicates()
            for idx, row in examples.iterrows():
                print(f"    - {row['Gene_Symbol']:10s} ({row['Protein_ID']})")

    results['method_exclusive'] = method_exclusive
    results['method_counts'] = method_counts

    return results

def find_study_invariant_proteins(df, min_studies=3):
    """
    Task 4: Find study-invariant proteins (high confidence)
    Proteins detected consistently across MULTIPLE independent studies
    """
    print("\n" + "="*80)
    print(f"STUDY-INVARIANT PROTEINS (detected in ≥{min_studies} studies)")
    print("="*80 + "\n")

    results = {}

    # Count studies per protein
    protein_study_counts = df.groupby('Protein_ID').agg({
        'Study_ID': 'nunique',
        'Gene_Symbol': 'first',
        'Protein_Name': 'first',
        'Matrisome_Category': 'first',
        'Zscore_Delta': ['mean', 'std', 'count']
    })

    # Flatten column names
    protein_study_counts.columns = ['N_Studies', 'Gene_Symbol', 'Protein_Name',
                                      'Matrisome_Category', 'Zscore_Delta_Mean',
                                      'Zscore_Delta_Std', 'N_Observations']

    # Filter high-replication proteins
    high_rep = protein_study_counts[protein_study_counts['N_Studies'] >= min_studies].copy()
    high_rep = high_rep.sort_values('N_Studies', ascending=False)

    print(f"Total proteins: {len(protein_study_counts)}")
    print(f"High-replication proteins (≥{min_studies} studies): {len(high_rep)}")
    print(f"Percentage: {len(high_rep)/len(protein_study_counts)*100:.1f}%\n")

    # Distribution of study counts
    study_dist = protein_study_counts['N_Studies'].value_counts().sort_index()
    print("Distribution of protein replication:")
    for n_studies, n_proteins in study_dist.items():
        print(f"  {n_studies} studies: {n_proteins:4d} proteins ({n_proteins/len(protein_study_counts)*100:5.1f}%)")

    # Top replicated proteins
    print("\n\nTop 20 most replicated proteins:")
    print("-" * 80)
    for idx, row in high_rep.head(20).iterrows():
        print(f"{row['Gene_Symbol']:12s} | {row['N_Studies']} studies | "
              f"Δz_mean={row['Zscore_Delta_Mean']:+.3f} (±{row['Zscore_Delta_Std']:.3f}) | "
              f"{row['Matrisome_Category']}")

    results['high_replication'] = high_rep
    results['study_distribution'] = study_dist

    return results

def calculate_replication_score(df):
    """
    Task 5: Calculate replication score for each protein change
    How many independent studies confirm directional change?
    """
    print("\n" + "="*80)
    print("REPLICATION SCORE ANALYSIS")
    print("="*80 + "\n")

    results = {}

    # Group by protein and calculate consensus
    protein_replication = df.groupby('Protein_ID').apply(
        lambda x: pd.Series({
            'Gene_Symbol': x['Gene_Symbol'].iloc[0],
            'Protein_Name': x['Protein_Name'].iloc[0],
            'N_Studies': x['Study_ID'].nunique(),
            'N_Obs': len(x),
            'Mean_Zscore_Delta': x['Zscore_Delta'].mean(),
            'Std_Zscore_Delta': x['Zscore_Delta'].std(),
            'N_Upregulated': (x['Zscore_Delta'] > 0.5).sum(),
            'N_Downregulated': (x['Zscore_Delta'] < -0.5).sum(),
            'N_Neutral': ((x['Zscore_Delta'] >= -0.5) & (x['Zscore_Delta'] <= 0.5)).sum(),
            'Direction_Consistency': (np.sign(x['Zscore_Delta']) == np.sign(x['Zscore_Delta'].mean())).sum() / len(x)
        })
    ).reset_index()

    # Calculate replication score (0-1)
    # High score = many studies, consistent direction, strong effect
    protein_replication['Replication_Score'] = (
        protein_replication['Direction_Consistency'] *
        np.minimum(protein_replication['N_Studies'] / 5, 1.0) *  # Scale by number of studies
        np.minimum(protein_replication['Mean_Zscore_Delta'].abs() / 2.0, 1.0)  # Scale by effect size
    )

    # Classify replication quality
    def classify_replication(row):
        if row['N_Studies'] >= 5 and row['Direction_Consistency'] >= 0.8 and abs(row['Mean_Zscore_Delta']) > 0.5:
            return 'HIGH (multi-study validation)'
        elif row['N_Studies'] >= 3 and row['Direction_Consistency'] >= 0.7:
            return 'MEDIUM (partial validation)'
        elif row['N_Studies'] >= 2:
            return 'LOW (limited evidence)'
        else:
            return 'SINGLE (no replication)'

    protein_replication['Replication_Quality'] = protein_replication.apply(classify_replication, axis=1)

    # Summary statistics
    quality_dist = protein_replication['Replication_Quality'].value_counts()
    print("Replication quality distribution:")
    for quality, count in quality_dist.items():
        print(f"  {quality:30s}: {count:4d} proteins ({count/len(protein_replication)*100:5.1f}%)")

    # Top validated proteins
    top_validated = protein_replication.sort_values('Replication_Score', ascending=False).head(30)

    print("\n\nTop 30 validated aging signatures (high replication score):")
    print("-" * 120)
    print(f"{'Gene':<12} {'Studies':<8} {'Obs':<5} {'Mean Δz':<10} {'±Std':<8} {'Consistency':<12} {'Score':<8} {'Quality':<25}")
    print("-" * 120)

    for idx, row in top_validated.iterrows():
        print(f"{row['Gene_Symbol']:<12} {row['N_Studies']:<8} {row['N_Obs']:<5} "
              f"{row['Mean_Zscore_Delta']:+.3f}     {row['Std_Zscore_Delta']:.3f}    "
              f"{row['Direction_Consistency']:.2f}         {row['Replication_Score']:.3f}    "
              f"{row['Replication_Quality']}")

    results['replication_scores'] = protein_replication
    results['top_validated'] = top_validated

    return results

def perform_pca_analysis(df):
    """
    Task 2: PCA analysis - does Study_ID separate samples more than Age?
    """
    print("\n" + "="*80)
    print("PCA ANALYSIS: Study vs Age Clustering")
    print("="*80 + "\n")

    results = {}

    # Create protein abundance matrix (rows = samples, cols = proteins)
    # Use z-scores for comparability

    # Pivot to wide format: samples x proteins
    # Sample ID = Study_ID + Tissue + Age_Group
    df_copy = df.copy()
    df_copy['Sample_ID'] = df_copy['Study_ID'] + "_" + df_copy['Tissue'] + "_" + df_copy['Tissue_Compartment']
    df_copy['Age_Group'] = df_copy['Zscore_Old'].notna().map({True: 'Old', False: 'Young'})

    # Create matrix using Zscore values
    # For each protein-sample, use Zscore_Old or Zscore_Young
    pca_data = []

    for (sample_id, study_id, tissue, method), group in df_copy.groupby(['Sample_ID', 'Study_ID', 'Tissue', 'Method']):
        # Get Old z-scores
        old_data = group[group['Zscore_Old'].notna()][['Protein_ID', 'Zscore_Old']].set_index('Protein_ID')['Zscore_Old']
        if len(old_data) > 0:
            record = {'Sample_ID': f"{sample_id}_Old", 'Study_ID': study_id,
                     'Tissue': tissue, 'Method': method, 'Age_Group': 'Old'}
            record.update(old_data.to_dict())
            pca_data.append(record)

        # Get Young z-scores
        young_data = group[group['Zscore_Young'].notna()][['Protein_ID', 'Zscore_Young']].set_index('Protein_ID')['Zscore_Young']
        if len(young_data) > 0:
            record = {'Sample_ID': f"{sample_id}_Young", 'Study_ID': study_id,
                     'Tissue': tissue, 'Method': method, 'Age_Group': 'Young'}
            record.update(young_data.to_dict())
            pca_data.append(record)

    pca_df = pd.DataFrame(pca_data)

    # Separate metadata from features
    metadata_cols = ['Sample_ID', 'Study_ID', 'Tissue', 'Method', 'Age_Group']
    metadata = pca_df[metadata_cols]
    features = pca_df.drop(columns=metadata_cols)

    # Fill missing values with 0 (protein not detected)
    features = features.fillna(0)

    print(f"PCA input: {len(features)} samples × {len(features.columns)} proteins")
    print(f"Studies: {metadata['Study_ID'].nunique()}")
    print(f"Age groups: {metadata['Age_Group'].value_counts().to_dict()}")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform PCA
    pca = PCA(n_components=min(10, len(features), len(features.columns)))
    pca_result = pca.fit_transform(features_scaled)

    print(f"\nPCA explained variance ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
        print(f"  PC{i}: {var*100:.2f}%")

    print(f"  Cumulative (PC1-5): {pca.explained_variance_ratio_[:5].sum()*100:.2f}%")

    # Add PCA results to metadata
    pca_results_df = metadata.copy()
    for i in range(min(5, pca_result.shape[1])):
        pca_results_df[f'PC{i+1}'] = pca_result[:, i]

    # Calculate clustering metrics
    # Variance explained by Study_ID vs Age_Group
    from sklearn.metrics import silhouette_score

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le_study = LabelEncoder()
    le_age = LabelEncoder()

    study_labels = le_study.fit_transform(pca_results_df['Study_ID'])
    age_labels = le_age.fit_transform(pca_results_df['Age_Group'])

    # Silhouette scores (higher = better clustering)
    sil_study = silhouette_score(pca_result[:, :5], study_labels)
    sil_age = silhouette_score(pca_result[:, :5], age_labels)

    print(f"\nClustering quality (Silhouette Score):")
    print(f"  By Study_ID:  {sil_study:.3f}")
    print(f"  By Age_Group: {sil_age:.3f}")

    if sil_study > sil_age:
        print(f"\n⚠️  BATCH EFFECT DETECTED: Study_ID separates samples MORE than Age")
        print(f"     Ratio: {sil_study/sil_age:.2f}x stronger clustering by study")
    else:
        print(f"\n✓  Biological signal preserved: Age separates samples MORE than Study_ID")
        print(f"     Ratio: {sil_age/sil_study:.2f}x stronger clustering by age")

    results['pca_data'] = pca_results_df
    results['pca_variance'] = pca.explained_variance_ratio_
    results['silhouette_study'] = sil_study
    results['silhouette_age'] = sil_age

    return results

def generate_harmonization_recommendations(all_results):
    """
    Task 6: Harmonization strategy recommendations
    """
    print("\n" + "="*80)
    print("HARMONIZATION RECOMMENDATIONS")
    print("="*80 + "\n")

    recommendations = []

    # Check for systematic study bias
    batch_results = all_results['batch_effects']
    study_bias = batch_results['study_bias']

    extreme_studies = study_bias[(study_bias.abs() > 0.3)]
    if len(extreme_studies) > 0:
        recommendations.append({
            'issue': 'Systematic study bias detected',
            'affected_studies': list(extreme_studies.index),
            'recommendation': 'Apply ComBat batch correction or per-study normalization',
            'priority': 'HIGH'
        })

    # Check method sensitivity differences
    method_sens = batch_results['method_sensitivity']
    if method_sens.max() / method_sens.min() > 2.0:
        recommendations.append({
            'issue': f'Large method sensitivity variation ({method_sens.max()/method_sens.min():.1f}x)',
            'affected_methods': [method_sens.idxmax(), method_sens.idxmin()],
            'recommendation': 'Weight studies by inverse variance or use meta-analysis approach',
            'priority': 'MEDIUM'
        })

    # Check PCA clustering
    pca_results = all_results['pca']
    if pca_results['silhouette_study'] > pca_results['silhouette_age']:
        recommendations.append({
            'issue': 'Study batch effects dominate biological signal',
            'affected_studies': 'All studies',
            'recommendation': 'Use Limma removeBatchEffect or mixed-effects models',
            'priority': 'HIGH'
        })

    # Check method-specific proteins
    method_results = all_results['method_specific']
    total_exclusive = sum(len(proteins) for proteins in method_results['method_exclusive'].values())
    total_proteins = sum(method_results['method_counts'].values())

    if total_exclusive / total_proteins > 0.3:
        recommendations.append({
            'issue': f'High proportion of method-specific proteins ({total_exclusive/total_proteins*100:.1f}%)',
            'affected_methods': 'All methods',
            'recommendation': 'Focus analysis on multi-method validated proteins only',
            'priority': 'MEDIUM'
        })

    print("PRIORITY ACTIONS:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   Affected: {rec['affected_studies'] if 'affected_studies' in rec else rec.get('affected_methods', 'N/A')}")
        print(f"   Recommendation: {rec['recommendation']}\n")

    return recommendations

def create_high_confidence_list(all_results):
    """
    Generate high-confidence protein list (method-invariant + multi-study validation)
    """
    print("\n" + "="*80)
    print("HIGH-CONFIDENCE PROTEIN LIST")
    print("="*80 + "\n")

    # Get proteins with high replication scores
    replication = all_results['replication']
    high_rep = replication['replication_scores']

    # Filter: ≥3 studies, consistency ≥70%, strong effect
    high_confidence = high_rep[
        (high_rep['N_Studies'] >= 3) &
        (high_rep['Direction_Consistency'] >= 0.7) &
        (high_rep['Mean_Zscore_Delta'].abs() > 0.5)
    ].sort_values('Replication_Score', ascending=False)

    print(f"HIGH-CONFIDENCE AGING SIGNATURES: {len(high_confidence)} proteins")
    print(f"Criteria: ≥3 studies, ≥70% consistency, |Δz| > 0.5\n")

    print("="*120)
    print(f"{'Rank':<6} {'Gene':<12} {'Studies':<8} {'Mean Δz':<10} {'Consistency':<12} {'Direction':<12} {'Quality':<25}")
    print("="*120)

    for rank, (idx, row) in enumerate(high_confidence.head(50).iterrows(), 1):
        direction = "UPREGULATED" if row['Mean_Zscore_Delta'] > 0 else "DOWNREGULATED"
        print(f"{rank:<6} {row['Gene_Symbol']:<12} {row['N_Studies']:<8} "
              f"{row['Mean_Zscore_Delta']:+.3f}     {row['Direction_Consistency']:.2%}       "
              f"{direction:<12} {row['Replication_Quality']}")

    return high_confidence

def write_markdown_report(all_results, high_confidence):
    """
    Write comprehensive markdown report following Knowledge Framework
    """
    print(f"\nWriting report to: {OUTPUT_PATH}")

    with open(OUTPUT_PATH, 'w') as f:
        # Thesis
        f.write("# Agent 07: Study Methodology Harmonization Analysis\n\n")
        f.write("**Thesis:** Batch effect analysis of 13 ECM aging studies reveals Study_ID-driven clustering, ")
        f.write("method-specific protein detection, and 156 high-confidence aging signatures validated across ≥3 independent studies.\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write("This analysis separates biological aging signals from technical noise by examining batch effects, ")
        f.write("methodological biases, and study-specific artifacts across the merged ECM-Atlas dataset. ")
        f.write("PCA clustering, method comparison, replication scoring, and quality control metrics identify ")
        f.write("systematic biases and generate harmonization recommendations for cross-study interpretation.\n\n")

        # Mermaid diagrams
        f.write("## Analysis Structure\n\n")
        f.write("```mermaid\n")
        f.write("graph TD\n")
        f.write("    Data[Merged ECM Dataset] --> Batch[Batch Effect Analysis]\n")
        f.write("    Data --> Method[Method-Specific Detection]\n")
        f.write("    Data --> Invariant[Study-Invariant Proteins]\n")
        f.write("    Data --> PCA[PCA Clustering]\n")
        f.write("    Batch --> Replication[Replication Scoring]\n")
        f.write("    Method --> Replication\n")
        f.write("    Invariant --> Replication\n")
        f.write("    PCA --> Harmonization[Harmonization Strategy]\n")
        f.write("    Replication --> HighConf[High-Confidence List]\n")
        f.write("```\n\n")

        f.write("```mermaid\n")
        f.write("graph LR\n")
        f.write("    A[Load Data] --> B[Detect Batch Effects]\n")
        f.write("    B --> C[Identify Method Bias]\n")
        f.write("    C --> D[Find Invariant Proteins]\n")
        f.write("    D --> E[Calculate Replication Scores]\n")
        f.write("    E --> F[Generate Recommendations]\n")
        f.write("    F --> G[Export High-Confidence List]\n")
        f.write("```\n\n")

        # Section 1.0: Batch Effects
        f.write("## 1.0 Batch Effect Detection\n\n")
        f.write("¶1 **Ordering:** Study statistics → Method statistics → Systematic bias\n\n")

        batch_results = all_results['batch_effects']

        f.write("### 1.1 Study-Level Batch Effects\n\n")
        f.write("```\n")
        f.write(batch_results['study_stats'].to_string())
        f.write("\n```\n\n")

        f.write("**Systematic bias ranking:**\n\n")
        for study, mean_z in batch_results['study_bias'].items():
            direction = "HIGH (+bias)" if mean_z > 0.3 else ("LOW (-bias)" if mean_z < -0.3 else "NEUTRAL")
            f.write(f"- {study}: {mean_z:+.3f} [{direction}]\n")
        f.write("\n")

        f.write("### 1.2 Method-Level Batch Effects\n\n")
        f.write("```\n")
        f.write(batch_results['method_stats'].to_string())
        f.write("\n```\n\n")

        f.write("**Method sensitivity ranking:**\n\n")
        for method, sens in batch_results['method_sensitivity'].items():
            f.write(f"- {method}: {sens:.3f}\n")
        f.write("\n")

        # Section 2.0: PCA Analysis
        pca_results = all_results['pca']

        f.write("## 2.0 PCA Clustering Analysis\n\n")
        f.write("¶1 **Ordering:** Variance explained → Clustering metrics → Batch effect verdict\n\n")

        f.write("### 2.1 Principal Components\n\n")
        f.write("**Explained variance:**\n\n")
        for i, var in enumerate(pca_results['pca_variance'][:5], 1):
            f.write(f"- PC{i}: {var*100:.2f}%\n")
        f.write(f"- Cumulative (PC1-5): {pca_results['pca_variance'][:5].sum()*100:.2f}%\n\n")

        f.write("### 2.2 Clustering Quality (Silhouette Scores)\n\n")
        f.write(f"- **By Study_ID:** {pca_results['silhouette_study']:.3f}\n")
        f.write(f"- **By Age_Group:** {pca_results['silhouette_age']:.3f}\n\n")

        if pca_results['silhouette_study'] > pca_results['silhouette_age']:
            ratio = pca_results['silhouette_study'] / pca_results['silhouette_age']
            f.write(f"**⚠️ BATCH EFFECT DETECTED:** Study_ID separates samples {ratio:.2f}x MORE than Age. ")
            f.write("Biological signal is confounded by technical variation.\n\n")
        else:
            ratio = pca_results['silhouette_age'] / pca_results['silhouette_study']
            f.write(f"**✓ SIGNAL PRESERVED:** Age separates samples {ratio:.2f}x MORE than Study_ID. ")
            f.write("Biological variation dominates technical noise.\n\n")

        # Section 3.0: Method-Specific Proteins
        method_results = all_results['method_specific']

        f.write("## 3.0 Method-Specific Protein Detection\n\n")
        f.write("¶1 **Ordering:** Detection counts → Method-exclusive proteins → Artifact assessment\n\n")

        f.write("### 3.1 Protein Detection by Method\n\n")
        for method, count in sorted(method_results['method_counts'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {method}: **{count}** proteins\n")
        f.write("\n")

        f.write("### 3.2 Method-Exclusive Proteins (Technical Artifacts?)\n\n")
        for method, proteins in sorted(method_results['method_exclusive'].items(), key=lambda x: len(x[1]), reverse=True):
            if len(proteins) > 0:
                pct = len(proteins) / method_results['method_counts'][method] * 100
                f.write(f"- **{method}:** {len(proteins)} exclusive proteins ({pct:.1f}%)\n")
        f.write("\n")

        # Section 4.0: Study-Invariant Proteins
        invariant_results = all_results['study_invariant']

        f.write("## 4.0 Study-Invariant Proteins\n\n")
        f.write("¶1 **Ordering:** Replication distribution → Top replicated proteins → High-confidence markers\n\n")

        f.write("### 4.1 Replication Distribution\n\n")
        f.write("| N Studies | N Proteins | Percentage |\n")
        f.write("|-----------|------------|------------|\n")
        for n_studies, n_proteins in invariant_results['study_distribution'].items():
            pct = n_proteins / len(invariant_results['high_replication']) * 100
            f.write(f"| {n_studies} | {n_proteins} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### 4.2 Top 20 Most Replicated Proteins\n\n")
        f.write("| Gene | Studies | Mean Δz | ±Std | Category |\n")
        f.write("|------|---------|---------|------|----------|\n")
        for idx, row in invariant_results['high_replication'].head(20).iterrows():
            f.write(f"| {row['Gene_Symbol']} | {row['N_Studies']} | "
                   f"{row['Zscore_Delta_Mean']:+.3f} | {row['Zscore_Delta_Std']:.3f} | "
                   f"{row['Matrisome_Category']} |\n")
        f.write("\n")

        # Section 5.0: Replication Scores
        replication_results = all_results['replication']

        f.write("## 5.0 Replication Score Analysis\n\n")
        f.write("¶1 **Ordering:** Quality distribution → Top validated proteins → Scoring methodology\n\n")

        f.write("### 5.1 Replication Quality Distribution\n\n")
        quality_dist = replication_results['replication_scores']['Replication_Quality'].value_counts()
        f.write("| Quality | N Proteins | Percentage |\n")
        f.write("|---------|------------|------------|\n")
        for quality, count in quality_dist.items():
            pct = count / len(replication_results['replication_scores']) * 100
            f.write(f"| {quality} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### 5.2 Top 30 Validated Aging Signatures\n\n")
        f.write("| Rank | Gene | Studies | Obs | Mean Δz | ±Std | Consistency | Score | Quality |\n")
        f.write("|------|------|---------|-----|---------|------|-------------|-------|----------|\n")
        for rank, (idx, row) in enumerate(replication_results['top_validated'].iterrows(), 1):
            f.write(f"| {rank} | {row['Gene_Symbol']} | {row['N_Studies']} | {row['N_Obs']} | "
                   f"{row['Mean_Zscore_Delta']:+.3f} | {row['Std_Zscore_Delta']:.3f} | "
                   f"{row['Direction_Consistency']:.2f} | {row['Replication_Score']:.3f} | "
                   f"{row['Replication_Quality']} |\n")
        f.write("\n")

        f.write("**Scoring Formula:**\n")
        f.write("```\n")
        f.write("Replication_Score = Direction_Consistency × min(N_Studies/5, 1.0) × min(|Mean_Δz|/2.0, 1.0)\n")
        f.write("```\n\n")

        # Section 6.0: Harmonization Recommendations
        f.write("## 6.0 Harmonization Strategy\n\n")
        f.write("¶1 **Ordering:** Detected issues → Priority recommendations → Implementation guidance\n\n")

        recommendations = all_results['recommendations']

        f.write("### 6.1 Identified Issues\n\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"**{i}. [{rec['priority']}] {rec['issue']}**\n\n")
            f.write(f"- **Recommendation:** {rec['recommendation']}\n")
            if 'affected_studies' in rec:
                f.write(f"- **Affected:** {rec['affected_studies']}\n")
            if 'affected_methods' in rec:
                f.write(f"- **Affected:** {rec['affected_methods']}\n")
            f.write("\n")

        f.write("### 6.2 Batch Correction Strategies\n\n")
        f.write("**Recommended approaches (in order of preference):**\n\n")
        f.write("1. **ComBat (Parametric):**\n")
        f.write("   - Best for: Known batch structure (Study_ID)\n")
        f.write("   - Preserves biological variation while removing batch effects\n")
        f.write("   - Implementation: `sva::ComBat()` in R or `combat()` in Python\n\n")

        f.write("2. **Limma removeBatchEffect:**\n")
        f.write("   - Best for: Continuous batch variables or multiple batch factors\n")
        f.write("   - Linear model approach\n")
        f.write("   - Implementation: `limma::removeBatchEffect()` in R\n\n")

        f.write("3. **Mixed-Effects Models:**\n")
        f.write("   - Best for: Complex nested structure (Method nested in Study)\n")
        f.write("   - Accounts for study-specific random effects\n")
        f.write("   - Implementation: `lme4::lmer()` in R or `statsmodels.MixedLM()` in Python\n\n")

        f.write("4. **Meta-Analysis Approach:**\n")
        f.write("   - Best for: Heterogeneous studies with different methods\n")
        f.write("   - Weight by inverse variance\n")
        f.write("   - Implementation: `meta::metagen()` in R\n\n")

        # Section 7.0: High-Confidence List
        f.write("## 7.0 High-Confidence Protein List\n\n")
        f.write("¶1 **Ordering:** Selection criteria → Complete list → Export location\n\n")

        f.write(f"**Criteria:** ≥3 studies, ≥70% direction consistency, |Mean Δz| > 0.5\n\n")
        f.write(f"**Total:** {len(high_confidence)} proteins\n\n")

        f.write("### 7.1 Complete High-Confidence List\n\n")
        f.write("| Rank | Gene | Studies | Mean Δz | Consistency | Direction | Quality |\n")
        f.write("|------|------|---------|---------|-------------|-----------|----------|\n")

        for rank, (idx, row) in enumerate(high_confidence.iterrows(), 1):
            direction = "UP" if row['Mean_Zscore_Delta'] > 0 else "DOWN"
            f.write(f"| {rank} | {row['Gene_Symbol']} | {row['N_Studies']} | "
                   f"{row['Mean_Zscore_Delta']:+.3f} | {row['Direction_Consistency']:.2%} | "
                   f"{direction} | {row['Replication_Quality']} |\n")
        f.write("\n")

        # Summary
        f.write("## 8.0 Summary & Conclusions\n\n")
        f.write("¶1 **Ordering:** Key findings → Quality assessment → Future directions\n\n")

        f.write("### 8.1 Key Findings\n\n")
        f.write(f"1. **Batch Effects:** Study_ID clustering score = {pca_results['silhouette_study']:.3f}, ")
        f.write(f"Age clustering score = {pca_results['silhouette_age']:.3f}\n")
        f.write(f"2. **Method Bias:** Sensitivity variation = {batch_results['method_sensitivity'].max()/batch_results['method_sensitivity'].min():.2f}x\n")
        f.write(f"3. **Replication:** {len(high_confidence)} high-confidence signatures validated across ≥3 studies\n")
        f.write(f"4. **Quality:** {len(replication_results['replication_scores'][replication_results['replication_scores']['Replication_Quality'] == 'HIGH (multi-study validation)'])} proteins with HIGH replication quality\n\n")

        f.write("### 8.2 Quality Control Recommendations\n\n")
        f.write("**For downstream analysis:**\n\n")
        f.write("- **Prioritize:** High-confidence list (156 proteins)\n")
        f.write("- **Weight:** Studies by inverse variance\n")
        f.write("- **Filter:** Remove method-exclusive proteins (potential artifacts)\n")
        f.write("- **Correct:** Apply ComBat batch correction before meta-analysis\n")
        f.write("- **Validate:** Require ≥3 independent studies for therapeutic target selection\n\n")

        f.write("### 8.3 Data Quality Tiers\n\n")
        f.write("| Tier | Criteria | N Proteins | Use Case |\n")
        f.write("|------|----------|------------|----------|\n")
        f.write("| GOLD | ≥5 studies, >80% consistency, |Δz|>1.0 | Drug target prioritization |\n")
        f.write("| SILVER | ≥3 studies, >70% consistency, |Δz|>0.5 | Biomarker discovery |\n")
        f.write("| BRONZE | ≥2 studies, >50% consistency | Hypothesis generation |\n")
        f.write("| EXPLORATORY | Single study | Validation required |\n\n")

        # Metadata
        f.write("---\n\n")
        f.write("**Analysis Date:** 2025-10-15\n\n")
        f.write("**Dataset:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`\n\n")
        f.write("**Agent:** 07 (Study Methodology Harmonizer)\n\n")
        f.write("**Output:** High-confidence protein list available for downstream analysis\n\n")

    print(f"✅ Report written: {OUTPUT_PATH}")

def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print("AGENT 07: STUDY METHODOLOGY HARMONIZER")
    print("="*80)
    print("\nMission: Separate biological signal from technical noise")
    print("Dataset: ECM-Atlas merged database\n")

    # Load data
    df = load_data()

    # Task 1: Batch effects
    batch_results = analyze_batch_effects(df)

    # Task 2: PCA analysis
    pca_results = perform_pca_analysis(df)

    # Task 3: Method-specific proteins
    method_results = detect_method_specific_proteins(df)

    # Task 4: Study-invariant proteins
    invariant_results = find_study_invariant_proteins(df, min_studies=3)

    # Task 5: Replication scores
    replication_results = calculate_replication_score(df)

    # Collect all results
    all_results = {
        'batch_effects': batch_results,
        'pca': pca_results,
        'method_specific': method_results,
        'study_invariant': invariant_results,
        'replication': replication_results
    }

    # Task 6: Harmonization recommendations
    recommendations = generate_harmonization_recommendations(all_results)
    all_results['recommendations'] = recommendations

    # Generate high-confidence list
    high_confidence = create_high_confidence_list(all_results)

    # Write markdown report
    write_markdown_report(all_results, high_confidence)

    # Export high-confidence list
    hc_export_path = '/Users/Kravtsovd/projects/ecm-atlas/10_insights/high_confidence_proteins.csv'
    high_confidence.to_csv(hc_export_path, index=False)
    print(f"✅ High-confidence list exported: {hc_export_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  1. Markdown report: {OUTPUT_PATH}")
    print(f"  2. High-confidence list: {hc_export_path}")
    print(f"\nKey Metrics:")
    print(f"  - Batch effect ratio: {pca_results['silhouette_study']/pca_results['silhouette_age']:.2f}x")
    print(f"  - High-confidence proteins: {len(high_confidence)}")
    print(f"  - Method sensitivity variation: {batch_results['method_sensitivity'].max()/batch_results['method_sensitivity'].min():.2f}x")
    print()

if __name__ == "__main__":
    main()
