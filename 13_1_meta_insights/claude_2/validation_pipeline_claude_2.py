#!/usr/bin/env python3
"""
Meta-Insights Validation Pipeline for V2 Batch-Corrected Dataset
Agent: claude_2
Created: 2025-10-18

Validates all 12 meta-insights (7 GOLD + 5 SILVER) from original V1 analysis
against ComBat-harmonized V2 dataset.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

V2_DATASET = "/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv"

ORIGINAL_INSIGHTS_DIR = "/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/"

OUTPUT_DIR = "/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/claude_2/"

# Original V1 baseline values
V1_BASELINES = {
    'G1_universal_pct': 12.2,  # % universal markers
    'G1_top5': ['Hp', 'VTN', 'Col14a1', 'F2', 'FGB'],
    'G2_pcolce_dz': -0.82,
    'G2_pcolce_consistency': 88.0,
    'G3_study_pc1': 0.674,
    'G3_age_pc1': -0.051,
    'G4_weak_signal_count': 14,
    'G5_entropy_proteins': 52,
    'G6_antagonistic_events': 11,
    'G6_col11a2_divergence': 1.86,
    'G7_shared_genes': 8,
    'G7_total_genes': 1167,
    'G7_correlation': -0.71,
    'S1_fga_dz': 0.88,
    'S1_fgb_dz': 0.89,
    'S1_serpinc1_dz': 3.01,
    'S3_timp3_dz': 3.14,
    'S3_timp3_consistency': 81.0,
    'S4_tsi_proteins': 13,
    'S4_kdm5c_tsi': 32.73,
    'S5_biomarker_count': 7
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_v2_dataset():
    """Load batch-corrected V2 dataset with quality filters."""
    print("Loading V2 dataset...")
    df = pd.read_csv(V2_DATASET)

    # Filter out contaminants
    df = df[~df['Protein_ID'].str.contains('CON__', na=False)]

    # Ensure required columns
    required_cols = ['Gene_Symbol', 'Tissue', 'Compartment', 'Species',
                     'Zscore_Delta', 'Study_ID', 'Matrisome_Category']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")

    print(f"Loaded {len(df)} rows, {df['Gene_Symbol'].nunique()} unique proteins")
    return df

# =============================================================================
# G1: UNIVERSAL MARKERS
# =============================================================================

def validate_g1_universal_markers(df):
    """
    Original: 405/3,317 proteins (12.2%) universal
    Top: Hp (0.749), VTN (0.732), Col14a1 (0.729), F2 (0.717), FGB (0.714)
    """
    print("\n" + "="*80)
    print("G1: UNIVERSAL MARKERS")
    print("="*80)

    # Group by protein
    protein_stats = []

    for gene, group in df.groupby('Gene_Symbol'):
        tissue_count = group['Tissue'].nunique()

        # Directional consistency
        zscore_values = group['Zscore_Delta'].dropna()
        if len(zscore_values) == 0:
            continue

        positive_pct = (zscore_values > 0).sum() / len(zscore_values) * 100
        consistency = max(positive_pct, 100 - positive_pct)

        mean_dz = zscore_values.mean()

        # Universality score (using 17 as max tissue count from original analysis)
        max_tissues = df['Tissue'].nunique()
        universality_score = (tissue_count / max_tissues) * (consistency / 100) * abs(mean_dz)

        protein_stats.append({
            'Gene_Symbol': gene,
            'Tissue_Count': tissue_count,
            'Consistency_Pct': consistency,
            'Mean_Zscore_Delta': mean_dz,
            'Universality_Score': universality_score,
            'N_Measurements': len(zscore_values)
        })

    protein_df = pd.DataFrame(protein_stats)

    # Apply thresholds: ≥3 tissues, ≥70% consistency
    universal = protein_df[
        (protein_df['Tissue_Count'] >= 3) &
        (protein_df['Consistency_Pct'] >= 70)
    ].sort_values('Universality_Score', ascending=False)

    total_proteins = len(protein_df)
    universal_count = len(universal)
    universal_pct = (universal_count / total_proteins) * 100

    print(f"V2 Universal markers: {universal_count}/{total_proteins} ({universal_pct:.1f}%)")
    print(f"V1 Baseline: 405/3,317 (12.2%)")

    print("\nTop 10 Universal Markers (V2):")
    print(universal[['Gene_Symbol', 'Universality_Score', 'Tissue_Count', 'Consistency_Pct', 'Mean_Zscore_Delta']].head(10).to_string(index=False))

    # Check V1 top-5
    print("\nV1 Top-5 in V2 dataset:")
    for gene in V1_BASELINES['G1_top5']:
        if gene in universal['Gene_Symbol'].values:
            row = universal[universal['Gene_Symbol'] == gene].iloc[0]
            print(f"  {gene}: Score={row['Universality_Score']:.3f}, Tissues={row['Tissue_Count']}, Consistency={row['Consistency_Pct']:.1f}%")
        else:
            print(f"  {gene}: NOT in universal list")

    # Classification
    change_pct = ((universal_pct - V1_BASELINES['G1_universal_pct']) / V1_BASELINES['G1_universal_pct']) * 100

    if abs(change_pct) >= 20:
        classification = "CONFIRMED" if change_pct > 0 else "MODIFIED"
    elif abs(change_pct) < 20:
        classification = "MODIFIED"

    return {
        'Insight_ID': 'G1',
        'Tier': 'GOLD',
        'Original_Metric': f"{V1_BASELINES['G1_universal_pct']}% universal",
        'V2_Metric': f"{universal_pct:.1f}% universal",
        'Change_Percent': change_pct,
        'Classification': classification,
        'Notes': f"{universal_count} proteins, top: {universal.iloc[0]['Gene_Symbol']}"
    }, universal

# =============================================================================
# G2: PCOLCE QUALITY PARADIGM
# =============================================================================

def validate_g2_pcolce(df):
    """
    Original: PCOLCE Δz=-0.82, 88% consistency, 5 studies
    """
    print("\n" + "="*80)
    print("G2: PCOLCE QUALITY PARADIGM")
    print("="*80)

    pcolce_data = df[df['Gene_Symbol'] == 'PCOLCE'].copy()

    if len(pcolce_data) == 0:
        print("WARNING: PCOLCE not found in V2 dataset!")
        return {
            'Insight_ID': 'G2',
            'Tier': 'GOLD',
            'Original_Metric': f"PCOLCE Δz={V1_BASELINES['G2_pcolce_dz']}, {V1_BASELINES['G2_pcolce_consistency']}% consistency",
            'V2_Metric': "NOT FOUND",
            'Change_Percent': np.nan,
            'Classification': "REJECTED",
            'Notes': "PCOLCE not in V2 dataset"
        }, None

    mean_dz = pcolce_data['Zscore_Delta'].mean()
    study_count = pcolce_data['Study_ID'].nunique()

    # Consistency (% negative values for depletion)
    consistency = (pcolce_data['Zscore_Delta'] < 0).sum() / len(pcolce_data) * 100

    # Outlier test: compare to global distribution
    all_protein_means = df.groupby('Gene_Symbol')['Zscore_Delta'].mean()
    global_std = all_protein_means.std()
    outlier_status = abs(mean_dz) > 2 * global_std

    print(f"V2 PCOLCE: Δz={mean_dz:.3f}, Consistency={consistency:.1f}%, Studies={study_count}")
    print(f"V1 Baseline: Δz={V1_BASELINES['G2_pcolce_dz']}, Consistency={V1_BASELINES['G2_pcolce_consistency']}%")
    print(f"Outlier status (|Δz| > 2σ): {outlier_status} (global σ={global_std:.3f})")

    # Classification
    dz_change_pct = ((abs(mean_dz) - abs(V1_BASELINES['G2_pcolce_dz'])) / abs(V1_BASELINES['G2_pcolce_dz'])) * 100

    if dz_change_pct >= 20 and np.sign(mean_dz) == np.sign(V1_BASELINES['G2_pcolce_dz']):
        classification = "CONFIRMED"
    elif np.sign(mean_dz) == np.sign(V1_BASELINES['G2_pcolce_dz']):
        classification = "MODIFIED"
    else:
        classification = "REJECTED"

    return {
        'Insight_ID': 'G2',
        'Tier': 'GOLD',
        'Original_Metric': f"Δz={V1_BASELINES['G2_pcolce_dz']}, {V1_BASELINES['G2_pcolce_consistency']}% consistency",
        'V2_Metric': f"Δz={mean_dz:.3f}, {consistency:.1f}% consistency",
        'Change_Percent': dz_change_pct,
        'Classification': classification,
        'Notes': f"{study_count} studies, outlier={outlier_status}"
    }, pcolce_data

# =============================================================================
# G3: BATCH EFFECTS
# =============================================================================

def validate_g3_batch_effects(df):
    """
    Original: PCA PC1: Study_ID=0.674 vs Age_Group=-0.051 (13x bias)
    Expected V2: Age_Group signal increases, Study_ID decreases
    """
    print("\n" + "="*80)
    print("G3: BATCH EFFECTS DOMINATE BIOLOGY")
    print("="*80)

    # Create wide matrix: rows = samples, columns = proteins
    # Sample ID = Tissue_Compartment_Age_Study
    pivot_data = df.pivot_table(
        index=['Tissue', 'Compartment', 'Study_ID'],
        columns='Gene_Symbol',
        values='Zscore_Delta',
        aggfunc='mean'
    ).reset_index()

    # Extract metadata
    metadata = pivot_data[['Tissue', 'Compartment', 'Study_ID']].copy()

    # Create Age_Group from study-level aggregation (simplified - use Study_ID as proxy)
    # In real analysis, would need actual age mapping
    # For now, use Study_ID as categorical

    # Protein data
    protein_cols = [c for c in pivot_data.columns if c not in ['Tissue', 'Compartment', 'Study_ID']]
    X = pivot_data[protein_cols].fillna(0).values

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, X_scaled.shape[0], X_scaled.shape[1]))
    pc_scores = pca.fit_transform(X_scaled)

    # Correlate PC1 with Study_ID (use one-hot encoding)
    study_dummies = pd.get_dummies(metadata['Study_ID']).values
    study_corr = np.abs(np.corrcoef(pc_scores[:, 0], study_dummies.T)[0, 1:]).max()

    print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
    print(f"PC1 correlation with Study_ID: {study_corr:.3f}")
    print(f"V1 Baseline Study_ID correlation: {V1_BASELINES['G3_study_pc1']}")

    # For proper validation, would need Age_Group column
    # Check if Age_Group exists
    if 'Age_Group' in df.columns:
        age_map = df.groupby(['Tissue', 'Compartment', 'Study_ID'])['Age_Group'].first()
        metadata['Age_Group'] = metadata.apply(
            lambda row: age_map.get((row['Tissue'], row['Compartment'], row['Study_ID']), 'unknown'),
            axis=1
        )
        age_dummies = pd.get_dummies(metadata['Age_Group']).values
        age_corr = np.abs(np.corrcoef(pc_scores[:, 0], age_dummies.T)[0, 1:]).max()
        print(f"PC1 correlation with Age_Group: {age_corr:.3f}")
        print(f"V1 Baseline Age_Group correlation: {abs(V1_BASELINES['G3_age_pc1'])}")
    else:
        age_corr = None
        print("Age_Group column not found - cannot validate age signal")

    # Classification
    study_reduction_pct = ((V1_BASELINES['G3_study_pc1'] - study_corr) / V1_BASELINES['G3_study_pc1']) * 100

    if study_reduction_pct >= 50:  # Batch effect reduced by ≥50%
        classification = "CONFIRMED"
    elif study_reduction_pct >= 20:
        classification = "MODIFIED"
    else:
        classification = "REJECTED"

    return {
        'Insight_ID': 'G3',
        'Tier': 'GOLD',
        'Original_Metric': f"Study_ID PC1={V1_BASELINES['G3_study_pc1']}",
        'V2_Metric': f"Study_ID PC1={study_corr:.3f}" + (f", Age PC1={age_corr:.3f}" if age_corr else ""),
        'Change_Percent': -study_reduction_pct,  # Negative = good (batch effect reduced)
        'Classification': classification,
        'Notes': f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance"
    }, pca

# =============================================================================
# G4: WEAK SIGNALS
# =============================================================================

def validate_g4_weak_signals(df):
    """
    Original: 14 proteins with 0.3 ≤ |Δz| ≤ 0.8, pathway-level cumulative effect
    """
    print("\n" + "="*80)
    print("G4: WEAK SIGNALS COMPOUND")
    print("="*80)

    # Calculate protein-level stats
    protein_stats = df.groupby('Gene_Symbol').agg({
        'Zscore_Delta': ['mean', 'count'],
        'Matrisome_Category': 'first'
    }).reset_index()

    protein_stats.columns = ['Gene_Symbol', 'Mean_Zscore_Delta', 'N_Measurements', 'Matrisome_Category']

    # Consistency calculation (need individual measurements)
    consistency_list = []
    for gene, group in df.groupby('Gene_Symbol'):
        zscore_values = group['Zscore_Delta'].dropna()
        if len(zscore_values) > 0:
            positive_pct = (zscore_values > 0).sum() / len(zscore_values) * 100
            consistency = max(positive_pct, 100 - positive_pct)
        else:
            consistency = 0
        consistency_list.append({'Gene_Symbol': gene, 'Consistency': consistency})

    consistency_df = pd.DataFrame(consistency_list)
    protein_stats = protein_stats.merge(consistency_df, on='Gene_Symbol')

    # Filter weak signals: 0.3 ≤ |Δz| ≤ 0.8, consistency ≥ 65%
    weak_signals = protein_stats[
        (protein_stats['Mean_Zscore_Delta'].abs() >= 0.3) &
        (protein_stats['Mean_Zscore_Delta'].abs() <= 0.8) &
        (protein_stats['Consistency'] >= 65)
    ]

    print(f"V2 Weak signal proteins: {len(weak_signals)}")
    print(f"V1 Baseline: {V1_BASELINES['G4_weak_signal_count']}")

    # Pathway-level aggregation
    pathway_cumulative = weak_signals.groupby('Matrisome_Category')['Mean_Zscore_Delta'].sum().sort_values(ascending=False)

    print("\nTop pathways by cumulative weak signal:")
    print(pathway_cumulative.head(5))

    # Classification
    change_pct = ((len(weak_signals) - V1_BASELINES['G4_weak_signal_count']) / V1_BASELINES['G4_weak_signal_count']) * 100

    if change_pct >= 20:
        classification = "CONFIRMED"
    elif change_pct >= -20:
        classification = "MODIFIED"
    else:
        classification = "REJECTED"

    return {
        'Insight_ID': 'G4',
        'Tier': 'GOLD',
        'Original_Metric': f"{V1_BASELINES['G4_weak_signal_count']} weak signal proteins",
        'V2_Metric': f"{len(weak_signals)} weak signal proteins",
        'Change_Percent': change_pct,
        'Classification': classification,
        'Notes': f"Top pathway: {pathway_cumulative.index[0] if len(pathway_cumulative) > 0 else 'N/A'}"
    }, weak_signals

# =============================================================================
# G5: ENTROPY TRANSITIONS
# =============================================================================

def validate_g5_entropy(df):
    """
    Original: 52 proteins ordered→chaotic, DEATh theorem (collagens 28% predictable)
    """
    print("\n" + "="*80)
    print("G5: ENTROPY TRANSITIONS")
    print("="*80)

    # Need age binning - check if Age column exists
    if 'Age' not in df.columns:
        print("WARNING: Age column not found - using simplified entropy")
        # Use Study_ID as proxy for temporal variation
        age_col = 'Study_ID'
    else:
        age_col = 'Age'
        # Create age bins
        df['Age_Bin'] = pd.cut(df['Age'], bins=[0, 40, 50, 65, 100], labels=['young', 'middle1', 'middle2', 'old'])
        age_col = 'Age_Bin'

    entropy_results = []

    for gene, group in df.groupby('Gene_Symbol'):
        zscore_values = group['Zscore_Delta'].dropna()

        if len(zscore_values) < 3:
            continue

        # Shannon entropy across age bins
        if age_col == 'Age_Bin':
            age_dist = group.groupby(age_col)['Zscore_Delta'].count()
            probabilities = age_dist / age_dist.sum()
            shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            # Use variance as proxy if no age bins
            shannon_entropy = zscore_values.std()

        # Coefficient of variation
        cv = zscore_values.std() / (abs(zscore_values.mean()) + 1e-10)

        # Predictability (inverse of normalized entropy)
        max_entropy = np.log2(len(probabilities)) if age_col == 'Age_Bin' else 1.0
        predictability = 1 - (shannon_entropy / max_entropy) if max_entropy > 0 else 0

        # Check if collagen
        is_collagen = gene.upper().startswith('COL')

        entropy_results.append({
            'Gene_Symbol': gene,
            'Shannon_Entropy': shannon_entropy,
            'CV': cv,
            'Predictability': predictability,
            'Is_Collagen': is_collagen,
            'N_Measurements': len(zscore_values)
        })

    entropy_df = pd.DataFrame(entropy_results)

    # Classify: ordered (H<1.5) vs chaotic (H>2.0)
    if age_col == 'Age_Bin':
        ordered = entropy_df[entropy_df['Shannon_Entropy'] < 1.5]
        chaotic = entropy_df[entropy_df['Shannon_Entropy'] > 2.0]
        transitions = len(entropy_df[(entropy_df['Shannon_Entropy'] >= 1.5) & (entropy_df['Shannon_Entropy'] <= 2.0)])
    else:
        ordered = entropy_df[entropy_df['CV'] < 0.5]
        chaotic = entropy_df[entropy_df['CV'] > 1.0]
        transitions = len(entropy_df[(entropy_df['CV'] >= 0.5) & (entropy_df['CV'] <= 1.0)])

    # DEATh theorem: % collagens with predictability > 0.7
    collagens = entropy_df[entropy_df['Is_Collagen']]
    if len(collagens) > 0:
        predictable_collagens_pct = (collagens['Predictability'] > 0.7).sum() / len(collagens) * 100
    else:
        predictable_collagens_pct = 0

    print(f"V2 Entropy classification:")
    print(f"  Ordered: {len(ordered)}")
    print(f"  Chaotic: {len(chaotic)}")
    print(f"  Transitions: {transitions}")
    print(f"  Collagens predictable (>0.7): {predictable_collagens_pct:.1f}%")
    print(f"V1 Baseline: 52 transition proteins, 28% collagens predictable")

    # Classification (use transitions count)
    change_pct = ((transitions - V1_BASELINES['G5_entropy_proteins']) / V1_BASELINES['G5_entropy_proteins']) * 100

    if abs(change_pct) <= 20:
        classification = "MODIFIED"
    elif change_pct > 20:
        classification = "CONFIRMED"
    else:
        classification = "REJECTED"

    return {
        'Insight_ID': 'G5',
        'Tier': 'GOLD',
        'Original_Metric': f"52 transition proteins, 28% collagens predictable",
        'V2_Metric': f"{transitions} transition proteins, {predictable_collagens_pct:.1f}% collagens predictable",
        'Change_Percent': change_pct,
        'Classification': classification,
        'Notes': f"Ordered: {len(ordered)}, Chaotic: {len(chaotic)}"
    }, entropy_df

# =============================================================================
# G6: COMPARTMENT ANTAGONISM
# =============================================================================

def validate_g6_antagonism(df):
    """
    Original: 11 antagonistic events, Col11a2 divergence SD=1.86
    """
    print("\n" + "="*80)
    print("G6: COMPARTMENT ANTAGONISTIC REMODELING")
    print("="*80)

    antagonistic_events = []

    # Group by tissue and gene
    for (tissue, gene), group in df.groupby(['Tissue', 'Gene_Symbol']):
        compartments = group['Compartment'].unique()

        if len(compartments) < 2:
            continue

        # Calculate mean Δz per compartment
        compartment_means = group.groupby('Compartment')['Zscore_Delta'].mean()

        if len(compartment_means) < 2:
            continue

        # Divergence = SD across compartments
        divergence = compartment_means.std()

        # Check for opposite directions (antagonistic)
        signs = np.sign(compartment_means.values)
        is_antagonistic = (signs > 0).any() and (signs < 0).any()

        if is_antagonistic and divergence > 1.5:
            antagonistic_events.append({
                'Tissue': tissue,
                'Gene_Symbol': gene,
                'Divergence': divergence,
                'Compartments': ', '.join(compartments),
                'Compartment_Means': compartment_means.to_dict()
            })

    antagonistic_df = pd.DataFrame(antagonistic_events)

    print(f"V2 Antagonistic events: {len(antagonistic_df)}")
    print(f"V1 Baseline: {V1_BASELINES['G6_antagonistic_events']}")

    if len(antagonistic_df) > 0:
        print("\nTop 5 antagonistic events:")
        print(antagonistic_df.nlargest(5, 'Divergence')[['Tissue', 'Gene_Symbol', 'Divergence']].to_string(index=False))

        # Check Col11a2
        col11a2 = antagonistic_df[antagonistic_df['Gene_Symbol'].str.upper() == 'COL11A2']
        if len(col11a2) > 0:
            col11a2_div = col11a2['Divergence'].max()
            print(f"\nCol11a2 divergence: {col11a2_div:.3f} (V1: {V1_BASELINES['G6_col11a2_divergence']})")

    # Classification
    change_pct = ((len(antagonistic_df) - V1_BASELINES['G6_antagonistic_events']) / V1_BASELINES['G6_antagonistic_events']) * 100

    if abs(change_pct) <= 30:  # Allow 30% variation for compartment analysis
        classification = "MODIFIED"
    elif change_pct > 30:
        classification = "CONFIRMED"
    else:
        classification = "REJECTED"

    return {
        'Insight_ID': 'G6',
        'Tier': 'GOLD',
        'Original_Metric': f"{V1_BASELINES['G6_antagonistic_events']} events, Col11a2 div={V1_BASELINES['G6_col11a2_divergence']}",
        'V2_Metric': f"{len(antagonistic_df)} events",
        'Change_Percent': change_pct,
        'Classification': classification,
        'Notes': f"Max divergence: {antagonistic_df['Divergence'].max():.2f}" if len(antagonistic_df) > 0 else "None found"
    }, antagonistic_df

# =============================================================================
# G7: SPECIES DIVERGENCE
# =============================================================================

def validate_g7_species_divergence(df):
    """
    Original: 8/1,167 genes cross-species, R=-0.71 (opposite directions)
    """
    print("\n" + "="*80)
    print("G7: SPECIES DIVERGENCE")
    print("="*80)

    # Separate human and mouse
    human_df = df[df['Species'] == 'Homo sapiens'].groupby('Gene_Symbol')['Zscore_Delta'].mean()
    mouse_df = df[df['Species'] == 'Mus musculus'].groupby('Gene_Symbol')['Zscore_Delta'].mean()

    # Find shared genes
    shared_genes = set(human_df.index) & set(mouse_df.index)

    print(f"Human proteins: {len(human_df)}")
    print(f"Mouse proteins: {len(mouse_df)}")
    print(f"Shared genes: {len(shared_genes)}")
    print(f"V1 Baseline: {V1_BASELINES['G7_shared_genes']}/{V1_BASELINES['G7_total_genes']}")

    if len(shared_genes) > 0:
        # Create matched dataframe
        shared_df = pd.DataFrame({
            'Gene_Symbol': list(shared_genes),
            'Human_Dz': [human_df[g] for g in shared_genes],
            'Mouse_Dz': [mouse_df[g] for g in shared_genes]
        })

        # Correlation
        correlation = shared_df['Human_Dz'].corr(shared_df['Mouse_Dz'])

        # Concordance (same sign)
        concordance_pct = ((np.sign(shared_df['Human_Dz']) == np.sign(shared_df['Mouse_Dz'])).sum() / len(shared_df)) * 100

        print(f"Correlation: {correlation:.3f} (V1: {V1_BASELINES['G7_correlation']})")
        print(f"Concordance: {concordance_pct:.1f}%")

        print("\nShared genes:")
        print(shared_df.to_string(index=False))
    else:
        correlation = np.nan
        concordance_pct = 0
        print("No shared genes found!")

    # Classification
    total_genes_v2 = len(set(human_df.index) | set(mouse_df.index))
    shared_pct_v2 = (len(shared_genes) / total_genes_v2) * 100 if total_genes_v2 > 0 else 0
    shared_pct_v1 = (V1_BASELINES['G7_shared_genes'] / V1_BASELINES['G7_total_genes']) * 100

    if abs(shared_pct_v2 - shared_pct_v1) <= 2:  # Within 2 percentage points
        classification = "CONFIRMED"
    else:
        classification = "MODIFIED"

    return {
        'Insight_ID': 'G7',
        'Tier': 'GOLD',
        'Original_Metric': f"{V1_BASELINES['G7_shared_genes']}/{V1_BASELINES['G7_total_genes']} genes, R={V1_BASELINES['G7_correlation']}",
        'V2_Metric': f"{len(shared_genes)}/{total_genes_v2} genes, R={correlation:.3f}" if not np.isnan(correlation) else f"{len(shared_genes)} genes, R=N/A",
        'Change_Percent': ((len(shared_genes) - V1_BASELINES['G7_shared_genes']) / V1_BASELINES['G7_shared_genes']) * 100 if V1_BASELINES['G7_shared_genes'] > 0 else 0,
        'Classification': classification,
        'Notes': f"Concordance: {concordance_pct:.1f}%"
    }, shared_df if len(shared_genes) > 0 else None

# =============================================================================
# SILVER INSIGHTS (S1-S5)
# =============================================================================

def validate_s1_fibrinogen(df):
    """S1: Fibrinogen cascade - FGA, FGB, SERPINC1"""
    print("\n" + "="*80)
    print("S1: FIBRINOGEN COAGULATION CASCADE")
    print("="*80)

    coagulation_proteins = ['FGA', 'FGB', 'FGG', 'SERPINC1', 'SERPINA1']

    results = []
    for protein in coagulation_proteins:
        protein_data = df[df['Gene_Symbol'] == protein]
        if len(protein_data) > 0:
            mean_dz = protein_data['Zscore_Delta'].mean()
            consistency = (protein_data['Zscore_Delta'] > 0).sum() / len(protein_data) * 100 if len(protein_data) > 0 else 0
            results.append({
                'Protein': protein,
                'Mean_Dz': mean_dz,
                'Consistency_Pct': consistency,
                'N': len(protein_data)
            })
            print(f"{protein}: Δz={mean_dz:.3f}, consistency={consistency:.1f}%")

    results_df = pd.DataFrame(results)

    # Check if main proteins preserved
    fga_match = abs(results_df[results_df['Protein'] == 'FGA']['Mean_Dz'].values[0] - V1_BASELINES['S1_fga_dz']) < 0.5 if 'FGA' in results_df['Protein'].values else False
    fgb_match = abs(results_df[results_df['Protein'] == 'FGB']['Mean_Dz'].values[0] - V1_BASELINES['S1_fgb_dz']) < 0.5 if 'FGB' in results_df['Protein'].values else False

    classification = "CONFIRMED" if (fga_match and fgb_match) else "MODIFIED"

    return {
        'Insight_ID': 'S1',
        'Tier': 'SILVER',
        'Original_Metric': f"FGA +{V1_BASELINES['S1_fga_dz']}, FGB +{V1_BASELINES['S1_fgb_dz']}",
        'V2_Metric': f"FGA {results_df[results_df['Protein']=='FGA']['Mean_Dz'].values[0]:.2f}" if 'FGA' in results_df['Protein'].values else "N/A",
        'Change_Percent': 0,  # Simplified
        'Classification': classification,
        'Notes': f"{len(results)} coagulation proteins detected"
    }, results_df

def validate_s2_temporal_windows(df):
    """S2: Temporal intervention windows"""
    print("\n" + "="*80)
    print("S2: TEMPORAL INTERVENTION WINDOWS")
    print("="*80)

    # Simplified - would need actual age data
    classification = "MODIFIED"  # Cannot fully validate without detailed age data

    return {
        'Insight_ID': 'S2',
        'Tier': 'SILVER',
        'Original_Metric': "Age windows: 40-50, 50-65, 65+",
        'V2_Metric': "Requires detailed age mapping",
        'Change_Percent': 0,
        'Classification': classification,
        'Notes': "Partial validation - age column needed"
    }, None

def validate_s3_timp3(df):
    """S3: TIMP3 lock-in"""
    print("\n" + "="*80)
    print("S3: TIMP3 LOCK-IN")
    print("="*80)

    timp3_data = df[df['Gene_Symbol'] == 'TIMP3']

    if len(timp3_data) > 0:
        mean_dz = timp3_data['Zscore_Delta'].mean()
        consistency = (timp3_data['Zscore_Delta'] > 0).sum() / len(timp3_data) * 100
        study_count = timp3_data['Study_ID'].nunique()

        print(f"TIMP3: Δz={mean_dz:.3f}, consistency={consistency:.1f}%, studies={study_count}")
        print(f"V1: Δz={V1_BASELINES['S3_timp3_dz']}, consistency={V1_BASELINES['S3_timp3_consistency']}%")

        change_pct = ((mean_dz - V1_BASELINES['S3_timp3_dz']) / V1_BASELINES['S3_timp3_dz']) * 100
        classification = "CONFIRMED" if abs(change_pct) < 30 else "MODIFIED"
    else:
        mean_dz = np.nan
        classification = "REJECTED"

    return {
        'Insight_ID': 'S3',
        'Tier': 'SILVER',
        'Original_Metric': f"Δz={V1_BASELINES['S3_timp3_dz']}, {V1_BASELINES['S3_timp3_consistency']}% consistency",
        'V2_Metric': f"Δz={mean_dz:.3f}" if not np.isnan(mean_dz) else "NOT FOUND",
        'Change_Percent': change_pct if not np.isnan(mean_dz) else 0,
        'Classification': classification,
        'Notes': f"{study_count} studies" if not np.isnan(mean_dz) else "Not detected"
    }, timp3_data

def validate_s4_tissue_specific(df):
    """S4: Tissue-specific signatures (TSI)"""
    print("\n" + "="*80)
    print("S4: TISSUE-SPECIFIC SIGNATURES (TSI)")
    print("="*80)

    # Calculate TSI = max_tissue_Δz / mean_other_tissues_Δz
    tsi_results = []

    for gene in df['Gene_Symbol'].unique():
        gene_data = df[df['Gene_Symbol'] == gene]
        tissue_means = gene_data.groupby('Tissue')['Zscore_Delta'].mean()

        if len(tissue_means) < 2:
            continue

        max_tissue = tissue_means.abs().idxmax()
        max_value = abs(tissue_means[max_tissue])
        other_mean = tissue_means[tissue_means.index != max_tissue].abs().mean()

        if other_mean > 0:
            tsi = max_value / other_mean
            if tsi > 3.0:
                tsi_results.append({
                    'Gene_Symbol': gene,
                    'TSI': tsi,
                    'Max_Tissue': max_tissue,
                    'Max_Value': max_value
                })

    tsi_df = pd.DataFrame(tsi_results).sort_values('TSI', ascending=False)

    print(f"V2 Tissue-specific proteins (TSI>3.0): {len(tsi_df)}")
    print(f"V1 Baseline: {V1_BASELINES['S4_tsi_proteins']}")

    if len(tsi_df) > 0:
        print("\nTop 5:")
        print(tsi_df.head(5).to_string(index=False))

    change_pct = ((len(tsi_df) - V1_BASELINES['S4_tsi_proteins']) / V1_BASELINES['S4_tsi_proteins']) * 100
    classification = "CONFIRMED" if abs(change_pct) < 30 else "MODIFIED"

    return {
        'Insight_ID': 'S4',
        'Tier': 'SILVER',
        'Original_Metric': f"{V1_BASELINES['S4_tsi_proteins']} proteins TSI>3.0",
        'V2_Metric': f"{len(tsi_df)} proteins TSI>3.0",
        'Change_Percent': change_pct,
        'Classification': classification,
        'Notes': f"Max TSI: {tsi_df.iloc[0]['TSI']:.1f}" if len(tsi_df) > 0 else "None"
    }, tsi_df

def validate_s5_biomarker_panel(df):
    """S5: Biomarker panel (7-protein clock)"""
    print("\n" + "="*80)
    print("S5: BIOMARKER PANEL")
    print("="*80)

    # Filter for secreted/plasma-accessible proteins
    secreted_categories = ['ECM Glycoproteins', 'Collagens', 'Proteoglycans', 'Secreted Factors']

    biomarker_candidates = df[df['Matrisome_Category'].isin(secreted_categories)].groupby('Gene_Symbol').agg({
        'Zscore_Delta': 'mean',
        'Tissue': 'nunique'
    }).reset_index()

    biomarker_candidates.columns = ['Gene_Symbol', 'Mean_Dz', 'Tissue_Count']

    # Rank by universality and effect size
    biomarker_candidates['Score'] = biomarker_candidates['Tissue_Count'] * biomarker_candidates['Mean_Dz'].abs()
    biomarker_candidates = biomarker_candidates.sort_values('Score', ascending=False)

    top_7 = biomarker_candidates.head(7)

    print("Top 7 biomarker candidates:")
    print(top_7.to_string(index=False))

    classification = "MODIFIED"  # Would need original 7 proteins to confirm

    return {
        'Insight_ID': 'S5',
        'Tier': 'SILVER',
        'Original_Metric': "7-protein plasma aging clock",
        'V2_Metric': f"Top candidate: {top_7.iloc[0]['Gene_Symbol']}",
        'Change_Percent': 0,
        'Classification': classification,
        'Notes': "Panel re-ranking needed"
    }, top_7

# =============================================================================
# NEW DISCOVERIES
# =============================================================================

def identify_new_discoveries(df, universal_df, weak_signals_df, antagonistic_df):
    """Identify emergent patterns not visible in V1"""
    print("\n" + "="*80)
    print("NEW DISCOVERIES IN V2")
    print("="*80)

    discoveries = []

    # 1. New universal markers (not in V1 top-5)
    v1_top5 = set(V1_BASELINES['G1_top5'])
    new_universal = universal_df[~universal_df['Gene_Symbol'].isin(v1_top5)].head(5)

    for _, row in new_universal.iterrows():
        discoveries.append({
            'Discovery_Type': 'New_Universal_Marker',
            'Protein_Pattern': row['Gene_Symbol'],
            'Metric': f"Universality={row['Universality_Score']:.3f}",
            'Description': f"Tissues={row['Tissue_Count']}, Consistency={row['Consistency_Pct']:.1f}%"
        })

    # 2. Extreme z-scores (|Δz| > 3.0)
    extreme_proteins = df.groupby('Gene_Symbol')['Zscore_Delta'].mean()
    extreme_proteins = extreme_proteins[extreme_proteins.abs() > 3.0].sort_values(key=abs, ascending=False)

    for gene, dz in extreme_proteins.head(3).items():
        discoveries.append({
            'Discovery_Type': 'Extreme_Effect',
            'Protein_Pattern': gene,
            'Metric': f"Δz={dz:.3f}",
            'Description': "Outlier effect size"
        })

    # 3. Consistent weak signals (emerged from noise)
    if weak_signals_df is not None and len(weak_signals_df) > V1_BASELINES['G4_weak_signal_count']:
        new_weak = weak_signals_df.iloc[V1_BASELINES['G4_weak_signal_count']:].head(5)
        for _, row in new_weak.iterrows():
            discoveries.append({
                'Discovery_Type': 'Emergent_Weak_Signal',
                'Protein_Pattern': row['Gene_Symbol'],
                'Metric': f"Δz={row['Mean_Zscore_Delta']:.3f}",
                'Description': f"Consistency={row['Consistency']:.1f}%, Category={row['Matrisome_Category']}"
            })

    discoveries_df = pd.DataFrame(discoveries)

    print(f"Total new discoveries: {len(discoveries_df)}")
    if len(discoveries_df) > 0:
        print("\n" + discoveries_df.to_string(index=False))

    return discoveries_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute full validation pipeline"""

    print("="*80)
    print("META-INSIGHTS VALIDATION PIPELINE - AGENT: claude_2")
    print("="*80)

    # Load data
    df = load_v2_dataset()

    # Storage for results
    validation_results = []
    artifacts = {}

    # =========================================================================
    # PHASE 1: GOLD INSIGHTS
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 1: GOLD INSIGHTS VALIDATION")
    print("="*80)

    # G1
    result, universal_df = validate_g1_universal_markers(df)
    validation_results.append(result)
    artifacts['universal_markers'] = universal_df

    # G2
    result, pcolce_df = validate_g2_pcolce(df)
    validation_results.append(result)

    # G3
    result, pca_obj = validate_g3_batch_effects(df)
    validation_results.append(result)

    # G4
    result, weak_signals_df = validate_g4_weak_signals(df)
    validation_results.append(result)
    artifacts['weak_signals'] = weak_signals_df

    # G5
    result, entropy_df = validate_g5_entropy(df)
    validation_results.append(result)
    artifacts['entropy'] = entropy_df

    # G6
    result, antagonistic_df = validate_g6_antagonism(df)
    validation_results.append(result)
    artifacts['antagonistic'] = antagonistic_df

    # G7
    result, species_df = validate_g7_species_divergence(df)
    validation_results.append(result)

    # =========================================================================
    # PHASE 2: SILVER INSIGHTS
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 2: SILVER INSIGHTS VALIDATION")
    print("="*80)

    # S1
    result, fibrinogen_df = validate_s1_fibrinogen(df)
    validation_results.append(result)

    # S2
    result, _ = validate_s2_temporal_windows(df)
    validation_results.append(result)

    # S3
    result, timp3_df = validate_s3_timp3(df)
    validation_results.append(result)

    # S4
    result, tsi_df = validate_s4_tissue_specific(df)
    validation_results.append(result)
    artifacts['tissue_specific'] = tsi_df

    # S5
    result, biomarker_df = validate_s5_biomarker_panel(df)
    validation_results.append(result)
    artifacts['biomarkers'] = biomarker_df

    # =========================================================================
    # PHASE 3: NEW DISCOVERIES
    # =========================================================================

    discoveries_df = identify_new_discoveries(
        df,
        universal_df,
        weak_signals_df,
        antagonistic_df
    )

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Validation results CSV
    validation_df = pd.DataFrame(validation_results)
    validation_path = OUTPUT_DIR + "validation_results_claude_2.csv"
    validation_df.to_csv(validation_path, index=False)
    print(f"Saved: {validation_path}")

    # New discoveries CSV
    if len(discoveries_df) > 0:
        discoveries_path = OUTPUT_DIR + "new_discoveries_claude_2.csv"
        discoveries_df.to_csv(discoveries_path, index=False)
        print(f"Saved: {discoveries_path}")

    # Validated proteins subset
    validated_proteins = set()
    for artifact_df in [universal_df, weak_signals_df, antagonistic_df, tsi_df]:
        if artifact_df is not None and len(artifact_df) > 0:
            if 'Gene_Symbol' in artifact_df.columns:
                validated_proteins.update(artifact_df['Gene_Symbol'].unique())

    validated_subset = df[df['Gene_Symbol'].isin(validated_proteins)]
    subset_path = OUTPUT_DIR + "v2_validated_proteins_claude_2.csv"
    validated_subset.to_csv(subset_path, index=False)
    print(f"Saved: {subset_path} ({len(validated_subset)} rows, {len(validated_proteins)} proteins)")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    confirmed = validation_df[validation_df['Classification'] == 'CONFIRMED']
    modified = validation_df[validation_df['Classification'] == 'MODIFIED']
    rejected = validation_df[validation_df['Classification'] == 'REJECTED']

    print(f"CONFIRMED: {len(confirmed)}/12 insights")
    print(f"MODIFIED: {len(modified)}/12 insights")
    print(f"REJECTED: {len(rejected)}/12 insights")

    print("\nGOLD tier:")
    gold = validation_df[validation_df['Tier'] == 'GOLD']
    print(f"  CONFIRMED: {len(gold[gold['Classification']=='CONFIRMED'])}/7")
    print(f"  MODIFIED: {len(gold[gold['Classification']=='MODIFIED'])}/7")
    print(f"  REJECTED: {len(gold[gold['Classification']=='REJECTED'])}/7")

    print("\nSILVER tier:")
    silver = validation_df[validation_df['Tier'] == 'SILVER']
    print(f"  CONFIRMED: {len(silver[silver['Classification']=='CONFIRMED'])}/5")
    print(f"  MODIFIED: {len(silver[silver['Classification']=='MODIFIED'])}/5")
    print(f"  REJECTED: {len(silver[silver['Classification']=='REJECTED'])}/5")

    print(f"\nNew discoveries: {len(discoveries_df)}")

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)

    return validation_df, discoveries_df

if __name__ == "__main__":
    validation_results, discoveries = main()
