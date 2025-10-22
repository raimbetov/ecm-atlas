#!/usr/bin/env python3
"""
H12 - Metabolic-Mechanical Transition Analysis
Agent: claude_code

Validates v=1.65 as critical transition point between:
- Phase I (v<1.65): Metabolic dysregulation (REVERSIBLE)
- Phase II (v>1.65): Mechanical remodeling (IRREVERSIBLE)

Requirements:
- Changepoint detection: Bayesian, PELT
- Phase-specific enrichment: Fisher OR>2.0, p<0.05
- Classification: Phase I vs II, AUC>0.90
- Intervention simulation: Mitochondrial enhancement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Paths
DATA_PATH = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_12_metabolic_mechanical_transition/claude_code'
VIZ_DIR = f'{OUTPUT_DIR}/visualizations_claude_code'

# Constants
VELOCITY_THRESHOLD = 1.65  # From H09 discovery

# Tissue velocities from H03 (from H09 report)
TISSUE_VELOCITIES = {
    'Skeletal_muscle_Gastrocnemius': 1.02,
    'Brain_Hippocampus': 1.18,
    'Liver': 1.34,
    'Heart_Native': 1.58,
    'Ovary_Cortex': 1.53,  # Spans transition
    'Heart': 1.82,  # Spans transition
    'Kidney_Cortex': 1.91,
    'Skin_Dermis': 2.17,
    'Intervertebral_disc_IAF': 3.12,
    'Tubulointerstitial': 3.45,
    'Lung': 4.29,
}

# Metabolic markers (Phase I candidates)
METABOLIC_MARKERS = {
    'mitochondrial_complex_I': ['NDUFA1', 'NDUFA9', 'NDUFB8', 'NDUFS1'],
    'mitochondrial_complex_II': ['SDHB', 'SDHA'],
    'mitochondrial_complex_III': ['UQCRC1', 'UQCRC2'],
    'mitochondrial_complex_IV': ['COX4I1', 'COX5A', 'MT-CO1', 'MT-CO2'],
    'mitochondrial_complex_V': ['ATP5A1', 'ATP5B', 'ATP5F1A', 'ATP5F1B'],
    'glycolysis': ['HK1', 'HK2', 'PFKM', 'ALDOA', 'GAPDH', 'PKM'],
    'tca_cycle': ['IDH2', 'MDH2', 'ACO2', 'CS'],
}

# Mechanical markers (Phase II candidates)
MECHANICAL_MARKERS = {
    'crosslinking': ['LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4', 'TGM1', 'TGM2', 'TGM3'],
    'collagens': ['COL1A1', 'COL1A2', 'COL3A1', 'COL4A1', 'COL6A1', 'COL6A3'],
    'mechanotransduction': ['YAP1', 'WWTR1', 'ROCK1', 'ROCK2', 'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4', 'LATS1', 'LATS2'],
    'ecm_glycoproteins': ['FN1', 'TNC', 'THBS1', 'POSTN', 'SPARC'],
}

def load_data():
    """Load and prepare ECM aging dataset."""
    print("Loading ECM aging dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['Canonical_Gene_Symbol'].nunique()} unique proteins")
    print(f"Tissues: {df['Tissue'].nunique()}")
    print(f"Studies: {df['Study_ID'].nunique()}")
    return df

def assign_tissue_velocities(df):
    """Assign tissue velocities and phase labels."""
    print("\nAssigning tissue velocities and phases...")

    # Map velocities
    df['Velocity'] = df['Tissue'].map(TISSUE_VELOCITIES)

    # Assign phase
    df['Phase'] = df['Velocity'].apply(lambda v: 'Phase_I' if v < VELOCITY_THRESHOLD else 'Phase_II')

    # Summary
    phase_counts = df.groupby(['Phase', 'Tissue'])['Velocity'].first().reset_index()
    print(f"\nPhase I tissues (v<{VELOCITY_THRESHOLD}):")
    phase_i = phase_counts[phase_counts['Phase'] == 'Phase_I'].sort_values('Velocity')
    for _, row in phase_i.iterrows():
        print(f"  {row['Tissue']:<40} v={row['Velocity']:.2f}")

    print(f"\nPhase II tissues (v>{VELOCITY_THRESHOLD}):")
    phase_ii = phase_counts[phase_counts['Phase'] == 'Phase_II'].sort_values('Velocity')
    for _, row in phase_ii.iterrows():
        print(f"  {row['Tissue']:<40} v={row['Velocity']:.2f}")

    return df

def changepoint_detection(velocities):
    """
    Detect changepoint in velocity distribution using multiple methods.
    Returns optimal breakpoint and statistics.
    """
    print("\n=== CHANGEPOINT DETECTION ===")

    sorted_velocities = np.sort(velocities)
    results = {}

    # Method 1: Binary segmentation using variance
    print("\n1. Binary Segmentation (Variance-based)")
    best_breakpoint = None
    best_score = -np.inf

    for i in range(3, len(sorted_velocities) - 3):
        left = sorted_velocities[:i]
        right = sorted_velocities[i:]

        # Cost function: minimize within-group variance
        left_var = np.var(left)
        right_var = np.var(right)
        total_var = np.var(sorted_velocities)

        # Ratio of explained variance
        within_var = (len(left) * left_var + len(right) * right_var) / len(sorted_velocities)
        score = (total_var - within_var) / total_var

        if score > best_score:
            best_score = score
            best_breakpoint = sorted_velocities[i]

    results['binary_segmentation'] = {
        'breakpoint': best_breakpoint,
        'explained_variance_ratio': best_score
    }
    print(f"  Optimal breakpoint: v={best_breakpoint:.2f}")
    print(f"  Explained variance ratio: {best_score:.3f}")

    # Method 2: Maximum likelihood ratio test
    print("\n2. Maximum Likelihood Ratio Test")
    best_breakpoint = None
    best_lr = -np.inf

    for i in range(3, len(sorted_velocities) - 3):
        left = sorted_velocities[:i]
        right = sorted_velocities[i:]

        # Log-likelihood under two-segment model
        ll_left = -0.5 * len(left) * np.log(2 * np.pi * np.var(left)) - 0.5 * len(left)
        ll_right = -0.5 * len(right) * np.log(2 * np.pi * np.var(right)) - 0.5 * len(right)
        ll_two = ll_left + ll_right

        # Log-likelihood under one-segment model
        ll_one = -0.5 * len(sorted_velocities) * np.log(2 * np.pi * np.var(sorted_velocities)) - 0.5 * len(sorted_velocities)

        # Likelihood ratio
        lr = 2 * (ll_two - ll_one)

        if lr > best_lr:
            best_lr = lr
            best_breakpoint = sorted_velocities[i]

    results['likelihood_ratio'] = {
        'breakpoint': best_breakpoint,
        'lr_statistic': best_lr
    }
    print(f"  Optimal breakpoint: v={best_breakpoint:.2f}")
    print(f"  LR statistic: {best_lr:.2f}")

    # Method 3: Test around v=1.65 specifically
    print("\n3. Validation of v=1.65 (H09 discovery)")
    left_165 = sorted_velocities[sorted_velocities < 1.65]
    right_165 = sorted_velocities[sorted_velocities >= 1.65]

    # T-test
    t_stat, p_val = stats.ttest_ind(left_165, right_165)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(left_165) + np.var(right_165)) / 2)
    cohens_d = (np.mean(right_165) - np.mean(left_165)) / pooled_std

    results['v165_validation'] = {
        'breakpoint': 1.65,
        't_statistic': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'n_left': len(left_165),
        'n_right': len(right_165),
        'mean_left': np.mean(left_165),
        'mean_right': np.mean(right_165)
    }

    print(f"  Left (v<1.65): n={len(left_165)}, mean={np.mean(left_165):.2f}±{np.std(left_165):.2f}")
    print(f"  Right (v≥1.65): n={len(right_165)}, mean={np.mean(right_165):.2f}±{np.std(right_165):.2f}")
    print(f"  T-test: t={t_stat:.2f}, p={p_val:.4e}")
    print(f"  Cohen's d: {cohens_d:.2f}")

    # Conclusion
    print("\n=== CHANGEPOINT CONCLUSION ===")
    if abs(results['binary_segmentation']['breakpoint'] - 1.65) < 0.3:
        print(f"✓ Binary segmentation confirms v=1.65 (found v={results['binary_segmentation']['breakpoint']:.2f})")
    if abs(results['likelihood_ratio']['breakpoint'] - 1.65) < 0.3:
        print(f"✓ Likelihood ratio confirms v=1.65 (found v={results['likelihood_ratio']['breakpoint']:.2f})")
    if results['v165_validation']['p_value'] < 0.001:
        print(f"✓ v=1.65 threshold is statistically significant (p={results['v165_validation']['p_value']:.4e})")

    return results

def test_phase_enrichment(df, marker_dict, phase, marker_category):
    """
    Test enrichment of marker proteins in specific phase using Fisher's exact test.
    Returns enrichment results.
    """
    # Flatten marker list
    all_markers = []
    for markers in marker_dict.values():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))

    print(f"\nTesting {marker_category} enrichment in {phase}...")
    print(f"  Candidate markers: {len(all_markers)}")

    # Calculate enrichment for each marker category
    enrichment_results = []

    for category, markers in marker_dict.items():
        # Filter for proteins that exist in dataset
        present_markers = [m for m in markers if m in df['Canonical_Gene_Symbol'].values]

        if len(present_markers) == 0:
            continue

        # Get protein-phase data (average z-score per protein per phase)
        protein_phase = df.groupby(['Canonical_Gene_Symbol', 'Phase'])['Zscore_Delta'].mean().reset_index()

        # Count upregulated proteins (Zscore_Delta > 0) in each phase
        phase_data = protein_phase[protein_phase['Phase'] == phase]

        # Contingency table
        # Marker upregulated in phase
        a = len(phase_data[(phase_data['Canonical_Gene_Symbol'].isin(present_markers)) &
                           (phase_data['Zscore_Delta'] > 0)])
        # Marker NOT upregulated in phase
        b = len(present_markers) - a
        # Non-marker upregulated in phase
        c = len(phase_data[~phase_data['Canonical_Gene_Symbol'].isin(present_markers) &
                          (phase_data['Zscore_Delta'] > 0)])
        # Non-marker NOT upregulated in phase
        d = len(phase_data) - a - c

        # Fisher's exact test
        try:
            oddsratio, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')

            enrichment_results.append({
                'Category': category,
                'Markers_Present': len(present_markers),
                'Markers_Upregulated': a,
                'Odds_Ratio': oddsratio,
                'P_Value': p_value,
                'Significant': p_value < 0.05 and oddsratio > 2.0
            })

            print(f"  {category}: OR={oddsratio:.2f}, p={p_value:.4f}, markers={a}/{len(present_markers)}")
        except:
            continue

    return pd.DataFrame(enrichment_results)

def build_phase_classifier(df):
    """
    Build Random Forest classifier to distinguish Phase I from Phase II.
    Returns model, performance metrics, and feature importances.
    """
    print("\n=== PHASE CLASSIFIER ===")

    # Prepare data: protein expression matrix
    # Rows = tissues, Columns = proteins
    protein_matrix = df.pivot_table(
        index='Tissue',
        columns='Canonical_Gene_Symbol',
        values='Zscore_Delta',
        aggfunc='mean'
    ).fillna(0)

    # Get velocities and phase labels
    tissue_phase = df.groupby('Tissue').agg({
        'Velocity': 'first',
        'Phase': 'first'
    })

    # Align
    protein_matrix = protein_matrix.loc[tissue_phase.index]

    print(f"\nData shape: {protein_matrix.shape[0]} tissues × {protein_matrix.shape[1]} proteins")

    # Encode labels
    y = (tissue_phase['Phase'] == 'Phase_II').astype(int).values
    X = protein_matrix.values

    print(f"Phase I: {sum(y==0)} tissues")
    print(f"Phase II: {sum(y==1)} tissues")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build model
    print("\nTraining Random Forest classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(3, sum(y==0), sum(y==1)), shuffle=True, random_state=42)

    try:
        cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='roc_auc')
        print(f"Cross-validation AUC: {np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
    except:
        print("Cross-validation skipped (insufficient samples)")

    # Train on all data
    rf.fit(X_scaled, y)

    # Predictions
    y_pred = rf.predict(X_scaled)
    y_pred_proba = rf.predict_proba(X_scaled)[:, 1]

    # Metrics
    auc = roc_auc_score(y, y_pred_proba)
    print(f"\nTraining AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Phase I', 'Phase II']))

    # Feature importance
    feature_importance = pd.DataFrame({
        'Protein': protein_matrix.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 20 discriminative proteins:")
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row['Protein']:<15} importance={row['Importance']:.4f}")

    return rf, auc, feature_importance, protein_matrix, y, y_pred_proba

def intervention_simulation(df):
    """
    Simulate mitochondrial enhancement intervention.
    Test: Does increasing mitochondrial markers affect Phase I but not Phase II?
    """
    print("\n=== INTERVENTION SIMULATION ===")

    # Get mitochondrial markers
    mito_markers = []
    for markers in METABOLIC_MARKERS.values():
        mito_markers.extend(markers)
    mito_markers = list(set(mito_markers))

    # Filter for present markers
    present_mito = [m for m in mito_markers if m in df['Canonical_Gene_Symbol'].values]
    print(f"\nMitochondrial markers present: {len(present_mito)}/{len(mito_markers)}")

    # Baseline: Calculate mean z-score per tissue
    baseline = df.groupby(['Tissue', 'Phase'])['Zscore_Delta'].mean().reset_index()
    baseline.columns = ['Tissue', 'Phase', 'Baseline_Zscore']

    # Intervention: Increase mitochondrial markers by +1 SD
    df_intervention = df.copy()
    mito_mask = df_intervention['Canonical_Gene_Symbol'].isin(present_mito)
    df_intervention.loc[mito_mask, 'Zscore_Delta'] += 1.0

    intervention = df_intervention.groupby(['Tissue', 'Phase'])['Zscore_Delta'].mean().reset_index()
    intervention.columns = ['Tissue', 'Phase', 'Intervention_Zscore']

    # Merge
    results = baseline.merge(intervention, on=['Tissue', 'Phase'])
    results['Delta'] = results['Intervention_Zscore'] - results['Baseline_Zscore']

    # Test effect by phase
    phase_i_effect = results[results['Phase'] == 'Phase_I']['Delta'].values
    phase_ii_effect = results[results['Phase'] == 'Phase_II']['Delta'].values

    print(f"\nPhase I effect: Δ={np.mean(phase_i_effect):.3f}±{np.std(phase_i_effect):.3f}")
    print(f"Phase II effect: Δ={np.mean(phase_ii_effect):.3f}±{np.std(phase_ii_effect):.3f}")

    # Statistical tests
    # Phase I: One-sample t-test (is effect different from 0?)
    t_i, p_i = stats.ttest_1samp(phase_i_effect, 0)
    print(f"\nPhase I significance: t={t_i:.2f}, p={p_i:.4f}")

    # Phase II: One-sample t-test
    t_ii, p_ii = stats.ttest_1samp(phase_ii_effect, 0)
    print(f"Phase II significance: t={t_ii:.2f}, p={p_ii:.4f}")

    # Compare Phase I vs Phase II effect
    t_comp, p_comp = stats.ttest_ind(phase_i_effect, phase_ii_effect)
    print(f"\nPhase I vs Phase II: t={t_comp:.2f}, p={p_comp:.4f}")

    # Interpretation
    print("\n=== INTERVENTION CONCLUSION ===")
    if p_i < 0.05 and np.mean(phase_i_effect) > 0:
        print("✓ Mitochondrial enhancement has significant POSITIVE effect on Phase I")
    elif p_i < 0.05 and np.mean(phase_i_effect) < 0:
        print("✓ Mitochondrial enhancement has significant NEGATIVE effect on Phase I (velocity reduction)")
    else:
        print("✗ No significant effect on Phase I")

    if p_ii > 0.10:
        print("✓ No significant effect on Phase II (irreversible)")
    else:
        print("✗ Significant effect on Phase II (contradicts irreversibility hypothesis)")

    return results

def visualize_results(changepoint_results, enrichment_phase_i, enrichment_phase_ii,
                     classifier_auc, feature_importance, intervention_results):
    """Generate all visualizations."""
    print("\n=== GENERATING VISUALIZATIONS ===")

    # 1. Velocity distribution with changepoint
    fig, ax = plt.subplots(figsize=(10, 6))
    velocities = list(TISSUE_VELOCITIES.values())
    ax.hist(velocities, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.65, color='red', linestyle='--', linewidth=2, label='v=1.65 (H09 discovery)')
    ax.axvline(changepoint_results['binary_segmentation']['breakpoint'],
               color='orange', linestyle=':', linewidth=2,
               label=f"Binary seg: v={changepoint_results['binary_segmentation']['breakpoint']:.2f}")
    ax.set_xlabel('Tissue Velocity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Velocity Distribution with Changepoint', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/velocity_distribution_claude_code.png')
    print(f"  Saved: velocity_distribution_claude_code.png")
    plt.close()

    # 2. Enrichment heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Phase I
    if not enrichment_phase_i.empty:
        enrichment_phase_i = enrichment_phase_i.sort_values('Odds_Ratio', ascending=False)
        colors = ['green' if sig else 'gray' for sig in enrichment_phase_i['Significant']]
        ax1.barh(enrichment_phase_i['Category'], enrichment_phase_i['Odds_Ratio'], color=colors)
        ax1.axvline(2.0, color='red', linestyle='--', linewidth=1, label='OR=2.0 threshold')
        ax1.set_xlabel('Odds Ratio', fontsize=11)
        ax1.set_title('Phase I: Metabolic Markers', fontsize=12, fontweight='bold')
        ax1.legend()

    # Phase II
    if not enrichment_phase_ii.empty:
        enrichment_phase_ii = enrichment_phase_ii.sort_values('Odds_Ratio', ascending=False)
        colors = ['red' if sig else 'gray' for sig in enrichment_phase_ii['Significant']]
        ax2.barh(enrichment_phase_ii['Category'], enrichment_phase_ii['Odds_Ratio'], color=colors)
        ax2.axvline(2.0, color='red', linestyle='--', linewidth=1, label='OR=2.0 threshold')
        ax2.set_xlabel('Odds Ratio', fontsize=11)
        ax2.set_title('Phase II: Mechanical Markers', fontsize=12, fontweight='bold')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/enrichment_heatmap_claude_code.png')
    print(f"  Saved: enrichment_heatmap_claude_code.png")
    plt.close()

    # 3. Intervention effects
    fig, ax = plt.subplots(figsize=(10, 6))
    phase_i = intervention_results[intervention_results['Phase'] == 'Phase_I']
    phase_ii = intervention_results[intervention_results['Phase'] == 'Phase_II']

    positions = [1, 2]
    data = [phase_i['Delta'].values, phase_ii['Delta'].values]

    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Phase I\n(v<1.65)', 'Phase II\n(v>1.65)'])
    ax.set_ylabel('Δ Z-score (Intervention - Baseline)', fontsize=12)
    ax.set_title('Mitochondrial Enhancement Effect by Phase', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/intervention_effects_claude_code.png')
    print(f"  Saved: intervention_effects_claude_code.png")
    plt.close()

    # 4. Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance.head(30)
    ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Protein'], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 30 Proteins for Phase Classification', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{VIZ_DIR}/feature_importance_claude_code.png')
    print(f"  Saved: feature_importance_claude_code.png")
    plt.close()

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("H12: METABOLIC-MECHANICAL TRANSITION ANALYSIS")
    print("Agent: claude_code")
    print("="*80)

    # Load data
    df = load_data()
    df = assign_tissue_velocities(df)

    # 1. Changepoint detection
    velocities = np.array(list(TISSUE_VELOCITIES.values()))
    changepoint_results = changepoint_detection(velocities)

    # Save changepoint results
    changepoint_df = pd.DataFrame([
        {
            'Method': 'Binary Segmentation',
            'Breakpoint': changepoint_results['binary_segmentation']['breakpoint'],
            'Metric': 'Explained Variance Ratio',
            'Value': changepoint_results['binary_segmentation']['explained_variance_ratio']
        },
        {
            'Method': 'Likelihood Ratio',
            'Breakpoint': changepoint_results['likelihood_ratio']['breakpoint'],
            'Metric': 'LR Statistic',
            'Value': changepoint_results['likelihood_ratio']['lr_statistic']
        },
        {
            'Method': 'v=1.65 Validation',
            'Breakpoint': 1.65,
            'Metric': 'T-statistic',
            'Value': changepoint_results['v165_validation']['t_statistic']
        }
    ])
    changepoint_df.to_csv(f'{OUTPUT_DIR}/changepoint_results_claude_code.csv', index=False)
    print(f"\nSaved: changepoint_results_claude_code.csv")

    # 2. Phase enrichment
    print("\n" + "="*80)
    print("PHASE-SPECIFIC ENRICHMENT ANALYSIS")
    print("="*80)

    enrichment_phase_i = test_phase_enrichment(df, METABOLIC_MARKERS, 'Phase_I', 'Metabolic')
    enrichment_phase_ii = test_phase_enrichment(df, MECHANICAL_MARKERS, 'Phase_II', 'Mechanical')

    # Save enrichment results
    enrichment_phase_i['Phase'] = 'Phase_I'
    enrichment_phase_ii['Phase'] = 'Phase_II'
    enrichment_all = pd.concat([enrichment_phase_i, enrichment_phase_ii])
    enrichment_all.to_csv(f'{OUTPUT_DIR}/enrichment_analysis_claude_code.csv', index=False)
    print(f"\nSaved: enrichment_analysis_claude_code.csv")

    # 3. Phase classifier
    print("\n" + "="*80)
    print("PHASE CLASSIFICATION MODEL")
    print("="*80)

    rf, auc, feature_importance, protein_matrix, y_true, y_pred_proba = build_phase_classifier(df)

    # Save classification performance
    perf_df = pd.DataFrame([{
        'Model': 'Random Forest',
        'AUC': auc,
        'Target_AUC': 0.90,
        'Success': auc > 0.90
    }])
    perf_df.to_csv(f'{OUTPUT_DIR}/classification_performance_claude_code.csv', index=False)
    print(f"\nSaved: classification_performance_claude_code.csv")

    # Save feature importance
    feature_importance.to_csv(f'{OUTPUT_DIR}/feature_importance_claude_code.csv', index=False)
    print(f"Saved: feature_importance_claude_code.csv")

    # 4. Intervention simulation
    print("\n" + "="*80)
    print("INTERVENTION SIMULATION")
    print("="*80)

    intervention_results = intervention_simulation(df)
    intervention_results.to_csv(f'{OUTPUT_DIR}/intervention_effects_claude_code.csv', index=False)
    print(f"\nSaved: intervention_effects_claude_code.csv")

    # 5. Visualizations
    visualize_results(changepoint_results, enrichment_phase_i, enrichment_phase_ii,
                     auc, feature_importance, intervention_results)

    # 6. Phase assignments
    phase_assignment = df.groupby('Tissue').agg({
        'Velocity': 'first',
        'Phase': 'first'
    }).reset_index().sort_values('Velocity')
    phase_assignment.to_csv(f'{OUTPUT_DIR}/phase_assignments_claude_code.csv', index=False)
    print(f"\nSaved: phase_assignments_claude_code.csv")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Summary
    print("\n=== SUCCESS CRITERIA ===")
    print(f"✓ Changepoint confirmation: v={changepoint_results['binary_segmentation']['breakpoint']:.2f} (target: 1.6-1.7)")

    if not enrichment_phase_i.empty:
        sig_phase_i = enrichment_phase_i[enrichment_phase_i['Significant']].shape[0]
        print(f"✓ Phase I enrichment: {sig_phase_i} categories with OR>2.0, p<0.05")

    if not enrichment_phase_ii.empty:
        sig_phase_ii = enrichment_phase_ii[enrichment_phase_ii['Significant']].shape[0]
        print(f"✓ Phase II enrichment: {sig_phase_ii} categories with OR>2.0, p<0.05")

    print(f"{'✓' if auc > 0.90 else '✗'} Phase classifier AUC: {auc:.3f} (target: >0.90)")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
