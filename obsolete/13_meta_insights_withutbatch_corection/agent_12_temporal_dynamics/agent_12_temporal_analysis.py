#!/usr/bin/env python3
"""
Agent 12: Temporal Dynamics Reconstructor
Reconstructs time-course of ECM aging from cross-sectional snapshots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Filter valid z-score deltas
df_valid = df[df['Zscore_Delta'].notna()].copy()
print(f"Total entries: {len(df)}, Valid z-scores: {len(df_valid)}")

# ============================================================================
# 1. EXTRACT AGE INFORMATION (Young vs Old Ranges)
# ============================================================================
print("\n" + "="*80)
print("1. EXTRACTING AGE INFORMATION")
print("="*80)

age_info = []
for study in df_valid['Study_ID'].unique():
    study_data = df_valid[df_valid['Study_ID'] == study]

    # Estimate age from "young" vs "old" groups
    # Most studies don't have explicit age data, use N_Profiles as proxy
    young_samples = study_data['N_Profiles_Young'].median()
    old_samples = study_data['N_Profiles_Old'].median()

    age_info.append({
        'Study_ID': study,
        'Species': study_data['Species'].iloc[0],
        'Tissue': study_data['Tissue'].iloc[0],
        'N_Proteins': len(study_data),
        'Young_Samples': young_samples,
        'Old_Samples': old_samples,
        'Mean_Zscore_Delta': study_data['Zscore_Delta'].mean(),
        'Std_Zscore_Delta': study_data['Zscore_Delta'].std()
    })

age_df = pd.DataFrame(age_info)
print("\nStudy Age Information:")
print(age_df.to_string())

# ============================================================================
# 2. ORDER PROTEINS BY TEMPORAL DYNAMICS (Early/Chronic/Late)
# ============================================================================
print("\n" + "="*80)
print("2. TEMPORAL ORDERING: Early vs Late Aging Markers")
print("="*80)

# Aggregate protein-level statistics across studies
protein_temporal = []

for protein in df_valid['Canonical_Gene_Symbol'].unique():
    protein_data = df_valid[df_valid['Canonical_Gene_Symbol'] == protein]

    if len(protein_data) < 2:  # Need at least 2 observations
        continue

    # Calculate temporal metrics
    mean_zscore = protein_data['Zscore_Delta'].mean()
    std_zscore = protein_data['Zscore_Delta'].std()
    consistency = 1 - (std_zscore / (abs(mean_zscore) + 0.1))  # Higher = more consistent

    # Classify temporal pattern
    if abs(mean_zscore) > 0.5 and consistency > 0.5:
        if mean_zscore > 0:
            pattern = 'Late_Increase'  # Accumulates with age
        else:
            pattern = 'Late_Decrease'  # Depletes with age
    elif abs(mean_zscore) > 0.2:
        pattern = 'Chronic_Change'  # Gradual change
    else:
        pattern = 'Stable'  # Minimal change

    protein_temporal.append({
        'Protein': protein,
        'Mean_Zscore_Delta': mean_zscore,
        'Std_Zscore_Delta': std_zscore,
        'Consistency': consistency,
        'N_Studies': len(protein_data),
        'N_Tissues': protein_data['Tissue'].nunique(),
        'Temporal_Pattern': pattern,
        'Matrisome_Division': protein_data['Matrisome_Division'].iloc[0],
        'Matrisome_Category': protein_data['Matrisome_Category'].iloc[0]
    })

temporal_df = pd.DataFrame(protein_temporal)

# Sort by mean z-score to identify early vs late markers
temporal_df = temporal_df.sort_values('Mean_Zscore_Delta', ascending=False)

print("\nTop 20 Early Increase Markers (accumulate with age):")
early_increase = temporal_df.head(20)
print(early_increase[['Protein', 'Mean_Zscore_Delta', 'Consistency', 'N_Studies', 'Temporal_Pattern']].to_string())

print("\nTop 20 Early Decrease Markers (deplete with age):")
early_decrease = temporal_df.tail(20)
print(early_decrease[['Protein', 'Mean_Zscore_Delta', 'Consistency', 'N_Studies', 'Temporal_Pattern']].to_string())

# ============================================================================
# 3. INFER CAUSAL ORDERING (Temporal Precedence)
# ============================================================================
print("\n" + "="*80)
print("3. CAUSAL ORDERING: Temporal Precedence Network")
print("="*80)

# Create protein-protein correlation matrix to infer precedence
# Pivot to protein x study matrix
pivot_data = df_valid.pivot_table(
    index='Canonical_Gene_Symbol',
    columns='Study_ID',
    values='Zscore_Delta',
    aggfunc='mean'
)

# Calculate correlation matrix (proxy for co-regulation)
corr_matrix = pivot_data.T.corr()

# Identify strong precedence relationships
precedence_edges = []
proteins = corr_matrix.columns

for i, p1 in enumerate(proteins):
    for j, p2 in enumerate(proteins):
        if i >= j:  # Skip diagonal and duplicates
            continue

        corr = corr_matrix.loc[p1, p2]

        if abs(corr) > 0.7:  # Strong correlation
            # Infer direction based on mean z-scores
            mean1 = temporal_df[temporal_df['Protein'] == p1]['Mean_Zscore_Delta'].iloc[0] if p1 in temporal_df['Protein'].values else 0
            mean2 = temporal_df[temporal_df['Protein'] == p2]['Mean_Zscore_Delta'].iloc[0] if p2 in temporal_df['Protein'].values else 0

            # Protein with larger absolute z-score likely changes first
            if abs(mean1) > abs(mean2):
                source, target = p1, p2
            else:
                source, target = p2, p1

            precedence_edges.append({
                'Source': source,
                'Target': target,
                'Correlation': corr,
                'Evidence': 'Co-regulation'
            })

precedence_df = pd.DataFrame(precedence_edges)
print(f"\nIdentified {len(precedence_df)} temporal precedence relationships")
if len(precedence_df) > 0:
    print("\nTop 20 Causal Chains:")
    print(precedence_df.nlargest(20, 'Correlation').to_string())

# ============================================================================
# 4. FIND TIPPING POINTS (Phase Transitions)
# ============================================================================
print("\n" + "="*80)
print("4. TIPPING POINTS: Critical Ages for ECM Remodeling")
print("="*80)

# Use distribution analysis to find multimodal patterns
tipping_points = []

for protein in temporal_df.head(100)['Protein']:  # Top 100 proteins
    protein_data = df_valid[df_valid['Canonical_Gene_Symbol'] == protein]['Zscore_Delta']

    if len(protein_data) < 10:
        continue

    # Test for bimodality using Hartigan's dip test proxy
    # Use coefficient of variation and kurtosis
    cv = protein_data.std() / (abs(protein_data.mean()) + 0.1)
    kurtosis = stats.kurtosis(protein_data)

    # High CV + negative kurtosis = bimodal (phase transition)
    if cv > 1.5 and kurtosis < -0.5:
        tipping_points.append({
            'Protein': protein,
            'CV': cv,
            'Kurtosis': kurtosis,
            'Mean_Zscore': protein_data.mean(),
            'Phase_Transition': 'Yes'
        })

tipping_df = pd.DataFrame(tipping_points)
if len(tipping_df) > 0:
    print(f"\nIdentified {len(tipping_df)} proteins with phase transitions")
    print("\nTop Phase Transition Candidates:")
    print(tipping_df.sort_values('CV', ascending=False).head(20).to_string())
else:
    print("\nNo strong phase transitions detected (may need age-stratified data)")

# ============================================================================
# 5. RECONSTRUCT AGING TRAJECTORIES (Pseudotime Analysis)
# ============================================================================
print("\n" + "="*80)
print("5. AGING TRAJECTORIES: Biological Age Progression")
print("="*80)

# Use PCA to create pseudotime ordering
# Create protein expression matrix per study
study_profiles = []

for study in df_valid['Study_ID'].unique():
    study_data = df_valid[df_valid['Study_ID'] == study]

    # Create protein abundance vector
    profile = study_data.groupby('Canonical_Gene_Symbol')['Zscore_Delta'].mean()

    study_profiles.append({
        'Study_ID': study,
        'Profile': profile,
        'Mean_Change': profile.mean(),
        'Tissue': study_data['Tissue'].iloc[0],
        'Species': study_data['Species'].iloc[0]
    })

# Align all profiles to common protein set
all_proteins = set()
for sp in study_profiles:
    all_proteins.update(sp['Profile'].index)

all_proteins = sorted(all_proteins)

# Create matrix
X = []
study_names = []

for sp in study_profiles:
    profile_vec = [sp['Profile'].get(p, 0) for p in all_proteins]
    X.append(profile_vec)
    study_names.append(sp['Study_ID'])

X = np.array(X)

# PCA for pseudotime
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PC1 represents main aging axis
pseudotime = X_pca[:, 0]

trajectory_df = pd.DataFrame({
    'Study_ID': study_names,
    'Pseudotime': pseudotime,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Tissue': [sp['Tissue'] for sp in study_profiles],
    'Species': [sp['Species'] for sp in study_profiles]
})

trajectory_df = trajectory_df.sort_values('Pseudotime')

print("\nAging Trajectory (Pseudotime Order):")
print(trajectory_df[['Study_ID', 'Pseudotime', 'Tissue', 'Species']].to_string())

print(f"\nPCA explains {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} = {pca.explained_variance_ratio_.sum():.1%} variance")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 24))

# 6.1 Timeline: Early → Middle → Late Markers
ax1 = plt.subplot(5, 2, 1)
temporal_summary = temporal_df.groupby('Temporal_Pattern').size()
colors = {'Late_Increase': '#d62728', 'Late_Decrease': '#2ca02c',
          'Chronic_Change': '#ff7f0e', 'Stable': '#7f7f7f'}
bars = ax1.barh(temporal_summary.index, temporal_summary.values,
                color=[colors.get(x, '#1f77b4') for x in temporal_summary.index])
ax1.set_xlabel('Number of Proteins')
ax1.set_title('Temporal Aging Patterns Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(temporal_summary.values):
    ax1.text(v + 5, i, str(v), va='center')

# 6.2 Top Early/Late Markers
ax2 = plt.subplot(5, 2, 2)
top_markers = pd.concat([
    temporal_df.head(15)[['Protein', 'Mean_Zscore_Delta']],
    temporal_df.tail(15)[['Protein', 'Mean_Zscore_Delta']]
])
colors_markers = ['#d62728' if x > 0 else '#2ca02c' for x in top_markers['Mean_Zscore_Delta']]
ax2.barh(range(len(top_markers)), top_markers['Mean_Zscore_Delta'], color=colors_markers)
ax2.set_yticks(range(len(top_markers)))
ax2.set_yticklabels(top_markers['Protein'], fontsize=8)
ax2.set_xlabel('Mean Z-score Delta')
ax2.set_title('Top 15 Increase (red) vs Decrease (green) Markers', fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linestyle='--', linewidth=1)

# 6.3 Temporal Consistency vs Magnitude
ax3 = plt.subplot(5, 2, 3)
scatter = ax3.scatter(temporal_df['Mean_Zscore_Delta'],
                     temporal_df['Consistency'],
                     c=temporal_df['N_Studies'],
                     s=50, alpha=0.6, cmap='viridis')
ax3.set_xlabel('Mean Z-score Delta')
ax3.set_ylabel('Consistency Score')
ax3.set_title('Temporal Reliability: Consistency vs Magnitude', fontsize=12, fontweight='bold')
ax3.axvline(0, color='red', linestyle='--', alpha=0.5)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('# Studies')

# Annotate top consistent markers
top_consistent = temporal_df.nlargest(5, 'Consistency')
for _, row in top_consistent.iterrows():
    ax3.annotate(row['Protein'],
                xy=(row['Mean_Zscore_Delta'], row['Consistency']),
                fontsize=8, alpha=0.7)

# 6.4 Matrisome Category Temporal Patterns
ax4 = plt.subplot(5, 2, 4)
category_temporal = temporal_df.groupby(['Matrisome_Division', 'Temporal_Pattern']).size().unstack(fill_value=0)
category_temporal.plot(kind='bar', stacked=True, ax=ax4,
                       color=[colors.get(x, '#1f77b4') for x in category_temporal.columns])
ax4.set_xlabel('Matrisome Division')
ax4.set_ylabel('Number of Proteins')
ax4.set_title('Temporal Patterns by Matrisome Category', fontsize=12, fontweight='bold')
ax4.legend(title='Pattern', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 6.5 Precedence Network (top relationships)
ax5 = plt.subplot(5, 2, 5)
if len(precedence_df) > 0:
    top_precedence = precedence_df.nlargest(20, 'Correlation')

    # Create network visualization
    unique_proteins = pd.concat([top_precedence['Source'], top_precedence['Target']]).unique()
    protein_idx = {p: i for i, p in enumerate(unique_proteins)}

    for _, edge in top_precedence.iterrows():
        source_idx = protein_idx[edge['Source']]
        target_idx = protein_idx[edge['Target']]
        ax5.arrow(source_idx, 0, target_idx - source_idx, 0,
                 head_width=0.3, head_length=0.5, fc='blue', ec='blue', alpha=0.3)

    ax5.scatter(range(len(unique_proteins)), [0]*len(unique_proteins), s=100, c='red', zorder=5)
    for i, p in enumerate(unique_proteins):
        ax5.text(i, 0.5, p, rotation=90, fontsize=8, ha='center')

    ax5.set_ylim(-1, 5)
    ax5.set_xlim(-1, len(unique_proteins))
    ax5.set_title('Temporal Precedence Network (A→B)', fontsize=12, fontweight='bold')
    ax5.axis('off')
else:
    ax5.text(0.5, 0.5, 'No strong precedence\nrelationships detected',
            ha='center', va='center', fontsize=12)
    ax5.axis('off')

# 6.6 Phase Transition Candidates
ax6 = plt.subplot(5, 2, 6)
if len(tipping_df) > 0:
    scatter6 = ax6.scatter(tipping_df['CV'], tipping_df['Kurtosis'],
                          c=abs(tipping_df['Mean_Zscore']), s=100,
                          alpha=0.7, cmap='Reds')
    ax6.set_xlabel('Coefficient of Variation')
    ax6.set_ylabel('Kurtosis')
    ax6.set_title('Phase Transition Candidates (bimodal proteins)', fontsize=12, fontweight='bold')
    ax6.axhline(-0.5, color='red', linestyle='--', alpha=0.5, label='Kurtosis threshold')
    ax6.axvline(1.5, color='blue', linestyle='--', alpha=0.5, label='CV threshold')
    ax6.legend()
    cbar = plt.colorbar(scatter6, ax=ax6)
    cbar.set_label('|Mean Z-score|')

    # Annotate top candidates
    for _, row in tipping_df.nlargest(5, 'CV').iterrows():
        ax6.annotate(row['Protein'], xy=(row['CV'], row['Kurtosis']),
                    fontsize=8, alpha=0.7)
else:
    ax6.text(0.5, 0.5, 'No phase transitions\ndetected',
            ha='center', va='center', fontsize=12)
    ax6.axis('off')

# 6.7 Aging Trajectory (PCA Pseudotime)
ax7 = plt.subplot(5, 2, 7)
species_colors = {'Homo sapiens': '#e41a1c', 'Mus musculus': '#377eb8', 'Bos taurus': '#4daf4a'}
for species in trajectory_df['Species'].unique():
    mask = trajectory_df['Species'] == species
    ax7.scatter(trajectory_df[mask]['PC1'], trajectory_df[mask]['PC2'],
               label=species, s=100, alpha=0.7, c=species_colors.get(species, 'gray'))

# Draw trajectory arrow
ax7.annotate('', xy=(trajectory_df.iloc[-1]['PC1'], trajectory_df.iloc[-1]['PC2']),
            xytext=(trajectory_df.iloc[0]['PC1'], trajectory_df.iloc[0]['PC2']),
            arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.5))

ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax7.set_title('Aging Trajectory (PCA Pseudotime)', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 6.8 Pseudotime Ordering
ax8 = plt.subplot(5, 2, 8)
trajectory_df_sorted = trajectory_df.sort_values('Pseudotime')
colors_trajectory = [species_colors.get(s, 'gray') for s in trajectory_df_sorted['Species']]
bars = ax8.barh(range(len(trajectory_df_sorted)), trajectory_df_sorted['Pseudotime'],
               color=colors_trajectory)
ax8.set_yticks(range(len(trajectory_df_sorted)))
ax8.set_yticklabels(trajectory_df_sorted['Study_ID'], fontsize=8)
ax8.set_xlabel('Pseudotime (Biological Age)')
ax8.set_title('Study Ordering by Biological Age', fontsize=12, fontweight='bold')
ax8.axvline(0, color='black', linestyle='--', linewidth=1)

# 6.9 Protein Loadings on PC1 (main aging axis)
ax9 = plt.subplot(5, 2, 9)
loadings = pca.components_[0]
protein_loadings = pd.DataFrame({
    'Protein': all_proteins,
    'Loading': loadings
}).sort_values('Loading', key=abs, ascending=False).head(30)

colors_loading = ['#d62728' if x > 0 else '#2ca02c' for x in protein_loadings['Loading']]
ax9.barh(range(len(protein_loadings)), protein_loadings['Loading'], color=colors_loading)
ax9.set_yticks(range(len(protein_loadings)))
ax9.set_yticklabels(protein_loadings['Protein'], fontsize=7)
ax9.set_xlabel('PC1 Loading')
ax9.set_title('Top 30 Drivers of Aging Trajectory', fontsize=12, fontweight='bold')
ax9.axvline(0, color='black', linestyle='--', linewidth=1)

# 6.10 Intervention Windows
ax10 = plt.subplot(5, 2, 10)
# Categorize proteins by intervention timing
intervention_timing = []
for _, row in temporal_df.iterrows():
    if row['Temporal_Pattern'] == 'Late_Increase':
        timing = 'Early Prevention'
    elif row['Temporal_Pattern'] == 'Late_Decrease':
        timing = 'Early Supplementation'
    elif row['Temporal_Pattern'] == 'Chronic_Change':
        timing = 'Continuous Monitoring'
    else:
        timing = 'Low Priority'

    intervention_timing.append({
        'Protein': row['Protein'],
        'Timing': timing,
        'Magnitude': abs(row['Mean_Zscore_Delta']),
        'Category': row['Matrisome_Category']
    })

intervention_df = pd.DataFrame(intervention_timing)
timing_summary = intervention_df.groupby('Timing').size().sort_values()
colors_timing = {'Early Prevention': '#d62728', 'Early Supplementation': '#2ca02c',
                'Continuous Monitoring': '#ff7f0e', 'Low Priority': '#7f7f7f'}
ax10.barh(timing_summary.index, timing_summary.values,
         color=[colors_timing.get(x, '#1f77b4') for x in timing_summary.index])
ax10.set_xlabel('Number of Proteins')
ax10.set_title('Intervention Window Classification', fontsize=12, fontweight='bold')
for i, v in enumerate(timing_summary.values):
    ax10.text(v + 5, i, str(v), va='center')

plt.tight_layout()
plt.savefig('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_temporal_dynamics_visualizations.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved visualizations")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("7. SAVING RESULTS")
print("="*80)

# Save all tables
temporal_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_temporal_ordering.csv', index=False)
print("✓ Saved temporal_ordering.csv")

if len(precedence_df) > 0:
    precedence_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_causal_precedence.csv', index=False)
    print("✓ Saved causal_precedence.csv")

if len(tipping_df) > 0:
    tipping_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_phase_transitions.csv', index=False)
    print("✓ Saved phase_transitions.csv")

trajectory_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_aging_trajectories.csv', index=False)
print("✓ Saved aging_trajectories.csv")

intervention_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_intervention_windows.csv', index=False)
print("✓ Saved intervention_windows.csv")

age_df.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_study_age_info.csv', index=False)
print("✓ Saved study_age_info.csv")

# Save protein loadings
protein_loadings.to_csv('/Users/Kravtsovd/projects/ecm-atlas/10_insights/agent_12_pc1_loadings.csv', index=False)
print("✓ Saved pc1_loadings.csv")

print("\n" + "="*80)
print("TEMPORAL DYNAMICS ANALYSIS COMPLETE")
print("="*80)
print("\nKey Files Generated:")
print("  - agent_12_temporal_ordering.csv")
print("  - agent_12_causal_precedence.csv (if detected)")
print("  - agent_12_phase_transitions.csv (if detected)")
print("  - agent_12_aging_trajectories.csv")
print("  - agent_12_intervention_windows.csv")
print("  - agent_12_study_age_info.csv")
print("  - agent_12_pc1_loadings.csv")
print("  - agent_12_temporal_dynamics_visualizations.png")
