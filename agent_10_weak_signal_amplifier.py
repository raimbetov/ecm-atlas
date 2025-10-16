#!/usr/bin/env python3
"""
Agent 10: Weak Signal Amplifier
Identifies proteins with small but consistent changes across many tissues/studies
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import combine_pvalues
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Filter to rows with valid Zscore_Delta (both old and young present)
df_valid = df[df['Zscore_Delta'].notna()].copy()

print(f"Total rows: {len(df)}")
print(f"Rows with Zscore_Delta: {len(df_valid)}")
print(f"Unique proteins: {df_valid['Canonical_Gene_Symbol'].nunique()}")
print(f"Unique studies: {df_valid['Study_ID'].nunique()}")

# 1. WEAK SIGNAL DETECTION
print("\n=== WEAK SIGNAL DETECTION ===")

# Group by protein
protein_stats = []
for gene in df_valid['Canonical_Gene_Symbol'].unique():
    gene_data = df_valid[df_valid['Canonical_Gene_Symbol'] == gene]

    zscore_deltas = gene_data['Zscore_Delta'].values
    n_studies = gene_data['Study_ID'].nunique()
    n_measurements = len(zscore_deltas)

    if n_measurements < 2:
        continue

    # Calculate key metrics
    mean_delta = np.mean(zscore_deltas)
    abs_mean_delta = np.abs(mean_delta)
    std_delta = np.std(zscore_deltas)

    # Count directional consistency
    positive_count = np.sum(zscore_deltas > 0)
    negative_count = np.sum(zscore_deltas < 0)
    direction_consistency = max(positive_count, negative_count) / n_measurements
    dominant_direction = 'increase' if positive_count > negative_count else 'decrease'

    # Weak signal criteria:
    # 1. Small effect: |mean| between 0.3-0.8
    # 2. High consistency: ≥8 measurements, same direction ≥70%
    # 3. Low variance: std < 0.3
    is_weak_signal = (
        0.3 <= abs_mean_delta <= 0.8 and
        n_measurements >= 8 and
        direction_consistency >= 0.70 and
        std_delta < 0.3
    )

    # Meta-analysis: calculate combined p-value using Fisher's method
    # For each measurement, calculate one-sided p-value
    p_values = []
    for z in zscore_deltas:
        if dominant_direction == 'increase':
            p = 1 - stats.norm.cdf(z)
        else:
            p = stats.norm.cdf(z)
        p_values.append(max(p, 1e-16))  # avoid zero

    # Combine p-values
    try:
        chi2_stat, combined_p = combine_pvalues(p_values, method='fisher')
    except:
        combined_p = 1.0

    # Calculate cumulative effect size
    cumulative_effect = np.sum(zscore_deltas)

    protein_stats.append({
        'Gene_Symbol': gene,
        'N_Measurements': n_measurements,
        'N_Studies': n_studies,
        'Mean_Zscore_Delta': mean_delta,
        'Abs_Mean_Zscore_Delta': abs_mean_delta,
        'Std_Zscore_Delta': std_delta,
        'Direction_Consistency': direction_consistency,
        'Dominant_Direction': dominant_direction,
        'Cumulative_Effect': cumulative_effect,
        'Combined_P_Value': combined_p,
        'Is_Weak_Signal': is_weak_signal
    })

df_proteins = pd.DataFrame(protein_stats)

# 2. HIGH-CONSISTENCY WEAK SIGNALS
print(f"\nTotal proteins analyzed: {len(df_proteins)}")
weak_signals = df_proteins[df_proteins['Is_Weak_Signal'] == True].copy()
print(f"Weak signal proteins identified: {len(weak_signals)}")

if len(weak_signals) > 0:
    # Sort by combined p-value (best meta-analysis result)
    weak_signals = weak_signals.sort_values('Combined_P_Value')

    # Also calculate random effects meta-analysis estimate
    print("\n=== TOP 20 WEAK SIGNAL PROTEINS ===")
    print(weak_signals[['Gene_Symbol', 'N_Measurements', 'N_Studies',
                        'Mean_Zscore_Delta', 'Std_Zscore_Delta',
                        'Direction_Consistency', 'Combined_P_Value']].head(20).to_string(index=False))

# 3. EXPAND CRITERIA - MODERATE WEAK SIGNALS
print("\n=== MODERATE WEAK SIGNALS (relaxed criteria) ===")
moderate_weak = df_proteins[
    (df_proteins['Abs_Mean_Zscore_Delta'].between(0.3, 1.0)) &
    (df_proteins['N_Measurements'] >= 6) &
    (df_proteins['Direction_Consistency'] >= 0.65) &
    (df_proteins['Std_Zscore_Delta'] < 0.4)
].sort_values('Combined_P_Value')

print(f"Moderate weak signal proteins: {len(moderate_weak)}")
print(moderate_weak[['Gene_Symbol', 'N_Measurements', 'N_Studies',
                    'Mean_Zscore_Delta', 'Direction_Consistency',
                    'Combined_P_Value']].head(30).to_string(index=False))

# 4. MATRISOME CATEGORY ENRICHMENT
print("\n=== WEAK SIGNALS BY MATRISOME CATEGORY ===")
for idx, row in moderate_weak.head(50).iterrows():
    gene = row['Gene_Symbol']
    gene_categories = df_valid[df_valid['Canonical_Gene_Symbol'] == gene]['Matrisome_Division'].unique()
    if len(gene_categories) > 0:
        moderate_weak.loc[idx, 'Matrisome_Division'] = gene_categories[0]

category_counts = moderate_weak['Matrisome_Division'].value_counts()
print(category_counts)

# 5. PATHWAY-LEVEL ANALYSIS (ECM-specific)
print("\n=== COLLAGEN FAMILY WEAK SIGNALS ===")
collagens = moderate_weak[moderate_weak['Gene_Symbol'].str.startswith('COL')]
print(collagens[['Gene_Symbol', 'N_Measurements', 'Mean_Zscore_Delta',
                 'Direction_Consistency', 'Combined_P_Value']].to_string(index=False))

print("\n=== METALLOPROTEINASE/INHIBITOR WEAK SIGNALS ===")
mmps = moderate_weak[moderate_weak['Gene_Symbol'].str.contains('MMP|TIMP|ADAM')]
print(mmps[['Gene_Symbol', 'N_Measurements', 'Mean_Zscore_Delta',
            'Direction_Consistency', 'Combined_P_Value']].to_string(index=False))

# 6. STATISTICAL POWER ANALYSIS
print("\n=== STATISTICAL POWER ANALYSIS ===")
# For a typical weak signal (delta=0.5, std=0.25), how many studies needed?
effect_sizes = [0.3, 0.5, 0.8]
alpha = 0.05
power = 0.80

print(f"For power={power}, alpha={alpha}:")
for effect in effect_sizes:
    # Approximate sample size for one-sample t-test
    # n ≈ (Zα + Zβ)² * (2σ²/δ²)
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    std_assumed = 0.25

    n = ((z_alpha + z_beta)**2 * 2 * std_assumed**2) / effect**2
    print(f"  Effect size {effect}: ~{int(np.ceil(n))} measurements needed")

# 7. EARLY WARNING SIGNALS
print("\n=== EARLY WARNING CANDIDATES ===")
print("Proteins with consistent weak signals across diverse tissues may indicate early aging:")
early_warning = moderate_weak[moderate_weak['N_Studies'] >= 3].copy()

# Get tissue diversity for top candidates
for idx, row in early_warning.head(20).iterrows():
    gene = row['Gene_Symbol']
    tissues = df_valid[df_valid['Canonical_Gene_Symbol'] == gene]['Tissue'].unique()
    early_warning.loc[idx, 'N_Tissues'] = len(tissues)
    early_warning.loc[idx, 'Tissues'] = ', '.join(list(tissues)[:3])

print(early_warning[['Gene_Symbol', 'N_Studies', 'N_Tissues', 'Mean_Zscore_Delta',
                     'Direction_Consistency', 'Tissues']].head(20).to_string(index=False))

# 8. SAVE RESULTS
print("\n=== SAVING RESULTS ===")
output_file = '/Users/Kravtsovd/projects/ecm-atlas/weak_signal_proteins.csv'
moderate_weak.to_csv(output_file, index=False)
print(f"Saved {len(moderate_weak)} weak signal proteins to {output_file}")

# Save detailed protein-level data
all_proteins_file = '/Users/Kravtsovd/projects/ecm-atlas/all_protein_statistics.csv'
df_proteins.to_csv(all_proteins_file, index=False)
print(f"Saved {len(df_proteins)} protein statistics to {all_proteins_file}")

print("\n=== KEY FINDINGS ===")
print(f"1. Identified {len(weak_signals)} high-confidence weak signals")
print(f"2. Identified {len(moderate_weak)} moderate weak signals")
print(f"3. Most enriched category: {category_counts.index[0] if len(category_counts) > 0 else 'N/A'}")
print(f"4. For typical weak signal (δ=0.5), need ~{int(np.ceil(((stats.norm.ppf(0.975) + stats.norm.ppf(0.80))**2 * 2 * 0.25**2) / 0.5**2))} measurements")
print("\nAnalysis complete!")
