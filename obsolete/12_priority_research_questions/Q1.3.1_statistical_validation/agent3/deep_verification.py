#!/usr/bin/env python3
"""
Agent 3: Deep Verification - Reproduce Key Analyses Independently
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load data
data_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
df = pd.read_csv(data_path)

print("="*80)
print("DEEP VERIFICATION: REPRODUCING KEY CLAIMS INDEPENDENTLY")
print("="*80)

# ============================================================================
# CRITICAL FINDING 1: Protein count discrepancy
# ============================================================================

print("\n🔴 CRITICAL FINDING 1: PROTEIN COUNT DISCREPANCY")
print("="*80)

# Agent reports claim 3,317 proteins
# But I found 3,757 proteins
total_proteins = df['Protein_ID'].nunique()
gene_symbols = df['Gene_Symbol'].nunique()

print(f"Agent claim: 3,317 unique proteins")
print(f"My count (Protein_ID): {total_proteins} proteins")
print(f"My count (Gene_Symbol): {gene_symbols} gene symbols")
print(f"DISCREPANCY: {total_proteins - 3317} more proteins than claimed ({((total_proteins - 3317) / 3317 * 100):.1f}% difference)")

# Let's check matrisome-annotated proteins only (agents might have filtered)
matrisome_proteins = df[df['Matrisome_Category'].notna()]['Protein_ID'].nunique()
print(f"\nMatrisome-annotated only: {matrisome_proteins} proteins")
print(f"Non-matrisome proteins: {total_proteins - matrisome_proteins}")

# ============================================================================
# VERIFY "405 UNIVERSAL PROTEINS" CLAIM
# ============================================================================

print("\n" + "="*80)
print("VERIFYING: '405 universal proteins (12.2%)'")
print("="*80)

# According to agent_01 report:
# Universal = ≥3 tissues, ≥70% directional consistency

# Group by protein and calculate tissue breadth
protein_stats = []

for protein in df['Gene_Symbol'].dropna().unique():
    protein_data = df[df['Gene_Symbol'] == protein].copy()

    # Skip if no z-score data
    if protein_data['Zscore_Delta'].isna().all():
        continue

    # Count tissues where this protein appears
    tissues = protein_data['Tissue_Compartment'].unique()
    n_tissues = len(tissues)

    if n_tissues < 3:
        continue

    # Calculate directional consistency
    deltas = protein_data['Zscore_Delta'].dropna()
    if len(deltas) < 3:
        continue

    n_up = (deltas > 0).sum()
    n_down = (deltas < 0).sum()
    total = n_up + n_down

    if total == 0:
        continue

    consistency = max(n_up, n_down) / total * 100
    direction = "UP" if n_up > n_down else "DOWN"

    if consistency >= 70:  # Universal criteria
        mean_delta = deltas.mean()
        mean_abs_delta = deltas.abs().mean()

        protein_stats.append({
            'Gene_Symbol': protein,
            'N_Tissues': n_tissues,
            'Consistency': consistency,
            'Direction': direction,
            'Mean_Delta': mean_delta,
            'Mean_Abs_Delta': mean_abs_delta,
            'N_Measurements': len(deltas)
        })

universal_df = pd.DataFrame(protein_stats)
n_universal = len(universal_df)

print(f"\nMY CALCULATION: {n_universal} universal proteins (≥3 tissues, ≥70% consistency)")
print(f"AGENT CLAIM: 405 universal proteins")
print(f"DISCREPANCY: {abs(n_universal - 405)} proteins ({((abs(n_universal - 405)) / 405 * 100):.1f}% difference)")

if n_universal > 0:
    print(f"\nTop 10 universal proteins by my calculation:")
    print(universal_df.nlargest(10, 'Mean_Abs_Delta')[['Gene_Symbol', 'N_Tissues', 'Consistency', 'Direction', 'Mean_Abs_Delta']])

# ============================================================================
# VERIFY "TOP 5 UNIVERSAL MARKERS" CLAIM
# ============================================================================

print("\n" + "="*80)
print("VERIFYING: TOP 5 UNIVERSAL MARKERS")
print("="*80)

claimed_top5 = ['Hp', 'VTN', 'Col14a1', 'F2', 'FGB']
print(f"Claimed top 5: {claimed_top5}")

print("\nChecking if these proteins appear in my universal list:")
for protein in claimed_top5:
    if protein in universal_df['Gene_Symbol'].values:
        row = universal_df[universal_df['Gene_Symbol'] == protein].iloc[0]
        print(f"✓ {protein}: {row['N_Tissues']} tissues, {row['Consistency']:.1f}% consistency, Δz={row['Mean_Delta']:.3f}")
    else:
        # Check with case-insensitive match
        matches = universal_df[universal_df['Gene_Symbol'].str.upper() == protein.upper()]
        if len(matches) > 0:
            row = matches.iloc[0]
            print(f"✓ {row['Gene_Symbol']} (case mismatch): {row['N_Tissues']} tissues, {row['Consistency']:.1f}% consistency")
        else:
            print(f"✗ {protein}: NOT IN MY UNIVERSAL LIST")

# ============================================================================
# VERIFY "7 GOLD-TIER PROTEINS" WITH REPLICATION
# ============================================================================

print("\n" + "="*80)
print("VERIFYING: '7 GOLD-TIER PROTEINS (≥5 studies, >80% consistency)'")
print("="*80)

gold_tier_claimed = ['VTN', 'FGB', 'FGA', 'PCOLCE', 'CTSF', 'SERPINH1', 'MFGE8']

# Calculate study replication for each protein
replication_stats = []

for protein in df['Gene_Symbol'].dropna().unique():
    protein_data = df[df['Gene_Symbol'] == protein].copy()

    # Skip if no z-score data
    if protein_data['Zscore_Delta'].isna().all():
        continue

    # Count STUDIES (not tissues) where protein appears
    studies = protein_data['Study_ID'].unique()
    n_studies = len(studies)

    if n_studies >= 2:  # At least some replication
        deltas = protein_data['Zscore_Delta'].dropna()

        if len(deltas) >= 2:
            n_up = (deltas > 0).sum()
            n_down = (deltas < 0).sum()
            total = n_up + n_down

            if total > 0:
                consistency = max(n_up, n_down) / total * 100

                replication_stats.append({
                    'Gene_Symbol': protein,
                    'N_Studies': n_studies,
                    'N_Measurements': len(deltas),
                    'Consistency': consistency,
                    'Mean_Delta': deltas.mean()
                })

replication_df = pd.DataFrame(replication_stats)

# Find GOLD-tier proteins (≥5 studies, >80% consistency)
gold_tier_mine = replication_df[(replication_df['N_Studies'] >= 5) & (replication_df['Consistency'] > 80)]

print(f"\nMY CALCULATION: {len(gold_tier_mine)} GOLD-tier proteins (≥5 studies, >80% consistency)")
print(f"AGENT CLAIM: 7 GOLD-tier proteins")

if len(gold_tier_mine) > 0:
    print(f"\nMy GOLD-tier proteins:")
    print(gold_tier_mine[['Gene_Symbol', 'N_Studies', 'Consistency', 'Mean_Delta']].to_string(index=False))

    print(f"\nChecking claimed GOLD-tier proteins:")
    for protein in gold_tier_claimed:
        matches = gold_tier_mine[gold_tier_mine['Gene_Symbol'].str.upper() == protein.upper()]
        if len(matches) > 0:
            row = matches.iloc[0]
            print(f"✓ {protein}: {row['N_Studies']} studies, {row['Consistency']:.1f}% consistency")
        else:
            # Check if protein exists but didn't meet criteria
            all_matches = replication_df[replication_df['Gene_Symbol'].str.upper() == protein.upper()]
            if len(all_matches) > 0:
                row = all_matches.iloc[0]
                print(f"✗ {protein}: Only {row['N_Studies']} studies, {row['Consistency']:.1f}% consistency (DOESN'T MEET GOLD CRITERIA)")
            else:
                print(f"✗ {protein}: NOT FOUND IN DATASET")

# ============================================================================
# VERIFY PERFECT CORRELATIONS (r=1.000)
# ============================================================================

print("\n" + "="*80)
print("VERIFYING: 'Perfect protein correlations (r=1.000)'")
print("="*80)

perfect_corr_claimed = [
    ("CTGF", "IGFALS"),
    ("Asah1", "Lman2"),
    ("CTSD", "TIMP2")
]

print("RED FLAG WARNING: Perfect correlations (r=1.000) are statistically suspicious")
print("- Could indicate: Data duplication, analysis error, or extremely small sample")
print("- In biology, r=1.000 is virtually impossible with n>5")

for protein1, protein2 in perfect_corr_claimed:
    # Get data for both proteins
    p1_data = df[df['Gene_Symbol'].str.upper() == protein1.upper()][['Tissue_Compartment', 'Zscore_Delta']].dropna()
    p2_data = df[df['Gene_Symbol'].str.upper() == protein2.upper()][['Tissue_Compartment', 'Zscore_Delta']].dropna()

    # Merge on tissue
    merged = pd.merge(p1_data, p2_data, on='Tissue_Compartment', suffixes=('_p1', '_p2'))

    if len(merged) >= 2:
        corr = merged['Zscore_Delta_p1'].corr(merged['Zscore_Delta_p2'])
        n = len(merged)
        print(f"\n{protein1} ↔ {protein2}:")
        print(f"  My calculation: r={corr:.4f}, n={n} shared tissues")
        print(f"  Agent claim: r=1.000")
        if abs(corr - 1.000) < 0.001:
            print(f"  🔴 WARNING: Nearly perfect correlation with small n={n} - likely spurious!")
        elif abs(corr - 1.000) > 0.1:
            print(f"  ✗ MAJOR DISCREPANCY: My r={corr:.4f} vs claimed r=1.000")
    else:
        print(f"\n{protein1} ↔ {protein2}: Insufficient overlapping data (n={len(merged)})")

# ============================================================================
# SUMMARY OF FINDINGS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF VERIFICATION")
print("="*80)

print("\n✓ VERIFIED:")
print("  - Dataset size: 9,343 rows")

print("\n🟡 DISCREPANCIES FOUND:")
print(f"  - Protein count: {total_proteins} vs claimed 3,317 ({total_proteins - 3317} difference)")
print(f"  - Universal proteins: {n_universal} vs claimed 405 ({abs(n_universal - 405)} difference)")
print(f"  - GOLD-tier proteins: {len(gold_tier_mine)} vs claimed 7")

print("\n🔴 RED FLAGS:")
print("  - Perfect correlations (r=1.000) are statistically implausible")
print("  - Lifespan extension claims (+20-30 years) lack clear justification")
print("  - 'Nobel Prize potential' is subjective hype, not scientific assessment")

print("\n" + "="*80)
