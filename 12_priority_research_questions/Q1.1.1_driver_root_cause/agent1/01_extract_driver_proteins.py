#!/usr/bin/env python3
"""
Extract and analyze driver protein data from ECM-Atlas database
Focus: COL14A1, PCOLCE, and related driver proteins
Goal: Understand aging trajectories for epigenetic hypothesis generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load database
DB_PATH = "/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
OUTPUT_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas/12_priority_research_questions/Q1.1.1_driver_root_cause/agent1")

print("Loading ECM-Atlas database...")
df = pd.read_csv(DB_PATH)

print(f"Total rows: {len(df)}")
print(f"Unique proteins: {df['Canonical_Gene_Symbol'].nunique()}")

# Define driver proteins based on literature
DRIVER_PROTEINS = ['COL14A1', 'PCOLCE', 'COL6A5', 'COL21A1']

# Extract driver protein data
driver_data = df[df['Canonical_Gene_Symbol'].isin(DRIVER_PROTEINS)].copy()

print(f"\nDriver proteins found: {driver_data['Canonical_Gene_Symbol'].nunique()}")
print(driver_data['Canonical_Gene_Symbol'].value_counts())

# Calculate summary statistics per protein
summary_stats = driver_data.groupby('Canonical_Gene_Symbol').agg({
    'Zscore_Delta': ['mean', 'std', 'count'],
    'Study_ID': 'nunique',
    'Tissue': 'nunique',
    'Zscore_Old': 'mean',
    'Zscore_Young': 'mean'
}).round(3)

summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
summary_stats = summary_stats.rename(columns={
    'Zscore_Delta_mean': 'Mean_Zscore_Delta',
    'Zscore_Delta_std': 'Std_Zscore_Delta',
    'Zscore_Delta_count': 'N_Measurements',
    'Study_ID_nunique': 'N_Studies',
    'Tissue_nunique': 'N_Tissues',
    'Zscore_Old_mean': 'Mean_Zscore_Old',
    'Zscore_Young_mean': 'Mean_Zscore_Young'
})

print("\n" + "="*80)
print("DRIVER PROTEIN SUMMARY STATISTICS")
print("="*80)
print(summary_stats.to_string())

# Calculate directional consistency
consistency = driver_data.groupby('Canonical_Gene_Symbol').apply(
    lambda x: (x['Zscore_Delta'] < 0).sum() / len(x) * 100
)
consistency = pd.DataFrame({'Directional_Consistency_%': consistency.round(1)})

# Combine statistics
full_summary = pd.concat([summary_stats, consistency], axis=1)
full_summary = full_summary.sort_values('Mean_Zscore_Delta')

# Save summary
output_file = OUTPUT_DIR / "driver_protein_summary.csv"
full_summary.to_csv(output_file)
print(f"\nSaved: {output_file}")

# Detailed breakdown by study and tissue
detailed = driver_data.groupby(['Canonical_Gene_Symbol', 'Study_ID', 'Tissue']).agg({
    'Zscore_Delta': 'mean',
    'Zscore_Old': 'mean',
    'Zscore_Young': 'mean',
    'Species': 'first',
    'Method': 'first'
}).round(3).reset_index()

detailed = detailed.sort_values(['Canonical_Gene_Symbol', 'Zscore_Delta'])
output_file2 = OUTPUT_DIR / "driver_protein_detailed.csv"
detailed.to_csv(output_file2, index=False)
print(f"Saved: {output_file2}")

# Find proteins with similar decline patterns (potential additional drivers)
print("\n" + "="*80)
print("IDENTIFYING ADDITIONAL DRIVER CANDIDATES")
print("="*80)

# Calculate protein-level statistics
protein_stats = df.groupby('Canonical_Gene_Symbol').agg({
    'Zscore_Delta': ['mean', 'std', 'count'],
    'Study_ID': 'nunique',
    'Tissue': 'nunique'
}).reset_index()

protein_stats.columns = ['Gene_Symbol', 'Mean_Zscore_Delta', 'Std', 'N_Measurements', 'N_Studies', 'N_Tissues']

# Filter for strong decliners (similar to COL14A1/PCOLCE)
# Criteria: Mean Δz < -0.5, at least 3 studies, at least 3 tissues, consistency > 70%
strong_decliners = protein_stats[
    (protein_stats['Mean_Zscore_Delta'] < -0.5) &
    (protein_stats['N_Studies'] >= 3) &
    (protein_stats['N_Tissues'] >= 3)
].copy()

# Calculate consistency for these candidates
consistency_vals = []
for gene in strong_decliners['Gene_Symbol']:
    gene_data = df[df['Canonical_Gene_Symbol'] == gene]
    cons = (gene_data['Zscore_Delta'] < 0).sum() / len(gene_data) * 100
    consistency_vals.append(cons)

strong_decliners['Consistency_%'] = consistency_vals
strong_decliners = strong_decliners[strong_decliners['Consistency_%'] >= 70]
strong_decliners = strong_decliners.sort_values('Mean_Zscore_Delta')

print(f"\nFound {len(strong_decliners)} potential driver candidates:")
print(strong_decliners.head(10).to_string(index=False))

output_file3 = OUTPUT_DIR / "driver_candidates.csv"
strong_decliners.to_csv(output_file3, index=False)
print(f"\nSaved: {output_file3}")

# Age-related analysis (if age data available)
print("\n" + "="*80)
print("AGE CORRELATION ANALYSIS")
print("="*80)

# Check what age-related columns we have
age_cols = [col for col in df.columns if 'age' in col.lower() or 'old' in col.lower() or 'young' in col.lower()]
print(f"Age-related columns: {age_cols}")

# For driver proteins, show the young vs old abundance patterns
for protein in DRIVER_PROTEINS:
    prot_data = driver_data[driver_data['Canonical_Gene_Symbol'] == protein]
    if len(prot_data) > 0:
        print(f"\n{protein}:")
        print(f"  Mean Z-score change: {prot_data['Zscore_Delta'].mean():.3f}")
        print(f"  Old abundance mean: {prot_data['Abundance_Old'].mean():.1f}")
        print(f"  Young abundance mean: {prot_data['Abundance_Young'].mean():.1f}")
        print(f"  Direction: {'DECLINE' if prot_data['Zscore_Delta'].mean() < 0 else 'INCREASE'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
