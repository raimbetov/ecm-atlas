#!/usr/bin/env python3
"""
Transform ECM-Atlas Dataset from Wide to Long Format

Purpose: Convert wide format (Abundance_Old/Young columns) to long format
         (single Abundance column with Age_Group identifier) for batch
         correction analyses

Input:  ../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv (wide)
Output: ../data/merged_ecm_aging_long_format.csv (long)

Author: Exploratory Batch Correction Analysis
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

print("=" * 70)
print("ECM-ATLAS: WIDE TO LONG FORMAT TRANSFORMATION")
print("=" * 70)
print()

# =============================================================================
# 1. SETUP
# =============================================================================

print("Setting up environment...")

# Create output directory
Path("../data").mkdir(parents=True, exist_ok=True)

print("✓ Output directory ready\n")

# =============================================================================
# 2. LOAD WIDE FORMAT DATA
# =============================================================================

print("Loading wide format dataset...")

input_path = "../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

if not Path(input_path).exists():
    raise FileNotFoundError(f"Dataset not found: {input_path}")

df_wide = pd.read_csv(input_path)

print(f"✓ Loaded {len(df_wide):,} rows, {len(df_wide.columns)} columns")
print(f"✓ Unique proteins: {df_wide['Protein_ID'].nunique():,}")
print(f"✓ Unique studies: {df_wide['Study_ID'].nunique()}")
print()

# Display column structure
print("Wide format columns:")
print(f"  Abundance: Abundance_Old, Abundance_Young")
print(f"  Z-scores: Zscore_Old, Zscore_Young")
print()

# =============================================================================
# 3. TRANSFORM TO LONG FORMAT
# =============================================================================

print("=" * 70)
print("TRANSFORMING TO LONG FORMAT")
print("=" * 70)
print()

print("Creating long format records...")

# Base columns to preserve
base_columns = [
    'Protein_ID', 'Gene_Symbol', 'Canonical_Gene_Symbol',
    'Study_ID', 'Dataset_Name', 'Tissue', 'Tissue_Compartment',
    'Organ', 'Compartment', 'Species', 'Method',
    'Matrisome_Category', 'Matrisome_Division',
    'Protein_Name', 'Match_Level', 'Match_Confidence',
    'Data_Quality'
]

# Keep only columns that exist
base_columns = [col for col in base_columns if col in df_wide.columns]

records = []

for idx, row in df_wide.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing row {idx:,}/{len(df_wide):,}...", end='\r')

    # Extract base information
    base = {col: row[col] for col in base_columns}

    # Add Old age group record
    if pd.notna(row['Abundance_Old']):
        old_record = base.copy()
        old_record.update({
            'Age_Group': 'Old',
            'Abundance': row['Abundance_Old'],
            'Z_score': row['Zscore_Old'],
            'N_Profiles': row.get('N_Profiles_Old', np.nan)
        })
        records.append(old_record)

    # Add Young age group record
    if pd.notna(row['Abundance_Young']):
        young_record = base.copy()
        young_record.update({
            'Age_Group': 'Young',
            'Abundance': row['Abundance_Young'],
            'Z_score': row['Zscore_Young'],
            'N_Profiles': row.get('N_Profiles_Young', np.nan)
        })
        records.append(young_record)

print()
print(f"✓ Created {len(records):,} long format records")

# Create DataFrame
df_long = pd.DataFrame(records)

print(f"✓ Long format shape: {df_long.shape}")
print()

# =============================================================================
# 4. VALIDATE TRANSFORMATION
# =============================================================================

print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)
print()

# Check 1: Age group distribution
age_counts = df_long['Age_Group'].value_counts()
print("Age group distribution:")
print(f"  Old:   {age_counts.get('Old', 0):,} records")
print(f"  Young: {age_counts.get('Young', 0):,} records")
print()

# Check 2: Missing values
missing_abundance = df_long['Abundance'].isna().sum()
missing_zscore = df_long['Z_score'].isna().sum()
print("Missing values:")
print(f"  Abundance: {missing_abundance:,} ({missing_abundance/len(df_long)*100:.1f}%)")
print(f"  Z_score:   {missing_zscore:,} ({missing_zscore/len(df_long)*100:.1f}%)")
print()

# Check 3: Studies and proteins
print("Summary statistics:")
print(f"  Unique proteins: {df_long['Protein_ID'].nunique():,}")
print(f"  Unique studies: {df_long['Study_ID'].nunique()}")
print(f"  Unique tissues: {df_long['Tissue_Compartment'].nunique()}")
print()

# Check 4: Z-score statistics
zscore_stats = df_long['Z_score'].describe()
print("Z-score distribution:")
print(f"  Mean: {zscore_stats['mean']:.3f} (should be ~0)")
print(f"  Std:  {zscore_stats['std']:.3f} (should be ~1)")
print(f"  Min:  {zscore_stats['min']:.3f}")
print(f"  Max:  {zscore_stats['max']:.3f}")
print()

# Check 5: Sample sizes per group
samples_per_study = df_long.groupby(['Study_ID', 'Age_Group']).size().reset_index(name='Count')
print("Sample sizes per study:")
for study in df_long['Study_ID'].unique():
    study_data = samples_per_study[samples_per_study['Study_ID'] == study]
    old_count = study_data[study_data['Age_Group'] == 'Old']['Count'].values
    young_count = study_data[study_data['Age_Group'] == 'Young']['Count'].values
    print(f"  {study}: Old={old_count[0] if len(old_count) > 0 else 0}, Young={young_count[0] if len(young_count) > 0 else 0}")
print()

# =============================================================================
# 5. SAVE OUTPUT
# =============================================================================

print("=" * 70)
print("SAVING OUTPUT")
print("=" * 70)
print()

output_path = "../data/merged_ecm_aging_long_format.csv"

# Save long format data
df_long.to_csv(output_path, index=False)

file_size_mb = Path(output_path).stat().st_size / 1e6

print(f"✓ Long format data saved: {output_path}")
print(f"  Size: {file_size_mb:.2f} MB")
print(f"  Rows: {len(df_long):,}")
print(f"  Columns: {len(df_long.columns)}")
print()

# Save transformation metadata
metadata = {
    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_file': str(input_path),
    'output_file': str(output_path),
    'input_rows': len(df_wide),
    'output_rows': len(df_long),
    'expansion_ratio': len(df_long) / len(df_wide),
    'n_proteins': int(df_long['Protein_ID'].nunique()),
    'n_studies': int(df_long['Study_ID'].nunique()),
    'n_tissues': int(df_long['Tissue_Compartment'].nunique()),
    'age_groups': {
        'Old': int(age_counts.get('Old', 0)),
        'Young': int(age_counts.get('Young', 0))
    },
    'missing_values': {
        'Abundance': int(missing_abundance),
        'Z_score': int(missing_zscore)
    },
    'zscore_stats': {
        'mean': float(zscore_stats['mean']),
        'std': float(zscore_stats['std']),
        'min': float(zscore_stats['min']),
        'max': float(zscore_stats['max'])
    }
}

metadata_path = "../data/transformation_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved: {metadata_path}")
print()

# =============================================================================
# 6. SUMMARY
# =============================================================================

print("=" * 70)
print("TRANSFORMATION SUMMARY")
print("=" * 70)
print()

print("Input (Wide Format):")
print(f"  - File: {input_path}")
print(f"  - Rows: {len(df_wide):,}")
print(f"  - Format: One row per protein-study (separate Old/Young columns)")
print()

print("Output (Long Format):")
print(f"  - File: {output_path}")
print(f"  - Rows: {len(df_long):,} ({len(df_long)/len(df_wide):.1f}× expansion)")
print(f"  - Format: Two rows per protein-study (one Old, one Young)")
print()

print("Data Quality:")
print(f"  - Proteins: {df_long['Protein_ID'].nunique():,}")
print(f"  - Studies: {df_long['Study_ID'].nunique()}")
print(f"  - Age groups: Old={age_counts.get('Old', 0):,}, Young={age_counts.get('Young', 0):,}")
print(f"  - Z-score mean: {zscore_stats['mean']:.3f} (target: 0)")
print(f"  - Z-score std: {zscore_stats['std']:.3f} (target: 1)")
print()

print("Next Steps:")
print("  1. Update script paths to use: ../data/merged_ecm_aging_long_format.csv")
print("  2. Run ComBat correction: combat_correction/01_apply_combat.R")
print("  3. Run Percentile normalization: percentile_normalization/01_apply_percentile.py")
print()

print("=" * 70)
print("TRANSFORMATION COMPLETE")
print("=" * 70)
