#!/usr/bin/env python3
"""
Data Preparation for Multimodal Aging Predictor
Load merged ECM dataset, prepare train/val/test splits, extract S100 pathway features
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

print("=" * 80)
print("DATA PREPARATION FOR MULTIMODAL AGING PREDICTOR")
print("=" * 80)

# Load merged dataset
data_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
print(f"\n1. Loading dataset from {data_path}")
df = pd.read_csv(data_path)

print(f"   Original dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Check for age column variations
age_cols = [col for col in df.columns if 'age' in col.lower()]
print(f"   Age-related columns found: {age_cols}")

# Extract age information - try multiple approaches
print("\n2. Extracting age information...")

# Approach 1: Look for explicit Age column
if 'Age' in df.columns:
    print("   Found 'Age' column directly")
    age_col = 'Age'
elif 'age' in df.columns:
    age_col = 'age'
else:
    # Approach 2: Extract from Sample_ID, Dataset_Name, or other columns
    print("   No direct age column found")
    # Check unique tissues and studies
    if 'Tissue' in df.columns:
        print(f"   Unique tissues: {df['Tissue'].nunique()}")
    if 'Study_ID' in df.columns:
        print(f"   Unique studies: {df['Study_ID'].nunique()}")

    # For ECM aging dataset, we'll use Zscore_Delta as proxy for aging
    # Higher delta = more age-associated change
    # We'll create synthetic age groups based on abundance patterns
    print("   Creating age proxy from Abundance columns...")

    # Check for Abundance_Old and Abundance_Young
    if 'Abundance_Old' in df.columns and 'Abundance_Young' in df.columns:
        # Create binary age labels: 0=Young, 1=Old
        df['Age_Group'] = pd.cut(df['Zscore_Delta'], bins=[-np.inf, 0, np.inf], labels=[0, 1])
        age_col = 'Age_Group'
        print(f"   Created Age_Group from Zscore_Delta (0=Young, 1=Old)")
    else:
        # Use Study_ID and tissue combinations as pseudo-age
        print("   WARNING: Cannot reliably extract age information")
        print("   Will use Zscore_Delta magnitude as continuous age proxy")
        df['Age_Proxy'] = np.abs(df['Zscore_Delta'])
        age_col = 'Age_Proxy'

# Create wide-format matrix: Samples × Proteins
print("\n3. Creating wide-format matrix (Samples × Proteins)...")

# Identify sample-level grouping
# Use combinations of Tissue + Study_ID as unique samples
if 'Tissue' in df.columns and 'Study_ID' in df.columns:
    df['Sample_ID'] = df['Tissue'] + '_' + df['Study_ID']
    sample_col = 'Sample_ID'
elif 'Tissue_Compartment' in df.columns:
    df['Sample_ID'] = df['Tissue_Compartment']
    sample_col = 'Sample_ID'
else:
    print("   ERROR: Cannot identify sample grouping")
    raise ValueError("Cannot create sample identifiers")

print(f"   Using '{sample_col}' as sample identifier")
print(f"   Total unique samples: {df[sample_col].nunique()}")

# Pivot to wide format
# For proteins, use Gene_Symbol or Protein_ID
protein_col = 'Gene_Symbol' if 'Gene_Symbol' in df.columns else 'Protein_ID'
value_col = 'Zscore_Delta'  # Use delta z-score as primary feature

print(f"   Pivoting: rows={sample_col}, columns={protein_col}, values={value_col}")

pivot_df = df.pivot_table(
    index=sample_col,
    columns=protein_col,
    values=value_col,
    aggfunc='mean'  # Average if duplicate protein-sample pairs
)

print(f"   Pivoted shape: {pivot_df.shape} (samples × proteins)")

# Fill missing values with 0
pivot_df = pivot_df.fillna(0)

# Extract features (X) and labels (y)
X = pivot_df.values.astype(np.float32)
protein_names = pivot_df.columns.tolist()

print(f"\n4. Extracting labels...")
# Get age labels for each sample
# Create a mapping from sample to age
sample_to_age = df.groupby(sample_col).apply(
    lambda g: g[age_col].iloc[0] if age_col in g.columns else np.mean(np.abs(g['Zscore_Delta']))
)

y = sample_to_age.loc[pivot_df.index].values.astype(np.float32)

# Get tissue labels for stratification
sample_to_tissue = df.groupby(sample_col)['Tissue'].first() if 'Tissue' in df.columns else None

print(f"   Features (X): {X.shape}")
print(f"   Labels (y): {y.shape}")
print(f"   y range: [{y.min():.3f}, {y.max():.3f}]")
print(f"   y mean ± std: {y.mean():.3f} ± {y.std():.3f}")

# S100 pathway proteins (from H08)
print("\n5. Identifying S100 pathway proteins...")
s100_pathway = [
    'S100A1', 'S100A4', 'S100A6', 'S100A8', 'S100A9', 'S100A10',
    'S100B', 'S100A11', 'S100A12', 'S100A13',
    'LOX', 'LOXL1', 'LOXL2', 'LOXL3', 'LOXL4',
    'TGM2', 'TGM3', 'F13B', 'FN1', 'COL1A1'
]

s100_indices = [i for i, p in enumerate(protein_names) if p in s100_pathway]
s100_proteins_found = [protein_names[i] for i in s100_indices]

print(f"   S100 pathway proteins in dataset: {len(s100_proteins_found)}/{len(s100_pathway)}")
print(f"   Found: {s100_proteins_found}")
print(f"   S100 feature indices: {s100_indices[:10]}..." if len(s100_indices) > 10 else f"   S100 feature indices: {s100_indices}")

# Train/Val/Test split (70/15/15)
print("\n6. Creating train/validation/test splits...")

# For small datasets, don't stratify (need at least 2 samples per class)
print(f"   Dataset has {X.shape[0]} samples - using simple split without stratification")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"   Train: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   Val:   {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"   Test:  {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

# Standardization (fit on train, apply to all)
print("\n7. Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   Mean: {scaler.mean_[:5]}... (first 5 features)")
print(f"   Std:  {scaler.scale_[:5]}... (first 5 features)")

# Save all preprocessed data
print("\n8. Saving preprocessed data...")
output_dir = '/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_18_multimodal_integration/claude_code'

# Save as numpy arrays
np.save(f'{output_dir}/X_train_claude_code.npy', X_train_scaled)
np.save(f'{output_dir}/X_val_claude_code.npy', X_val_scaled)
np.save(f'{output_dir}/X_test_claude_code.npy', X_test_scaled)
np.save(f'{output_dir}/y_train_claude_code.npy', y_train)
np.save(f'{output_dir}/y_val_claude_code.npy', y_val)
np.save(f'{output_dir}/y_test_claude_code.npy', y_test)

# Save metadata
metadata = {
    'n_samples': X.shape[0],
    'n_proteins': X.shape[1],
    'protein_names': protein_names,
    's100_indices': s100_indices,
    's100_proteins': s100_proteins_found,
    'train_size': len(y_train),
    'val_size': len(y_val),
    'test_size': len(y_test),
    'y_mean': float(y.mean()),
    'y_std': float(y.std()),
    'y_min': float(y.min()),
    'y_max': float(y.max())
}

with open(f'{output_dir}/data_metadata_claude_code.pkl', 'wb') as f:
    pickle.dump(metadata, f)

with open(f'{output_dir}/scaler_claude_code.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save protein list as CSV
pd.DataFrame({
    'Index': range(len(protein_names)),
    'Protein': protein_names,
    'In_S100_Pathway': [p in s100_pathway for p in protein_names]
}).to_csv(f'{output_dir}/protein_list_claude_code.csv', index=False)

print(f"   ✓ Saved train/val/test arrays")
print(f"   ✓ Saved metadata and scaler")
print(f"   ✓ Saved protein list")

# Summary statistics
print("\n" + "=" * 80)
print("DATA PREPARATION SUMMARY")
print("=" * 80)
print(f"Total samples:        {X.shape[0]}")
print(f"Total proteins:       {X.shape[1]}")
print(f"S100 pathway proteins: {len(s100_proteins_found)}")
print(f"")
print(f"Train set:            {len(y_train)} samples, y={y_train.mean():.3f}±{y_train.std():.3f}")
print(f"Validation set:       {len(y_val)} samples, y={y_val.mean():.3f}±{y_val.std():.3f}")
print(f"Test set:             {len(y_test)} samples, y={y_test.mean():.3f}±{y_test.std():.3f}")
print("")
print("Ready for multimodal model training!")
print("=" * 80)
