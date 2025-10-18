#!/usr/bin/env python3
"""Fill missing Compartment column from Tissue_Compartment"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Paths
PROJECT_ROOT = Path("/Users/Kravtsovd/projects/ecm-atlas")
INPUT_CSV = PROJECT_ROOT / "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
BACKUP_DIR = PROJECT_ROOT / "08_merged_ecm_dataset/backups"

print("="*70)
print("FIX MISSING COMPARTMENT COLUMN")
print("="*70)

# Backup
BACKUP_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
backup_path = BACKUP_DIR / f"merged_ecm_aging_zscore_{timestamp}_before_compartment_fix.csv"
shutil.copy(INPUT_CSV, backup_path)
print(f"✓ Backup created: {backup_path}")

# Load data
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df)} rows")

# Check missing Compartment
mask_missing = df['Compartment'].isna() | (df['Compartment'] == '')
n_missing = mask_missing.sum()
print(f"Found {n_missing} rows with missing Compartment")

# Show which Study_IDs are affected
affected = df[mask_missing]['Study_ID'].value_counts()
print(f"\nAffected Study_IDs:")
for study, count in affected.items():
    print(f"  {study}: {count} rows")

# Fix: Extract compartment from Tissue_Compartment
def extract_compartment(tissue_compartment):
    """Extract compartment from Tissue_Compartment
    Examples:
      Kidney_Glomerular → Glomerular
      Glomerular → Glomerular
      Heart_Native_Tissue → Native_Tissue
    """
    if pd.isna(tissue_compartment):
        return None
    tc_str = str(tissue_compartment).strip()
    # If contains underscore, take part after first underscore
    if '_' in tc_str:
        parts = tc_str.split('_', 1)
        # If first part looks like organ (capitalized single word), take second part
        if parts[0] and parts[0][0].isupper() and len(parts) > 1:
            return parts[1]
    # Otherwise use as-is
    return tc_str

df.loc[mask_missing, 'Compartment'] = df.loc[mask_missing, 'Tissue_Compartment'].apply(extract_compartment)

# Verify
mask_fixed = mask_missing & df['Compartment'].notna()
print(f"\n✓ Fixed {mask_fixed.sum()} rows with Compartment from Tissue_Compartment")

# Show sample of fixed data
print("\nSample of fixed data:")
sample = df[mask_fixed & (df['Study_ID'] == 'Randles_2021')][['Study_ID', 'Tissue_Compartment', 'Compartment']].head(10)
print(sample.to_string(index=False))

# Save
df.to_csv(INPUT_CSV, index=False)
print(f"\n✓ Saved fixed data to {INPUT_CSV}")

print("\n" + "="*70)
print("FIX COMPLETE")
print("="*70)
