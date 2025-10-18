#!/usr/bin/env python3
"""Fill missing Dataset_Name column from Study_ID"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Paths
PROJECT_ROOT = Path("/Users/Kravtsovd/projects/ecm-atlas")
INPUT_CSV = PROJECT_ROOT / "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
BACKUP_DIR = PROJECT_ROOT / "08_merged_ecm_dataset/backups"

print("="*70)
print("FIX MISSING DATASET_NAME COLUMN")
print("="*70)

# Backup
BACKUP_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
backup_path = BACKUP_DIR / f"merged_ecm_aging_zscore_{timestamp}_before_dataset_name_fix.csv"
shutil.copy(INPUT_CSV, backup_path)
print(f"✓ Backup created: {backup_path}")

# Load data
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df)} rows")

# Check missing Dataset_Name
mask_missing = df['Dataset_Name'].isna() | (df['Dataset_Name'] == '')
n_missing = mask_missing.sum()
print(f"Found {n_missing} rows with missing Dataset_Name")

# Show which Study_IDs are affected
affected = df[mask_missing]['Study_ID'].value_counts()
print(f"\nAffected Study_IDs:")
for study, count in affected.items():
    print(f"  {study}: {count} rows")

# Fix: Copy Study_ID to Dataset_Name where missing
df.loc[mask_missing, 'Dataset_Name'] = df.loc[mask_missing, 'Study_ID']

# Verify
mask_fixed = mask_missing & df['Dataset_Name'].notna()
print(f"\n✓ Fixed {mask_fixed.sum()} rows with Dataset_Name from Study_ID")

# Save
df.to_csv(INPUT_CSV, index=False)
print(f"\n✓ Saved fixed data to {INPUT_CSV}")

print("\n" + "="*70)
print("FIX COMPLETE")
print("="*70)
