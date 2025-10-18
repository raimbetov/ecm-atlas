#!/usr/bin/env python3
"""Fix missing Organ column for Randles_2021 study"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Paths
PROJECT_ROOT = Path("/Users/Kravtsovd/projects/ecm-atlas")
INPUT_CSV = PROJECT_ROOT / "08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"
BACKUP_DIR = PROJECT_ROOT / "08_merged_ecm_dataset/backups"

print("="*70)
print("FIX MISSING ORGAN COLUMN")
print("="*70)

# Backup
BACKUP_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
backup_path = BACKUP_DIR / f"merged_ecm_aging_zscore_{timestamp}_before_organ_fix.csv"
shutil.copy(INPUT_CSV, backup_path)
print(f"✓ Backup created: {backup_path}")

# Load data
df = pd.read_csv(INPUT_CSV)
print(f"\nLoaded {len(df)} rows")

# Check missing Organ
mask_missing = (df['Study_ID'] == 'Randles_2021') & df['Organ'].isna()
n_missing = mask_missing.sum()
print(f"Found {n_missing} rows with missing Organ for Randles_2021")

# Fix: Extract organ from Tissue column
def extract_organ_from_tissue(tissue):
    """Extract organ name from Tissue column"""
    if pd.isna(tissue):
        return None
    tissue_str = str(tissue)
    # Kidney_Glomerular → Kidney
    # Heart_Native_Tissue → Heart
    if '_' in tissue_str:
        return tissue_str.split('_')[0]
    return tissue_str

# Apply fix
df.loc[mask_missing, 'Organ'] = df.loc[mask_missing, 'Tissue'].apply(extract_organ_from_tissue)

# Verify
mask_fixed = (df['Study_ID'] == 'Randles_2021') & (df['Organ'] == 'Kidney')
print(f"✓ Fixed {mask_fixed.sum()} rows with Organ='Kidney'")

# Save
df.to_csv(INPUT_CSV, index=False)
print(f"\n✓ Saved fixed data to {INPUT_CSV}")

print("\n" + "="*70)
print("FIX COMPLETE")
print("="*70)
