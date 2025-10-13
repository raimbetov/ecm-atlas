#!/usr/bin/env python3
"""
Migrate old merged_ecm_aging_zscore.csv to new ECM_Atlas_Unified.csv format
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

def migrate():
    """Migrate old format to new format and merge."""

    print("="*70)
    print("MIGRATE OLD DATA TO NEW UNIFIED FORMAT")
    print("="*70)

    # Paths
    old_file = Path("08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
    new_file = Path("08_merged_ecm_dataset/ECM_Atlas_Unified.csv")
    backup_dir = Path("08_merged_ecm_dataset/backups")
    backup_dir.mkdir(exist_ok=True)

    # 1. Backup both files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if old_file.exists():
        backup_old = backup_dir / f"merged_ecm_aging_zscore_{timestamp}.csv"
        shutil.copy(old_file, backup_old)
        print(f"✅ Backed up old file: {backup_old.name}")

    if new_file.exists():
        backup_new = backup_dir / f"ECM_Atlas_Unified_{timestamp}.csv"
        shutil.copy(new_file, backup_new)
        print(f"✅ Backed up new file: {backup_new.name}")

    # 2. Load old data
    print(f"\n📥 Loading old data...")
    df_old = pd.read_csv(old_file)
    print(f"   Rows: {len(df_old)}")
    print(f"   Studies: {df_old['Study_ID'].unique().tolist()}")
    print(f"   Columns: {len(df_old.columns)}")

    # 3. Load new data
    print(f"\n📥 Loading new data...")
    df_new = pd.read_csv(new_file)
    print(f"   Rows: {len(df_new)}")
    print(f"   Studies: {df_new['Study_ID'].unique().tolist()}")
    print(f"   Columns: {len(df_new.columns)}")

    # 4. Normalize old schema to match new schema
    print(f"\n🔄 Normalizing old schema...")

    # Rename columns with underscores to match new (with spaces)
    df_old_normalized = df_old.rename(columns={
        'Matrisome_Category': 'Matrisome Category',
        'Matrisome_Division': 'Matrisome Division'
    })

    # Select only columns that exist in new schema
    new_columns = df_new.columns.tolist()

    # Map old columns to new columns
    column_mapping = {
        'Protein_ID': 'Protein_ID',
        'Protein_Name': 'Protein_Name',
        'Gene_Symbol': 'Gene_Symbol',
        'Canonical_Gene_Symbol': 'Canonical_Gene_Symbol',
        'Matrisome Category': 'Matrisome Category',
        'Matrisome Division': 'Matrisome Division',
        'Tissue': 'Tissue',
        'Tissue_Compartment': 'Tissue_Compartment',
        'Species': 'Species',
        'Abundance_Young': 'Abundance_Young',
        'Abundance_Old': 'Abundance_Old',
        'Method': 'Method',
        'Study_ID': 'Study_ID',
        'Match_Level': 'Match_Level',
        'Match_Confidence': 'Match_Confidence',
        'Zscore_Young': 'Zscore_Young',
        'Zscore_Old': 'Zscore_Old',
        'Zscore_Delta': 'Zscore_Delta'
    }

    # Select and reorder columns
    df_old_normalized = df_old_normalized[list(column_mapping.keys())]

    print(f"✅ Old data normalized to new schema")
    print(f"   Kept {len(df_old_normalized.columns)} columns")

    # 5. Check for overlapping studies
    old_studies = set(df_old_normalized['Study_ID'].unique())
    new_studies = set(df_new['Study_ID'].unique())
    overlap = old_studies & new_studies

    if overlap:
        print(f"\n⚠️  WARNING: Overlapping studies found: {overlap}")
        print(f"   Will keep NEW version (from ECM_Atlas_Unified.csv)")

        # Remove overlapping studies from old data
        df_old_normalized = df_old_normalized[~df_old_normalized['Study_ID'].isin(overlap)]
        print(f"   Removed {len(df_old) - len(df_old_normalized)} rows from old data")

    # 6. Concatenate
    print(f"\n🔗 Merging old + new...")
    df_merged = pd.concat([df_old_normalized, df_new], ignore_index=True)

    print(f"\n📊 Merged result:")
    print(f"   Total rows: {len(df_merged)}")
    print(f"   Total studies: {df_merged['Study_ID'].nunique()}")
    print(f"   Studies: {df_merged['Study_ID'].unique().tolist()}")

    studies_breakdown = df_merged.groupby('Study_ID').size()
    for study, count in studies_breakdown.items():
        print(f"     - {study}: {count} rows")

    # 7. Check for duplicates
    duplicates = df_merged.duplicated(subset=['Protein_ID', 'Tissue_Compartment', 'Study_ID']).sum()
    if duplicates > 0:
        print(f"\n⚠️  Found {duplicates} duplicate rows - removing...")
        df_merged = df_merged.drop_duplicates(subset=['Protein_ID', 'Tissue_Compartment', 'Study_ID'], keep='last')
        print(f"✅ Duplicates removed")

    # 8. Save merged result
    print(f"\n💾 Saving merged unified CSV...")
    df_merged.to_csv(new_file, index=False)
    print(f"✅ Saved: {new_file}")

    # 9. Rename old file
    old_file_renamed = old_file.with_suffix('.csv.MIGRATED')
    old_file.rename(old_file_renamed)
    print(f"✅ Renamed old file: {old_file_renamed.name}")

    print(f"\n{'='*70}")
    print("✅ MIGRATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFiles:")
    print(f"  - New unified: {new_file}")
    print(f"  - Old (renamed): {old_file_renamed}")
    print(f"  - Backups: {backup_dir}/")

    return df_merged

if __name__ == '__main__':
    try:
        df_result = migrate()
        print("\n✅ Success!")
        print(f"\nFinal stats:")
        print(f"  Total rows: {len(df_result)}")
        print(f"  Studies: {df_result['Study_ID'].nunique()}")
        print(f"  Species: {df_result['Species'].unique().tolist()}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
