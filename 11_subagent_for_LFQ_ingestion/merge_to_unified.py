#!/usr/bin/env python3
"""
Merge Study to Unified CSV
===========================

Add a processed study to the unified ECM Atlas database.

Usage:
    python merge_to_unified.py <study_csv_file>

Example:
    python merge_to_unified.py 05_Randles_paper_to_csv/Randles_2021_wide_format.csv
"""

import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
import json
import sys


def merge_study_to_unified(
    study_csv: str,
    unified_csv: str = '08_merged_ecm_dataset/merged_ecm_aging_zscore.csv',
    project_root: Path = None
):
    """
    Add new study to unified CSV.

    Parameters:
    -----------
    study_csv : str
        Path to study wide-format CSV (relative to project root)
    unified_csv : str
        Path to unified CSV (relative to project root)
    project_root : Path
        Project root directory (auto-detected if None)

    Returns:
    --------
    pd.DataFrame: Merged unified dataframe
    """

    # Auto-detect project root if not specified
    if project_root is None:
        # Try to find project root by looking for characteristic files
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / 'references' / 'human_matrisome_v2.csv').exists():
                project_root = parent
                break
        if project_root is None:
            project_root = Path.cwd()
            print(f"⚠️  Could not auto-detect project root, using current directory: {project_root}")

    # Convert to absolute paths
    study_path = project_root / study_csv
    unified_path = project_root / unified_csv

    print(f"\n{'='*70}")
    print("MERGE STUDY TO UNIFIED CSV")
    print(f"{'='*70}")
    print(f"Project root: {project_root}")
    print(f"Study CSV: {study_path}")
    print(f"Unified CSV: {unified_path}")

    # Validate study CSV exists
    if not study_path.exists():
        raise FileNotFoundError(f"❌ Study CSV not found: {study_path}")

    # Ensure output directory exists
    unified_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Backup existing unified CSV
    if unified_path.exists():
        backup_dir = unified_path.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        backup_path = backup_dir / f"merged_ecm_aging_zscore_{timestamp}.csv"
        shutil.copy(unified_path, backup_path)
        print(f"✅ Backup created: {backup_path.relative_to(project_root)}")
    else:
        print(f"ℹ️  No existing unified CSV - creating new one")

    # 2. Load new study
    df_new = pd.read_csv(study_path)
    study_id = df_new['Study_ID'].iloc[0]
    print(f"\n📥 Loading new study: {study_id}")
    print(f"   Rows: {len(df_new)}")
    print(f"   Proteins: {df_new['Protein_ID'].nunique()}")
    print(f"   Compartments: {df_new['Tissue_Compartment'].unique().tolist() if 'Tissue_Compartment' in df_new.columns else 'N/A'}")

    # 3. Load existing unified (if exists)
    if unified_path.exists():
        df_existing = pd.read_csv(unified_path)
        print(f"\n📊 Existing unified CSV:")
        print(f"   Rows: {len(df_existing)}")
        print(f"   Studies: {df_existing['Study_ID'].unique().tolist()}")

        # 4. Validate schema match
        missing_cols = set(df_new.columns) - set(df_existing.columns)
        extra_cols = set(df_existing.columns) - set(df_new.columns)

        if missing_cols:
            print(f"   ⚠️  New columns in study (will add to unified): {missing_cols}")
            for col in missing_cols:
                df_existing[col] = None

        if extra_cols:
            print(f"   ⚠️  Missing columns in study (will add NaN): {extra_cols}")
            for col in extra_cols:
                df_new[col] = None

        # Align column order
        df_new = df_new[df_existing.columns]

    else:
        df_existing = pd.DataFrame()

    # 5. Concatenate
    df_merged = pd.concat([df_existing, df_new], ignore_index=True)

    print(f"\n🔗 Merged result:")
    print(f"   Total rows: {len(df_merged)} (added {len(df_new)})")
    print(f"   Total studies: {df_merged['Study_ID'].nunique()}")
    print(f"   Studies: {df_merged['Study_ID'].unique().tolist()}")

    # 6. Check for duplicates
    # NOTE: Use Tissue_Compartment instead of Tissue to preserve proteins in multiple compartments
    duplicates = df_merged.duplicated(subset=['Protein_ID', 'Tissue_Compartment', 'Study_ID']).sum()
    if duplicates > 0:
        print(f"   ⚠️  WARNING: {duplicates} duplicate rows detected!")
        # Remove duplicates, keeping last occurrence
        df_merged = df_merged.drop_duplicates(subset=['Protein_ID', 'Tissue_Compartment', 'Study_ID'], keep='last')
        print(f"   ✅ Duplicates removed")

    # 7. Save updated unified CSV
    df_merged.to_csv(unified_path, index=False)
    print(f"\n✅ Saved updated unified CSV: {unified_path.relative_to(project_root)}")

    # 8. Update unified metadata
    metadata_path = unified_path.parent / 'unified_metadata.json'

    metadata = {
        "last_updated": datetime.now().isoformat(),
        "total_rows": int(len(df_merged)),
        "total_studies": int(df_merged['Study_ID'].nunique()),
        "studies": df_merged['Study_ID'].unique().tolist(),
        "total_proteins": int(df_merged['Protein_ID'].nunique()),
        "compartments": df_merged['Tissue'].unique().tolist(),
        "species": df_merged['Species'].unique().tolist(),
        "rows_per_study": {k: int(v) for k, v in df_merged.groupby('Study_ID').size().items()}
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Updated metadata: {metadata_path.relative_to(project_root)}")

    print(f"\n{'='*70}")
    print("✅ MERGE COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"1. Review unified CSV: {unified_path.relative_to(project_root)}")
    print(f"2. Calculate z-scores: python universal_zscore_function.py {study_id} Tissue")
    print(f"3. Verify backup created: {unified_path.parent / 'backups'}")

    return df_merged


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing required argument")
        print("\nUsage:")
        print("  python merge_to_unified.py <study_csv_file>")
        print("\nExamples:")
        print("  python merge_to_unified.py 05_Randles_paper_to_csv/Randles_2021_wide_format.csv")
        print("  python merge_to_unified.py 07_Tam_2020_paper_to_csv/Tam_2020_wide_format.csv")
        sys.exit(1)

    study_csv = sys.argv[1]

    # Optional: specify unified CSV path
    unified_csv = sys.argv[2] if len(sys.argv) > 2 else '08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'

    try:
        df_merged = merge_study_to_unified(
            study_csv=study_csv,
            unified_csv=unified_csv
        )
        print("\n✅ Done!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
