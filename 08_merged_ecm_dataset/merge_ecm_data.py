#!/usr/bin/env python3
"""
ECM Dataset Merge Script
Combines z-score normalized data from multiple studies into a single standardized CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define input files
INPUT_FILES = {
    'Randles_Glomerular': {
        'path': '../06_Randles_z_score_by_tissue_compartment/claude_code/Randles_2021_Glomerular_zscore.csv',
        'dataset': 'Randles_2021',
        'organ': 'Kidney',
        'compartment': 'Glomerular',
        'filter_ecm': True
    },
    'Randles_Tubulointerstitial': {
        'path': '../06_Randles_z_score_by_tissue_compartment/claude_code/Randles_2021_Tubulointerstitial_zscore.csv',
        'dataset': 'Randles_2021',
        'organ': 'Kidney',
        'compartment': 'Tubulointerstitial',
        'filter_ecm': True
    },
    'Tam_NP': {
        'path': '../07_Tam_2020_paper_to_csv/claude_code/Tam_2020_NP_zscore.csv',
        'dataset': 'Tam_2020',
        'organ': 'Intervertebral_disc',
        'compartment': 'NP',
        'filter_ecm': False  # Already filtered
    },
    'Tam_IAF': {
        'path': '../07_Tam_2020_paper_to_csv/claude_code/Tam_2020_IAF_zscore.csv',
        'dataset': 'Tam_2020',
        'organ': 'Intervertebral_disc',
        'compartment': 'IAF',
        'filter_ecm': False  # Already filtered
    },
    'Tam_OAF': {
        'path': '../07_Tam_2020_paper_to_csv/claude_code/Tam_2020_OAF_zscore.csv',
        'dataset': 'Tam_2020',
        'organ': 'Intervertebral_disc',
        'compartment': 'OAF',
        'filter_ecm': False  # Already filtered
    }
}

def load_and_process_file(file_info):
    """Load CSV file and add metadata"""
    print(f"\n📂 Loading {file_info['dataset']} - {file_info['compartment']}...")

    # Load CSV
    df = pd.read_csv(file_info['path'])
    print(f"   Loaded {len(df)} proteins")

    # Filter ECM proteins if needed
    if file_info['filter_ecm']:
        ecm_mask = df['Matrisome_Division'].notna() & (
            df['Matrisome_Division'].str.contains('Core matrisome', na=False) |
            df['Matrisome_Division'].str.contains('Matrisome-associated', na=False)
        )
        df = df[ecm_mask].copy()
        print(f"   Filtered to {len(df)} ECM proteins")

    # Add metadata columns at the beginning
    df.insert(0, 'Dataset_Name', file_info['dataset'])
    df.insert(1, 'Organ', file_info['organ'])
    df.insert(2, 'Compartment', file_info['compartment'])

    return df

def standardize_columns(dfs):
    """Standardize columns across all dataframes"""
    print("\n🔧 Standardizing columns...")

    # Get all unique columns
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)

    print(f"   Found {len(all_columns)} unique columns")

    # Add missing columns to each dataframe
    for i, df in enumerate(dfs):
        missing_cols = all_columns - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan

        # Reorder columns: metadata first, then alphabetical
        metadata_cols = ['Dataset_Name', 'Organ', 'Compartment']
        other_cols = sorted([c for c in df.columns if c not in metadata_cols])
        dfs[i] = df[metadata_cols + other_cols]

    return dfs

def validate_data(df):
    """Validate merged data quality"""
    print("\n✅ Validating data quality...")

    issues = []

    # Check for required columns (but allow NaN in z-scores - biologically meaningful)
    required_cols_strict = ['Protein_ID', 'Gene_Symbol']  # Must have no NaN
    required_cols_allow_nan = ['Zscore_Young', 'Zscore_Old', 'Zscore_Delta']  # NaN allowed (protein not detected)

    for col in required_cols_strict:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
        elif df[col].isna().sum() > 0:
            issues.append(f"Column {col} has {df[col].isna().sum()} missing values")

    for col in required_cols_allow_nan:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # Check z-score ranges (on non-NaN values)
    for col in ['Zscore_Young', 'Zscore_Old', 'Zscore_Delta']:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()
                if min_val < -10 or max_val > 10:
                    issues.append(f"{col} has unusual range: [{min_val:.2f}, {max_val:.2f}]")

    # Report missing z-scores (informational, not an issue)
    print(f"\n   ℹ️  Missing z-scores (biologically meaningful - protein not detected):")
    for col in required_cols_allow_nan:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100
            print(f"      {col}: {n_missing}/{len(df)} ({pct_missing:.1f}%)")

    # Check for duplicates
    duplicate_mask = df.duplicated(subset=['Protein_ID', 'Compartment'], keep=False)
    if duplicate_mask.sum() > 0:
        issues.append(f"Found {duplicate_mask.sum()} duplicate (Protein_ID, Compartment) pairs")

    if issues:
        print("   ⚠️  Issues found:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ All validation checks passed!")

    return issues

def generate_statistics(df):
    """Generate statistics about the merged data"""
    print("\n📊 Generating statistics...")

    stats = {
        'total_proteins': len(df),
        'unique_proteins': df['Protein_ID'].nunique(),
        'by_dataset': df.groupby('Dataset_Name').size().to_dict(),
        'by_organ': df.groupby('Organ').size().to_dict(),
        'by_compartment': df.groupby('Compartment').size().to_dict(),
        'by_matrisome_category': df['Matrisome_Category'].value_counts().to_dict(),
        'by_matrisome_division': df['Matrisome_Division'].value_counts().to_dict(),
        'zscore_stats': {
            'young_mean': float(df['Zscore_Young'].mean()),
            'young_std': float(df['Zscore_Young'].std()),
            'old_mean': float(df['Zscore_Old'].mean()),
            'old_std': float(df['Zscore_Old'].std()),
            'delta_mean': float(df['Zscore_Delta'].mean()),
            'delta_std': float(df['Zscore_Delta'].std())
        }
    }

    print(f"   Total proteins: {stats['total_proteins']}")
    print(f"   Unique proteins: {stats['unique_proteins']}")
    print(f"   By dataset: {stats['by_dataset']}")
    print(f"   By compartment: {stats['by_compartment']}")

    return stats

def main():
    print("="*70)
    print("ECM Dataset Merge Script")
    print("="*70)

    # Load all files
    dataframes = []
    for name, file_info in INPUT_FILES.items():
        df = load_and_process_file(file_info)
        dataframes.append(df)

    # Standardize columns
    dataframes = standardize_columns(dataframes)

    # Merge all dataframes
    print("\n🔗 Merging all datasets...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"   Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Validate data
    issues = validate_data(merged_df)

    # Generate statistics
    stats = generate_statistics(merged_df)

    # Save merged data
    output_file = 'merged_ecm_aging_zscore.csv'
    print(f"\n💾 Saving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    print(f"   Saved {len(merged_df)} rows")

    # Generate report
    print("\n📝 Generating merge report...")
    report_lines = [
        "# ECM Dataset Merge Report",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total proteins:** {stats['total_proteins']}",
        f"- **Unique proteins:** {stats['unique_proteins']}",
        f"- **Datasets:** {len(INPUT_FILES)}",
        f"- **Output file:** {output_file}",
        "",
        "## Datasets Included",
        ""
    ]

    for name, info in INPUT_FILES.items():
        count = stats['by_dataset'].get(info['dataset'], 0)
        report_lines.append(f"### {info['dataset']} - {info['compartment']}")
        report_lines.append(f"- **Organ:** {info['organ']}")
        report_lines.append(f"- **Compartment:** {info['compartment']}")
        report_lines.append(f"- **Proteins:** {count}")
        report_lines.append(f"- **ECM Filtered:** {'Yes' if info['filter_ecm'] else 'No (pre-filtered)'}")
        report_lines.append("")

    report_lines.extend([
        "## Statistics by Compartment",
        ""
    ])
    for comp, count in sorted(stats['by_compartment'].items()):
        report_lines.append(f"- **{comp}:** {count} proteins")

    report_lines.extend([
        "",
        "## Matrisome Categories",
        ""
    ])
    for cat, count in sorted(stats['by_matrisome_category'].items(), key=lambda x: x[1], reverse=True):
        if pd.notna(cat):
            report_lines.append(f"- **{cat}:** {count} proteins")

    report_lines.extend([
        "",
        "## Z-Score Statistics",
        "",
        f"### Young",
        f"- Mean: {stats['zscore_stats']['young_mean']:.3f}",
        f"- Std Dev: {stats['zscore_stats']['young_std']:.3f}",
        "",
        f"### Old",
        f"- Mean: {stats['zscore_stats']['old_mean']:.3f}",
        f"- Std Dev: {stats['zscore_stats']['old_std']:.3f}",
        "",
        f"### Delta (Old - Young)",
        f"- Mean: {stats['zscore_stats']['delta_mean']:.3f}",
        f"- Std Dev: {stats['zscore_stats']['delta_std']:.3f}",
        ""
    ])

    if issues:
        report_lines.extend([
            "## ⚠️  Data Quality Issues",
            ""
        ])
        for issue in issues:
            report_lines.append(f"- {issue}")
        report_lines.append("")
    else:
        report_lines.extend([
            "## ✅ Data Quality",
            "",
            "All validation checks passed successfully!",
            ""
        ])

    report_lines.extend([
        "## Column Schema",
        "",
        "### Metadata Columns (Added)",
        "- `Dataset_Name` - Study identifier (Randles_2021 / Tam_2020)",
        "- `Organ` - Tissue/organ (Kidney / Intervertebral_disc)",
        "- `Compartment` - Tissue compartment (Glomerular, Tubulointerstitial, NP, IAF, OAF)",
        "",
        "### Core Data Columns",
        "- `Protein_ID` - UniProt ID",
        "- `Protein_Name` - Full protein name",
        "- `Gene_Symbol` - Gene symbol",
        "- `Canonical_Gene_Symbol` - Canonical gene symbol",
        "- `Species` - Organism (Homo sapiens)",
        "",
        "### Abundance Columns",
        "- `Abundance_Young` - Abundance in young samples",
        "- `Abundance_Old` - Abundance in old samples",
        "- `Abundance_Young_transformed` - log2-transformed (Randles only)",
        "- `Abundance_Old_transformed` - log2-transformed (Randles only)",
        "",
        "### Z-Score Columns",
        "- `Zscore_Young` - Z-score for young samples",
        "- `Zscore_Old` - Z-score for old samples",
        "- `Zscore_Delta` - Change in z-score (Old - Young)",
        "",
        "### Annotation Columns",
        "- `Matrisome_Category` - ECM protein category",
        "- `Matrisome_Division` - Core matrisome / Matrisome-associated",
        "- `Match_Level` - Annotation match level",
        "- `Match_Confidence` - Confidence score (0-100)",
        "",
        "### Method & Study",
        "- `Method` - Proteomics method",
        "- `Study_ID` - Study identifier",
        "- `Tissue` - Original tissue annotation",
        "- `Tissue_Compartment` - Original compartment annotation",
        "",
        "### Additional (Tam 2020 only)",
        "- `N_Profiles_Young` - Number of spatial profiles (young)",
        "- `N_Profiles_Old` - Number of spatial profiles (old)",
        ""
    ])

    report_file = 'merge_report.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"   Report saved to {report_file}")

    print("\n" + "="*70)
    print("✅ ECM Dataset Merge Complete!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"   - {output_file}")
    print(f"   - {report_file}")
    print("")

if __name__ == '__main__':
    main()
