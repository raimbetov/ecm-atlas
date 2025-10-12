#!/usr/bin/env python3
"""
Convert Randles 2021 long-format CSV to wide-format with Young/Old columns.

INPUT (long-format):
- 31,320 rows (2,610 proteins × 12 samples)
- Each protein appears 12 times (6 Young + 6 Old samples)
- Age_Bin: Young or Old

OUTPUT (wide-format):
- ~5,220 rows (2,610 proteins × 2 compartments)
- Each protein appears once per compartment
- Separate columns: Abundance_Young, Abundance_Old
- Young/Old values are AVERAGED across replicates within compartment
"""

import pandas as pd
import json
from pathlib import Path

# Paths
input_csv = "claude_code/Randles_2021_parsed.csv"
output_csv = "claude_code/Randles_2021_wide_format.csv"
metadata_json = "claude_code/Randles_2021_wide_format_metadata.json"

print("=" * 80)
print("RANDLES 2021: LONG → WIDE FORMAT CONVERSION")
print("=" * 80)

# Load long-format CSV
print(f"\n1. Loading long-format CSV: {input_csv}")
df_long = pd.read_csv(input_csv)
print(f"   ✅ Loaded {len(df_long):,} rows × {len(df_long.columns)} columns")

# Verify structure
print(f"\n2. Verifying data structure:")
print(f"   - Unique proteins: {df_long['Gene_Symbol'].nunique():,}")
print(f"   - Unique samples: {df_long['Sample_ID'].nunique()}")
print(f"   - Age bins: {df_long['Age_Bin'].unique().tolist()}")
print(f"   - Compartments: {df_long['Tissue_Compartment'].unique().tolist()}")

# Check duplicates per protein per compartment per age_bin
duplicates_check = df_long.groupby(['Gene_Symbol', 'Tissue_Compartment', 'Age_Bin']).size()
print(f"   - Replicates per protein-compartment-age: {duplicates_check.value_counts().to_dict()}")

# Identify columns to keep (non-redundant metadata)
# These columns are the same for each protein regardless of sample
keep_cols = [
    'Protein_ID', 'Protein_Name', 'Gene_Symbol',
    'Species', 'Method', 'Study_ID',
    'Canonical_Gene_Symbol', 'Matrisome_Category',
    'Matrisome_Division', 'Match_Level', 'Match_Confidence'
]

# Columns that vary by compartment
compartment_cols = ['Tissue', 'Tissue_Compartment']

# Columns to average: Abundance
abundance_col = 'Abundance'

print(f"\n3. Pivoting to wide format:")
print(f"   Strategy: Average {abundance_col} across replicates within each Age_Bin + Compartment")

# Group by protein + compartment, then pivot Age_Bin
# First, calculate mean abundance for each protein-compartment-age_bin combination
# IMPORTANT: Include dropna=False to keep proteins with empty Matrisome_Category (non-ECM proteins)
df_aggregated = df_long.groupby(
    ['Protein_ID', 'Gene_Symbol', 'Tissue', 'Tissue_Compartment', 'Age_Bin'],
    dropna=False  # Keep rows with NaN values in grouping columns
).agg({
    'Abundance': 'mean',  # Average across replicates
    'Protein_Name': 'first',
    'Species': 'first',
    'Method': 'first',
    'Study_ID': 'first',
    'Canonical_Gene_Symbol': 'first',
    'Matrisome_Category': 'first',
    'Matrisome_Division': 'first',
    'Match_Level': 'first',
    'Match_Confidence': 'first'
}).reset_index()

print(f"   ✅ Aggregated to {len(df_aggregated):,} rows (protein × compartment × age_bin)")

# CRITICAL FIX: Fill NaN values in annotation columns to prevent pivot_table from dropping rows
# For non-ECM proteins, these columns are empty, but we still want to keep them
df_aggregated['Canonical_Gene_Symbol'] = df_aggregated['Canonical_Gene_Symbol'].fillna('')
df_aggregated['Matrisome_Category'] = df_aggregated['Matrisome_Category'].fillna('')
df_aggregated['Matrisome_Division'] = df_aggregated['Matrisome_Division'].fillna('')
df_aggregated['Match_Level'] = df_aggregated['Match_Level'].fillna('unmatched')
df_aggregated['Match_Confidence'] = df_aggregated['Match_Confidence'].fillna(0)

# Pivot Age_Bin to columns
df_wide = df_aggregated.pivot_table(
    index=[
        'Protein_ID', 'Protein_Name', 'Gene_Symbol',
        'Tissue', 'Tissue_Compartment', 'Species',
        'Method', 'Study_ID',
        'Canonical_Gene_Symbol', 'Matrisome_Category',
        'Matrisome_Division', 'Match_Level', 'Match_Confidence'
    ],
    columns='Age_Bin',
    values='Abundance',
    aggfunc='first'  # Already aggregated, so just take first value
).reset_index()

# Rename columns
df_wide.columns.name = None  # Remove 'Age_Bin' from column name
df_wide = df_wide.rename(columns={
    'Young': 'Abundance_Young',
    'Old': 'Abundance_Old'
})

# Reorder columns for readability
column_order = [
    'Protein_ID', 'Protein_Name', 'Gene_Symbol',
    'Tissue', 'Tissue_Compartment', 'Species',
    'Abundance_Young', 'Abundance_Old',
    'Method', 'Study_ID',
    'Canonical_Gene_Symbol', 'Matrisome_Category',
    'Matrisome_Division', 'Match_Level', 'Match_Confidence'
]
df_wide = df_wide[column_order]

print(f"   ✅ Pivoted to {len(df_wide):,} rows × {len(df_wide.columns)} columns")

# Validation
print(f"\n4. Validating wide-format data:")
print(f"   - Unique proteins: {df_wide['Gene_Symbol'].nunique():,}")
print(f"   - Unique compartments: {df_wide['Tissue_Compartment'].nunique()}")
print(f"   - Rows with Young data: {df_wide['Abundance_Young'].notna().sum():,}")
print(f"   - Rows with Old data: {df_wide['Abundance_Old'].notna().sum():,}")
print(f"   - Rows with both: {((df_wide['Abundance_Young'].notna()) & (df_wide['Abundance_Old'].notna())).sum():,}")

# ECM protein count
ecm_count = (df_wide['Matrisome_Category'].notna()).sum()
print(f"   - ECM proteins: {ecm_count:,} ({100*ecm_count/len(df_wide):.1f}%)")

# Save wide-format CSV
print(f"\n5. Exporting wide-format CSV: {output_csv}")
df_wide.to_csv(output_csv, index=False)
file_size = Path(output_csv).stat().st_size / (1024**2)
print(f"   ✅ Saved {len(df_wide):,} rows × {len(df_wide.columns)} columns ({file_size:.2f} MB)")

# Generate metadata
metadata = {
    "conversion_info": {
        "input_file": input_csv,
        "output_file": output_csv,
        "conversion_date": pd.Timestamp.now().isoformat(),
        "format": "wide-format (Age_Bin as columns)"
    },
    "input_stats": {
        "rows": int(len(df_long)),
        "columns": int(len(df_long.columns)),
        "unique_proteins": int(df_long['Gene_Symbol'].nunique()),
        "unique_samples": int(df_long['Sample_ID'].nunique())
    },
    "output_stats": {
        "rows": int(len(df_wide)),
        "columns": int(len(df_wide.columns)),
        "unique_proteins": int(df_wide['Gene_Symbol'].nunique()),
        "unique_compartments": int(df_wide['Tissue_Compartment'].nunique())
    },
    "aggregation_method": {
        "abundance_young": "Mean of 3 young samples (ages 15, 29, 37) per compartment",
        "abundance_old": "Mean of 3 old samples (ages 61, 67, 69) per compartment"
    },
    "validation": {
        "rows_with_young_data": int(df_wide['Abundance_Young'].notna().sum()),
        "rows_with_old_data": int(df_wide['Abundance_Old'].notna().sum()),
        "rows_with_both": int(((df_wide['Abundance_Young'].notna()) & (df_wide['Abundance_Old'].notna())).sum()),
        "ecm_proteins": int(ecm_count),
        "ecm_percentage": round(100*ecm_count/len(df_wide), 2)
    }
}

# Save metadata
print(f"\n6. Exporting metadata: {metadata_json}")
with open(metadata_json, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ Metadata saved")

# Show sample of results
print(f"\n7. Sample rows (first 3 proteins, Glomerular compartment):")
sample = df_wide[df_wide['Tissue_Compartment'] == 'Glomerular'].head(3)
for idx, row in sample.iterrows():
    print(f"\n   Protein: {row['Gene_Symbol']}")
    print(f"   - Young: {row['Abundance_Young']:.2f}")
    print(f"   - Old: {row['Abundance_Old']:.2f}")
    print(f"   - Change: {((row['Abundance_Old'] / row['Abundance_Young']) - 1) * 100:+.1f}%")

print(f"\n{'='*80}")
print(f"✅ CONVERSION COMPLETE!")
print(f"{'='*80}")
print(f"Output: {output_csv}")
print(f"Metadata: {metadata_json}")
