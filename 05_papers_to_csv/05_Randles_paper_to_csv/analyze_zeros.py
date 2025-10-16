#!/usr/bin/env python3
"""
Analyze zero values in Randles_2021 dataset to assess impact of 0→NaN conversion.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
SOURCE_FILE = "/Users/Kravtsovd/projects/ecm-atlas/data_raw/Randles et al. - 2021/ASN.2020101442-File027.xlsx"
PROCESSED_FILE = "/Users/Kravtsovd/projects/ecm-atlas/05_papers_to_csv/05_Randles_paper_to_csv/claude_code/Randles_2021_wide_format.csv"

def analyze_source_excel():
    """Analyze zeros in source Excel file."""
    print("=" * 80)
    print("ANALYZING SOURCE EXCEL FILE")
    print("=" * 80)

    # Load Excel - header is on row 1
    df = pd.read_excel(SOURCE_FILE, sheet_name="Human data matrix fraction", header=1)

    print(f"\nFile: {Path(SOURCE_FILE).name}")
    print(f"Sheet: Human data matrix fraction")
    print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

    # Intensity columns (excluding .1 detection flags)
    intensity_cols = ['G15', 'T15', 'G29', 'T29', 'G37', 'T37', 'G61', 'T61', 'G67', 'T67', 'G69', 'T69']

    print(f"\nIntensity columns analyzed: {len(intensity_cols)}")
    print(f"Intensity columns: {', '.join(intensity_cols)}")

    # Total proteins
    total_proteins = len(df)
    print(f"\nTotal proteins: {total_proteins}")

    # Analyze zeros per column
    print("\n" + "-" * 80)
    print("ZERO VALUE STATISTICS BY COLUMN")
    print("-" * 80)

    zero_stats = []
    for col in intensity_cols:
        if col in df.columns:
            total_values = len(df[col])
            zero_count = (df[col] == 0).sum()
            nan_count = df[col].isna().sum()
            non_zero_count = ((df[col] != 0) & (~df[col].isna())).sum()
            zero_pct = (zero_count / total_values) * 100

            zero_stats.append({
                'Column': col,
                'Total': total_values,
                'Zeros': zero_count,
                'NaN': nan_count,
                'Non-zero': non_zero_count,
                'Zero %': zero_pct
            })

            print(f"{col:6s}: Zeros={zero_count:4d} ({zero_pct:5.1f}%), NaN={nan_count:4d}, Non-zero={non_zero_count:4d}")

    # Summary statistics
    total_zeros = sum([s['Zeros'] for s in zero_stats])
    total_values = sum([s['Total'] for s in zero_stats])
    total_nans = sum([s['NaN'] for s in zero_stats])
    total_nonzeros = sum([s['Non-zero'] for s in zero_stats])
    total_zero_pct = (total_zeros / total_values) * 100

    print("\n" + "-" * 80)
    print("OVERALL SUMMARY")
    print("-" * 80)
    print(f"Total values across all intensity columns: {total_values:,}")
    print(f"Total zero values: {total_zeros:,} ({total_zero_pct:.2f}%)")
    print(f"Total NaN values: {total_nans:,}")
    print(f"Total non-zero values: {total_nonzeros:,}")

    # Check for replicate patterns with zeros
    print("\n" + "-" * 80)
    print("TECHNICAL REPLICATE ANALYSIS")
    print("-" * 80)

    # Group samples by age and compartment
    age_groups = {
        'Young_Glomerular': ['G15', 'G29', 'G37'],
        'Young_Tubulointerstitial': ['T15', 'T29', 'T37'],
        'Old_Glomerular': ['G61', 'G67', 'G69'],
        'Old_Tubulointerstitial': ['T61', 'T67', 'T69']
    }

    print("\nTechnical replicates: Yes (3 donors per age group per compartment)")
    print("\nReplicate groups:")
    for group_name, cols in age_groups.items():
        print(f"  {group_name}: {', '.join(cols)}")

    # Find examples where zeros affect averaging
    print("\n" + "-" * 80)
    print("EXAMPLE REPLICATE PATTERNS WITH ZEROS")
    print("-" * 80)

    examples_found = 0
    for group_name, cols in age_groups.items():
        if examples_found >= 2:  # Show max 2 examples per group
            break

        # Find proteins where at least one replicate is zero and others are non-zero
        for idx, row in df.iterrows():
            if examples_found >= 2:
                break

            values = [row[col] for col in cols if col in df.columns]

            # Check if we have mix of zeros and non-zeros
            has_zero = any(v == 0 for v in values if pd.notna(v))
            has_nonzero = any(v > 0 for v in values if pd.notna(v))

            if has_zero and has_nonzero:
                protein_id = row.get('Accession', 'Unknown')
                gene = row.get('Gene name', 'Unknown')

                print(f"\n{group_name} - Protein: {protein_id} ({gene})")
                print(f"  Replicate values: {', '.join([f'{col}={row[col]:.1f}' if pd.notna(row[col]) else f'{col}=NaN' for col in cols])}")

                # Calculate current average (including zeros)
                valid_values = [v for v in values if pd.notna(v)]
                if valid_values:
                    current_avg = np.mean(valid_values)
                    # Calculate what average would be if zeros → NaN
                    nonzero_values = [v for v in valid_values if v > 0]
                    if nonzero_values:
                        new_avg = np.mean(nonzero_values)
                        print(f"  Current average (0 as value): {current_avg:.2f}")
                        print(f"  After 0→NaN conversion: {new_avg:.2f}")
                        print(f"  Impact: {((new_avg - current_avg) / current_avg * 100):.1f}% increase")

                examples_found += 1

    return df, intensity_cols, zero_stats

def analyze_processed_csv():
    """Analyze zeros in processed wide-format CSV."""
    print("\n\n" + "=" * 80)
    print("ANALYZING PROCESSED WIDE-FORMAT CSV")
    print("=" * 80)

    df = pd.read_csv(PROCESSED_FILE)

    print(f"\nFile: {Path(PROCESSED_FILE).name}")
    print(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")

    # Analyze Abundance_Young and Abundance_Old columns
    abundance_cols = ['Abundance_Young', 'Abundance_Old']

    print(f"\nAbundance columns: {', '.join(abundance_cols)}")
    print(f"Total protein-compartment pairs: {len(df)}")

    print("\n" + "-" * 80)
    print("ZERO VALUE STATISTICS IN PROCESSED DATA")
    print("-" * 80)

    for col in abundance_cols:
        if col in df.columns:
            total = len(df[col])
            zeros = (df[col] == 0).sum()
            nans = df[col].isna().sum()
            nonzeros = ((df[col] != 0) & (~df[col].isna())).sum()
            zero_pct = (zeros / total) * 100

            print(f"{col:17s}: Zeros={zeros:4d} ({zero_pct:5.1f}%), NaN={nans:4d}, Non-zero={nonzeros:4d}")

    # Examples of proteins with zeros
    print("\n" + "-" * 80)
    print("EXAMPLE PROTEINS WITH ZERO ABUNDANCES")
    print("-" * 80)

    for col in abundance_cols:
        df_zeros = df[df[col] == 0].head(3)
        if len(df_zeros) > 0:
            print(f"\n{col} examples:")
            for idx, row in df_zeros.iterrows():
                print(f"  {row['Protein_ID']} ({row['Gene_Symbol']}) in {row['Tissue']}: {col}={row[col]}")

def main():
    """Run complete analysis."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "RANDLES_2021 ZERO VALUE AUDIT" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")

    # Analyze source Excel
    df_source, intensity_cols, zero_stats = analyze_source_excel()

    # Analyze processed CSV
    analyze_processed_csv()

    # Impact assessment
    print("\n\n" + "=" * 80)
    print("IMPACT ASSESSMENT: 0→NaN CONVERSION")
    print("=" * 80)

    total_zeros = sum([s['Zeros'] for s in zero_stats])
    total_values = sum([s['Total'] for s in zero_stats])

    print(f"""
When converting zeros to NaN:

1. DATA VOLUME IMPACT:
   - {total_zeros:,} zero values ({(total_zeros/total_values*100):.2f}%) will become NaN
   - These zeros are currently treated as "detected with zero abundance"
   - After conversion, they will be treated as "missing/not detected"

2. AVERAGING IMPACT (Technical replicates):
   - Current behavior: 0 values are included in mean calculation
   - After 0→NaN: 0 values excluded from mean calculation
   - Effect: Averages will increase when zeros are present alongside non-zero values
   - Example: [100, 0, 150] → mean=83.3 BECOMES [100, NaN, 150] → mean=125.0 (+50%)

3. BIOLOGICAL INTERPRETATION:
   - Current: "Protein detected but with zero abundance" (questionable)
   - After fix: "Protein not detected" (more accurate)
   - Rationale: Zero abundance in LC-MS is measurement artifact, not biological reality

4. DOWNSTREAM ANALYSIS IMPACT:
   - Z-score calculation: Will use only non-zero values (reduces denominator inflation)
   - Statistical tests: Sample sizes may vary per protein (appropriate for missing data)
   - Heatmaps: Zeros currently appear as low values; will appear as missing (more honest)

5. RECOMMENDATION:
   - PROCEED with 0→NaN conversion
   - This is a data quality improvement, not data loss
   - Aligns with proteomics field standards (PRIDE, ProteomeXchange)
   - Better reflects technical reality of LC-MS detection limits
    """)

    print("=" * 80)

if __name__ == "__main__":
    main()
