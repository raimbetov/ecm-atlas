#!/usr/bin/env python3
"""
Find common aging signatures across tissues (Disc vs Kidney)
Identify proteins that change in the SAME DIRECTION in both datasets
"""

import pandas as pd
import numpy as np

def load_dataset(path, name):
    """Load and prepare dataset"""
    df = pd.read_csv(path)
    # Filter ECM proteins only
    if 'Match_Confidence' in df.columns:
        df = df[df['Match_Confidence'] > 0].copy()

    # Keep only proteins with valid aging data
    df = df.dropna(subset=['Zscore_Delta'])

    # Normalize gene symbols
    df['Gene_Upper'] = df['Gene_Symbol'].str.upper().str.strip()

    print(f"\n{name}: {len(df)} ECM proteins with aging data")

    return df[['Gene_Upper', 'Gene_Symbol', 'Protein_Name', 'Zscore_Delta',
               'Zscore_Young', 'Zscore_Old', 'Matrisome_Category',
               'Abundance_Young', 'Abundance_Old']]

def find_common_proteins(tam_df, randles_df):
    """Find proteins present in both datasets"""

    tam_genes = set(tam_df['Gene_Upper'].unique())
    randles_genes = set(randles_df['Gene_Upper'].unique())

    common_genes = tam_genes & randles_genes

    print(f"\n{'='*80}")
    print(f"OVERLAP ANALYSIS")
    print(f"{'='*80}")
    print(f"Tam 2020 (Disc) unique proteins: {len(tam_genes)}")
    print(f"Randles 2021 (Kidney) unique proteins: {len(randles_genes)}")
    print(f"Common proteins (overlap): {len(common_genes)}")
    print(f"Overlap percentage: {len(common_genes)/len(tam_genes)*100:.1f}% (Tam) | {len(common_genes)/len(randles_genes)*100:.1f}% (Randles)")

    return common_genes

def analyze_common_signatures(tam_df, randles_df, common_genes):
    """Analyze aging signatures for common proteins"""

    # Filter to common proteins
    tam_common = tam_df[tam_df['Gene_Upper'].isin(common_genes)].copy()
    randles_common = randles_df[randles_df['Gene_Upper'].isin(common_genes)].copy()

    # Merge on gene symbol
    merged = pd.merge(
        tam_common,
        randles_common,
        on='Gene_Upper',
        suffixes=('_Disc', '_Kidney')
    )

    # Calculate direction concordance
    merged['Same_Direction'] = np.sign(merged['Zscore_Delta_Disc']) == np.sign(merged['Zscore_Delta_Kidney'])
    merged['Both_Upregulated'] = (merged['Zscore_Delta_Disc'] > 0.5) & (merged['Zscore_Delta_Kidney'] > 0.5)
    merged['Both_Downregulated'] = (merged['Zscore_Delta_Disc'] < -0.5) & (merged['Zscore_Delta_Kidney'] < -0.5)

    # Calculate average z-score delta
    merged['Avg_Zscore_Delta'] = (merged['Zscore_Delta_Disc'] + merged['Zscore_Delta_Kidney']) / 2

    # Classify aging signature
    def classify_signature(row):
        if row['Both_Upregulated']:
            return 'Pan-tissue UPREGULATION'
        elif row['Both_Downregulated']:
            return 'Pan-tissue DOWNREGULATION'
        elif row['Same_Direction']:
            if row['Zscore_Delta_Disc'] > 0:
                return 'Concordant upregulation (weak)'
            else:
                return 'Concordant downregulation (weak)'
        else:
            return 'Discordant (tissue-specific)'

    merged['Aging_Signature'] = merged.apply(classify_signature, axis=1)

    return merged

def print_pan_tissue_signatures(merged):
    """Print proteins with consistent aging signatures across tissues"""

    print(f"\n{'='*80}")
    print(f"PAN-TISSUE AGING SIGNATURES (Common to Both Disc & Kidney)")
    print(f"{'='*80}\n")

    # Pan-tissue upregulation
    pan_up = merged[merged['Both_Upregulated']].sort_values('Avg_Zscore_Delta', ascending=False)

    print(f"🔴 PAN-TISSUE UPREGULATED PROTEINS ({len(pan_up)} proteins)")
    print(f"   (Increased with aging in BOTH tissues)")
    print(f"{'='*80}\n")

    if len(pan_up) > 0:
        for idx, row in pan_up.iterrows():
            print(f"Gene: {row['Gene_Symbol_Disc']}")
            print(f"Protein: {row['Protein_Name_Disc']}")
            print(f"Category: {row['Matrisome_Category_Disc']}")
            print(f"  Disc:   Δz = {row['Zscore_Delta_Disc']:+.3f} (Young: {row['Zscore_Young_Disc']:.2f} → Old: {row['Zscore_Old_Disc']:.2f})")
            print(f"  Kidney: Δz = {row['Zscore_Delta_Kidney']:+.3f} (Young: {row['Zscore_Young_Kidney']:.2f} → Old: {row['Zscore_Old_Kidney']:.2f})")
            print(f"  Average: Δz = {row['Avg_Zscore_Delta']:+.3f}")
            print()
    else:
        print("  None found with Δz > 0.5 in both tissues\n")

    # Pan-tissue downregulation
    pan_down = merged[merged['Both_Downregulated']].sort_values('Avg_Zscore_Delta', ascending=True)

    print(f"{'='*80}")
    print(f"🔵 PAN-TISSUE DOWNREGULATED PROTEINS ({len(pan_down)} proteins)")
    print(f"   (Decreased with aging in BOTH tissues)")
    print(f"{'='*80}\n")

    if len(pan_down) > 0:
        for idx, row in pan_down.iterrows():
            print(f"Gene: {row['Gene_Symbol_Disc']}")
            print(f"Protein: {row['Protein_Name_Disc']}")
            print(f"Category: {row['Matrisome_Category_Disc']}")
            print(f"  Disc:   Δz = {row['Zscore_Delta_Disc']:+.3f} (Young: {row['Zscore_Young_Disc']:.2f} → Old: {row['Zscore_Old_Disc']:.2f})")
            print(f"  Kidney: Δz = {row['Zscore_Delta_Kidney']:+.3f} (Young: {row['Zscore_Young_Kidney']:.2f} → Old: {row['Zscore_Old_Kidney']:.2f})")
            print(f"  Average: Δz = {row['Avg_Zscore_Delta']:+.3f}")
            print()
    else:
        print("  None found with Δz < -0.5 in both tissues\n")

    # Concordant weak signals
    concordant = merged[merged['Same_Direction'] & ~merged['Both_Upregulated'] & ~merged['Both_Downregulated']]
    concordant_up = concordant[concordant['Zscore_Delta_Disc'] > 0].sort_values('Avg_Zscore_Delta', ascending=False)
    concordant_down = concordant[concordant['Zscore_Delta_Disc'] < 0].sort_values('Avg_Zscore_Delta', ascending=True)

    print(f"{'='*80}")
    print(f"↗️ CONCORDANT UPREGULATION (Weak, {len(concordant_up)} proteins)")
    print(f"   (Same direction but |Δz| < 0.5 in at least one tissue)")
    print(f"{'='*80}\n")

    if len(concordant_up) > 0:
        for idx, row in concordant_up.head(10).iterrows():
            print(f"{row['Gene_Symbol_Disc']:12s} | Disc: {row['Zscore_Delta_Disc']:+.3f} | Kidney: {row['Zscore_Delta_Kidney']:+.3f} | Avg: {row['Avg_Zscore_Delta']:+.3f}")
        if len(concordant_up) > 10:
            print(f"... and {len(concordant_up)-10} more")
        print()

    print(f"{'='*80}")
    print(f"↘️ CONCORDANT DOWNREGULATION (Weak, {len(concordant_down)} proteins)")
    print(f"   (Same direction but |Δz| < 0.5 in at least one tissue)")
    print(f"{'='*80}\n")

    if len(concordant_down) > 0:
        for idx, row in concordant_down.head(10).iterrows():
            print(f"{row['Gene_Symbol_Disc']:12s} | Disc: {row['Zscore_Delta_Disc']:+.3f} | Kidney: {row['Zscore_Delta_Kidney']:+.3f} | Avg: {row['Avg_Zscore_Delta']:+.3f}")
        if len(concordant_down) > 10:
            print(f"... and {len(concordant_down)-10} more")
        print()

    # Discordant
    discordant = merged[~merged['Same_Direction']]

    print(f"{'='*80}")
    print(f"🔀 DISCORDANT (TISSUE-SPECIFIC) RESPONSES ({len(discordant)} proteins)")
    print(f"   (Opposite directions in Disc vs Kidney)")
    print(f"{'='*80}\n")

    if len(discordant) > 0:
        # Sort by absolute difference
        discordant['Abs_Diff'] = (discordant['Zscore_Delta_Disc'] - discordant['Zscore_Delta_Kidney']).abs()
        discordant = discordant.sort_values('Abs_Diff', ascending=False)

        for idx, row in discordant.head(10).iterrows():
            direction_disc = "↑" if row['Zscore_Delta_Disc'] > 0 else "↓"
            direction_kidney = "↑" if row['Zscore_Delta_Kidney'] > 0 else "↓"
            print(f"{row['Gene_Symbol_Disc']:12s} | Disc: {direction_disc} {row['Zscore_Delta_Disc']:+.3f} | Kidney: {direction_kidney} {row['Zscore_Delta_Kidney']:+.3f} | Diff: {row['Abs_Diff']:.3f}")

        if len(discordant) > 10:
            print(f"... and {len(discordant)-10} more")
        print()

    return pan_up, pan_down, concordant_up, concordant_down, discordant

def summarize_statistics(merged):
    """Print summary statistics"""

    total = len(merged)
    pan_up = len(merged[merged['Both_Upregulated']])
    pan_down = len(merged[merged['Both_Downregulated']])
    concordant = len(merged[merged['Same_Direction'] & ~merged['Both_Upregulated'] & ~merged['Both_Downregulated']])
    discordant = len(merged[~merged['Same_Direction']])

    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    print(f"Total common proteins: {total}")
    print(f"  Pan-tissue upregulated (strong):   {pan_up:3d} ({pan_up/total*100:5.1f}%)")
    print(f"  Pan-tissue downregulated (strong): {pan_down:3d} ({pan_down/total*100:5.1f}%)")
    print(f"  Concordant (weak):                 {concordant:3d} ({concordant/total*100:5.1f}%)")
    print(f"  Discordant (tissue-specific):      {discordant:3d} ({discordant/total*100:5.1f}%)")
    print(f"\nOverall concordance rate: {(pan_up + pan_down + concordant)/total*100:.1f}%")
    print()

def main():
    """Main analysis"""

    print("\n" + "="*80)
    print("COMMON AGING SIGNATURES ACROSS TISSUES")
    print("="*80)
    print("\nDatasets:")
    print("1. Tam 2020 - Intervertebral Disc NP (Human)")
    print("2. Randles 2021 - Kidney Glomerular (Mouse)")
    print("\nGoal: Identify ECM proteins that change consistently across tissues")

    # Load datasets
    tam_df = load_dataset(
        "07_Tam_2020_paper_to_csv/claude_code/Tam_2020_NP_zscore.csv",
        "Tam 2020 (Disc)"
    )

    randles_df = load_dataset(
        "06_Randles_z_score_by_tissue_compartment/claude_code/Randles_2021_Glomerular_zscore.csv",
        "Randles 2021 (Kidney)"
    )

    # Find overlap
    common_genes = find_common_proteins(tam_df, randles_df)

    # Analyze common signatures
    merged = analyze_common_signatures(tam_df, randles_df, common_genes)

    # Print results
    pan_up, pan_down, concordant_up, concordant_down, discordant = print_pan_tissue_signatures(merged)

    # Summary statistics
    summarize_statistics(merged)

    # Export results
    output_file = "COMMON_AGING_SIGNATURES.csv"
    merged_export = merged.sort_values('Avg_Zscore_Delta', ascending=False)
    merged_export.to_csv(output_file, index=False)
    print(f"\n✅ Results exported to: {output_file}")

    # Print conclusion
    print(f"\n{'='*80}")
    print(f"BIOLOGICAL INTERPRETATION")
    print(f"{'='*80}\n")

    print("🔬 Pan-tissue aging signatures represent:")
    print("  • Universal ECM aging mechanisms across organs")
    print("  • Potential systemic aging biomarkers")
    print("  • High-priority therapeutic targets (affect multiple tissues)")
    print()

    print("🔬 Discordant signatures represent:")
    print("  • Tissue-specific adaptations to aging")
    print("  • Organ-specific therapeutic opportunities")
    print("  • Context-dependent ECM remodeling")
    print()

if __name__ == "__main__":
    main()
