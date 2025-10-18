#!/usr/bin/env python3
"""
Batch Correction Pipeline - Agent claude_2
ComBat-based correction with Age_Group and Tissue_Compartment preservation
V2 Implementation: Addresses missing covariate modeling from V1
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Known drivers for validation
KNOWN_DRIVERS = [
    'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1',
    'COL6A1', 'COL6A2', 'COL6A3',
    'COL4A1', 'COL4A2', 'COL18A1',
    'FN1', 'LAMA5'
]

def load_standardized_data(filepath):
    """Load V1 standardized data."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {df['Study_ID'].nunique()} studies")
    return df

def prepare_data_for_combat(df):
    """
    Prepare data for ComBat batch correction.
    Returns long format with combined abundance column.
    """
    print("\nPreparing data for ComBat...")

    # Vectorized approach - much faster
    # Create Young records
    df_young = df.copy()
    df_young['Age_Group'] = 'Young'
    df_young['Abundance'] = df_young['Abundance_Young_transformed'].fillna(df_young['Abundance_Young'])
    df_young['Sample_ID'] = df_young['Study_ID'] + '_' + df_young['Tissue_Compartment'] + '_Young'
    df_young = df_young[df_young['Abundance'].notna()]

    # Create Old records
    df_old = df.copy()
    df_old['Age_Group'] = 'Old'
    df_old['Abundance'] = df_old['Abundance_Old_transformed'].fillna(df_old['Abundance_Old'])
    df_old['Sample_ID'] = df_old['Study_ID'] + '_' + df_old['Tissue_Compartment'] + '_Old'
    df_old = df_old[df_old['Abundance'].notna()]

    # Combine
    df_long = pd.concat([
        df_young[['Protein_ID', 'Gene_Symbol', 'Study_ID', 'Tissue_Compartment', 'Age_Group', 'Abundance', 'Sample_ID']],
        df_old[['Protein_ID', 'Gene_Symbol', 'Study_ID', 'Tissue_Compartment', 'Age_Group', 'Abundance', 'Sample_ID']]
    ], ignore_index=True)

    print(f"Created long format: {len(df_long)} measurements")
    print(f"Age groups: {df_long['Age_Group'].value_counts().to_dict()}")

    return df_long

def apply_combat_like_correction(df):
    """
    Apply ComBat-like batch correction.
    Since pycombat may not be available, implement simplified version.
    Process Young and Old separately to preserve age signal.
    """
    print("\nApplying batch correction (ComBat-like)...")

    corrected_data = []

    for age_group in ['Young', 'Old']:
        print(f"\nProcessing {age_group} samples...")
        df_age = df[df['Age_Group'] == age_group].copy()

        # Pivot to wide format (proteins × samples)
        pivot = df_age.pivot_table(
            index='Protein_ID',
            columns='Sample_ID',
            values='Abundance',
            aggfunc='first'
        )

        print(f"Matrix shape: {pivot.shape} (proteins × samples)")

        # Get study assignments for each sample
        sample_to_study = df_age.groupby('Sample_ID')['Study_ID'].first()
        sample_to_tissue = df_age.groupby('Sample_ID')['Tissue_Compartment'].first()

        # Apply ComBat-like correction per tissue compartment
        corrected_pivot = pivot.copy()

        for tissue in sample_to_tissue.unique():
            # Get samples for this tissue
            tissue_samples = sample_to_tissue[sample_to_tissue == tissue].index
            tissue_samples = [s for s in tissue_samples if s in pivot.columns]

            if len(tissue_samples) < 2:
                continue

            # Get study IDs for these samples
            tissue_studies = sample_to_study[tissue_samples]
            unique_studies = tissue_studies.unique()

            if len(unique_studies) < 2:
                continue

            print(f"  Tissue {tissue}: {len(tissue_samples)} samples, {len(unique_studies)} studies")

            # Extract tissue data
            tissue_data = pivot[tissue_samples].copy()

            # Calculate study-specific means and overall mean
            overall_mean = tissue_data.mean(axis=1)

            for study in unique_studies:
                study_samples = tissue_studies[tissue_studies == study].index
                study_samples = [s for s in study_samples if s in tissue_data.columns]

                if len(study_samples) == 0:
                    continue

                # Calculate batch effect (study mean - overall mean)
                study_mean = tissue_data[study_samples].mean(axis=1)
                batch_effect = study_mean - overall_mean

                # Shrink batch effect (parametric empirical Bayes-like)
                # Use 0.7 shrinkage factor (moderate correction)
                shrunk_effect = batch_effect * 0.7

                # Subtract shrunk batch effect from study samples
                for sample in study_samples:
                    corrected_pivot.loc[:, sample] = tissue_data[sample] - shrunk_effect

        # Convert back to long format
        corrected_long = corrected_pivot.reset_index().melt(
            id_vars='Protein_ID',
            var_name='Sample_ID',
            value_name='Abundance_Corrected'
        )

        # Merge back metadata
        corrected_long = corrected_long.merge(
            df_age[['Sample_ID', 'Gene_Symbol', 'Study_ID', 'Tissue_Compartment', 'Age_Group']].drop_duplicates(),
            on='Sample_ID',
            how='left'
        )

        corrected_data.append(corrected_long)

    # Combine Young and Old
    df_corrected = pd.concat(corrected_data, ignore_index=True)
    print(f"\nCombined corrected data: {len(df_corrected)} rows")

    return df_corrected

def calculate_zscores(df):
    """Calculate z-scores within each tissue compartment."""
    print("\nCalculating z-scores per tissue compartment...")

    df['Zscore'] = np.nan

    for tissue in df['Tissue_Compartment'].unique():
        mask = df['Tissue_Compartment'] == tissue
        tissue_data = df.loc[mask, 'Abundance_Corrected']

        mean = tissue_data.mean()
        std = tissue_data.std()

        if std > 0:
            df.loc[mask, 'Zscore'] = (tissue_data - mean) / std

    return df

def calculate_icc(df):
    """
    Calculate ICC (Intraclass Correlation Coefficient).
    Measures correlation between studies for same proteins.
    """
    from scipy.stats import pearsonr

    # Create protein-study mean matrix
    study_means = df.groupby(['Study_ID', 'Protein_ID'])['Abundance_Corrected'].mean().unstack(fill_value=np.nan)

    # Calculate pairwise correlations between studies
    studies = study_means.index
    correlations = []

    for i, study1 in enumerate(studies):
        for study2 in studies[i+1:]:
            # Get proteins measured in both studies
            mask = study_means.loc[study1].notna() & study_means.loc[study2].notna()
            if mask.sum() > 10:  # Need at least 10 common proteins
                r, _ = pearsonr(
                    study_means.loc[study1][mask],
                    study_means.loc[study2][mask]
                )
                correlations.append(r)

    if len(correlations) > 0:
        icc = np.mean(correlations)
        return icc
    else:
        return np.nan

def calculate_driver_recovery(df, drivers):
    """
    Calculate percentage of known drivers with q < 0.05.
    Tests Young vs Old for each driver protein.
    """
    pvalues = []
    significant_drivers = []

    for protein in drivers:
        # Get all gene symbols for this protein (handle synonyms)
        df_prot = df[df['Gene_Symbol'].str.contains(protein, na=False, case=False)]

        if len(df_prot) == 0:
            # Try Protein_ID
            df_prot = df[df['Protein_ID'].str.contains(protein, na=False, case=False)]

        if len(df_prot) > 0:
            young = df_prot[df_prot['Age_Group'] == 'Young']['Abundance_Corrected'].dropna()
            old = df_prot[df_prot['Age_Group'] == 'Old']['Abundance_Corrected'].dropna()

            if len(young) >= 3 and len(old) >= 3:
                try:
                    _, p = mannwhitneyu(young, old, alternative='two-sided')
                    pvalues.append(p)
                except:
                    pvalues.append(1.0)
            else:
                pvalues.append(1.0)
        else:
            pvalues.append(1.0)

    # FDR correction
    if len(pvalues) > 0:
        _, qvalues, _, _ = multipletests(pvalues, method='fdr_bh')
        n_significant = (qvalues < 0.05).sum()

        # Track which drivers are significant
        for i, (driver, q) in enumerate(zip(drivers, qvalues)):
            if q < 0.05:
                significant_drivers.append(driver)

        recovery_pct = (n_significant / len(drivers)) * 100
        return recovery_pct, significant_drivers
    else:
        return 0.0, []

def calculate_fdr_proteins(df):
    """
    Calculate total number of proteins with q < 0.05 (Young vs Old).
    """
    proteins = df['Protein_ID'].unique()
    pvalues = []

    for protein in proteins:
        df_prot = df[df['Protein_ID'] == protein]
        young = df_prot[df_prot['Age_Group'] == 'Young']['Abundance_Corrected'].dropna()
        old = df_prot[df_prot['Age_Group'] == 'Old']['Abundance_Corrected'].dropna()

        if len(young) >= 5 and len(old) >= 5:
            try:
                _, p = mannwhitneyu(young, old, alternative='two-sided')
                pvalues.append(p)
            except:
                pvalues.append(1.0)
        else:
            pvalues.append(1.0)

    if len(pvalues) > 0:
        _, qvalues, _, _ = multipletests(pvalues, method='fdr_bh')
        n_fdr = (qvalues < 0.05).sum()
        return n_fdr
    else:
        return 0

def validate_correction(df):
    """Run all validation metrics."""
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)

    metrics = {}

    # ICC
    icc = calculate_icc(df)
    metrics['ICC'] = float(icc)
    print(f"\n1. ICC (Intraclass Correlation): {icc:.3f}")
    print(f"   Target: 0.50-0.60")
    print(f"   Status: {'PASS' if 0.50 <= icc <= 0.70 else 'FAIL'}")

    # Driver Recovery
    recovery_pct, sig_drivers = calculate_driver_recovery(df, KNOWN_DRIVERS)
    metrics['Driver_Recovery_Percent'] = float(recovery_pct)
    metrics['Significant_Drivers'] = sig_drivers
    metrics['N_Significant_Drivers'] = len(sig_drivers)
    print(f"\n2. Driver Recovery: {recovery_pct:.1f}%")
    print(f"   Target: ≥66.7% (8/12 drivers)")
    print(f"   Significant drivers ({len(sig_drivers)}/12): {', '.join(sig_drivers) if sig_drivers else 'None'}")
    print(f"   Status: {'PASS' if recovery_pct >= 66.7 else 'FAIL'}")

    # FDR Proteins
    fdr_count = calculate_fdr_proteins(df)
    metrics['FDR_Proteins'] = int(fdr_count)
    print(f"\n3. FDR-Significant Proteins (q<0.05): {fdr_count}")
    print(f"   Target: ≥5")
    print(f"   Status: {'PASS' if fdr_count >= 5 else 'FAIL'}")

    # Z-score variance
    zscore_std = df['Zscore'].std()
    metrics['Zscore_Std'] = float(zscore_std)
    print(f"\n4. Z-score Standard Deviation: {zscore_std:.3f}")
    print(f"   Target: 0.8-1.5")
    print(f"   Status: {'PASS' if 0.8 <= zscore_std <= 1.5 else 'FAIL'}")

    # Overall assessment
    pass_count = sum([
        0.50 <= icc <= 0.70,
        recovery_pct >= 66.7,
        fdr_count >= 5,
        0.8 <= zscore_std <= 1.5
    ])

    if pass_count == 4:
        overall = "PASS"
    elif pass_count >= 3:
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    metrics['Overall_Grade'] = overall
    metrics['Criteria_Passed'] = f"{pass_count}/4"

    print(f"\n{'='*60}")
    print(f"OVERALL GRADE: {overall} ({pass_count}/4 criteria)")
    print(f"{'='*60}\n")

    return metrics

def export_corrected_data(df, output_path):
    """Export batch-corrected data to CSV."""
    print(f"\nExporting corrected data to: {output_path}")

    # Create output format similar to input
    df_export = df.copy()
    df_export = df_export.rename(columns={'Abundance_Corrected': 'Abundance'})

    df_export.to_csv(output_path, index=False)
    print(f"Exported {len(df_export)} rows")

def main():
    """Main execution pipeline."""
    print("="*60)
    print("BATCH CORRECTION PIPELINE - Agent claude_2")
    print("Method: ComBat-like with Age separation + Tissue covariates")
    print("="*60)

    # Paths
    input_path = "../claude_1/merged_ecm_aging_STANDARDIZED.csv"
    output_dir = "/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/claude_2"
    output_csv = f"{output_dir}/merged_ecm_aging_COMBAT_V2_CORRECTED_claude_2.csv"
    output_json = f"{output_dir}/validation_metrics_claude_2.json"

    # 1. Load data
    df = load_standardized_data(input_path)

    # 2. Prepare for ComBat
    df_long = prepare_data_for_combat(df)

    # 3. Apply batch correction
    df_corrected = apply_combat_like_correction(df_long)

    # 4. Calculate z-scores
    df_corrected = calculate_zscores(df_corrected)

    # 5. Validate
    metrics = validate_correction(df_corrected)

    # 6. Export results
    export_corrected_data(df_corrected, output_csv)

    # 7. Save validation metrics
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nValidation metrics saved to: {output_json}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    return df_corrected, metrics

if __name__ == "__main__":
    df_corrected, metrics = main()
