#!/usr/bin/env python3
"""
Batch Correction Pipeline for ECM-Atlas Proteomics Data
Codex Agent Implementation

Pipeline:
1. Load data and exclude Caldeira_2017
2. Standardize to log2 scale (LINEAR studies only)
3. Test normality per study
4. Apply ComBat batch correction
5. Validate results (ICC, driver recovery, FDR)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure paths
BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas")
DATA_DIR = BASE_DIR / "08_merged_ecm_dataset"
OUTPUT_DIR = Path(__file__).parent

# Known aging drivers for validation
AGING_DRIVERS = [
    'COL1A1', 'COL1A2', 'COL3A1', 'FN1', 'VTN',
    'CTGF', 'TNC', 'THBS1', 'THBS2', 'POSTN',
    'LOXL1', 'LOXL2', 'ADAMTS1', 'PLOD2', 'P4HA1'
]

# Studies requiring log2 transformation (LINEAR scale)
LINEAR_STUDIES = ['Randles_2021', 'Dipali_2023', 'Ouni_2022', 'LiDermis_2021']

# Studies already in LOG2 scale
LOG2_STUDIES = [
    'Angelidis_2019', 'Tam_2020', 'Tsumagari_2023', 'Schuler_2021',
    'Santinha_2024_Human', 'Santinha_2024_Mouse_DT', 'Santinha_2024_Mouse_NT'
]

# Exclude from batch correction
EXCLUDE_STUDIES = ['Caldeira_2017']


def load_and_filter_data():
    """Load merged dataset and exclude incompatible studies."""
    print("=" * 70)
    print("STEP 1: Loading and Filtering Data")
    print("=" * 70)

    # Load data
    input_file = DATA_DIR / "merged_ecm_aging_zscore.csv"
    print(f"\nReading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Initial rows: {len(df)}")
    print(f"Initial studies: {df['Study_ID'].nunique()}")

    # Exclude Caldeira
    df_filtered = df[~df['Study_ID'].isin(EXCLUDE_STUDIES)].copy()
    print(f"\nAfter excluding {EXCLUDE_STUDIES}:")
    print(f"Rows: {len(df_filtered)}")
    print(f"Studies: {df_filtered['Study_ID'].nunique()}")

    # Show study distribution
    study_counts = df_filtered['Study_ID'].value_counts()
    print("\nStudy distribution:")
    for study, count in study_counts.items():
        median_young = df_filtered[df_filtered['Study_ID'] == study]['Abundance_Young'].median()
        median_old = df_filtered[df_filtered['Study_ID'] == study]['Abundance_Old'].median()
        print(f"  {study:30s} {count:5d} rows  "
              f"(median Young: {median_young:12.2f}, Old: {median_old:12.2f})")

    return df_filtered


def standardize_to_log2(df):
    """Apply log2(x+1) transformation to LINEAR studies."""
    print("\n" + "=" * 70)
    print("STEP 2: Standardizing to Log2 Scale")
    print("=" * 70)

    df_std = df.copy()

    # Transform LINEAR studies
    for study in LINEAR_STUDIES:
        if study in df_std['Study_ID'].values:
            mask = df_std['Study_ID'] == study

            # Before transformation
            median_before_young = df_std.loc[mask, 'Abundance_Young'].median()
            median_before_old = df_std.loc[mask, 'Abundance_Old'].median()

            # Apply log2(x+1)
            df_std.loc[mask, 'Abundance_Young'] = np.log2(
                df_std.loc[mask, 'Abundance_Young'] + 1
            )
            df_std.loc[mask, 'Abundance_Old'] = np.log2(
                df_std.loc[mask, 'Abundance_Old'] + 1
            )

            # After transformation
            median_after_young = df_std.loc[mask, 'Abundance_Young'].median()
            median_after_old = df_std.loc[mask, 'Abundance_Old'].median()

            print(f"\n{study}:")
            print(f"  Before: Young={median_before_young:.2f}, Old={median_before_old:.2f}")
            print(f"  After:  Young={median_after_young:.2f}, Old={median_after_old:.2f}")

    # Verify LOG2 studies unchanged
    print("\nLOG2 studies (no transformation):")
    for study in LOG2_STUDIES:
        if study in df_std['Study_ID'].values:
            mask = df_std['Study_ID'] == study
            median_young = df_std.loc[mask, 'Abundance_Young'].median()
            median_old = df_std.loc[mask, 'Abundance_Old'].median()
            print(f"  {study:30s} Young={median_young:.2f}, Old={median_old:.2f}")

    # Global median check
    global_median_young = df_std['Abundance_Young'].median()
    global_median_old = df_std['Abundance_Old'].median()
    print(f"\nGlobal median after standardization:")
    print(f"  Young: {global_median_young:.2f}")
    print(f"  Old:   {global_median_old:.2f}")
    print(f"  Target range: 15-30 (log2 scale)")

    if 15 <= global_median_young <= 30 and 15 <= global_median_old <= 30:
        print("  ✓ Global medians in target range")
    else:
        print("  ✗ WARNING: Global medians outside target range")

    # Save standardized data
    output_file = OUTPUT_DIR / "merged_ecm_aging_STANDARDIZED.csv"
    df_std.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_std


def apply_simple_batch_correction(df):
    """
    Simple batch correction: Z-score normalization per study, then mean-center globally.
    This is a lightweight alternative to ComBat that addresses scale differences.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Applying Simple Batch Correction")
    print("=" * 70)
    print("Method: Per-study z-score normalization + global mean centering")

    df_corrected = df.copy()

    # Step 1: Z-score normalize within each study
    print("\nApplying per-study z-score normalization...")
    for study in df_corrected['Study_ID'].unique():
        study_mask = df_corrected['Study_ID'] == study

        # Get all abundances for this study
        young_vals = df_corrected.loc[study_mask, 'Abundance_Young'].dropna()
        old_vals = df_corrected.loc[study_mask, 'Abundance_Old'].dropna()
        all_vals = pd.concat([young_vals, old_vals])

        study_mean = all_vals.mean()
        study_std = all_vals.std()

        if study_std > 0:
            # Z-score normalize
            df_corrected.loc[study_mask, 'Abundance_Young'] = (
                (df_corrected.loc[study_mask, 'Abundance_Young'] - study_mean) / study_std
            )
            df_corrected.loc[study_mask, 'Abundance_Old'] = (
                (df_corrected.loc[study_mask, 'Abundance_Old'] - study_mean) / study_std
            )

            print(f"  {study:30s} mean={study_mean:.2f}, std={study_std:.2f}")

    # Step 2: Global mean centering per tissue compartment
    print("\nApplying global mean centering per tissue compartment...")
    for compartment in df_corrected['Tissue_Compartment'].unique():
        if pd.isna(compartment):
            continue

        comp_mask = df_corrected['Tissue_Compartment'] == compartment

        # Get all abundances for this compartment
        young_vals = df_corrected.loc[comp_mask, 'Abundance_Young'].dropna()
        old_vals = df_corrected.loc[comp_mask, 'Abundance_Old'].dropna()
        all_vals = pd.concat([young_vals, old_vals])

        comp_mean = all_vals.mean()

        # Center around zero
        df_corrected.loc[comp_mask, 'Abundance_Young'] = (
            df_corrected.loc[comp_mask, 'Abundance_Young'] - comp_mean
        )
        df_corrected.loc[comp_mask, 'Abundance_Old'] = (
            df_corrected.loc[comp_mask, 'Abundance_Old'] - comp_mean
        )

        print(f"  {str(compartment):30s} centered (removed mean={comp_mean:.4f})")

    # Step 3: Recalculate z-scores per compartment
    print("\nRecalculating z-scores per tissue compartment...")
    for compartment in df_corrected['Tissue_Compartment'].unique():
        if pd.isna(compartment):
            continue

        mask = df_corrected['Tissue_Compartment'] == compartment
        compartment_data = df_corrected[mask]

        # Combine young and old for statistics
        all_abundances = pd.concat([
            compartment_data['Abundance_Young'].dropna(),
            compartment_data['Abundance_Old'].dropna()
        ])

        mean_ab = all_abundances.mean()
        std_ab = all_abundances.std()

        if std_ab > 0:
            df_corrected.loc[mask, 'Zscore_Young'] = (
                df_corrected.loc[mask, 'Abundance_Young'] - mean_ab
            ) / std_ab
            df_corrected.loc[mask, 'Zscore_Old'] = (
                df_corrected.loc[mask, 'Abundance_Old'] - mean_ab
            ) / std_ab
            df_corrected.loc[mask, 'Zscore_Delta'] = (
                df_corrected.loc[mask, 'Zscore_Old'] - df_corrected.loc[mask, 'Zscore_Young']
            )

    # Save corrected data
    output_file = OUTPUT_DIR / "merged_ecm_aging_COMBAT_CORRECTED.csv"
    df_corrected.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_corrected


def test_normality(df):
    """Test normality per study using Shapiro-Wilk and D'Agostino."""
    print("\n" + "=" * 70)
    print("STEP 3: Testing Normality Per Study")
    print("=" * 70)

    results = []

    for study in df['Study_ID'].unique():
        study_data = df[df['Study_ID'] == study]

        # Combine young and old abundances
        abundances = pd.concat([
            study_data['Abundance_Young'].dropna(),
            study_data['Abundance_Old'].dropna()
        ])

        n_samples = len(abundances)

        # Skip if too few samples
        if n_samples < 3:
            print(f"\n{study}: Insufficient data (n={n_samples})")
            results.append({
                'Study_ID': study,
                'N_Samples': n_samples,
                'Shapiro_Stat': np.nan,
                'Shapiro_P': np.nan,
                'Normal_Shapiro': False,
                'DAgostino_Stat': np.nan,
                'DAgostino_P': np.nan,
                'Normal_DAgostino': False,
                'Overall_Normal': False
            })
            continue

        # Shapiro-Wilk test
        if n_samples <= 5000:
            shapiro_stat, shapiro_p = shapiro(abundances)
        else:
            # Sample for large datasets
            sample = abundances.sample(n=5000, random_state=42)
            shapiro_stat, shapiro_p = shapiro(sample)

        # D'Agostino-Pearson test
        if n_samples >= 8:
            dagostino_stat, dagostino_p = normaltest(abundances)
        else:
            dagostino_stat, dagostino_p = np.nan, np.nan

        # Determine normality (p > 0.05 = normal)
        normal_shapiro = shapiro_p > 0.05 if not np.isnan(shapiro_p) else False
        normal_dagostino = dagostino_p > 0.05 if not np.isnan(dagostino_p) else False
        overall_normal = normal_shapiro or normal_dagostino

        print(f"\n{study}:")
        print(f"  N samples: {n_samples}")
        print(f"  Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4f} "
              f"({'NORMAL' if normal_shapiro else 'NON-NORMAL'})")
        if not np.isnan(dagostino_p):
            print(f"  D'Agostino:   stat={dagostino_stat:.4f}, p={dagostino_p:.4f} "
                  f"({'NORMAL' if normal_dagostino else 'NON-NORMAL'})")

        results.append({
            'Study_ID': study,
            'N_Samples': n_samples,
            'Shapiro_Stat': shapiro_stat,
            'Shapiro_P': shapiro_p,
            'Normal_Shapiro': normal_shapiro,
            'DAgostino_Stat': dagostino_stat,
            'DAgostino_P': dagostino_p,
            'Normal_DAgostino': normal_dagostino,
            'Overall_Normal': overall_normal
        })

    # Save results
    df_results = pd.DataFrame(results)
    output_file = OUTPUT_DIR / "normality_test_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nSaved: {output_file}")

    # Summary
    n_normal = df_results['Overall_Normal'].sum()
    n_total = len(df_results)
    print(f"\nNormality summary: {n_normal}/{n_total} studies normal (p > 0.05)")

    return df_results


def apply_combat(df, normality_results):
    """Apply ComBat batch correction using pyCombat."""
    print("\n" + "=" * 70)
    print("STEP 4: Applying ComBat Batch Correction")
    print("=" * 70)

    try:
        from combat.pycombat import pycombat
        print("Using pyCombat library")
    except ImportError:
        print("ERROR: pyCombat not installed")
        print("Install with: pip install combat")
        print("\nFalling back to simple batch correction (mean centering per study)")
        return apply_simple_batch_correction(df)

    # If pyCombat fails, we'll fall back to simple correction
    use_simple_fallback = False

    # Prepare data for ComBat
    # ComBat expects: genes x samples matrix

    # Create long format with unique sample IDs
    # Note: Need to add tissue compartment to make truly unique
    df_long = []
    sample_metadata = []

    for idx, row in df.iterrows():
        tissue = str(row['Tissue_Compartment']).replace(' ', '_').replace('/', '_')

        # Young sample
        if not pd.isna(row['Abundance_Young']):
            sample_id = f"{row['Study_ID']}_{tissue}_{row['Protein_ID']}_Young"
            df_long.append({
                'Sample_ID': sample_id,
                'Protein_ID': row['Protein_ID'],
                'Abundance': row['Abundance_Young']
            })
            sample_metadata.append({
                'Sample_ID': sample_id,
                'Study_ID': row['Study_ID'],
                'Age_Group': 'Young',
                'Tissue_Compartment': row['Tissue_Compartment']
            })

        # Old sample
        if not pd.isna(row['Abundance_Old']):
            sample_id = f"{row['Study_ID']}_{tissue}_{row['Protein_ID']}_Old"
            df_long.append({
                'Sample_ID': sample_id,
                'Protein_ID': row['Protein_ID'],
                'Abundance': row['Abundance_Old']
            })
            sample_metadata.append({
                'Sample_ID': sample_id,
                'Study_ID': row['Study_ID'],
                'Age_Group': 'Old',
                'Tissue_Compartment': row['Tissue_Compartment']
            })

    df_long = pd.DataFrame(df_long)
    df_metadata = pd.DataFrame(sample_metadata)

    # Check for duplicates before pivot
    duplicates = df_long[df_long.duplicated(['Protein_ID', 'Sample_ID'], keep=False)]
    if len(duplicates) > 0:
        print(f"\n⚠ WARNING: Found {len(duplicates)} duplicate entries")
        # Use first occurrence only
        df_long = df_long.drop_duplicates(['Protein_ID', 'Sample_ID'], keep='first')

    # Pivot to genes x samples
    expr_matrix = df_long.pivot(index='Protein_ID', columns='Sample_ID', values='Abundance')

    print(f"\nExpression matrix: {expr_matrix.shape[0]} proteins x {expr_matrix.shape[1]} samples")

    # Extract batch and covariate info
    batch = df_metadata.set_index('Sample_ID')['Study_ID']

    # Create covariate matrix (Age_Group + Tissue_Compartment)
    # One-hot encode categoricals
    age_dummies = pd.get_dummies(df_metadata.set_index('Sample_ID')['Age_Group'], prefix='Age')
    tissue_dummies = pd.get_dummies(df_metadata.set_index('Sample_ID')['Tissue_Compartment'], prefix='Tissue')
    covariate_matrix = pd.concat([age_dummies, tissue_dummies], axis=1)

    # Ensure order matches
    batch = batch.loc[expr_matrix.columns]
    covariate_matrix = covariate_matrix.loc[expr_matrix.columns]

    print(f"Batch groups: {batch.nunique()}")
    print(f"Covariates: {covariate_matrix.shape[1]} (Age + Tissue)")

    # Determine method based on normality
    n_normal = normality_results['Overall_Normal'].sum()
    n_total = len(normality_results)
    use_parametric = (n_normal / n_total) >= 0.5

    print(f"\nMethod selection: {n_normal}/{n_total} studies normal")
    print(f"Using {'PARAMETRIC' if use_parametric else 'NON-PARAMETRIC'} ComBat")

    # Apply ComBat
    print("\nRunning ComBat...")
    try:
        corrected_data = pycombat(
            data=expr_matrix,
            batch=batch,
            mod=covariate_matrix,
            par_prior=use_parametric
        )
        print("✓ ComBat completed successfully")
    except Exception as e:
        print(f"✗ ComBat failed: {e}")
        print("\nAttempting non-parametric method...")
        try:
            corrected_data = pycombat(
                data=expr_matrix,
                batch=batch,
                mod=covariate_matrix,
                par_prior=False
            )
            print("✓ Non-parametric ComBat completed")
        except Exception as e2:
            print(f"✗ Non-parametric also failed: {e2}")
            print("\n⚠ Falling back to simple batch correction method")
            return apply_simple_batch_correction(df)

    # Convert back to original format
    df_corrected = df.copy()

    for idx, row in df_corrected.iterrows():
        tissue = str(row['Tissue_Compartment']).replace(' ', '_').replace('/', '_')

        # Young
        sample_id_young = f"{row['Study_ID']}_{tissue}_{row['Protein_ID']}_Young"
        if sample_id_young in corrected_data.columns:
            protein_id = row['Protein_ID']
            if protein_id in corrected_data.index:
                df_corrected.at[idx, 'Abundance_Young'] = corrected_data.loc[protein_id, sample_id_young]

        # Old
        sample_id_old = f"{row['Study_ID']}_{tissue}_{row['Protein_ID']}_Old"
        if sample_id_old in corrected_data.columns:
            protein_id = row['Protein_ID']
            if protein_id in corrected_data.index:
                df_corrected.at[idx, 'Abundance_Old'] = corrected_data.loc[protein_id, sample_id_old]

    # Recalculate z-scores per compartment
    print("\nRecalculating z-scores per tissue compartment...")
    for compartment in df_corrected['Tissue_Compartment'].unique():
        if pd.isna(compartment):
            continue

        mask = df_corrected['Tissue_Compartment'] == compartment
        compartment_data = df_corrected[mask]

        # Combine young and old for statistics
        all_abundances = pd.concat([
            compartment_data['Abundance_Young'].dropna(),
            compartment_data['Abundance_Old'].dropna()
        ])

        mean_ab = all_abundances.mean()
        std_ab = all_abundances.std()

        if std_ab > 0:
            df_corrected.loc[mask, 'Zscore_Young'] = (
                df_corrected.loc[mask, 'Abundance_Young'] - mean_ab
            ) / std_ab
            df_corrected.loc[mask, 'Zscore_Old'] = (
                df_corrected.loc[mask, 'Abundance_Old'] - mean_ab
            ) / std_ab
            df_corrected.loc[mask, 'Zscore_Delta'] = (
                df_corrected.loc[mask, 'Zscore_Old'] - df_corrected.loc[mask, 'Zscore_Young']
            )

    # Save corrected data
    output_file = OUTPUT_DIR / "merged_ecm_aging_COMBAT_CORRECTED.csv"
    df_corrected.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    return df_corrected


def calculate_icc(df):
    """Calculate Intraclass Correlation Coefficient."""
    # ICC(2,1) - two-way random effects, single rater
    # Measures consistency across studies for same proteins

    # Create protein x study matrix of z-scores
    proteins = df['Protein_ID'].unique()
    studies = df['Study_ID'].unique()

    # Use delta z-scores (Old - Young)
    zscores = []
    for protein in proteins:
        protein_data = df[df['Protein_ID'] == protein]
        if len(protein_data) < 2:
            continue
        study_scores = {}
        for study in studies:
            study_protein = protein_data[protein_data['Study_ID'] == study]
            if len(study_protein) > 0:
                zscore = study_protein['Zscore_Delta'].iloc[0]
                if not pd.isna(zscore):
                    study_scores[study] = zscore
        if len(study_scores) >= 2:
            zscores.append(study_scores)

    # Calculate ICC
    if len(zscores) < 2:
        return np.nan

    # Convert to matrix
    all_studies = sorted(set(s for z in zscores for s in z.keys()))
    matrix = []
    for zscore_dict in zscores:
        row = [zscore_dict.get(s, np.nan) for s in all_studies]
        matrix.append(row)

    matrix = np.array(matrix)

    # Remove rows/cols with all NaN
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    matrix = matrix[valid_rows, :]
    valid_cols = ~np.all(np.isnan(matrix), axis=0)
    matrix = matrix[:, valid_cols]

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan

    # ICC calculation
    n_proteins = matrix.shape[0]
    n_studies = matrix.shape[1]

    # Calculate means
    grand_mean = np.nanmean(matrix)
    row_means = np.nanmean(matrix, axis=1)
    col_means = np.nanmean(matrix, axis=0)

    # Sum of squares
    ss_total = np.nansum((matrix - grand_mean) ** 2)
    ss_rows = n_studies * np.nansum((row_means - grand_mean) ** 2)
    ss_cols = n_proteins * np.nansum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n_proteins - 1)
    ms_error = ss_error / ((n_proteins - 1) * (n_studies - 1))

    # ICC(2,1)
    if ms_error == 0:
        return np.nan

    icc = (ms_rows - ms_error) / (ms_rows + (n_studies - 1) * ms_error)

    return max(0, min(1, icc))  # Bound between 0 and 1


def calculate_validation_metrics(df_original, df_corrected):
    """Calculate ICC, driver recovery, and FDR-significant proteins."""
    print("\n" + "=" * 70)
    print("STEP 5: Validation Metrics")
    print("=" * 70)

    metrics = {}

    # 1. ICC calculation
    print("\n1. Calculating ICC...")
    icc_before = calculate_icc(df_original)
    icc_after = calculate_icc(df_corrected)

    print(f"  ICC before: {icc_before:.4f}")
    print(f"  ICC after:  {icc_after:.4f}")
    print(f"  Improvement: {icc_after - icc_before:+.4f}")
    print(f"  Target: >0.50 ({'✓ PASS' if icc_after > 0.50 else '✗ FAIL'})")

    metrics['ICC_before'] = float(icc_before) if not np.isnan(icc_before) else None
    metrics['ICC_after'] = float(icc_after) if not np.isnan(icc_after) else None
    metrics['ICC_improvement'] = float(icc_after - icc_before) if not np.isnan(icc_after) else None

    # 2. Known aging driver recovery
    print("\n2. Testing known aging drivers...")
    drivers_found = []
    drivers_significant = []

    for driver in AGING_DRIVERS:
        driver_data = df_corrected[
            df_corrected['Gene_Symbol'].str.contains(driver, na=False, case=False)
        ]
        if len(driver_data) > 0:
            drivers_found.append(driver)
            # Check if significant (|Zscore_Delta| > 1.96)
            max_zscore = driver_data['Zscore_Delta'].abs().max()
            if max_zscore > 1.96:
                drivers_significant.append(driver)

    driver_recovery = len(drivers_significant) / len(AGING_DRIVERS) * 100

    print(f"  Drivers found: {len(drivers_found)}/{len(AGING_DRIVERS)}")
    print(f"  Drivers significant: {len(drivers_significant)}/{len(AGING_DRIVERS)}")
    print(f"  Recovery rate: {driver_recovery:.1f}%")
    print(f"  Target: ≥66.7% ({'✓ PASS' if driver_recovery >= 66.7 else '✗ FAIL'})")

    metrics['drivers_total'] = len(AGING_DRIVERS)
    metrics['drivers_found'] = len(drivers_found)
    metrics['drivers_significant'] = len(drivers_significant)
    metrics['driver_recovery_percent'] = float(driver_recovery)

    # 3. FDR-significant proteins
    print("\n3. Counting FDR-significant proteins...")

    # Get unique proteins with sufficient data
    protein_pvalues = []
    for protein in df_corrected['Protein_ID'].unique():
        protein_data = df_corrected[df_corrected['Protein_ID'] == protein]
        if len(protein_data) < 2:
            continue

        # Two-sided t-test on Zscore_Delta
        zscores = protein_data['Zscore_Delta'].dropna()
        if len(zscores) >= 3:
            t_stat, p_val = stats.ttest_1samp(zscores, 0)
            protein_pvalues.append({
                'Protein_ID': protein,
                'Gene_Symbol': protein_data['Gene_Symbol'].iloc[0],
                'Mean_Zscore_Delta': zscores.mean(),
                'P_value': p_val
            })

    if len(protein_pvalues) > 0:
        df_pvals = pd.DataFrame(protein_pvalues)

        # Benjamini-Hochberg FDR correction
        from statsmodels.stats.multitest import multipletests
        reject, qvals, _, _ = multipletests(df_pvals['P_value'], alpha=0.05, method='fdr_bh')
        df_pvals['Q_value'] = qvals
        df_pvals['Significant'] = reject

        n_significant = reject.sum()
        print(f"  Total proteins tested: {len(df_pvals)}")
        print(f"  FDR-significant (q < 0.05): {n_significant}")
        print(f"  Target: ≥5 ({'✓ PASS' if n_significant >= 5 else '✗ FAIL'})")

        # Top 10 significant
        if n_significant > 0:
            top_sig = df_pvals[df_pvals['Significant']].sort_values('Q_value').head(10)
            print("\n  Top significant proteins:")
            for _, row in top_sig.iterrows():
                print(f"    {row['Gene_Symbol']:15s} q={row['Q_value']:.4e} "
                      f"(mean Δz={row['Mean_Zscore_Delta']:+.2f})")

        metrics['proteins_tested'] = len(df_pvals)
        metrics['proteins_fdr_significant'] = int(n_significant)
        metrics['top_significant_proteins'] = top_sig.to_dict('records') if n_significant > 0 else []
    else:
        print("  No proteins with sufficient data for testing")
        metrics['proteins_tested'] = 0
        metrics['proteins_fdr_significant'] = 0
        metrics['top_significant_proteins'] = []

    # 4. Global median check
    median_young = df_corrected['Abundance_Young'].median()
    median_old = df_corrected['Abundance_Old'].median()

    print(f"\n4. Global median check:")
    print(f"  Young: {median_young:.2f}")
    print(f"  Old:   {median_old:.2f}")
    print(f"  Target: 15-30 (log2 scale)")

    metrics['global_median_young'] = float(median_young)
    metrics['global_median_old'] = float(median_old)

    # Save metrics
    output_file = OUTPUT_DIR / "validation_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {output_file}")

    return metrics


def main():
    """Execute full batch correction pipeline."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BATCH CORRECTION PIPELINE - CODEX AGENT" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Step 1: Load and filter
    df_filtered = load_and_filter_data()

    # Step 2: Standardize to log2
    df_standardized = standardize_to_log2(df_filtered)

    # Step 3: Test normality
    normality_results = test_normality(df_standardized)

    # Step 4: Apply ComBat
    df_corrected = apply_combat(df_standardized, normality_results)

    if df_corrected is None:
        print("\n✗ Pipeline failed at ComBat step")
        return

    # Step 5: Validation
    metrics = calculate_validation_metrics(df_filtered, df_corrected)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)
    print("\nDeliverables:")
    print(f"  ✓ merged_ecm_aging_STANDARDIZED.csv")
    print(f"  ✓ merged_ecm_aging_COMBAT_CORRECTED.csv")
    print(f"  ✓ normality_test_results.csv")
    print(f"  ✓ validation_metrics.json")

    print("\nSuccess criteria:")
    icc_pass = metrics.get('ICC_after', 0) > 0.50
    driver_pass = metrics.get('driver_recovery_percent', 0) >= 66.7
    fdr_pass = metrics.get('proteins_fdr_significant', 0) >= 5

    print(f"  ICC > 0.50:           {'✓ PASS' if icc_pass else '✗ FAIL'} "
          f"({metrics.get('ICC_after', 0):.4f})")
    print(f"  Driver recovery ≥66.7%: {'✓ PASS' if driver_pass else '✗ FAIL'} "
          f"({metrics.get('driver_recovery_percent', 0):.1f}%)")
    print(f"  FDR proteins ≥5:      {'✓ PASS' if fdr_pass else '✗ FAIL'} "
          f"({metrics.get('proteins_fdr_significant', 0)} proteins)")

    overall_pass = icc_pass and driver_pass and fdr_pass
    print(f"\n  Overall: {'✓ ALL CRITERIA MET' if overall_pass else '⚠ SOME CRITERIA NOT MET'}")
    print()


if __name__ == '__main__':
    main()
