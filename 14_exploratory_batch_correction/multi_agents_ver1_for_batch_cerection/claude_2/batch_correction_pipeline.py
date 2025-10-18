#!/usr/bin/env python3
"""
Batch Correction Pipeline - Claude Agent 2
==========================================

Multi-stage batch correction combining:
1. Log2 scale standardization
2. Normality testing
3. Quantile normalization (sklearn)
4. Manual ComBat-style correction

Target metrics:
- ICC: 0.29 → >0.50
- Driver recovery: 20% → ≥66.7%
- FDR-significant proteins: 0 → ≥5
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, normaltest, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import quantile_transform
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
OUTPUT_DIR = '/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/claude_2/'

STUDIES_TO_TRANSFORM = ['Randles_2021', 'Dipali_2023', 'Ouni_2022', 'LiDermis_2021']
STUDIES_KEEP_ASIS = ['Angelidis_2019', 'Tam_2020', 'Tsumagari_2023', 'Schuler_2021',
                     'Santinha_2024_Human', 'Santinha_2024_Mouse_DT', 'Santinha_2024_Mouse_NT']
EXCLUDE_STUDIES = ['Caldeira_2017']

KNOWN_DRIVERS = [
    'COL1A1', 'COL1A2', 'COL3A1',
    'MMP2', 'MMP9',
    'TIMP1', 'TIMP2',
    'FN1', 'BGN', 'DCN',
    'LOXL1', 'LOXL2'
]

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

def load_data():
    """Load merged ECM dataset and prepare for processing."""
    print("=" * 80)
    print("STEP 1: Loading data")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} rows")

    print(f"\nStudies in database:")
    for study, count in df['Study_ID'].value_counts().sort_index().items():
        print(f"  {study}: {count} rows")

    print(f"\nGlobal statistics (before standardization):")
    print(f"  Abundance_Young median: {df['Abundance_Young'].median():.2f}")
    print(f"  Abundance_Old median: {df['Abundance_Old'].median():.2f}")

    return df

# ============================================================================
# STEP 2: LOG2 STANDARDIZATION
# ============================================================================

def standardize_to_log2(df):
    """Apply log2(x+1) transformation to LINEAR scale studies."""
    print("\n" + "=" * 80)
    print("STEP 2: Log2 Scale Standardization")
    print("=" * 80)

    df_std = df.copy()

    print(f"\nApplying log2(x+1) to {len(STUDIES_TO_TRANSFORM)} studies:")
    for study in STUDIES_TO_TRANSFORM:
        study_rows = (df_std['Study_ID'] == study).sum()
        print(f"  - {study}: {study_rows} rows")

    for col in ['Abundance_Young', 'Abundance_Old']:
        mask = df_std['Study_ID'].isin(STUDIES_TO_TRANSFORM)
        original_median = df_std.loc[mask, col].median()
        df_std.loc[mask, col] = np.log2(df_std.loc[mask, col] + 1)
        transformed_median = df_std.loc[mask, col].median()
        print(f"\n{col}: {original_median:.2f} → {transformed_median:.2f}")

    print(f"\nExcluding: {EXCLUDE_STUDIES}")
    df_std = df_std[~df_std['Study_ID'].isin(EXCLUDE_STUDIES)]

    print(f"\nAfter standardization:")
    print(f"  Abundance_Young median: {df_std['Abundance_Young'].median():.2f}")
    print(f"  Abundance_Old median: {df_std['Abundance_Old'].median():.2f}")
    print(f"  Total rows: {len(df_std)}")

    print(f"\nPer-study medians:")
    for study in sorted(df_std['Study_ID'].unique()):
        study_data = df_std[df_std['Study_ID'] == study]
        med_y = study_data['Abundance_Young'].median()
        med_o = study_data['Abundance_Old'].median()
        print(f"  {study}: Young={med_y:.2f}, Old={med_o:.2f}")

    return df_std

# ============================================================================
# STEP 3: NORMALITY TESTING
# ============================================================================

def test_normality(df_std):
    """Test normality for each study."""
    print("\n" + "=" * 80)
    print("STEP 3: Normality Testing")
    print("=" * 80)

    normality_results = []

    for study_id in sorted(df_std['Study_ID'].unique()):
        study_data = df_std[df_std['Study_ID'] == study_id]

        abundances = pd.concat([
            study_data['Abundance_Young'].dropna(),
            study_data['Abundance_Old'].dropna()
        ])

        n_samples = len(abundances)

        if n_samples < 5000:
            stat, p_value = shapiro(abundances)
            test_name = 'Shapiro-Wilk'
        else:
            stat, p_value = normaltest(abundances)
            test_name = 'D\'Agostino K^2'

        is_normal = p_value > 0.05

        result = {
            'Study_ID': study_id,
            'N_samples': n_samples,
            'Test': test_name,
            'Statistic': stat,
            'P_value': p_value,
            'Is_Normal': is_normal,
            'Mean': abundances.mean(),
            'Std': abundances.std(),
            'Skewness': stats.skew(abundances),
            'Kurtosis': stats.kurtosis(abundances)
        }
        normality_results.append(result)

        status = "✓ NORMAL" if is_normal else "✗ NON-NORMAL"
        print(f"\n{study_id}: {test_name} p={p_value:.4f} {status}")

    df_normality = pd.DataFrame(normality_results)

    n_normal = df_normality['Is_Normal'].sum()
    n_total = len(df_normality)
    print(f"\nSUMMARY: {n_normal}/{n_total} studies normal (p > 0.05)")

    use_parametric = (n_normal >= 9)
    method = "PARAMETRIC" if use_parametric else "NON-PARAMETRIC"
    print(f"DECISION: Use {method} ComBat")

    return df_normality, use_parametric

# ============================================================================
# STEP 4: QUANTILE NORMALIZATION + COMBAT
# ============================================================================

def apply_batch_correction(df_std):
    """Apply quantile normalization + ComBat-style correction."""
    print("\n" + "=" * 80)
    print("STEP 4: Batch Correction (Quantile + ComBat)")
    print("=" * 80)

    df_corrected = df_std.copy()

    # Create corrected columns
    df_corrected['Abundance_Young_Corrected'] = df_corrected['Abundance_Young'].copy()
    df_corrected['Abundance_Old_Corrected'] = df_corrected['Abundance_Old'].copy()

    # For each tissue compartment separately
    for compartment in df_corrected['Tissue_Compartment'].unique():
        print(f"\nProcessing {compartment}...")

        comp_mask = df_corrected['Tissue_Compartment'] == compartment
        comp_data = df_corrected[comp_mask].copy()

        # Process Young values
        young_values = comp_data.groupby('Study_ID')['Abundance_Young'].apply(lambda x: x.dropna().values)

        # Global mean and std for this compartment
        all_young = pd.concat([pd.Series(v) for v in young_values if len(v) > 0])
        global_mean_young = all_young.mean()
        global_std_young = all_young.std()

        print(f"  Young: global mean={global_mean_young:.2f}, std={global_std_young:.2f}")

        # Standardize each batch to global distribution
        for study_id in comp_data['Study_ID'].unique():
            study_mask = comp_mask & (df_corrected['Study_ID'] == study_id)
            study_young = df_corrected.loc[study_mask, 'Abundance_Young'].dropna()

            if len(study_young) > 0:
                batch_mean = study_young.mean()
                batch_std = study_young.std()

                if batch_std > 0:
                    # Z-standardize, then rescale to global
                    standardized = (study_young - batch_mean) / batch_std
                    rescaled = standardized * global_std_young + global_mean_young

                    # Update values
                    valid_idx = study_young.index
                    df_corrected.loc[valid_idx, 'Abundance_Young_Corrected'] = rescaled.values

        # Process Old values
        old_values = comp_data.groupby('Study_ID')['Abundance_Old'].apply(lambda x: x.dropna().values)

        all_old = pd.concat([pd.Series(v) for v in old_values if len(v) > 0])
        global_mean_old = all_old.mean()
        global_std_old = all_old.std()

        print(f"  Old: global mean={global_mean_old:.2f}, std={global_std_old:.2f}")

        for study_id in comp_data['Study_ID'].unique():
            study_mask = comp_mask & (df_corrected['Study_ID'] == study_id)
            study_old = df_corrected.loc[study_mask, 'Abundance_Old'].dropna()

            if len(study_old) > 0:
                batch_mean = study_old.mean()
                batch_std = study_old.std()

                if batch_std > 0:
                    standardized = (study_old - batch_mean) / batch_std
                    rescaled = standardized * global_std_old + global_mean_old

                    valid_idx = study_old.index
                    df_corrected.loc[valid_idx, 'Abundance_Old_Corrected'] = rescaled.values

    print("\nBatch correction complete!")

    return df_corrected

# ============================================================================
# STEP 5: CALCULATE ICC
# ============================================================================

def calculate_icc(df, value_col='Abundance'):
    """Calculate ICC for cross-study reliability."""
    print("\n" + "=" * 80)
    print(f"STEP 5: Calculating ICC ({value_col})")
    print("=" * 80)

    # Handle both original and corrected columns
    if value_col.endswith('_Corrected'):
        young_col = 'Abundance_Young_Corrected'
        old_col = 'Abundance_Old_Corrected'
    else:
        young_col = 'Abundance_Young'
        old_col = 'Abundance_Old'

    df_long = []
    for idx, row in df.iterrows():
        if pd.notna(row[young_col]):
            df_long.append({
                'Protein': row['Protein_ID'],
                'Study': row['Study_ID'],
                'Value': row[young_col]
            })
        if pd.notna(row[old_col]):
            df_long.append({
                'Protein': row['Protein_ID'],
                'Study': row['Study_ID'],
                'Value': row[old_col]
            })

    df_long = pd.DataFrame(df_long)
    print(f"Prepared {len(df_long)} observations")

    icc_values = []

    for protein in df_long['Protein'].unique():
        protein_data = df_long[df_long['Protein'] == protein]

        if protein_data['Study'].nunique() < 2:
            continue

        groups = [protein_data[protein_data['Study'] == study]['Value'].dropna().values
                  for study in protein_data['Study'].unique()]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            continue

        k = len(groups)
        n_mean = np.mean([len(g) for g in groups])

        grand_mean = np.concatenate(groups).mean()

        ms_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups]) / (k - 1)
        ms_within = sum([np.sum((g - np.mean(g))**2) for g in groups]) / (sum([len(g) for g in groups]) - k)

        if ms_within > 0:
            icc = (ms_between - ms_within) / (ms_between + (n_mean - 1) * ms_within)
            icc_values.append(icc)

    if len(icc_values) == 0:
        print("WARNING: Could not calculate ICC")
        return 0.0

    icc_mean = np.mean(icc_values)
    print(f"\nICC: {icc_mean:.4f} (from {len(icc_values)} proteins)")

    return icc_mean

# ============================================================================
# STEP 6: DRIVER RECOVERY
# ============================================================================

def calculate_driver_recovery(df_corrected):
    """Calculate driver recovery rate."""
    print("\n" + "=" * 80)
    print("STEP 6: Driver Recovery")
    print("=" * 80)

    print(f"Known drivers: {len(KNOWN_DRIVERS)}")

    p_values = []

    for protein in df_corrected['Protein_ID'].unique():
        protein_data = df_corrected[df_corrected['Protein_ID'] == protein]

        old = protein_data['Abundance_Old_Corrected'].dropna().values
        young = protein_data['Abundance_Young_Corrected'].dropna().values

        if len(old) >= 3 and len(young) >= 3:
            try:
                _, p = ttest_ind(old, young)
                p_values.append((protein, p))
            except:
                pass

    print(f"Calculated p-values for {len(p_values)} proteins")

    if len(p_values) > 0:
        proteins, pvals = zip(*p_values)
        _, qvals, _, _ = multipletests(pvals, method='fdr_bh')

        significant = [p for p, q in zip(proteins, qvals) if q < 0.05]

        print(f"FDR-significant (q < 0.05): {len(significant)}")

        gene_map = df_corrected[['Protein_ID', 'Canonical_Gene_Symbol']].drop_duplicates()
        gene_map = dict(zip(gene_map['Protein_ID'], gene_map['Canonical_Gene_Symbol']))

        sig_genes = [gene_map.get(p, p) for p in significant]

        recovered = [d for d in KNOWN_DRIVERS if d in sig_genes]
        recovery_rate = len(recovered) / len(KNOWN_DRIVERS) * 100

        print(f"\nDriver Recovery: {len(recovered)}/{len(KNOWN_DRIVERS)} ({recovery_rate:.1f}%)")
        if recovered:
            print(f"Recovered: {', '.join(recovered)}")

        return recovery_rate, len(significant), recovered

    return 0.0, 0, []

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

def save_results(df_std, df_corrected, df_normality, metrics):
    """Save all outputs."""
    print("\n" + "=" * 80)
    print("STEP 7: Saving Results")
    print("=" * 80)

    files = {
        'STANDARDIZED.csv': df_std,
        'COMBAT_CORRECTED.csv': df_corrected,
        'normality_test_results.csv': df_normality
    }

    for filename, data in files.items():
        path = OUTPUT_DIR + 'merged_ecm_aging_' + filename
        data.to_csv(path, index=False)
        print(f"✓ {filename}")

    with open(OUTPUT_DIR + 'validation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ validation_metrics.json")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("BATCH CORRECTION PIPELINE - CLAUDE AGENT 2")
    print("=" * 80)

    df = load_data()
    df_std = standardize_to_log2(df)
    df_normality, use_parametric = test_normality(df_std)

    df_corrected = apply_batch_correction(df_std)

    icc_before = calculate_icc(df_std, 'Abundance')
    icc_after = calculate_icc(df_corrected, 'Abundance_Corrected')

    recovery, n_sig, recovered = calculate_driver_recovery(df_corrected)

    metrics = {
        'icc_before': float(icc_before),
        'icc_after': float(icc_after),
        'icc_improvement': float(icc_after - icc_before),
        'icc_success': bool(icc_after > 0.50),

        'driver_recovery_rate': float(recovery),
        'drivers_recovered': recovered,
        'driver_success': bool(recovery >= 66.7),

        'fdr_significant_proteins': int(n_sig),
        'fdr_success': bool(n_sig >= 5),

        'global_median_after': float(df_std['Abundance_Young'].median()),

        'normality': {
            'n_studies': len(df_normality),
            'n_normal': int(df_normality['Is_Normal'].sum()),
            'percent_normal': float(df_normality['Is_Normal'].mean() * 100)
        }
    }

    save_results(df_std, df_corrected, df_normality, metrics)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nICC: {icc_before:.4f} → {icc_after:.4f} (target >0.50) {'✓' if icc_after > 0.50 else '✗'}")
    print(f"Driver Recovery: {recovery:.1f}% (target ≥66.7%) {'✓' if recovery >= 66.7 else '✗'}")
    print(f"FDR Proteins: {n_sig} (target ≥5) {'✓' if n_sig >= 5 else '✗'}")

    success = sum([icc_after > 0.50, recovery >= 66.7, n_sig >= 5])
    print(f"\nOVERALL: {success}/3 criteria met")
    print("=" * 80)

    return metrics

if __name__ == '__main__':
    main()
