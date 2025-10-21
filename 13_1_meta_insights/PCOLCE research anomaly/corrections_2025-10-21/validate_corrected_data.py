#!/usr/bin/env python3
"""
Validation script for corrected PCOLCE evidence document
Verifies all study IDs in Table 2.3 exist in database
"""

import pandas as pd
import sys

# Study IDs claimed in corrected Table 2.3 (v1.1)
CLAIMED_STUDIES = {
    'Schuler_2021': {
        'tissues': ['Skeletal_muscle_Soleus', 'Skeletal_muscle_TA',
                   'Skeletal_muscle_EDL', 'Skeletal_muscle_Gastrocnemius'],
        'n_expected': 4,
        'species': 'Mus musculus'
    },
    'Tam_2020': {
        'tissues': ['Intervertebral_disc_NP', 'Intervertebral_disc_IAF',
                   'Intervertebral_disc_OAF'],
        'n_expected': 3,
        'species': 'Homo sapiens'
    },
    'LiDermis_2021': {
        'tissues': ['Skin dermis'],
        'n_expected': 1,
        'species': 'Homo sapiens'
    },
    'Angelidis_2019': {
        'tissues': ['Lung'],
        'n_expected': 1,
        'species': 'Mus musculus'
    },
    'Santinha_2024_Mouse_NT': {
        'tissues': ['Heart_Native_Tissue'],
        'n_expected': 1,
        'species': 'Mus musculus'
    },
    'Santinha_2024_Mouse_DT': {
        'tissues': ['Heart_Decellularized_Tissue'],
        'n_expected': 1,
        'species': 'Mus musculus'
    },
    'Dipali_2023': {
        'tissues': ['Ovary'],
        'n_expected': 1,
        'species': 'Mus musculus'
    }
}

# Studies that should NOT exist (v1.0 errors)
SPURIOUS_STUDIES = [
    'Baranyi_2020', 'Carlson_2019', 'Vogel_2021',
    'Tabula_2020', 'Li_2021', 'Dall_2023'
]

def main():
    print("="*80)
    print("PCOLCE Evidence Document Validation (v1.1)")
    print("="*80)

    # Load database
    try:
        df = pd.read_csv('08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
    except FileNotFoundError:
        print("❌ ERROR: Database not found at 08_merged_ecm_dataset/merged_ecm_aging_zscore.csv")
        print("   Run this script from repository root: python 13_1_meta_insights/PCOLCE research anomaly/validate_corrected_data.py")
        sys.exit(1)

    # Get PCOLCE data (case-insensitive)
    pcolce_df = df[df['Gene_Symbol'].str.upper() == 'PCOLCE'].copy()

    print(f"\n✓ Database loaded: {len(df)} total observations")
    print(f"✓ PCOLCE observations: {len(pcolce_df)}")
    print(f"✓ Unique studies with PCOLCE: {pcolce_df['Study_ID'].nunique()}")

    # Validation 1: Check all claimed studies exist
    print("\n" + "="*80)
    print("VALIDATION 1: Verify all claimed studies exist")
    print("="*80)

    all_valid = True
    for study_id, expected in CLAIMED_STUDIES.items():
        study_data = pcolce_df[pcolce_df['Study_ID'] == study_id]

        if len(study_data) == 0:
            print(f"❌ FAIL: {study_id} not found in database")
            all_valid = False
            continue

        if len(study_data) != expected['n_expected']:
            print(f"⚠️  WARNING: {study_id} has {len(study_data)} observations, expected {expected['n_expected']}")
            all_valid = False

        species = study_data['Species'].iloc[0]
        if species != expected['species']:
            print(f"❌ FAIL: {study_id} species is {species}, expected {expected['species']}")
            all_valid = False
            continue

        tissues = set(study_data['Tissue'].unique())
        expected_tissues = set(expected['tissues'])

        if tissues != expected_tissues:
            print(f"⚠️  WARNING: {study_id} tissue mismatch")
            print(f"   Found: {tissues}")
            print(f"   Expected: {expected_tissues}")
            all_valid = False
        else:
            print(f"✅ PASS: {study_id} ({species}, n={len(study_data)})")

    # Validation 2: Check spurious studies don't exist
    print("\n" + "="*80)
    print("VALIDATION 2: Verify spurious studies are absent")
    print("="*80)

    actual_studies = set(df['Study_ID'].unique())
    spurious_found = []

    for spurious in SPURIOUS_STUDIES:
        if spurious in actual_studies:
            print(f"❌ FAIL: Spurious study '{spurious}' found in database!")
            spurious_found.append(spurious)
            all_valid = False
        else:
            print(f"✅ PASS: '{spurious}' correctly absent")

    # Validation 3: Statistical consistency
    print("\n" + "="*80)
    print("VALIDATION 3: Verify statistical results")
    print("="*80)

    # Overall stats
    mean_dz = pcolce_df['Zscore_Delta'].mean()
    n_decrease = len(pcolce_df[pcolce_df['Zscore_Delta'] < 0])
    consistency = 100 * n_decrease / len(pcolce_df)

    print(f"Mean Δz (pooled): {mean_dz:.3f}")
    print(f"  Expected: -1.41")
    if abs(mean_dz - (-1.41)) < 0.05:
        print(f"  ✅ PASS: Within tolerance")
    else:
        print(f"  ❌ FAIL: Outside expected range")
        all_valid = False

    print(f"\nDirectional consistency: {consistency:.1f}% ({n_decrease}/{len(pcolce_df)} decrease)")
    print(f"  Expected: 91.7% (11/12)")
    if consistency >= 90 and consistency <= 93:
        print(f"  ✅ PASS: Matches expected")
    else:
        print(f"  ⚠️  WARNING: Different from expected")

    # Muscle-specific
    muscle_data = pcolce_df[pcolce_df['Tissue'].str.contains('muscle', case=False, na=False)]
    muscle_mean = muscle_data['Zscore_Delta'].mean()

    print(f"\nMean Δz (muscle only): {muscle_mean:.3f}")
    print(f"  Expected: -3.69")
    if abs(muscle_mean - (-3.69)) < 0.05:
        print(f"  ✅ PASS: Within tolerance")
    else:
        print(f"  ❌ FAIL: Outside expected range")
        all_valid = False

    # Validation 4: Data export
    print("\n" + "="*80)
    print("VALIDATION 4: Export corrected data")
    print("="*80)

    try:
        export_path = '13_1_meta_insights/PCOLCE research anomaly/corrected_table_2.3_data.csv'
        pcolce_df[['Study_ID', 'Species', 'Tissue', 'Method', 'Zscore_Delta']].to_csv(
            export_path, index=False
        )
        print(f"✅ Data exported to: {export_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        all_valid = False

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
        print("   Evidence document v1.1 is accurate and ready for publication")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("   Review errors above and update document accordingly")
        return 1

if __name__ == '__main__':
    sys.exit(main())
