#!/usr/bin/env python3
"""
Agent 3: Adversarial Audit of ECM-Atlas Claims
Purpose: Independently verify all major quantitative claims from agent reports
Approach: Load raw data, reproduce analyses, identify discrepancies
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD RAW DATA
# ============================================================================

print("="*80)
print("AGENT 3: ADVERSARIAL AUDIT - CLAIM VERIFICATION")
print("="*80)

# Load the merged dataset
data_path = '/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv'
print(f"\nLoading data from: {data_path}")
df = pd.read_csv(data_path)

print(f"✓ Data loaded: {len(df)} rows × {len(df.columns)} columns")
print(f"✓ Columns: {list(df.columns)}")

# ============================================================================
# CLAIM 1: Dataset size = 9,343 observations
# ============================================================================

print("\n" + "="*80)
print("CLAIM 1: '9,343 proteomic measurements'")
print("="*80)
expected = 9343
actual = len(df)
print(f"Expected: {expected}")
print(f"Actual:   {actual}")
print(f"Status:   {'✓ VERIFIED' if actual == expected else '✗ DISCREPANCY'}")

# ============================================================================
# CLAIM 2: 405 universal proteins (12.2% of 3,317)
# ============================================================================

print("\n" + "="*80)
print("CLAIM 2: '405 universal proteins (12.2% of 3,317 proteins)'")
print("="*80)

# Need to understand how "universal" is defined
# According to agent_01 report: ≥3 tissues, ≥70% directional consistency

# First, let's see what proteins we have
unique_proteins = df['Protein_ID'].nunique() if 'Protein_ID' in df.columns else df['Gene_Symbol'].nunique()
print(f"Unique proteins in dataset: {unique_proteins}")

# Check if we have Age_Group column
if 'Age_Group' in df.columns:
    print(f"Age groups: {df['Age_Group'].unique()}")
elif 'Age' in df.columns:
    print(f"Age range: {df['Age'].min()} - {df['Age'].max()}")

# ============================================================================
# CLAIM 3: Top proteins by ML methods
# ============================================================================

print("\n" + "="*80)
print("CLAIM 3: 'COL1A1 is #1 ranked protein / master regulator'")
print("="*80)

print("Reports claim:")
print("- Random Forest: COL1A1 #3")
print("- Gradient Boosting: COL1A1 #1")
print("- Overall: Master regulator")

# To verify this, I need to check if we have the right data structure
print(f"\nDataset structure check:")
print(f"- Study_ID present: {'Study_ID' in df.columns}")
print(f"- Tissue present: {'Tissue' in df.columns}")
print(f"- Z_score present: {'Z_score' in df.columns}")
print(f"- Age present: {'Age' in df.columns or 'Age_Group' in df.columns}")

# ============================================================================
# CLAIM 4: 7 GOLD-tier proteins with ≥5 studies, >80% consistency
# ============================================================================

print("\n" + "="*80)
print("CLAIM 4: '7 GOLD-tier proteins (≥5 studies, >80% consistency)'")
print("="*80)

gold_tier_claimed = ['VTN', 'FGB', 'FGA', 'PCOLCE', 'CTSF', 'SERPINH1', 'MFGE8']
print(f"Claimed GOLD-tier proteins: {gold_tier_claimed}")

# ============================================================================
# CLAIM 5: Batch effects dominate 13.34x
# ============================================================================

print("\n" + "="*80)
print("CLAIM 5: 'Study origin separates samples 13.34x MORE than biological age'")
print("="*80)

print("Claimed:")
print("- Study_ID clustering: 0.674")
print("- Age_Group clustering: -0.051")
print("- Ratio: 0.674 / 0.051 ≈ 13.2x")

# ============================================================================
# CLAIM 6: Perfect correlations (r=1.000)
# ============================================================================

print("\n" + "="*80)
print("CLAIM 6: 'Perfect protein correlations (r=1.000)'")
print("="*80)

perfect_corr_claimed = [
    ("CTGF", "IGFALS"),
    ("Asah1", "Lman2"),
    ("CTSD", "TIMP2")
]
print(f"Claimed perfect correlations:")
for p1, p2 in perfect_corr_claimed:
    print(f"  - {p1} ↔ {p2}: r=1.000")

# ============================================================================
# CLAIM 7: Expected lifespan extension
# ============================================================================

print("\n" + "="*80)
print("CLAIM 7: 'Expected lifespan extension: +20-30 years (combination therapy)'")
print("="*80)

print("RED FLAG: This is an extraordinary claim requiring extraordinary evidence")
print("- Based on: Mouse studies? Human data? Modeling?")
print("- Source of estimate: Not clearly documented")
print("- Clinical trial data: None (all targets are pre-clinical)")

# ============================================================================
# DATA EXPLORATION FOR VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("RAW DATA EXPLORATION")
print("="*80)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

# ============================================================================
# SAVE INITIAL VERIFICATION SUMMARY
# ============================================================================

print("\n" + "="*80)
print("INITIAL VERIFICATION COMPLETE")
print("="*80)

print("\nNext steps:")
print("1. Calculate universal protein metrics independently")
print("2. Run ML models to verify protein rankings")
print("3. Calculate batch effect metrics (PCA clustering)")
print("4. Compute protein correlations")
print("5. Trace specific claims to source data")

print("\n" + "="*80)
