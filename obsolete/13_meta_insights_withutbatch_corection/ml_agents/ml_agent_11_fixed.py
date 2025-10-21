#!/usr/bin/env python3
"""ML AGENT 11: RANDOM FOREST FEATURE IMPORTANCE (FIXED)"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML AGENT 11: RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)

df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
print(f"Total records: {len(df):,}")

# Create pivot table
pivot = df[df['Zscore_Delta'].notna()].pivot_table(
    index=['Study_ID', 'Tissue_Compartment'],
    columns='Gene_Symbol',
    values='Zscore_Delta',
    aggfunc='mean'
).fillna(0)

print(f"\nFeature matrix: {pivot.shape[0]} samples × {pivot.shape[1]} proteins")

# Target: average z-score delta (regression instead of classification)
y = df.groupby(['Study_ID', 'Tissue_Compartment'])['Zscore_Delta'].mean()
y = y.loc[pivot.index]

print(f"\nTarget statistics:")
print(f"  Mean: {y.mean():.3f}")
print(f"  Std:  {y.std():.3f}")
print(f"  Range: [{y.min():.3f}, {y.max():.3f}]")

# Train Random Forest Regressor
print("\n🌲 Training Random Forest Regressor...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_split=3,
    random_state=42,
    n_jobs=-1
)

cv_scores = cross_val_score(rf, pivot, y, cv=3, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

rf.fit(pivot, y)

# Extract feature importance
feature_importance = pd.DataFrame({
    'Protein': pivot.columns,
    'RF_Importance': rf.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print("\n🏆 TOP 30 MOST IMPORTANT PROTEINS:")
for i, row in feature_importance.head(30).iterrows():
    print(f"{row['Protein']:15s} → {row['RF_Importance']:.4f}")

# Train Gradient Boosting Regressor
print("\n🚀 Training Gradient Boosting Regressor...")
gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

gb.fit(pivot, y)
cv_scores_gb = cross_val_score(gb, pivot, y, cv=3, scoring='r2')
print(f"GradientBoosting R²: {cv_scores_gb.mean():.3f} ± {cv_scores_gb.std():.3f}")

gb_importance = pd.DataFrame({
    'Protein': pivot.columns,
    'GB_Importance': gb.feature_importances_
}).sort_values('GB_Importance', ascending=False)

print("\n🏆 TOP 20 PROTEINS (Gradient Boosting):")
for i, row in gb_importance.head(20).iterrows():
    print(f"{row['Protein']:15s} → {row['GB_Importance']:.4f}")

# Consensus
comparison = feature_importance.merge(gb_importance, on='Protein')
comparison['Avg_Importance'] = (comparison['RF_Importance'] + comparison['GB_Importance']) / 2
comparison = comparison.sort_values('Avg_Importance', ascending=False)

print("\n🎯 TOP 20 CONSENSUS PROTEINS:")
for i, row in comparison.head(20).iterrows():
    print(f"{row['Protein']:15s} → Avg: {row['Avg_Importance']:.4f}")

comparison.to_csv('10_insights/ml_consensus_importance.csv', index=False)
feature_importance.to_csv('10_insights/ml_rf_feature_importance.csv', index=False)
print("\n✅ ML AGENT 11 COMPLETED")
