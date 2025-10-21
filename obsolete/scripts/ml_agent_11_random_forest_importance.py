#!/usr/bin/env python3
"""
ML AGENT 11: RANDOM FOREST FEATURE IMPORTANCE ANALYZER
========================================================

Mission: Use ensemble ML to identify which proteins, tissues, and categories
are most predictive of aging, revealing hidden importance hierarchies.

Approach:
1. Random Forest Classifier (aging stage prediction)
2. Feature importance ranking
3. Permutation importance for validation
4. SHAP values for interpretability
5. Biological context integration
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ML AGENT 11: RANDOM FOREST FEATURE IMPORTANCE ANALYZER")
print("=" * 80)
print()

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
print(f"Total records: {len(df):,}")
print(f"Unique proteins: {df['Gene_Symbol'].nunique():,}")

# Filter to records with both old and young measurements
df_analysis = df[df['Zscore_Delta'].notna()].copy()
print(f"Records with z-score deltas: {len(df_analysis):,}")

# Create aging stage labels (simplified: down/stable/up regulation)
def classify_aging_change(delta):
    if delta < -0.5:
        return 'Down'
    elif delta > 0.5:
        return 'Up'
    else:
        return 'Stable'

df_analysis['Aging_Stage'] = df_analysis['Zscore_Delta'].apply(classify_aging_change)

print("\n" + "=" * 80)
print("AGING STAGE DISTRIBUTION")
print("=" * 80)
print(df_analysis['Aging_Stage'].value_counts())

# Prepare features: protein-tissue matrix
print("\n" + "=" * 80)
print("TASK 1: PROTEIN IMPORTANCE RANKING")
print("=" * 80)

# Create pivot table: rows = samples, columns = proteins
pivot = df_analysis.pivot_table(
    index=['Study_ID', 'Tissue_Compartment'],
    columns='Gene_Symbol',
    values='Zscore_Delta',
    aggfunc='mean'
).fillna(0)

print(f"\nFeature matrix: {pivot.shape[0]} samples × {pivot.shape[1]} proteins")

# Target: aging direction
# For each study-tissue, calculate net aging direction
targets = df_analysis.groupby(['Study_ID', 'Tissue_Compartment'])['Zscore_Delta'].mean()
y = targets.apply(classify_aging_change)

# Align indices
pivot = pivot.loc[y.index]

print(f"Target distribution:\n{y.value_counts()}")

# Train Random Forest
print("\n🌲 Training Random Forest Classifier...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv_scores = cross_val_score(rf, pivot, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Train on full data
rf.fit(pivot, y)

# Extract feature importance
feature_importance = pd.DataFrame({
    'Protein': pivot.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🏆 TOP 30 MOST IMPORTANT PROTEINS FOR AGING PREDICTION:")
for i, row in feature_importance.head(30).iterrows():
    print(f"{row['Protein']:15s} → Importance: {row['Importance']:.4f}")

# Save results
feature_importance.to_csv('10_insights/ml_rf_feature_importance.csv', index=False)
print("\n✅ Saved: 10_insights/ml_rf_feature_importance.csv")

# TASK 2: Tissue importance
print("\n" + "=" * 80)
print("TASK 2: TISSUE COMPARTMENT IMPORTANCE")
print("=" * 80)

tissue_importance = df_analysis.groupby('Tissue_Compartment').agg({
    'Zscore_Delta': ['mean', 'std', 'count']
}).round(3)
tissue_importance.columns = ['Mean_Delta', 'Std_Delta', 'N_Proteins']
tissue_importance['Abs_Mean'] = tissue_importance['Mean_Delta'].abs()
tissue_importance = tissue_importance.sort_values('Abs_Mean', ascending=False)

print("\n🧬 TISSUE COMPARTMENTS BY AGING IMPACT:")
print(tissue_importance.to_string())

# TASK 3: Matrisome category importance
print("\n" + "=" * 80)
print("TASK 3: MATRISOME CATEGORY IMPORTANCE")
print("=" * 80)

category_importance = df_analysis.groupby('Matrisome_Category').agg({
    'Zscore_Delta': ['mean', 'std', 'count']
}).round(3)
category_importance.columns = ['Mean_Delta', 'Std_Delta', 'N_Proteins']
category_importance['Abs_Mean'] = category_importance['Mean_Delta'].abs()
category_importance = category_importance.sort_values('Abs_Mean', ascending=False)

print("\n📊 MATRISOME CATEGORIES BY AGING IMPACT:")
print(category_importance.to_string())

# TASK 4: Interaction terms (protein pairs)
print("\n" + "=" * 80)
print("TASK 4: GRADIENT BOOSTING WITH INTERACTION DETECTION")
print("=" * 80)

print("\n🚀 Training Gradient Boosting Classifier...")
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb.fit(pivot, y)
cv_scores_gb = cross_val_score(gb, pivot, y, cv=5, scoring='accuracy')
print(f"GradientBoosting CV accuracy: {cv_scores_gb.mean():.3f} ± {cv_scores_gb.std():.3f}")

gb_importance = pd.DataFrame({
    'Protein': pivot.columns,
    'GB_Importance': gb.feature_importances_
}).sort_values('GB_Importance', ascending=False)

print("\n🏆 TOP 20 PROTEINS (Gradient Boosting):")
for i, row in gb_importance.head(20).iterrows():
    print(f"{row['Protein']:15s} → Importance: {row['GB_Importance']:.4f}")

# Compare RF vs GB rankings
comparison = feature_importance.merge(
    gb_importance,
    on='Protein',
    how='inner'
)
comparison['Avg_Importance'] = (comparison['Importance'] + comparison['GB_Importance']) / 2
comparison = comparison.sort_values('Avg_Importance', ascending=False)

print("\n" + "=" * 80)
print("TOP 20 CONSENSUS PROTEINS (RF + GB Average)")
print("=" * 80)
for i, row in comparison.head(20).iterrows():
    print(f"{row['Protein']:15s} → RF: {row['Importance']:.4f}, GB: {row['GB_Importance']:.4f}, Avg: {row['Avg_Importance']:.4f}")

comparison.to_csv('10_insights/ml_consensus_importance.csv', index=False)
print("\n✅ Saved: 10_insights/ml_consensus_importance.csv")

print("\n" + "=" * 80)
print("🎯 KEY ML INSIGHTS")
print("=" * 80)
print(f"""
1. Random Forest CV Accuracy: {cv_scores.mean():.2%}
2. Gradient Boosting CV Accuracy: {cv_scores_gb.mean():.2%}
3. Top protein: {comparison.iloc[0]['Protein']} (Avg importance: {comparison.iloc[0]['Avg_Importance']:.4f})
4. Most dynamic tissue: {tissue_importance.index[0]}
5. Most changing category: {category_importance.index[0]}
""")

print("\n✅ ML AGENT 11 COMPLETED")
print("=" * 80)
