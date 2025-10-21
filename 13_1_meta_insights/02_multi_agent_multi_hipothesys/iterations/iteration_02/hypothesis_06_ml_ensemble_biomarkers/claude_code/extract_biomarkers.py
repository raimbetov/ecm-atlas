#!/usr/bin/env python3
"""
Extract biomarker panel from trained models
Uses feature importances from RF and GB models
"""

import pandas as pd
import numpy as np
import joblib

# Load models
print("Loading models...")
rf_model = joblib.load('rf_model_claude_code.pkl')
lgb_model = joblib.load('gradientboosting_model_claude_code.pkl')

# Load dataset to get feature names
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')
pivot = df.pivot_table(values='Zscore_Delta', index='Tissue', columns='Gene_Symbol', aggfunc='mean')
pivot_filled = pivot.fillna(0)
feature_names = pivot_filled.columns.tolist()

print(f"Number of proteins: {len(feature_names)}")

# Get feature importances
rf_importance = rf_model.feature_importances_
lgb_importance = lgb_model.feature_importances_

# Load SHAP values
shap_rf = np.load('shap_values_rf_claude_code.npy')
shap_lgb = np.load('shap_values_lightgbm_claude_code.npy')
shap_nn = np.load('shap_values_nn_claude_code.npy')

print(f"SHAP RF shape: {shap_rf.shape}")
print(f"SHAP LGB shape: {shap_lgb.shape}")
print(f"SHAP NN shape: {shap_nn.shape}")

# Handle SHAP dimensions
if len(shap_rf.shape) == 3:
    # (n_samples, n_features, n_classes) - get class 1 and average over samples
    mean_abs_shap_rf = np.abs(shap_rf[:, :, 1]).mean(axis=0)
elif len(shap_rf.shape) == 2:
    mean_abs_shap_rf = np.abs(shap_rf).mean(axis=0)
else:
    mean_abs_shap_rf = np.abs(shap_rf)

mean_abs_shap_lgb = np.abs(shap_lgb).mean(axis=0)
mean_abs_shap_nn = np.abs(shap_nn).mean(axis=0)

print(f"Mean SHAP RF shape: {mean_abs_shap_rf.shape}")
print(f"Mean SHAP LGB shape: {mean_abs_shap_lgb.shape}")
print(f"Mean SHAP NN shape: {mean_abs_shap_nn.shape}")

# Normalize all scores
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

norm_rf_imp = normalize(rf_importance)
norm_lgb_imp = normalize(lgb_importance)
norm_rf_shap = normalize(mean_abs_shap_rf)
norm_lgb_shap = normalize(mean_abs_shap_lgb)
norm_nn_shap = normalize(mean_abs_shap_nn)

# Consensus score: average of all normalized scores
consensus_score = (norm_rf_imp + norm_lgb_imp + norm_rf_shap + norm_lgb_shap + norm_nn_shap) / 5

# Create biomarker ranking
biomarker_ranking = pd.DataFrame({
    'Protein': feature_names,
    'RF_Importance': rf_importance,
    'LGB_Importance': lgb_importance,
    'SHAP_RF': mean_abs_shap_rf,
    'SHAP_LGB': mean_abs_shap_lgb,
    'SHAP_NN': mean_abs_shap_nn,
    'Consensus_Score': consensus_score
}).sort_values('Consensus_Score', ascending=False)

print("\nTop 20 proteins by consensus score:")
print(biomarker_ranking.head(20)[['Protein', 'RF_Importance', 'LGB_Importance', 'Consensus_Score']])

# Select top 8 biomarkers
n_biomarkers = 8
biomarker_panel = biomarker_ranking.head(n_biomarkers)

print(f"\n=== BIOMARKER PANEL ({n_biomarkers} proteins) ===")
for idx, row in biomarker_panel.iterrows():
    print(f"{row['Protein']:15s} | Consensus: {row['Consensus_Score']:.4f}")

# Save
biomarker_panel.to_csv('biomarker_panel_claude_code.csv', index=False)
biomarker_ranking.to_csv('biomarker_ranking_full_claude_code.csv', index=False)

print("\n✓ Saved: biomarker_panel_claude_code.csv")
print("✓ Saved: biomarker_ranking_full_claude_code.csv")
