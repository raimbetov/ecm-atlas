# Hypothesis 06: ML Ensemble Discovers Optimal Aging Biomarker Panel

## Scientific Question

Can ensemble machine learning (Random Forest + XGBoost + Neural Networks) combined with SHAP interpretability identify a minimal 5-10 protein biomarker panel that predicts tissue aging velocity with >85% accuracy, outperforming single-algorithm approaches?

## Background Context

**Clinical Need:** Simple blood test for "aging clock" requires minimal biomarkers.

**ML Hypothesis:** Ensemble methods combine complementary algorithms:
- Random Forest: Captures non-linear interactions, feature importance
- XGBoost: Gradient boosting for complex patterns
- Neural Networks: Universal function approximation

SHAP values provide unified interpretability across models.

**Expected Discovery:** 5-10 protein panel (likely serpins + collagens + coagulation factors) predicts aging with AUC > 0.90, validated across tissues.

## Data Source

```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

## ML Requirements (MANDATORY)

### Must Use All of These:

1. **Random Forest (Required):**
   - 100-500 trees
   - Feature importance extraction
   - Out-of-bag (OOB) error for validation

2. **XGBoost (Required):**
   - Gradient boosting decision trees
   - Hyperparameter tuning (learning rate, max_depth, n_estimators)
   - Early stopping to prevent overfitting

3. **Neural Network Classifier (Required):**
   - Multi-layer perceptron (MLP)
   - Architecture: Input → 128 → 64 → 32 → Output
   - Dropout, batch normalization

4. **Ensemble Stacking:**
   - Meta-learner combines RF + XGBoost + NN predictions
   - Logistic regression or another XGBoost as meta-model

5. **SHAP Interpretability (Required):**
   - TreeSHAP for RF and XGBoost
   - DeepSHAP or KernelSHAP for neural network
   - Aggregate SHAP values across models
   - Identify consensus important proteins

6. **Feature Selection:**
   - Recursive Feature Elimination (RFE)
   - Select minimal biomarker panel (5-10 proteins)
   - Validate panel on hold-out test set

## Success Criteria

### Criterion 1: Multi-Model Training (40 pts)

**Required:**
1. Define prediction task:
   - **Binary:** Fast-aging vs Slow-aging tissues (median split on velocity)
   - **OR Multi-class:** Low / Medium / High aging velocity
   - **OR Regression:** Predict continuous aging velocity
2. Train 3 models independently:
   - Random Forest: n_estimators=200, max_depth=10
   - XGBoost: learning_rate=0.1, max_depth=6, n_estimators=100
   - Neural Network: 3 hidden layers, Adam optimizer
3. Evaluate each model:
   - Classification: Accuracy, Precision, Recall, F1, AUC-ROC
   - Regression: R², MAE, RMSE
4. Compare model performance:
   - Which single model is best?
   - Does ensemble beat best single model?

**Deliverables:**
- `rf_model_[agent].pkl` - Trained Random Forest
- `xgboost_model_[agent].pkl` - Trained XGBoost
- `nn_model_[agent].pth` - Trained Neural Network
- `model_comparison_[agent].csv` - Performance metrics per model

### Criterion 2: Ensemble Stacking (30 pts)

**Required:**
1. Build stacking ensemble:
   - Level 0: RF + XGBoost + NN (base learners)
   - Level 1: Meta-model (Logistic Regression or XGBoost)
2. Train meta-model on out-of-fold predictions from base learners
3. Evaluate ensemble on test set:
   - Does ensemble AUC > individual models?
   - Target: >85% accuracy or >0.90 AUC
4. Cross-validation:
   - 5-fold CV to ensure robustness
   - Report mean ± std of metrics

**Deliverables:**
- `ensemble_model_[agent].pkl` - Stacking ensemble
- `ensemble_performance_[agent].csv` - CV results, test metrics
- `ensemble_roc_curve_[agent].png` - ROC curves for all models

### Criterion 3: SHAP Interpretability & Biomarker Panel (20 pts)

**Required:**
1. Compute SHAP values for each model:
   - TreeSHAP for RF and XGBoost
   - DeepSHAP for neural network
2. Aggregate SHAP values:
   - Mean absolute SHAP across models
   - Rank proteins by importance
3. Select biomarker panel:
   - Top 5-10 proteins by consensus SHAP
   - Retrain ensemble using ONLY these proteins
   - Validate: Does reduced panel maintain >80% of full-model performance?
4. Visualize SHAP:
   - Summary plot: SHAP values for top 20 proteins
   - Dependence plots: SHAP vs protein expression

**Deliverables:**
- `shap_values_rf_[agent].npy` - SHAP values from RF
- `shap_values_xgboost_[agent].npy` - SHAP from XGBoost
- `biomarker_panel_[agent].csv` - Final 5-10 proteins with SHAP scores
- `shap_summary_plot_[agent].png`
- `reduced_panel_performance_[agent].csv` - Panel validation

### Criterion 4: Clinical Translation (10 pts)

**Required:**
1. Biomarker feasibility:
   - Are selected proteins measurable in blood?
   - Check: Serpins, coagulation factors (YES), Collagens (tissue-specific, harder)
2. Therapeutic implications:
   - Can biomarker panel guide interventions?
   - Example: High serpin dysregulation → Anti-inflammatory therapy
3. Comparison with published aging clocks:
   - Do our ML-selected proteins overlap with Horvath clock, GrimAge?

**Deliverables:**
- `biomarker_feasibility_[agent].md` - Clinical assessment
- `therapeutic_strategy_[agent].md` - Intervention guide based on panel

## Required Artifacts

1. **01_plan_[agent].md**
2. **analysis_ensemble_[agent].py** - Full ML pipeline
3. **rf_model_[agent].pkl**, **xgboost_model_[agent].pkl**, **nn_model_[agent].pth**
4. **ensemble_model_[agent].pkl**
5. **biomarker_panel_[agent].csv**
6. **visualizations_[agent]/**:
   - model_comparison_bar_[agent].png
   - ensemble_roc_curve_[agent].png
   - shap_summary_plot_[agent].png
   - shap_dependence_plots_[agent]/ (folder with individual protein plots)
7. **90_results_[agent].md** - Knowledge Framework format

## ML Implementation Template

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import shap
import torch
import torch.nn as nn

# 1. Load data
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# 2. Prepare features: Proteins × Tissues → Tissue aging velocity
# (Example: Predict if tissue ages fast based on protein profile)
pivot = df.pivot_table(values='Zscore_Delta', index='Tissue', columns='Gene_Symbol')
X = pivot.fillna(0).values  # Features: protein expression per tissue
y = (X.mean(axis=1) > X.mean(axis=1).median()).astype(int)  # Binary: fast vs slow aging

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"RF AUC: {rf_auc:.3f}")

# 4. XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_pred)
print(f"XGBoost AUC: {xgb_auc:.3f}")

# 5. Neural Network
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

nn_model = MLPClassifier(input_dim=X_train.shape[1])
# ... training loop ...

# 6. SHAP
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_test)

explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# Aggregate SHAP
mean_shap = (np.abs(shap_values_rf).mean(axis=0) + np.abs(shap_values_xgb).mean(axis=0)) / 2
top_proteins_idx = np.argsort(mean_shap)[-10:]  # Top 10
biomarker_panel = pivot.columns[top_proteins_idx].tolist()

# 7. Reduced panel validation
X_train_reduced = X_train[:, top_proteins_idx]
X_test_reduced = X_test[:, top_proteins_idx]

rf_reduced = RandomForestClassifier(n_estimators=200, random_state=42)
rf_reduced.fit(X_train_reduced, y_train)
rf_reduced_auc = roc_auc_score(y_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
print(f"Reduced Panel AUC: {rf_reduced_auc:.3f} (vs Full: {rf_auc:.3f})")

# Save
import joblib
joblib.dump(rf, 'rf_model_[agent].pkl')
joblib.dump(xgb_model, 'xgboost_model_[agent].pkl')
pd.DataFrame({'Protein': biomarker_panel, 'SHAP': mean_shap[top_proteins_idx]}).to_csv('biomarker_panel_[agent].csv')
```

## Documentation Standards

Follow Knowledge Framework.

Reference: `/Users/Kravtsovd/projects/ecm-atlas/ADVANCED_ML_REQUIREMENTS.md`

## Expected Results

- Ensemble AUC > 0.90 (beats individual models)
- Biomarker panel: 5-10 proteins (serpins, fibrinogen, TIMP3 expected)
- Reduced panel maintains >85% performance
- SHAP reveals protein synergies

---

**Hypothesis ID:** H06
**Iteration:** 02
**Predicted Scores:** Novelty 9/10, Impact 10/10
**ML Focus:** ✅ Ensemble Learning, SHAP, Biomarker Discovery
