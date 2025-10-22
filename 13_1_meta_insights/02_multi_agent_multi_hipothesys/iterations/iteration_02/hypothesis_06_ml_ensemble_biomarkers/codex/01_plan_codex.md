# Plan: Ensemble ML Biomarker Discovery (Agent: codex)

## 1. Data Understanding & Preparation
- Inspect `merged_ecm_aging_zscore.csv`, define tissue identifier, compute aging velocity metric (abs mean z-score delta) and binary labels via median split.
- Pivot tissue × protein matrix, handle missing values, split into train/validation/test and set random seeds.

## 2. Baseline Model Training & Evaluation
- Train Random Forest (n=200, max_depth=10, OOB) and log metrics.
- Train XGBoost with early stopping (learning_rate=0.1, max_depth=6, n_estimators=100) and record metrics.
- Implement PyTorch MLP (128-64-32, dropout+batchnorm) with validation early stopping, capture metrics.
- Save individual models and metrics table `model_comparison_codex.csv`.

## 3. Stacking Ensemble Construction
- Wrap base models for stacking with 5-fold CV to generate out-of-fold predictions.
- Train logistic regression meta-learner, evaluate on hold-out test set, compute ROC/AUC, export `ensemble_model_codex.pkl` and `ensemble_performance_codex.csv`.

## 4. Interpretability & Feature Selection
- Compute SHAP values: TreeSHAP (RF/XGBoost), DeepSHAP (NN) on validation/test split.
- Aggregate mean |SHAP| across models, run RFE (logistic) to cross-check feature list, pick consensus 5-10 protein panel.
- Save SHAP arrays, biomarker table, and plots (summary + dependence) under `visualizations_codex/`.

## 5. Reduced Panel Validation & Outputs
- Retrain ensemble using selected biomarkers, evaluate performance retention (>80%).
- Generate comparison bar chart, ensemble ROC, reduced panel metrics CSV, and finalize documentation (`90_results_codex.md`, feasibility & therapeutic notes).

## 6. Verification & Packaging
- Ensure all artifacts saved with `_codex` suffix, verify paths, outline suggested next steps.
