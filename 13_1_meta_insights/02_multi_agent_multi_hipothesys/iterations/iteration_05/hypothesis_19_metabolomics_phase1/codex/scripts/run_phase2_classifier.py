#!/usr/bin/env python3
"""Logistic regression for Phase II risk prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_19_metabolomics_phase1/codex')
ANALYSES_DIR = ROOT / 'analyses_codex'

FEATURES = ['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate', 'Lactate/Pyruvate',
            'COL1A1', 'COL3A1', 'COL5A1', 'FN1', 'ELN', 'LOX', 'TGM2', 'MMP2', 'MMP9', 'PLOD1', 'COL4A1', 'LAMC1']


def main() -> None:
    data = pd.read_csv(ANALYSES_DIR / 'multiomics_samples_codex.csv')
    data = data[data['is_control'] == False].reset_index(drop=True)
    X = data[FEATURES].fillna(0.0)
    y = (data['phase'] == 'Phase II').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)

    results = {
        'accuracy': accuracy,
        'auc': auc,
        'precision_phase2': report['1']['precision'],
        'recall_phase2': report['1']['recall'],
        'f1_phase2': report['1']['f1-score'],
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    output_path = ANALYSES_DIR / 'phase2_risk_prediction_codex.csv'
    pd.DataFrame([results]).to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
