# Ranks features by SHAP importance → retrains model with top-K → evaluates performance.

import shap
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
from src.features import build_feature_view

def train_and_score(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    model = LGBMClassifier(class_weight='balanced', n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(classification_report(y_test, preds))
    return model, acc

def run_rfe_shap(top_k=10):
    X, y, _ = build_feature_view(return_shap_ready=True)
    print(f"[RFE] Original shape: {X.shape}")

    model, _ = train_and_score(X, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X).values

    shap_mean = np.abs(shap_values).mean(axis=0)
    top_k_idx = np.argsort(shap_mean)[-top_k:]
    top_k_features = X.columns[top_k_idx]

    print(f"[RFE] Top {top_k} features based on SHAP:")
    for f in top_k_features:
        print(f"- {f}")

    # Retrain on top-k features
    X_reduced = X[top_k_features]
    print(f"\n[RFE] Retraining on {top_k} features...")
    _, acc = train_and_score(X_reduced, y)
    print(f"[RFE] Accuracy after RFE: {acc:.4f}")

if __name__ == "__main__":
    run_rfe_shap(top_k=10)
