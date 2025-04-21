# src/explain_interactions.py

import shap
import joblib
import pandas as pd
import argparse
import logging
from src.features import build_feature_view
from src.config import MODEL_PATH

logging.basicConfig(level=logging.INFO)

def main(feature_name: str, other_feature: str = None):
    print("[INTERACT] Loading model and features...")
    X, y, _ = build_feature_view(return_shap_ready=True)
    model = joblib.load(MODEL_PATH)

    print("[INTERACT] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if feature_name not in X.columns:
        raise ValueError(f"Feature '{feature_name}' not found in features.")

    print(f"[INTERACT] Plotting SHAP dependence plot for: {feature_name}")
    shap.dependence_plot(
        feature_name,
        shap_values,
        X,
        interaction_index=other_feature,
        show=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", required=True, help="Primary feature for dependence plot")
    parser.add_argument("--other", help="Optional second feature for interaction coloring")
    args = parser.parse_args()

    main(args.feature, args.other)