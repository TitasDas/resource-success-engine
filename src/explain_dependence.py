# Visualize how a single feature influences the model’s output — across all data points.

import shap
import joblib
import argparse
from src.features import build_feature_view
from src.config import MODEL_PATH

def run_dependence_plot(feature: str):
    X, y, _ = build_feature_view(return_shap_ready=True)
    model = joblib.load(MODEL_PATH)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    print(f"[DEP] Plotting dependence for: {feature}")
    shap.dependence_plot(feature, shap_values.values, X, show=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=str, required=True)
    args = parser.parse_args()
    run_dependence_plot(args.feature)
