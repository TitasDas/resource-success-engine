# Ensure SHAP explanation runs without error
# Confirm SHAP values match feature dimensions
# Validate top features list

import shap
import joblib
import numpy as np
from src.features import build_feature_view
from src.config import MODEL_PATH


def test_shap_output_shape():
    model = joblib.load(MODEL_PATH)
    X, y, _ = build_feature_view(return_shap_ready=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    assert shap_values.shape == X.shape, "SHAP output shape mismatch"


def test_top_shap_features_are_named():
    model = joblib.load(MODEL_PATH)
    X, y, _ = build_feature_view(return_shap_ready=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:10]
    top_feature_names = X.columns[top_indices]
    assert all(isinstance(name, str) for name in top_feature_names), "Top feature names not valid"
