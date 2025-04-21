# Generates a SHAP summary plot for the trained model

import joblib
import shap
import pandas as pd
import logging
from src.features import build_feature_view
from src.config import MODEL_PATH, LOGISTIC_MODEL_PATH, TUNED_MODEL_PATH, AUTO_MODEL_PATH

logging.basicConfig(level=logging.INFO)

def explain_model(use_logistic=False, use_tuned=False, use_auto=False):
    """
    Generates a SHAP summary plot for the trained model.
    Supports LightGBM, Tuned LightGBM, Logistic Regression, and Auto model.
    """
    logging.info("Loading model and features for SHAP explanation...")

    if use_logistic:
        model_path = LOGISTIC_MODEL_PATH
    elif use_tuned:
        model_path = TUNED_MODEL_PATH
    elif use_auto:
        model_path = AUTO_MODEL_PATH
    else:
        model_path = MODEL_PATH

    model = joblib.load(model_path)
    X, y, _ = build_feature_view(return_shap_ready=True)

    model_name = model.__class__.__name__
    logging.info(f"Using model: {model_name}")

    if model_name == "LGBMClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError(f"Unsupported model type for SHAP: {model_name}")

    logging.info("SHAP explanation ready. Launching summary plot...")
    shap.summary_plot(shap_values, X, show=True)
    logging.info("[EXPLAIN] SHAP explanation created.")

if __name__ == "__main__":
    explain_model()