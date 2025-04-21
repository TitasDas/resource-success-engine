# Evaluation script for synthetic data for all the different models along with SHAP plots

import argparse
import logging
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import sys

from src.features import build_feature_view
from src.config import MODEL_PATH, LOGISTIC_MODEL_PATH, TUNED_MODEL_PATH, AUTO_MODEL_PATH, SYN_DB_PATH

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def evaluate_model_on_synthetic(model_type="lightgbm"):
    """
    Evaluate a trained model on synthetic data and generate SHAP explanations.
    
    Args:
        model_type: Type of model to evaluate ("lightgbm", "logistic", "tuned", "auto")
    """
    try:
        logger.info(f"Evaluating {model_type} model on synthetic dataset...")

        # Select model path
        if model_type == "logistic":
            model_path = LOGISTIC_MODEL_PATH
        elif model_type == "tuned":
            model_path = TUNED_MODEL_PATH
        elif model_type == "auto":
            model_path = AUTO_MODEL_PATH
        else:
            model_path = MODEL_PATH  # default lightgbm

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info(f"Loading model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(...)
        model = joblib.load(model_path)

        # Load synthetic data from SQLite
        logger.info(f"Loading synthetic data from {SYN_DB_PATH}")
        X, y, _ = build_feature_view(return_shap_ready=True, override_db_path=SYN_DB_PATH)

        # Predict and evaluate
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        logger.info(f"Accuracy on synthetic data: {acc:.4f}")
        print(classification_report(y, y_pred))

        # SHAP summary plot
        model_name = model.__class__.__name__
        logger.info(f"Generating SHAP explanation for: {model_name}")

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

        shap.summary_plot(shap_values, X, show=False)
        plot_path = PROJECT_ROOT / "docs" / f"shap_summary_{model_type}.png"
        plt.savefig(plot_path)
        logger.info(f"SHAP summary plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error in synthetic evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logistic", action="store_true", help="Evaluate logistic regression model")
    parser.add_argument("--tuned", action="store_true", help="Evaluate tuned LightGBM model")
    parser.add_argument("--auto", action="store_true", help="Evaluate auto-selected stacked model")
    args = parser.parse_args()

    try:
        if args.logistic:
            evaluate_model_on_synthetic(model_type="logistic")
        elif args.tuned:
            evaluate_model_on_synthetic(model_type="tuned")
        elif args.auto:
            evaluate_model_on_synthetic(model_type="auto")
        else:
            evaluate_model_on_synthetic(model_type="lightgbm")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        sys.exit(1)