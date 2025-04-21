import argparse
import logging
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from src.features import build_feature_view
from src.config import MODEL_PATH, LOGISTIC_MODEL_PATH, TUNED_MODEL_PATH, AUTO_MODEL_PATH, DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model_type="lightgbm"):
    logger.info(f"Evaluating {model_type} model on original data...")
    model_path = {
        "logistic": LOGISTIC_MODEL_PATH,
        "tuned": TUNED_MODEL_PATH,
        "auto": AUTO_MODEL_PATH,
        "lightgbm": MODEL_PATH
    }.get(model_type, MODEL_PATH)

    model = joblib.load(model_path)
    X, y, _ = build_feature_view(return_shap_ready=True, override_db_path=DB_PATH)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], va='center', ha='center')
    plt.title(f"Confusion Matrix - {model_type}")
    plt.savefig(PLOT_DIR / f"confusion_{model_type}.png")
    plt.close()

    # ROC curve
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f"ROC Curve - {model_type}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(PLOT_DIR / f"roc_{model_type}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logistic", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.logistic:
        evaluate("logistic")
    elif args.tuned:
        evaluate("tuned")
    elif args.auto:
        evaluate("auto")
    else:
        evaluate()