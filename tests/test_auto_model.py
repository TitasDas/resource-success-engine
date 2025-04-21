# Tests that auto_model selects a valid model, saves it, and accuracy is reasonable

import joblib
import os
from src.config import MODEL_PATH
from src.auto_model import auto_select_best_model
from src.features import build_feature_view
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_auto_model_runs_and_saves():
    auto_select_best_model()
    assert os.path.exists(MODEL_PATH), "Auto-selected model was not saved."

def test_auto_model_predicts_with_reasonable_accuracy():
    model = joblib.load(MODEL_PATH)
    X, y, _ = build_feature_view(return_shap_ready=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc >= 0.5, f"Auto model underperformed — accuracy was {acc:.2f}"
