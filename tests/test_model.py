# Model file is saved and can be loaded with .predict method
# Ensure model has some skill (acc > 0.5) and can learn
# Confirm .predict_proba() returns valid outputs

import joblib
import os
from src.config import MODEL_PATH
from src.lightgbm_model import train_model
from src.features import build_feature_view
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


def test_model_training_and_save():
    train_model()
    assert os.path.exists(MODEL_PATH), "Model file not found after training"
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Loaded model is not valid"

def test_model_can_learn():
    df = build_feature_view()
    X = df.drop(columns=["is_successful", "status", "start_date", "end_date"])
    y = df["is_successful"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=1
    )

    model = LGBMClassifier(
        class_weight="balanced",
        min_data_in_leaf=1,
        min_data_in_bin=1,
        max_depth=3,
        n_estimators=50,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.5, f" Model underperforming — accuracy: {acc:.2f}"

def test_prediction_probabilities():
    model = joblib.load(MODEL_PATH)
    df = build_feature_view()
    X = df.drop(columns=["is_successful", "status", "start_date", "end_date"])
    preds = model.predict_proba(X)
    assert preds.shape[1] == 2, "Model should output 2-class probabilities"
    assert (preds >= 0).all() and (preds <= 1).all(), "Probabilities are out of bounds"