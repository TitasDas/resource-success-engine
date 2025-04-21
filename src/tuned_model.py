# Performs robust hyperparameter tuning for LightGBM using randomized search with cross-validation.

import joblib
import logging
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.features import build_feature_view
from src.config import TUNED_MODEL_PATH

logging.basicConfig(level=logging.INFO)

def tune_model(override_db_path=None):
    logging.info("[TUNE] Starting randomized hyperparameter tuning...")

    df = build_feature_view(override_db_path=override_db_path)
    X = df.drop(columns=["is_successful", "status", "start_date", "end_date"])
    y = df["is_successful"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_data_in_leaf": [1, 5, 10, 20],
        "min_child_samples": [5, 10, 20],
    }

    base_model = LGBMClassifier(
        class_weight="balanced",
        min_data_in_bin=1,
        random_state=42,
        verbose=-1
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_weighted",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"[TUNE] Best accuracy on test set: {acc:.4f}")
    logging.info(f"[TUNE] Best params: {search.best_params_}")
    print("\n Classification Report (Test Set):\n", classification_report(y_test, y_pred))

    # Cross-validation report
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="f1_weighted")
    print(f"\nCross-validated F1-weighted score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump(best_model, TUNED_MODEL_PATH)
    logging.info(f"[TUNE] Best model saved to {TUNED_MODEL_PATH}")

if __name__ == "__main__":
    tune_model()
