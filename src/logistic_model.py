import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from src.features import build_feature_view
from src.config import LOGISTIC_MODEL_PATH

logging.basicConfig(level=logging.INFO)

FEATURE_GROUPS = {
    "Cost": [
        "budget", "avg_rate", "actual_cost", "planned_cost",
        "cost_variance_pct", "log_vs_cost_ratio",
        "cost_per_task", "avg_cost_per_person", "is_overbudget"
    ],
    "Effort": [
        "hours_logged_per_allocated", "logged_hours_total", "planned_hours_total",
        "overtime_ratio", "overtime_hours", "overtime_efficiency", "is_underlogged"
    ],
    "PM": [
        "pm_success_rate", "pm_completed_projects", "pm_is_full_time",
        "pm_success_weighted", "is_high_variance_pm"
    ],
    "Planning": [
        "project_duration_days", "avg_allocation_lag_days", "active_days",
        "duration_per_person"
    ],
    "Allocation": [
        "allocation_std_per_person", "allocation_spikiness"
    ],
    "Complexity": [
        "dropoff_rate", "tasks_per_day", "task_density",
        "tasks_per_person", "is_high_complexity"
    ]
}

def get_feature_metadata(feature):
    for group, features in FEATURE_GROUPS.items():
        if feature in features:
            return group, "Core"
    return "Other", "Derived"

def plot_coefficients(model, feature_names):
    coefs = model.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[::-1][:10]
    top_features = [(feature_names[i], coefs[i]) for i in top_idx]

    labels = []
    values = []
    for feat, weight in top_features:
        group, origin = get_feature_metadata(feat)
        label = f"{feat} ({group}, {origin})"
        labels.append(label)
        values.append(weight)

    plt.figure(figsize=(10, 6))
    colors = ['green' if w > 0 else 'red' for w in values]
    plt.barh(labels[::-1], values[::-1], color=colors[::-1])
    plt.xlabel("Coefficient Weight")
    plt.title("Top 10 Influential Logistic Regression Features")
    plt.tight_layout()
    plt.savefig("logistic_coefficients.png")
    logging.info("Coefficient plot saved to logistic_coefficients.png")

def train_logistic_model(override_db_path=None):
    logging.info("[LOGISTIC] Training Logistic Regression on dataset...")
    df = build_feature_view(override_db_path=override_db_path)
    X = df.drop(columns=["is_successful", "status", "start_date", "end_date"])
    y = df["is_successful"]

    model = LogisticRegression(penalty="l2", solver="liblinear", C=1.0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    logging.info(f"Cross-validated accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    for i, score in enumerate(scores, 1):
        logging.info(f"Fold {i}: {score:.4f}")

    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    logging.info(f"[LOGISTIC] Accuracy on full data: {acc:.4f}")
    print("\nClassification Report (Full Data):")
    print(classification_report(y, y_pred))

    plot_coefficients(model, X.columns)
    joblib.dump(model, LOGISTIC_MODEL_PATH)
    logging.info(f"[LOGISTIC] Model saved to {LOGISTIC_MODEL_PATH}")
    print(" [LOGISTIC] Model training complete.")

if __name__ == "__main__":
    train_logistic_model()