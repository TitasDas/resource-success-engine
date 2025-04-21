# CLI tool to estimate success probability for a given project

import argparse
import joblib
import numpy as np
import pandas as pd
import shap
import logging
from src.features import build_feature_view
from src.config import MODEL_PATH, LOGISTIC_MODEL_PATH

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

def suggest_for_project(project_id: int, use_logistic=False):
    """
    Estimate the success probability for a specific project and show top contributing factors.
    Works for both LightGBM and LogisticRegression models.
    """
    logging.info(f"Loading model from: {model_path}")

    X, y, df = build_feature_view(return_shap_ready=True)

    model_path = LOGISTIC_MODEL_PATH if use_logistic else MODEL_PATH
    model = joblib.load(model_path)

    if project_id not in df.index:
        raise ValueError("Project ID not found")

    input_df = X.loc[[project_id]]

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    pred_label = "SUCCESS" if pred == 1 else "FAILURE"

    print(f"\n--- Prediction Summary ---")
    print(f"Project ID: {project_id}")
    print(f"Predicted Class: {pred_label}")
    print(f"Estimated Success Probability: {proba:.2%}")

    # Get SHAP values
    model_name = model.__class__.__name__

    if model_name == "LGBMClassifier":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError(f"Unsupported model type for SHAP: {model_name}")

    local_shap = shap_values[project_id]
    abs_sorted_indices = np.argsort(np.abs(local_shap))[::-1]

    print(f"\n--- Top 10 Factors Contributing to This Prediction ---")
    for idx in abs_sorted_indices[:10]:
        feature = X.columns[idx]
        value = X.loc[project_id, feature]
        impact = local_shap[idx]
        effect = "increases" if impact > 0 else "decreases"

        group = next((k for k, v in FEATURE_GROUPS.items() if feature in v), "Other")
        label = "# Derived" if feature not in df.columns else "# Core"

        print(f"{feature:30} | {value:.3f} | {effect:>9} likelihood of success | Group: {group:<11} {label} | Impact: {impact:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, required=True)
    parser.add_argument("--logistic", action="store_true", help="Use logistic regression model")
    args = parser.parse_args()

    suggest_for_project(project_id=args.project_id, use_logistic=args.logistic)
# CLI tool to estimate success probability for a given project

import argparse
import joblib
import numpy as np
import pandas as pd
import shap
from src.features import build_feature_view
from src.config import MODEL_PATH, LOGISTIC_MODEL_PATH

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

def suggest_for_project(project_id: int, use_logistic=False):
    """
    Estimate the success probability for a specific project and show top contributing factors.
    Works for both LightGBM and LogisticRegression models.
    """
    X, y, df = build_feature_view(return_shap_ready=True)

    model_path = LOGISTIC_MODEL_PATH if use_logistic else MODEL_PATH
    model = joblib.load(model_path)

    if project_id not in df.index:
        raise ValueError("Project ID not found")

    input_df = X.loc[[project_id]]

    proba = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    pred_label = "SUCCESS" if pred == 1 else "FAILURE"

    print(f"\n--- Prediction Summary ---")
    print(f"Project ID: {project_id}")
    print(f"Predicted Class: {pred_label}")
    print(f"Estimated Success Probability: {proba:.2%}")

    # Get SHAP values
    model_name = model.__class__.__name__

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

    local_shap = shap_values[project_id]
    abs_sorted_indices = np.argsort(np.abs(local_shap))[::-1]

    print(f"\n--- Top 10 Factors Contributing to This Prediction ---")
    for idx in abs_sorted_indices[:10]:
        feature = X.columns[idx]
        value = X.loc[project_id, feature]
        impact = local_shap[idx]
        effect = "increases" if impact > 0 else "decreases"

        group = next((k for k, v in FEATURE_GROUPS.items() if feature in v), "Other")
        label = "# Derived" if feature not in df.columns else "# Core"

        print(f"{feature:30} | {value:.3f} | {effect:>9} likelihood of success | Group: {group:<11} {label} | Impact: {impact:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=int, required=True)
    parser.add_argument("--logistic", action="store_true", help="Use logistic regression model")
    args = parser.parse_args()

    suggest_for_project(project_id=args.project_id, use_logistic=args.logistic)