# At least 1 row, >10 columns
# Critical columns like is_successful, cost_variance_pct exist
# No missing values in feature columns
# feature health tests
# a label test? : Confirm class diversity exists for learning

import pytest
from src.features import build_feature_view


def test_feature_view_shape():
    df = build_feature_view()
    assert df.shape[0] > 0, "Feature view has no rows"
    assert df.shape[1] > 10, "Feature view has too few columns"


def test_required_columns_exist():
    df = build_feature_view()
    required_cols = [
        "is_successful",
        "planned_hours_total",
        "logged_hours_total",
        "cost_variance_pct",
        "pm_success_rate"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


def test_no_nulls_in_features():
    df = build_feature_view()
    feature_cols = df.drop(columns=["status", "start_date", "end_date"], errors="ignore")
    assert not feature_cols.isnull().any().any(), "Nulls found in feature columns"

def test_feature_distributions_are_reasonable():
    df = build_feature_view()
    assert df["planned_hours_total"].max() < 1e5, "Planned hours too high"
    assert (df["cost_variance_pct"] > -10).all(), "Cost variance seems unreasonable"
    assert (df["overtime_ratio"] <= 5).all(), "Overtime ratio unusually high"

def test_success_label_variation():
    df = build_feature_view()
    unique_labels = df["is_successful"].nunique()
    assert unique_labels > 1, "All projects have same success label — no signal!"


