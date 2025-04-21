import shap
import joblib
import pandas as pd
from src.features import build_feature_view
from src.config import MODEL_PATH

def get_shap_explainer(model=None):
    if model is None:
        model = joblib.load(MODEL_PATH)
    df = build_feature_view()
    X = df.drop(columns=["is_successful", "status", "start_date", "end_date"])
    explainer = shap.Explainer(model, X)
    return explainer, X

def get_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values, pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
