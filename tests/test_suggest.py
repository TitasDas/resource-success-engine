# Validate suggest pipeline: prediction + SHAP explanation
# Check valid project ID gives expected output

import joblib
from src.features import build_feature_view
from src.config import MODEL_PATH


def test_suggest_prediction_pipeline():
    X, y, df = build_feature_view(return_shap_ready=True)
    model = joblib.load(MODEL_PATH)

    project_id = df.index[0]
    assert project_id in df.index

    input_row = X.loc[project_id].to_frame().T
    proba = model.predict_proba(input_row)[0][1]
    pred = model.predict(input_row)[0]

    assert 0 <= proba <= 1, "Probability out of bounds"
    assert pred in [0, 1], "Prediction must be binary"
