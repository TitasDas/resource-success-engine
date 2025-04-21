# Trains multiple models (LightGBM, KNN, Stacked) and saves the best performing one.

import joblib
import logging
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.features import build_feature_view
from src.config import AUTO_MODEL_PATH

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, model

def auto_select_best_model(override_db_path=None):
    logging.info("Loading data for auto model selection...")

    X, y, _ = build_feature_view(return_shap_ready=True, override_db_path=override_db_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    candidates = {
        "lgb": LGBMClassifier(
            n_estimators=50, max_depth=3, min_data_in_leaf=1,
            min_data_in_bin=1, class_weight="balanced", random_state=42),

        "knn": KNeighborsClassifier(n_neighbors=3),

        "stacked": StackingClassifier(
            estimators=[
                ('lgb', LGBMClassifier(n_estimators=50, max_depth=3, min_data_in_leaf=1,
                                       min_data_in_bin=1, class_weight="balanced", random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=3))
            ],
            final_estimator=LogisticRegression(),
            cv=3, n_jobs=-1
        )
    }

    scores = {}
    best_model = None
    best_score = -1

    for name, model in candidates.items():
        logging.info(f"Evaluating model: {name}")
        acc, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test)
        scores[name] = acc
        logging.info(f"{name} accuracy: {acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_model = trained_model
            best_name = name

    joblib.dump(best_model, AUTO_MODEL_PATH)
    logging.info(f"Best model '{best_name}' with accuracy {best_score:.4f} saved to {AUTO_MODEL_PATH}")

if __name__ == "__main__":
    auto_select_best_model()
