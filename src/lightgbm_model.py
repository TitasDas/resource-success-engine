# Trains and evaluates a LightGBM model, saving the model to disk

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from src.features import build_feature_view
from src.config import MODEL_PATH

def train_model(override_db_path=None):
    """
    Train a LightGBM model on project features and save it to disk.
    """
    print(" [TRAIN] Model training started...")

    df = build_feature_view(override_db_path=override_db_path)

    # Drop non-feature columns
    X = df.drop(columns=["is_successful", "status", "start_date", "end_date"])
    y = df["is_successful"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    # Initialize and train model
    model = LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        min_data_in_leaf=1,
        min_data_in_bin=1,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print(" Accuracy:", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f" Model saved to {MODEL_PATH}")
    print(" [TRAIN] Model training complete.")

if __name__ == "__main__":
    train_model()
