# main orchestrator

import argparse
import logging
from src.lightgbm_model import train_model
from src.tuned_model import tune_model
from src.auto_model import auto_select_best_model
from src.explain import explain_model
from src.suggest import suggest_for_project
from src.logistic_model import train_logistic_model
from src.visualize_data import plot_logistic_coefficients
from src.features import build_feature_view
from src.config import SYN_DB_PATH

logging.basicConfig(level=logging.INFO)


def main(use_tuned=False, use_stacked=False, use_logistic=False, use_synthetic=False):
    train_db_path = SYN_DB_PATH if use_synthetic else None

    # Step 1: Train model
    print("\nStep 1: Training the model...")
    if use_stacked:
        print("\nStep 1: Training the stacked model...")
        auto_select_best_model(override_db_path=train_db_path)
    elif use_logistic:
        print("\nStep 1: Training the logistic model...")
        train_logistic_model(override_db_path=train_db_path)
        print("\nStep 1B: Generating logistic regression coefficient plot...")
        plot_logistic_coefficients()
    elif use_tuned:
        print("\nStep 1: Training the tuned lightgbm model...")
        tune_model(override_db_path=train_db_path)
    else:
        print("\nStep 1: Training the lightgbm model...")
        train_model(override_db_path=train_db_path)
    print("Finished model training.")

    # Step 2: SHAP explanation
    print("\nStep 2: Generating SHAP summary explanation...")
    if use_logistic:
        explain_model(use_logistic=True)
    else:
        explain_model()
    print("Finished SHAP explanation.")

    # Step 3: Suggest outcome for a project (always using original DB)
    print("\nStep 3: Predicting success probability for selected project...")
    _, _, df = build_feature_view(return_shap_ready=True)  # no override
    available_ids = df.index.tolist()

    print(f"Available Project IDs: {available_ids}")
    project_id = int(input("Enter a project ID to evaluate: "))
    if project_id not in available_ids:
        raise ValueError(f"Project ID {project_id} not found in original DB.")

    suggest_for_project(project_id=project_id, use_logistic=use_logistic)
    print("Finished project prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, explain, and predict project success")
    parser.add_argument("--tuned", action="store_true", help="Use hyperparameter-tuned LightGBM")
    parser.add_argument("--logistic", action="store_true", help="Use logistic regression on full data")
    parser.add_argument("--stacked", action="store_true", help="Use stacked model (LGBM + KNN)")
    parser.add_argument("--synthetic", action="store_true", help="Train model on synthetic data")
    args = parser.parse_args()

    main(
        use_tuned=args.tuned,
        use_stacked=args.stacked,
        use_logistic=args.logistic,
        use_synthetic=args.synthetic,
    )
