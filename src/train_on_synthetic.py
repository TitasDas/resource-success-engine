import argparse
import logging
from pathlib import Path
from src.features import build_feature_view
from src.lightgbm_model import train_model
from src.tuned_model import tune_model
from src.auto_model import auto_select_best_model
from src.logistic_model import train_logistic_model
from src.config import SYN_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(train_type="lightgbm"):
    logger.info(f"Training {train_type} model on synthetic data...")

    if train_type == "logistic":
        train_logistic_model(override_db_path=SYN_DB_PATH)
    elif train_type == "tuned":
        tune_model(override_db_path=SYN_DB_PATH)
    elif train_type == "auto":
        auto_select_best_model(override_db_path=SYN_DB_PATH)
    else:
        train_model(override_db_path=SYN_DB_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logistic", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.logistic:
        main("logistic")
    elif args.tuned:
        main("tuned")
    elif args.auto:
        main("auto")
    else:
        main()