# Color codes
YELLOW=\033[1;33m
CYAN=\033[1;36m
RESET=\033[0m

.PHONY: help

help:
	@echo ""
	@echo "$(CYAN)Available Make Commands:$(RESET)"
	@echo "  $(YELLOW)install$(RESET)                 - Install Python dependencies"
	@echo "  $(YELLOW)train$(RESET)                   - Train LightGBM model"
	@echo "  $(YELLOW)explain$(RESET)                 - Generate SHAP summary for last trained model"
	@echo "  $(YELLOW)suggest$(RESET)                 - Predict success probability for project_id=1"
	@echo "  $(YELLOW)run$(RESET)                     - Run pipeline with default LightGBM model"
	@echo "  $(YELLOW)run_tuned$(RESET)               - Run pipeline using tuned LightGBM model"
	@echo "  $(YELLOW)run_logistic$(RESET)            - Run pipeline using logistic regression model"
	@echo "  $(YELLOW)auto$(RESET)                    - Run auto-model selection script"
	@echo "  $(YELLOW)test$(RESET)                    - Run all pytest tests"
	@echo "  $(YELLOW)rfe$(RESET)                     - Run recursive feature elimination"
	@echo "  $(YELLOW)dependence$(RESET)              - Plot SHAP dependence for selected feature"
	@echo "  $(YELLOW)visualize$(RESET)               - Run base visualizations on input data"
	@echo "  $(YELLOW)visualize_logistic$(RESET)      - Plot logistic regression coefficients"
	@echo "  $(YELLOW)interactions$(RESET)            - Run SHAP interaction analysis between features"
	@echo "  $(YELLOW)clean$(RESET)                   - Delete generated .pkl and .pyc files and cache"
	@echo "  $(YELLOW)evaluate_synthetic$(RESET)      - Evaluate LightGBM model on synthetic data"
	@echo "  $(YELLOW)evaluate_logistic_synthetic$(RESET) - Evaluate logistic regression on synthetic data"
	@echo "  $(YELLOW)evaluate_tuned_synthetic$(RESET)    - Evaluate tuned LightGBM on synthetic data"
	@echo "  $(YELLOW)evaluate_auto_synthetic$(RESET)     - Evaluate auto-stacked model on synthetic data"
	@echo "  $(YELLOW)run_synthetic$(RESET)           - Run pipeline using LightGBM model trained on synthetic data"
	@echo "  $(YELLOW)run_logistic_synthetic$(RESET)  - Run pipeline using logistic regression trained on synthetic data"
	@echo "  $(YELLOW)run_tuned_synthetic$(RESET)     - Run pipeline using tuned LightGBM on synthetic data"
	@echo "  $(YELLOW)run_stacked_synthetic$(RESET)   - Run pipeline using auto-stacked model on synthetic data"
	@echo "  $(YELLOW)train_synthetic$(RESET)             - Train LightGBM model on synthetic data"
	@echo "  $(YELLOW)train_synthetic_logistic$(RESET)    - Train logistic regression model on synthetic data"
	@echo "  $(YELLOW)train_synthetic_tuned$(RESET)       - Train tuned LightGBM model on synthetic data"
	@echo "  $(YELLOW)train_synthetic_auto$(RESET)        - Train auto-stacked model on synthetic data"
	@echo "  $(YELLOW)evaluate_original$(RESET)           - Evaluate LightGBM model on original data"
	@echo "  $(YELLOW)evaluate_original_logistic$(RESET)  - Evaluate logistic regression model on original data"
	@echo "  $(YELLOW)evaluate_original_tuned$(RESET)     - Evaluate tuned LightGBM model on original data"
	@echo "  $(YELLOW)evaluate_original_auto$(RESET)      - Evaluate auto-stacked model on original data"


	@echo ""

PYTHON=python
SRC_DIR=src
TEST_DIR=tests

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/model.py

explain:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/explain.py

suggest:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/suggest.py --project_id=1

run:
	PYTHONPATH=. $(PYTHON) run.py

run_tuned:
	PYTHONPATH=. $(PYTHON) run.py --tuned

run_logistic:
	PYTHONPATH=. $(PYTHON) run.py --logistic

run_synthetic:
	PYTHONPATH=. $(PYTHON) run.py --synthetic

run_logistic_synthetic:
	PYTHONPATH=. $(PYTHON) run.py --logistic --synthetic

run_tuned_synthetic:
	PYTHONPATH=. $(PYTHON) run.py --tuned --synthetic

run_stacked_synthetic:
	PYTHONPATH=. $(PYTHON) run.py --stacked --synthetic

auto:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/auto_model.py

test:
	PYTHONPATH=. pytest $(TEST_DIR)


rfe:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/feature_select_rfe.py

dependence:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/explain_dependence.py --feature=$(FEATURE)

visualize:
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/visualize_data.py

visualize_logistic:
	PYTHONPATH=. $(PYTHON) -c "from src.visualize_data import plot_logistic_coefficients; plot_logistic_coefficients()"

interactions:
ifeq ($(OTHER),)
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/explain_interactions.py --feature=$(FEATURE)
else
	PYTHONPATH=. $(PYTHON) $(SRC_DIR)/explain_interactions.py --feature=$(FEATURE) --other=$(OTHER)
endif

evaluate_synthetic:
	PYTHONPATH=. python src/evaluate_on_synthetic.py

evaluate_logistic_synthetic:
	PYTHONPATH=. python src/evaluate_on_synthetic.py --logistic

evaluate_tuned_synthetic:
	PYTHONPATH=. python src/evaluate_on_synthetic.py --tuned

evaluate_auto_synthetic:
	PYTHONPATH=. python src/evaluate_on_synthetic.py --auto

train_synthetic:
	PYTHONPATH=. python src/train_on_synthetic.py

train_synthetic_logistic:
	PYTHONPATH=. python src/train_on_synthetic.py --logistic

train_synthetic_tuned:
	PYTHONPATH=. python src/train_on_synthetic.py --tuned

train_synthetic_auto:
	PYTHONPATH=. python src/train_on_synthetic.py --auto

evaluate_original:
	PYTHONPATH=. python src/evaluate_on_original.py

evaluate_original_logistic:
	PYTHONPATH=. python src/evaluate_on_original.py --logistic

evaluate_original_tuned:
	PYTHONPATH=. python src/evaluate_on_original.py --tuned

evaluate_original_auto:
	PYTHONPATH=. python src/evaluate_on_original.py --auto


clean:
	find . -type f -name '*.pkl' -delete
	find . -type f -name '*.pyc' -delete
	rm -rf __pycache__ .pytest_cache
