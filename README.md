# Project Success Predictor

This project predicts the likelihood of project success using resource allocation, team structure, and cost-performance indicators. It leverages interpretable machine learning techniques and provides tooling for explainability, tuning, and modeling experimentation.

## Overview

The solution includes:

- A baseline LightGBM classifier  
- Optional model tuning and stacking  
- SHAP-based global and local explanations  
- Feature interaction and dependence analysis  
- Feature selection utilities  
- Auto model evaluation and best-model selection  

## Setup

Clone the repository and install the requirements:

```bash
git clone https://github.com/julia-fulton/titas-das.git
cd float-project
pip install -r requirements.txt
```

## Running the Pipeline

This project supports multiple modes of execution depending on the model variant you wish to run. Use the following `make` commands:


### Baseline Execution

Runs the standard LightGBM model training and evaluation pipeline. Ensure you close the SHAP plot window while executing the run commands to continue the execution.

```bash
make run
```

This command will:

- Train a baseline LightGBM model  
- Generate SHAP global explanations (close the plot window to proceed during a run)
- Prompt the user to input a project ID  
- Predict the outcome and explain the top contributing features  

### Tuned LightGBM

Performs hyperparameter tuning on the LightGBM model before training and evaluation.

```bash
make run_tuned
```


## Core Commands

### Train the baseline model

```bash
make train
```

### Generate global SHAP explanation

```bash
make explain
```

### Predict for a specific project

```bash
make suggest
```

### Run all unit tests

```bash
make test
```

## Explainability Utilities

### Generate SHAP dependence plot

```bash
make dependence FEATURE=feature_name
```

Plots how model output changes with a specific feature value.

### Generate SHAP interaction plot

```bash
make interactions FEATURE=feature_name
```

Plots pairwise interactions between features and their combined impact on the model.

## Model Enhancement

Run automated model selection (LightGBM, KNN, Stacked) and use the best one:
```bash
make auto
```

## Visualizations

This project includes a rich set of visual tools to explore and understand the dataset before modeling. Running the following command will automatically generate all visualizations:

```bash
make visualize
```

The outputs help in assessing feature distributions, relationships, and potential model signals.

### Generated Visuals

- **Interactive Scatter Matrix (Plotly)**  
  A dynamic pairwise scatter plot across all numeric features, colored by `is_successful`. Enables quick exploration of feature interactions.  
  Output: `interactive_scatter_matrix.html` (open in browser)

- **Correlation Heatmap**  
  Displays Pearson correlation coefficients between all numeric features to reveal linear dependencies and multicollinearity.

- **Boxplots of Key Features**  
  Shows how distributions of selected important features vary by project outcome (`is_successful`).

- **UMAP Projection**  
  Reduces high-dimensional feature space into a 2D representation for visual clustering and separation patterns.

- **LightGBM Feature Importance**  
  Highlights the top features influencing model predictions, based on the trained LightGBM model.

These plots aid in data sanity checks, model feature engineering, and communication of insights to stakeholders.


## Cleanup Artifacts

```bash
make clean
```

Deletes temporary `.pkl`, `.pyc`, and cached files. The `models/` directory is preserved.

## Directory Structure

```
.
├── docs/
├── models/
├── src/
|   ├── __init__.py
│   ├── config.py
│   ├── etl.py
│   ├── features.py
│   ├── model.py
│   ├── explain.py
│   ├── suggest.py
│   ├── auto_model.py
│   ├── explain_dependence.py
│   ├── explain_interactions.py
│   ├── explain_helper.py
│   ├── visualize_data.py
│   └── feature_select_rfe.py
|   └── README.md (Titas' explainer doc)
|    
├── run.py
├── makefile
├── requirements.txt
└── README.md (original readme given by Float)
```

## Future Directions and Model Strategy

This project compares multiple model configurations to ensure we always use the most performant model for predicting project success. Currently, the system supports:

- **Baseline LightGBM**: Fast, interpretable, and performs well on tabular data even with small datasets. This serves as our default model.
- **Auto Model Selection**: Uses the `auto_model.py` script to compare:
  - LightGBM
  - K-Nearest Neighbors (KNN)
  - A StackingClassifier that combines both with Logistic Regression as a meta-learner

Each model is trained and evaluated on the same data split using accuracy as the comparison metric. The best model is saved and used for downstream tasks like SHAP explanations and project recommendations.

### Why LightGBM Remains the Best

In evaluations so far, **LightGBM consistently outperforms both KNN and the stacked ensemble**, which is expected due to:

- The small number of projects (≈26), where simpler models generalize better
- LightGBM's native support for feature importance and class imbalance
- Limited gains from ensembling when the additional models (like KNN) contribute less predictive power

### When to Rerun Model Selection

We may rerun `make auto` or execute `src/auto_model.py` to reselect the best model if:

- We have more data
- New features are introduced
- We want to validate against alternative metrics like precision, recall, or AUC

The auto-selection logic ensures the system always picks the most accurate model from the set, and stores it in `models/lgbm_model.pkl` for consistent use across the pipeline.

## Caveats and Considerations

While the tuned model (`--tuned` flag) performs randomized hyperparameter optimization with cross-validation, it may not always outperform the baseline LightGBM model in test set accuracy due to the limited dataset size (26 projects).

Key points to note:

- The **baseline model** allows finer splits (`min_data_in_leaf=1`), which may overfit but perform better on small holdout sets.
- The **tuned model** shows higher *cross-validated F1 scores*, indicating better generalization, but may underperform on the specific test partition.
- This behavior is expected in low-data regimes where even small validation set changes introduce high variance in metrics.

The tuning and evaluation logic is preserved to ensure the pipeline is extensible and ready for more data in production environments. As the dataset grows, the benefit of tuning is expected to become more consistent and significant.



