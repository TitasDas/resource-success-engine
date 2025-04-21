import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import umap
import joblib
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.features import build_feature_view
from src.config import MODEL_PATH, LOGISTIC_MODEL_PATH
import os


def plot_interactive_scatter(df):
    numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()
    numeric_df["is_successful"] = df["is_successful"]

    fig = px.scatter_matrix(
        numeric_df,
        dimensions=[col for col in numeric_df.columns if col != "is_successful"],
        color="is_successful",
        title="Interactive Scatter Matrix of Project Features",
        labels={col: col.replace("_", " ") for col in numeric_df.columns}
    )
    fig.update_traces(diagonal_visible=False)
    fig.write_html("interactive_scatter_matrix.html")
    print("Interactive scatter matrix saved to interactive_scatter_matrix.html")


def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved to correlation_heatmap.png")
    plt.close()


def plot_boxplots(df):
    features_to_plot = ["planned_hours_total", "cost_variance_pct", "overtime_ratio", "hours_logged_per_allocated"]
    for feat in features_to_plot:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="is_successful", y=feat, data=df)
        plt.title(f"Distribution of {feat} by Success")
        plt.tight_layout()
        plt.savefig(f"boxplot_{feat}.png")
        print(f"Boxplot saved to boxplot_{feat}.png")
        plt.close()


def plot_umap(df):
    features = df.drop(columns=["is_successful", "status", "start_date", "end_date"], errors="ignore")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    umap_df["is_successful"] = df["is_successful"].values

    fig = px.scatter(
        umap_df,
        x="UMAP1", y="UMAP2",
        color="is_successful",
        title="UMAP Projection of Projects"
    )
    fig.write_html("umap_projection.html")
    print("UMAP projection saved to umap_projection.html")


def plot_feature_importance():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train the model first using make run.")
        return

    model = joblib.load(MODEL_PATH)
    importances = model.feature_importances_
    features = model.feature_name_

    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
    imp_df = imp_df.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=imp_df)
    plt.title("Feature Importance (LightGBM)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot saved to feature_importance.png")
    plt.close()


def plot_logistic_coefficients():
    if not os.path.exists(LOGISTIC_MODEL_PATH):
        print("Logistic regression model not found. Train it first using make run or an appropriate command.")
        return

    model = joblib.load(LOGISTIC_MODEL_PATH)
    coef = model.coef_.flatten()
    features = model.feature_names_in_

    coef_df = pd.DataFrame({"Feature": features, "Coefficient": coef})
    coef_df = coef_df.sort_values(by="Coefficient", key=abs, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df)
    plt.title("Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig("logistic_coefficients.png")
    print("Logistic regression coefficient plot saved to logistic_coefficients.png")
    plt.close()


if __name__ == "__main__":
    df = build_feature_view()
    print("Running visualizations on project data...")
    plot_interactive_scatter(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)
    plot_umap(df)
    plot_feature_importance()
    print("All visualizations complete.")