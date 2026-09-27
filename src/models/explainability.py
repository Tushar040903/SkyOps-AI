"""
Model Explainability and Feature Attribution Module using SHAP (SHapley Additive exPlanations).
Performs TreeSHAP analysis to:
1. Identify global operational drivers of flight complexity.
2. Validate data-driven feature attributions against heuristic hand-coded weights.
3. Generate local flight-level waterfall explanations for frontline dispatchers.
"""

from typing import Dict, Any, List
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.config import (
    ENRICHED_COMPLEXITY_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
)
from src.utils import get_logger, set_plot_style

logger = get_logger(__name__)

def run_shap_analysis():
    """
    Execute full SHAP explainability pipeline on best complexity classification model.
    """
    logger.info("Initializing SHAP explainability and feature importance analysis...")
    set_plot_style()

    model_artifact_path = MODELS_DIR / "best_complexity_classifier.joblib"
    if not model_artifact_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at {model_artifact_path}")

    artifact = joblib.load(model_artifact_path)
    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    model_name = artifact["model_name"]
    features_meta = artifact["features"]

    # Extract pipeline stages
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Load master dataset
    df = pd.read_csv(ENRICHED_COMPLEXITY_PATH)
    num_cols = features_meta["numeric"]
    cat_cols = features_meta["categorical"]

    X = df[num_cols + cat_cols].copy()
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    # Transform features
    X_transformed = preprocessor.transform(X)

    # Retrieve feature names from ColumnTransformer
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names

    logger.info(f"Total features after one-hot encoding: {len(all_feature_names)}")

    # Sample for SHAP computation efficiency (e.g., 1000 representative flights)
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample_trans = X_transformed[sample_indices]
    X_sample_df = pd.DataFrame(X_sample_trans, columns=all_feature_names)

    logger.info("Computing TreeSHAP values...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_sample_df)

    # 1. Global Feature Importance Analysis
    # Depending on multiclass / binary, shap_values.values has shape (N, D, C) or (N, D)
    if len(shap_values.shape) == 3:
        # Multi-class: compute mean absolute SHAP value across all classes
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=(0, 2))
        # High complexity class is index 2 ('Hard')
        hard_class_idx = classes.index("Hard") if "Hard" in classes else 2
        shap_values_hard = shap_values[:, :, hard_class_idx]
    else:
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        shap_values_hard = shap_values

    importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap,
    }).sort_values("Mean_Absolute_SHAP", ascending=False)

    shap_csv_path = REPORTS_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(shap_csv_path, index=False)
    logger.info(f"Saved SHAP feature importances to: {shap_csv_path}")

    # 2. Plot Global SHAP Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    top20 = importance_df.head(20)
    sns.barplot(data=top20, x="Mean_Absolute_SHAP", y="Feature", palette="Blues_r", ax=ax)
    ax.set_title(f"Top 20 Operational Drivers of Flight Complexity\n(TreeSHAP Attribution - {model_name})", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Complexity)", labelpad=8)
    ax.set_ylabel("Operational Feature", labelpad=8)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "shap_feature_importance_bar.png"
    fig.savefig(bar_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved SHAP bar plot to {bar_path}")

    # 3. Plot SHAP Beeswarm Summary for High-Complexity ('Hard' Class)
    plt.figure(figsize=(11, 8), dpi=300)
    shap.plots.beeswarm(shap_values_hard, max_display=18, show=False)
    plt.title("TreeSHAP Beeswarm: Impact of Operational Factors on High Complexity", pad=14, fontsize=13, weight="bold")
    plt.tight_layout()
    beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP beeswarm plot to {beeswarm_path}")

    # 4. Data-Driven vs Heuristic Weight Validation Analysis
    # Group features into the 3 core dimensions: Delay, Passenger, Baggage
    delay_keywords = ["delay", "buffer", "ground_time", "pressure", "turnaround"]
    passenger_keywords = ["pax", "child", "basic_economy", "stroller", "ssr", "seats", "load_factor"]
    baggage_keywords = ["bag", "transfer"]

    def categorize_feature(fname: str) -> str:
        f_lower = fname.lower()
        if any(k in f_lower for k in baggage_keywords):
            return "Baggage"
        elif any(k in f_lower for k in passenger_keywords):
            return "Passenger & SSR"
        elif any(k in f_lower for k in delay_keywords):
            return "Delay & Turnaround"
        return "Network & Schedule"

    importance_df["Category"] = importance_df["Feature"].apply(categorize_feature)
    category_importance = importance_df.groupby("Category")["Mean_Absolute_SHAP"].sum()
    # Normalize to 100%
    data_driven_weights = (category_importance / category_importance.sum()) * 100

    comparison_df = pd.DataFrame({
        "Dimension": ["Delay & Turnaround", "Passenger & SSR", "Baggage"],
        "Heuristic Hand-Coded Weight (%)": [25.0, 50.0, 25.0],
        "SHAP Data-Driven Weight (%)": [
            round(data_driven_weights.get("Delay & Turnaround", 0), 1),
            round(data_driven_weights.get("Passenger & SSR", 0), 1),
            round(data_driven_weights.get("Baggage", 0), 1),
        ],
    })

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width/2, comparison_df["Heuristic Hand-Coded Weight (%)"], width, label="Heuristic Hand-Picked", color="#7FB3D5")
    ax.bar(x + width/2, comparison_df["SHAP Data-Driven Weight (%)"], width, label="Data-Driven SHAP Learned", color="#1A5276")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Dimension"], fontweight="bold")
    ax.set_ylabel("Relative Operational Weight (%)", fontweight="bold")
    ax.set_title("Weight Validation: Heuristic Formula vs Data-Driven Learned Importance", pad=12)
    ax.legend(frameon=True)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height + 0.8),
                        ha="center", va="baseline", fontsize=10, weight="bold")
    ax.set_ylim(0, max(comparison_df["SHAP Data-Driven Weight (%)"].max(), 55) + 8)
    plt.tight_layout()
    weights_path = FIGURES_DIR / "data_driven_vs_heuristic_weights.png"
    fig.savefig(weights_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved weights comparison plot to {weights_path}")

    # 5. Local Waterfall Case Studies for Operational Dispatch
    for label, target_class in [("easy", "Easy"), ("medium", "Medium"), ("hard", "Hard")]:
        match_indices = np.where(df.iloc[sample_indices]["complexity_class"].values == target_class)[0]
        if len(match_indices) > 0:
            idx = match_indices[0]
            orig_row = df.iloc[sample_indices[idx]]
            flight_desc = f"{orig_row.get('company_id', 'UA')} {orig_row.get('flight_number', '123')} ({orig_row.get('route', 'ORD-LHR')})"

            fig = plt.figure(figsize=(10, 6), dpi=300)
            class_idx = classes.index(target_class) if target_class in classes else 0
            if len(shap_values.shape) == 3:
                flight_shap = shap_values[idx, :, class_idx]
            else:
                flight_shap = shap_values[idx]

            shap.plots.waterfall(flight_shap, max_display=12, show=False)
            plt.title(f"Operational Case Study: {target_class} Flight\n{flight_desc}", pad=14, fontsize=12, weight="bold")
            plt.tight_layout()
            case_path = FIGURES_DIR / f"shap_waterfall_{label}.png"
            plt.savefig(case_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved local waterfall plot for {label} flight to {case_path}")

    print("\n" + "=" * 60)
    print("WEIGHT VALIDATION SUMMARY (Heuristic vs. Data-Driven SHAP):")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("\nTOP 10 OPERATIONAL DRIVERS (SHAP):")
    print(importance_df.head(10).to_string(index=False))

    return importance_df, comparison_df

if __name__ == "__main__":
    run_shap_analysis()
