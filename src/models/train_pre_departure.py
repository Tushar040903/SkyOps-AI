"""
Pre-Departure Operational Risk & Delay Prediction Pipeline.
Trains predictive machine learning models using ONLY pre-flight features (T-2 hours before departure).
Forecasts whether a flight will experience severe operational complexity or delay (>15 min),
enabling proactive dispatch and frontline ground handling interventions.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from src.config import (
    ENRICHED_COMPLEXITY_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    PRE_DEPARTURE_NUMERIC_FEATURES,
    PRE_DEPARTURE_CATEGORICAL_FEATURES,
    TARGET_DELAY_15MIN,
)
from src.utils import get_logger, evaluate_classification, plot_confusion_matrix, set_plot_style

logger = get_logger(__name__)

def prepare_pre_departure_data() -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Extract strict pre-departure feature space and binary operational delay target."""
    df = pd.read_csv(ENRICHED_COMPLEXITY_PATH)
    logger.info(f"Loaded master dataset for pre-departure modeling: {df.shape}")

    num_cols = [c for c in PRE_DEPARTURE_NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in PRE_DEPARTURE_CATEGORICAL_FEATURES if c in df.columns]

    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_DELAY_15MIN].copy().astype(int)

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    logger.info(f"Pre-departure features: {len(num_cols)} numeric, {len(cat_cols)} categorical.")
    logger.info(f"Target '{TARGET_DELAY_15MIN}' class balance: {y.value_counts(normalize=True).to_dict()}")
    return X, y, preprocessor

def run_pre_departure_benchmark() -> pd.DataFrame:
    """
    Train and benchmark models for pre-departure prediction using Stratified 5-Fold CV.
    Generates ROC/PR trade-off curves, threshold tuning table, and serializes best model.
    """
    logger.info("Initializing pre-departure delay & operational risk prediction...")
    X, y, preprocessor = prepare_pre_departure_data()
    classes = ["On-Time / Low Risk", "Delayed > 15m / High Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Compute scale_pos_weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0

    models = {
        "Logistic Regression (Pre-Flight)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest (Pre-Flight)": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM (Pre-Flight)": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost (Pre-Flight)": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    benchmark_rows = []
    trained_pipelines = {}
    test_probs = {}

    for name, model in models.items():
        logger.info(f"Cross-validating {name}...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        scoring = {
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc",
        }
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        test_probs[name] = y_prob

        test_eval = evaluate_classification(y_test, y_pred, y_prob, classes=classes, model_name=name)

        benchmark_rows.append({
            "Model": name,
            "CV_ROC_AUC": round(cv_res["test_roc_auc"].mean(), 4),
            "CV_F1": round(cv_res["test_f1"].mean(), 4),
            "CV_Recall": round(cv_res["test_recall"].mean(), 4),
            "Test_Accuracy": round(test_eval["accuracy"], 4),
            "Test_Precision": round(test_eval["precision_weighted"], 4),
            "Test_Recall": round(test_eval["recall_weighted"], 4),
            "Test_F1": round(test_eval["f1_weighted"], 4),
            "Test_ROC_AUC": round(test_eval["roc_auc"], 4),
        })

    results_df = pd.DataFrame(benchmark_rows).sort_values("Test_ROC_AUC", ascending=False)
    csv_path = REPORTS_DIR / "pre_departure_benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved pre-departure benchmark to: {csv_path}")

    # Best model
    best_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_name]
    logger.info(f"Top Pre-Departure Model: {best_name}")

    # Plot ROC and PR curves
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    for name in models.keys():
        prob = test_probs[name]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_val = roc_auc_score(y_test, prob)
        ax1.plot(fpr, tpr, lw=2, label=f"{name.split(' (')[0]} (AUC={roc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_val = average_precision_score(y_test, prob)
        ax2.plot(rec, prec, lw=2, label=f"{name.split(' (')[0]} (AP={pr_val:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
    ax1.set_title("Pre-Departure ROC Curves (T-2h Prediction)", pad=10)
    ax1.set_xlabel("False Positive Rate", labelpad=8)
    ax1.set_ylabel("True Positive Rate (Recall)", labelpad=8)
    ax1.legend(loc="lower right")

    ax2.plot([0, 1], [y_test.mean(), y_test.mean()], "k--", lw=1.5, alpha=0.6, label=f"Baseline ({y_test.mean():.2f})")
    ax2.set_title("Pre-Departure Precision-Recall Curves", pad=10)
    ax2.set_xlabel("Recall", labelpad=8)
    ax2.set_ylabel("Precision", labelpad=8)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    roc_pr_path = FIGURES_DIR / "pre_departure_roc_pr_curve.png"
    fig.savefig(roc_pr_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved ROC/PR curves to {roc_pr_path}")

    # Threshold Optimization Table (Crucial for operations!)
    best_prob = test_probs[best_name]
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    thresh_rows = []
    for t in thresholds:
        pred_t = (best_prob >= t).astype(int)
        rep = classification_report(y_test, pred_t, output_dict=True, zero_division=0)
        thresh_rows.append({
            "Decision_Threshold": t,
            "High_Risk_Precision": round(rep["1"]["precision"], 4),
            "High_Risk_Recall": round(rep["1"]["recall"], 4),
            "High_Risk_F1": round(rep["1"]["f1-score"], 4),
            "Overall_Accuracy": round(rep["accuracy"], 4),
        })
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_csv = REPORTS_DIR / "pre_departure_threshold_tuning.csv"
    thresh_df.to_csv(thresh_csv, index=False)

    # Save best model
    best_path = MODELS_DIR / "pre_departure_model.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "classes": classes,
        "model_name": best_name,
        "features": {
            "numeric": [c for c in PRE_DEPARTURE_NUMERIC_FEATURES if c in X.columns],
            "categorical": [c for c in PRE_DEPARTURE_CATEGORICAL_FEATURES if c in X.columns],
        },
        "threshold_tuning": thresh_df,
    }, best_path)
    logger.info(f"Serialized pre-departure model to {best_path}")

    print("\n" + "=" * 60)
    print("PRE-DEPARTURE BENCHMARK SUMMARY:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nOPERATIONAL THRESHOLD TUNING (High-Risk Alert Trade-off):")
    print(thresh_df.to_string(index=False))

    return results_df

if __name__ == "__main__":
    run_pre_departure_benchmark()
