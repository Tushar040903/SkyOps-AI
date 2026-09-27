"""
Supervised Multi-Class Flight Complexity Classification Pipeline.
Benchmarks Logistic Regression, Random Forest, LightGBM, and XGBoost using Stratified 5-Fold CV.
Generates evaluation metrics, confusion matrices, and serializes the best performing model.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.config import (
    ENRICHED_COMPLEXITY_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    FULL_NUMERIC_FEATURES,
    FULL_CATEGORICAL_FEATURES,
    TARGET_MULTICLASS,
)
from src.utils import get_logger, evaluate_classification, plot_confusion_matrix

logger = get_logger(__name__)

def prepare_data() -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, ColumnTransformer]:
    """Load enriched data, extract feature matrices, and build preprocessing pipeline."""
    df = pd.read_csv(ENRICHED_COMPLEXITY_PATH)
    logger.info(f"Loaded master dataset: {df.shape}")

    # Validate feature presence
    num_cols = [c for c in FULL_NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in FULL_CATEGORICAL_FEATURES if c in df.columns]

    X = df[num_cols + cat_cols].copy()
    y_raw = df[TARGET_MULTICLASS].copy()

    # Encode target labels
    label_encoder = LabelEncoder()
    # Ensure ordered classes: Easy=0, Medium=1, Hard=2
    label_order = ["Easy", "Medium", "Hard"]
    label_encoder.fit(label_order)
    y = label_encoder.transform(y_raw)

    # Impute and clean
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    # Preprocessor: Scaler for numeric, OneHot for categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    logger.info(f"Feature space: {len(num_cols)} numeric features, {len(cat_cols)} categorical features.")
    return X, pd.Series(y, name=TARGET_MULTICLASS), label_encoder, preprocessor

def run_classification_benchmark() -> pd.DataFrame:
    """
    Train and evaluate multiple classification models using 5-Fold Stratified Cross-Validation.
    Serializes best model and exports performance comparison.
    """
    logger.info("Initializing multi-class classification benchmark...")
    X, y, label_encoder, preprocessor = prepare_data()
    classes = list(label_encoder.classes_)

    # Holdout split: 80% train-validation, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Model definitions
    candidate_models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    benchmark_rows = []
    trained_pipelines = {}

    for name, model in candidate_models.items():
        logger.info(f"Evaluating {name} with {CV_FOLDS}-fold Stratified CV...")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        # Cross-validation scoring
        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "f1_macro": "f1_macro",
            "roc_auc_ovr": "roc_auc_ovr_weighted",
        }
        cv_res = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        mean_acc = cv_res["test_accuracy"].mean()
        std_acc = cv_res["test_accuracy"].std()
        mean_f1_w = cv_res["test_f1_weighted"].mean()
        std_f1_w = cv_res["test_f1_weighted"].std()
        mean_f1_m = cv_res["test_f1_macro"].mean()
        mean_roc = cv_res["test_roc_auc_ovr"].mean()

        # Fit on entire training set and evaluate on test set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        test_eval = evaluate_classification(y_test, y_pred, y_prob, classes=classes, model_name=name)

        benchmark_rows.append({
            "Model": name,
            "CV_Accuracy_Mean": round(mean_acc, 4),
            "CV_Accuracy_Std": round(std_acc, 4),
            "CV_F1_Weighted": round(mean_f1_w, 4),
            "CV_F1_Macro": round(mean_f1_m, 4),
            "CV_ROC_AUC": round(mean_roc, 4),
            "Test_Accuracy": round(test_eval["accuracy"], 4),
            "Test_F1_Weighted": round(test_eval["f1_weighted"], 4),
            "Test_F1_Macro": round(test_eval["f1_macro"], 4),
            "Test_ROC_AUC": round(test_eval["roc_auc"], 4) if test_eval["roc_auc"] else None,
        })

        logger.info(f"[{name}] Test Accuracy: {test_eval['accuracy']:.4f} | F1 Weighted: {test_eval['f1_weighted']:.4f} | Macro: {test_eval['f1_macro']:.4f}")

    results_df = pd.DataFrame(benchmark_rows).sort_values("Test_F1_Weighted", ascending=False)
    benchmark_csv_path = REPORTS_DIR / "classification_benchmark_results.csv"
    results_df.to_csv(benchmark_csv_path, index=False)
    logger.info(f"Saved benchmark results to {benchmark_csv_path}")

    # Best model selection based on Test F1 Weighted
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    logger.info(f"Top Model Selected: {best_model_name}")

    # Evaluate best model confusion matrix
    y_test_pred = best_pipeline.predict(X_test)
    test_eval = evaluate_classification(y_test, y_test_pred, classes=classes, model_name=best_model_name)

    cm_save_path = FIGURES_DIR / "confusion_matrix_multiclass.png"
    plot_confusion_matrix(
        test_eval["confusion_matrix"],
        classes=classes,
        title=f"Multi-Class Complexity Confusion Matrix\n({best_model_name})",
        save_path=cm_save_path,
    )

    # Save best model artifact
    best_model_path = MODELS_DIR / "best_complexity_classifier.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "classes": classes,
        "model_name": best_model_name,
        "features": {
            "numeric": [c for c in FULL_NUMERIC_FEATURES if c in X.columns],
            "categorical": [c for c in FULL_CATEGORICAL_FEATURES if c in X.columns],
        },
        "test_metrics": test_eval,
    }, best_model_path)
    logger.info(f"Serialized best model artifact to: {best_model_path}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION BENCHMARK SUMMARY:")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 60)
    print(f"BEST MODEL ({best_model_name}) DETAILED CLASSIFICATION REPORT:")
    print("=" * 60)
    print(test_eval["report_text"])

    return results_df

if __name__ == "__main__":
    run_classification_benchmark()
