"""
Utility Functions for Flight Complexity Analysis and Prediction.
Provides standardized logging, plotting aesthetics, metric reports, and directory helpers.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# Configure standard logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with standard project formatting."""
    return logging.getLogger(name)

logger = get_logger(__name__)

def set_plot_style():
    """Apply professional, publication-ready plotting aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
        "grid.color": "#EAEAEA",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.autolayout": True,
    })

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    classes: Optional[list] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    Supports binary and multi-class classification.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    roc_auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
                roc_auc = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class classification (OVR)
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC for {model_name}: {e}")

    # Handle label mapping if y_true is numeric and classes are strings
    unique_true = np.unique(y_true)
    if classes is not None and isinstance(classes[0], str) and np.issubdtype(unique_true.dtype, np.integer):
        numeric_labels = list(range(len(classes)))
        cm = confusion_matrix(y_true, y_pred, labels=numeric_labels)
        report_dict = classification_report(y_true, y_pred, labels=numeric_labels, target_names=classes, output_dict=True, zero_division=0)
        report_text = classification_report(y_true, y_pred, labels=numeric_labels, target_names=classes, zero_division=0)
    else:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        report_text = classification_report(y_true, y_pred, target_names=classes, zero_division=0)

    results = {
        "model_name": model_name,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_text": report_text,
    }
    return results

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot and optionally save a publication-quality confusion matrix heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Ground Truth Label", labelpad=8)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix plot to {save_path}")

    return fig
