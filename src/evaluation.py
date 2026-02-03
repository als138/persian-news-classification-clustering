"""
Evaluation metrics and visualization for classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> dict:
    """
    Compute Accuracy and F1-Score (macro and weighted).
    
    Returns:
        Dict with accuracy, f1_macro, f1_weighted
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_evaluation_report(
    metrics: dict,
    model_name: str,
) -> None:
    """Print formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"  {model_name} - Evaluation Report")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"{'='*50}\n")
