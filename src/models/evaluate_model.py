"""
evaluate_model.py

Model evaluation utilities for the Healthcare Analytics project.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate(
    y_true,
    y_pred,
    y_proba=None,
) -> Dict[str, float]:
    """
    Compute common classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Optional predicted probabilities (for ROC AUC).

    Returns:
        dict of metrics: accuracy, precision, recall, f1, and optionally roc_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # Only compute ROC AUC for binary classification when proba is provided
    if y_proba is not None:
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


if __name__ == "__main__":
    # Small self-test
    import numpy as np

    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.3, 0.2, 0.9])

    metrics = evaluate(y_true, y_pred, y_proba)
    print("Self-test metrics:", metrics)