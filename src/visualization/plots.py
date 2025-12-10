"""
Visualization utilities for model diagnostics and reporting
in the Healthcare Analytics diabetes project.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def feature_importance_plot(
    model,
    feature_names: List[str],
    top_n: int | None = None,
    title: str = "Feature Importances",
) -> None:
    """
    Plot feature importances for tree-based models.

    Args:
        model: Trained model with `feature_importances_` attribute.
        feature_names: List of feature names corresponding to model input.
        top_n: If provided, only show the top N most important features.
        title: Plot title.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not have `feature_importances_` attribute.")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    if top_n is not None:
        indices = indices[:top_n]

    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(sorted_features)), sorted_importances[::-1])
    plt.yticks(range(len(sorted_features)), sorted_features[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def roc_curve_plot(
    model,
    X_test,
    y_test,
    title: str = "ROC Curve",
) -> None:
    """
    Plot ROC curve for a binary classifier.

    Args:
        model: Trained classifier with `predict_proba` or `decision_function`.
        X_test: Test features.
        y_test: True labels for the test set.
        title: Plot title.
    """
    plt.figure(figsize=(6, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def confusion_matrix_plot(
    model,
    X_test,
    y_test,
    normalize: str | None = "true",
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix for a classifier.

    Args:
        model: Trained classifier with `predict`.
        X_test: Test features.
        y_test: True labels.
        normalize: Normalization mode for confusion matrix (None, 'true', 'pred', 'all').
        title: Plot title.
    """
    y_pred = model.predict(X_test)

    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        normalize=normalize,
        cmap="Blues",
        colorbar=True,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()