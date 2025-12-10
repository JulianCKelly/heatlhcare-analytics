"""
train_model.py

Training script for a RandomForestClassifier on the Healthcare
Analytics diabetes dataset, aligned with the notebook:

- Train/test split with stratification
- Class imbalance handled via class_weight="balanced"
- GridSearchCV over RandomForest hyperparameters
  with recall as the primary scoring metric
- Evaluation using accuracy, precision, recall, F1, ROC AUC
"""

import os
from typing import Tuple, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from src.data.clean_data import load_and_clean_data
from src.models.evaluate_model import evaluate


def train_model(
    test_size: float = 0.2,
    random_state: int = 42,
    raw_path: str = "data/raw/diabetes.csv",
    target_col: str = "Outcome",
    use_grid_search: bool = True,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a RandomForest model and evaluate it.

    Args:
        test_size: Proportion of data used for test set.
        random_state: Random seed.
        raw_path: Path to the raw CSV data.
        target_col: Name of the target column.
        use_grid_search: Whether to run GridSearchCV with recall optimization.

    Returns:
        model: Trained RandomForestClassifier (best estimator if grid search is used).
        metrics: Dict of evaluation metrics on the test set.
    """
    # Load and clean data
    X, y = load_and_clean_data(raw_path=raw_path, target_col=target_col)

    # Train/test split with stratification (preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if use_grid_search:
        # Hyperparameter grid similar to your notebook
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        base_model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",  # bias toward recall on minority class
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="recall",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        print("Best parameters from GridSearchCV:")
        print(grid_search.best_params_)
        print(f"Best cross-validated recall: {grid_search.best_score_:.4f}")
    else:
        # Simple fallback model (no grid search)
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)

    # Predictions on hold-out test set
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = evaluate(y_test, y_pred, y_proba)
    return model, metrics


def save_model(model, path: str = "models/random_forest_diabetes.pkl") -> None:
    """
    Save the trained model to disk.
    """
    import joblib

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


if __name__ == "__main__":
    # Train, evaluate, and save
    model, metrics = train_model()
    print("Training complete. Evaluation metrics on hold-out test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    save_model(model)
    print("Model saved to models/random_forest_diabetes.pkl")