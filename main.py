"""
Command-line pipeline runner

Usage examples (from project root):

    # Train a model on the diabetes dataset
    python main.py train

    # Train with custom paths/params
    python main.py train --raw-path data/raw/diabetes.csv --test-size 0.25

    # Run predictions on a CSV file using the saved model
    python main.py predict --input data/raw/diabetes.csv --output predictions.csv
"""

import argparse
import os
from typing import List

import pandas as pd
import joblib

from src.models.train_model import train_model, save_model
from src.data.clean_data import load_and_clean_data


# ---------------------------
# Helpers
# ---------------------------

def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


# ---------------------------
# Commands
# ---------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """
    Train the model and save it to disk.
    """
    print("🔧 Training model...")
    model, metrics = train_model(
        test_size=args.test_size,
        random_state=args.random_state,
        raw_path=args.raw_path,
        target_col=args.target_col,
    )

    print("\n✅ Training complete. Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    save_model(model, path=args.model_path)
    print(f"\n💾 Model saved to: {args.model_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    """
    Load a trained model and generate predictions for a CSV file.
    """
    print("🔮 Running predictions...")

    # Load model
    ensure_file_exists(args.model_path)
    model = joblib.load(args.model_path)
    print(f"✅ Loaded model from: {args.model_path}")

    # Load input data
    ensure_file_exists(args.input)
    df_input = pd.read_csv(args.input)

    # Optionally drop target column if present
    if args.target_col in df_input.columns:
        df_input_features = df_input.drop(columns=[args.target_col])
    else:
        df_input_features = df_input

    # Predict
    preds = model.predict(df_input_features)

    # Attach predictions to original data and save
    df_output = df_input.copy()
    df_output[args.prediction_col] = preds

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
    df_output.to_csv(args.output, index=False)

    print(f"✅ Predictions written to: {args.output}")
    print(f"🧮 Sample predictions: {df_output[args.prediction_col].head().tolist()}")


# ---------------------------
# CLI setup
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Healthcare Analytics pipeline runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = subparsers.add_parser("train", help="Train a RandomForest model")
    train_parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/diabetes.csv",
        help="Path to raw diabetes CSV data.",
    )
    train_parser.add_argument(
        "--target-col",
        type=str,
        default="Outcome",
        help="Name of the target column in the dataset.",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for test set.",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    train_parser.add_argument(
        "--model-path",
        type=str,
        default="models/random_forest_diabetes.pkl",
        help="Where to save the trained model.",
    )
    train_parser.set_defaults(func=cmd_train)

    # PREDICT
    predict_parser = subparsers.add_parser(
        "predict", help="Generate predictions using a trained model"
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file for prediction.",
    )
    predict_parser.add_argument(
        "--model-path",
        type=str,
        default="models/random_forest_diabetes.pkl",
        help="Path to trained model .pkl file.",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save CSV with predictions.",
    )
    predict_parser.add_argument(
        "--target-col",
        type=str,
        default="Outcome",
        help="Optional: target column to drop from input (if present).",
    )
    predict_parser.add_argument(
        "--prediction-col",
        type=str,
        default="prediction",
        help="Name of the prediction column in the output CSV.",
    )
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()