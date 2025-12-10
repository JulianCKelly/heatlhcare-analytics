"""
clean_data.py

Data loading, cleaning, and feature/target splitting utilities
for the Healthcare Analytics diabetes risk project.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw CSV data from the given path.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Cleaning logic aligned with the notebook and the typical Pima dataset:

    - Strip whitespace from column names
    - Replace blank strings with NaN
    - Treat 0 as missing for:
        Glucose, BloodPressure, SkinThickness, Insulin, BMI
      (physiologically implausible → missing)
    - Drop rows with missing target
    - Median-impute numeric columns
    - Mode-impute non-numeric columns
    """
    df = df.copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Replace empty strings with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Columns where 0 usually actually means "missing"
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    for col in zero_as_missing:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Drop any rows where the target is missing
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    df = df.dropna(subset=[target_col])

    # Impute numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Impute categorical / non-numeric columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    return df


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a cleaned DataFrame into features (X) and target (y).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def load_and_clean_data(
    raw_path: str = "data/raw/diabetes.csv",
    target_col: str = "Outcome",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function:
    - Load raw data from CSV
    - Clean it
    - Return X, y
    """
    df_raw = load_raw_data(raw_path)
    df_clean = basic_clean(df_raw, target_col=target_col)
    X, y = split_features_target(df_clean, target_col=target_col)
    return X, y


if __name__ == "__main__":
    # Manual sanity check (will raise if the CSV is missing)
    try:
        X, y = load_and_clean_data()
        print("Data loaded and cleaned successfully.")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print("Columns:", list(X.columns))
    except FileNotFoundError as e:
        print(e)