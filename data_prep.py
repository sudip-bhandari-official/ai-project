"""
Data Preprocessing Module (data_prep.py)
Loads the agricultural dataset, validates features, and performs train-test split.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

FEATURE_METADATA = {
    "N": {"label": "Nitrogen (N)", "unit": "ratio in soil", "min": 0, "max": 140},
    "P": {"label": "Phosphorus (P)", "unit": "ratio in soil", "min": 5, "max": 145},
    "K": {"label": "Potassium (K)", "unit": "ratio in soil", "min": 5, "max": 205},
    "temperature": {"label": "Temperature", "unit": "deg C", "min": 8.8, "max": 43.7},
    "humidity": {"label": "Relative Humidity", "unit": "%", "min": 14.0, "max": 100.0},
    "ph": {"label": "pH Value", "unit": "scale (3.5 - 9.9)", "min": 3.5, "max": 9.9},
    "rainfall": {"label": "Rainfall", "unit": "mm", "min": 20.0, "max": 298.6},
}


def load_and_preprocess_data(dataset_path: str = "Crop_recommendation_1000.csv", test_size: float = 0.2, random_state: int = 42):
    """Loads CSV dataset, validates schema, drops nulls, and returns stratified train-test splits."""
    if not os.path.exists(dataset_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(base_dir, "Crop_recommendation_1000.csv")
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}' or '{alt_path}'.")

    df = pd.read_csv(dataset_path)

    expected_cols = FEATURE_NAMES + [TARGET_COLUMN]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    if df[expected_cols].isnull().sum().sum() > 0:
        df = df.dropna(subset=expected_cols)

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Data Loaded: {len(X_train)} train, {len(X_test)} test across {y_train.nunique()} crops.")

