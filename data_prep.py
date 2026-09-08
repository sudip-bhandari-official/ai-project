"""
Data Preprocessing Module (data_prep.py)
---------------------------------------
Responsibility:
- Ingests 'Crop_recommendation.csv' via pandas.read_csv.
- Extracts features X (N, P, K, temperature, humidity, ph, rainfall) and target y (label).
- Performs an 80/20 train-test split for model training and validation.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Feature definitions and recommended environmental bounds
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
    """
    Loads the crop recommendation dataset, validates schema, and performs an 80/20 train-test split.
    Prefers 'Crop_recommendation_1000.csv' (1,000 crops) and falls back to 'Crop_recommendation.csv' (22 crops).

    Parameters:
        dataset_path (str): Relative or absolute path to CSV dataset
        test_size (float): Proportion of dataset to include in test split (default: 0.2)
        random_state (int): Seed for reproducible random splitting (default: 42)

    Returns:
        tuple: (X_train, X_test, y_train, y_test) as pandas DataFrames/Series.
    """
    # Locate dataset
    if not os.path.exists(dataset_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cand_1000 = os.path.join(base_dir, "Crop_recommendation_1000.csv")
        cand_22 = os.path.join(base_dir, "Crop_recommendation.csv")
        if os.path.exists(cand_1000):
            dataset_path = cand_1000
        elif os.path.exists(cand_22):
            dataset_path = cand_22
        elif os.path.exists("Crop_recommendation.csv"):
            dataset_path = "Crop_recommendation.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at '{dataset_path}' or fallback locations.")

    # Ingest CSV
    df = pd.read_csv(dataset_path)

    # Validate presence of all expected columns
    expected_cols = FEATURE_NAMES + [TARGET_COLUMN]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Check for missing/null values
    null_counts = df[expected_cols].isnull().sum().sum()
    if null_counts > 0:
        # Impute or drop if missing
        df = df.dropna(subset=expected_cols)

    # Separate features (X) and target (y)
    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]

    # Perform 80/20 train-test split with stratification across 22 classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Component 1: Data Preprocessing verification")
    print(f"  Training Features (X_train) Shape: {X_train.shape}")
    print(f"  Testing Features  (X_test)  Shape: {X_test.shape}")
    print(f"  Training Labels   (y_train) Shape: {y_train.shape}")
    print(f"  Testing Labels    (y_test)  Shape: {y_test.shape}")
    print(f"  Total Unique Classes: {y_train.nunique()}")
    print("  Preprocessing completed successfully!")
