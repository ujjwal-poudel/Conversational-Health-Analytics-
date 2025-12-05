import os
import pandas as pd
from typing import Tuple
from ..config.paths import DATASET_DIR


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the train, dev, and test CSV files from the datasets directory.

    Returns:
        train_df, dev_df, test_df (DataFrames)
    """

    train_path = os.path.join(DATASET_DIR, "train.csv")
    dev_path = os.path.join(DATASET_DIR, "dev.csv")
    test_path = os.path.join(DATASET_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"dev.csv not found at {dev_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    return train_df, dev_df, test_df


def prepare_training_data(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare the model-ready datasets.

    Steps:
        - Merge train and dev for cross-validation tuning.
        - Identify feature columns (all but 'PHQ_score').
        - Extract X and y for train, dev, and full_train.

    Returns:
        X_train_full, y_train_full, X_dev, y_dev, full_train_df
    """

    if "PHQ_score" not in train_df.columns:
        raise ValueError("PHQ_score column missing in train dataset.")

    # Merge train + dev for CV-based hyperparameter tuning
    full_train_df = pd.concat([train_df, dev_df], ignore_index=True)

    # Features = all columns except the target
    feature_cols = [c for c in full_train_df.columns if c != "PHQ_score"]

    # Extract label
    y_train_full = full_train_df["PHQ_score"]
    X_train_full = full_train_df[feature_cols]

    # Dev set split (kept separately for historical use, but not used for tuning)
    y_dev = dev_df["PHQ_score"]
    X_dev = dev_df[feature_cols]

    return X_train_full, y_train_full, X_dev, y_dev, full_train_df


def prepare_test_data(test_df: pd.DataFrame):
    """
    Prepare test set for final evaluation.

    Returns:
        X_test, y_test, feature_cols
    """

    if "PHQ_score" not in test_df.columns:
        raise ValueError("PHQ_score column missing in test dataset.")

    feature_cols = [c for c in test_df.columns if c != "PHQ_score"]

    X_test = test_df[feature_cols]
    y_test = test_df["PHQ_score"]

    return X_test, y_test, feature_cols