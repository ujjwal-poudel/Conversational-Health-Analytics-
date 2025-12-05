"""
Dataset loader for deep learning features (wav2vec2 + prosody).

Loads the .npy files created by build_tabular_dataset.py and converts
them to DataFrames compatible with the existing training pipeline.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

from ..config.paths_deep import NPY_DATA_DIR


def load_npy_as_dataframe(npy_path: str) -> pd.DataFrame:
    """
    Load a .npy file and convert to DataFrame.
    
    The .npy files have shape (N, D+1) where the last column is PHQ_score.
    Feature columns are named feature_0, feature_1, ..., feature_D-1.
    """
    data = np.load(npy_path)
    n_features = data.shape[1] - 1
    
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    columns = feature_cols + ["PHQ_score"]
    
    df = pd.DataFrame(data, columns=columns)
    return df


def load_deep_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the train, dev, and test .npy files as DataFrames.
    
    Returns:
        train_df, dev_df, test_df (DataFrames)
    """
    train_path = os.path.join(NPY_DATA_DIR, "train_data.npy")
    dev_path = os.path.join(NPY_DATA_DIR, "dev_data.npy")
    test_path = os.path.join(NPY_DATA_DIR, "test_data.npy")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train_data.npy not found at {train_path}")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"dev_data.npy not found at {dev_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test_data.npy not found at {test_path}")
    
    train_df = load_npy_as_dataframe(train_path)
    dev_df = load_npy_as_dataframe(dev_path)
    test_df = load_npy_as_dataframe(test_path)
    
    print(f"Loaded train: {train_df.shape}, dev: {dev_df.shape}, test: {test_df.shape}")
    
    return train_df, dev_df, test_df


def prepare_deep_training_data(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare model-ready datasets from deep features.
    
    Returns:
        X_train_full, y_train_full, X_dev, y_dev, full_train_df
    """
    # Merge train + dev for CV-based tuning
    full_train_df = pd.concat([train_df, dev_df], ignore_index=True)
    
    feature_cols = [c for c in full_train_df.columns if c != "PHQ_score"]
    
    y_train_full = full_train_df["PHQ_score"]
    X_train_full = full_train_df[feature_cols]
    
    y_dev = dev_df["PHQ_score"]
    X_dev = dev_df[feature_cols]
    
    return X_train_full, y_train_full, X_dev, y_dev, full_train_df


def prepare_deep_test_data(test_df: pd.DataFrame):
    """
    Prepare test set for final evaluation.
    
    Returns:
        X_test, y_test, feature_cols
    """
    feature_cols = [c for c in test_df.columns if c != "PHQ_score"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["PHQ_score"]
    
    return X_test, y_test, feature_cols
