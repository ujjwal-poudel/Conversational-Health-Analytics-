"""
Load PHQ regression labels for DAIC-WOZ.

Handles:
- train/dev splits with column 'PHQ8_Score'
- test split with column 'PHQ_Score'
- participant naming pattern '{id}_P'
"""

import os
import pandas as pd

from src.utils.config import (
    TRAIN_LABEL_FILE,
    DEV_LABEL_FILE,
    TEST_LABEL_FILE,
)


def _load_label_file(path, score_column):
    """
    Load a label file and return a dict mapping participant_id -> label.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    df = pd.read_csv(path)

    if score_column not in df.columns:
        raise ValueError(f"Missing column '{score_column}' in: {path}")

    mapping = {}

    for _, row in df.iterrows():
        pid = f"{int(row['Participant_ID'])}_P"
        mapping[pid] = float(row[score_column])

    return mapping


def load_train_labels():
    return _load_label_file(TRAIN_LABEL_FILE, "PHQ8_Score")


def load_dev_labels():
    return _load_label_file(DEV_LABEL_FILE, "PHQ8_Score")


def load_test_labels():
    return _load_label_file(TEST_LABEL_FILE, "PHQ_Score")
