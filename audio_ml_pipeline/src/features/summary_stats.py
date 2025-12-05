"""
Summary statistics extraction for combined features.

Computes mean, std, min, max for each feature dimension and
injects PHQ regression label per participant.
"""

import os
import numpy as np
import pandas as pd

from src.utils.config import (
    FEATURE_COMBINED_DIR,
    get_summary_feature_path,
    SUMMARY_STATS,
)


def load_combined_features(participant_id):
    """
    Load combined feature matrix for the participant.
    """
    path = os.path.join(FEATURE_COMBINED_DIR, f"{participant_id}.npy")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing combined feature file: {path}")

    return np.load(path)


def compute_summary(matrix):
    """
    Compute summary-level statistics across frames.
    """
    stats = {}

    if "mean" in SUMMARY_STATS:
        vec = matrix.mean(axis=0)
        for idx, val in enumerate(vec):
            stats[f"mean_{idx}"] = val

    if "std" in SUMMARY_STATS:
        vec = matrix.std(axis=0)
        for idx, val in enumerate(vec):
            stats[f"std_{idx}"] = val

    if "min" in SUMMARY_STATS:
        vec = matrix.min(axis=0)
        for idx, val in enumerate(vec):
            stats[f"min_{idx}"] = val

    if "max" in SUMMARY_STATS:
        vec = matrix.max(axis=0)
        for idx, val in enumerate(vec):
            stats[f"max_{idx}"] = val

    return pd.DataFrame([stats])


def save_summary_with_label(participant_id, label):
    """
    Compute summary features + attach PHQ label, save as CSV.
    """
    matrix = load_combined_features(participant_id)
    summary_df = compute_summary(matrix)

    summary_df["PHQ_score"] = label

    out_path = get_summary_feature_path(participant_id)
    summary_df.to_csv(out_path, index=False)

    return out_path
