"""
Build ML-ready dataset files (train/dev/test) using
summary-level feature CSVs and PHQ labels.

Outputs:
    datasets/train.csv
    datasets/dev.csv
    datasets/test.csv
"""

from src.labels.load_labels import (
    load_train_labels,
    load_dev_labels,
    load_test_labels,
)

from src.utils.config import SUMMARY_FEATURE_DIR

import os
import pandas as pd

def load_summary_row(participant_id):
    path = os.path.join(SUMMARY_FEATURE_DIR, f"{participant_id}.csv")

    if not os.path.exists(path):
        return None

    return pd.read_csv(path)


def build_split(label_dict, output_path):
    """
    Build a dataset split from summary files + label mapping.
    """
    rows = []

    for pid, label in label_dict.items():
        df = load_summary_row(pid)
        if df is None:
            continue
        rows.append(df)

    if len(rows) == 0:
        raise ValueError("No summary files found for this split.")

    data = pd.concat(rows, axis=0)
    data.to_csv(output_path, index=False)
    return output_path


def main():
    os.makedirs("datasets", exist_ok=True)

    train_labels = load_train_labels()
    dev_labels = load_dev_labels()
    test_labels = load_test_labels()

    build_split(train_labels, "datasets/train.csv")
    build_split(dev_labels,   "datasets/dev.csv")
    build_split(test_labels,  "datasets/test.csv")

    print("Datasets created successfully.")


if __name__ == "__main__":
    main()