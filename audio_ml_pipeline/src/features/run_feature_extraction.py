"""
Feature extraction pipeline for DAIC-WOZ audio data.

This script runs the full feature extraction pipeline:
1. Loads labels for all participants.
2. For each participant with processed audio:
   - Extracts combined features (MFCC, chroma, spectral, RMS, ZCR, tonnetz)
   - Computes summary statistics
   - Saves summary CSV with PHQ label

Usage:
    python -m src.features.run_feature_extraction
"""

import os

from src.features.feature_pipeline import save_combined_features
from src.features.summary_stats import save_summary_with_label
from src.labels.load_labels import load_train_labels, load_dev_labels, load_test_labels
from src.utils.config import get_processed_audio_path, get_summary_feature_path


def get_all_labels():
    """
    Merge all labels from train, dev, and test splits.
    """
    all_labels = {}

    try:
        all_labels.update(load_train_labels())
    except Exception as e:
        print(f"Warning: Could not load train labels: {e}")

    try:
        all_labels.update(load_dev_labels())
    except Exception as e:
        print(f"Warning: Could not load dev labels: {e}")

    try:
        all_labels.update(load_test_labels())
    except Exception as e:
        print(f"Warning: Could not load test labels: {e}")

    return all_labels


def process_participant(participant_id, label):
    """
    Extract features and save summary for a single participant.
    """
    # Check if summary already exists
    summary_path = get_summary_feature_path(participant_id)
    if os.path.exists(summary_path):
        print(f"[SKIP] {participant_id}: Summary already exists.")
        return summary_path

    # Check if processed audio exists
    audio_path = get_processed_audio_path(participant_id)
    if not os.path.exists(audio_path):
        print(f"[SKIP] {participant_id}: No processed audio found.")
        return None

    try:
        # Extract and save combined features
        save_combined_features(participant_id)

        # Compute summary stats and save with label
        save_summary_with_label(participant_id, label)

        print(f"[OK] {participant_id}: Features extracted (label={label}).")
        return summary_path

    except Exception as e:
        print(f"[ERROR] {participant_id}: {e}")
        return None


def main():
    print("=" * 60)
    print("DAIC-WOZ Feature Extraction Pipeline")
    print("=" * 60)

    all_labels = get_all_labels()
    print(f"Found {len(all_labels)} participants with labels.\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for pid, label in sorted(all_labels.items()):
        result = process_participant(pid, label)
        if result:
            success_count += 1
        elif result is None:
            # Could be skip or error - check audio
            audio_path = get_processed_audio_path(pid)
            if not os.path.exists(audio_path):
                skip_count += 1
            else:
                error_count += 1

    print("\n" + "=" * 60)
    print(f"Feature extraction complete.")
    print(f"  Extracted: {success_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
