"""
Preprocessing pipeline for DAIC-WOZ audio data.

This script runs the full preprocessing pipeline:
1. Extracts participant-only audio from raw recordings using transcript timestamps.
2. Applies high-pass filtering to remove low-frequency noise.
3. Saves cleaned audio to the processed directory.

Usage:
    python -m src.preprocessing.pipeline
"""

import os

from src.preprocessing.diarization import extract_participant_audio, save_clean_audio
from src.preprocessing.filtering import apply_highpass_filter
from src.labels.load_labels import load_train_labels, load_dev_labels, load_test_labels
from src.utils.config import PARTICIPANT_MIN_ID, PARTICIPANT_MAX_ID, get_processed_audio_path


def get_all_participant_ids():
    """
    Collect all participant IDs from train, dev, and test labels.
    """
    all_ids = set()

    try:
        all_ids.update(load_train_labels().keys())
    except Exception as e:
        print(f"Warning: Could not load train labels: {e}")

    try:
        all_ids.update(load_dev_labels().keys())
    except Exception as e:
        print(f"Warning: Could not load dev labels: {e}")

    try:
        all_ids.update(load_test_labels().keys())
    except Exception as e:
        print(f"Warning: Could not load test labels: {e}")

    return sorted(all_ids)


def process_participant(participant_id):
    """
    Process a single participant: extract, filter, and save audio.
    """
    # Check if already processed
    output_path = get_processed_audio_path(participant_id)
    if os.path.exists(output_path):
        print(f"[SKIP] {participant_id} already processed.")
        return output_path

    try:
        # Extract participant-only audio
        waveform = extract_participant_audio(participant_id)

        if len(waveform) == 0 or (len(waveform) == 1 and waveform[0] == 0):
            print(f"[WARN] {participant_id}: No participant audio found.")
            return None

        # Apply high-pass filter
        filtered = apply_highpass_filter(waveform)

        # Save cleaned audio
        save_clean_audio(filtered, participant_id)
        print(f"[OK] {participant_id}: Processed and saved.")
        return output_path

    except FileNotFoundError as e:
        print(f"[ERROR] {participant_id}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] {participant_id}: Unexpected error - {e}")
        return None


def main():
    print("=" * 60)
    print("DAIC-WOZ Audio Preprocessing Pipeline")
    print("=" * 60)

    participant_ids = get_all_participant_ids()
    print(f"Found {len(participant_ids)} participants to process.\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for pid in participant_ids:
        result = process_participant(pid)
        if result:
            if "SKIP" in str(result):
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1

    print("\n" + "=" * 60)
    print(f"Preprocessing complete.")
    print(f"  Processed: {success_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
