"""
Diarization module for extracting participant-only audio
from DAIC-WOZ recordings using transcript timestamps.

This module:
1. Loads transcript segments (start_time, stop_time, speaker).
2. Removes Ellie segments entirely.
3. Concatenates participant-only speech.
4. Provides error handling for missing files or bad timestamps.
"""

import os
import csv
import numpy as np
import soundfile as sf

from src.utils.config import (
    AUDIO_SR,
    ELLIE_LABELS,
    PARTICIPANT_LABEL,
    build_raw_audio_path,
    build_transcript_path
)


def load_transcript(path):
    """
    Load transcript rows with columns:
    start_time, stop_time, speaker, value
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")

    segments = []

    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                seg = {
                    "start": float(row["start_time"]),
                    "end": float(row["stop_time"]),
                    "speaker": row["speaker"].strip()
                }
                segments.append(seg)
            except Exception:
                continue

    if len(segments) == 0:
        raise ValueError(f"No valid transcript segments found: {path}")

    return segments


def extract_participant_audio(participant_id):
    """
    Extract only Participant speech from the audio.

    Returns a waveform or raises informative errors.
    """
    audio_path = build_raw_audio_path(participant_id)
    transcript_path = build_transcript_path(participant_id)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file missing: {audio_path}")

    audio, sr = sf.read(audio_path)

    if sr != AUDIO_SR:
        raise ValueError(
            f"Sample rate mismatch for {participant_id}. "
            f"Expected {AUDIO_SR}, got {sr}."
        )

    transcript = load_transcript(transcript_path)

    segments = []
    for seg in transcript:
        if seg["speaker"] in ELLIE_LABELS:
            continue

        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)

        if start_idx < 0 or end_idx > len(audio):
            continue

        segments.append(audio[start_idx:end_idx])

    if len(segments) == 0:
        return np.zeros(1, dtype=np.float32)

    return np.concatenate(segments).astype(np.float32)


def save_clean_audio(waveform, participant_id):
    """
    Save cleaned participant audio to the processed directory.
    """
    from src.utils.config import get_processed_audio_path
    out_path = get_processed_audio_path(participant_id)

    sf.write(out_path, waveform, AUDIO_SR)

    return out_path