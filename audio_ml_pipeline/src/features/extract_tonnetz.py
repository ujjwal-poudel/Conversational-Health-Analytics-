"""
Tonal centroid (tonnetz) extraction.

Tonnetz describes harmonic relationships. Requires a harmonic component.
"""

import os
import numpy as np
import librosa

from src.utils.config import (
    WINDOW_LENGTH,
    HOP_LENGTH,
    FEATURE_TONNETZ_DIR,
    get_processed_audio_path,
)


def extract_tonnetz(participant_id):
    audio_path = get_processed_audio_path(participant_id)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing cleaned audio for {participant_id}")

    waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    harmonic = librosa.effects.harmonic(waveform)

    tonnetz = librosa.feature.tonnetz(
        y=harmonic,
        sr=sr,
    )

    return tonnetz.T.astype(np.float32)


def save_tonnetz(participant_id):
    tonnetz = extract_tonnetz(participant_id)
    out_path = os.path.join(FEATURE_TONNETZ_DIR, f"{participant_id}.npy")
    np.save(out_path, tonnetz)
    return out_path
