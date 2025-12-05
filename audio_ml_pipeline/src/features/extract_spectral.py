"""
Spectral feature extraction.

Features:
- spectral centroid
- spectral rolloff
- spectral flatness

Window/hop follow the paper settings.
"""

import os
import numpy as np
import librosa

from src.utils.config import (
    WINDOW_LENGTH,
    HOP_LENGTH,
    FEATURE_SPECTRAL_DIR,
    get_processed_audio_path,
)


def extract_spectral(participant_id):
    """
    Extract centroid, rolloff, and flatness as a single matrix.

    Returns a matrix of shape (frames, 3).
    """
    audio_path = get_processed_audio_path(participant_id)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing cleaned audio for {participant_id}")

    waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sr,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=sr,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=waveform,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    stacked = np.vstack([centroid, rolloff, flatness]).T
    return stacked.astype(np.float32)


def save_spectral(participant_id):
    """
    Compute and save spectral features.
    """
    spectral = extract_spectral(participant_id)
    out_path = os.path.join(FEATURE_SPECTRAL_DIR, f"{participant_id}.npy")
    np.save(out_path, spectral)
    return out_path
