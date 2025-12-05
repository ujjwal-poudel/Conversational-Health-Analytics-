"""
Chroma feature extraction module.

Computes 12-bin chroma features using paper window/hop settings.
"""

import os
import numpy as np
import librosa

from src.utils.config import (
    CHROMA_N_CHROMA,
    WINDOW_LENGTH,
    HOP_LENGTH,
    FEATURE_CHROMA_DIR,
    get_processed_audio_path,
)


def extract_chroma(participant_id):
    """
    Extract chroma features for the participant.
    """
    audio_path = get_processed_audio_path(participant_id)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing cleaned audio for {participant_id}")

    waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    chroma = librosa.feature.chroma_stft(
        y=waveform,
        sr=sr,
        n_chroma=CHROMA_N_CHROMA,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
    )

    return chroma.T.astype(np.float32)


def save_chroma(participant_id):
    """
    Compute and save chroma features.
    """
    chroma = extract_chroma(participant_id)
    out_path = os.path.join(FEATURE_CHROMA_DIR, f"{participant_id}.npy")
    np.save(out_path, chroma)
    return out_path
