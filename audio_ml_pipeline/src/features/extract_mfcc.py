"""
MFCC feature extraction module.

Computes MFCC coefficients using paper settings:
- window length = 4096 samples
- hop length = 2205 samples
- n_mfcc = 80

Outputs a frame-by-feature matrix and saves it as .npy.
"""

import os
import numpy as np
import librosa

from src.utils.config import (
    MFCC_N_MFCC,
    WINDOW_LENGTH,
    HOP_LENGTH,
    FEATURE_MFCC_DIR,
    get_processed_audio_path,
)


def extract_mfcc(participant_id):
    """
    Extract MFCC features for a participant.

    Returns
    -------
    np.ndarray
        MFCC matrix of shape (frames, n_mfcc)
    """
    audio_path = get_processed_audio_path(participant_id)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing cleaned audio for {participant_id}")

    waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=MFCC_N_MFCC,
        n_fft=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
    )

    return mfcc.T.astype(np.float32)


def save_mfcc(participant_id):
    """
    Compute and save MFCC as .npy for the participant.
    """
    mfcc = extract_mfcc(participant_id)
    out_path = os.path.join(FEATURE_MFCC_DIR, f"{participant_id}.npy")
    np.save(out_path, mfcc)
    return out_path
