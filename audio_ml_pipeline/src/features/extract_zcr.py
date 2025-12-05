"""
Zero-crossing rate extraction.
"""

import os
import numpy as np
import librosa

from src.utils.config import (
    WINDOW_LENGTH,
    HOP_LENGTH,
    FEATURE_ZCR_DIR,
    get_processed_audio_path,
)


def extract_zcr(participant_id):
    audio_path = get_processed_audio_path(participant_id)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Missing cleaned audio for {participant_id}")

    waveform, sr = librosa.load(audio_path, sr=None, mono=True)

    zcr = librosa.feature.zero_crossing_rate(
        y=waveform,
        frame_length=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
    )[0]

    return zcr.reshape(-1, 1).astype(np.float32)


def save_zcr(participant_id):
    zcr = extract_zcr(participant_id)
    out_path = os.path.join(FEATURE_ZCR_DIR, f"{participant_id}.npy")
    np.save(out_path, zcr)
    return out_path
