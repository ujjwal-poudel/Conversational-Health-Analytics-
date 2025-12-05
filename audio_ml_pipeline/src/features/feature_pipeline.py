"""
Feature extraction pipeline for a participant.

Runs all individual feature extractors and stores:
- individual .npy feature files
- combined feature .npy
"""

import os
import numpy as np

from src.features.extract_mfcc import extract_mfcc
from src.features.extract_chroma import extract_chroma
from src.features.extract_spectral import extract_spectral
from src.features.extract_rms import extract_rms
from src.features.extract_zcr import extract_zcr
from src.features.extract_tonnetz import extract_tonnetz

from src.utils.config import FEATURE_COMBINED_DIR

def extract_all_features(participant_id):
    """
    Extract all feature types and return them in a dictionary.
    """
    f_mfcc = extract_mfcc(participant_id)
    f_chroma = extract_chroma(participant_id)
    f_spectral = extract_spectral(participant_id)
    f_rms = extract_rms(participant_id)
    f_zcr = extract_zcr(participant_id)
    f_tonnetz = extract_tonnetz(participant_id)

    return {
        "mfcc": f_mfcc,
        "chroma": f_chroma,
        "spectral": f_spectral,
        "rms": f_rms,
        "zcr": f_zcr,
        "tonnetz": f_tonnetz,
    }


def save_combined_features(participant_id):
    """
    Extract and concatenate all feature matrices (frame-aligned).
    Saves a single combined .npy file.
    """
    feats = extract_all_features(participant_id)

    # Find minimum frame count across all feature types
    min_frames = min(feat.shape[0] for feat in feats.values())

    # Trim all features to same frame count
    trimmed = [feat[:min_frames] for feat in feats.values()]

    combined = np.concatenate(trimmed, axis=1).astype(np.float32)

    out_path = os.path.join(FEATURE_COMBINED_DIR, f"{participant_id}.npy")
    np.save(out_path, combined)

    return out_path
