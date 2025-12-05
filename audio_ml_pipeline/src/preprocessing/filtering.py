"""
High-pass filtering utilities for audio preprocessing.

Implements a standard Butterworth high-pass filter.
Used to remove low-frequency noise below 300 Hz.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from src.utils.config import AUDIO_SR, HIGHPASS_CUTOFF, HIGHPASS_ORDER


def butter_highpass(cutoff=HIGHPASS_CUTOFF, sr=AUDIO_SR, order=HIGHPASS_ORDER):
    """
    Compute Butterworth high-pass filter coefficients.
    """
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype="high", analog=False)
    return b, a


def apply_highpass_filter(waveform):
    """
    Apply the high-pass filter to a waveform.
    """
    if len(waveform) == 0:
        return waveform.astype(np.float32)

    b, a = butter_highpass()
    filtered = filtfilt(b, a, waveform)
    return filtered.astype(np.float32)