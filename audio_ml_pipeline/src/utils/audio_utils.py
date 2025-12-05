"""
Utility functions for audio preprocessing:
- load and normalize audio
- chunk long audio into fixed-duration segments
"""

import numpy as np
import soundfile as sf


def load_audio_mono(wav_path):
    """
    Loads audio and converts stereo to mono.
    """
    audio, sr = sf.read(wav_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr


def normalize_audio(audio):
    """
    Normalizes audio amplitude to [-1, 1].
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def chunk_audio(audio, sr, chunk_seconds=20):
    """
    Splits audio into chunks of fixed duration without loss.
    """
    samples_per_chunk = chunk_seconds * sr
    chunks = []

    start = 0
    n = len(audio)

    while start < n:
        end = min(start + samples_per_chunk, n)
        chunks.append(audio[start:end])
        start = end

    return chunks