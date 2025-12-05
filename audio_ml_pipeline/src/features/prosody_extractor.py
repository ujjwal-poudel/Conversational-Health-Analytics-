"""
Prosody feature extraction.

Computes frame-level prosodic features for each participant:
- pitch (fundamental frequency, F0)
- energy (RMS)
- zero-crossing rate (ZCR)
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral contrast

Outputs a matrix of shape (T, D) per participant and saves it as .npy
in FEATURE_PROSODY_DIR.
"""

import os
import numpy as np
import librosa

from src.utils import config
from src.utils.logging_utils import get_logger
from src.utils.audio_utils import load_audio_mono, normalize_audio

logger = get_logger(__name__)


class ProsodyExtractor:
    def __init__(self):
        """
        Initialize parameters from the config.
        """
        self.target_sr = config.AUDIO_SR
        self.n_fft = config.WINDOW_LENGTH
        self.hop_length = config.HOP_LENGTH

    def _ensure_sr(self, audio, sr):
        """
        Resamples audio to target_sr if needed.
        """
        if sr == self.target_sr:
            return audio, sr
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio_resampled, self.target_sr

    def extract_features(self, wav_path):
        """
        Extracts prosodic features for a single WAV file.

        Returns
        -------
        np.ndarray or None
            Matrix of shape (T, D), where T is number of frames
            and D is number of prosodic features.
        """
        try:
            audio, sr = load_audio_mono(wav_path)
            audio = normalize_audio(audio)
            audio, sr = self._ensure_sr(audio, sr)

            n_fft = self.n_fft
            hop = self.hop_length

            # RMS energy
            rms = librosa.feature.rms(
                y=audio,
                frame_length=n_fft,
                hop_length=hop,
                center=True,
            )  # shape (1, T)

            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio,
                frame_length=n_fft,
                hop_length=hop,
                center=True,
            )  # shape (1, T)

            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                center=True,
            )  # shape (1, T)

            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                center=True,
            )  # shape (1, T)

            # Spectral rolloff (high-frequency energy)
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                center=True,
                roll_percent=0.85,
            )  # shape (1, T)

            # Spectral contrast (multi-band)
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop,
                center=True,
            )  # shape (7, T) by default

            # Pitch (fundamental frequency) using YIN
            f0 = librosa.yin(
                y=audio,
                fmin=50,
                fmax=400,
                sr=sr,
                frame_length=n_fft,
                hop_length=hop,
            )  # shape (T,)

            # Replace NaNs (unvoiced) with 0
            f0 = np.nan_to_num(f0, nan=0.0)
            f0 = f0[np.newaxis, :]  # shape (1, T)

            # Align all features to the same number of frames
            feature_list = [rms, zcr, centroid, bandwidth, rolloff, contrast, f0]
            min_frames = min(feat.shape[1] for feat in feature_list)

            aligned = [feat[:, :min_frames] for feat in feature_list]

            # Stack features along feature dimension: (D, T)
            stacked = np.vstack(aligned)

            # Transpose to (T, D)
            return stacked.T

        except Exception as e:
            logger.error(f"Prosody extraction error for {wav_path}: {e}")
            return None


def run_prosody_extraction():
    """
    Runs prosody feature extraction for all processed participant audio files.
    Saves features to FEATURE_PROSODY_DIR as <participant_id>.npy.
    """
    processed_dir = config.PROCESSED_AUDIO_DIR
    output_dir = config.FEATURE_PROSODY_DIR

    extractor = ProsodyExtractor()

    wav_files = [
        f for f in os.listdir(processed_dir)
        if f.endswith(".wav")
    ]

    logger.info(f"Found {len(wav_files)} processed audio files for prosody extraction.")

    for fname in wav_files:
        pid = fname.replace(".wav", "")
        wav_path = os.path.join(processed_dir, fname)
        out_path = os.path.join(output_dir, f"{pid}.npy")

        if os.path.exists(out_path):
            logger.info(f"Prosody features already exist for {pid}, skipping.")
            continue

        logger.info(f"Extracting prosody features for participant {pid}")

        feats = extractor.extract_features(wav_path)

        if feats is None:
            logger.warning(f"Skipping participant {pid} due to extraction failure.")
            continue

        np.save(out_path, feats)
        logger.info(f"Saved prosody features → {out_path}")

    logger.info("Prosody extraction finished.")


if __name__ == "__main__":
    run_prosody_extraction()