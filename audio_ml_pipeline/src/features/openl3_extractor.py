"""
OpenL3 Embedding Extraction

Extracts high-level audio embeddings using OpenL3.
Embeddings capture timbre, spectral structure, and emotional texture
useful for depression regression.

Optimizations:
- Full audio preserved
- Chunked inference for long recordings
- Amplitude normalization
- Mono conversion
- Saves raw embeddings (T, 512)
"""

import os
import numpy as np
import soundfile as sf

import openl3

from src.utils import config
from src.utils.logging_utils import get_logger
from src.utils.audio_utils import (
    load_audio_mono,
    normalize_audio,
    chunk_audio,
)

logger = get_logger(__name__)


class OpenL3Extractor:
    def __init__(self, 
                 input_repr="mel256",
                 content_type="music",
                 embedding_size=512):
        """
        Load the OpenL3 model with recommended settings.
        
        Parameters
        ----------
        input_repr : str
            "mel256" recommended for emotional acoustic features.
        content_type : str
            "music" or "env". Music content type performs better for emotion tasks.
        embedding_size : int
            512 dim embeddings recommended for audio emotion/depression tasks.
        """
        logger.info("Loading OpenL3 model")

        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size
        )

        # OpenL3 requires 48k Hz or 32k Hz internally
        self.required_sr = 48000
        logger.info("OpenL3 model loaded successfully")

    def extract_embeddings(self, wav_path):
        """
        Extracts raw OpenL3 embeddings (T, 512) for the given WAV file.
        """
        try:
            audio, sr = load_audio_mono(wav_path)
            audio = normalize_audio(audio)

            # Resample if needed
            if sr != self.required_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.required_sr)
                sr = self.required_sr

            chunks = chunk_audio(audio, sr, chunk_seconds=20)

            all_embeddings = []

            for chunk in chunks:
                # OpenL3 returns frame-wise embeddings
                emb, _ = openl3.get_audio_embedding(
                    chunk,
                    sr,
                    model=self.model,
                    center=False,      # prevent misalignment
                    hop_size=0.1,      # 100ms hop recommended
                    verbose=False
                )

                # OpenL3 gives shape (n_frames, 512)
                if emb is not None and len(emb) > 0:
                    all_embeddings.append(emb)

            if not all_embeddings:
                logger.warning(f"No embeddings extracted for {wav_path}")
                return None

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"OpenL3 extraction error for {wav_path}: {e}")
            return None


def run_openl3_extraction():
    """
    Runs OpenL3 extraction for all processed participant audio files.
    """
    processed_dir = config.PROCESSED_AUDIO_DIR
    output_dir = config.FEATURE_OPENL3_DIR

    extractor = OpenL3Extractor()

    wav_files = [
        f for f in os.listdir(processed_dir)
        if f.endswith(".wav")
    ]

    logger.info(f"Found {len(wav_files)} processed audio files.")

    for fname in wav_files:
        pid = fname.replace(".wav", "")
        wav_path = os.path.join(processed_dir, fname)
        out_path = os.path.join(output_dir, f"{pid}.npy")

        if os.path.exists(out_path):
            logger.info(f"Already extracted: {pid}, skipping.")
            continue

        logger.info(f"Extracting OpenL3 embeddings for participant {pid}")

        emb = extractor.extract_embeddings(wav_path)

        if emb is None:
            logger.warning(f"Skipping participant {pid} due to extraction issue.")
            continue

        np.save(out_path, emb)
        logger.info(f"Saved OpenL3 embeddings → {out_path}")

    logger.info("OpenL3 extraction finished.")


if __name__ == "__main__":
    run_openl3_extraction()
