"""
Optimized Wav2Vec2 embedding extraction.

Uses the emotion-tuned model:
    superb/wav2vec2-base-superb-er

Optimizations included:
- FP16 inference on MPS
- Chunked inference (20s chunks)
- Frame-level subsampling (::2)
- Amplitude normalization
- Full audio preserved

Embeddings are saved as raw (T, 768) arrays.
"""

import os
import numpy as np
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from src.utils import config
from src.utils.logging_utils import get_logger
from src.utils.audio_utils import (
    load_audio_mono,
    normalize_audio,
    chunk_audio,
)

logger = get_logger(__name__)


class Wav2Vec2Extractor:
    def __init__(self, model_name="superb/wav2vec2-base-superb-er"):
        """
        Load Wav2Vec2 with optimized inference settings.
        """
        logger.info(f"Loading Wav2Vec2 model: {model_name}")

        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        if self.device == "mps":
            self.model = self.model.to(self.device, dtype=torch.float16)
        else:
            self.model = self.model.to(self.device)

        self.model.eval()
        torch.set_grad_enabled(False)

    def extract_embeddings(self, wav_path):
        """
        Extracts embeddings from full audio using chunked inference.
        """
        try:
            audio, sr = load_audio_mono(wav_path)
            audio = normalize_audio(audio)
            chunks = chunk_audio(audio, sr, chunk_seconds=20)

            all_embeddings = []

            for chunk in chunks:
                inputs = self.feature_extractor(
                    chunk, sampling_rate=sr, return_tensors="pt"
                )

                if self.device == "mps":
                    inp = inputs.input_values.to(self.device, dtype=torch.float16)
                else:
                    inp = inputs.input_values.to(self.device)

                with torch.inference_mode():
                    outputs = self.model(inp)

                hidden = outputs.last_hidden_state.squeeze(0)
                hidden = hidden[::2]  # frame subsampling

                all_embeddings.append(hidden.cpu().numpy())

            return np.vstack(all_embeddings)

        except Exception as e:
            logger.error(f"Extraction failed for {wav_path}: {e}")
            return None


def run_wav2vec2_extraction():
    """
    Iterates through processed audio files and saves embeddings.
    """
    processed_dir = config.PROCESSED_AUDIO_DIR
    output_dir = config.FEATURE_WAV2VEC2_DIR

    extractor = Wav2Vec2Extractor()

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
            logger.info(f"Already exists, skipping {pid}")
            continue

        logger.info(f"Extracting embeddings for participant {pid}")

        emb = extractor.extract_embeddings(wav_path)
        if emb is None:
            logger.warning(f"Skipping participant {pid}")
            continue

        np.save(out_path, emb)
        logger.info(f"Saved embeddings: {out_path}")

    logger.info("Wav2Vec2 extraction finished.")


if __name__ == "__main__":
    run_wav2vec2_extraction()