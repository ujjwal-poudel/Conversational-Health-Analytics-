"""
Audio Inference Service for Depression Score Prediction

This module provides a modular, professional audio processing pipeline for
extracting features from audio and predicting depression scores using
the Lasso model trained on Wav2Vec2 + Prosody features.

Pipeline:
1. AudioPreprocessor: load → resample → normalize → 300Hz high-pass filter
2. Wav2Vec2Extractor: audio → embeddings (768 dims)
3. ProsodyExtractor: audio → prosody features (13 dims)
4. FeatureProcessor: PCA → segment pooling → feature vector
5. AudioScorer: scaler → selector → Lasso → score

All models are loaded once at startup for fast inference.
"""

import os
import numpy as np
import torch
import joblib
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from typing import Optional, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


# =============================================================================
# CONFIGURATION
# =============================================================================

class AudioConfig:
    """Audio processing configuration matching training pipeline."""
    
    # Paths to model files - environment variables for production, fallback to local
    # Get the backend directory (parent of src/)
    backend_dir = os.path.dirname(os.path.dirname(__file__))
    
    LASSO_MODEL_DIR = os.getenv(
        "LASSO_MODEL_DIR",
        os.path.join(backend_dir, "models/lasso/lasso_final_v8")
    )
    PCA_PATH = os.getenv(
        "PCA_MODEL_PATH",
        os.path.join(backend_dir, "models/pca/pca_wav2vec2.joblib")
    )
    

    
    # Audio parameters (must match training)
    AUDIO_SR = 16000
    HIGHPASS_CUTOFF = 300
    HIGHPASS_ORDER = 5
    
    # Feature extraction parameters
    WINDOW_LENGTH = 4096
    HOP_LENGTH = 2205
    
    # Wav2Vec2 model
    WAV2VEC2_MODEL = "superb/wav2vec2-base-superb-er"
    CHUNK_SECONDS = 20  # Process audio in 20s chunks
    
    # PCA settings (must match training)
    PCA_N_COMPONENTS = 200  # 768 → 200
    USE_SEGMENT_POOLING = True


# =============================================================================
# AUDIO PREPROCESSOR
# =============================================================================

class AudioPreprocessor:
    """
    Handles audio loading, resampling, normalization, and filtering.
    Matches the training pipeline exactly.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._setup_filter_coefficients()
    
    def _setup_filter_coefficients(self):
        """Pre-compute Butterworth filter coefficients."""
        nyquist = 0.5 * self.config.AUDIO_SR
        norm_cutoff = self.config.HIGHPASS_CUTOFF / nyquist
        self.b, self.a = butter(
            self.config.HIGHPASS_ORDER, 
            norm_cutoff, 
            btype="high", 
            analog=False
        )
    
    def load_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """Load audio and convert to mono."""
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        
        audio, sr = sf.read(wav_path)
        
        # Convert stereo to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return audio, sr
    
    def resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample to target sample rate if needed."""
        if sr == self.config.AUDIO_SR:
            return audio
        return librosa.resample(audio, orig_sr=sr, target_sr=self.config.AUDIO_SR)
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize amplitude to [-1, 1]."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def apply_highpass_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply 300Hz Butterworth high-pass filter (matches training)."""
        if len(audio) == 0:
            return audio.astype(np.float32)
        
        filtered = filtfilt(self.b, self.a, audio)
        return filtered.astype(np.float32)
    
    def process(self, wav_path: str) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Returns processed audio ready for feature extraction.
        """
        # Load
        audio, sr = self.load_audio(wav_path)
        
        # Resample to 16kHz
        audio = self.resample(audio, sr)
        
        # Normalize
        audio = self.normalize(audio)
        
        # Apply 300Hz high-pass filter
        audio = self.apply_highpass_filter(audio)
        
        return audio


# =============================================================================
# WAV2VEC2 EXTRACTOR
# =============================================================================

class Wav2Vec2Extractor:
    """
    Extracts Wav2Vec2 embeddings from audio.
    Uses the emotion-tuned model with optimizations for MPS.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.model = None
        self.feature_extractor = None
        self.device = None
        self._loaded = False
    
    def load(self):
        """Load Wav2Vec2 model (call once at startup)."""
        if self._loaded:
            return
        
        print(f"[AUDIO_SERVICE] Loading Wav2Vec2 model: {self.config.WAV2VEC2_MODEL}")
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"[AUDIO_SERVICE] Using device: {self.device}")
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config.WAV2VEC2_MODEL
        )
        self.model = Wav2Vec2Model.from_pretrained(self.config.WAV2VEC2_MODEL)
        
        # Optimize for MPS
        if self.device == "mps":
            self.model = self.model.half()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        
        print("[AUDIO_SERVICE] Wav2Vec2 model loaded successfully")
    
    def _chunk_audio(self, audio: np.ndarray) -> list:
        """Split audio into fixed-duration chunks."""
        samples_per_chunk = self.config.CHUNK_SECONDS * self.config.AUDIO_SR
        chunks = []
        
        start = 0
        n = len(audio)
        
        while start < n:
            end = min(start + samples_per_chunk, n)
            chunks.append(audio[start:end])
            start = end
        
        return chunks
    
    def extract(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract Wav2Vec2 embeddings from preprocessed audio.
        
        Returns (T, 768) embedding matrix.
        """
        if not self._loaded:
            self.load()
        
        try:
            chunks = self._chunk_audio(audio)
            all_embeddings = []
            
            for chunk in chunks:
                # Prepare input
                inputs = self.feature_extractor(
                    chunk,
                    sampling_rate=self.config.AUDIO_SR,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to device with appropriate dtype
                if self.device == "mps":
                    inp = inputs.input_values.to(self.device, dtype=torch.float16)
                else:
                    inp = inputs.input_values.to(self.device)
                
                # Extract embeddings
                with torch.inference_mode():
                    outputs = self.model(inp)
                
                hidden = outputs.last_hidden_state.squeeze(0)
                hidden = hidden[::2]  # Frame subsampling (matches training)
                
                all_embeddings.append(hidden.cpu().numpy())
            
            return np.vstack(all_embeddings)
        
        except Exception as e:
            print(f"[AUDIO_SERVICE] Wav2Vec2 extraction failed: {e}")
            return None


# =============================================================================
# PROSODY EXTRACTOR
# =============================================================================

class ProsodyExtractor:
    """
    Extracts prosodic features from audio.
    Features: F0, RMS, ZCR, spectral centroid/bandwidth/rolloff/contrast (13 dims).
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
    
    def extract(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract prosody features from preprocessed audio.
        
        Returns (T, 13) feature matrix.
        """
        try:
            sr = self.config.AUDIO_SR
            n_fft = self.config.WINDOW_LENGTH
            hop = self.config.HOP_LENGTH
            
            # RMS energy
            rms = librosa.feature.rms(
                y=audio, frame_length=n_fft, hop_length=hop, center=True
            )
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, frame_length=n_fft, hop_length=hop, center=True
            )
            
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop, center=True
            )
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop, center=True
            )
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop, 
                center=True, roll_percent=0.85
            )
            
            # Spectral contrast (7 bands)
            contrast = librosa.feature.spectral_contrast(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop, center=True
            )
            
            # Pitch (F0) using YIN
            f0 = librosa.yin(
                y=audio, fmin=50, fmax=400, sr=sr,
                frame_length=n_fft, hop_length=hop
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            f0 = f0[np.newaxis, :]
            
            # Align to same number of frames
            feature_list = [rms, zcr, centroid, bandwidth, rolloff, contrast, f0]
            min_frames = min(feat.shape[1] for feat in feature_list)
            aligned = [feat[:, :min_frames] for feat in feature_list]
            
            # Stack: (D, T) -> transpose to (T, D)
            stacked = np.vstack(aligned)
            return stacked.T
        
        except Exception as e:
            print(f"[AUDIO_SERVICE] Prosody extraction failed: {e}")
            return None


# =============================================================================
# FEATURE PROCESSOR
# =============================================================================

class FeatureProcessor:
    """
    Applies PCA dimensionality reduction and segment pooling.
    Matches the training pipeline exactly.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.pca_model = None
        self._loaded = False
    
    def load(self):
        """Load PCA model (call once at startup)."""
        if self._loaded:
            return
        
        if not os.path.exists(self.config.PCA_PATH):
            raise FileNotFoundError(f"PCA model not found: {self.config.PCA_PATH}")
        
        print(f"[AUDIO_SERVICE] Loading PCA model from: {self.config.PCA_PATH}")
        self.pca_model = joblib.load(self.config.PCA_PATH)
        print(f"[AUDIO_SERVICE] PCA loaded: {self.pca_model.n_components_} components")
        self._loaded = True
    
    def _compute_summary_stats(self, matrix: np.ndarray) -> np.ndarray:
        """Compute 4 statistics (mean, std, min, max) for each feature."""
        stats = [
            np.mean(matrix, axis=0),
            np.std(matrix, axis=0),
            np.min(matrix, axis=0),
            np.max(matrix, axis=0),
        ]
        pooled = np.concatenate(stats)
        return np.nan_to_num(pooled)
    
    def _segment_pool(self, matrix: np.ndarray, segments: int = 3) -> np.ndarray:
        """Apply pooling to each segment (beginning/middle/end)."""
        splits = np.array_split(matrix, segments)
        pooled_segments = [self._compute_summary_stats(seg) for seg in splits]
        return np.concatenate(pooled_segments)
    
    def _pool_embeddings(self, matrix: np.ndarray) -> np.ndarray:
        """Pool embeddings with optional segment pooling."""
        if self.config.USE_SEGMENT_POOLING:
            return self._segment_pool(matrix, segments=3)
        return self._compute_summary_stats(matrix)
    
    def process(
        self, 
        wav2vec_emb: np.ndarray, 
        prosody_emb: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Process embeddings through PCA and pooling.
        
        Returns final feature vector ready for Lasso model.
        """
        if not self._loaded:
            self.load()
        
        try:
            # Apply PCA to Wav2Vec2 (768 → 200)
            wav2vec_reduced = self.pca_model.transform(wav2vec_emb)
            
            # Apply segment pooling (3 segments × 4 stats)
            pooled_wav2vec = self._pool_embeddings(wav2vec_reduced)
            pooled_prosody = self._pool_embeddings(prosody_emb)
            
            # Concatenate
            feature_vector = np.concatenate([pooled_wav2vec, pooled_prosody])
            return feature_vector.reshape(1, -1)  # Shape for sklearn
        
        except Exception as e:
            print(f"[AUDIO_SERVICE] Feature processing failed: {e}")
            return None


# =============================================================================
# AUDIO SCORER
# =============================================================================

class AudioScorer:
    """
    Runs inference using the trained Lasso model.
    Applies scaler and feature selector before prediction.
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.model = None
        self.scaler = None
        self.selector = None
        self._loaded = False
    
    def load(self):
        """Load Lasso model, scaler, and selector (call once at startup)."""
        if self._loaded:
            return
        
        model_dir = self.config.LASSO_MODEL_DIR
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Lasso model directory not found: {model_dir}")
        
        print(f"[AUDIO_SERVICE] Loading Lasso model from: {model_dir}")
        
        self.model = joblib.load(os.path.join(model_dir, "lasso_model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self.selector = joblib.load(os.path.join(model_dir, "selector.joblib"))
        
        print(f"[AUDIO_SERVICE] Lasso model loaded (K={self.selector.k} features)")
        self._loaded = True
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        """
        Predict depression score from feature vector.
        
        Returns PHQ score prediction.
        """
        if not self._loaded:
            self.load()
        
        try:
            # Apply scaler
            X_scaled = self.scaler.transform(features)
            
            # Apply feature selection
            X_selected = self.selector.transform(X_scaled)
            
            # Predict
            prediction = self.model.predict(X_selected)
            return float(prediction[0])
        
        except Exception as e:
            print(f"[AUDIO_SERVICE] Prediction failed: {e}")
            return None


# =============================================================================
# MAIN SERVICE CLASS
# =============================================================================

class AudioInferenceService:
    """
    Main audio inference service.
    Coordinates all components for end-to-end audio scoring.
    
    Usage:
        service = AudioInferenceService()
        service.load_models()  # Call once at startup
        score = service.predict(wav_path)
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(self.config)
        self.wav2vec_extractor = Wav2Vec2Extractor(self.config)
        self.prosody_extractor = ProsodyExtractor(self.config)
        self.feature_processor = FeatureProcessor(self.config)
        self.scorer = AudioScorer(self.config)
        
        self._loaded = False
    
    def load_models(self):
        """Load all models. Call once at server startup."""
        if self._loaded:
            return
        
        print("[AUDIO_SERVICE] Loading all audio models...")
        
        try:
            self.wav2vec_extractor.load()
            self.feature_processor.load()
            self.scorer.load()
            self._loaded = True
            print("[AUDIO_SERVICE] All audio models loaded successfully!")
        except Exception as e:
            print(f"[AUDIO_SERVICE] Failed to load models: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._loaded
    
    def predict(self, wav_path: str) -> Optional[float]:
        """
        Run full inference pipeline on audio file.
        
        Parameters
        ----------
        wav_path : str
            Path to the merged user audio WAV file.
        
        Returns
        -------
        float or None
            Predicted PHQ score, or None if prediction fails.
        """
        if not self._loaded:
            print("[AUDIO_SERVICE] WARNING: Models not loaded. Call load_models() first.")
            return None
        
        try:
            # Step 1: Preprocess audio
            audio = self.preprocessor.process(wav_path)
            
            if len(audio) == 0:
                print("[AUDIO_SERVICE] WARNING: Empty audio after preprocessing")
                return None
            
            # Step 2: Extract Wav2Vec2 embeddings
            wav2vec_emb = self.wav2vec_extractor.extract(audio)
            if wav2vec_emb is None:
                return None
            
            # Step 3: Extract prosody features
            prosody_emb = self.prosody_extractor.extract(audio)
            if prosody_emb is None:
                return None
            
            # Step 4: Process features (PCA + pooling)
            features = self.feature_processor.process(wav2vec_emb, prosody_emb)
            if features is None:
                return None
            
            # Step 5: Predict and clamp to valid PHQ range
            raw_score = self.scorer.predict(features)
            
            if raw_score is not None:
                # Clamp to valid PHQ-8 range [0, 24]
                score = max(0.0, min(24.0, raw_score))
                if raw_score != score:
                    print(f"[AUDIO_SERVICE] Raw prediction {raw_score:.4f} clamped to {score:.4f}")
                else:
                    print(f"[AUDIO_SERVICE] Audio prediction: {score:.4f}")
                return score
            
            return None
        
        except Exception as e:
            print(f"[AUDIO_SERVICE] Inference failed: {e}")
            return None


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    service = AudioInferenceService()
    service.load_models()
    print("Audio Inference Service loaded successfully.")

