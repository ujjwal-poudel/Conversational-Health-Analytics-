"""
stt_service.py

Speech-to-Text (STT) module using the Whisper model.

Purpose
-------
This module handles audio transcription for user responses. It loads the
Whisper model once during server startup and exposes a function that saves raw
audio, transcribes it, and returns both the saved audio path and the text
transcript with timestamps.

Key Features
------------
1. Whisper model is loaded once at import time (efficient for multiple users).
2. Audio files are saved for later feature extraction in the regression model.
3. Word-level timestamps are captured for detailed analysis.
4. Errors during transcription or file handling are caught and logged.
5. Uses centralized configuration from config.py
"""

import os
import uuid
import whisper
from typing import Tuple, Dict, Any
from app.audio.config import AudioConfig
import warnings

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Note: MPS (Mac GPU) doesn't support Whisper's sparse tensors, so we use CPU
# For faster transcription, use WHISPER_MODEL_SIZE=tiny in your .env file

# Load Whisper model once at module import time.
try:
    WHISPER_MODEL = whisper.load_model(AudioConfig.WHISPER_MODEL_SIZE)
    print(f"[STT] Whisper '{AudioConfig.WHISPER_MODEL_SIZE}' model loaded successfully")
except Exception as error:
    raise RuntimeError(f"Failed to load Whisper model: {error}")


def transcribe_user_audio(audio_bytes: bytes, output_dir: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Transcribe user audio, save it, and return metadata.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file content (may be WebM, WAV, or other formats).
    output_dir : str
        Directory to save the audio file.

    Returns
    -------
    tuple of (str, str, dict)
        transcript : str
            The transcribed text from the audio.
        file_path : str
            Absolute path to the saved audio file (converted to WAV).
        timestamps : dict
            Timing information from Whisper transcription.

    Raises
    ------
    RuntimeError
        If transcription fails or audio saving encounters errors.
    """
    try:
        from pydub import AudioSegment
        import io
        
        # Generate unique filename with session ID prefix
        filename = f"{os.path.basename(output_dir)}_{uuid.uuid4()}.wav"
        file_path = os.path.join(output_dir, filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert audio to proper WAV format (frontend often sends WebM)
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio.export(file_path, format="wav")
        except Exception as e:
            # Fallback: save raw bytes if conversion fails
            print(f"[STT] Warning: Audio conversion failed, saving raw bytes: {e}")
            with open(file_path, "wb") as f:
                f.write(audio_bytes)
    except Exception as error:
        raise RuntimeError(f"Failed to save user audio: {error}")

    # Perform transcription with word timestamps
    try:
        # word_timestamps=True enables word-level timing information
        result = WHISPER_MODEL.transcribe(file_path, word_timestamps=True)
        transcript = result.get("text", "").strip()
        
        # Extract timestamp information
        timestamps = {
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown")
        }
        
        print(f"[STT] Transcribed: {transcript[:50]}...")
        
    except Exception as error:
        raise RuntimeError(f"Whisper transcription failed: {error}")

    return transcript, file_path, timestamps