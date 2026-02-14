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
from faster_whisper import WhisperModel
from typing import Tuple, Dict, Any, List
from app.audio.config import AudioConfig
import warnings

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Note: faster-whisper is optimized for CPU with int8 quantization
# For faster transcription, use WHISPER_MODEL_SIZE=tiny in your .env file

# Load Whisper model once at module import time.
try:
    # faster-whisper uses int8 quantization for fast CPU inference
    WHISPER_MODEL = WhisperModel(
        AudioConfig.WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8"
    )
    print(f"[STT] faster-whisper '{AudioConfig.WHISPER_MODEL_SIZE}' model loaded successfully")
except Exception as error:
    raise RuntimeError(f"Failed to load faster-whisper model: {error}")


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
    tuple of (str, None, dict)
        transcript : str
            The transcribed text from the audio.
        file_path : None
            Always None (audio not saved to disk).
        timestamps : dict
            Timing information from Whisper transcription.

    Raises
    ------
    RuntimeError
        If transcription fails.
    """
    try:
        from pydub import AudioSegment
        import io
        import tempfile
        
        # Convert audio to proper WAV format in memory
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Export to temporary WAV file (Whisper requires file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = temp_wav.name
            audio.export(temp_path, format="wav")
        
    except Exception as error:
        raise RuntimeError(f"Failed to process audio: {error}")

    # Perform transcription with word timestamps
    try:
        # faster-whisper returns a generator (segments, info)
        segments_gen, info = WHISPER_MODEL.transcribe(
            temp_path,
            word_timestamps=True
        )
        
        # Convert generator to list and build transcript
        segments_list = list(segments_gen)
        transcript = " ".join([seg.text for seg in segments_list]).strip()
        
        # Extract timestamp information (convert to dict format for compatibility)
        timestamps = {
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in (seg.words or [])
                    ] if seg.words else []
                }
                for seg in segments_list
            ],
            "language": info.language
        }
        
        print(f"[STT] Transcribed: {transcript[:50]}...")
        
    except Exception as error:
        raise RuntimeError(f"faster-whisper transcription failed: {error}")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

    return transcript, None, timestamps  # Don't save audio, return None for path