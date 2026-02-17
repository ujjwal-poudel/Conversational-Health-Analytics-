"""
config.py

Audio Configuration

Centralizes all audio-related configuration using environment variables.
"""

import os

# Suppress verbose warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv

load_dotenv()


class AudioConfig:
    """Configuration for audio services (STT, TTS, paths)."""
    
    # Piper TTS - Use system piper command from venv (installed via pip)
    # In Docker/production, piper will be in /opt/venv/bin/piper
    # In local dev, it will be in .newvenv/bin/piper or system PATH
    PIPER_EXECUTABLE = os.getenv("PIPER_EXECUTABLE_PATH", "piper")  # Use system PATH by default
    PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "models/piper/en_US-lessac-medium.onnx")
    
    # Whisper STT
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
    
    # Audio Output Directories
    # Session-based: audio_data/<session_id>/...
    AUDIO_BASE_DIR = "audio_data"
    
    # Bot TTS audio (temporary, cleaned up per-session)
    BOT_AUDIO_OUTPUT_DIR = os.path.join(AUDIO_BASE_DIR, "questions/")
    
    os.makedirs(BOT_AUDIO_OUTPUT_DIR, exist_ok=True)

    # Add FFmpeg to PATH (for Whisper)
    # We add the local bin directory where we installed ffmpeg
    _ffmpeg_dir = os.path.join(os.getcwd(), "app/audio/bin")
    if os.path.exists(_ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + _ffmpeg_dir
