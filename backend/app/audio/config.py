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
    
    # Piper TTS - paths relative to backend/ directory where uvicorn runs
    # Using venv installation as requested
    PIPER_EXECUTABLE = os.getenv("PIPER_EXECUTABLE_PATH", "../.newvenv/bin/piper")
    PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "app/audio/piper_models/en_US-lessac-medium.onnx")
    
    # Whisper STT
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
    
    # Audio Output Directories
    # We now use a cleaner structure: audio_data/<session_id>/...
    AUDIO_BASE_DIR = "audio_data"
    
    # Deprecated specific paths (kept for safety but should use session dirs)
    USER_AUDIO_OUTPUT_DIR = os.path.join(AUDIO_BASE_DIR, "users/")
    BOT_AUDIO_OUTPUT_DIR = os.path.join(AUDIO_BASE_DIR, "questions/")
    
    # Merged audio output (can be external drive)
    # Merged audio output (can be external drive)
    MERGED_AUDIO_OUTPUT_DIR = os.getenv("MERGED_AUDIO_OUTPUT_DIR", "/Volumes/MACBACKUP/conversation_audio/")
    
    try:
        # Attempt to create the directory (will fail if drive doesn't exist)
        os.makedirs(MERGED_AUDIO_OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        print(f"[CONFIG] Warning: Could not create external audio path '{MERGED_AUDIO_OUTPUT_DIR}': {e}")
        # Fallback to local project directory
        MERGED_AUDIO_OUTPUT_DIR = os.path.join(AUDIO_BASE_DIR, "merged/")
        os.makedirs(MERGED_AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # Ensure directories exist
    os.makedirs(USER_AUDIO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BOT_AUDIO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_AUDIO_OUTPUT_DIR, exist_ok=True)
    
    # User data JSONL path - relative to backend/ directory
    USER_DATA_JSONL = os.getenv("USER_DATA_JSONL_PATH", "data/user_data.jsonl")

    # Add FFmpeg to PATH (for Whisper)
    # We add the local bin directory where we installed ffmpeg
    _ffmpeg_dir = os.path.join(os.getcwd(), "app/audio/bin")
    if os.path.exists(_ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + _ffmpeg_dir
