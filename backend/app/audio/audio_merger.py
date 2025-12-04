"""
audio_merger.py

Audio Merging Service

Purpose
-------
This module handles merging multiple audio files into single WAV files.
It is used to create:
1. User-only audio: All user responses concatenated
2. Full conversation audio: Bot questions + user answers in correct order

Key Features
------------
1. Uses pydub for audio concatenation
2. Maintains consistent audio format (16kHz, mono)
3. Saves to configurable output directories
4. Generates unique filenames based on user_data.jsonl ID
"""

import os
from typing import List
from pydub import AudioSegment


def merge_audio_files(audio_paths: List[str], output_path: str) -> str:
    """
    Merge multiple audio files into a single WAV file.

    Parameters
    ----------
    audio_paths : list of str
        List of paths to audio files to merge (in order).
    output_path : str
        Path where the merged audio file will be saved.

    Returns
    -------
    str
        Path to the merged audio file.

    Raises
    ------
    RuntimeError
        If merging fails or file writing encounters errors.
    """
    if not audio_paths:
        raise ValueError("No audio files provided for merging")

    # Start with empty audio segment
    merged_audio = AudioSegment.empty()
    valid_files_count = 0

    for audio_path in audio_paths:
        if not os.path.exists(audio_path):
            print(f"[AUDIO_MERGER] Warning: File not found: {audio_path}")
            continue

        try:
            # Attempt to load the audio file
            segment = AudioSegment.from_wav(audio_path)
            merged_audio += segment
            valid_files_count += 1
        except Exception as e:
            print(f"[AUDIO_MERGER] Warning: Skipping corrupted/invalid file {audio_path}: {e}")
            # Continue to next file instead of failing the whole process
            continue

    if valid_files_count == 0:
        raise RuntimeError("No valid audio files could be loaded for merging")

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export merged audio as WAV
        merged_audio.export(output_path, format="wav")

        print(f"[AUDIO_MERGER] Successfully merged {valid_files_count}/{len(audio_paths)} files to: {output_path}")
        return output_path

    except Exception as error:
        raise RuntimeError(f"Failed to export merged audio: {error}")


def create_user_only_audio(user_audio_paths: List[str], output_dir: str, conversation_id: str) -> str:
    """
    Create merged audio file containing only user responses.

    Parameters
    ----------
    user_audio_paths : list of str
        Paths to all user audio files from the conversation.
    output_dir : str
        Directory where the merged file will be saved.
    conversation_id : str
        ID from user_data.jsonl to use in filename.

    Returns
    -------
    str
        Path to the merged user-only audio file.
    """
    filename = f"{conversation_id}_user_only.wav"
    output_path = os.path.join(output_dir, filename)
    return merge_audio_files(user_audio_paths, output_path)


def create_full_conversation_audio(
    interleaved_paths: List[str], 
    output_dir: str, 
    conversation_id: str
) -> str:
    """
    Create merged audio file with bot questions and user answers alternating.

    Parameters
    ----------
    interleaved_paths : list of str
        Paths to audio files in order: [Q1, A1, Q2, A2, ...]
    output_dir : str
        Directory where the merged file will be saved.
    conversation_id : str
        ID from user_data.jsonl to use in filename.

    Returns
    -------
    str
        Path to the merged full conversation audio file.
    """
    filename = f"{conversation_id}_full_conversation.wav"
    output_path = os.path.join(output_dir, filename)
    return merge_audio_files(interleaved_paths, output_path)
