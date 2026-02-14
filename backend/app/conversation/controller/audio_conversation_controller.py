"""
audio_conversation_controller.py

This module defines the controller responsible for handling audio-based
conversation interactions at the application level.

It serves as the interface between the API layer (FastAPI router) and the
audio conversation orchestration layer. The controller does not contain
any business logic for transcription, conversation progression, or speech
synthesis. Instead, it delegates those responsibilities to the
AudioConversationOrchestrator.

Responsibilities
----------------
1. Receive raw audio input from the API layer.
2. Delegate audio processing and conversation flow to the orchestrator.
3. Return a structured result containing:
       - Transcript of the user's audio
       - Saved user audio file path
       - Next question text
       - Next question audio file path
4. Ensure each session ID maintains its own conversation state.

This modular separation ensures:
- Cleaner API routes
- Better maintainability
- Easier debugging and testing
- Minimal duplication of logic
"""

from typing import Dict

from app.conversation.audio_flow.audio_conversation_orchestrator import (
    AudioConversationOrchestrator,
)


class AudioConversationController:
    """
    High-level controller for managing audio conversation turns.

    This class acts as the entry point for the API router. It manages:
        - Session binding
        - Instantiation of the orchestrator
        - Delegation of audio processing tasks

    No transcription, TTS generation, or LLM logic is implemented here.
    The controller's sole role is request/response handling at the
    application level.
    """

    def __init__(self, session_id: str, templates_base_path: str):
        """
        Initialize the audio conversation controller.

        Parameters
        ----------
        session_id : str
            Unique identifier used to maintain conversation state and
            correctly store user audio, transcript, and JSONL turn data.
        templates_base_path : str
            Path to the conversation templates directory.
        """
        self.session_id = session_id
        self.orchestrator = AudioConversationOrchestrator(session_id, templates_base_path)

    async def start_conversation(self) -> Dict:
        """
        Start a new audio conversation.

        Returns
        -------
        dict
            A structured response containing:
            - "response_text": list of str
            - "response_audio_paths": list of str
            - "is_finished": bool
        """
        return await self.orchestrator.start_conversation()

    async def process_audio_turn(self, wav_bytes: bytes) -> Dict:
        """
        Handle a single conversational turn based on audio input.

        This method delegates the actual processing to the orchestrator:
            1. Audio transcription via Whisper (STT)
            2. Conversation state update + next question generation
            3. Text-to-speech synthesis of next question via Piper (TTS)

        Parameters
        ----------
        wav_bytes : bytes
            Raw WAV audio data provided by the client.

        Returns
        -------
        dict
            A structured response containing:
            - "transcript": str
            - "timestamps": dict
            - "user_audio_path": str
            - "response_text": list of str
            - "response_audio_paths": list of str
            - "is_finished": bool
        """
        return await self.orchestrator.process_turn(wav_bytes)

    # DISABLED: No longer saving audio files to disk
    # def finalize_conversation(self, conversation_id: str) -> Dict[str, str]:
    #     """
    #     Finalize the conversation by merging all audio files.

    #     Parameters
    #     ----------
    #     conversation_id : str
    #         ID from user_data.jsonl to use in audio filenames.

    #     Returns
    #     -------
    #     dict
    #         {
    #             "user_only_path": str,
    #             "full_conversation_path": str
    #         }
    #     """
    #     return self.orchestrator.finalize_conversation(conversation_id)

    def get_inference_pairs(self):
        """
        Get the conversation transcript for regression model inference.

        Returns
        -------
        list
            Alternating list of [question, answer, question, answer...]
        """
        return self.orchestrator.conversation.get_inference_pairs()

    def cleanup_session(self):
        """
        Clean up the session audio directory.
        """
        self.orchestrator.cleanup_session()