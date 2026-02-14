"""
audio_conversation_orchestrator.py

This module defines the orchestration layer for the audio-based conversation flow.
It serves as a mediator between the following components:

1. Speech-to-Text (STT) Layer
   - Converts user audio input into text using Whisper.

2. Conversation Engine Layer
   - Uses the existing text-based ConversationController for:
       - Turn management
       - Conversation state progression
       - LLM question generation
       - JSONL turn storage

3. Text-to-Speech (TTS) Layer
   - Converts generated text questions into spoken audio using Piper.

Purpose
-------
This orchestrator isolates the core conversational logic from the API layer and
keeps audio processing responsibilities modular. Each step in the audio
conversation loop is clearly separated: transcription, conversation update,
question generation, and audio synthesis.

This modular approach improves testability, maintainability, and scalability
by ensuring that each component performs one well-defined task.
"""

from typing import Tuple, Dict, List, Union, Any

from app.conversation.engine import ConversationEngine
from app.audio.stt_service import transcribe_user_audio
from app.audio.tts_service import synthesize_question_audio


class AudioConversationOrchestrator:
    """
    Orchestrates the end-to-end flow of the audio conversation.

    The orchestrator does not handle HTTP requests, file uploads, or routing.
    It strictly coordinates:
        - STT processing
        - Conversation state updates
        - LLM-based question generation
        - TTS generation
        - Audio file tracking for end-of-conversation merging

    This class ensures that the audio conversation flow remains modular
    and that the existing text conversation system is reused without modification.
    """

    def __init__(self, session_id: str, templates_base_path: str):
        """
        Initialize a new orchestrator tied to a specific session.

        Parameters
        ----------
        session_id : str
            Unique identifier for the conversation session.
        templates_base_path : str
            Path to the conversation templates directory.
        """
        self.session_id = session_id
        self.conversation = ConversationEngine(templates_base_path=templates_base_path)
        
        # Track audio files for merging at conversation end
        self.user_audio_paths: List[str] = []
        self.bot_audio_paths: List[str] = []
        self.interleaved_audio_paths: List[str] = []  # [Q1, A1, Q2, A2, ...]
        
        # Track timestamps from user responses
        self.transcription_timestamps: List[Dict[str, Any]] = []
        
        # Create session-specific audio directory
        from app.audio.config import AudioConfig
        import os
        self.session_audio_dir = os.path.join(AudioConfig.AUDIO_BASE_DIR, session_id)
        os.makedirs(self.session_audio_dir, exist_ok=True)

    def transcribe_user_audio(self, wav_bytes: bytes) -> Tuple[str, str, Dict[str, Any]]:
        """
        Convert raw user audio bytes into a transcript with timestamps using Whisper.
        """
        # Transcribe user audio (no longer saves to disk)
        transcript, _, timestamps = transcribe_user_audio(
            audio_bytes=wav_bytes,
            output_dir=self.session_audio_dir  # Ignored, kept for compatibility
        )
        
        # No longer track audio paths (not saved)
        # self.user_audio_paths.append(user_audio_path)
        # self.transcription_timestamps.append(timestamps)
        
        return None, transcript, timestamps  # user_audio_path is None

    def update_conversation_with_transcript(self, transcript: str) -> List[str]:
        """
        Update the conversation state with the user's transcribed answer
        and request the next question from the conversation engine.
        """
        bot_response = self.conversation.process(transcript)
        
        # Ensure response is always a list
        if isinstance(bot_response, str):
            bot_response = [bot_response]
        
        return bot_response

    async def synthesize_question_audio(self, question_text: str) -> str:
        """
        Convert a question string into audio using Piper TTS.
        """
        # Save to session directory
        return await synthesize_question_audio(question_text, output_dir=self.session_audio_dir)

    async def start_conversation(self) -> Dict[str, Union[str, List[str], List[str]]]:
        """
        Start a new audio conversation.
        """
        first_message = self.conversation.start()
        
        # Ensure response is always a list
        if isinstance(first_message, str):
            first_message = [first_message]
        
        # Synthesize each message part into audio (paths still returned for playback)
        audio_paths = []
        for text in first_message:
            audio_path = await self.synthesize_question_audio(text)
            audio_paths.append(audio_path)
            # No longer track for merging
            # self.bot_audio_paths.append(audio_path)
            # self.interleaved_audio_paths.append(audio_path)
        
        return {
            "response_text": first_message,
            "response_audio_paths": audio_paths,
            "is_finished": False
        }

    async def process_turn(self, wav_bytes: bytes) -> Dict[str, Union[str, List[str], List[str], bool, Dict[str, Any]]]:
        """
        Execute the complete audio turn-processing pipeline.
        """
        # Step 1: STT with timestamps (user audio not saved)
        user_audio_path, transcript, timestamps = self.transcribe_user_audio(wav_bytes)
        
        # No longer track audio paths (not saved)
        # self.interleaved_audio_paths.append(user_audio_path)

        # Step 2: Conversation logic (returns list of message parts)
        bot_response_text = self.update_conversation_with_transcript(transcript)

        # Step 3: TTS - synthesize each message part
        audio_paths = []
        for text in bot_response_text:
            audio_path = await self.synthesize_question_audio(text)
            audio_paths.append(audio_path)
            # Still track bot audio for cleanup, but don't merge
            # self.bot_audio_paths.append(audio_path)
            # self.interleaved_audio_paths.append(audio_path)

        return {
            "transcript": transcript,
            "timestamps": timestamps,
            "user_audio_path": None,  # No longer saved
            "response_text": bot_response_text,
            "response_audio_paths": audio_paths,
            "is_finished": self.conversation.is_finished()
        }

    # DISABLED: No longer saving audio files to disk
    # def finalize_conversation(self, conversation_id: str) -> Dict[str, str]:
    #     """
    #     Merge and save audio files when conversation is finished.
    #     """
    #     from app.audio.audio_merger import create_user_only_audio, create_full_conversation_audio
    #     from app.audio.config import AudioConfig
    #     import shutil
    #     import os

    #     print(f"[ORCHESTRATOR] Finalizing conversation {conversation_id}")
        
    #     try:
    #         # Create user-only merged audio
    #         user_only_path = create_user_only_audio(
    #             self.user_audio_paths,
    #             AudioConfig.MERGED_AUDIO_OUTPUT_DIR,
    #             conversation_id
    #         )

    #         # Create full conversation merged audio
    #         full_conversation_path = create_full_conversation_audio(
    #             self.interleaved_audio_paths,
    #             AudioConfig.MERGED_AUDIO_OUTPUT_DIR,
    #             conversation_id
    #         )

    #         print(f"[ORCHESTRATOR] User-only audio saved: {user_only_path}")
    #         # Note: Full conversation path is already logged by audio_merger
            
    #         # Note: We do NOT delete the session directory here anymore.
    #         # The frontend needs the final audio files to play them.
    #         # Cleanup should be handled separately (e.g., via a specific endpoint or periodic task).
            
    #         return {
    #             "user_only_path": user_only_path,
    #             "full_conversation_path": full_conversation_path
    #         }

    #     except Exception as error:
    #         raise RuntimeError(f"Failed to finalize conversation audio: {error}")

    def cleanup_session(self):
        """
        Delete the temporary session audio directory.
        """
        import os
        if os.path.exists(self.session_audio_dir):
            import shutil
            shutil.rmtree(self.session_audio_dir)
            print(f"[ORCHESTRATOR] Deleted session audio directory: {self.session_audio_dir}")