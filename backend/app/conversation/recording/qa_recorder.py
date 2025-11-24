"""
QA Recorder

This module records all messages exchanged during the conversation.
It keeps two separate logs:

1. full_transcript:
   Every message from the bot and the user, including the intro phase.
   This is useful for debugging or for showing the conversation later.

2. qa_sequence:
   A list of alternating questions and answers from the content phase only.
   This list is used as input to the regression model. The intro phase
   is never included in this list.

The recorder is designed to be controlled by the conversation controller,
which specifies whether a message should be included in inference or not.
"""

from typing import List, Dict


class QARecorder:
    """
    Handles storage of conversation transcripts and Q/A for inference.
    """

    def __init__(self):
        """
        Initializes empty storage for transcripts and Q/A sequences.
        """
        self.full_transcript: List[Dict[str, str]] = []
        self.qa_sequence: List[str] = []

    def record_bot_message(self, text: str, include_in_inference: bool):
        """
        Records a bot message.

        text:
            The bot's message in plain text.
        include_in_inference:
            If True, the message is treated as a question and added to
            the qa_sequence list.
        """
        self.full_transcript.append({"sender": "bot", "text": text})

        if include_in_inference:
            self.qa_sequence.append(text)

    def record_user_message(self, text: str, include_in_inference: bool):
        """
        Records a user message.

        text:
            The user's message.
        include_in_inference:
            If True, the message is treated as an answer and added to
            the qa_sequence list.
        """
        self.full_transcript.append({"sender": "user", "text": text})

        if include_in_inference:
            self.qa_sequence.append(text)

    def get_qa_sequence(self) -> List[str]:
        """
        Returns the list of alternating questions and answers.
        """
        return self.qa_sequence

    def get_full_transcript(self) -> List[Dict[str, str]]:
        """
        Returns the complete conversation transcript.
        """
        return self.full_transcript

    def reset(self):
        """
        Clears both transcripts. Useful when starting a new conversation.
        """
        self.full_transcript = []
        self.qa_sequence = []
