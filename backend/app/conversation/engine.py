"""
Conversation Engine

This module wires together:
- Template loading
- LLM rewriting
- Topic detection
- Question sequencing
- Filler responses
- Vague detection
- Sufficiency logic
- Q/A recording

The ConversationEngine exposes:
    start()        -> intro + first question
    process(text)  -> next bot response
    is_finished()  -> True/False
    get_inference_pairs() -> Q/A list
"""

from typing import Optional

from .data.template_manager import TemplateManager
from .generation.llm_rewriter import LLMRewriter
from .generation.llm_client import LLMClient

from .analysis.vague_detector import VagueDetector
from .analysis.sufficiency_checker import SufficiencyChecker

from .recording.qa_recorder import QARecorder
from .controller.conversation_controller import ConversationController

import random

class ConversationEngine:
    """
    High-level wrapper around the entire conversation system.
    """

    def __init__(self, templates_base_path: str, model_path: Optional[str] = None):
        """
        Initializes all components.

        templates_base_path:
            Folder containing the JSON template files.

        model_path:
            Full path to the GGUF model file.
        """

        # Load all template resources
        self.template_manager = TemplateManager(templates_base_path)

        # Extract topic keywords from templates (required for vague detection)
        topic_keywords = self.template_manager.topic_keywords

        # Initialize LLM + rewriter
        self.llm_client = LLMClient(model_path=model_path)      
        self.llm_rewriter = LLMRewriter(self.llm_client)

        # Initialize vague + sufficiency systems with topic keywords
        self.vague_detector = VagueDetector(topic_keywords)
        self.sufficiency_checker = SufficiencyChecker(self.vague_detector)

        # Recorder for inference
        self.qa_recorder = QARecorder()

        # Main conversation controller
        self.controller = ConversationController(
            template_manager=self.template_manager,
            llm_rewriter=self.llm_rewriter,
            qa_recorder=self.qa_recorder,
            sufficiency_checker=self.sufficiency_checker,
            vague_detector=self.vague_detector
        )

        self.finished = False

    def start(self) -> str:
        """
        Starts the conversation by immediately beginning the first topic.
        No intro is generated here because the frontend handles it.
        """
        first_question = self.controller.start_first_topic()

        if first_question is None:
            self.finished = True
            return self.template_manager.get_conversation_end()

        return first_question

    def process(self, user_text: str) -> str:
        """
        Processes a user message and returns the next bot turn.
        """
        if self.finished:
            return self.template_manager.get_conversation_end()

        bot_message = self.controller.handle_user_answer(user_text)

        if self.controller.get_current_topic() is None:
            self.finished = True

        return bot_message

    def is_finished(self) -> bool:
        """
        Returns True if conversation is complete.
        """
        return self.finished

    def get_inference_pairs(self):
        """
        Returns alternating list of [question, answer, question, answer...]
        """
        return self.qa_recorder.get_qa_sequence()