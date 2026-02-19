"""
Template Manager

This module loads and serves all template-based text resources used by
the conversation system, including:

- Topic questions (primary and secondary)
- Ending message
- Keyword lists for topic detection
"""

import json
import random
from pathlib import Path
from typing import List, Optional


class TemplateManager:
    """
    Central loader for all JSON template resources.
    Provides clean helper functions for accessing specific templates.
    """

    def __init__(self, base_path: str):
        """
        Initializes the manager.

        base_path:
            Root directory containing the 'data' folder.

        Expected directory layout:
            data/templates/topics.json
            data/keywords/topic_keywords.json
        """

        base = Path(base_path)

        # Template file paths
        self.topics_path = base / "templates" / "topics.json"

        # Keyword path
        self.keywords_path = base / "keywords" / "topic_keywords.json"

        # Load JSON files
        self.topics_data = self._load_json(self.topics_path, default={})

        # Load topic keywords for dynamic topic detection
        self.keywords_data = self._load_json(self.keywords_path, default={})

        # Create attribute expected by ConversationEngine and VagueDetector
        self.topic_keywords = self.keywords_data

        # List of topics in order defined in topics.json
        self.topics_list = list(self.topics_data.keys())

    def _load_json(self, path: Path, default):
        """
        Safely loads a JSON file.
        If the file is missing or invalid, returns the default value.
        """
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return default

    """
    Topic questions
    """

    def get_topic_list(self) -> List[str]:
        """
        Returns the list of topics in the order defined in topics.json.
        """
        return self.topics_list

    def get_primary_question(self, topic: str) -> str:
        """
        Returns the primary question for a topic.
        """
        entry = self.topics_data.get(topic, {})
        primary = entry.get("primary", [])
        if primary:
            return primary[0]
        return "Could you tell me about this?"

    def get_secondary_question(self, topic: str) -> str:
        """
        Returns a random secondary question for a topic.
        """
        entry = self.topics_data.get(topic, {})
        secondary = entry.get("secondary", [])
        if secondary:
            return random.choice(secondary)
        return "Can you tell me a bit more about that?"

    def get_secondary_questions(self, topic: str) -> List[str]:
        """
        Returns all secondary questions for a topic.
        """
        entry = self.topics_data.get(topic, {})
        return entry.get("secondary", [])

    def get_random_secondary_questions(self, topic: str, count: int = 2) -> List[str]:
        """
        Returns up to `count` randomly selected (non-repeating) secondary questions.
        """
        all_questions = self.get_secondary_questions(topic)
        if not all_questions:
            return ["Can you tell me a bit more about that?"]
        return random.sample(all_questions, min(count, len(all_questions)))

    """
    Topic keywords for topic detection
    """

    def get_topic_keywords(self, topic: str) -> List[str]:
        """
        Returns the predefined keywords for a given topic.
        Used by the topic detector in the controller.
        """
        return self.keywords_data.get(topic, [])

    """
    Ending
    """

    def get_conversation_end(self) -> str:
        """
        Returns the ending message.
        """
        return "Thanks for talking with me today. I really appreciate you sharing."
