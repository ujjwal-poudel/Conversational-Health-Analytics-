"""
Template Manager

This module loads and serves all template-based text resources used by
the conversation system, including:

- Introduction text
- Topic questions
- Follow-up templates
- Filler responses
- Transition lines
- Ending message
- Keyword lists for topic detection
"""

import json
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
            data/templates/intro.json
            data/templates/topics.json
            data/templates/followups.json
            data/templates/transitions.json
            data/templates/end.json
            data/fillers/filler_responses.json
            data/keywords/topic_keywords.json
        """

        base = Path(base_path)

        # Template file paths
        self.intro_path = base / "templates" / "intro.json"
        self.topics_path = base / "templates" / "topics.json"
        self.followups_path = base / "templates" / "followups.json"
        self.transitions_path = base / "templates" / "transitions.json"
        self.end_path = base / "templates" / "end.json"

        # Filler file path
        self.fillers_path = base / "fillers" / "filler_responses.json"

        # Keyword path
        self.keywords_path = base / "keywords" / "topic_keywords.json"

        # Load all JSON files
        self.intro_data = self._load_json(self.intro_path, default={})
        self.topics_data = self._load_json(self.topics_path, default={})
        self.followups_data = self._load_json(self.followups_path, default={})
        self.fillers_data = self._load_json(self.fillers_path, default={})
        self.transitions_data = self._load_json(
            self.transitions_path,
            default={"default_transition": "Let's talk about something else."}
        )
        self.end_data = self._load_json(
            self.end_path,
            default={"end": "Thanks for talking with me."}
        )

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
    Introduction
    """

    def get_intro(self) -> str:
        """
        Returns a simple introduction text.

        If an 'intro' field exists in intro.json, it is used.
        Otherwise, a basic default is returned.
        """
        # If you ever add a flat "intro" field, this will prefer it
        if "intro" in self.intro_data:
            return self.intro_data.get(
                "intro",
                "Hello, I’m here to talk with you today."
            )

        # Fallback default if using segment-based intro only
        return "Hello, I’m here to talk with you today."
    
    def get_intro_segments(self):
        """
        Returns the structured intro segments as defined in intro.json.

        Expected structure in intro.json:
            {
                "greeting": [...],
                "purpose": [...],
                "comfort": [...],
                "disclosure": [...],
                "transition": [...]
            }

        Each value should be a list of alternative phrasings. If a key
        is missing, an empty list is returned for that segment.
        """
        return {
            "greeting": self.intro_data.get("greeting", []),
            "purpose": self.intro_data.get("purpose", []),
            "comfort": self.intro_data.get("comfort", []),
            "disclosure": self.intro_data.get("disclosure", []),
            "transition": self.intro_data.get("transition", []),
        }

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
        Returns the secondary question for a topic.
        """
        entry = self.topics_data.get(topic, {})
        secondary = entry.get("secondary", [])
        if secondary:
            return secondary[0]
        return "Can you tell me a bit more about that?"

    """
    Follow-ups
    """

    def get_followup_template(self, topic: str) -> Optional[str]:
        """
        Returns a topic-based follow-up template.
        """
        entry = self.followups_data.get(topic, [])
        if entry:
            return entry[0]
        return None

    def get_safe_generic_followup(self, topic: str) -> str:
        """
        Returns a safe generic follow-up if semantic follow-up fails.
        """
        entry = self.followups_data.get(topic, [])
        if entry:
            return entry[0]
        return "Could you tell me a little more?"

    """
    Fillers
    """

    def get_filler_for_intent(self, intent: str) -> Optional[str]:
        """
        Returns a filler response for the given intent category.
        """
        entry = self.fillers_data.get(intent, [])
        if entry:
            return entry[0]
        return None

    """
    Topic keywords for topic detection
    """

    def get_topic_keywords(self, topic: str) -> List[str]:
        """
        Returns the predefined keywords for a given topic.
        Used by the topic detector in the controller.

        Example of expected structure in topic_keywords.json:
            {
                "sleep": ["sleep", "insomnia", "wake up", "night"],
                "energy": ["tired", "fatigue", "exhausted"],
                ...
            }
        """
        return self.keywords_data.get(topic, [])

    """
    Transitions and ending
    """

    def get_transition(self) -> str:
        """
        Returns a transition line.
        """
        return self.transitions_data.get(
            "default_transition",
            "Let's talk about something else."
        )

    def get_conversation_end(self) -> str:
        """
        Returns the ending message.
        """
        return self.end_data.get("end", "Thanks for talking with me.")
