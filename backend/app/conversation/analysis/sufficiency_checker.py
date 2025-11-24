"""
Sufficiency Checker

Evaluates whether a conversation topic has reached enough depth to move on.

Rules (as agreed):

1. A topic needs at least 25 total non-vague words.
2. Vague answers contribute 0 words.
3. Non-vague answers contribute ALL of their words.
4. Relevance is detected ONLY on the first meaningful answer,
   based on topic keyword match.
5. Each topic allows at most 2 follow-ups total.
6. A topic is complete if:
       - total words >= 25, OR
       - follow-ups >= 2
7. Topic keyword matches do NOT affect word count,
   only the relevance flag.

This file does NOT generate follow-ups.
It only tracks depth, relevance, and follow-up limits.
"""

from typing import Dict, List, Union
from .vague_detector import VagueDetector


class SufficiencyChecker:
    """
    Implements topic sufficiency logic exactly as designed.
    """

    def __init__(self, vague_detector: VagueDetector):
        self.vague_detector = vague_detector

        # Track per-topic state:
        #   words: total non-vague word count
        #   relevant: True if first meaningful response referenced topic keywords
        #   followups: number of follow-ups already asked
        self.topic_status: Dict[str, Dict[str, Union[int, bool]]] = {}

        # Minimum required raw words per topic
        self.required_raw_words = 25

        # Maximum number of follow-ups per topic
        self.max_followups = 2

    def reset_topic(self, topic: str):
        """
        Initialize tracking for a new topic.
        """
        self.topic_status[topic] = {
            "words": 0,
            "relevant": False,
            "followups": 0
        }

    def add_response(self, topic: str, response: str, topic_keywords: List[str]):
        """
        Adds a user response to the topic counters.
        Applies vague rules and relevance rules.
        """

        if topic not in self.topic_status:
            self.reset_topic(topic)

        status = self.topic_status[topic]

        # If vague → contributes zero words
        if self.vague_detector.is_vague(response):
            return

        # Non-vague → add ALL raw words
        wc = len(response.strip().split())
        status["words"] += wc

        # Only set relevance ONCE — first meaningful answer
        if not status["relevant"]:
            lower = response.lower()
            if any(k in lower for k in topic_keywords):
                status["relevant"] = True

    def increment_followup(self, topic: str):
        """
        Increment follow-up count after a follow-up is asked.
        """
        if topic in self.topic_status:
            self.topic_status[topic]["followups"] += 1

    def needs_followup(self, topic: str) -> bool:
        """
        Returns True if this topic requires another follow-up.
        """

        status = self.topic_status.get(topic)
        if not status:
            return True

        # If enough words gathered → no follow-up needed
        if status["words"] >= self.required_raw_words:
            return False

        # If follow-up limit reached → stop
        if status["followups"] >= self.max_followups:
            return False

        return True

    def is_topic_complete(self, topic: str) -> bool:
        """
        Topic is complete when:
        - it meets the word requirement, OR
        - it has reached the follow-up limit
        """
        status = self.topic_status.get(topic)
        if not status:
            return False

        if status["words"] >= self.required_raw_words:
            return True

        if status["followups"] >= self.max_followups:
            return True

        return False

    def get_topic_state(self, topic: str):
        """
        Returns dictionary of:
            { "words": int, "relevant": bool, "followups": int }
        """
        return self.topic_status.get(topic, {})