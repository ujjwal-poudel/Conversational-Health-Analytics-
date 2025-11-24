"""
Vague Detector

Determines whether a user response is too vague to be useful.

Final agreed logic:

1. Word count < 5:
   - Check topic keywords.
   - Check vague words.
   - If neither present → vague.
   - If topic keywords present → NOT vague.
   - If vague words present → vague.

2. Word count ≥ 10:
   - Always not vague.

3. Word count 5–9:
   - Topic keywords override → not vague.
   - vague_count >= 2 → vague.
   - Otherwise → not vague.
"""

from typing import List, Dict


class VagueDetector:
    """
    Detects vague or low-information responses using the final rules.
    """

    def __init__(self, topic_keywords: Dict[str, List[str]]):
        """
        topic_keywords:
            A dict: topic -> list of meaningful keywords.
        """
        self.topic_keywords = topic_keywords

        self.vague_words: List[str] = [
            "fine",
            "okay",
            "ok",
            "normal",
            "idk",
            "maybe",
            "not sure",
            "good",
            "same",
            "whatever",
            "nothing much",
            "i guess",
            "i think so"
        ]

    def word_count(self, text: str) -> int:
        if not text:
            return 0
        return len(text.strip().split())

    def count_vague_words(self, text: str) -> int:
        lower = text.lower()
        return sum(1 for phrase in self.vague_words if phrase in lower)

    def contains_topic_keywords(self, text: str) -> bool:
        lower = text.lower()
        for keywords in self.topic_keywords.values():
            for w in keywords:
                if w in lower:
                    return True
        return False

    def is_vague(self, text: str) -> bool:
        wc = self.word_count(text)
        text = text.lower()

        has_topic = self.contains_topic_keywords(text)
        vague_count = self.count_vague_words(text)

        # Rule 1: <5 words
        if wc < 5:
            if has_topic:
                # Topic content overrides vagueness even if vague words exist
                return False
            if vague_count > 0:
                return True
            return True  # too short + no content

        # Rule 2: >=10 words
        if wc >= 10:
            return False

        # Rule 3: 5–9 words, topic still overrides
        if has_topic:
            return False

        # vague density rule
        if vague_count >= 2:
            return True

        return False