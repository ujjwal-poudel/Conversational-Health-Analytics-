"""
Conversation Controller

Coordinates the conversation flow, including:
- Topic progression
- Question selection
- Follow-ups
- Minimal fillers only at transitions
- Semantic follow-ups when relevant
- Topic detection based on user content
- Safe fallbacks if LLM fails
"""

import random
from typing import List, Optional, Dict


class ConversationController:
    """
    Manages the assessment conversation for a single user.
    """

    def __init__(
        self,
        template_manager,
        llm_rewriter,
        qa_recorder,
        sufficiency_checker,
        vague_detector,
        max_followups: int = 2
    ):
        """
        Initializes the controller.

        template_manager:
            Provides topics, questions, follow-ups, keywords, and end message.

        llm_rewriter:
            Handles rewriting of questions and follow-ups.

        qa_recorder:
            Records bot and user messages, and Q/A pairs for inference.

        sufficiency_checker:
            Tracks non-vague word counts and follow-up counts per topic.

        vague_detector:
            Used by the sufficiency checker to decide vagueness.

        max_followups:
            Maximum number of follow-ups per topic (including secondary).
        """

        self.template_manager = template_manager
        self.llm = llm_rewriter
        self.qa_recorder = qa_recorder
        self.sufficiency_checker = sufficiency_checker
        self.vague_detector = vague_detector

        self.topics: List[str] = self.template_manager.get_topic_list()
        self.current_topic_index: int = -1

        self.current_topic_responses: List[str] = []
        self.max_followups: int = max_followups

        self.transition_fillers = [
            "Okay.",
            "Alright.",
            "Sure.",
            "Okay then.",
            "Alright, let’s shift."
        ]

    def _safe_rewrite(self, rewrite_fn, *args, fallback_text: str) -> str:
        """
        Safely calls an LLM rewrite function with a fallback.
        """
        try:
            result = rewrite_fn(*args)
            if not result or not result.strip():
                return fallback_text
            return result.strip()
        except Exception:
            return fallback_text

    def _minimal_transition(self, next_topic: str) -> str:
        """
        Returns a short, natural transition line using the LLM.
        Falls back to a simple filler if LLM fails.
        """
        template = random.choice(self.transition_fillers)
        
        # Use LLM to create a more natural transition
        transition = self._safe_rewrite(
            self.llm.rewrite_transition,
            template,
            next_topic,
            fallback_text=template
        )
        
        return transition

    def get_current_topic(self) -> Optional[str]:
        """
        Returns the current topic label, or None if all topics are finished.
        """
        if 0 <= self.current_topic_index < len(self.topics):
            return self.topics[self.current_topic_index]
        return None

    def move_to_next_topic(self) -> Optional[str]:
        """
        Moves to the next topic.

        Uses topic detection to optionally override the default ordering.
        Resets per-topic tracking in the sufficiency checker.
        """
        override_topic = self._detect_next_topic()

        if override_topic:
            self.current_topic_index = self.topics.index(override_topic)
        else:
            self.current_topic_index += 1

        if self.current_topic_index >= len(self.topics):
            return None

        self.current_topic_responses = []

        next_topic = self.topics[self.current_topic_index]
        self.sufficiency_checker.reset_topic(next_topic)

        return next_topic

    def _detect_next_topic(self) -> Optional[str]:
        """
        Detects the next topic based on user content.

        Logic:
        - Concatenate all responses within the current topic.
        - For each remaining topic, count occurrences of its keywords.
        - Choose the topic with highest keyword count.
        - If tie, choose randomly among tied topics.
        - If all scores are zero, return None (use default order).
        """
        current_topic = self.get_current_topic()
        if current_topic is None:
            return None

        combined = " ".join(self.current_topic_responses).lower()
        remaining_topics = self.topics[self.current_topic_index + 1:]

        if not remaining_topics:
            return None

        scores: Dict[str, int] = {}
        for topic in remaining_topics:
            keywords = self.template_manager.get_topic_keywords(topic)
            score = 0
            for kw in keywords:
                kw_low = kw.lower()
                if kw_low in combined:
                    score += combined.count(kw_low)
            scores[topic] = score

        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return None

        candidates = [t for t, s in scores.items() if s == max_score]
        return random.choice(candidates)

    def _generate_primary_question(self, topic: str) -> str:
        """
        Returns the rewritten primary question for a topic.
        """
        template = self.template_manager.get_primary_question(topic)
        fallback = template or "Could you tell me a bit about this?"
        question = self._safe_rewrite(
            self.llm.rewrite_question,
            template,
            topic,
            fallback_text=fallback
        )
        self.qa_recorder.record_bot_message(question, include_in_inference=True)
        return question

    def _generate_secondary_question(self, topic: str) -> str:
        """
        Returns the rewritten secondary question for a topic.
        """
        template = self.template_manager.get_secondary_question(topic)
        fallback = template or "Can you tell me more about that?"
        question = self._safe_rewrite(
            self.llm.rewrite_question,
            template,
            topic,
            fallback_text=fallback
        )
        self.qa_recorder.record_bot_message(question, include_in_inference=True)
        return question

    def _generate_followup(self, topic: str, user_text: str, relevant: bool) -> str:
        """
        Returns either a semantic follow-up (if relevant) or a template-based
        follow-up question. Relevance is based on topic keywords and is used
        only to decide which style of follow-up to use, not to block flow.
        """
        if relevant:
            fallback = "Could you tell me a bit more?"
            followup = self._safe_rewrite(
                self.llm.generate_semantic_followup,
                user_text,
                topic,
                fallback_text=fallback
            )
        else:
            template = self.template_manager.get_followup_template(topic)
            template = template or "Could you tell me a little more?"
            followup = self._safe_rewrite(
                self.llm.rewrite_followup,
                template,
                topic,
                fallback_text=template
            )

        self.qa_recorder.record_bot_message(followup, include_in_inference=True)
        return followup

    def start_first_topic(self) -> Optional[str]:
        """
        Starts the conversation with the first topic and question.
        """
        next_topic = self.move_to_next_topic()
        if next_topic is None:
            return None
        return self._generate_primary_question(next_topic)

    def handle_user_answer(self, user_text: str) -> str:
        """
        Processes a user answer and decides the next bot message.

        Flow:
        - Record the user answer.
        - Update sufficiency tracking for this topic.
        - If topic is complete → transition to next topic.
        - Else → secondary or follow-up question.
        """

        topic = self.get_current_topic()

        if topic is None:
            closing = self.template_manager.get_conversation_end()
            self.qa_recorder.record_user_message(user_text, include_in_inference=True)
            return closing

        self.qa_recorder.record_user_message(user_text, include_in_inference=True)
        self.current_topic_responses.append(user_text)

        topic_keywords = self.template_manager.get_topic_keywords(topic)
        self.sufficiency_checker.add_response(topic, user_text, topic_keywords)
        status = self.sufficiency_checker.get_topic_state(topic)

        # If the topic is complete (words >= 25 or followups >= max), transition
        if self.sufficiency_checker.is_topic_complete(topic):
            next_topic = self.move_to_next_topic()
            if next_topic is None:
                end = self.template_manager.get_conversation_end()
                transition_line = self._minimal_transition("conversation end")
                self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
                return f"{transition_line} {end}"

            transition_line = self._minimal_transition(next_topic)
            self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
            next_question = self._generate_primary_question(next_topic)
            return f"{transition_line} {next_question}"

        followups_used = int(status.get("followups", 0))
        relevant = bool(status.get("relevant", False))

        # First follow-up is always the secondary question
        if followups_used == 0:
            self.sufficiency_checker.increment_followup(topic)
            return self._generate_secondary_question(topic)

        # If still allowed by SufficiencyChecker, ask another follow-up
        if self.sufficiency_checker.needs_followup(topic):
            self.sufficiency_checker.increment_followup(topic)
            return self._generate_followup(topic, user_text, relevant)

        # Safety: if for some reason needs_followup is False but topic not complete,
        # force a transition so the conversation never gets stuck.
        next_topic = self.move_to_next_topic()
        if next_topic is None:
            end = self.template_manager.get_conversation_end()
            transition_line = self._minimal_transition("conversation end")
            self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
            return f"{transition_line} {end}"

        transition_line = self._minimal_transition(next_topic)
        self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
        next_question = self._generate_primary_question(next_topic)
        return f"{transition_line} {next_question}"