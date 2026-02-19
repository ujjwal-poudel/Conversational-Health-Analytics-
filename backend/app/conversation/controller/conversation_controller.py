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
        # Use a list of remaining topics to allow dynamic reordering
        self.remaining_topics: List[str] = list(self.topics)
        self.completed_topics: List[str] = []  # Track completed topics to prevent repeats
        self.current_topic: Optional[str] = None

        self.current_topic_responses: List[str] = []
        self.max_followups: int = max_followups
        self._selected_secondary: List[str] = []  # Pre-selected secondary questions for current topic

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

    def _get_recent_conversation_history(self, n: int = 2) -> list:
        """
        Retrieves the last N question-answer pairs from the conversation.
        
        Args:
            n: Number of recent pairs to retrieve (default: 2)
            
        Returns:
            List of (question, answer) tuples
        """
        return self.qa_recorder.get_recent_pairs(n)

    def _minimal_transition(self, next_topic: str) -> str:
        """
        Returns a short, natural transition line using the LLM.
        Falls back to a simple filler if LLM fails.
        Includes conversation history for context-aware transitions.
        """
        template = random.choice(self.transition_fillers)
        history = self._get_recent_conversation_history()
        
        # Use LLM to create a more natural transition with context
        transition = self._safe_rewrite(
            self.llm.rewrite_transition,
            template,
            next_topic,
            history,
            fallback_text=template
        )
        
        return transition

    def get_current_topic(self) -> Optional[str]:
        """
        Returns the current topic label, or None if all topics are finished.
        """
        return self.current_topic

    def move_to_next_topic(self) -> Optional[str]:
        """
        Moves to the next topic.

        Uses topic detection to optionally reorder the remaining topics.
        Resets per-topic tracking in the sufficiency checker.
        Prevents revisiting completed topics.
        """
        # Mark current topic as completed before moving to next
        if self.current_topic and self.current_topic not in self.completed_topics:
            self.completed_topics.append(self.current_topic)
        
        # Check if user input triggers a jump to a remaining topic
        override_topic = self._detect_next_topic()

        if override_topic and override_topic in self.remaining_topics:
            # Only use override if it hasn't been completed yet
            if override_topic not in self.completed_topics:
                # Move the detected topic to the front of the list
                self.remaining_topics.remove(override_topic)
                self.remaining_topics.insert(0, override_topic)

        if not self.remaining_topics:
            self.current_topic = None
            return None

        # Pop the next topic from the front of the list
        next_topic = self.remaining_topics.pop(0)
        
        # Double-check it's not already completed (safety check)
        if next_topic in self.completed_topics:
            # Skip this topic and try the next one
            return self.move_to_next_topic()
        
        self.current_topic = next_topic
        self.current_topic_responses = []
        self.sufficiency_checker.reset_topic(next_topic)

        # Pre-select 2 random secondary questions for this topic
        self._selected_secondary = self.template_manager.get_random_secondary_questions(next_topic, count=self.max_followups)

        return next_topic

    def _detect_next_topic(self) -> Optional[str]:
        """
        Detects the next topic based on user content.

        Logic:
        - Concatenate all responses within the current topic.
        - For each REMAINING topic (excluding completed ones), count occurrences of its keywords.
        - Choose the topic with highest keyword count.
        - If tie, choose randomly among tied topics.
        - If all scores are zero, return None (use default order).
        """
        if not self.remaining_topics:
            return None

        combined = " ".join(self.current_topic_responses).lower()
        
        scores: Dict[str, int] = {}
        for topic in self.remaining_topics:
            # Skip topics that are already completed
            if topic in self.completed_topics:
                continue
                
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
        Includes conversation history so the question flows naturally.
        """
        template = self.template_manager.get_primary_question(topic)
        fallback = template or "Could you tell me a bit about this?"
        history = self._get_recent_conversation_history()
        question = self._safe_rewrite(
            self.llm.rewrite_question,
            template,
            topic,
            history,
            fallback_text=fallback
        )
        self.qa_recorder.record_bot_message(question, include_in_inference=True)
        return question

    def _generate_secondary_question(self, topic: str, index: int = 0) -> str:
        """
        Returns the rewritten secondary question for a topic.
        Uses the pre-selected secondary question at the given index.
        Includes conversation history to avoid repeating the primary question.
        Passes all secondary questions as examples for the LLM.
        """
        # Use pre-selected secondary question at index, fallback to random
        if index < len(self._selected_secondary):
            template = self._selected_secondary[index]
        else:
            template = self.template_manager.get_secondary_question(topic)
        
        fallback = template or "Can you tell me more about that?"
        history = self._get_recent_conversation_history()
        all_secondary = self.template_manager.get_secondary_questions(topic)
        
        question = self._safe_rewrite(
            self.llm.rewrite_followup,
            template,
            topic,
            history,
            all_secondary,
            fallback_text=fallback
        )
        self.qa_recorder.record_bot_message(question, include_in_inference=True)
        return question

    def _generate_followup(self, topic: str, user_text: str, relevant: bool) -> str:
        """
        Returns either a semantic follow-up (if relevant) or a template-based
        follow-up question. Relevance is based on topic keywords and is used
        only to decide which style of follow-up to use, not to block flow.
        Includes conversation history and example secondary questions for context.
        """
        history = self._get_recent_conversation_history()
        all_secondary = self.template_manager.get_secondary_questions(topic)
        
        if relevant:
            fallback = "Could you tell me a bit more?"
            followup = self._safe_rewrite(
                self.llm.generate_semantic_followup,
                user_text,
                topic,
                history,
                all_secondary,
                fallback_text=fallback
            )
        else:
            # Use a random secondary question as the template
            template = self.template_manager.get_secondary_question(topic)
            template = template or "Could you tell me a little more?"
            followup = self._safe_rewrite(
                self.llm.rewrite_followup,
                template,
                topic,
                history,
                all_secondary,
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
                # Check if transition can be split into multiple parts
                parts = self.llm._split_response(transition_line)
                if len(parts) > 1:
                    # Record each part separately and return as list
                    for part in parts:
                        self.qa_recorder.record_bot_message(part, include_in_inference=False)
                    return parts + [end]  # Return array of message parts
                else:
                    self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
                    return [transition_line, end]  # Return as array

            transition_line = self._minimal_transition(next_topic)
            # Check if transition can be split into multiple parts
            parts = self.llm._split_response(transition_line)
            
            if len(parts) > 1:
                # Transition already includes the question (split by |||)
                # Record each part separately and return as list
                for part in parts[:-1]:  # All parts except the last (question)
                    self.qa_recorder.record_bot_message(part, include_in_inference=False)
                # Last part is the question - record it as inference
                self.qa_recorder.record_bot_message(parts[-1], include_in_inference=True)
                return parts  # Return array of message parts (no duplicate question)
            else:
                # Transition doesn't include question, so generate it separately
                self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
                next_question = self._generate_primary_question(next_topic)
                return [transition_line, next_question]  # Return as array

        followups_used = int(status.get("followups", 0))
        relevant = bool(status.get("relevant", False))

        # Follow-ups use pre-selected secondary questions
        if followups_used < len(self._selected_secondary):
            self.sufficiency_checker.increment_followup(topic)
            return self._generate_secondary_question(topic, index=followups_used)

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
            parts = self.llm._split_response(transition_line)
            if len(parts) > 1:
                for part in parts:
                    self.qa_recorder.record_bot_message(part, include_in_inference=False)
                return parts + [end]  # Return array of message parts
            else:
                self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
                return [transition_line, end]  # Return as array

        transition_line = self._minimal_transition(next_topic)
        parts = self.llm._split_response(transition_line)
        
        if len(parts) > 1:
            # Transition already includes the question (split by |||)
            for part in parts[:-1]:  # All parts except the last (question)
                self.qa_recorder.record_bot_message(part, include_in_inference=False)
            # Last part is the question - record it as inference
            self.qa_recorder.record_bot_message(parts[-1], include_in_inference=True)
            return parts  # Return array of message parts (no duplicate question)
        else:
            # Transition doesn't include question, so generate it separately
            self.qa_recorder.record_bot_message(transition_line, include_in_inference=False)
            next_question = self._generate_primary_question(next_topic)
            return [transition_line, next_question]  # Return as array