"""
LLMRewriter
Handles LLM rewriting and intent classification for the conversation system.

This component performs:
1. Rewriting topic questions.
2. Rewriting follow-up questions.
3. Rewriting fillers.
4. Rewriting transitions.
5. Rewriting introductions.
6. Generating semantic follow-ups.
7. Classifying user intent.

All rewriting must stay natural, simple, and must not add new meaning.
"""

import re

class LLMRewriter:
    """
    Wrapper around an LLM client. Ensures the LLM output is cleaned so that
    the system receives only a single rewritten line without explanations.
    """

    def __init__(self, llm_client):
        """
        llm_client must implement:
            llm_client.generate(prompt: str) -> str
        """
        self.llm = llm_client

    def _clean_response(self, text: str) -> str:
        """
        Cleans the LLM output so that only the rewritten line is returned.

        Cleaning rules:
        1. If the response contains a quoted rewritten sentence, return the first quoted text.
        2. Otherwise return the first non-empty line.
        3. Remove prefixes like "Rewritten:" or "Note:" if present.
        """
        if not text:
            return ""

        quotes = re.findall(r'"(.*?)"', text, re.DOTALL)
        if quotes:
            cleaned = quotes[0].strip()
            if cleaned:
                return cleaned

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines:
            line = lines[0]
            line = re.sub(r'^[Rr]ewritten:? ?', "", line)
            line = re.sub(r'^[Nn]ote:? ?', "", line)
            return line.strip()

        return text.strip()

    def _format_conversation_history(self, history: list) -> str:
        """
        Formats conversation history (list of Q&A tuples) for LLM prompts.
        
        Args:
            history: List of tuples [(question, answer), ...]
            
        Returns:
            Formatted string representation of the conversation history.
        """
        if not history:
            return ""
        
        formatted = "\n".join([
            f"Bot: {q}\nUser: {a}" 
            for q, a in history
        ])
        return f"\nPrevious conversation:\n{formatted}\n"

    def _split_response(self, text: str) -> list:
        """
        Splits a response into multiple natural parts if appropriate.
        
        The LLM can use markers like "|||" or return multi-line responses
        to indicate natural split points. For fallback templates, also looks
        for natural transition patterns like "Okay, let's shift [topic]"
        followed by a question.
        
        Args:
            text: The response text to potentially split
            
        Returns:
            List of text parts (single item if no split needed)
        """
        if not text:
            return [text]
        
        # Check for explicit split marker from LLM
        if "|||" in text:
            parts = [p.strip() for p in text.split("|||") if p.strip()]
            return parts if len(parts) > 1 else [text]
        
        # For fallback templates, try to detect natural transition + question pattern
        # Common patterns: "Okay. What...", "Alright, let's shift. How...", etc.
        import re
        
        # Pattern: [transition phrase] [punctuation] [question]
        # Look for transition words followed by a question
        transition_patterns = [
            r'((?:Okay|Alright|Sure|Okay then|Alright, let\'s shift)[.,\s]+)(.+)',
            r'((?:Let\'s move on|Moving on)[.,\s]+)(.+)',
        ]
        
        for pattern in transition_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                transition = match.group(1).strip()
                question = match.group(2).strip()
                # Only split if both parts are substantial
                if len(transition) > 3 and len(question) > 5:
                    return [transition, question]
        
        # No natural split found, return as single
        return [text]

    def _query_llm(self, prompt: str, system_instruction: str = None, fallback_text: str = "") -> str:
        """
        Helper to query the LLM and clean the response.
        If LLM fails or is inactive, returns fallback_text.
        """
        is_active = getattr(self.llm, 'is_active', False)
        status_icon = "LLM Working" if is_active else "LLM Not Working"
        print(f"\n[{status_icon} LLM STATUS] Active: {is_active}")
        print(f"[ORIGINAL/FALLBACK] {fallback_text}")
        
        raw = self.llm.get_response(prompt, system_instruction=system_instruction)
        
        # If LLM is inactive (returns None) or fails, use fallback
        if raw is None:
            print(f"[RESULT] Using Fallback (LLM returned None)")
            return fallback_text
            
        cleaned = self._clean_response(raw)
        print(f"[RESULT] LLM Generated: {cleaned}")
        return cleaned

    def classify_intent(self, user_text: str) -> str:
        """
        Returns one of the following labels:
        positive, negative, neutral, vague, off_topic, emotional
        """
        prompt = f"""
Classify the user's message using only one label:
positive, negative, neutral, vague, off_topic, emotional

Rules:
Return only the label.
Do not explain.
Do not add extra text.

User message:
"{user_text}"
"""
        return self._query_llm(prompt)

    def rewrite_question(self, template: str, topic: str) -> str:
        """
        Rewrites a topic question in a natural tone without changing meaning.
        """
        prompt = f"""
Rewrite this question in a simple and natural conversational tone.
Keep the meaning exactly the same.
Do not add anything new.
Return only the rewritten question.

Topic: {topic}
Original: "{template}"
"""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_followup(self, template: str, topic: str, conversation_history: list = None) -> str:
        """
        Rewrites a follow-up question in a gentle tone.
        
        Args:
            template: The template follow-up question
            topic: The current topic
            conversation_history: Optional list of (question, answer) tuples for context
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        prompt = f"""
Rewrite this follow-up question naturally based on the conversation context.
Do not add new ideas.
Return only the rewritten follow-up.
{history_text}
Topic: {topic}
Original: "{template}"
"""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_filler(self, template: str, conversation_history: list = None) -> str:
        """
        Rewrites a filler or acknowledgment line.
        
        Args:
            template: The template filler text
            conversation_history: Optional list of (question, answer) tuples for context
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        prompt = f"""
Rewrite this acknowledgment in a natural conversational tone based on context.
Do not add new meaning.
Return only the rewritten acknowledgment.
{history_text}
Original: "{template}"
"""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_transition(self, template: str, next_topic: str, conversation_history: list = None) -> str:
        """
        Rewrites a transition line naturally, optionally splitting into multiple parts.
        
        Args:
            template: The template transition text
            next_topic: The upcoming topic
            conversation_history: Optional list of (question, answer) tuples for context
            
        Returns:
            Either a single string or can be processed for multi-part responses.
            Use "|||" in the LLM response to split into multiple parts.
            Example: "Okay, let's shift to the next topic|||How is your daily routine?"
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        prompt = f"""
Rewrite this transition line naturally based on the conversation context.
You can optionally split the response into two parts using "|||" as a separator for a more natural flow.
Example: "Okay, let's move on|||What about your sleep schedule?"
Keep it short and conversational.
Do not add new content beyond natural transitions.
{history_text}
Next topic: {next_topic}
Original: "{template}"
"""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_intro(self, template: str) -> str:
        """
        Rewrites an introduction line.
        """
        prompt = f"""
Rewrite this introduction in a warm but neutral conversational tone.
Return only the rewritten line.

Original: "{template}"
"""
        return self._query_llm(prompt, fallback_text=template)

    def generate_semantic_followup(self, user_text: str, topic: str, conversation_history: list = None) -> str:
        """
        Generates a short follow-up based strictly on what the user said.
        No assumptions, no advice, no clinical references.
        
        Args:
            user_text: The user's most recent message
            topic: The current topic
            conversation_history: Optional list of (question, answer) tuples for context
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        prompt = f"""
Generate one gentle follow-up question based strictly on the user's message and conversation context.

Rules:
Stay within the topic: {topic}
Do not add new details.
Do not infer anything.
Keep it short.
Return only the follow-up question.
{history_text}
User's latest message:
"{user_text}"
"""
        return self._query_llm(prompt)