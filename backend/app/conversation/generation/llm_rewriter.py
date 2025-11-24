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

    def _query_llm(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the cleaned result.
        """
        raw = self.llm.generate(prompt)
        return self._clean_response(raw)

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
        return self._query_llm(prompt)

    def rewrite_followup(self, template: str, topic: str) -> str:
        """
        Rewrites a follow-up question in a gentle tone.
        """
        prompt = f"""
Rewrite this follow-up question naturally.
Do not add new ideas.
Return only the rewritten follow-up.

Topic: {topic}
Original: "{template}"
"""
        return self._query_llm(prompt)

    def rewrite_filler(self, template: str) -> str:
        """
        Rewrites a filler or acknowledgment line.
        """
        prompt = f"""
Rewrite this acknowledgment in a natural conversational tone.
Do not add new meaning.
Return only the rewritten acknowledgment.

Original: "{template}"
"""
        return self._query_llm(prompt)

    def rewrite_transition(self, template: str, next_topic: str) -> str:
        """
        Rewrites a transition line naturally.
        """
        prompt = f"""
Rewrite this transition line naturally.
Keep it short.
Do not add new content.
Return only the rewritten line.

Next topic: {next_topic}
Original: "{template}"
"""
        return self._query_llm(prompt)

    def rewrite_intro(self, template: str) -> str:
        """
        Rewrites an introduction line.
        """
        prompt = f"""
Rewrite this introduction in a warm but neutral conversational tone.
Return only the rewritten line.

Original: "{template}"
"""
        return self._query_llm(prompt)

    def generate_semantic_followup(self, user_text: str, topic: str) -> str:
        """
        Generates a short follow-up based strictly on what the user said.
        No assumptions, no advice, no clinical references.
        """
        prompt = f"""
Generate one gentle follow-up question based strictly on the user's message.

Rules:
Stay within the topic: {topic}
Do not add new details.
Do not infer anything.
Keep it short.
Return only the follow-up question.

User message:
"{user_text}"
"""
        return self._query_llm(prompt)