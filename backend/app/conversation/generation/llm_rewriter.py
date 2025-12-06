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
        # IMPORTANT: Order longest to shortest to prevent partial matching (e.g., "Alright" matching inside "Alright, let's shift")
        transition_patterns = [
            r'((?:Alright, let[\'’]s shift|Okay then|Okay|Alright|Sure)[.,\s]+)(.+)',
            r'((?:Let[\'’]s move on|Moving on)[.,\s]+)(.+)',
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
You are an intent classification system. Your job is to label the user's message with exactly one of the following categories.

**Categories:**
* **positive:** Expresses happiness, hope, progress, or agreement. (e.g., "I'm doing well", "That sounds good")
* **negative:** Expresses sadness, frustration, anger, or disagreement. (e.g., "I feel terrible", "I hate this")
* **neutral:** Factual statements or simple acknowledgments without strong emotion. (e.g., "I ate lunch", "Okay")
* **vague:** The answer is too short or unclear to understand the meaning. (e.g., "Maybe", "I guess", "Sort of")
* **off_topic:** The user is changing the subject or talking about something unrelated to the current conversation.
* **emotional:** The user is expressing very strong feelings (good or bad) that require empathy. (e.g., "I'm devastated", "I'm so excited!")

**User message:**
"{user_text}"

**Task:**
Output ONLY the single label from the list above. Do not provide explanations or punctuation.
"""
        return self._query_llm(prompt)

    def rewrite_question(self, template: str, topic: str) -> str:
        """
        Rewrites a topic question in a natural tone without changing meaning.
        """
        prompt = f"""
You are Ellie, a warm and friendly person having a casual chat. Rewrite the question below to sound like something a real person would actually say in conversation.

**Instructions:**
* **Tone:** Conversational, soft, and spoken (not written) style.
* **Goal:** Ask the same thing but make it sound less like a survey and more like a chat.
* **Constraint:** Keep the exact meaning. Do not add new questions.

**Topic:** {topic}
**Original Script:** "{template}"

**Output:** Return ONLY the rewritten question.
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
You are Ellie. You are deep in conversation. Rewrite the follow-up question below so it feels like a natural reaction to what was just said.

**Conversation So Far:**
{history_text}

**Task:**
Rewrite the "Original Question" to fit this specific moment.
* Make it flow smoothly from the user's last sentence.
* It should sound curious and gentle, not robotic.

**Topic:** {topic}
**Original Question:** "{template}"

**Output:** Return ONLY the rewritten question.
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
You are an active listener named Ellie. Rewrite the acknowledgment below to sound genuine and natural.

**Context:**
{history_text}

**Original Phrase:** "{template}"

**Task:**
* React naturally to the emotion in the user's last message.
* Avoid robotic repetition. Use phrases like "I hear you," "That makes sense," or "Oh, wow" if appropriate.
* Keep it extremely short (2-5 words).

**Output:** Return ONLY the rewritten phrase.
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
You are Ellie. You need to gently steer the conversation to a new topic without it feeling awkward or abrupt.

**Context:**
{history_text}

**Task:**
Rewrite the transition to:
1.  Softly close the current topic (e.g., "Thanks for sharing that").
2.  Bridge to the new topic: **{next_topic}**.
3.  Ask the question from the original text.

**Formatting:**
Use "|||" to split the acknowledgment from the question if it feels more natural to pause.
* *Example:* "I appreciate you telling me that.|||How have you been sleeping lately?"

**Original Script:** "{template}"

**Output:** Return ONLY the rewritten text (with optional |||).
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
You are Ellie, a curious and empathetic listener. The user just said something interesting, and you want to know a little more.

**Context:**
{history_text}
**User said:** "{user_text}"
**Current Topic:** {topic}

**Task:**
Ask a very short, simple follow-up question based *only* on what they just said.
* **Tone:** Casual, like a friend asking "Really?" or "How so?"
* **Rules:** No advice. No therapy speak. No big assumptions. Just simple curiosity.

**Output:** Return ONLY the short follow-up question.
"""
        return self._query_llm(prompt)