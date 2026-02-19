"""
LLMRewriter
Handles LLM rewriting and intent classification for the conversation system.

This component performs:
1. Rewriting primary topic questions naturally.
2. Rewriting follow-up questions based on conversation context.
3. Generating context-aware semantic follow-ups.
4. Generating natural transitions with empathetic acknowledgment.
5. Classifying user intent.

All rewriting produces natural, human-like conversational questions
grounded in PHQ-8 assessment topics.
"""

import re
import logging

logger = logging.getLogger(__name__)


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
        1. If the response contains a quoted sentence that looks like a real question
           (contains '?' or is longer than 15 chars), return it.
        2. Otherwise return the first non-empty line.
        3. Remove prefixes like "Rewritten:", "Output:", or "Note:" if present.
        """
        if not text:
            return ""

        # Only extract quoted text if it looks like an actual question/sentence,
        # not a short echo of user words
        quotes = re.findall(r'"(.*?)"', text, re.DOTALL)
        for q in quotes:
            cleaned = q.strip()
            # Accept quoted text only if it's a question or a substantial sentence
            if cleaned and ("?" in cleaned or len(cleaned) > 15):
                return cleaned

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if lines:
            line = lines[0]
            line = re.sub(r'^[Rr]ewritten:? ?', "", line)
            line = re.sub(r'^[Nn]ote:? ?', "", line)
            line = re.sub(r'^[Oo]utput:? ?', "", line)
            return line.strip()

        return text.strip()

    def _format_conversation_history(self, history: list) -> str:
        """
        Formats conversation history (list of Q&A tuples) for LLM prompts.
        """
        if not history:
            return "No conversation yet — this is the start of the session."
        
        formatted = "\n".join([
            f"You (Ellie): {q}\nPatient: {a}" 
            for q, a in history
        ])
        return formatted

    def _split_response(self, text: str) -> list:
        """
        Splits a response into multiple natural parts if appropriate.
        
        The LLM can use markers like "|||" to indicate split points.
        Example: "I hear you, that sounds rough.|||How has that been affecting your sleep?"
        """
        if not text:
            return [text]
        
        # Check for explicit split marker from LLM
        if "|||" in text:
            parts = [p.strip() for p in text.split("|||") if p.strip()]
            return parts if len(parts) > 1 else [text]
        
        # For fallback templates, try to detect natural transition + question pattern
        import re
        
        transition_patterns = [
            r'((?:Alright, let[\''']s shift|Okay then|Okay|Alright|Sure)[.,\s]+)(.+)',
            r'((?:Let[\''']s move on|Moving on)[.,\s]+)(.+)',
        ]
        
        for pattern in transition_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                transition = match.group(1).strip()
                question = match.group(2).strip()
                if len(transition) > 3 and len(question) > 5:
                    return [transition, question]
        
        return [text]

    def _query_llm(self, prompt: str, system_instruction: str = None, fallback_text: str = "") -> str:
        """
        Helper to query the LLM and clean the response.
        If LLM fails or is inactive, returns fallback_text.
        """
        raw = self.llm.get_response(prompt, system_instruction=system_instruction)
        
        if raw is None:
            logger.warning("[LLM] FALLBACK - LLM returned None, using: %s", fallback_text)
            return fallback_text
        
        cleaned = self._clean_response(raw)
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

    def rewrite_question(self, template: str, topic: str, conversation_history: list = None) -> str:
        """
        Rewrites a primary topic question so it sounds like a natural, human conversation.
        Uses conversation history so the question flows naturally from what was said before.
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else "No conversation yet — this is the start of the session."
        
        prompt = f"""You are Ellie, a warm and empathetic conversational agent conducting a mental health check-in with a patient. You are covering topics related to the PHQ-8 questionnaire, but in a natural, human, conversational way — NOT like a clinical survey.

Your job right now is to ask the patient about a new topic. You must paraphrase the question below so it sounds like something a caring friend or therapist would naturally say in conversation.

**Conversation history so far:**
{history_text}

**Current topic you need to ask about:** {topic}
**Question to paraphrase naturally:** "{template}"

**Rules:**
- Make it sound like a real person talking — warm, casual, curious
- CRITICAL: Keep it to 1 short sentence, 8-12 words max
- The question must still cover the same topic and meaning
- Flow naturally from the conversation history — don't ignore what was said before
- Do NOT add advice, commentary, or extra questions
- Do NOT use clinical language

**Output:** ONLY the paraphrased question, nothing else."""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_followup(self, template: str, topic: str, conversation_history: list = None, example_questions: list = None) -> str:
        """
        Rewrites a follow-up question based on what the patient just said.
        The follow-up should directly reference or build on the patient's last response.
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        examples_text = ""
        if example_questions:
            examples_text = "\n**Example follow-up questions you can draw inspiration from (do NOT copy verbatim):**\n" + "\n".join(f"- {q}" for q in example_questions) + "\n"
        
        prompt = f"""You are Ellie, a warm and empathetic conversational agent having a mental health check-in with a patient. You are covering PHQ-8 topics naturally through conversation.

You just asked the patient a question and they responded. Now you need to ask a follow-up question that digs deeper into what they shared. The follow-up must directly relate to what the patient said — reference their words, their situation, their feelings.

**Conversation so far:**
{history_text}

**Current topic:** {topic}
**Template question to paraphrase:** "{template}"
{examples_text}
**Rules:**
- Your follow-up MUST connect to what the patient just said — reference specific things they mentioned
- Sound like a caring person genuinely curious about their experience
- CRITICAL: Keep it to 1 short sentence, 6-10 words max
- Do NOT give advice, do NOT make assumptions beyond what they said
- Do NOT use clinical language
- Make it feel like a natural back-and-forth conversation

**Example of good follow-up behavior:**
If the patient said "I've been feeling really down because of a breakup", a good follow-up would be something like "Do you think the breakup is the main thing weighing on you?" — notice how it directly references what they shared.

**Output:** ONLY the follow-up question, nothing else."""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_filler(self, template: str, conversation_history: list = None) -> str:
        """
        Rewrites a filler or acknowledgment line.
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        prompt = f"""You are Ellie, a warm conversational agent having a mental health check-in. The patient just shared something with you. You need to give a brief, empathetic acknowledgment before asking your next question.

**Conversation so far:**
{history_text}

**Template acknowledgment:** "{template}"

**Rules:**
- React genuinely to what the patient just said — show you heard them
- Keep it very short (3-5 words)
- Match the emotional tone — if they shared something painful, be gentle; if positive, be warm
- Examples of good acknowledgments: "That sounds really tough.", "I hear you.", "That makes a lot of sense.", "I appreciate you sharing that."
- Do NOT give advice or ask questions here — just acknowledge

**Output:** ONLY the short acknowledgment, nothing else."""
        return self._query_llm(prompt, fallback_text=template)

    def rewrite_transition(self, template: str, next_topic: str, conversation_history: list = None) -> str:
        """
        Creates a natural transition from the current topic to the next one.
        First acknowledges what the patient shared, then gently moves to the new topic.
        Uses "|||" to separate the acknowledgment from the question.
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        prompt = f"""You are Ellie, a warm and empathetic conversational agent having a mental health check-in with a patient. You've been discussing one topic and now need to naturally transition to a new topic.

**Conversation so far:**
{history_text}

**New topic to transition to:** {next_topic}

**Rules:**
- First, give a brief, genuine acknowledgment of what the patient just shared (3-8 words) — show empathy and that you were listening
- Then, naturally lead into a question about the new topic
- Use "|||" to separate the acknowledgment from the question
- The acknowledgment should reference what they actually said — don't be generic
- The question should feel like a natural shift, not an abrupt topic change
- Total: maximum 2 short sentences
- Do NOT use clinical language

**Example format:**
"I really appreciate you telling me that.|||How have you been sleeping lately?"
"That sounds like it's been weighing on you.|||What about your energy — how's that been?"

**Output:** Short acknowledgment|||Short question about {next_topic}"""
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

    def generate_semantic_followup(self, user_text: str, topic: str, conversation_history: list = None, example_questions: list = None) -> str:
        """
        Generates a context-aware follow-up based strictly on what the patient said.
        This is for when the patient's response is relevant and we want to dig deeper
        into their specific experience.
        """
        history_text = self._format_conversation_history(conversation_history) if conversation_history else ""
        
        examples_text = ""
        if example_questions:
            examples_text = "\n**Example follow-up angles you can draw inspiration from (do NOT copy verbatim):**\n" + "\n".join(f"- {q}" for q in example_questions) + "\n"
        
        prompt = f"""You are Ellie, a warm and empathetic conversational agent having a mental health check-in with a patient. You are covering PHQ-8 topics through natural conversation.

The patient just responded to your question. You need to ask a brief follow-up that goes deeper into what THEY specifically said. React to their actual words and experience.

**Conversation so far:**
{history_text}

**Patient's latest response:** "{user_text}"
**Current topic:** {topic}
{examples_text}
**Rules:**
- Your follow-up MUST directly reference something specific the patient said
- Sound genuinely curious about their experience — like a friend who cares
- CRITICAL: Keep it to 1 sentence, 5-10 words max
- Do NOT give advice, do NOT make assumptions beyond what they shared
- Do NOT repeat questions already asked in the conversation history
- Make it conversational — phrases like "Really?", "How so?", "What was that like?" are fine if they fit

**Example of what good looks like:**
Patient says: "I've been feeling really bad and depressed because I had a breakup."
Good follow-up: "Do you think your mood is mainly because of the breakup?"
— Notice how it directly picks up on what they said and gently explores it.

**Output:** ONLY the follow-up question, nothing else."""
        return self._query_llm(prompt)