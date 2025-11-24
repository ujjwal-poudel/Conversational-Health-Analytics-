"""
LLM Client (Groq Version)

Uses the Groq API with Llama 3.1 or Mixtral models.
Faster than OpenAI and 100% free for this use-case.
"""

from groq import Groq
import os


class LLMClient:
    """
    Thin wrapper for Groq chat completion API.
    """

    def __init__(self, model_path=None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-120b"  # Great quality

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            content = response.choices[0].message.content

            if content is None:
                return ""

            return content.strip()

        except Exception as e:
            print("LLM ERROR:", e)
            return ""

