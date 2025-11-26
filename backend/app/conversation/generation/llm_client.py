"""
LLM Client (Groq Version)

Uses the Groq API with Llama 3 or Mixtral models.
"""

from groq import Groq
import os
from typing import Optional

class LLMClient:
    """
    Wrapper for Groq chat completion API.
    """

    def __init__(self, model: Optional[object] = None, model_path: Optional[str] = None):
        # We accept model/model_path arguments to maintain compatibility with engine.py signature,
        # but we don't use them for Groq.
        api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        
        if not api_key:
            print("WARNING: GROQ_API_KEY not found. LLM paraphrasing will be DISABLED (using default templates).")
        else:
            try:
                self.client = Groq(api_key=api_key)
            except Exception as e:
                print(f"WARNING: Failed to initialize Groq client: {e}. LLM paraphrasing will be DISABLED.")
                self.client = None
                
        self.model_name = "llama-3.3-70b-versatile" 

    def generate(self, prompt: str) -> str:
        if not self.client:
            return ""

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

