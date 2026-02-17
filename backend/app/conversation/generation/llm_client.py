"""
LLM Client Module

Description:
    This module provides a unified interface for interacting with Large Language Models (LLMs).
    It supports Groq API (llama-3.1-8b-instant) as the primary provider, with Google Gemini
    as a legacy fallback.

Requirements:
    - groq library (primary)
    - python-dotenv library
    - GROQ_API_KEY environment variable must be set in backend/.env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Make Groq imports optional
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq library not installed. Groq LLM features will be disabled.")
    print("To enable: pip install groq")

# Make Gemini imports optional (legacy fallback)
try:
    from google import genai
    from google.genai import types
    from google.genai import errors
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class LLMClient:
    """
    A client wrapper for LLM APIs to handle text generation requests.
    Priority: Groq → Gemini → disabled.
    """

    def __init__(self):
        """
        Initializes the LLM client with the best available provider.

        Provider priority:
            1. Groq (if GROQ_API_KEY is set and groq library installed)
            2. Gemini (if GEMINI_API_KEY is set and google-genai installed)
            3. Disabled (fallback mode)
        """
        # Load environment variables from backend/.env
        current_dir = Path(__file__).resolve().parent
        env_path = current_dir.parents[2] / '.env'
        
        if not env_path.exists():
            cwd = Path.cwd()
            potential_path = cwd / 'backend' / '.env'
            if potential_path.exists():
                env_path = potential_path
        
        load_dotenv(dotenv_path=env_path)
        
        self.is_active = False
        self.client = None
        self.provider = None  # 'groq' or 'gemini'
        
        # Groq config
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = "llama-3.1-8b-instant"
        
        # Gemini config (legacy)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_model = "gemini-2.5-flash"
        
        # --- Provider selection (priority chain) ---
        
        # 1. Try Groq first
        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.client = Groq(api_key=self.groq_api_key)
                self.provider = "groq"
                self.is_active = True
                print(f"✅ Using Groq with model: {self.groq_model}")
                return
            except Exception as e:
                print(f"WARNING: Failed to initialize Groq client: {e}")
        elif not GROQ_AVAILABLE:
            print("Groq library not available, trying fallbacks...")
        elif not self.groq_api_key:
            print("GROQ_API_KEY not set, trying fallbacks...")
        
        # 2. Try Gemini (legacy fallback)
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                self.client = genai.Client(api_key=self.gemini_api_key)
                self.provider = "gemini"
                self.is_active = True
                print(f"✅ Using Gemini with model: {self.gemini_model}")
                return
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini client: {e}")
        
        # 3. Nothing available
        print("⚠️  No LLM provider available. LLM features will be disabled (fallback mode).")
    
    def _groq_generate(self, prompt, system_instruction=None):
        """Generate text using Groq API (non-streaming)."""
        try:
            messages = []
            
            if system_instruction:
                messages.append({
                    "role": "system",
                    "content": system_instruction
                })
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            completion = self.client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=0.7,
                max_completion_tokens=1024,
                top_p=0.95,
                stream=False,
                stop=None
            )
            
            if completion and completion.choices:
                return completion.choices[0].message.content
            else:
                print("Warning: Received an empty response from Groq.")
                return None
                
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                print(f"\n[LLM WARNING] Groq rate limit hit. Switching to fallback templates.")
            else:
                print(f"\n[LLM Client Error] Groq: {type(e).__name__}: {e}")
            return None

    def _gemini_generate(self, prompt, system_instruction=None):
        """Generate text using Google Gemini API (legacy fallback)."""
        if not GEMINI_AVAILABLE or not self.client:
            return None
        
        try:
            from google.genai import types, errors
            
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=8192,
                system_instruction=system_instruction
            )

            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=config
            )
            
            if response and response.text:
                return response.text
            else:
                print("Warning: Received an empty response from Gemini.")
                return None

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"\n[LLM WARNING] Gemini quota exhausted. Switching to fallback templates.")
            elif "404" in str(e):
                print(f"\n[LLM WARNING] Model {self.gemini_model} not found.")
            else:
                print(f"\n[LLM Client Error] Gemini: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            print(f"\n[LLM Client Error] Gemini: {type(e).__name__}: {e}")
            return None

    def get_response(self, prompt, system_instruction=None):
        """
        Sends a prompt to the active LLM provider and retrieves the generated text.

        Args:
            prompt (str): The main input text or question from the user.
            system_instruction (str, optional): Instructions that define the model's behavior.

        Returns:
            str: The content of the model's response as a string.
                 Returns None if the client is inactive (fallback mode).
        """
        if not self.is_active:
            print("LLM Client is inactive. Skipping generation.")
            return None

        if self.provider == "groq":
            return self._groq_generate(prompt, system_instruction)
        elif self.provider == "gemini":
            return self._gemini_generate(prompt, system_instruction)
        
        return None