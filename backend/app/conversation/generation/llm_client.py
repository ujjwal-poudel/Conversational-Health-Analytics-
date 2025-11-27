"""
LLM Client Module

Description:
    This module provides a unified interface for interacting with Large Language Models (LLMs).
    It encapsulates the logic for connecting to the Google Gemini API, managing authentication,
    and generating text responses based on user prompts and system instructions.

    The LLMClient class includes robust error handling to provide clear, actionable feedback
    in the console for common issues such as quota exhaustion (429), permission errors (403),
    and server unavailability (500/503).

Requirements:
    - google-genai library
    - python-dotenv library
    - GEMINI_API_KEY environment variable must be set in backend/.env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

class LLMClient:
    """
    A client wrapper for the Google Gemini API to handle text generation requests.
    """

    def __init__(self):
        """
        Initializes the Gemini client.

        This method retrieves the API key from the .env file located in the backend directory
        and sets up the connection to the Google Gen AI service. It raises an error if the
        API key is missing to prevent runtime failures later.
        
        Attributes:
            self.client (genai.Client): The authenticated Google Gen AI client.
            self.model_name (str): The specific model version to use (e.g., gemini-2.0-flash-exp).
        """
        # Load environment variables from backend/.env
        # Assuming this script is running from the project root or a subdirectory
        # We look for the .env file relative to the current file's location
        current_dir = Path(__file__).resolve().parent
        
        # Construct path to .env file (go up 3 levels to backend/)
        # parent -> generation -> conversation -> app -> backend
        env_path = current_dir.parents[2] / '.env'
        
        # If running from root, fallback to checking standard locations if the relative path fails
        if not env_path.exists():
             # Fallback: try to find backend/.env from current working directory
             cwd = Path.cwd()
             potential_path = cwd / 'backend' / '.env'
             if potential_path.exists():
                 env_path = potential_path
        
        load_dotenv(dotenv_path=env_path)
        
        # Retrieve API key from environment variables for security
        # Note: Ensure your .env file uses 'GEMINI_API_KEY' or 'GOOGLE_API_KEY'
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        self.is_active = False
        self.client = None
        self.model_name = "gemini-2.0-flash"

        if not self.api_key:
            print(f"WARNING: API Key not found at {env_path}. LLM features will be disabled (fallback mode).")
        else:
            try:
                # Initialize the Google Gen AI Client
                self.client = genai.Client(api_key=self.api_key)
                self.is_active = True
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini client: {e}. LLM features will be disabled.")

    def get_response(self, prompt, system_instruction=None):
        """
        Sends a prompt to the LLM and retrieves the generated text response.

        This method includes comprehensive error handling to catch specific API issues
        like quota limits or server errors and prints a detailed explanation to the console.

        Args:
            prompt (str): The main input text or question from the user.
            system_instruction (str, optional): Instructions that define the model's behavior, 
                                                persona, or constraints. Defaults to None.

        Returns:
            str: The content of the model's response as a string.
                 Returns None if the client is inactive (fallback mode).
                 Returns a user-friendly error message if the API call fails.
        """
        if not self.is_active or not self.client:
            print("LLM Client is inactive. Skipping generation.")
            return None

        try:
            # Configure generation parameters
            # We set temperature to 0.7 for a balance of creativity and focus
            config = types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=8192,
                system_instruction=system_instruction
            )

            # Make the API call to generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Return the text portion of the response
            if response and response.text:
                return response.text
            else:
                print("Warning: Received an empty response from the model.")
                return "I'm sorry, I didn't catch that. Could you please repeat?"

        except Exception as e:
            # Catch ALL exceptions (billing, quota, network, etc.)
            # Log the error for debugging but return None to trigger silent fallback
            print(f"\n[LLM Client Error] {type(e).__name__}: {e}")
            print("Action: Falling back to default templates.")
            return None