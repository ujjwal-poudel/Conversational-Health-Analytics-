"""
Test script to verify correct model behavior and controlled outputs
"""

from gpt4all import GPT4All
import os

def main():
    MODEL_DIR = "/Volumes/MACBACKUP/models/chat_model"
    MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print("Loading model using GPT4All...")
    model = GPT4All(
        model_name=MODEL_NAME,
        model_path=MODEL_DIR
    )

    print("Model loaded successfully.")
    print("Testing inference...\n")

    prompt = (
        "Rewrite this politely and concisely as a single sentence: "
        "I'm really tired."
    )

    response = model.generate(
        prompt,
        max_tokens=40,
        temp=0.2,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1
    )

    print("Model Response:")
    print(response)


if __name__ == "__main__":
    main()
