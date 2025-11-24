"""
Test Conversation Script

This script runs a complete simulation of your conversation engine
using your actual templates, LLM, sufficiency checker, and controller.

Run using:
    python test_conversation.py
"""

from conversation.engine import ConversationEngine

def main():

    # Adjust this to your real folder
    templates_base_path = "/Users/ujjwalpoudel/Documents/insane_projects/Conversational-Health-Analytics-/backend/app/conversation/data"

    # Adjust this to your model path
    model_path = "/Volumes/MACBACKUP/models/chat_model/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

    print("Initializing engine...")
    engine = ConversationEngine(
        templates_base_path=templates_base_path,
        model_path=model_path
    )

    print("\nStarting conversation...\n")
    bot = engine.start()
    print(f"Bot: {bot}")

    # Simple loop for manual testing
    while not engine.is_finished():
        user = input("You: ")
        bot = engine.process(user)
        print(f"Bot: {bot}")

    print("\nConversation finished.\n")

    print("Final Q/A Pairs for inference:")
    pairs = engine.get_inference_pairs()
    for i, item in enumerate(pairs):
        print(f"{i+1}. {item}")


if __name__ == "__main__":
    main()