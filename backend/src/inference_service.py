#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference Script for Depression Score Prediction

This script loads a pre-trained multi-target (8-symptom) regression model 
and uses it to predict a single, total depression score from a list of 
transcript turns.

This is intended for a backend service.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Tuple

# --- CRITICAL DEPENDENCY ---
# The following import MUST work. This means the file
# 'hdsc/model.py' (or wherever your model class is defined)
# must be available to this script (e.g., in the same directory
# or in your project's Python path).
try:
    from hdsc.model import PHQTotalMulticlassAttentionModelBERT
except ImportError:
    print("Error: Could not import 'PHQTotalMulticlassAttentionModelBERT' from 'hdsc.model'.")
    print("Please make sure the file 'hdsc/model.py' is in your Python path.")
    exit(1)


# --- CONFIGURATION ---
# Set these variables to match your environment.

# Path to the single, best model checkpoint you want to use in production.
MODEL_PATH = "saved_models/your_model_name/model_best.pt" 

# Pre-trained tokenizer name (must match what was used in training).
TOKENIZER_NAME = "sentence-transformers/all-distilroberta-v1"


def set_device() -> torch.device:
    """
    Checks for available hardware (MPS for Apple, CUDA for NVIDIA) 
    and returns the appropriate torch.device, defaulting to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def load_artifacts(
    model_path: str, 
    tokenizer_name: str, 
    device: torch.device
) -> Tuple[PHQTotalMulticlassAttentionModelBERT, AutoTokenizer]:
    """
    Loads the trained model checkpoint and tokenizer from disk.
    This function should be run ONCE when your backend service starts.

    Args:
        model_path: Filepath to the .pt model checkpoint.
        tokenizer_name: Name of the tokenizer from Hugging Face.
        device: The torch.device to load the model onto.

    Returns:
        A tuple containing the loaded (model, tokenizer).
    """
    
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Loading model from: {model_path}...")
    # We use map_location=device to ensure the model loads correctly
    # on any hardware, even if it was trained on a different machine (e.g., CUDA).
    try:
        loaded_dict = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please update the 'MODEL_PATH' variable in this script.")
        raise
    
    # Re-create the model structure using saved hyperparameters
    model_kwargs = loaded_dict["kwargs"]
    model_state_dict = loaded_dict["model"]
    
    model = PHQTotalMulticlassAttentionModelBERT(
        device=device,
        **model_kwargs,
    )
    
    # Load the trained weights into the model
    model.load_state_dict(model_state_dict)
    
    # Move model to the correct device (e.g., MPS, CUDA, or CPU)
    model = model.to(device)
    
    # --- CRUCIAL ---
    # Set the model to evaluation mode. This disables
    # operations like dropout, which are only used during training.
    model.eval()
    
    print("Artifacts loaded successfully.")
    return model, tokenizer


@torch.no_grad()  # Decorator to disable gradient calculations
def get_depression_score(
    transcript_turns: List[str], 
    model: PHQTotalMulticlassAttentionModelBERT, 
    tokenizer: AutoTokenizer, 
    device: torch.device
) -> float:
    """
    Runs a single inference on a list of transcript turns.
    This is the function your backend will call for each new request.

    Args:
        transcript_turns: A list of strings, where each string is a 
                          turn in the conversation (e.g., ["Hello", "I feel sad"]).
        model: The loaded PHQTotalMulticlassAttentionModelBERT model.
        tokenizer: The loaded tokenizer.
        device: The torch.device the model is on.

    Returns:
        A single float representing the total predicted depression score.
    """
    
    # 1. Preprocessing (Tokenization)
    # Tokenize the list of turns, padding and truncating to the model's max length.
    inputs = tokenizer(
        transcript_turns, 
        padding="max_length",  # Pad to the max length the model expects
        truncation=True,       # Truncate if longer than max length
        return_tensors="pt"    # Return PyTorch tensors
    )

    # 2. Create a "batch" of 1
    # The model was trained on batches of data, so we must replicate
    # that structure, even for a single prediction.
    # This structure must match the 'collate_fn' from your test script.
    
    # 'text_lens' in your original script was the number of turns.
    text_lens = torch.tensor([len(transcript_turns)]).to(device)
    
    # Add a batch dimension (from [seq_len] to [1, seq_len])
    # and move tensors to the correct device.
    input_ids = inputs["input_ids"].unsqueeze(0).to(device)
    attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)

    # Assemble the batch dictionary that the model's .forward() method expects
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_lens": text_lens
    }
    
    # 3. Run Inference
    # Call the model's forward() method with the prepared batch.
    # The raw output will be a tensor of 8 symptom scores (e.g., shape [1, 8]).
    symptom_scores_tensor = model(batch) 
    
    # 4. Sum the 8 scores to get the total
    # As confirmed by the research paper, the total score is the
    # [cite_start]sum of the individual predicted symptom scores[cite: 235].
    # We sum along dimension 1 (the dimension with the 8 scores).
    total_score_tensor = torch.sum(symptom_scores_tensor, dim=1)
    
    # 5. Format and Return the Final Value
    # .squeeze() removes extra dimensions (e.g., [1, 1] -> [1])
    # .cpu() moves the tensor from GPU/MPS back to the CPU
    # .item() converts the single-value tensor to a plain Python number (float)
    final_score = total_score_tensor.squeeze().cpu().item()
    
    return final_score


# --- Main Execution Block ---
# This block runs ONLY when you execute the script directly 
# (e.g., `python inference_service.py`).
# It's an example of how to use the functions.
if __name__ == "__main__":
    
    """
    In a real backend service (like FastAPI or Flask), you would:
    1. Call set_device() and load_artifacts() *ONCE* when the server starts.
    2. Call get_depression_score() *inside your API endpoint* every time you get a new request.
    """
    
    # 1. Set the device
    device = set_device()
    
    try:
        # 2. Load the model and tokenizer (this can take a few seconds)
        print("Initializing service...")
        model, tokenizer = load_artifacts(
            model_path=MODEL_PATH,
            tokenizer_name=TOKENIZER_NAME,
            device=device
        )
        
        # 3. Simulate a new input (this would come from your API request)
        # This input MUST match the 'chunking' you used in training.
        # If CHUNKING was 'lines', this is just a list of lines.
        new_transcript = [
            "Hello, I'm here to talk today.",
            "I've been feeling pretty down.",
            "It's hard to get out of bed in the morning.",
            "I just don't feel like myself.",
            "I don't enjoy things I used to.",
            "My sleep is all messed up.",
            "I feel like a failure sometimes.",
            "It's hard to focus on my work."
        ]

        # 4. Get the prediction
        print("\nRunning inference on sample transcript...")
        total_phq_score = get_depression_score(
            new_transcript, 
            model, 
            tokenizer, 
            device
        )

        print("\n--- INFERENCE COMPLETE ---")
        print(f"Predicted Total Depression Score: {total_phq_score:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during initialization or inference:")
        print(f"{e}")
        print("Please check your file paths and dependencies.")