#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Script for Depression Score Prediction

This script loads all models from a directory, runs inference on a
test set from a .jsonl file, and calculates detailed metrics
for each model.

All results are saved to a JSON file.
"""

import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from torchmetrics.functional import mean_absolute_error, mean_squared_error, f1_score

# --- CRITICAL DEPENDENCY ---
# Make sure the 'hdsc' folder is in the same directory
try:
    from hdsc.model import PHQTotalMulticlassAttentionModelBERT
except ImportError:
    print("Error: Could not import 'PHQTotalMulticlassAttentionModelBERT' from 'hdsc.model'.")
    print("Please make sure the file 'hdsc/model.py' is in your Python path.")
    exit(1)


# --- CONFIGURATION ---
# Adjust these paths to match your environment

# Path to the directory containing all your .pt model files
MODEL_DIR = "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_"

# Path to your test.jsonl file
TEST_FILE_PATH = "test.jsonl"

# Path for the final JSON output file
RESULTS_JSON_PATH = "model_evaluation_results.json"

# Pre-trained tokenizer name (must match what was used in training).
TOKENIZER_NAME = "sentence-transformers/all-distilroberta-v1"

# Batch size for processing turns (to prevent OOM errors)
TURN_BATCH_SIZE = 16

# --- HELPER FUNCTIONS (From inference_service.py) ---

def set_device() -> torch.device:
    """
    Checks for available hardware (MPS for Apple, CUDA for NVIDIA)
    and returns the appropriate torch.device, defaulting to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    return device


def load_model_only(
    model_path: str,
    device: torch.device
) -> PHQTotalMulticlassAttentionModelBERT:
    """
    Loads a single model checkpoint from disk.
    """
    print(f"\n--- Loading model: {Path(model_path).name} ---")
    try:
        loaded_dict = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        raise
    
    model_kwargs = loaded_dict["kwargs"]
    model_state_dict = loaded_dict["model"]
    
    model = PHQTotalMulticlassAttentionModelBERT(
        device=device,
        **model_kwargs,
    )
    
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval() # Set to evaluation mode
    
    return model


@torch.no_grad()  # Decorator to disable gradient calculations
def get_depression_score(
    transcript_turns: List[str],
    model: PHQTotalMulticlassAttentionModelBERT,
    tokenizer: AutoTokenizer,
    device: torch.device,
    turn_batch_size: int = 16
) -> float:
    """
    Runs a single inference on a list of transcript turns.
    Includes turn-level batching to prevent OOM errors.
    """
    
    inputs = tokenizer(
        transcript_turns,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    
    all_sentence_embeddings = []

    for i in range(0, len(transcript_turns), turn_batch_size):
        batch_input_ids = torch.tensor(
            inputs["input_ids"][i:i + turn_batch_size]
        ).to(device)
        batch_attention_mask = torch.tensor(
            inputs["attention_mask"][i:i + turn_batch_size]
        ).to(device)

        model_output = model.encoder(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        sentence_embeddings = model.mean_pooling(
            model_output, batch_attention_mask
        )
        all_sentence_embeddings.append(sentence_embeddings.cpu())

    sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0).to(device)
    
    text_lens = torch.tensor([len(transcript_turns)]).to(device)
    word_outputs = torch.split(sentence_embeddings, text_lens.tolist())
    output = torch.nn.utils.rnn.pad_sequence(
        word_outputs, batch_first=True, padding_value=0
    )

    batch_size = text_lens.size(0)
    sent_h0, sent_c0 = model.init_hidden(batch_size, device=device)
    
    (
        final_hidden_binary,
        final_hidden_regression,
        attn_binary,
        attn_regression,
        sent_conicity,
    ) = model.sent_encoder(
        output,
        text_lens.cpu(),
        sent_h0,
        sent_c0,
        device,
        pooling=model.pooling,
    )

    pred_binary = model.clf_binary(final_hidden_binary)
    symptom_scores_tensor = pred_binary

    total_score_tensor = torch.sum(symptom_scores_tensor, dim=1)
    
    final_score = total_score_tensor.squeeze().cpu().item()
    
    return final_score

# --- NEW FUNCTIONS FOR EVALUATION ---

def load_test_data(filepath: str) -> Tuple[List[List[str]], List[int]]:
    """
    Loads the test.jsonl file.
    """
    all_turns = []
    all_true_scores = []
    
    print(f"Loading test data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    all_turns.append(data['turns'])
                    all_true_scores.append(data['labels'][0])
                except (json.JSONDecodeError, KeyError):
                    print(f"Warning: Skipping malformed line: {line}")
    
    print(f"Loaded {len(all_true_scores)} test items.")
    return all_turns, all_true_scores

# --- MAIN EXECUTION BLOCK ---

def main():
    device = set_device()
    
    # 1. Load Tokenizer (once)
    print(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # 2. Load Test Data (once)
    all_turns, all_true_scores = load_test_data(TEST_FILE_PATH)
    # Move to CPU for now, we'll move to device inside the loop
    true_labels_tensor_cpu = torch.tensor(all_true_scores, dtype=torch.float32)
    
    # 3. Find all models in the directory
    model_dir = Path(MODEL_DIR)
    model_paths = sorted(model_dir.glob("model_2_*.pt"))
    
    if not model_paths:
        print(f"Error: No models found at '{MODEL_DIR}'.")
        print("Please check the 'MODEL_DIR' variable.")
        return

    print(f"Found {len(model_paths)} models to evaluate.")
    
    results_list = []

    # 4. Loop over each model, run inference, and get metrics
    for model_path in model_paths:
        try:
            model = load_model_only(str(model_path), device)
            
            model_predictions = []
            
            # Create a progress bar for the inference step
            pbar = tqdm(all_turns, desc=f"Evaluating {model_path.name}")
            for turns_list in pbar:
                pred_score = get_depression_score(
                    turns_list,
                    model,
                    tokenizer,
                    device,
                    TURN_BATCH_SIZE
                )
                model_predictions.append(pred_score)
            
            # Close the progress bar
            pbar.close()

            # --- Calculate Detailed Metrics ---
            pred_tensor = torch.tensor(model_predictions, dtype=torch.float32).to(device)
            true_labels_tensor = true_labels_tensor_cpu.to(device)

            # Regression Metrics
            mae = mean_absolute_error(pred_tensor, true_labels_tensor).item()
            mse = mean_squared_error(pred_tensor, true_labels_tensor).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()

            # Classification Metrics (Binarized)
            # Create binary tensors (Depressed if score >= 10)
            pred_bin = (pred_tensor >= 10).to(torch.long)
            true_bin = (true_labels_tensor >= 10).to(torch.long)

            f1_micro = f1_score(pred_bin, true_bin, average="micro", task="binary").item()
            f1_macro = f1_score(pred_bin, true_bin, average="macro", task="binary", num_classes=2).item()
            
            # --- Print Detailed Report for this Model ---
            print(f"--- Results for: {model_path.name} ---")
            print(f"  Regression Metrics (Total Score):")
            print(f"    MAE:   {mae:.4f}")
            print(f"    MSE:   {mse:.4f}")
            print(f"    RMSE:  {rmse:.4f}")
            print(f"  Classification Metrics (Binary, cutoff=10):")
            print(f"    F1 Micro: {f1_micro:.4f}")
            print(f"    F1 Macro: {f1_macro:.4f}")
            print("--------------------------------" + "-" * len(model_path.name))

            # Store all metrics in the dictionary
            model_results = {
                "model": model_path.name,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "f1_micro_binary": f1_micro,
                "f1_macro_binary": f1_macro
            }
            results_list.append(model_results)

        except Exception as e:
            print(f"Failed to evaluate model {model_path.name}. Error: {e}")
            
    # 5. Report and Save the best model
    if results_list:
        print("\n--- Evaluation Complete ---")
        
        # Sort results by MAE (lowest first)
        results_list.sort(key=lambda x: x["mae"])
        
        best_model = results_list[0]
        
        print("\n--- Best Model (by MAE) ---")
        print(f"Model: {best_model['model']}")
        print(f"MAE:   {best_model['mae']:.4f}")
        print(f"MSE:   {best_model['mse']:.4f}")
        print(f"RMSE:  {best_model['rmse']:.4f}")
        print(f"F1 Macro: {best_model['f1_macro_binary']:.4f}")

        # --- Save results to JSON file ---
        final_output = {
            "best_model_by_mae": best_model,
            "all_results": results_list
        }
        
        print(f"\nSaving results to {RESULTS_JSON_PATH}...")
        with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4)
        
        print("Done.")

    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()