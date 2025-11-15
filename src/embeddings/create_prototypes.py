#!/usr/bin/env python3

"""
create_prototypes.py

This script processes a JSONL file of conversational data, calculates a total 
PHQ8 score for each conversation, and then generates two "prototype" embeddings
(one for scores < 10 and one for scores >= 10) using a sentence-transformer 
model.

It saves:
1. The full sentence-transformer model.
2. The two prototype embedding vectors as .npy files.

To run:
    pip install numpy sentence-transformers
    python create_prototypes.py
"""

import json
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---

# Path to your input data file
JSONL_FILE_PATH = '/Volumes/MACBACKUP/data/json/lines/train.jsonl'

# Directory where the model and prototype vectors will be saved
MODEL_SAVE_PATH = '/Volumes/MACBACKUP/embeddings'

# The pre-trained sentence-transformer model to use
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# The PHQ8 score threshold
SCORE_THRESHOLD = 10


def clean_text(text):
    """
    Applies "janitorial" cleaning to the text.
    - Converts to lowercase
    - Removes artifacts like [laughter]
    - Removes URLs and HTML tags
    - Collapses whitespace
    """
    text = text.lower()
    
    # 2. Remove [laughter], [sighs], etc. (Corrected regex)
    text = re.sub(r'\[.*?\]', '', text) 
    
    # 3. Remove URLs (Corrected regex)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 5. Collapse whitespace (Corrected regex)
    text = re.sub(r'\s+', ' ', text)
    
    # 6. Remove leading/trailing whitespace
    return text.strip()


def load_and_process_data(file_path):
    """
    Reads the JSONL file, processes labels and text, and sorts
    text into two classes based on the score threshold.
    """
    texts_class_0 = []
    texts_class_1 = []

    print(f"Reading and processing {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # --- 1. Process Labels ---
                    
                    # Check for 'labels' key
                    if 'labels' not in data:
                        continue
                    
                    individual_scores = data['labels']
                    
                    # Check if it's a non-empty list
                    if not isinstance(individual_scores, list) or not individual_scores:
                        continue
                    
                    # Sum the list to get the final score
                    score_value = sum(individual_scores)

                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    # Catches bad JSON, non-numeric list items, or missing keys
                    continue 
                
                # --- 2. Process and Clean Text ---
                if 'turns' not in data or not isinstance(data['turns'], list):
                    continue

                full_conversation_text = " ".join(data['turns'])
                full_conversation_text = clean_text(full_conversation_text)

                # --- 3. Apply Rule and Sort ---
                if score_value >= SCORE_THRESHOLD:
                    texts_class_1.append(full_conversation_text)
                else:
                    texts_class_0.append(full_conversation_text)

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None

    print(f"Processing complete.")
    print(f"Found {len(texts_class_0)} documents for Class 0 (Score < {SCORE_THRESHOLD}).")
    print(f"Found {len(texts_class_1)} documents for Class 1 (Score >= {SCORE_THRESHOLD}).")
    
    return texts_class_0, texts_class_1


def create_and_save_prototypes(model, save_path, texts_class_0, texts_class_1):
    """
    Encodes texts, calculates mean prototypes, and saves
    both the model and the prototype vectors.
    """
    
    # --- 1. Generate Embeddings ---
    print("Generating embeddings for Class 0...")
    embeddings_class_0 = model.encode(texts_class_0, show_progress_bar=True)

    print("Generating embeddings for Class 1...")
    embeddings_class_1 = model.encode(texts_class_1, show_progress_bar=True)

    # --- 2. Create the Prototype (Centroid) ---
    print("Averaging embeddings to create prototypes...")
    emb_0 = np.mean(embeddings_class_0, axis=0)
    emb_1 = np.mean(embeddings_class_1, axis=0)
    
    print(f"Class 0 prototype shape: {emb_0.shape}")
    print(f"Class 1 prototype shape: {emb_1.shape}")

    # --- 3. Save Everything for Production ---
    
    # Create the target directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Saving model to {save_path}...")
    model.save(save_path)

    # Define full paths for the .npy files
    proto_0_path = os.path.join(save_path, 'prototype_emb_0.npy')
    proto_1_path = os.path.join(save_path, 'prototype_emb_1.npy')

    print(f"Saving prototype vectors to {save_path}...")
    np.save(proto_0_path, emb_0)
    np.save(proto_1_path, emb_1)

    print("\nSetup complete. Model and prototypes are saved.")


def main():
    """
    Main function to orchestrate the entire process.
    """
    # --- 1. Load your Embedding Model ---
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # --- 2. Process the .jsonl File ---
    texts_class_0, texts_class_1 = load_and_process_data(JSONL_FILE_PATH)

    # --- 3. Check if we found any data ---
    if not texts_class_0 or not texts_class_1:
        print("\n--- !! WARNING !! ---")
        print("One or both classes have 0 documents.")
        print("This could be due to a file error or no data matching the criteria.")
        print("------------------------")
        return # Exit the script

    # --- 4. Generate, Average, and Save Embeddings ---
    create_and_save_prototypes(model, MODEL_SAVE_PATH, texts_class_0, texts_class_1)


if __name__ == "__main__":
    main()