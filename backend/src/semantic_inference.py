import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

class SemanticClassifier:
    def __init__(self, model_path: str, prototypes_dir: str):
        """
        Initializes the semantic classifier by loading the SentenceTransformer model
        and the pre-computed prototype vectors.
        
        Args:
            model_path (str): Path to the saved SentenceTransformer model (or model name).
            prototypes_dir (str): Directory containing the .npy prototype files.
        """
        # Device Selection Logic
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        print(f"Loading Semantic Model on {self.device}...")
        
        # 1. Load the Model
        # We load from local path if it exists, otherwise download the default
        if os.path.exists(model_path):
            self.model = SentenceTransformer(model_path, device=self.device)
        else:
            print(f"Warning: Local model not found at {model_path}. Downloading base model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

        # 2. Load the Prototypes
        print("Loading Prototype Vectors...")
        try:
            self.emb_0 = np.load(os.path.join(prototypes_dir, 'prototype_emb_0.npy'))
            self.emb_1 = np.load(os.path.join(prototypes_dir, 'prototype_emb_1.npy'))
            
            # Ensure they are converted to tensors for efficient calculation with sentence-transformers
            self.emb_0 = torch.tensor(self.emb_0).to(self.device)
            self.emb_1 = torch.tensor(self.emb_1).to(self.device)
            
            print("Semantic Classifier initialized successfully.")
        except FileNotFoundError as e:
            print(f"CRITICAL ERROR: Could not load prototype files from {prototypes_dir}")
            print("Make sure you ran 'create_prototypes.py' first.")
            raise e

    def clean_text(self, text: str) -> str:
        """
        Applies the SAME "janitorial" cleaning used during training.
        CRITICAL: Do not change this function unless you change it in training too.
        """
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)      # Remove [laughter], etc.
        text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
        text = re.sub(r'<.*?>', '', text)        # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)         # Collapse whitespace
        return text.strip()

    def predict(self, conversation_text: str):
        """
        Generates an embedding for the input text and calculates similarity
        to the 'Non-Depressed' (Class 0) and 'Depressed' (Class 1) prototypes.
        
        Returns:
            dict: Contains similarity scores and the predicted class label.
        """
        # 1. Clean
        clean_input = self.clean_text(conversation_text)
        
        # 2. Encode (Create Embedding)
        # convert_to_tensor=True ensures output is a torch tensor on the correct device
        input_embedding = self.model.encode(clean_input, convert_to_tensor=True)
        
        # 3. Calculate Cosine Similarity
        # util.cos_sim returns a matrix, we just want the float value
        sim_0 = util.cos_sim(input_embedding, self.emb_0).item()
        sim_1 = util.cos_sim(input_embedding, self.emb_1).item()
        
        # 4. Determine Result
        predicted_class = 1 if sim_1 > sim_0 else 0
        class_label = "High Risk (Depressed) (>=10)" if predicted_class == 1 else "Low Risk (<10) (Non-Depressed)"
        
        print("\n" + "="*40)
        print(" SEMANTIC SIMILARITY REPORT ".center(40))
        print("="*40)
        print(f"Similarity to Class 0 (Low Risk):  {sim_0:.4f}")
        print(f"Similarity to Class 1 (High Risk): {sim_1:.4f}")
        print("-" * 40)
        if predicted_class == 1:
            print(f"result: MORE SIMILAR TO CLASS 1 (High Risk)")
        else:
            print(f"result: MORE SIMILAR TO CLASS 0 (Low Risk)")
        print("="*40 + "\n")

        return {
            "similarity_class_0": round(sim_0, 4), # Similarity to Non-Depressed
            "similarity_class_1": round(sim_1, 4), # Similarity to Depressed
            "predicted_class": predicted_class,
            "predicted_label": class_label
        }

# --- Singleton Instance Management ---
# This helps you easily import and use a shared instance in FastAPI

_classifier_instance = None

def get_semantic_classifier():
    """
    Returns the singleton instance of the classifier. Initializes it if necessary.
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        MODEL_PATH = '/Volumes/MACBACKUP/embeddings'
        PROTOTYPES_DIR = '/Volumes/MACBACKUP/prototypes'
        
        _classifier_instance = SemanticClassifier(MODEL_PATH, PROTOTYPES_DIR)
        
    return _classifier_instance

if __name__ == "__main__":
    # Tests the script independently
    classifier = get_semantic_classifier()
    
    test_text = "I've been feeling excited lately and I enjoy my hobbies way too much."
    result = classifier.predict(test_text)
    
    print("\n--- Test Result ---")
    print(f"Input: {test_text}")
    print(f"Similarity to Class 0 (Healthy):   {result['similarity_class_0']}")
    print(f"Similarity to Class 1 (Depressed): {result['similarity_class_1']}")
    print(f"Prediction: {result['predicted_label']}")