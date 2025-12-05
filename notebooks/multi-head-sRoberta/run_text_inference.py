"""
Simple Text Model Inference for Fusion

Runs the best text model (Epoch 15) on all 47 test samples 
from test.jsonl and saves predictions for multimodal fusion.

Usage:
    python run_text_inference.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Import model class
import sys
sys.path.insert(0, str(Path(__file__).parent))
from hdsc.model import PHQTotalMulticlassAttentionModelBERT

# =============================================================================
# CONFIGURATION - Update these paths as needed
# =============================================================================
# Log epoch 14 has best dev_mae (3.5704) = model_2_13.pt (0-indexed)
MODEL_PATH = "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_/model_2_13.pt"
TEST_JSONL_PATH = "/Users/ujjwalpoudel/Documents/insane_projects/Conversational-Health-Analytics-/backend/src/original_data/test.jsonl"
OUTPUT_PATH = "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_/preds_2_13_full.json"
TOKENIZER_NAME = "sentence-transformers/all-distilroberta-v1"


def set_device():
    """Set compute device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using: CUDA")
    else:
        device = torch.device("cpu")
        print("Using: CPU")
    return device


def load_model(model_path, device):
    """Load the trained model."""
    print(f"Loading model: {Path(model_path).name}")
    loaded_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    model = PHQTotalMulticlassAttentionModelBERT(
        device=device,
        **loaded_dict["kwargs"],
    )
    model.load_state_dict(loaded_dict["model"])
    model = model.to(device)
    model.eval()
    
    return model


def load_test_data(filepath):
    """Load test.jsonl file."""
    print(f"Loading test data from: {filepath}")
    
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append({
                    'id': data.get('id'),
                    'turns': data['turns'],
                    'labels': data['labels']  # 8 PHQ item scores
                })
    
    print(f"Loaded {len(samples)} test samples")
    return samples


@torch.no_grad()
def run_inference(model, tokenizer, samples, device):
    """Run inference on all samples."""
    
    all_preds = []
    all_true = []
    all_ids = []
    
    print("\nRunning inference...")
    for sample in tqdm(samples):
        turns = sample['turns']
        
        # Tokenize each turn separately
        encoded = tokenizer(
            turns,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Build batch dict expected by model
        batch = {
            'input_ids': encoded['input_ids'].to(device),
            'attention_mask': encoded['attention_mask'].to(device),
            'text_lens': torch.tensor([len(turns)], device=device)
        }
        
        # Forward pass
        try:
            pred = model(batch)
            pred_np = pred.cpu().numpy().squeeze().tolist()
            # Ensure it's a list of 8 values
            if isinstance(pred_np, float):
                pred_np = [pred_np]
        except Exception as e:
            print(f"Error on sample {sample['id']}: {e}")
            pred_np = [0.0] * 8  # Default
        
        all_preds.append(pred_np)
        all_true.append(sample['labels'])
        all_ids.append(sample['id'])
    
    return all_preds, all_true, all_ids


def compute_metrics(preds, true_labels):
    """Compute MAE and RMSE on total scores."""
    preds_sum = np.sum(preds, axis=1)
    true_sum = np.sum(true_labels, axis=1)
    
    mae = np.mean(np.abs(preds_sum - true_sum))
    rmse = np.sqrt(np.mean((preds_sum - true_sum) ** 2))
    
    return mae, rmse


def main():
    print("=" * 60)
    print("Text Model Inference (Epoch 15)")
    print("=" * 60)
    
    device = set_device()
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Load test data
    samples = load_test_data(TEST_JSONL_PATH)
    
    # Run inference
    preds, true_labels, ids = run_inference(model, tokenizer, samples, device)
    
    # Compute metrics
    preds_np = np.array(preds)
    true_np = np.array(true_labels)
    mae, rmse = compute_metrics(preds_np, true_np)
    
    print(f"\n{'=' * 40}")
    print(f"Results on {len(preds)} samples:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"{'=' * 40}")
    
    # Save predictions
    results = {
        'pred': preds,
        'true': true_labels,
        'ids': ids,
        'n_samples': len(preds),
        'mae': float(mae),
        'rmse': float(rmse),
        'model': 'model_2_15.pt'
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved predictions to: {OUTPUT_PATH}")
    print("Ready for multimodal fusion!")


if __name__ == "__main__":
    main()
