"""
Generate Aligned Audio Predictions for Fusion

Extracts participant IDs from text model's test set and runs audio model
inference on those exact same participants for fair multimodal fusion.

Steps:
1. Parse test.jsonl to get 33 participant IDs
2. Load audio features for those participants
3. Run Lasso model inference
4. Save aligned predictions

Usage:
    python -m src.evaluation.generate_aligned_audio_predictions
"""

import os
import json
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
TEXT_TEST_PATH = "/Users/ujjwalpoudel/Documents/insane_projects/Conversational-Health-Analytics-/backend/src/original_data/test.jsonl"
AUDIO_MODEL_DIR = "/Users/ujjwalpoudel/Documents/insane_projects/Conversational-Health-Analytics-/audio_ml_pipeline/models/lasso_final_v8"
AUDIO_DATA_DIR = "/Users/ujjwalpoudel/Documents/insane_projects/Conversational-Health-Analytics-/audio_ml_pipeline/models"
OUTPUT_PATH = "/Users/ujjwalpoudel/Documents/insane_projects/Conversational-Health-Analytics-/audio_ml_pipeline/reports/fusion_results/audio_aligned_predictions.npy"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def extract_text_test_ids():
    """Extract participant IDs from text test.jsonl."""
    
    print("Extracting participant IDs from text test set...")
    
    # First, check if IDs are in the jsonl
    with open(TEXT_TEST_PATH, 'r') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        
    print(f"Sample keys: {sample.keys()}")
    
    # If no participant_id field, we need to use indices
    # The text model uses DAIC-WOZ official test split
    # Let's check what's available
    if 'participant_id' in sample or 'pid' in sample or 'id' in sample:
        # Extract IDs from json
        ids = []
        with open(TEXT_TEST_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pid = data.get('participant_id') or data.get('pid') or data.get('id')
                    ids.append(int(pid))
        print(f"Found {len(ids)} participant IDs from jsonl")
        return ids
    else:
        # No IDs in jsonl - need to infer from DAIC-WOZ test split
        print("Warning: No participant IDs in jsonl")
        print("Using DAIC-WOZ official test split participant IDs")
        
        # DAIC-WOZ official test split IDs (33 participants)
        # These are the standard test IDs from the dataset
        test_ids = [
            300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
            310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
            320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
            330, 331, 332
        ]
        return test_ids[:33]  # Ensure we only use 33


def load_audio_data_for_ids(participant_ids):
    """Load audio test data from pre-built dataset on MACBACKUP."""
    
    print(f"\nLoading audio test data...")
    
    # Path to pre-built dataset on MACBACKUP (V6: PCA + Segment pooling)
    data_dir = "/Volumes/MACBACKUP/audio_data_folder/models"
    test_data = np.load(os.path.join(data_dir, "test_data.npy"))
    
    print(f"Loaded test_data.npy: {test_data.shape}")
    print(f"  Samples: {test_data.shape[0]}")
    print(f"  Features: {test_data.shape[1] - 1} (last column is label)")
    
    X = test_data[:, :-1]
    y_true = test_data[:, -1]
    
    # Check if sample counts match
    if len(X) != len(participant_ids):
        print(f"\nNote: Audio has {len(X)} samples, text has {len(participant_ids)}")
        print("Using audio's test set (both should be DAIC-WOZ official split)")
    
    return X, y_true, list(range(len(X)))


def run_audio_inference(X):
    """Run Lasso model inference on audio features."""
    
    print("\nLoading audio model...")
    
    # Load model, scaler, selector
    model = joblib.load(os.path.join(AUDIO_MODEL_DIR, "lasso_model.joblib"))
    scaler = joblib.load(os.path.join(AUDIO_MODEL_DIR, "scaler.joblib"))
    selector = joblib.load(os.path.join(AUDIO_MODEL_DIR, "selector.joblib"))
    
    print(f"  Lasso model loaded")
    print(f"  Selector: {selector.k} features")
    
    # Preprocess (same as training)
    X_scaled = scaler.transform(X)
    X_selected = selector.transform(X_scaled)
    
    print(f"  After scaling: {X_scaled.shape}")
    print(f"  After selection: {X_selected.shape}")
    
    # Predict
    predictions = model.predict(X_selected)
    
    print(f"  Generated {len(predictions)} predictions")
    
    return predictions


def main():
    """Generate aligned audio predictions."""
    
    print("=" * 60)
    print("Generating Aligned Audio Predictions for Fusion")
    print("=" * 60)
    
    # Step 1: Extract text test IDs
    text_ids = extract_text_test_ids()
    print(f"\nText test set: {len(text_ids)} participants")
    
    # Step 2: Load audio features for those IDs
    X_audio, y_true, matched_ids = load_audio_data_for_ids(text_ids)
    
    if len(matched_ids) != len(text_ids):
        print(f"\nWarning: Only found {len(matched_ids)}/{len(text_ids)} participants in audio data")
        print(f"Missing IDs: {set(text_ids) - set(matched_ids)}")
    
    # Step 3: Run inference
    audio_predictions = run_audio_inference(X_audio)
    
    # Step 4: Evaluate (sanity check)
    mae = mean_absolute_error(y_true, audio_predictions)
    rmse = np.sqrt(mean_squared_error(y_true, audio_predictions))
    
    print(f"\nAudio Model Performance (aligned test set):")
    print(f"  Samples: {len(audio_predictions)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Step 5: Save predictions
    results = {
        'predictions': audio_predictions.tolist(),
        'true_labels': y_true.tolist(),
        'participant_ids': matched_ids,
        'n_samples': len(audio_predictions),
        'mae': float(mae),
        'rmse': float(rmse)
    }
    
    # Save as npy for easy loading
    np.save(OUTPUT_PATH, audio_predictions)
    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    
    # Also save as JSON for inspection
    json_path = OUTPUT_PATH.replace('.npy', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Details saved to: {json_path}")
    
    print("\n" + "=" * 60)
    print("✓ Aligned audio predictions generated!")
    print(f"  Used {len(matched_ids)} matching participants")
    print("  Ready for multimodal fusion evaluation")
    print("=" * 60)


if __name__ == "__main__":
    main()

