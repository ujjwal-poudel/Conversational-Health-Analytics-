import os
import re
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

"""
Cleaning function
"""
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)          # remove bracket content
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'<.*?>', '', text)            # remove HTML
    text = re.sub(r'\s+', ' ', text)             # collapse spaces
    return text.strip()


"""
Load model and prototype embeddings
"""
MODEL_PATH = "/Volumes/MACBACKUP/embeddings"
PROTOTYPES_DIR = "/Volumes/MACBACKUP/prototypes"

def load_components():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Load model
    if os.path.exists(MODEL_PATH):
        model = SentenceTransformer(MODEL_PATH, device=device)
    else:
        print("Warning: Local model not found, using base MiniLM model...")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Load prototypes
    emb0 = torch.tensor(np.load(os.path.join(PROTOTYPES_DIR, "prototype_emb_0.npy"))).to(device)
    emb1 = torch.tensor(np.load(os.path.join(PROTOTYPES_DIR, "prototype_emb_1.npy"))).to(device)

    return model, emb0, emb1, device


"""
Prediction using nearest prototype
"""
def predict_class(model, emb0, emb1, text, device):
    cleaned = clean_text(text)

    embedding = model.encode(cleaned, convert_to_tensor=True)
    embedding = embedding.to(device)

    sim0 = util.cos_sim(embedding, emb0).item()
    sim1 = util.cos_sim(embedding, emb1).item()

    return 0 if sim0 > sim1 else 1


"""
Load test.jsonl
"""
def load_test_data(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if "labels" not in data or not isinstance(data["labels"], list):
                continue

            try:
                score_value = sum(data["labels"])
            except Exception:
                continue

            if "turns" not in data or not isinstance(data["turns"], list):
                continue

            conversation_text = " ".join(data["turns"])
            conversation_text = clean_text(conversation_text)

            dataset.append((conversation_text, score_value))

    return dataset


"""
Evaluation logic
"""
def evaluate(model, emb0, emb1, dataset, device, threshold=10):

    preds = []
    labels = []

    for text, score_value in dataset:
        pred = predict_class(model, emb0, emb1, text, device)
        preds.append(pred)

        true_label = 0 if score_value < threshold else 1
        labels.append(true_label)

    preds = np.array(preds)
    labels = np.array(labels)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=4)

    return {
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "predictions": preds.tolist(),
        "true_labels": labels.tolist()
    }


"""
Save results to output file
"""
def save_results(results, out="raw_results.txt"):
    with open(out, "w", encoding="utf-8") as f:
        f.write("Raw Prototype Model Evaluation\n\n")

        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n\n")

        f.write("Confusion Matrix (rows=actual, cols=predicted):\n")
        f.write(str(results["confusion_matrix"]) + "\n\n")

        f.write("Classification Report:\n")
        f.write(results["classification_report"] + "\n\n")

        f.write("Predictions:\n")
        f.write(str(results["predictions"]) + "\n\n")

        f.write("True Labels:\n")
        f.write(str(results["true_labels"]) + "\n\n")

    print(f"Saved results to {out}")


"""
Main entry
"""
if __name__ == "__main__":
    model, emb0, emb1, device = load_components()

    test_data_path = "/Volumes/MACBACKUP/data/json/lines/test.jsonl"
    test_data = load_test_data(test_data_path)

    results = evaluate(model, emb0, emb1, test_data, device)
    save_results(results)