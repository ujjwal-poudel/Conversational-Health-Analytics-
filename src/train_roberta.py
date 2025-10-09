import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler, AutoConfig
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
from pathlib import Path
import time

from preprocess_dataset import RobertaPreprocessor

MODEL_NAME = 'roberta-base'
DATA_DIR = Path("/Volumes/MACBACKUP/final_datasets/")
CHECKPOINT_PATH = Path("./models/roberta")
LEARNING_RATE = 1e-5 # RoBERTa often benefits from a slightly smaller learning rate
EPOCHS = 20
BATCH_SIZE = 4
EARLY_STOPPING_PATIENCE = 5

class DepressionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
    
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)
            outputs = model(**{k: v.to(device) for k, v in batch.items() if k != 'labels'})
            
            # Manual loss calculation is not strictly needed here but can be added for consistency
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, accuracy, f1


def plot_metrics(history):
    output_dir = Path("./plots/roberta")
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('RoBERTa: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'training_validation_loss.png')
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_accuracy'], 'c-o', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'g-o', label='Validation Accuracy')
    plt.plot(epochs, history['val_f1'], 'm-o', label='Validation F1-Score')
    plt.title('RoBERTa: Training/Validation Accuracy & F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'metrics.png')
    print(f"Saved plots to {output_dir}")
    
    
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} for training.")

    # Using RobertaPreprocessor
    preprocessor = RobertaPreprocessor()
    
    # Data loading
    print("Loading and preprocessing data...")
    X_train_df, y_train_df = preprocessor.load_and_preprocess(DATA_DIR / "final_train_dataset.csv")
    X_dev_df, y_dev_df = preprocessor.load_and_preprocess(DATA_DIR / "final_dev_dataset.csv")
    
    X_train_chunked, y_train_chunked = preprocessor.chunk_dataframe(X_train_df, y_train_df, overlap=100)
    X_dev_chunked, y_dev_chunked = preprocessor.chunk_dataframe(X_dev_df, y_dev_df, overlap=100)
    
    train_encodings = preprocessor.tokenize(X_train_chunked['text'])
    dev_encodings = preprocessor.tokenize(X_dev_chunked['text'])
    
    train_dataset = DepressionDataset(train_encodings, y_train_chunked['label'].tolist())
    dev_dataset = DepressionDataset(dev_encodings, y_dev_chunked['label'].tolist())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    # Calculate Class Weights
    print("Calculating class weights for imbalanced dataset...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_chunked['label']),
        y=y_train_chunked['label'].to_numpy()
    )
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Calculated weights: {class_weights}")

    # Model setup with dropout is the same as distillbert
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2, attention_dropout=0.2, dropout=0.2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Defining the loss function with our calculated weights
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    print("Starting training with RoBERTa and class weighting...")
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_train_loss, epoch_preds, epoch_labels = 0, [], []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            labels = batch['labels'].to(device)
            # not passing labels to the model, as we'll calculate loss manually
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            
            # Manual loss calculation
            loss = loss_fct(outputs.logits, labels)
            
            total_train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.set_postfix(loss=loss.item())

        # Rest of the loop: metrics, printing, checkpointing is the same as distilbert
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(epoch_labels, epoch_preds)
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)

        print("Evaluating on development set...")
        val_loss, val_accuracy, val_f1 = evaluate(model, dev_loader, device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

        if val_accuracy > best_val_accuracy:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}. Saving model...")
            best_val_accuracy = val_accuracy
            patience_counter = 0
            CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(CHECKPOINT_PATH)
            preprocessor.tokenizer.save_pretrained(CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epoch(s).")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Stopping early after {patience_counter} epochs with no improvement.")
            break

    plot_metrics(history)
    

if __name__ == "__main__":
    main()