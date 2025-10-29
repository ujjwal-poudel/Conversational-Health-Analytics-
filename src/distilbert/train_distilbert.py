import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AdamW, get_scheduler, AutoConfig
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
from pathlib import Path

# Importing the custom preprocessor
from preprocess_dataset import DistilBertPreprocessor

# CONFIGURATION
MODEL_NAME = 'distilbert-base-uncased'
DATA_DIR = Path("/Volumes/MACBACKUP/final_datasets/")
SAVE_PATH = Path("/models/")
CHECKPOINT_PATH = Path("./models")
LEARNING_RATE = 3e-5
EPOCHS = 20
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 3

# DATA PREPARATION
#  PYTORCH DATASET CLASS
class DepressionDataset(Dataset):
    """Custom Dataset class for our depression text data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # The encodings are already tensors from the tokenizer
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate(model, dataloader, device):
    """
    Evaluates the model on a given dataset.
    
    RETURNS: validation loss, accuracy, and f1 score.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Gets model outputs, which include the loss when labels are provided
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def plot_metrics(history):
    """Generates and saves plots for training metrics."""
    output_dir = Path("./plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)

    # Plots for Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'distilbert/training_validation_loss.png')
    print(f"Saved training and validation loss plot to {output_dir}")

    # Plots for Accuracy and F1-Score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_accuracy'], 'c-o', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'g-o', label='Validation Accuracy')
    plt.plot(epochs, history['val_f1'], 'm-o', label='Validation F1-Score')
    plt.title('Training/Validation Accuracy & F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'distilbert/metrics.png')
    print(f"Saved metrics plot to {output_dir}")

# MAIN TRAINING AND EVALUATION SCRIPT
def main():
    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (GPU) for training.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU for training.")
    
    # Data Loading and Preprocessing
    preprocessor = DistilBertPreprocessor()
    
    # Loading raw data
    print("Loading and preprocessing data...")
    X_train_df, y_train_df = preprocessor.load_and_preprocess(DATA_DIR / "final_train_dataset.csv")
    X_dev_df, y_dev_df = preprocessor.load_and_preprocess(DATA_DIR / "final_dev_dataset.csv")

    # Chunking the dataframes
    X_train_chunked, y_train_chunked = preprocessor.chunk_dataframe(X_train_df, y_train_df, overlap=100)
    X_dev_chunked, y_dev_chunked = preprocessor.chunk_dataframe(X_dev_df, y_dev_df, overlap=100)

    # Tokenizing the final chunked text
    train_encodings = preprocessor.tokenize(X_train_chunked['text'])
    dev_encodings = preprocessor.tokenize(X_dev_chunked['text'])

    # Creating PyTorch Datasets
    train_dataset = DepressionDataset(train_encodings, y_train_chunked['label'].tolist())
    dev_dataset = DepressionDataset(dev_encodings, y_dev_chunked['label'].tolist())

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    
    print("Setting up model with dropout regularization...")

    # Creating a custom configuration with dropout
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        attention_dropout=0.3, # Dropout in the multi-head attention layers
        dropout=0.2            # Dropout in the final classifier layer
    )
    
    # Loads the model with the new custom configuration
    # Model, Optimizer, and Scheduler Setup
    print("Setting up model, optimizer, and scheduler...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.to(device) # Moves model to GPU
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # Training & Evaluation Loop
    print("Starting training...")
    start_time = time.time()
    
    # Dictionary to store metrics history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_val_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.train()
        total_train_loss = 0
        
        # Lists to store predictions and labels for the epoch
        epoch_preds = []
        epoch_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            
            # Gets predictions and store them
            preds = torch.argmax(outputs.logits, dim=-1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.set_postfix(loss=loss.item())

        # Calculates and store metrics
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

        # Early stopping logic is the same
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
        
    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")

    # Generates and save plots
    plot_metrics(history)



if __name__ == "__main__":
    print("hello world")
    # main()