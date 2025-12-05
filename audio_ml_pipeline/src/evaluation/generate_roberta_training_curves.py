"""
Generate Multi-head RoBERTa Training & Validation Curves

Multi-label REGRESSION task (8 PHQ items)

Creates minimal visualizations for training progress:
1. Training vs Validation Loss
2. Validation MAE (Regression)
3. Validation MSE (Regression)
4. F1 Scores (Binary classification from sum > 9)

Reads from TSV training log file.

Usage:
    python -m src.evaluation.generate_roberta_training_curves
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Same color theme
COLORS = {
    'background': '#0A0A12',
    'card_bg': '#101020',
    'grid': '#2A2A4A',
    'text': '#FFFFFF',
    'text_secondary': '#8888AA',
    
    'train': '#0059FF',     # Blue for training
    'val': '#00FFFF',       # Cyan for validation
    'success': '#00FF84',   # Green accent
}

# Paths
TSV_PATH = "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_/log_robert_multilabel_no-regression__2.tsv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/roberta_training_curves")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_training_log():
    """Load training log from TSV file."""
    df = pd.read_csv(TSV_PATH, sep='\t')
    return df


def create_loss_curves():
    """Create training vs validation loss curves."""
    
    df = load_training_log()
    
    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    dev_loss = df['dev_loss'].values
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot lines
    ax.plot(epochs, train_loss, color=COLORS['train'], linewidth=2, 
            marker='o', markersize=5, label='Training Loss', alpha=0.9)
    ax.plot(epochs, dev_loss, color=COLORS['val'], linewidth=2,
            marker='o', markersize=5, label='Validation Loss', alpha=0.9)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Loss (Lower is Better)', fontsize=12, color=COLORS['text'])
    ax.set_title('Multi-head Distilled RoBERTa: Training vs Validation Loss',
                fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, 
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'loss_curves.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_mae_curves():
    """Create validation MAE curve."""
    
    df = load_training_log()
    
    epochs = df['epoch'].values
    dev_mae = df['dev_mae'].values
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot line
    ax.plot(epochs, dev_mae, color=COLORS['val'], linewidth=2, 
            marker='o', markersize=5, label='Validation MAE', alpha=0.9)
    
    # Find best epoch
    best_epoch_idx = np.argmin(dev_mae)
    best_epoch = epochs[best_epoch_idx]
    best_mae = dev_mae[best_epoch_idx]
    
    ax.scatter([best_epoch], [best_mae], color=COLORS['success'], 
               s=150, zorder=5, marker='o', edgecolors='white', linewidth=2)
    
    ax.annotate(f'Best: {best_mae:.3f}\nEpoch {best_epoch}', 
               xy=(best_epoch, best_mae), xytext=(best_epoch + 2, best_mae + 0.2),
               fontsize=10, color=COLORS['success'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('MAE (Lower is Better)', fontsize=12, color=COLORS['text'])
    ax.set_title('Multi-head Distilled RoBERTa: Validation MAE Over Training',
                fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, 
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'mae_curve.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_mse_curves():
    """Create validation MSE curve."""
    
    df = load_training_log()
    
    epochs = df['epoch'].values
    dev_mse = df['dev_mse'].values
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Plot line
    ax.plot(epochs, dev_mse, color=COLORS['train'], linewidth=2, 
            marker='o', markersize=5, label='Validation MSE', alpha=0.9)
    
    # Find best epoch
    best_epoch_idx = np.argmin(dev_mse)
    best_epoch = epochs[best_epoch_idx]
    best_mse = dev_mse[best_epoch_idx]
    
    ax.scatter([best_epoch], [best_mse], color=COLORS['success'], 
               s=150, zorder=5, marker='o', edgecolors='white', linewidth=2)
    
    ax.annotate(f'Best: {best_mse:.2f}\nEpoch {best_epoch}', 
               xy=(best_epoch, best_mse), xytext=(best_epoch + 2, best_mse + 2),
               fontsize=10, color=COLORS['success'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('MSE (Lower is Better)', fontsize=12, color=COLORS['text'])
    ax.set_title('Multi-head Distilled RoBERTa: Validation MSE Over Training',
                fontsize=14, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, 
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'mse_curve.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_mae_mse_combined():
    """Create MAE and MSE on same plot."""
    
    df = load_training_log()
    
    epochs = df['epoch'].values
    dev_mae = df['dev_mae'].values
    dev_mse = df['dev_mse'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['background'])
    
    # Plot 1: MAE
    ax1.set_facecolor(COLORS['background'])
    ax1.plot(epochs, dev_mae, color=COLORS['val'], linewidth=2, 
            marker='o', markersize=5, label='Validation MAE', alpha=0.9)
    
    best_mae_idx = np.argmin(dev_mae)
    ax1.scatter([epochs[best_mae_idx]], [dev_mae[best_mae_idx]], 
               color=COLORS['success'], s=150, zorder=5, marker='o', 
               edgecolors='white', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=11, color=COLORS['text'])
    ax1.set_ylabel('MAE (Lower is Better)', fontsize=11, color=COLORS['text'])
    ax1.set_title('Validation MAE', fontsize=12, color=COLORS['text'], fontweight='bold', pad=10)
    ax1.legend(fontsize=10, facecolor=COLORS['background'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'], framealpha=0.9)
    ax1.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])
    
    # Plot 2: MSE
    ax2.set_facecolor(COLORS['background'])
    ax2.plot(epochs, dev_mse, color=COLORS['train'], linewidth=2, 
            marker='o', markersize=5, label='Validation MSE', alpha=0.9)
    
    best_mse_idx = np.argmin(dev_mse)
    ax2.scatter([epochs[best_mse_idx]], [dev_mse[best_mse_idx]], 
               color=COLORS['success'], s=150, zorder=5, marker='o', 
               edgecolors='white', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=11, color=COLORS['text'])
    ax2.set_ylabel('MSE (Lower is Better)', fontsize=11, color=COLORS['text'])
    ax2.set_title('Validation MSE', fontsize=12, color=COLORS['text'], fontweight='bold', pad=10)
    ax2.legend(fontsize=10, facecolor=COLORS['background'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'], framealpha=0.9)
    ax2.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'mae_mse_combined.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate regression training curve plots."""
    print("=" * 60)
    print("Generating Multi-head RoBERTa Regression Training Curves")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Load and show summary
    df = load_training_log()
    print(f"\nTotal epochs: {len(df)}")
    print(f"Best validation MAE: {df['dev_mae'].min():.4f} at epoch {df['dev_mae'].idxmin() + 1}")
    print(f"Best validation MSE: {df['dev_mse'].min():.4f} at epoch {df['dev_mse'].idxmin() + 1}")
    print(f"Final validation MAE: {df['dev_mae'].iloc[-1]:.4f}")
    print(f"Final validation MSE: {df['dev_mse'].iloc[-1]:.4f}")
    
    create_loss_curves()
    create_mae_curves()
    create_mse_curves()
    create_mae_mse_combined()
    
    print("\n" + "=" * 60)
    print(f"All graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
