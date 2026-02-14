"""
Generate Multi-Target Model Training Graphs

Creates two graphs:
1. Train MAE vs Validation MAE over epochs
2. Validation F1 vs Test F1 over epochs (only val F1 available from logs)

Usage:
    python -m audio_ml_pipeline.src.evaluation.generate_multitarget_graphs
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Color theme with user-specified background
COLORS = {
    'background': '#040B10',
    'grid': '#1a2a3a',
    'text': '#FFFFFF',
    'text_secondary': '#8899AA',
    
    'train': '#FF6B6B',      # Red for training
    'val': '#4ECDC4',        # Teal for validation
    'test': '#00FF84',       # Green for test
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/multitarget_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training data from log (40 epochs)
# Columns: epoch, train_loss, train_loss_bin, train_loss_reg, dev_loss, dev_loss_bin, dev_loss_reg, 
#          train_acc(MAE), dev_acc(MAE), f1_micro, f1_macro, dev_mae, dev_mse
TRAINING_DATA = {
    'epochs': list(range(1, 41)),
    'train_mae': [
        1.1197, 0.8809, 0.7761, 0.7605, 0.7655, 0.7541, 0.7427, 0.7329, 0.7376, 0.7348,
        0.7394, 0.7411, 0.7499, 0.7181, 0.7334, 0.7350, 0.7278, 0.7253, 0.7192, 0.7038,
        0.7167, 0.6782, 0.6984, 0.6838, 0.6355, 0.6315, 0.6470, 0.6417, 0.6466, 0.6263,
        0.6351, 0.6039, 0.6332, 0.6164, 0.6146, 0.6073, 0.6045, 0.5944, 0.5599, 0.5995
    ],
    'val_mae': [
        3.9744, 3.8085, 3.7200, 3.8085, 3.8085, 3.7923, 3.8459, 3.8004, 3.6869, 3.6573,
        3.6900, 3.7604, 3.7732, 3.5704, 3.6159, 3.6203, 3.6401, 3.9233, 3.9416, 3.8013,
        3.6940, 3.7721, 3.7200, 3.7672, 4.0293, 3.8228, 3.8460, 3.9242, 4.0040, 3.8422,
        3.8830, 3.9524, 3.9109, 3.8991, 3.8978, 3.9337, 3.9342, 3.9505, 3.9593, 3.9479
    ],
    'val_f1_micro': [
        0.5714, 0.5455, 0.6000, 0.6364, 0.5455, 0.5000, 0.6957, 0.5714, 0.5714, 0.6667,
        0.5455, 0.5714, 0.5714, 0.6364, 0.6364, 0.5000, 0.6364, 0.6667, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000, 0.5714, 0.6364, 0.5000, 0.6364, 0.5263, 0.6957, 0.6364,
        0.6364, 0.5263, 0.6364, 0.6957, 0.6957, 0.7273, 0.6957, 0.6957, 0.6957, 0.6957
    ],
}

# Best epoch and test results
BEST_EPOCH = 14
TEST_MAE = 4.73
TEST_F1 = 0.68  # Approximate from presentation data
TEST_RMSE = 6.06

# R² calculation: R² = 1 - (MSE / Variance)
# PHQ target variance ≈ 56.25 (std ≈ 7.5)
TARGET_VARIANCE = 56.25

# MSE values at best epoch (14) from training log
TRAIN_MSE_EPOCH14 = 3.34  # From train_loss_reg column
VAL_MSE_EPOCH14 = 23.07   # From dev_mse column
TEST_MSE = TEST_RMSE ** 2  # 36.72

# Computed R² values
TRAIN_R2 = 1 - (TRAIN_MSE_EPOCH14 / TARGET_VARIANCE)  # 0.94
VAL_R2 = 1 - (VAL_MSE_EPOCH14 / TARGET_VARIANCE)       # 0.59
TEST_R2 = 1 - (TEST_MSE / TARGET_VARIANCE)             # 0.35


def create_mae_graph():
    """Create Train MAE vs Validation MAE graph."""
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    epochs = TRAINING_DATA['epochs']
    train_mae = TRAINING_DATA['train_mae']
    val_mae = TRAINING_DATA['val_mae']
    
    # Plot lines
    ax.plot(epochs, train_mae, '-o', color=COLORS['train'], linewidth=2.5, 
            markersize=6, label='Train MAE', alpha=0.9)
    ax.plot(epochs, val_mae, '-s', color=COLORS['val'], linewidth=2.5, 
            markersize=6, label='Validation MAE', alpha=0.9)
    
    # Mark best epoch
    ax.axvline(x=BEST_EPOCH, color=COLORS['test'], linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Best Epoch ({BEST_EPOCH})')
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_title('Multi-Target Model: Train vs Validation MAE', fontsize=18, 
                 color=COLORS['text'], fontweight='bold', pad=20)
    
    ax.tick_params(colors=COLORS['text'], labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    ax.legend(loc='upper right', fontsize=12, facecolor=COLORS['background'], 
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    ax.set_xlim(0, 41)
    ax.set_ylim(0, 4.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'train_val_mae.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_f1_graph():
    """Create Validation F1 over epochs graph."""
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    epochs = TRAINING_DATA['epochs']
    val_f1 = TRAINING_DATA['val_f1_micro']
    
    # Plot line
    ax.plot(epochs, val_f1, '-o', color=COLORS['val'], linewidth=2.5, 
            markersize=6, label='Validation F1 (Micro)', alpha=0.9)
    
    # Add test F1 as horizontal reference line
    ax.axhline(y=TEST_F1, color=COLORS['test'], linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Test F1 ({TEST_F1:.2f})')
    
    # Mark best epoch
    ax.axvline(x=BEST_EPOCH, color=COLORS['train'], linestyle=':', linewidth=2, 
               alpha=0.7, label=f'Best Epoch ({BEST_EPOCH})')
    
    # Add F1 value annotation at epoch 14 (best epoch)
    best_f1_val = val_f1[BEST_EPOCH - 1]  # Index is epoch-1
    ax.annotate(f'Val F1: {best_f1_val:.2f}', 
                xy=(BEST_EPOCH, best_f1_val), 
                xytext=(BEST_EPOCH + 5, best_f1_val + 0.08),
                fontsize=13, color=COLORS['val'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['val'], lw=2))
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_ylabel('F1 Score (Micro)', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_title('Multi-Target Model: Validation F1 Score', fontsize=18, 
                 color=COLORS['text'], fontweight='bold', pad=20)
    
    ax.tick_params(colors=COLORS['text'], labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    ax.legend(loc='lower right', fontsize=12, facecolor=COLORS['background'], 
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])
    
    ax.set_xlim(0, 41)
    ax.set_ylim(0.4, 0.8)
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'val_f1.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_val_vs_test_graph():
    """Create Validation vs Test MAE and RMSE comparison bar chart."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['background'])
    
    # Data at best epoch (14)
    val_mae = 3.5704
    test_mae = 4.73
    val_mse = 23.07
    val_rmse = np.sqrt(val_mse)  # 4.80
    test_rmse = 6.06
    
    # Plot 1: MAE Comparison
    ax1.set_facecolor(COLORS['background'])
    
    categories = ['Validation', 'Test']
    mae_values = [val_mae, test_mae]
    colors = [COLORS['val'], COLORS['test']]
    
    bars1 = ax1.bar(categories, mae_values, color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, val in zip(bars1, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=14, color=COLORS['text'], fontweight='bold')
    
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=13, color=COLORS['text'], fontweight='bold')
    ax1.set_title('MAE: Validation vs Test', fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    ax1.set_ylim(0, 6)
    ax1.tick_params(colors=COLORS['text'], labelsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], axis='y')
    
    # Plot 2: RMSE Comparison
    ax2.set_facecolor(COLORS['background'])
    
    rmse_values = [val_rmse, test_rmse]
    bars2 = ax2.bar(categories, rmse_values, color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, val in zip(bars2, rmse_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=14, color=COLORS['text'], fontweight='bold')
    
    ax2.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=13, color=COLORS['text'], fontweight='bold')
    ax2.set_title('RMSE: Validation vs Test', fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    ax2.set_ylim(0, 8)
    ax2.tick_params(colors=COLORS['text'], labelsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], axis='y')
    
    # Main title
    fig.suptitle(f'Multi-Target Model: Validation vs Test (Best Epoch {BEST_EPOCH})',
                fontsize=18, color=COLORS['text'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(OUTPUT_DIR, 'val_vs_test_mae_rmse.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_r2_comparison_graph():
    """Create R² comparison bar chart for Train/Val/Test."""
    
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Data
    categories = ['Train', 'Validation', 'Test']
    r2_values = [TRAIN_R2, VAL_R2, TEST_R2]
    
    # Low contrast, muted colors
    bar_colors = ['#C08070', '#70A0A0', '#90B090']  # Muted salmon, muted teal, sage green
    
    # Create bars
    bars = ax.bar(categories, r2_values, color=bar_colors, edgecolor='#444444', linewidth=1.5, width=0.6)
    
    # Add value labels
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=16, color=COLORS['text'], fontweight='bold')
    
    # Styling
    ax.set_ylabel('R² Score (Higher is Better)', fontsize=14, color=COLORS['text'], fontweight='bold')
    ax.set_title(f'Multi-Target Model: R² Comparison (Best Epoch {BEST_EPOCH})', fontsize=18, 
                 color=COLORS['text'], fontweight='bold', pad=20)
    
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors=COLORS['text'], labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'r2_comparison.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate all graphs."""
    print("=" * 60)
    print("Generating Multi-Target Model Training Graphs")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    print("\n1. Creating Train vs Validation MAE graph...")
    create_mae_graph()
    
    print("\n2. Creating Validation F1 graph...")
    create_f1_graph()
    
    print("\n3. Creating Validation vs Test MAE/RMSE comparison...")
    create_val_vs_test_graph()
    
    print("\n4. Creating R² comparison graph...")
    create_r2_comparison_graph()
    
    print("\n" + "=" * 60)
    print(f"Graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

