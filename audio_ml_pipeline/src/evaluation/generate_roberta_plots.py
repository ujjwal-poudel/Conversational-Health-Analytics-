"""
Generate Multi-head RoBERTa Training Results Graphs

Creates minimal visualizations for Multi-head distilled RoBERTa model:
1. Training progression (MAE across 8 PHQ items by epoch)
2. F1 scores progression
3. Epoch comparison (first, best, middle, final)

Reads JSON prediction files from MACBACKUP.

Usage:
    python -m src.evaluation.generate_roberta_plots
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Same color theme
COLORS = {
    'background': '#0A0A12',
    'card_bg': '#101020',
    'grid': '#2A2A4A',
    'text': '#FFFFFF',
    'text_secondary': '#8888AA',
    
    'rmse': '#0059FF',      # Blue
    'mae': '#00FFFF',       # Cyan
    'r2': '#57FD53',        # Bright green
    'f1': '#00FF84',        # Green accent
    'success': '#00FF84',
}

# Paths
JSON_DIR = "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/roberta_model_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_epoch_from_filename(filename):
    """Extract epoch number from 'preds_2_15.json' format."""
    parts = Path(filename).stem.split('_')
    return int(parts[-1])


def compute_mae_from_json(json_path):
    """Compute MAE and MSE from pred/true JSON file (test set)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    preds = np.array(data['pred'])
    trues = np.array(data['true'])
    
    # Total score metrics
    pred_total = np.sum(preds, axis=1)
    true_total = np.sum(trues, axis=1)
    
    mae_total = np.mean(np.abs(pred_total - true_total))
    mse_total = np.mean((pred_total - true_total) ** 2)
    rmse_total = np.sqrt(mse_total)
    
    # Binary classification for F1 (threshold at 9)
    pred_bin = (pred_total > 9).astype(int)
    true_bin = (true_total > 9).astype(int)
    
    # F1 score (manual calculation)
    tp = np.sum((pred_bin == 1) & (true_bin == 1))
    fp = np.sum((pred_bin == 1) & (true_bin == 0))
    fn = np.sum((pred_bin == 0) & (true_bin == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'mae_total': mae_total,
        'mse_total': mse_total,
        'rmse_total': rmse_total,
        'f1_binary': f1,
        'precision': precision,
        'recall': recall
    }


def load_all_results():
    """Load and process all JSON pred files."""
    json_files = glob.glob(os.path.join(JSON_DIR, "preds_2_*.json"))
    
    results = []
    for json_file in json_files:
        epoch = extract_epoch_from_filename(json_file)
        metrics = compute_mae_from_json(json_file)
        metrics['epoch'] = epoch
        metrics['filename'] = os.path.basename(json_file)
        results.append(metrics)
    
    # Sort by epoch
    results = sorted(results, key=lambda x: x['epoch'])
    return results


def select_key_epochs(results, n_epochs=12):
    """Select key epochs to avoid clutter."""
    n_total = len(results)
    
    if n_total <= n_epochs:
        return results
    
    # Always include: first, last, best
    indices = {0, n_total - 1}
    
    # Add best MAE
    best_idx = min(range(n_total), key=lambda i: results[i]['mae_total'])
    indices.add(best_idx)
    
    # Add evenly spaced epochs
    step = n_total // (n_epochs - 3)
    for i in range(1, n_total - 1, step):
        indices.add(i)
        if len(indices) >= n_epochs:
            break
    
    selected = [results[i] for i in sorted(indices)]
    return selected


def create_mae_progression():
    """Create MAE/RMSE/MSE progression graph (test set)."""
    
    all_results = load_all_results()
    selected = select_key_epochs(all_results, n_epochs=12)
    
    epochs = [r['epoch'] for r in selected]
    mae_values = [r['mae_total'] for r in selected]
    rmse_values = [r['rmse_total'] for r in selected]
    
    # Best model info
    best = min(all_results, key=lambda x: x['mae_total'])
    best_epoch = best['epoch']
    
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    x = np.arange(len(epochs))
    
    # Plot lines
    ax.plot(x, mae_values, color=COLORS['mae'], linewidth=2, 
            marker='o', markersize=7, label='MAE', alpha=0.9)
    ax.plot(x, rmse_values, color=COLORS['rmse'], linewidth=2,
            marker='o', markersize=7, label='RMSE', alpha=0.9)
    
    # Highlight best epoch
    if best_epoch in epochs:
        best_idx = epochs.index(best_epoch)
        ax.scatter([best_idx], [mae_values[best_idx]], color=COLORS['success'], 
                   s=150, zorder=5, marker='o', edgecolors='white', linewidth=2)
    
    # Add MAE value labels
    for i, (ep, mae) in enumerate(zip(epochs, mae_values)):
        ax.annotate(f'{mae:.2f}', (i, mae), textcoords="offset points", 
                   xytext=(0, -18), ha='center', fontsize=9, color=COLORS['mae'],
                   fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels([f'Epoch {e}' for e in epochs], fontsize=9, 
                       color=COLORS['text'], rotation=45, ha='right')
    ax.set_ylabel('Score (Lower is Better)', fontsize=12, color=COLORS['text'])
    ax.set_title('Multi-head Distilled RoBERTa: MAE & RMSE on Test Set',
                fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, 
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'mae_rmse_test.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_f1_progression():
    """Create F1 score progressiongraph (test set)."""
    
    all_results = load_all_results()
    selected = select_key_epochs(all_results, n_epochs=12)
    
    epochs = [r['epoch'] for r in selected]
    f1_values = [r['f1_binary'] for r in selected]
    
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    x = np.arange(len(epochs))
    
    # Plot F1 scores
    ax.plot(x, f1_values, color=COLORS['f1'], linewidth=2, 
            marker='o', markersize=7, label='F1 Binary', alpha=0.9)
    
    # Add value labels
    for i, (ep, f1) in enumerate(zip(epochs, f1_values)):
        ax.annotate(f'{f1:.3f}', (i, f1), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=9, color=COLORS['f1'],
                   fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels([f'Epoch {e}' for e in epochs], fontsize=9, 
                       color=COLORS['text'], rotation=45, ha='right')
    ax.set_ylabel('F1 Score (Higher is Better)', fontsize=12, color=COLORS['text'])
    ax.set_ylim(0, 1)
    
    # Title
    ax.set_title('Multi-head Distilled RoBERTa: F1 Binary Score on Test Set',
                fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='lower right', fontsize=10,
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'f1_test.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_epoch_comparison():
    """Create comparison of first, best, middle, final epochs (test set)."""
    
    all_results = load_all_results()
    
    first = all_results[0]
    last = all_results[-1]
    best = min(all_results, key=lambda x: x['mae_total'])
    
    # Middle between best and final
    best_idx = all_results.index(best)
    last_idx = len(all_results) - 1
    middle_idx = (best_idx + last_idx) // 2
    middle = all_results[middle_idx]
    
    epochs_data = [first, best, middle, last]
    stage_labels = ["First", "Best", "Middle", "Final"]
    epoch_labels = [f"Epoch {e['epoch']}" for e in epochs_data]
    labels = [f"{stage}\n{epoch}" for stage, epoch in zip(stage_labels, epoch_labels)]
    
    mae_vals = [e['mae_total'] for e in epochs_data]
    rmse_vals = [e['rmse_total'] for e in epochs_data]
    f1_vals = [e['f1_binary'] for e in epochs_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['background'])
    
    # Plot 1: MAE & RMSE
    ax1.set_facecolor(COLORS['background'])
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mae_vals, width, color=COLORS['mae'], 
                    label='MAE', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rmse_vals, width, color=COLORS['rmse'],
                    label='RMSE', alpha=0.8)
    
    # Highlight best
    best_bar_idx = 1
    bars1[best_bar_idx].set_edgecolor(COLORS['success'])
    bars1[best_bar_idx].set_linewidth(2)
    bars2[best_bar_idx].set_edgecolor(COLORS['success'])
    bars2[best_bar_idx].set_linewidth(2)
    
    # Add value labels
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=9, color=COLORS['text'])
    for bar, val in zip(bars2, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=9, color=COLORS['text'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, color=COLORS['text'])
    ax1.set_ylabel('Score (Lower is Better)', fontsize=11, color=COLORS['text'])
    ax1.set_title('MAE & RMSE on Test Set', fontsize=12, color=COLORS['text'], 
                 fontweight='bold', pad=10)
    ax1.legend(fontsize=10, facecolor=COLORS['background'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'], framealpha=0.9)
    ax1.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    ax1.tick_params(colors=COLORS['text'])
    
    # Plot 2: F1 Score
    ax2.set_facecolor(COLORS['background'])
    bars = ax2.bar(x, f1_vals, color=COLORS['f1'], alpha=0.8)
    
    # Highlight best
    bars[best_bar_idx].set_edgecolor(COLORS['success'])
    bars[best_bar_idx].set_linewidth(2)
    
    for bar, val in zip(bars, f1_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, color=COLORS['text'])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, color=COLORS['text'])
    ax2.set_ylabel('F1 Score (Higher is Better)', fontsize=11, color=COLORS['text'])
    ax2.set_title('F1 Score on Test Set', fontsize=12, color=COLORS['text'], 
                 fontweight='bold', pad=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    ax2.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'epoch_comparison.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate all RoBERTa plots."""
    print("=" * 60)
    print("Generating Multi-head RoBERTa Results Graphs")
    print("Model: Multi-head Distilled RoBERTa (8 PHQ items)")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Load data first to show best model
    all_results = load_all_results()
    best = min(all_results, key=lambda x: x['mae_total'])
    
    print(f"\nFound {len(all_results)} epochs")
    print(f"\nBest Model:")
    print(f"  Epoch: {best['epoch']}")
    print(f"  MAE Total: {best['mae_total']:.4f}")
    print(f"  RMSE Total: {best['rmse_total']:.4f}")
    print(f"  MSE Total: {best['mse_total']:.4f}")
    print(f"  F1 Binary: {best['f1_binary']:.4f}")
    
    create_mae_progression()
    create_f1_progression()
    create_epoch_comparison()
    
    print("\n" + "=" * 60)
    print(f"All graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
