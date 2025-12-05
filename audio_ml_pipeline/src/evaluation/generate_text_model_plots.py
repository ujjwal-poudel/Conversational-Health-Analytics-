"""
Generate Text Model Training Results Graphs

Creates minimal visualizations for Multi-target hierarchical regression model:
1. Training progression (MAE/RMSE by epoch - showing key epochs only)
2. F1 scores progression

Uses the same color theme as audio model graphs.

Usage:
    python -m src.evaluation.generate_text_model_plots
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Same color theme as audio graphs
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
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "../../datasets")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/text_model_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_text_results():
    """Load text model evaluation results."""
    json_path = os.path.join(DATASETS_DIR, "Text_model_evaluation_results.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_epoch_from_model_name(model_name):
    """Extract epoch number from model name like 'model_2_15.pt'."""
    parts = model_name.replace('.pt', '').split('_')
    return int(parts[-1])


def select_key_epochs(results, n_epochs=10):
    """
    Select key epochs to display (not all to avoid clutter).
    Includes: best, worst, first, last, and evenly spaced middle epochs.
    """
    sorted_by_epoch = sorted(results, key=lambda x: extract_epoch_from_model_name(x['model']))
    n_total = len(sorted_by_epoch)
    
    if n_total <= n_epochs:
        return sorted_by_epoch
    
    # Always include: first, last, best
    indices = {0, n_total - 1}  # first and last
    
    # Add best MAE
    best_idx = min(range(n_total), key=lambda i: sorted_by_epoch[i]['mae'])
    indices.add(best_idx)
    
    # Add evenly spaced epochs
    step = n_total // (n_epochs - 3)  # -3 for first, last, best
    for i in range(1, n_total - 1, step):
        indices.add(i)
        if len(indices) >= n_epochs:
            break
    
    # Return selected epochs
    selected = [sorted_by_epoch[i] for i in sorted(indices)]
    return selected


def create_mae_rmse_progression():
    """Create MAE and RMSE progression graph."""
    
    data = load_text_results()
    all_results = data['all_results']
    best_model = data['best_model_by_mae']
    
    # Select key epochs to avoid clutter
    selected = select_key_epochs(all_results, n_epochs=12)
    
    # Extract data
    epochs = [extract_epoch_from_model_name(r['model']) for r in selected]
    mae_values = [r['mae'] for r in selected]
    rmse_values = [r['rmse'] for r in selected]
    
    # Best model info
    best_epoch = extract_epoch_from_model_name(best_model['model'])
    best_mae = best_model['mae']
    best_rmse = best_model['rmse']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    x = np.arange(len(epochs))
    
    # Plot lines
    ax.plot(x, mae_values, color=COLORS['mae'], linewidth=2, 
            marker='o', markersize=7, label='MAE', alpha=0.9)
    ax.plot(x, rmse_values, color=COLORS['rmse'], linewidth=2,
            marker='o', markersize=7, label='RMSE', alpha=0.9)
    
    # Highlight best epoch
    best_idx = epochs.index(best_epoch)
    ax.scatter([best_idx], [best_mae], color=COLORS['success'], 
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
    ax.set_title('Multi-target Hierarchical Regression Model: Test Set Performance',
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
    
    save_path = os.path.join(OUTPUT_DIR, 'mae_rmse_progression.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_f1_progression():
    """Create F1 score progression graph."""
    
    data = load_text_results()
    all_results = data['all_results']
    
    # Select key epochs
    selected = select_key_epochs(all_results, n_epochs=12)
    
    # Extract data
    epochs = [extract_epoch_from_model_name(r['model']) for r in selected]
    f1_micro = [r['f1_micro_binary'] for r in selected]
    f1_macro = [r['f1_macro_binary'] for r in selected]
    mae_values = [r['mae'] for r in selected]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    x = np.arange(len(epochs))
    
    # Plot F1 scores
    ax.plot(x, f1_micro, color=COLORS['f1'], linewidth=2, 
            marker='o', markersize=7, label='F1 Binary (Micro=Macro)', alpha=0.9)
    
    # Create secondary axis for MAE
    ax2 = ax.twinx()
    ax2.plot(x, mae_values, color=COLORS['mae'], linewidth=2, 
             marker='s', markersize=6, label='MAE', alpha=0.7, linestyle='--')
    
    # Styling - primary axis
    ax.set_xticks(x)
    ax.set_xticklabels([f'Epoch {e}' for e in epochs], fontsize=9, 
                       color=COLORS['text'], rotation=45, ha='right')
    ax.set_ylabel('F1 Score (Higher is Better)', fontsize=12, color=COLORS['f1'])
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelcolor=COLORS['f1'])
    
    # Styling - secondary axis
    ax2.set_ylabel('MAE (Lower is Better)', fontsize=12, color=COLORS['mae'])
    ax2.tick_params(axis='y', labelcolor=COLORS['mae'])
    ax2.spines['right'].set_color(COLORS['grid'])
    
    # Title
    ax.set_title('Multi-target Hierarchical Regression Model: F1 Score & MAE Progression',
                fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10,
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(axis='x', colors=COLORS['text'])
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'f1_mae_progression.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_epoch_comparison():
    """Create comparison of key epochs: first, best, middle, and final."""
    
    data = load_text_results()
    all_results = data['all_results']
    
    # Sort by epoch
    sorted_results = sorted(all_results, key=lambda x: extract_epoch_from_model_name(x['model']))
    
    # Get key epochs
    first = sorted_results[0]
    last = sorted_results[-1]
    best = min(all_results, key=lambda x: x['mae'])
    
    # Get middle epoch between best and final
    best_idx = sorted_results.index(best)
    last_idx = len(sorted_results) - 1
    middle_idx = (best_idx + last_idx) // 2
    middle = sorted_results[middle_idx]
    
    # Create data
    epochs_data = [first, best, middle, last]
    epoch_labels = [f"Epoch {extract_epoch_from_model_name(e['model'])}" for e in epochs_data]
    stage_labels = ["First", "Best", "Middle", "Final"]
    labels = [f"{stage}\n{epoch}" for stage, epoch in zip(stage_labels, epoch_labels)]
    
    mae_vals = [e['mae'] for e in epochs_data]
    rmse_vals = [e['rmse'] for e in epochs_data]
    f1_vals = [e['f1_micro_binary'] for e in epochs_data]
    
    # Create plot with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['background'])
    
    # Plot 1: MAE and RMSE
    ax1.set_facecolor(COLORS['background'])
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mae_vals, width, color=COLORS['mae'], 
                    label='MAE', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rmse_vals, width, color=COLORS['rmse'],
                    label='RMSE', alpha=0.8)
    
    # Highlight best epoch
    best_bar_idx = 1  # Best is second in our list
    bars1[best_bar_idx].set_edgecolor(COLORS['success'])
    bars1[best_bar_idx].set_linewidth(2)
    bars2[best_bar_idx].set_edgecolor(COLORS['success'])
    bars2[best_bar_idx].set_linewidth(2)
    
    # Add value labels
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=10, color=COLORS['text'])
    for bar, val in zip(bars2, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=10, color=COLORS['text'])
    
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
    
    # Highlight best epoch
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
    """Generate all text model plots."""
    print("=" * 60)
    print("Generating Text Model Results Graphs")
    print("Model: Multi-target Hierarchical Regression Model")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    create_mae_rmse_progression()
    create_f1_progression()
    create_epoch_comparison()
    
    # Print best model info
    data = load_text_results()
    best = data['best_model_by_mae']
    print(f"\nBest Model: {best['model']}")
    print(f"  Epoch: {extract_epoch_from_model_name(best['model'])}")
    print(f"  MAE: {best['mae']:.4f}")
    print(f"  RMSE: {best['rmse']:.4f}")
    print(f"  F1 Binary: {best['f1_micro_binary']:.4f}")
    
    print("\n" + "=" * 60)
    print(f"All graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
