"""
Generate Phase 1 Baseline Model Results Graph

Shows DistilBERT and RoBERTa single-output regression baseline models
that preceded the multi-target hierarchical approach.

These models had high MAE (6+) which motivated switching to multi-target.

Usage:
    python -m src.evaluation.generate_phase1_baselines
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Same color theme
COLORS = {
    'background': '#0A0A12',
    'card_bg': '#101020',
    'grid': '#2A2A4A',
    'text': '#FFFFFF',
    'text_secondary': '#8888AA',
    
    'distilbert': '#FF6B6B',    # Red for DistilBERT
    'roberta_v2': '#4ECDC4',    # Teal for RoBERTa v2
    'roberta_v3': '#95E1D3',    # Light teal for RoBERTa v3
    'multi_target': '#00FF84',  # Green for final multi-target
    'success': '#00FF84',       # Green accent
}

#Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/phase1_baselines")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Phase 1 Models (Single-output regression baselines)
MODELS = [
    {
        'name': 'DistilBERT v4',
        'full_name': 'distilbert-base-uncased',
        'type': 'Classification',
        'task': 'Binary classification (>9 threshold)',
        'epochs': 5,
        'best_epoch': 2,
        'val_acc': 0.6122,
        'val_f1': 0.3596,
        'mae': None,  # Was classification
        'mse': None,
        'note': 'Early stopping, classification task',
        'color': COLORS['distilbert']
    },
    {
        'name': 'RoBERTa v2',
        'full_name': 'roberta-base',
        'type': 'Regression',
        'task': 'Single-output regression (total PHQ)',
        'epochs': 6,
        'best_epoch': 3,
        'val_mae': 6.01,
        'val_mse': 49.06,
        'val_rmse': np.sqrt(49.06),
        'note': 'Early stopping at epoch 6',
        'color': COLORS['roberta_v2']
    },
    {
        'name': 'RoBERTa v3',
        'full_name': 'roberta-base',
        'type': 'Regression',
        'task': 'Single-output regression (total PHQ)',
        'epochs': 16,
        'best_epoch': 13,
        'val_mae': 6.05,
        'val_mse': 48.11,
        'val_rmse': np.sqrt(48.11),
        'note': 'Early stopping at epoch 16',
        'color': COLORS['roberta_v3']
    },
]

# Current best model for comparison
MULTI_TARGET = {
    'name': 'Multi-target\nModel',
    'full_name': 'distilroberta-base',
    'type': 'Multi-label Regression',
    'task': '8-head hierarchical regression',
    'epochs': 40,
    'best_epoch': 14,
    'test_mae': 4.73,
    'test_rmse': 6.06,
    'note': 'Final approach - Phase 2',
    'color': COLORS['multi_target']
}


def create_phase1_comparison():
    """Create comparison of Phase 1 baseline models."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=COLORS['background'])
    
    # Filter regression models only
    regression_models = [m for m in MODELS if m['type'] == 'Regression']
    
    # Plot 1: MAE Comparison
    ax1.set_facecolor(COLORS['background'])
    
    model_names = [m['name'] for m in regression_models] + [MULTI_TARGET['name']]
    mae_values = [m['val_mae'] for m in regression_models] + [MULTI_TARGET['test_mae']]
    colors = [m['color'] for m in regression_models] + [MULTI_TARGET['color']]
    
    x = np.arange(len(model_names))
    bars = ax1.bar(x, mae_values, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    
    # Highlight best (multi-target)
    bars[-1].set_edgecolor(COLORS['multi_target'])
    bars[-1].set_linewidth(3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mae_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', fontsize=11, color=COLORS['text'], fontweight='bold')
    
    # Add epoch labels below bars
    for i, model in enumerate(regression_models + [MULTI_TARGET]):
        ax1.text(i, -0.4, f"Epoch {model.get('best_epoch', 'N/A')}",
                ha='center', fontsize=9, color=COLORS['text_secondary'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11, color=COLORS['text'])
    ax1.set_ylabel('MAE (Lower is Better)', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax1.set_title('Mean Absolute Error Comparison', fontsize=14, color=COLORS['text'], 
                 fontweight='bold', pad=15)
    ax1.set_ylim(0, 7)
    ax1.tick_params(colors=COLORS['text'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    ax1.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    
    # Plot 2: RMSE Comparison
    ax2.set_facecolor(COLORS['background'])
    
    rmse_values = [m['val_rmse'] for m in regression_models] + [MULTI_TARGET['test_rmse']]
    bars2 = ax2.bar(x, rmse_values, color=colors, alpha=0.9, edgecolor='white', linewidth=2)
    
    bars2[-1].set_edgecolor(COLORS['multi_target'])
    bars2[-1].set_linewidth(3)
    
    for i, (bar, val) in enumerate(zip(bars2, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', fontsize=11, color=COLORS['text'], fontweight='bold')
    
    # Add total epochs below bars
    for i, model in enumerate(regression_models + [MULTI_TARGET]):
        ax2.text(i, -0.4, f"{model.get('epochs', 'N/A')} epochs",
                ha='center', fontsize=9, color=COLORS['text_secondary'])
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=11, color=COLORS['text'])
    ax2.set_ylabel('RMSE (Lower is Better)', fontsize=12, color=COLORS['text'], fontweight='bold')
    ax2.set_title('Root Mean Squared Error Comparison', fontsize=14, color=COLORS['text'], 
                 fontweight='bold', pad=15)
    ax2.set_ylim(0, 8)
    ax2.tick_params(colors=COLORS['text'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    ax2.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    
    # Main title
    fig.suptitle('Phase 1 Baselines vs Multi-target Model (Phase 2)',
                fontsize=18, color=COLORS['text'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, 'phase1_vs_phase2.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate Phase 1 baseline comparison graphs."""
    print("=" * 60)
    print("Generating Phase 1 Baseline Model Comparison")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    print("\nPhase 1 Baseline Models (Single-output regression):")
    for model in MODELS:
        if model['type'] == 'Regression':
            print(f"  {model['name']}: MAE {model['val_mae']:.2f} (Epoch {model['best_epoch']})")
    
    print(f"\nPhase 2 Multi-target Model:")
    print(f"  {MULTI_TARGET['name'].replace(chr(10), ' ')}: Test MAE {MULTI_TARGET['test_mae']:.2f} (Epoch {MULTI_TARGET['best_epoch']})")
    
    improvement = ((6.01 - MULTI_TARGET['test_mae']) / 6.01) * 100
    print(f"\nImprovement: {improvement:.1f}% better MAE with multi-target approach\n")
    
    create_phase1_comparison()
    
    print("\n" + "=" * 60)
    print(f"Graph saved to: {OUTPUT_DIR}/phase1_vs_phase2.png")
    print("=" * 60)
    print("\nNote: Phase 1's high MAE (6+) motivated the switch to multi-target regression")


if __name__ == "__main__":
    main()
