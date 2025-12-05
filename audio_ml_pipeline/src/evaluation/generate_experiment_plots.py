"""
Generate Professional Experiment Results Graphs

Creates publication-quality visualizations for:
1. Test Set Results (RMSE, MAE, R²) across all experiments
2. Cross-Validation Results across all experiments

Color theme can be customized at the top of the file.

Usage:
    python -m src.evaluation.generate_experiment_plots
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================================================
# COLOR THEME (From Figma Design)
# =============================================================================

# Vibrant green/cyan/blue theme from Figma
COLORS = {
    'background': '#0A0A12',       # Dark background
    'card_bg': '#101020',          # Card background
    'grid': '#2A2A4A',             # Grid lines
    'text': '#FFFFFF',             # Main text
    'text_secondary': '#8888AA',   # Secondary text
    
    # Metric colors (from Figma gradient colors)
    'rmse': '#0059FF',             # Blue for RMSE
    'mae': '#00FFFF',              # Cyan for MAE
    'r2': '#57FD53',               # Bright green for R²
    
    # Accent
    'accent': '#00FF84',           # Green accent
    'highlight': '#57FD53',        # Bright green highlight
    'success': '#00FF84',          # Green for best result
    'warning': '#425311',          # Olive/yellow-green
}

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../reports/experiment_graphs"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# EXPERIMENT DATA (from all our experiments)
# =============================================================================

# Test set results for each experiment
EXPERIMENTS = [
    {
        'name': 'V1: 8-Stat Pooling',
        'algo': 'LightGBM',
        'features': 6248,
        'rmse': 6.97,
        'mae': 5.70,
        'r2': -0.18,
        'cv_rmse': 5.38
    },
    {
        'name': 'V2: 4-Stat Pooling',
        'algo': 'XGBoost',
        'features': 3124,
        'rmse': 6.26,
        'mae': 5.19,
        'r2': 0.04,
        'cv_rmse': 5.48
    },
    {
        'name': 'V3: Prosody Only',
        'algo': 'XGBoost',
        'features': 52,
        'rmse': 6.93,
        'mae': 5.66,
        'r2': -0.17,
        'cv_rmse': 5.47
    },
    {
        'name': 'V4: Segment Pooling',
        'algo': 'XGBoost',
        'features': 9372,
        'rmse': 5.94,
        'mae': 4.93,
        'r2': 0.14,
        'cv_rmse': 5.38
    },
    {
        'name': 'V5: PCA (50 dims)',
        'algo': 'XGBoost',
        'features': 252,
        'rmse': 6.25,
        'mae': 5.19,
        'r2': 0.05,
        'cv_rmse': 5.39
    },
    {
        'name': 'V6: PCA + Segment',
        'algo': 'XGBoost',
        'features': 2556,
        'rmse': 6.20,
        'mae': 5.10,
        'r2': 0.06,
        'cv_rmse': 5.48
    },
    {
        'name': 'V7: Lasso (K=100)',
        'algo': 'Lasso',
        'features': 100,
        'rmse': 5.91,
        'mae': 4.77,
        'r2': 0.15,
        'cv_rmse': 5.23
    },
    {
        'name': 'V8: Lasso (K=55)',
        'algo': 'Lasso',
        'features': 55,
        'rmse': 5.90,
        'mae': 4.71,
        'r2': 0.18,
        'cv_rmse': 5.18
    },
]


def create_test_results_plot():
    """Create minimal test results visualization."""
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Data
    x = np.arange(len(EXPERIMENTS))
    labels = [f"{e['name']}\n({e['algo']}, {e['features']} feat)" for e in EXPERIMENTS]
    rmse = [e['rmse'] for e in EXPERIMENTS]
    mae = [e['mae'] for e in EXPERIMENTS]
    r2 = [e['r2'] for e in EXPERIMENTS]
    
    # Scale R² to be visible (multiply by 10 for visualization)
    r2_scaled = [r * 10 for r in r2]
    
    # Plot lines with markers - minimal style
    line_width = 2
    marker_size = 8
    
    ax.plot(x, rmse, color=COLORS['rmse'], linewidth=line_width, 
            marker='o', markersize=marker_size, label='RMSE', alpha=0.9)
    ax.plot(x, mae, color=COLORS['mae'], linewidth=line_width,
            marker='o', markersize=marker_size, label='MAE', alpha=0.9)
    ax.plot(x, r2_scaled, color=COLORS['r2'], linewidth=line_width,
            marker='o', markersize=marker_size, label='R² (×10)', alpha=0.9)
    
    # Add value labels - only for MAE (most important)
    for i, ma in enumerate(mae):
        ax.annotate(f'{ma:.2f}', (i, ma), textcoords="offset points", 
                   xytext=(0, -18), ha='center', fontsize=10, color=COLORS['mae'],
                   fontweight='bold')
    
    # Highlight best result (V8)
    best_idx = len(EXPERIMENTS) - 1
    ax.scatter([best_idx], [mae[best_idx]], color=COLORS['success'], 
               s=150, zorder=5, marker='o', edgecolors='white', linewidth=2)
    
    # Baseline reference line
    ax.axhline(y=5.43, color=COLORS['text_secondary'], linestyle='--', 
               alpha=0.4, linewidth=1, label='Baseline MAE (5.43)')
    
    # Styling - minimal
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=COLORS['text'], rotation=0, ha='center')
    ax.set_ylabel('Score', fontsize=12, color=COLORS['text'])
    
    # Title
    ax.set_title('Test Set Performance Across Experiments',
                fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend - minimal
    legend = ax.legend(loc='upper right', fontsize=10, 
                      facecolor=COLORS['background'], edgecolor=COLORS['grid'],
                      labelcolor=COLORS['text'], framealpha=0.9)
    
    # Minimal grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, 'test_results.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'], 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_cv_results_plot():
    """Create minimal cross-validation results visualization."""
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Data
    x = np.arange(len(EXPERIMENTS))
    labels = [f"{e['name']}\n({e['algo']}, {e['features']} feat)" for e in EXPERIMENTS]
    cv_rmse = [e['cv_rmse'] for e in EXPERIMENTS]
    test_mae = [e['mae'] for e in EXPERIMENTS]
    
    # Bar width
    width = 0.35
    
    # Create bars - use accent colors from theme
    bars1 = ax.bar(x - width/2, cv_rmse, width, color=COLORS['rmse'], 
                   label='CV RMSE', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_mae, width, color=COLORS['mae'],
                   label='Test MAE', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars1, cv_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9,
               color=COLORS['text'])
    
    for bar, val in zip(bars2, test_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.2f}', ha='center', va='bottom', fontsize=9,
               color=COLORS['text'])
    
    # Highlight best (V8)
    best_idx = len(EXPERIMENTS) - 1
    bars2[best_idx].set_color(COLORS['success'])
    
    # Styling - minimal
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=COLORS['text'])
    ax.set_ylabel('Score (Lower is Better)', fontsize=12, color=COLORS['text'])
    
    # Title
    ax.set_title('Cross-Validation vs Test Performance',
                fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    
    # Legend - minimal
    legend = ax.legend(loc='upper right', fontsize=10,
                      facecolor=COLORS['background'], edgecolor=COLORS['grid'],
                      labelcolor=COLORS['text'], framealpha=0.9)
    
    # Minimal grid
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='-', 
            linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    ax.set_ylim(0, max(cv_rmse) + 0.6)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, 'cv_results.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def create_improvement_plot():
    """Create progress/improvement visualization."""
    
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['card_bg'])
    
    # Data - Focus on MAE progression
    experiments = ['Baseline', 'V1', 'V4\n(XGB)', 'V7\n(Lasso)', 'V8\n(Final)']
    mae_values = [5.43, 5.70, 4.93, 4.77, 4.71]
    
    x = np.arange(len(experiments))
    
    # Create gradient bar colors
    colors = [COLORS['text_secondary'], COLORS['rmse'], COLORS['accent'], 
              COLORS['mae'], COLORS['success']]
    
    bars = ax.bar(x, mae_values, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.2f}', ha='center', va='bottom', fontsize=14,
               color=COLORS['text'], fontweight='bold')
    
    # Add improvement arrows
    for i in range(1, len(mae_values)):
        if mae_values[i] < mae_values[i-1]:
            improvement = mae_values[i-1] - mae_values[i]
            pct = (improvement / mae_values[i-1]) * 100
            ax.annotate(f'↓{improvement:.2f}\n({pct:.1f}%)',
                       xy=(i, mae_values[i]),
                       xytext=(i, mae_values[i] - 0.4),
                       ha='center', fontsize=10, color=COLORS['success'],
                       fontweight='bold')
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=13, color=COLORS['text'])
    ax.set_ylabel('MAE (Lower is Better)', fontsize=14, color=COLORS['text'], fontweight='bold')
    
    ax.set_title('MAE Improvement Journey: From Baseline to Best Model\n',
                fontsize=18, color=COLORS['text'], fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', 
            linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Spine colors
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])
    
    ax.set_ylim(0, 6.5)
    
    # Add total improvement annotation
    total_improvement = mae_values[0] - mae_values[-1]
    total_pct = (total_improvement / mae_values[0]) * 100
    ax.text(len(experiments) - 1, 6.0, 
           f'Total Improvement: {total_improvement:.2f} ({total_pct:.1f}%)',
           ha='center', fontsize=14, color=COLORS['success'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=COLORS['card_bg'], 
                    edgecolor=COLORS['success'], linewidth=2))
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, 'improvement_journey.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate all plots."""
    print("=" * 60)
    print("Generating Experiment Results Graphs")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    create_test_results_plot()
    create_cv_results_plot()
    create_improvement_plot()
    
    print("\n" + "=" * 60)
    print(f"All graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
