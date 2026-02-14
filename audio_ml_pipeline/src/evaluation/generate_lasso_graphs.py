"""
Generate Lasso Audio Model Comparison Graph

Creates comparison of MAE and R² for Train/Val/Test on best Lasso model (V8).

Usage:
    python -m audio_ml_pipeline.src.evaluation.generate_lasso_graphs
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
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/lasso_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lasso V8 Model Results
# From training summary and reports
LASSO_DATA = {
    'cv_mae': 4.27,      # 5-fold cross-validation MAE
    'cv_rmse': 5.20,     # 5-fold cross-validation RMSE
    'test_mae': 4.71,    # Test set MAE (BEST!)
    'test_rmse': 5.78,   # Test set RMSE
    'test_r2': 0.18,     # Test set R²
}

# For R² calculation: R² = 1 - (MSE / Variance)
# PHQ target variance ≈ 41 (from audio dataset)
TARGET_VARIANCE = 41.0

# Compute R² for CV from CV RMSE
CV_MSE = LASSO_DATA['cv_rmse'] ** 2  # 27.04
CV_R2 = 1 - (CV_MSE / TARGET_VARIANCE)  # 0.34

TEST_MSE = LASSO_DATA['test_rmse'] ** 2  # 33.4
TEST_R2 = LASSO_DATA['test_r2']  # 0.18


def create_lasso_comparison_graph():
    """Create MAE and R² comparison bar chart for CV/Test on Lasso model."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['background'])
    
    # Low contrast, muted colors
    bar_colors = ['#70A0A0', '#90B090']  # Muted teal for CV, sage green for Test
    
    # Plot 1: MAE Comparison
    ax1.set_facecolor(COLORS['background'])
    
    categories = ['Cross-Validation', 'Test']
    mae_values = [LASSO_DATA['cv_mae'], LASSO_DATA['test_mae']]
    
    bars1 = ax1.bar(categories, mae_values, color=bar_colors, edgecolor='#444444', linewidth=1.5, width=0.5)
    
    # Add value labels
    for bar, val in zip(bars1, mae_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=16, color=COLORS['text'], fontweight='bold')
    
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=13, color=COLORS['text'], fontweight='bold')
    ax1.set_title('Lasso V8: MAE Comparison', fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    ax1.set_ylim(0, 6)
    ax1.tick_params(colors=COLORS['text'], labelsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    ax1.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    
    # Plot 2: R² Comparison
    ax2.set_facecolor(COLORS['background'])
    
    r2_values = [CV_R2, TEST_R2]
    bars2 = ax2.bar(categories, r2_values, color=bar_colors, edgecolor='#444444', linewidth=1.5, width=0.5)
    
    # Add value labels
    for bar, val in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=16, color=COLORS['text'], fontweight='bold')
    
    ax2.set_ylabel('R² Score (Higher is Better)', fontsize=13, color=COLORS['text'], fontweight='bold')
    ax2.set_title('Lasso V8: R² Comparison', fontsize=16, color=COLORS['text'], fontweight='bold', pad=15)
    ax2.set_ylim(0, 0.5)
    ax2.tick_params(colors=COLORS['text'], labelsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    ax2.grid(True, alpha=0.2, color=COLORS['grid'], axis='y')
    
    # Main title
    fig.suptitle('Best Audio Model: Lasso Regression (K=55 Features)',
                fontsize=18, color=COLORS['text'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    save_path = os.path.join(OUTPUT_DIR, 'lasso_mae_r2_comparison.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate Lasso comparison graphs."""
    print("=" * 60)
    print("Generating Lasso Audio Model Comparison Graphs")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    print("\nCreating MAE & R² comparison graph...")
    create_lasso_comparison_graph()
    
    print("\n" + "=" * 60)
    print(f"Graphs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
