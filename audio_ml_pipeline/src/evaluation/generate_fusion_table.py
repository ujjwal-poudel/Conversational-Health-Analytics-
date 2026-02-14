"""
Generate Fusion Comparison Table as Image

Creates a table image from fusion results data.

Usage:
    python -m audio_ml_pipeline.src.evaluation.generate_fusion_table
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Color theme
COLORS = {
    'background': '#040B10',
    'text': '#FFFFFF',
    'text_secondary': '#AABBCC',
    'row_alt': '#0a1520',
    'border': '#203040',
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/fusion_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fusion table data
FUSION_DATA = [
    ['Simple Average (50/50)', '4.34', '5.36', '+7.86%'],
    ['Weighted (50/50)', '4.34', '5.36', '+7.86%'],
    ['Weighted (40/60)', '4.38', '5.41', '+7.04%'],
    ['Weighted (30/70)', '4.44', '5.51', '+5.88%'],
    ['Weighted (19/80)', '4.52', '5.66', '+4.13%'],
    ['Min Fusion', '4.26', '5.73', '+9.55%'],
    ['Max Fusion', '5.18', '6.11', '-9.85%'],
]

COLUMNS = ['Strategy', 'MAE', 'RMSE', 'Improvement']


def create_fusion_table():
    """Create fusion comparison table as image."""
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=FUSION_DATA,
        colLabels=COLUMNS,
        loc='center',
        cellLoc='center',
    )
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)
    
    # Apply colors
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        cell.set_edgecolor(COLORS['border'])
        cell.set_text_props(color=COLORS['text'], fontweight='normal')
        
        if key[0] == 0:  # Header row
            cell.set_facecolor('#152535')
            cell.set_text_props(color=COLORS['text'], fontweight='bold')
        elif key[0] % 2 == 0:  # Even rows
            cell.set_facecolor(COLORS['row_alt'])
        else:  # Odd rows
            cell.set_facecolor(COLORS['background'])
        
        # Highlight Min Fusion row (best)
        if key[0] == 6:  # Min Fusion row
            cell.set_facecolor('#0a2020')
            cell.set_text_props(color='#90B090', fontweight='bold')
        
        # Highlight Max Fusion row (worst)
        if key[0] == 7:  # Max Fusion row
            cell.set_text_props(color='#C08070')
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'fusion_comparison_table.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate fusion table image."""
    print("=" * 60)
    print("Generating Fusion Comparison Table")
    print("=" * 60)
    
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    create_fusion_table()
    
    print("\n" + "=" * 60)
    print(f"Table saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
