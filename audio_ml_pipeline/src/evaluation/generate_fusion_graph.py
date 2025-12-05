"""
Generate Fusion Comparison Graph

Shows single modality vs fusion performance with SOTA comparison.

Usage:
    python -m src.evaluation.generate_fusion_graph
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Color theme
COLORS = {
    'background': '#0A0A12',
    'card_bg': '#101020',
    'grid': '#2A2A4A',
    'text': '#FFFFFF',
    'text_secondary': '#8888AA',
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/fusion_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model performance data
MODELS = {
    'Audio Only': {'mae': 4.71, 'color': '#3B82F6'},
    'Text Only': {'mae': 4.73, 'color': '#8B5CF6'},
    'Min Fusion': {'mae': 4.26, 'color': '#10B981'},
    'SOTA Text': {'mae': 3.78, 'color': '#F59E0B'},
}


def create_fusion_comparison():
    """Create fusion comparison bar chart."""
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    model_names = list(MODELS.keys())
    mae_values = [MODELS[m]['mae'] for m in model_names]
    colors = [MODELS[m]['color'] for m in model_names]
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, mae_values, color=colors, alpha=0.9, width=0.6,
                  edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, mae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=14, color=COLORS['text'],
                fontweight='bold')
    
    # Highlight fusion bar
    bars[2].set_edgecolor('#10B981')
    bars[2].set_linewidth(3)
    
    # Add improvement annotation
    improvement = ((4.71 - 4.26) / 4.71) * 100
    ax.annotate(f'+{improvement:.1f}% improvement',
                xy=(2, 4.26), xytext=(2.5, 3.5),
                fontsize=11, color='#10B981', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#10B981', lw=1.5))
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12, color=COLORS['text'])
    ax.set_ylabel('MAE (Lower is Better)', fontsize=13, color=COLORS['text'],
                  fontweight='bold')
    ax.set_ylim(0, 5.5)
    ax.tick_params(colors=COLORS['text'], length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, alpha=0.15, color=COLORS['grid'], linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    ax.set_title('Multimodal Fusion vs Single Modality Performance',
                fontsize=16, color=COLORS['text'], fontweight='bold', pad=20)
    
    # Add legend/categories at bottom
    ax.text(0.5, -0.12, 'Single Modality        |        Fusion        |        SOTA',
            transform=ax.transAxes, ha='center', fontsize=10,
            color=COLORS['text_secondary'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'fusion_comparison.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate fusion comparison graph."""
    print("=" * 60)
    print("Generating Fusion Comparison Graph")
    print("=" * 60)
    
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    print("\nModel Performance:")
    for name, data in MODELS.items():
        print(f"  {name}: MAE {data['mae']}")
    
    improvement = ((4.71 - 4.26) / 4.71) * 100
    print(f"\nFusion improvement: {improvement:.1f}% over best single modality")
    
    create_fusion_comparison()
    
    print("\n" + "=" * 60)
    print(f"Graph saved to: {OUTPUT_DIR}/fusion_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
