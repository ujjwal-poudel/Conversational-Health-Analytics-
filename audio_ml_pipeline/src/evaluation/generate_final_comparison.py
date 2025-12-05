"""
Generate Audio vs Text Model Comparison Graph

Creates a professional comparison visualization of the best audio and text models
for depression detection.

Usage:
    python -m src.evaluation.generate_final_comparison
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
    
    'audio': '#0059FF',     # Blue for audio
    'text': '#00FFFF',      # Cyan for text
    'baseline': '#8888AA',  # Gray for baseline
    'success': '#00FF84',   # Green for best
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../reports/final_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model details - Complete comparison spectrum
PHASE1_BASELINE = {
    'name': 'Phase 1\nBaseline\n(RoBERTa v2)',
    'mae': 6.01,
    'rmse': np.sqrt(49.06),
    'details': 'Single-output regression, Epoch 3',
}

AUDIO_MODEL = {
    'name': 'Our Audio\nLasso (K=55)',
    'mae': 4.71,
    'rmse': 5.90,
    'details': '55 features, Wav2Vec2 + Prosody',
}

TEXT_MODEL = {
    'name': 'Our Text\nMulti-target\n(Epoch 14)',
    'mae': 4.73,
    'rmse': 6.06,
    'details': '8-head hierarchical regression',
}

FUSION_MODEL = {
    'name': 'Our Best Fusion\n(Min Strategy)',
    'mae': 4.26,
    'rmse': 5.73,
    'details': 'Multimodal Late Fusion',
}

SOTA_AUDIO = {
    'name': 'SOTA Audio\n(Literature)',
    'mae': 4.45,  # Midpoint of 4.3-4.6 range
    'rmse': None,  # Not reported
    'details': 'COVAREP + Deep / Wav2Vec2',
}

SOTA_TEXT = {
    'name': 'SOTA Text\n(Literature)',
    'mae': 3.78,
    'rmse': None,  # Not reported
    'details': 'Hierarchical multi-target regression',
}


def create_comparison_bar_chart():
    """Create minimalist comparison bar chart."""
    
    # All 6 models now
    models = [PHASE1_BASELINE, AUDIO_MODEL, TEXT_MODEL, FUSION_MODEL, SOTA_AUDIO, SOTA_TEXT]
    model_names = [m['name'] for m in models]
    mae_values = [m['mae'] for m in models]
    
    # Minimalist 3-color scheme with special color for fusion
    colors = [
        '#6B7280',      # Phase 1 baseline (gray)
        '#3B82F6',      # Our audio (blue)
        '#3B82F6',      # Our text (blue)
        '#00FF84',      # Our Fusion (Bright Green/Success)
        '#10B981',      # SOTA audio (Green-ish)
        '#10B981',      # SOTA text (Green-ish)
    ]
    
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, mae_values, color=colors, alpha=0.9, width=0.6, 
                   edgecolor='white', linewidth=1.5)
    
    # highlight fusion bar even more
    bars[3].set_linewidth(3)
    bars[3].set_edgecolor('#FFFFFF')
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, mae_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}', ha='center', fontsize=14, color=COLORS['text'], 
                fontweight='bold')
    
    # Simple category labels below x-axis
    ax.text(0, -0.8, 'Phase 1', ha='center', fontsize=11, 
            color='#6B7280', fontweight='bold')
    ax.text(2, -0.8, 'Single Modality', ha='center', fontsize=11, 
            color='#3B82F6', fontweight='bold')
    ax.text(3, -0.8, 'Multimodal', ha='center', fontsize=11, 
            color='#00FF84', fontweight='bold')
    ax.text(4.5, -0.8, 'SOTA Reference', ha='center', fontsize=11, 
            color='#10B981', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11, color=COLORS['text'])
    ax.set_ylabel('MAE (Lower is Better)', fontsize=13, color=COLORS['text'], 
                  fontweight='bold')
    
    # Adjust Y-axis to see differences better
    # Using 0-7 range but adding more grid lines
    ax.set_ylim(0, 7.0)
    ax.set_yticks(np.arange(0, 7.5, 0.5))  # Grid lines every 0.5
    
    ax.tick_params(colors=COLORS['text'], length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, alpha=0.15, color=COLORS['grid'], linestyle='-', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Clean title
    ax.set_title('Depression Detection: Final Model Comparison',
                fontsize=18, color=COLORS['text'], fontweight='bold', pad=25)
    
    # Simple gap annotation at the bottom
    gap_text = f"Fusion Gap from SOTA Text: {abs(FUSION_MODEL['mae'] - SOTA_TEXT['mae']):.2f} (MAE)"
    ax.text(0.5, -0.15, gap_text,
           transform=ax.transAxes, ha='center', fontsize=12, 
           color=COLORS['text_secondary'])
    
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'complete_comparison.png')
    plt.savefig(save_path, dpi=300, facecolor=COLORS['background'],
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")
    return save_path


def main():
    """Generate comparison visualizations."""
    print("=" * 60)
    print("Generating Complete Model Comparison (with SOTA)")
    print("=" * 60)
    
    # Set matplotlib style
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    print("\nPhase 1 Baseline:")
    print(f"  RoBERTa v2: MAE {PHASE1_BASELINE['mae']}")
    
    print("\nOur Models:")
    print(f"  Audio (Lasso K=55): MAE {AUDIO_MODEL['mae']}")
    print(f"  Text (Multi-target Epoch 14): MAE {TEXT_MODEL['mae']}")
    
    print("\nSOTA (Literature):")
    print(f"  Audio: MAE {SOTA_AUDIO['mae']} (range 4.3-4.6)")
    print(f"  Text: MAE {SOTA_TEXT['mae']}")
    
    gap_audio = ((AUDIO_MODEL['mae'] - SOTA_AUDIO['mae']) / SOTA_AUDIO['mae']) * 100
    gap_text = ((TEXT_MODEL['mae'] - SOTA_TEXT['mae']) / SOTA_TEXT['mae']) * 100
    
    print(f"\nGap from SOTA:")
    print(f"  Audio: {gap_audio:.1f}% behind")
    print(f"  Text: {gap_text:.1f}% behind")
    
    create_comparison_bar_chart()
    
    print("\n" + "=" * 60)
    print(f"Graph saved to: {OUTPUT_DIR}/complete_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
