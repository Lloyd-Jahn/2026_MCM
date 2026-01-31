"""
Violin Plot + Box Plot: Scenario Comparison
============================================
Compares cost and time distributions across three transportation scenarios

USER ADJUSTMENT GUIDE:
----------------------
Line 24-28:  Color palette (Hex codes)
Line 31-32:  Figure size
Line 35-37:  Font sizes
Line 40-41:  Violin plot transparency and box width
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Professional Color Palette
COLOR_SCENARIO_A = '#3498db'  # Blue - Elevator
COLOR_SCENARIO_B = '#e74c3c'  # Red - Rocket
COLOR_SCENARIO_C = '#f39c12'  # Orange - Hybrid
PALETTE = [COLOR_SCENARIO_A, COLOR_SCENARIO_B, COLOR_SCENARIO_C]

# Plot Configuration
FIG_WIDTH = 18
FIG_HEIGHT = 8
DPI = 300

# Font Configuration
LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 18

# Violin plot configuration
VIOLIN_ALPHA = 0.7
BOX_WIDTH = 0.3

# ============================================================================
# VIOLIN + BOX PLOT CREATION
# ============================================================================

def create_violin_plot(scenario_data, output_path):
    """
    Create violin plot with embedded box plot for scenario comparison

    Parameters:
    -----------
    scenario_data : dict
        Dictionary containing 'cost' and 'time' data for each scenario
    output_path : Path
        Absolute path to save the PDF
    """

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    # Prepare data for cost comparison
    cost_df = pd.DataFrame({
        'Scenario': np.concatenate([
            ['Scenario A\n(Elevator)'] * len(scenario_data['cost']['Scenario A\n(Elevator)']),
            ['Scenario B\n(Rocket)'] * len(scenario_data['cost']['Scenario B\n(Rocket)']),
            ['Scenario C\n(Hybrid)'] * len(scenario_data['cost']['Scenario C\n(Hybrid)'])
        ]),
        'Cost (Billion USD)': np.concatenate([
            scenario_data['cost']['Scenario A\n(Elevator)'],
            scenario_data['cost']['Scenario B\n(Rocket)'],
            scenario_data['cost']['Scenario C\n(Hybrid)']
        ])
    })

    # Prepare data for time comparison
    time_df = pd.DataFrame({
        'Scenario': np.concatenate([
            ['Scenario A\n(Elevator)'] * len(scenario_data['time']['Scenario A\n(Elevator)']),
            ['Scenario B\n(Rocket)'] * len(scenario_data['time']['Scenario B\n(Rocket)']),
            ['Scenario C\n(Hybrid)'] * len(scenario_data['time']['Scenario C\n(Hybrid)'])
        ]),
        'Time (Years)': np.concatenate([
            scenario_data['time']['Scenario A\n(Elevator)'],
            scenario_data['time']['Scenario B\n(Rocket)'],
            scenario_data['time']['Scenario C\n(Hybrid)']
        ])
    })

    # === LEFT PANEL: Cost Distribution ===
    ax1 = axes[0]

    # Create violin plot
    parts = ax1.violinplot(
        [scenario_data['cost']['Scenario A\n(Elevator)'],
         scenario_data['cost']['Scenario B\n(Rocket)'],
         scenario_data['cost']['Scenario C\n(Hybrid)']],
        positions=[1, 2, 3],
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Color violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i])
        pc.set_alpha(VIOLIN_ALPHA)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Overlay box plot
    bp = ax1.boxplot(
        [scenario_data['cost']['Scenario A\n(Elevator)'],
         scenario_data['cost']['Scenario B\n(Rocket)'],
         scenario_data['cost']['Scenario C\n(Hybrid)']],
        positions=[1, 2, 3],
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
        medianprops=dict(color='red', linewidth=3),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2)
    )

    # Add mean markers (diamond)
    means = [np.mean(scenario_data['cost']['Scenario A\n(Elevator)']),
             np.mean(scenario_data['cost']['Scenario B\n(Rocket)']),
             np.mean(scenario_data['cost']['Scenario C\n(Hybrid)'])]
    ax1.scatter([1, 2, 3], means, marker='D', s=150, color='blue', edgecolor='black',
                linewidth=2, zorder=10, label='Mean')

    # Styling
    ax1.set_ylabel('Total Cost (Billion USD)', fontsize=LABEL_SIZE, fontweight='bold',
                   fontname='Times New Roman')
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Scenario A\n(Elevator)', 'Scenario B\n(Rocket)', 'Scenario C\n(Hybrid)'],
                        fontsize=TICK_SIZE, fontweight='bold', fontname='Times New Roman')
    ax1.tick_params(axis='y', labelsize=TICK_SIZE, width=2, length=6)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # Dynamic Y-axis limits (maximize variance visibility)
    cost_min = min([scenario_data['cost'][k].min() for k in scenario_data['cost']])
    cost_max = max([scenario_data['cost'][k].max() for k in scenario_data['cost']])
    margin = (cost_max - cost_min) * 0.1
    ax1.set_ylim(cost_min - margin, cost_max + margin)

    # Bold black borders
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Add grid
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax1.set_axisbelow(True)

    # === RIGHT PANEL: Time Distribution ===
    ax2 = axes[1]

    # Create violin plot
    parts = ax2.violinplot(
        [scenario_data['time']['Scenario A\n(Elevator)'],
         scenario_data['time']['Scenario B\n(Rocket)'],
         scenario_data['time']['Scenario C\n(Hybrid)']],
        positions=[1, 2, 3],
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Color violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE[i])
        pc.set_alpha(VIOLIN_ALPHA)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)

    # Overlay box plot
    bp = ax2.boxplot(
        [scenario_data['time']['Scenario A\n(Elevator)'],
         scenario_data['time']['Scenario B\n(Rocket)'],
         scenario_data['time']['Scenario C\n(Hybrid)']],
        positions=[1, 2, 3],
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='white', edgecolor='black', linewidth=2),
        medianprops=dict(color='red', linewidth=3),
        whiskerprops=dict(color='black', linewidth=2),
        capprops=dict(color='black', linewidth=2)
    )

    # Add mean markers
    means = [np.mean(scenario_data['time']['Scenario A\n(Elevator)']),
             np.mean(scenario_data['time']['Scenario B\n(Rocket)']),
             np.mean(scenario_data['time']['Scenario C\n(Hybrid)'])]
    ax2.scatter([1, 2, 3], means, marker='D', s=150, color='blue', edgecolor='black',
                linewidth=2, zorder=10, label='Mean')

    # Styling
    ax2.set_ylabel('Mission Duration (Years)', fontsize=LABEL_SIZE, fontweight='bold',
                   fontname='Times New Roman')
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['Scenario A\n(Elevator)', 'Scenario B\n(Rocket)', 'Scenario C\n(Hybrid)'],
                        fontsize=TICK_SIZE, fontweight='bold', fontname='Times New Roman')
    ax2.tick_params(axis='y', labelsize=TICK_SIZE, width=2, length=6)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # Dynamic Y-axis limits
    time_min = min([scenario_data['time'][k].min() for k in scenario_data['time']])
    time_max = max([scenario_data['time'][k].max() for k in scenario_data['time']])
    margin = (time_max - time_min) * 0.1
    ax2.set_ylim(time_min - margin, time_max + margin)

    # Bold black borders
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Add grid
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax2.set_axisbelow(True)

    # Add legend (only on right panel)
    legend = ax2.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True,
                       prop={'family': 'Times New Roman', 'weight': 'bold'})
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2)

    # Tight layout
    plt.tight_layout()

    # Save as high-resolution PDF
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Violin + box plot saved: {output_path}")


if __name__ == '__main__':
    # Test with sample data
    from pathlib import Path
    np.random.seed(42)

    scenario_data = {
        'cost': {
            'Scenario A\n(Elevator)': np.random.normal(28.3, 2.5, 10000),
            'Scenario B\n(Rocket)': np.random.normal(45.8, 5.2, 10000),
            'Scenario C\n(Hybrid)': np.random.normal(32.5, 2.0, 10000)
        },
        'time': {
            'Scenario A\n(Elevator)': np.random.normal(0.58, 0.05, 10000),
            'Scenario B\n(Rocket)': np.random.normal(0.25, 0.03, 10000),
            'Scenario C\n(Hybrid)': np.random.normal(0.35, 0.025, 10000)
        }
    }

    output_path = Path(__file__).parent / 'outputs' / 'test_violin.pdf'
    output_path.parent.mkdir(exist_ok=True)
    create_violin_plot(scenario_data, output_path)
