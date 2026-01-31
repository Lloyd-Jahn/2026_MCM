"""
Waterfall Chart: Cost Breakdown Comparison
===========================================
Compares cost components across three transportation scenarios using waterfall charts

USER ADJUSTMENT GUIDE:
----------------------
Line 24-28:  Color palette (Hex codes)
Line 31-32:  Figure size
Line 35-37:  Font sizes
Line 40:     Bar width
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Professional Color Palette
COLOR_POSITIVE = '#3498db'     # Blue - Positive increments
COLOR_NEGATIVE = '#e74c3c'     # Red - Negative increments (if any)
COLOR_TOTAL = '#2ecc71'        # Green - Total bar
COLOR_START = '#95a5a6'        # Gray - Start bar
EDGE_COLOR = '#000000'         # Black - Edge color

# Plot Configuration
FIG_WIDTH = 20
FIG_HEIGHT = 8
DPI = 300

# Font Configuration
LABEL_SIZE = 18
TICK_SIZE = 16
ANNOTATION_SIZE = 14

# Bar Configuration
BAR_WIDTH = 0.6

# ============================================================================
# WATERFALL CHART CREATION
# ============================================================================

def create_waterfall_chart(cost_breakdown, output_path):
    """
    Create waterfall chart showing cost breakdown for three scenarios

    Parameters:
    -----------
    cost_breakdown : dict
        Dictionary containing cost breakdown for Scenario A, B, C
    output_path : Path
        Absolute path to save the PDF
    """

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    scenario_names = ['Scenario A', 'Scenario B', 'Scenario C']

    for idx, (scenario_name, ax) in enumerate(zip(scenario_names, axes)):
        data = cost_breakdown[scenario_name]
        categories = data['categories']
        values = data['values']
        increments = data['increments']

        n_categories = len(categories)

        # Calculate bar positions and heights
        bar_starts = []
        bar_heights = []
        bar_colors = []

        cumulative = 0
        for i in range(n_categories):
            if i == 0:
                # Start bar
                bar_starts.append(0)
                bar_heights.append(0)
                bar_colors.append(COLOR_START)
            elif i == n_categories - 1:
                # Total bar
                bar_starts.append(0)
                bar_heights.append(values[i])
                bar_colors.append(COLOR_TOTAL)
            else:
                # Increment bars
                bar_starts.append(cumulative)
                bar_heights.append(increments[i])
                bar_colors.append(COLOR_POSITIVE if increments[i] >= 0 else COLOR_NEGATIVE)
                cumulative += increments[i]

        # Create bars
        bars = ax.bar(range(n_categories), bar_heights, bottom=bar_starts,
                      width=BAR_WIDTH, color=bar_colors, edgecolor=EDGE_COLOR,
                      linewidth=2, alpha=0.85)

        # Add connecting lines (dashed)
        for i in range(n_categories - 1):
            if i == n_categories - 2:
                # Don't connect second-to-last to total
                continue
            x1 = i + BAR_WIDTH / 2
            x2 = i + 1 - BAR_WIDTH / 2
            y = bar_starts[i] + bar_heights[i]
            ax.plot([x1, x2], [y, y], 'k--', linewidth=1.5, alpha=0.5)

        # Add value annotations on bars
        for i, (bar, value, increment) in enumerate(zip(bars, values, increments)):
            if i == 0:
                continue  # Skip start bar
            elif i == n_categories - 1:
                # Total bar - annotate total value
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                       f'${value:.2f}B',
                       ha='center', va='bottom', fontsize=ANNOTATION_SIZE,
                       fontweight='bold', fontname='Times New Roman',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='black', linewidth=1.5))
            else:
                # Increment bars - annotate increment value
                height = bar.get_height()
                y_pos = bar.get_y() + height / 2
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                       f'+${increment:.2f}B',
                       ha='center', va='center', fontsize=ANNOTATION_SIZE - 2,
                       fontweight='bold', fontname='Times New Roman', color='white',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Styling
        ax.set_ylabel('Cumulative Cost (Billion USD)', fontsize=LABEL_SIZE,
                     fontweight='bold', fontname='Times New Roman')
        ax.set_xticks(range(n_categories))
        ax.set_xticklabels(categories, fontsize=TICK_SIZE, fontweight='bold',
                          fontname='Times New Roman', rotation=0)
        ax.tick_params(axis='y', labelsize=TICK_SIZE, width=2, length=6)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontname('Times New Roman')

        # Dynamic Y-axis limits
        max_value = max(values)
        ax.set_ylim(0, max_value * 1.15)

        # Add scenario title as xlabel
        title_map = {
            'Scenario A': 'Scenario A: Space Elevator Only',
            'Scenario B': 'Scenario B: Rocket Only',
            'Scenario C': 'Scenario C: Hybrid System'
        }
        ax.set_xlabel(title_map[scenario_name], fontsize=LABEL_SIZE,
                     fontweight='bold', fontname='Times New Roman', labelpad=10)

        # Bold black borders
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_axisbelow(True)

        # Add cost efficiency annotation
        total_cost = values[-1]
        if scenario_name == 'Scenario A':
            efficiency = "Medium cost\nLong timeline"
        elif scenario_name == 'Scenario B':
            efficiency = "Highest cost\nShortest timeline"
        else:
            efficiency = "Optimal balance\nLow variance"

        ax.text(0.98, 0.98, efficiency, transform=ax.transAxes,
               fontsize=ANNOTATION_SIZE, fontweight='bold', fontname='Times New Roman',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor='black', linewidth=2))

    # Tight layout
    plt.tight_layout()

    # Save as high-resolution PDF
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Waterfall chart saved: {output_path}")


if __name__ == '__main__':
    # Test with sample data
    from pathlib import Path

    cost_breakdown = {
        'Scenario A': {
            'categories': ['Start', 'Fixed\nCost', 'Transport\nCost', 'Repair\nCost', 'Total'],
            'values': [0, 1.74, 26.24, 0.32, 28.30],
            'increments': [0, 1.74, 26.24, 0.32, 0]
        },
        'Scenario B': {
            'categories': ['Start', 'Fixed\nCost', 'Launch\nCost', 'Backup\nLaunches', 'Total'],
            'values': [0, 2.15, 38.50, 5.15, 45.80],
            'increments': [0, 2.15, 38.50, 5.15, 0]
        },
        'Scenario C': {
            'categories': ['Start', 'Fixed\nCost', 'Elevator\nTransport', 'Rocket\nLaunch', 'Repair\nCost', 'Total'],
            'values': [0, 1.95, 15.80, 13.95, 0.80, 32.50],
            'increments': [0, 1.95, 15.80, 13.95, 0.80, 0]
        }
    }

    output_path = Path(__file__).parent / 'outputs' / 'test_waterfall.pdf'
    output_path.parent.mkdir(exist_ok=True)
    create_waterfall_chart(cost_breakdown, output_path)
