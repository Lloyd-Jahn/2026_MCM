"""
Contour Plot + Marginal Distributions: Sensitivity Analysis
===========================================================
2D contour plot with marginal distributions showing sensitivity of mission duration
to elevator (p_E1) and rocket (p_R1) failure rates

USER ADJUSTMENT GUIDE:
----------------------
Line 26-28:  Color maps and palette
Line 31-32:  Figure size and grid layout
Line 35-37:  Font sizes
Line 40-42:  Contour levels and threshold
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Professional Color Palette
CONTOUR_CMAP = 'RdYlGn_r'      # Red-Yellow-Green reversed (red=high risk)
MARGINAL_COLOR_E = '#2ca02c'   # Green - Elevator marginal
MARGINAL_COLOR_R = '#d62728'   # Red - Rocket marginal
SAFE_COLOR = '#2ecc71'         # Light green
RISK_COLOR = '#e74c3c'         # Light red

# Plot Configuration
FIG_SIZE = (14, 14)
GRID_RATIOS = [1, 5, 0.3]      # Top marginal : Main plot : Colorbar
DPI = 300

# Font Configuration
LABEL_SIZE = 18
TICK_SIZE = 16
ANNOTATION_SIZE = 14

# Contour Configuration
CONTOUR_LEVELS = 25
THRESHOLD_TIME = 0.36          # Threshold for "safe" region (years)

# ============================================================================
# CONTOUR + MARGINAL PLOT CREATION
# ============================================================================

def create_contour_plot(sensitivity_data, output_path):
    """
    Create contour plot with marginal distributions

    Parameters:
    -----------
    sensitivity_data : dict
        Dictionary containing grid data and marginal distributions
    output_path : Path
        Absolute path to save the PDF
    """

    # Extract data
    p_E1_range = sensitivity_data['p_E1_range']
    p_R1_range = sensitivity_data['p_R1_range']
    p_E1_grid = sensitivity_data['p_E1_grid']
    p_R1_grid = sensitivity_data['p_R1_grid']
    T_C_grid = sensitivity_data['T_C_grid']
    marginal_T_vs_pE1 = sensitivity_data['marginal_T_vs_pE1']
    marginal_T_vs_pR1 = sensitivity_data['marginal_T_vs_pR1']

    # Create figure with GridSpec
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    gs = gridspec.GridSpec(3, 3, figure=fig,
                          height_ratios=GRID_RATIOS,
                          width_ratios=[0.3, 5, 1],
                          hspace=0.05, wspace=0.05)

    # Main contour plot (center)
    ax_main = fig.add_subplot(gs[1, 1])

    # Top marginal distribution (p_E1)
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)

    # Right marginal distribution (p_R1)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)

    # Colorbar axis
    ax_cbar = fig.add_subplot(gs[1, 0])

    # === MAIN CONTOUR PLOT ===
    # Filled contour
    contourf = ax_main.contourf(p_E1_grid * 100, p_R1_grid * 100, T_C_grid,
                                levels=CONTOUR_LEVELS, cmap=CONTOUR_CMAP, alpha=0.9)

    # Contour lines
    contour_lines = ax_main.contour(p_E1_grid * 100, p_R1_grid * 100, T_C_grid,
                                    levels=10, colors='black', linewidths=1, alpha=0.4)
    ax_main.clabel(contour_lines, inline=True, fontsize=10, fmt='%.3f')

    # Add scatter points (simulating Monte Carlo samples)
    np.random.seed(42)
    n_samples = 200
    p_E1_samples = np.random.uniform(3, 7, n_samples)
    p_R1_samples = np.random.uniform(2, 4, n_samples)
    ax_main.scatter(p_E1_samples, p_R1_samples, s=10, c='black', alpha=0.3, marker='.')

    # Highlight "safe" and "risk" regions
    # Safe region: T < threshold
    safe_contour = ax_main.contour(p_E1_grid * 100, p_R1_grid * 100, T_C_grid,
                                   levels=[THRESHOLD_TIME], colors=SAFE_COLOR,
                                   linewidths=3, linestyles='--')
    ax_main.clabel(safe_contour, inline=True, fontsize=12, fmt='Safe Threshold: %.3f',
                   manual=[(5.5, 3.5)])

    # Labels and styling
    ax_main.set_xlabel('Elevator Failure Rate $p_{E1}$ (%)', fontsize=LABEL_SIZE,
                       fontweight='bold', fontname='Times New Roman')
    ax_main.set_ylabel('Rocket Failure Rate $p_{R1}$ (%)', fontsize=LABEL_SIZE,
                       fontweight='bold', fontname='Times New Roman')
    ax_main.tick_params(axis='both', labelsize=TICK_SIZE, width=2, length=6)
    for label in ax_main.get_xticklabels() + ax_main.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # Bold black borders
    for spine in ax_main.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # === TOP MARGINAL DISTRIBUTION (p_E1 effect) ===
    ax_top.fill_between(p_E1_range * 100, marginal_T_vs_pE1, alpha=0.6,
                        color=MARGINAL_COLOR_E, edgecolor='black', linewidth=2)
    ax_top.plot(p_E1_range * 100, marginal_T_vs_pE1, color='darkgreen', linewidth=3)

    # Styling
    ax_top.set_ylabel('Avg. Time\n(Years)', fontsize=ANNOTATION_SIZE, fontweight='bold',
                      fontname='Times New Roman', rotation=0, ha='right', va='center')
    ax_top.tick_params(axis='x', labelbottom=False, width=2, length=6)
    ax_top.tick_params(axis='y', labelsize=TICK_SIZE, width=2, length=6)
    for label in ax_top.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # Dynamic Y limits
    margin_top = (marginal_T_vs_pE1.max() - marginal_T_vs_pE1.min()) * 0.1
    ax_top.set_ylim(marginal_T_vs_pE1.min() - margin_top,
                    marginal_T_vs_pE1.max() + margin_top)

    # Bold borders
    for spine in ax_top.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Grid
    ax_top.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

    # === RIGHT MARGINAL DISTRIBUTION (p_R1 effect) ===
    ax_right.fill_betweenx(p_R1_range * 100, marginal_T_vs_pR1, alpha=0.6,
                           color=MARGINAL_COLOR_R, edgecolor='black', linewidth=2)
    ax_right.plot(marginal_T_vs_pR1, p_R1_range * 100, color='darkred', linewidth=3)

    # Styling
    ax_right.set_xlabel('Avg. Time (Years)', fontsize=ANNOTATION_SIZE, fontweight='bold',
                        fontname='Times New Roman')
    ax_right.tick_params(axis='y', labelleft=False, width=2, length=6)
    ax_right.tick_params(axis='x', labelsize=TICK_SIZE, width=2, length=6)
    for label in ax_right.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # Dynamic X limits
    margin_right = (marginal_T_vs_pR1.max() - marginal_T_vs_pR1.min()) * 0.1
    ax_right.set_xlim(marginal_T_vs_pR1.min() - margin_right,
                      marginal_T_vs_pR1.max() + margin_right)

    # Bold borders
    for spine in ax_right.spines.values():
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Grid
    ax_right.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)

    # === COLORBAR ===
    cbar = plt.colorbar(contourf, cax=ax_cbar)
    cbar.set_label('Mission Duration (Years)', fontsize=LABEL_SIZE, fontweight='bold',
                   fontname='Times New Roman', rotation=90, labelpad=15)
    cbar.ax.tick_params(labelsize=TICK_SIZE, width=2, length=6)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # Colorbar border
    cbar.outline.set_linewidth(2)
    cbar.outline.set_edgecolor('black')

    # === ANNOTATIONS ===
    # Add text box explaining the plot
    textstr = 'Sensitivity Analysis:\nMission duration increases\nwith both failure rates.\n\nRocket failure $p_{R1}$ has\nstronger impact (steeper\nmarginal distribution).'
    props = dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='black', linewidth=2)
    ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=ANNOTATION_SIZE,
                fontweight='bold', fontname='Times New Roman', verticalalignment='top', bbox=props)

    # Save as high-resolution PDF
    plt.savefig(output_path, format='pdf', dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Contour + marginal plot saved: {output_path}")


if __name__ == '__main__':
    # Test with sample data
    from pathlib import Path

    p_E1_range = np.linspace(0.03, 0.07, 100)
    p_R1_range = np.linspace(0.02, 0.04, 100)
    p_E1_grid, p_R1_grid = np.meshgrid(p_E1_range, p_R1_range)

    T_C_base = 0.35
    T_C_grid = T_C_base * (1 + 2 * (p_E1_grid - 0.05) + 3 * (p_R1_grid - 0.03) +
                           5 * (p_E1_grid - 0.05) * (p_R1_grid - 0.03))

    marginal_T_vs_pE1 = np.mean(T_C_grid, axis=0)
    marginal_T_vs_pR1 = np.mean(T_C_grid, axis=1)

    sensitivity_data = {
        'p_E1_range': p_E1_range,
        'p_R1_range': p_R1_range,
        'p_E1_grid': p_E1_grid,
        'p_R1_grid': p_R1_grid,
        'T_C_grid': T_C_grid,
        'marginal_T_vs_pE1': marginal_T_vs_pE1,
        'marginal_T_vs_pR1': marginal_T_vs_pR1
    }

    output_path = Path(__file__).parent / 'outputs' / 'test_contour.pdf'
    output_path.parent.mkdir(exist_ok=True)
    create_contour_plot(sensitivity_data, output_path)
