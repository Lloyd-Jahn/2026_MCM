"""
Task 4 Visualization Module
============================
Purpose: Generate high-quality Nature-style publication figures
         All outputs saved as 300 DPI PDFs with absolute paths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from pathlib import Path
from parameters import *

# ============================================================================
# USER ADJUSTMENT GUIDE
# ============================================================================
# [Line 30-45]   Font sizes and style parameters
# [Line 50-65]   Color palette (modify COLOR_PALETTE in parameters.py)
# [Line 75-90]   Figure dimensions and DPI settings
# [Line 100-130] AHP weight comparison visualization
# [Line 200-250] Multi-objective radar chart
# [Line 350-400] Environmental impact heatmap
# [Line 500-550] Pareto frontier 3D visualization
# ============================================================================

# ============================================================================
# Global Style Settings (Nature-Style Requirements)
# ============================================================================

# Font settings (Times New Roman, Bold)
FONT_FAMILY = 'Times New Roman'
LABEL_SIZE = 18        # Axis labels
LEGEND_SIZE = 18       # Legend text
TICK_SIZE = 16         # Tick labels
TITLE_SIZE = 20        # Titles (only for internal reference, not displayed)

# Figure settings
FIG_DPI = 300          # High resolution for publication
FIG_FORMAT = 'pdf'     # Vector format

# Line and marker settings
LINE_WIDTH = 2.5
MARKER_SIZE = 10
GRID_ALPHA = 0.3

# Border settings
SPINE_WIDTH = 2.0      # Prominent black borders

# Output directory (absolute path)
OUTPUT_DIR = Path(__file__).parent.parent.resolve()  # Task4 folder


def setup_figure_style(ax):
    """
    Apply Nature-style formatting to matplotlib axes

    Args:
        ax: Matplotlib axes object
    """
    # Set all spines to be visible and bold
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_edgecolor('black')

    # Set tick parameters (bold)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE,
                   width=SPINE_WIDTH, length=6, direction='out')

    # Make tick labels bold
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily(FONT_FAMILY)

    # Grid (optional, light)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--', linewidth=0.8)


def set_label_style(ax, xlabel=None, ylabel=None):
    """
    Set bold labels for axes

    Args:
        ax: Matplotlib axes
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='bold',
                      fontfamily=FONT_FAMILY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='bold',
                      fontfamily=FONT_FAMILY)


def save_figure(fig, filename):
    """
    Save figure with absolute path as high-resolution PDF

    Args:
        fig: Matplotlib figure
        filename: Output filename (e.g., "fig1_ahp_weights.pdf")
    """
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, format=FIG_FORMAT, dpi=FIG_DPI,
                bbox_inches='tight', pad_inches=0.1)
    print(f"  Saved: {output_path}")


# ============================================================================
# Figure 1: AHP Weight Comparison (Main + Environmental Sub-weights)
# ============================================================================

def plot_ahp_weights():
    """
    Visualize AHP weights as horizontal bar chart

    Output: fig1_ahp_weights.pdf
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Subplot 1: Main Objective Weights ----
    objectives = ['Environment', 'Cost', 'Time', 'Reliability']
    weights = [w_environment, w_cost, w_time, w_reliability]
    colors = [COLOR_PALETTE['environment'], COLOR_PALETTE['cost'],
              COLOR_PALETTE['time'], COLOR_PALETTE['reliability']]

    # Sort by weight (descending)
    sorted_indices = np.argsort(weights)[::-1]
    objectives_sorted = [objectives[i] for i in sorted_indices]
    weights_sorted = [weights[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    # Horizontal bar chart
    y_pos = np.arange(len(objectives_sorted))
    bars1 = ax1.barh(y_pos, weights_sorted, color=colors_sorted,
                     edgecolor='black', linewidth=1.5, height=0.6)

    # Add value labels
    for i, (bar, weight) in enumerate(zip(bars1, weights_sorted)):
        ax1.text(weight + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{weight:.4f}\n({weight*100:.2f}%)',
                 va='center', ha='left', fontsize=TICK_SIZE - 2,
                 fontweight='bold', fontfamily=FONT_FAMILY)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(objectives_sorted, fontsize=LABEL_SIZE, fontweight='bold',
                        fontfamily=FONT_FAMILY)
    set_label_style(ax1, xlabel='Weight', ylabel='')
    ax1.set_xlim(0, 0.75)
    setup_figure_style(ax1)

    # ---- Subplot 2: Environmental Sub-Weights ----
    env_indicators = ['CO2', 'PM2.5', 'Ecology', 'Resource']
    env_weights = [w_CO2, w_PM25, w_eco, w_reso]
    env_colors = [COLOR_PALETTE['CO2'], COLOR_PALETTE['PM25'],
                  COLOR_PALETTE['ecology'], COLOR_PALETTE['resource']]

    # Sort by weight
    sorted_indices = np.argsort(env_weights)[::-1]
    env_indicators_sorted = [env_indicators[i] for i in sorted_indices]
    env_weights_sorted = [env_weights[i] for i in sorted_indices]
    env_colors_sorted = [env_colors[i] for i in sorted_indices]

    y_pos = np.arange(len(env_indicators_sorted))
    bars2 = ax2.barh(y_pos, env_weights_sorted, color=env_colors_sorted,
                     edgecolor='black', linewidth=1.5, height=0.6)

    # Add value labels
    for i, (bar, weight) in enumerate(zip(bars2, env_weights_sorted)):
        ax2.text(weight + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{weight:.4f}\n({weight*100:.2f}%)',
                 va='center', ha='left', fontsize=TICK_SIZE - 2,
                 fontweight='bold', fontfamily=FONT_FAMILY)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(env_indicators_sorted, fontsize=LABEL_SIZE, fontweight='bold',
                        fontfamily=FONT_FAMILY)
    set_label_style(ax2, xlabel='Weight', ylabel='')
    ax2.set_xlim(0, 0.75)
    setup_figure_style(ax2)

    plt.tight_layout()
    save_figure(fig, 'fig1_ahp_weights.pdf')
    plt.close()


# ============================================================================
# Figure 2: Multi-Objective Radar Chart for Three Scenarios
# ============================================================================

def plot_radar_chart(results):
    """
    Create radar chart comparing three scenarios on four objectives

    Args:
        results: Dict with keys 'A', 'B', 'C', each containing 'objectives_normalized'

    Output: fig2_multi_objective_radar.pdf
    """
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Four objectives (in order: Cost, Time, Reliability_inverted, Environment)
    categories = ['Cost', 'Time', 'Unreliability', 'Environment']
    N = len(categories)

    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Extract normalized objectives for each scenario
    values_a = results['A']['objectives_normalized'].tolist() + [results['A']['objectives_normalized'][0]]
    values_b = results['B']['objectives_normalized'].tolist() + [results['B']['objectives_normalized'][0]]
    values_c = results['C']['objectives_normalized'].tolist() + [results['C']['objectives_normalized'][0]]

    # Plot each scenario
    ax.plot(angles, values_a, 'o-', linewidth=LINE_WIDTH, label='Scenario A (Elevator)',
            color=COLOR_PALETTE['elevator'], markersize=MARKER_SIZE)
    ax.fill(angles, values_a, alpha=0.15, color=COLOR_PALETTE['elevator'])

    ax.plot(angles, values_b, 's-', linewidth=LINE_WIDTH, label='Scenario B (Rocket)',
            color=COLOR_PALETTE['rocket'], markersize=MARKER_SIZE)
    ax.fill(angles, values_b, alpha=0.15, color=COLOR_PALETTE['rocket'])

    ax.plot(angles, values_c, '^-', linewidth=LINE_WIDTH, label='Scenario C (Hybrid)',
            color=COLOR_PALETTE['hybrid'], markersize=MARKER_SIZE)
    ax.fill(angles, values_c, alpha=0.15, color=COLOR_PALETTE['hybrid'])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=LABEL_SIZE, fontweight='bold',
                       fontfamily=FONT_FAMILY)

    # Set radial limits (0 to 1 for normalized objectives)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                       fontsize=TICK_SIZE, fontweight='bold',
                       fontfamily=FONT_FAMILY)

    # Grid
    ax.grid(True, linestyle='--', alpha=GRID_ALPHA)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              fontsize=LEGEND_SIZE, frameon=True, edgecolor='black',
              prop={'weight': 'bold', 'family': FONT_FAMILY})

    save_figure(fig, 'fig2_multi_objective_radar.pdf')
    plt.close()


# ============================================================================
# Figure 3: Environmental Impact Heatmap (Scenarios vs Indicators)
# ============================================================================

def plot_environmental_heatmap(results):
    """
    Create heatmap showing environmental impact breakdown

    Args:
        results: Dict with optimization results for all scenarios

    Output: fig3_environmental_heatmap.pdf
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Environmental indicators
    indicators = ['PM2.5', 'CO2', 'Ecology', 'Resource']
    scenarios = ['Scenario A', 'Scenario B', 'Scenario C']

    # Extract environmental impact values for each scenario
    # Assuming we calculate component-wise contributions
    # For Scenario A:
    s_E_elec_a = results['A']['decision_vars']['s_E_elec']
    EI_components_a = np.array([
        w_PM25 * E_PM25_a / E_PM25_max,
        w_CO2 * E_CO2_a * (1 - s_E_elec_a) / E_CO2_max,
        w_reso * E_reso_a / E_reso_max,
        w_eco * E_eco_a / E_eco_max
    ])

    # For Scenario B:
    y_b = results['B']['decision_vars']['y']
    s_R_safe_avg_b = np.sum(y_b * s_R_safe) / np.sum(y_b)
    EI_components_b = np.array([
        w_PM25 * E_PM25_b / E_PM25_max,
        w_CO2 * E_CO2_b / E_CO2_max,
        w_reso * E_reso_b / E_reso_max,
        w_eco * E_eco_b * s_R_safe_avg_b / E_eco_max
    ])

    # For Scenario C:
    alpha_c = results['C']['decision_vars']['alpha']
    s_E_elec_c = results['C']['decision_vars']['s_E_elec']
    y_c = results['C']['decision_vars']['y']
    s_R_safe_avg_c = np.sum(y_c * s_R_safe) / np.sum(y_c)

    EI_components_c_elevator = np.array([
        w_PM25 * E_PM25_a / E_PM25_max,
        w_CO2 * E_CO2_a * (1 - s_E_elec_c) / E_CO2_max,
        w_reso * E_reso_a / E_reso_max,
        w_eco * E_eco_a / E_eco_max
    ])

    EI_components_c_rocket = np.array([
        w_PM25 * E_PM25_b / E_PM25_max,
        w_CO2 * E_CO2_b / E_CO2_max,
        w_reso * E_reso_b / E_reso_max,
        w_eco * E_eco_b * s_R_safe_avg_c / E_eco_max
    ])

    EI_components_c = alpha_c * EI_components_c_elevator + (1 - alpha_c) * EI_components_c_rocket

    # Create matrix for heatmap
    data = np.vstack([EI_components_a, EI_components_b, EI_components_c])

    # Plot heatmap
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=data.max())

    # Set ticks
    ax.set_xticks(np.arange(len(indicators)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(indicators, fontsize=LABEL_SIZE, fontweight='bold',
                       fontfamily=FONT_FAMILY)
    ax.set_yticklabels(scenarios, fontsize=LABEL_SIZE, fontweight='bold',
                       fontfamily=FONT_FAMILY)

    # Add value annotations
    for i in range(len(scenarios)):
        for j in range(len(indicators)):
            text = ax.text(j, i, f'{data[i, j]:.4f}',
                           ha="center", va="center", color="black",
                           fontsize=TICK_SIZE - 2, fontweight='bold',
                           fontfamily=FONT_FAMILY)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SIZE, width=SPINE_WIDTH)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily(FONT_FAMILY)
    cbar.set_label('Normalized Environmental Impact', fontsize=LABEL_SIZE,
                   fontweight='bold', fontfamily=FONT_FAMILY)

    # Labels
    set_label_style(ax, xlabel='Environmental Indicator', ylabel='')

    # Spine style
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)

    plt.tight_layout()
    save_figure(fig, 'fig3_environmental_heatmap.pdf')
    plt.close()


# ============================================================================
# Figure 4: Cost vs Time Pareto Frontier with Reliability Shading
# ============================================================================

def plot_pareto_frontier(results):
    """
    Create 2D Pareto frontier plot (Cost vs Time) with reliability as color

    Args:
        results: Dict with optimization results

    Output: fig4_pareto_cost_time.pdf
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract data for three scenarios
    scenarios = ['A', 'B', 'C']
    scenario_names = ['Elevator', 'Rocket', 'Hybrid']
    markers = ['o', 's', '^']
    colors = [COLOR_PALETTE['elevator'], COLOR_PALETTE['rocket'], COLOR_PALETTE['hybrid']]

    for i, (scenario, name, marker, color) in enumerate(zip(scenarios, scenario_names, markers, colors)):
        cost = results[scenario]['objectives_raw']['cost']
        time_years = results[scenario]['objectives_raw']['time_years']
        reliability = results[scenario]['objectives_raw']['reliability']

        # Plot point
        ax.scatter(cost, time_years, s=400, marker=marker, c=[reliability],
                   cmap='RdYlGn', vmin=0.8, vmax=1.0, edgecolors='black',
                   linewidths=2, label=f'Scenario {scenario} ({name})', zorder=3)

        # Add annotation
        ax.annotate(f'{name}\nR={reliability:.4f}',
                    xy=(cost, time_years),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=TICK_SIZE - 2, fontweight='bold',
                    fontfamily=FONT_FAMILY,
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.3,
                              edgecolor='black', linewidth=1.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    # Labels
    set_label_style(ax, xlabel='Total Cost (USD)', ylabel='Transport Time (years)')

    # Format axis to scientific notation
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

    # Colorbar for reliability
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0.8, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_SIZE, width=SPINE_WIDTH)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily(FONT_FAMILY)
    cbar.set_label('Reliability', fontsize=LABEL_SIZE, fontweight='bold',
                   fontfamily=FONT_FAMILY)

    # Legend
    ax.legend(loc='best', fontsize=LEGEND_SIZE - 2, frameon=True,
              edgecolor='black', prop={'weight': 'bold', 'family': FONT_FAMILY})

    setup_figure_style(ax)
    plt.tight_layout()
    save_figure(fig, 'fig4_pareto_cost_time.pdf')
    plt.close()


# ============================================================================
# Figure 5: Comprehensive Score Comparison Bar Chart
# ============================================================================

def plot_comprehensive_scores(results):
    """
    Create bar chart comparing comprehensive scores across scenarios

    Args:
        results: Dict with optimization results

    Output: fig5_comprehensive_scores.pdf
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ['A', 'B', 'C']
    scenario_names = ['Scenario A\n(Elevator)', 'Scenario B\n(Rocket)', 'Scenario C\n(Hybrid)']
    scores = [results[s]['comprehensive_score'] for s in scenarios]
    colors = [COLOR_PALETTE['elevator'], COLOR_PALETTE['rocket'], COLOR_PALETTE['hybrid']]

    # Bar chart
    x_pos = np.arange(len(scenarios))
    bars = ax.bar(x_pos, scores, color=colors, edgecolor='black',
                  linewidth=2, width=0.6)

    # Add value labels on top of bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                f'{score:.6f}',
                ha='center', va='bottom', fontsize=LABEL_SIZE - 2,
                fontweight='bold', fontfamily=FONT_FAMILY)

    # Add horizontal line for minimum (optimal)
    min_score = min(scores)
    optimal_scenario = scenarios[scores.index(min_score)]
    ax.axhline(y=min_score, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: Scenario {optimal_scenario}')

    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_names, fontsize=LABEL_SIZE, fontweight='bold',
                       fontfamily=FONT_FAMILY)
    set_label_style(ax, xlabel='', ylabel='Comprehensive Score (lower is better)')

    # Legend
    ax.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True,
              edgecolor='black', prop={'weight': 'bold', 'family': FONT_FAMILY})

    setup_figure_style(ax)
    plt.tight_layout()
    save_figure(fig, 'fig5_comprehensive_scores.pdf')
    plt.close()


# ============================================================================
# Master Visualization Function
# ============================================================================

def generate_all_figures(results):
    """
    Generate all publication-quality figures

    Args:
        results: Dict with keys 'A', 'B', 'C' containing optimization results
    """
    print("\nGenerating publication-quality visualizations...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Figure 1: AHP Weights
    print("  Creating Figure 1: AHP Weights...")
    plot_ahp_weights()

    # Figure 2: Multi-Objective Radar Chart
    print("  Creating Figure 2: Multi-Objective Radar Chart...")
    plot_radar_chart(results)

    # Figure 3: Environmental Heatmap
    print("  Creating Figure 3: Environmental Impact Heatmap...")
    plot_environmental_heatmap(results)

    # Figure 4: Pareto Frontier
    print("  Creating Figure 4: Pareto Frontier (Cost vs Time)...")
    plot_pareto_frontier(results)

    # Figure 5: Comprehensive Scores
    print("  Creating Figure 5: Comprehensive Score Comparison...")
    plot_comprehensive_scores(results)

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    # Test visualization functions with dummy data
    print("Testing visualization module...")

    # Create dummy results
    dummy_results = {
        'A': {
            'decision_vars': {'s_E_elec': 0.95, 'm_E': 3},
            'objectives_normalized': np.array([0.15, 0.45, 0.05, 0.02]),
            'objectives_raw': {
                'cost': 1.2e13,
                'time_days': 68000,
                'time_years': 186.3,
                'reliability': 0.998,
                'environmental_impact': 0.025
            },
            'comprehensive_score': 0.185
        },
        'B': {
            'decision_vars': {'y': np.ones(10), 'x': np.ones(10) * 50},
            'objectives_normalized': np.array([0.85, 0.12, 0.15, 0.92]),
            'objectives_raw': {
                'cost': 9.5e13,
                'time_days': 15000,
                'time_years': 41.1,
                'reliability': 0.950,
                'environmental_impact': 0.95
            },
            'comprehensive_score': 0.725
        },
        'C': {
            'decision_vars': {
                'alpha': 0.65,
                's_E_elec': 0.92,
                'm_E': 2,
                'y': np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
                'x': np.ones(10) * 30
            },
            'objectives_normalized': np.array([0.42, 0.22, 0.08, 0.35]),
            'objectives_raw': {
                'cost': 4.8e13,
                'time_days': 35000,
                'time_years': 95.9,
                'reliability': 0.985,
                'environmental_impact': 0.38
            },
            'comprehensive_score': 0.325
        }
    }

    generate_all_figures(dummy_results)
