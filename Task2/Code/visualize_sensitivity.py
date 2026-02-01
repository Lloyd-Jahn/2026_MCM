"""
visualize_sensitivity.py

Purpose: Generate Nature-style publication-quality visualizations for sensitivity analysis.

Key Features:
- Plot 1: Dual 3D surface plots (top 3 sensitivity factors: p_R1, p_E1, beta_E1)
- Plot 2: Tornado diagram (sensitivity ranking)
- Plot 3: Cumulative distribution functions (CDFs)
- Plot 4: 3D surface plot (time vs. dual-parameter variation)

All plots adhere to Nature-style standards:
- Times New Roman font, bold labels (18pt), bold ticks (16pt)
- Prominent black borders (all 4 spines)
- No internal titles (context from axis labels only)
- Dynamic Y-axis limits
- High-resolution PDF export (300 DPI)

User Adjustment Guide:
- Line 37-41: Font sizes (LEGEND_SIZE, LABEL_SIZE, TICK_SIZE, LINEWIDTH_SPINE)
- Line 46-52: Color palettes (Scenario colors and heatmap colors)
- Line 57-59: Figure sizes (FIG_WIDTH, FIG_HEIGHT, DPI)
- Line 66: Export format (FILE_FORMAT)
- Line 183: Viewing angles for 3D plots (elev, azim)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.font_manager as fm
import os
import matplotlib.tri as tri

# ============================================================================
# Typography Settings (Nature-style)
# ============================================================================
FONT_FAMILY = 'Times New Roman'
LEGEND_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 16
LINEWIDTH_SPINE = 2.0

# ============================================================================
# Color Palettes
# ============================================================================
# Gradient palette for tornado diagram (matching gradient image)
GRADIENT_COLORS = ['#5EB9C2', '#4BA6C9', '#5B8FC6', '#7B75B4', '#935891', '#9B3E63']

COLOR_PALETTE = {
    'scenario_A': '#1f77b4',  # Blue
    'scenario_B': '#ff7f0e',  # Orange
    'scenario_C': '#2ca02c',  # Green
    'heatmap_low': '#3288bd',  # Blue (low cost)
    'heatmap_high': '#d53e4f',  # Red (high cost)
}

# ============================================================================
# Figure Settings
# ============================================================================
FIG_WIDTH = 10
FIG_HEIGHT = 8
DPI = 300

# ============================================================================
# Export Settings
# ============================================================================
# OUTPUT_DIR is set dynamically in generate_all_plots() using absolute path
OUTPUT_DIR = None  # Will be set at runtime
FILE_FORMAT = 'pdf'


# ============================================================================
# Utility Function: Set Nature-style Axes
# ============================================================================

def set_nature_style_axes(ax, xlabel, ylabel, xlim=None, ylim=None):
    """
    Apply Nature-style formatting to matplotlib axes.

    Args:
        ax: Matplotlib axes object
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        xlim (tuple): X-axis limits (optional)
        ylim (tuple): Y-axis limits (optional)
    """
    # Set labels
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=LINEWIDTH_SPINE)

    # Set bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_family(FONT_FAMILY)

    # Set spine width and color
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH_SPINE)
        spine.set_color('black')
        spine.set_visible(True)

    # Set limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Ensure all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


# ============================================================================
# Plot 1: Dual 3D Surface (Top 3 Sensitivity Factors)
# ============================================================================

def plot_cost_heatmap(df_mc_samples, df_mc_results, output_path):
    """
    Nature-style Dual 3D Surface Plot using Polynomial Response Surface.
    Isolates target variable interactions by holding other parameters at median baselines.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # 1. Fit the Global Response Model
    # We use all 7 parameters to capture the true underlying physics
    X = df_mc_samples[['p_E1', 'beta_E1', 'p_R1', 'p_E2', 't_E2', 'p_R2', 't_R2']].values
    y = df_mc_results['C_C'].values / 1e12  # Trillion USD
    
    # Degree 2 captures the primary interactions (e.g., p_E1 * p_R1) smoothly
    poly = PolynomialFeatures(degree=2)
    model = LinearRegression().fit(poly.fit_transform(X), y)

    # Calculate neutral baselines (medians) for "zeroing out" other effects
    baselines = df_mc_samples.median().values
    grid_res = 50 

    # Define Nature-style divergent colormap
    colors = [COLOR_PALETTE['heatmap_low'], 'white', COLOR_PALETTE['heatmap_high']]
    cmap = LinearSegmentedColormap.from_list('nature_heat', colors, N=100)

    fig = plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT))

    # ========================================================================
    # Left Plot: Cost vs. p_E1 & p_R1 (Others held at baseline)
    # ========================================================================
    ax1 = fig.add_subplot(121, projection='3d')
    p_E1_range = np.linspace(df_mc_samples['p_E1'].min(), df_mc_samples['p_E1'].max(), grid_res)
    p_R1_range = np.linspace(df_mc_samples['p_R1'].min(), df_mc_samples['p_R1'].max(), grid_res)
    P_E1, P_R1 = np.meshgrid(p_E1_range, p_R1_range)

    # Create synthetic prediction matrix
    X_pred1 = np.tile(baselines, (grid_res**2, 1))
    X_pred1[:, 0] = P_E1.ravel() # p_E1 index
    X_pred1[:, 2] = P_R1.ravel() # p_R1 index
    Z1 = model.predict(poly.transform(X_pred1)).reshape(grid_res, grid_res)

    surf1 = ax1.plot_surface(P_R1*100, P_E1*100, Z1, cmap=cmap, alpha=0.9, antialiased=True, linewidth=0)

    # ========================================================================
    # Right Plot: Cost vs. p_E1 & beta_E1 (Others held at baseline)
    # ========================================================================
    ax2 = fig.add_subplot(122, projection='3d')
    beta_range = np.linspace(df_mc_samples['beta_E1'].min(), df_mc_samples['beta_E1'].max(), grid_res)
    BETA, P_E1_2 = np.meshgrid(beta_range, p_E1_range)

    X_pred2 = np.tile(baselines, (grid_res**2, 1))
    X_pred2[:, 0] = P_E1_2.ravel() # p_E1 index
    X_pred2[:, 1] = BETA.ravel() # beta_E1 index
    Z2 = model.predict(poly.transform(X_pred2)).reshape(grid_res, grid_res)

    surf2 = ax2.plot_surface(BETA*100, P_E1_2*100, Z2, cmap=cmap, alpha=0.9, antialiased=True, linewidth=0)

    # Formatting (Nature Style)
    for ax, xlab, ylab in zip([ax1, ax2], 
                             [r'$\mathbf{p_{R1}}$ (Launch Fail, %)', r'$\mathbf{\beta_{E1}}$ (Capacity Loss, %)'], 
                             [r'$\mathbf{p_{E1}}$ (Tether Sway, %)', r'$\mathbf{p_{E1}}$ (Tether Sway, %)']):
        ax.set_xlabel(xlab, fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY, labelpad=12)
        ax.set_ylabel(ylab, fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY, labelpad=12)
        ax.set_zlabel('Mean Cost\n(Trillion USD)', fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY, labelpad=15)
        
        # Set bold ticks
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=LINEWIDTH_SPINE)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontweight('bold')
            label.set_family(FONT_FAMILY)
        
        ax.view_init(elev=25, azim=135)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.invert_yaxis()

    # Shared Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(surf2, cax=cbar_ax)
    cbar.set_label('Mean Cost (Trillion USD)', fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_family(FONT_FAMILY)

    plt.subplots_adjust(left=0.05, right=0.90, wspace=0.3)
    
    # Save using absolute path
    abs_output = os.path.abspath(output_path)
    plt.savefig(abs_output, format=FILE_FORMAT, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Isolated 3D Surface plots saved to: {abs_output}")


# ============================================================================
# Plot 2: Tornado Diagram (Sensitivity Ranking)
# ============================================================================

def plot_tornado_diagram(sobol_indices, output_metric, output_path):
    """
    Generate tornado diagram showing ranked parameter sensitivities.

    Args:
        sobol_indices (dict): Sobol indices from analysis
        output_metric (str): Output metric name (e.g., 'C_C', 'T_C')
        output_path (str): Output file path
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # Extract total-order indices
    ST_dict = sobol_indices[output_metric]['ST']
    param_names = list(ST_dict.keys())
    ST_values = np.array(list(ST_dict.values()))

    # Sort by descending sensitivity
    sorted_indices = np.argsort(ST_values)[::-1]
    param_names_sorted = [param_names[i] for i in sorted_indices]
    ST_sorted = ST_values[sorted_indices]

    # Create parameter labels
    param_labels = {
        'p_E1': r'$p_{E1}$ (Tether Swaying)',
        'beta_E1': r'$\beta_{E1}$ (Capacity Reduction)',
        'p_E2': r'$p_{E2}$ (Climber Breakdown)',
        't_E2': r'$t_{E2}$ (Breakdown Duration)',
        'p_R1': r'$p_{R1}$ (Launch Failure)',
        'p_R2': r'$p_{R2}$ (Site Maintenance)',
        't_R2': r'$t_{R2}$ (Maintenance Duration)'
    }
    labels_sorted = [param_labels[name] for name in param_names_sorted]

    # Plot horizontal bars with gradient colors
    y_pos = np.arange(len(param_names_sorted))
    # Extend gradient colors if needed
    colors_tornado = GRADIENT_COLORS * (len(y_pos) // len(GRADIENT_COLORS) + 1)

    bars = ax.barh(y_pos, ST_sorted * 100, color=colors_tornado[:len(y_pos)])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted)
    ax.invert_yaxis()  # Highest at top

    # Set Nature-style axes
    set_nature_style_axes(ax,
                          xlabel='Total Sobol Index (%)',
                          ylabel='')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, ST_sorted)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%',
                va='center', fontsize=TICK_SIZE, fontweight='bold',
                family=FONT_FAMILY)

    plt.tight_layout()

    # Debug: print absolute path
    abs_path = os.path.abspath(output_path)
    print(f"  Attempting to save to: {abs_path}")

    try:
        plt.savefig(output_path, format=FILE_FORMAT, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved successfully")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        plt.close()
        raise


# ============================================================================
# Plot 3: Cumulative Distribution Functions (CDFs)
# ============================================================================

def plot_cdfs(df_mc_results, statistics, metric='cost', output_path=None):
    """
    Generate probability density plots for all three scenarios in vertical arrangement.

    Args:
        df_mc_results (pd.DataFrame): MC output results
        statistics (dict): Statistical summary
        metric (str): 'cost' or 'time'
        output_path (str): Output file path
    """
    from scipy.stats import gaussian_kde

    # Create vertical layout: 3 rows, 1 column (flatter rectangles)
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 1.2))

    colors = [COLOR_PALETTE['scenario_A'],
              COLOR_PALETTE['scenario_B'],
              COLOR_PALETTE['scenario_C']]
    scenario_names = ['Scenario A (Elevator)', 'Scenario B (Rockets)', 'Scenario C (Hybrid)']
    scenarios = ['A', 'B', 'C']

    # Create each subplot as a probability density function
    for idx, (ax, scenario, color, label) in enumerate(zip(axes, scenarios, colors, scenario_names)):
        if metric == 'cost':
            col = f'C_{scenario}'
            data = df_mc_results[col].values / 1e12  # Trillion USD
            unit = 'Trillion USD'
        else:
            col = f'T_{scenario}'
            data = df_mc_results[col].values
            unit = 'Years'

        # Calculate kernel density estimation
        kde = gaussian_kde(data, bw_method='scott')

        # Independent x-axis range for each subplot, centered on the peak
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        data_margin = data_range * 0.15  # 15% margin for better centering
        xlim = (data_min - data_margin, data_max + data_margin)

        x_plot = np.linspace(xlim[0], xlim[1], 1000)
        density = kde(x_plot)

        # Plot density curve
        ax.fill_between(x_plot, density, alpha=0.4, color=color, linewidth=0)
        ax.plot(x_plot, density, color=color, linewidth=3, label=label)

        # Add median line
        median = statistics[scenario][metric]['median']
        if metric == 'cost':
            median = median / 1e12

        median_density = kde(median)
        ax.axvline(median, color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax.plot(median, median_density, 'o', color=color,
                markersize=10, markeredgecolor='black', markeredgewidth=2)

        # Add 90% CI shaded region
        ci_90 = statistics[scenario][metric]['ci_90']
        if metric == 'cost':
            ci_90 = [c / 1e12 for c in ci_90]

        ax.axvspan(ci_90[0], ci_90[1], color=color, alpha=0.15, linewidth=0)

        # Set axes with larger font sizes for subplots
        xlabel = f'{"Cost" if metric == "cost" else "Time"} ({unit})' if idx == 2 else ''
        ylabel = 'Probability Density'

        # Use larger font sizes for subplot labels (22 for labels, 18 for ticks)
        ax.set_xlabel(xlabel, fontsize=22, fontweight='bold', family=FONT_FAMILY)
        ax.set_ylabel(ylabel, fontsize=22, fontweight='bold', family=FONT_FAMILY)

        # Set tick parameters
        ax.tick_params(axis='both', which='major', labelsize=18, width=LINEWIDTH_SPINE)

        # Set bold tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_family(FONT_FAMILY)

        # Set spine width and color
        for spine in ax.spines.values():
            spine.set_linewidth(LINEWIDTH_SPINE)
            spine.set_color('black')
            spine.set_visible(True)

        # Set limits
        if xlim is not None:
            ax.set_xlim(xlim)

        # Legend for each subplot with larger size
        legend_font = fm.FontProperties(family=FONT_FAMILY, weight='bold', size=20)
        ax.legend(frameon=True, loc='upper right', prop=legend_font)

        # Set y-axis to start from 0
        y_max = density.max()
        ax.set_ylim([0, y_max * 1.1])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, format=FILE_FORMAT, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


# ============================================================================
# Plot 4: 3D Surface (Time vs. Dual-Parameter Variation)
# ============================================================================

def plot_3d_surface(df_mc_samples, df_mc_results, output_path):
    """
    Generate 3D surface plot showing time as function of p_R1 and p_E1.

    Args:
        df_mc_samples (pd.DataFrame): MC parameter samples
        df_mc_results (pd.DataFrame): MC output results
        output_path (str): Output file path
    """
    fig = plt.figure(figsize=(FIG_WIDTH + 2, FIG_HEIGHT + 2))
    ax = fig.add_subplot(111, projection='3d')

    # Extract parameters and time
    p_R1 = df_mc_samples['p_R1'].values
    p_E1 = df_mc_samples['p_E1'].values
    time_C = df_mc_results['T_C'].values

    # Create grid
    grid_resolution = 40
    p_R1_grid = np.linspace(p_R1.min(), p_R1.max(), grid_resolution)
    p_E1_grid = np.linspace(p_E1.min(), p_E1.max(), grid_resolution)
    P_R1_grid, P_E1_grid = np.meshgrid(p_R1_grid, p_E1_grid)

    # Interpolate time on grid
    time_grid = griddata((p_R1, p_E1), time_C, (P_R1_grid, P_E1_grid), method='cubic')

    # Plot surface
    surf = ax.plot_surface(P_R1_grid * 100, P_E1_grid * 100, time_grid,
                           cmap=cm.viridis, alpha=0.9, edgecolor='none')

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('Mission Duration (Years)', fontsize=LABEL_SIZE,
                   fontweight='bold', family=FONT_FAMILY)
    cbar.ax.tick_params(labelsize=TICK_SIZE, width=LINEWIDTH_SPINE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_family(FONT_FAMILY)

    # Set labels
    ax.set_xlabel(r'$p_{R1}$ (Launch Failure Rate, %)',
                  fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY,
                  labelpad=10)
    ax.set_ylabel(r'$p_{E1}$ (Tether Swaying Probability, %)',
                  fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY,
                  labelpad=10)
    ax.set_zlabel('Mission Duration (Years)',
                  fontsize=LABEL_SIZE, fontweight='bold', family=FONT_FAMILY,
                  labelpad=10)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=LINEWIDTH_SPINE)

    # Set bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_fontweight('bold')
        label.set_family(FONT_FAMILY)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Debug: print absolute path
    abs_path = os.path.abspath(output_path)
    print(f"  Attempting to save to: {abs_path}")

    try:
        plt.savefig(output_path, format=FILE_FORMAT, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved successfully")
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        plt.close()
        raise


# ============================================================================
# Master Visualization Function
# ============================================================================

def generate_all_plots(sensitivity_results, output_dir='./'):
    """
    Generate all four Nature-style plots.

    Args:
        sensitivity_results (dict): Results from monte_carlo_sensitivity.py
        output_dir (str): Output directory for plots (default: current directory)
    """
    print("\n" + "="*70)
    print("GENERATING NATURE-STYLE VISUALIZATIONS")
    print("="*70)

    df_mc_samples = sensitivity_results['mc_samples']
    df_mc_results = sensitivity_results['mc_results']
    statistics = sensitivity_results['statistics']
    sobol_indices = sensitivity_results.get('sobol_indices', None)

    # Plot 1: Dual 3D surface plots (top 3 sensitivity factors)
    print("\n[1/4] Generating dual 3D surface plots (top 3 factors: p_R1, p_E1, beta_E1)...")
    output_path = os.path.join(output_dir, f'fig1_dual_3d_top3.{FILE_FORMAT}')
    plot_cost_heatmap(df_mc_samples, df_mc_results, output_path)

    # Plot 2: Tornado diagram (if Sobol indices available)
    if sobol_indices is not None:
        print("[2/4] Generating tornado diagram (Sobol sensitivity)...")
        output_path = os.path.join(output_dir, f'fig2_tornado_diagram.{FILE_FORMAT}')
        plot_tornado_diagram(sobol_indices, 'C_C', output_path)
    else:
        print("[2/4] Skipping tornado diagram (Sobol indices not available)")

    # Plot 3: CDFs for cost
    print("[3/4] Generating cumulative distribution functions (Cost)...")
    output_path = os.path.join(output_dir, f'fig3_cdf_cost.{FILE_FORMAT}')
    plot_cdfs(df_mc_results, statistics, metric='cost', output_path=output_path)

    # Plot 3b: CDFs for time
    print("      Generating cumulative distribution functions (Time)...")
    output_path = os.path.join(output_dir, f'fig4_cdf_time.{FILE_FORMAT}')
    plot_cdfs(df_mc_results, statistics, metric='time', output_path=output_path)

    # Plot 4: 3D surface
    print("[4/4] Generating 3D surface plot...")
    output_path = os.path.join(output_dir, f'fig5_3d_surface.{FILE_FORMAT}')
    plot_3d_surface(df_mc_samples, df_mc_results, output_path)

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED")
    print("="*70)


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 1000

    df_samples = pd.DataFrame({
        'p_E1': np.random.uniform(0.03, 0.07, n_samples),
        'beta_E1': np.random.uniform(0.2, 0.4, n_samples),
        'p_E2': np.random.uniform(0.01, 0.05, n_samples),
        't_E2': np.random.uniform(20, 40, n_samples),
        'p_R1': np.random.uniform(0.02, 0.04, n_samples),
        'p_R2': np.random.uniform(0.02, 0.06, n_samples),
        't_R2': np.random.uniform(45, 75, n_samples)
    })

    df_results = pd.DataFrame({
        'C_A': np.random.normal(11e12, 1e12, n_samples),
        'T_A': np.random.normal(190, 10, n_samples),
        'Z_A': np.random.normal(0.1, 0.01, n_samples),
        'C_B': np.random.normal(150e12, 15e12, n_samples),
        'T_B': np.random.normal(400, 30, n_samples),
        'Z_B': np.random.normal(0.9, 0.05, n_samples),
        'C_C': np.random.normal(45e12, 5e12, n_samples),
        'T_C': np.random.normal(145, 15, n_samples),
        'Z_C': np.random.normal(0.3, 0.02, n_samples)
    })

    # Calculate statistics
    from monte_carlo_sensitivity import calculate_statistics
    stats = calculate_statistics(df_results)

    # Test visualization
    test_results = {
        'mc_samples': df_samples,
        'mc_results': df_results,
        'statistics': stats
    }

    # calculate dynamic output path: ../ relative to this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir_resolved = os.path.join(script_dir, '..')

    print(f"Test mode: Saving plots to {output_dir_resolved}")
    generate_all_plots(test_results, output_dir=output_dir_resolved)
