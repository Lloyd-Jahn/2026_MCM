import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.scale import FuncScale

# --- 2. Visualization Aesthetics & Constraints (Guideline) ---
def set_nature_style():
    """
    Configures Matplotlib to match Nature-style publication standards.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.linewidth": 2,  # Prominent black bold borders
        "axes.edgecolor": "black",
        "xtick.major.width": 2,
        "ytick.major.width": 2,
        "xtick.labelsize": 16, # tick_size=16
        "ytick.labelsize": 16,
        "axes.labelsize": 18,  # label_size=18
        "legend.fontsize": 18, # legend_size=18
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "text.usetex": False # Simplified for standard installs, can enable if LaTeX available
    })

def plot_cumulative_mass_timeline(metrics, output_dir):
    """
    Generates a cumulative mass vs time line chart for Scenarios A, B, C.
    x-axis: Time (Years)
    y-axis: Cumulative Mass Delivered (tons)
    """
    set_nature_style()

    scenarios = ["A", "B", "C"]
    total_mass = metrics["Meta"]["Mass"]

    # Timeline settings
    start_year = 2050
    end_year = 2500
    plot_horizon_years = end_year - start_year
    year_grid = np.linspace(start_year, end_year, 800)

    # Professional Palette
    colors = {
        "A": "#377eb8",
        "B": "#e41a1c",
        "C": "#4daf4a"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for s in scenarios:
        rate = metrics[s]["AnnualCapacity"]
        elapsed = year_grid - start_year
        cumulative = np.minimum(rate * elapsed, total_mass)
        
        # For A and C, only plot until they reach total_mass
        if s in ["A", "C"]:
            completion_time = total_mass / rate
            completion_year = start_year + completion_time
            mask = year_grid <= completion_year
            year_plot = year_grid[mask]
            cumulative_plot = cumulative[mask]
        else:
            year_plot = year_grid
            cumulative_plot = cumulative
        
        ax.plot(year_plot, cumulative_plot, color=colors[s], linewidth=3.5, label=f"Scenario {s}")

    ax.set_xlabel("Year", weight="bold")
    ax.set_ylabel("Cumulative Mass Delivered/ 10$^8$ tons", weight="bold")

    # Apply non-linear y-axis scaling (log-like compression at top, moderate at bottom)
    forward = lambda x: np.log1p(x / 5e6) * 5e6  # Moderate compression function
    inverse = lambda x: (np.expm1(x / 5e6)) * 5e6
    ax.set_yscale(FuncScale(ax, (forward, inverse)))
    
    # Set linear-looking tick labels despite non-linear scaling (normalized to 10^8)
    yticks = np.linspace(0, 1e8, 11)  # Linear spacing in data space
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y/1e8:.1f}" for y in yticks], fontsize=16, weight='bold')

    ax.set_xlim([start_year, end_year])
    ax.set_ylim([0, total_mass * 1.05])

    ax.grid(True, which="both", color="#CCCCCC", linewidth=0.8)

    leg = ax.legend(loc="upper right", frameon=True, fancybox=False)
    leg.get_frame().set_edgecolor('lightgray')
    leg.get_frame().set_linewidth(1.5)

    # Framing (spines)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    output_path = os.path.join(output_dir, "fig1_cumulative_mass_timeline.pdf")
    plt.savefig(output_path)
    print(f"Generated: {output_path}")
    plt.close()
