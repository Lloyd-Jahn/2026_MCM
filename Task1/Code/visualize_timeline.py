import matplotlib.pyplot as plt
import numpy as np
import os

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
    
    # 定义目标线位置 (1.0 × 10^8 tons)
    target_y = 1.0e8

    # Timeline settings
    start_year = 2050
    end_year = 2500

    # Professional Palette
    colors = {
        "A": "#377eb8",
        "B": "#e41a1c",
        "C": "#4daf4a"
    }
    
    # 场景名称映射
    scenario_names = {
        "A": "Scenario A",
        "B": "Scenario B", 
        "C": "Scenario C"
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # 存储每个场景达到目标的时间点
    target_times = {}
    target_markers = {}

    for s in scenarios:
        # 直接获取 main.py 算好的时间序列数据
        years = metrics[s]["History_Years"]
        mass = metrics[s]["History_Mass"]
        
        # 找到质量首次超过目标值的索引
        valid_idx = np.searchsorted(mass, target_y)
        
        if valid_idx < len(mass) and valid_idx > 0:
            # 线性插值找到精确的到达时间
            x1, y1 = years[valid_idx-1], mass[valid_idx-1]
            x2, y2 = years[valid_idx], mass[valid_idx]
            
            # 线性插值公式
            target_time = x1 + (x2 - x1) * (target_y - y1) / (y2 - y1)
            target_times[s] = target_time
            
            # 记录用于标记的点
            target_markers[s] = (target_time, target_y)
            
            # 绘制线时截断到目标值
            plot_years = years[:valid_idx+1].copy()
            plot_mass = mass[:valid_idx+1].copy()
            
            # 将最后一个点修正为目标值
            plot_years[-1] = target_time
            plot_mass[-1] = target_y
        else:
            # 如果从未达到目标，使用完整数据
            plot_years = years
            plot_mass = mass
            if s in metrics[s] and "Target_Year" in metrics[s]:
                target_times[s] = metrics[s]["Target_Year"]

        ax.plot(plot_years, plot_mass, color=colors[s], linewidth=3.5, label=scenario_names[s])
        
        # 在达到目标的位置添加标记点
        if s in target_markers:
            marker_time, marker_mass = target_markers[s]
            ax.plot(marker_time, marker_mass, 'o', color=colors[s], 
                   markersize=12, markeredgecolor='white', markeredgewidth=2,
                   zorder=5)

    # 添加 y = 1.0 × 10^8 水平指示线
    ax.axhline(y=target_y, color='gray', linestyle='--', linewidth=2.5, 
               alpha=0.7)
    
    # 为指示线添加标签
    ax.text(end_year, target_y * 1.02, 
            fontsize=16, weight='bold', color='gray',
            ha='right', va='bottom', backgroundcolor='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='lightgray', alpha=0.9))

    ax.set_xlabel("Year", weight="bold")
    ax.set_ylabel("Cumulative Mass Delivered/ 10$^8$ tons", weight="bold")

    # 设置线性坐标轴刻度标签
    y_max = total_mass * 1.05
    yticks = np.linspace(0, y_max, 6)
    
    # 确保目标线出现在刻度上（如果不在已有的刻度中）
    if target_y <= y_max and target_y not in yticks:
        # 将目标值添加到刻度中
        all_ticks = np.sort(np.concatenate([yticks, [target_y]]))
        # 限制刻度数量，避免过多
        if len(all_ticks) <= 8:
            yticks = all_ticks
    
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y/1e8:.1f}" for y in yticks], fontsize=16, weight='bold')
    
    # 突出显示目标刻度
    for label in ax.get_yticklabels():
        if label.get_text() == "1.0":
            label.set_color('darkred')
            label.set_weight('bold')
            label.set_fontsize(18)

    ax.set_xlim([start_year, end_year])
    ax.set_ylim([0, y_max])

    ax.grid(True, which="both", color="#CCCCCC", linewidth=0.8)

    # 创建图例
    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, 
                   framealpha=0.9, edgecolor='black')
    leg.get_frame().set_linewidth(1.5)

    # Framing (spines)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # 调整布局
    plt.tight_layout()

    output_path = os.path.join(output_dir, "fig1_cumulative_mass_timeline.pdf")
    plt.savefig(output_path)
    print(f"Generated: {output_path}")
    plt.close()
    
    # 打印到达时间信息
    print("\nTarget (1.0 × 10^8 tons) Achievement Years:")
    for s in scenarios:
        if s in target_times:
            print(f"{scenario_names[s]}: Year {target_times[s]:.0f}")