import os
import sys
import calculate_time as calc
import visualize_timeline as viz

def main():
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.dirname(current_dir) # Go up one level to Task1/
    
    print("--- Starting Task 1 Analysis ---")
    
    # 1. Calculation
    metrics = calc.calculate_metrics()
    
    # 2. AHP & TOPSIS
    # User specified a=5 (Time is significantly more important than Cost)
    w_c, w_t = calc.run_ahp(a=5)
    topsis_res = calc.run_topsis(metrics, w_c, w_t)
    
    # 3. Visualization
    viz.plot_cumulative_mass_timeline(metrics, output_dir)
    
    # 4. Result Logging
    result_file = os.path.join(output_dir, "Result.txt")
    
    with open(result_file, "w") as f:
        f.write("# Task 1 Analysis Results\n")
        f.write("=========================\n\n")
        
        f.write("## 1. Parameters\n")
        f.write("Mass (M): 10^8 tons\n")
        f.write("AHP Preference parameter (a): 5 (Time is dominant)\n")
        f.write(f"Calculated Weights -> Cost: {w_c:.4f}, Time: {w_t:.4f}\n\n")
        
        f.write("## 2. Raw Metrics\n")
        for s in ["A", "B", "C"]:
            c_val = metrics[s]["Cost"]
            t_val = metrics[s]["Time"]
            f.write(f"Scenario {s}:\n")
            f.write(f"  - Cost: ${c_val:,.2f} USD ({c_val/1e12:.2f} Trillion)\n")
            f.write(f"  - Time: {t_val:.2f} Years\n")
        f.write("\n")
        
        f.write("## 3. TOPSIS Ranking\n")
        # Sort by score descending
        sorted_scenarios = sorted(topsis_res.items(), key=lambda item: item[1]['Score'], reverse=True)
        
        f.write(f"{'Rank':<6} | {'Scenario':<10} | {'Score':<10}\n")
        f.write("-" * 35 + "\n")
        for rank, (scenario, data) in enumerate(sorted_scenarios, 1):
            f.write(f"{rank:<6} | {scenario:<10} | {data['Score']:.4f}\n")
            
    print(f"Analysis complete. Results saved to {result_file}")

if __name__ == "__main__":
    main()
