import os
import sys
import numpy as np
import calculate_time as calc
import visualize_timeline as viz

def reconstruct_timeline_data(metrics):
    """
    Reconstructs timeline data based on analytical quadratic solutions.
    Physics: Cumulative Mass S(t) = (g/2)t^2 + (C0 - g/2)t
    """
    M_target = metrics["Meta"]["Mass"]
    
    # 1. Scenario A: Linear
    # S(t) = C0 * t
    t_A = metrics["A"]["Time"]
    C0_A = metrics["A"]["AnnualCapacity"]
    
    years_A = np.linspace(0, float(t_A), 100)
    mass_A = C0_A * years_A
    
    metrics["A"]["History_Years"] = 2050 + years_A
    metrics["A"]["History_Mass"] = mass_A
    
    # 2. Scenario B: Quadratic (Rocket Frequency Growth)
    # S(t) = (g/2)t^2 + (C0 - g/2)t
    t_B = metrics["B"]["Time"]
    C0_B = metrics["B"]["AnnualCapacity"]
    
    # Global Growth Rate: 10 sites * 1 launch/year * 150 tons/launch
    # FIXED: Hardcoded logic to match calculate_time.py
    g_B = 10 * 150 
    
    years_B = np.linspace(0, float(t_B), 100)
    mass_B = (g_B / 2.0) * years_B**2 + (C0_B - g_B / 2.0) * years_B
    
    metrics["B"]["History_Years"] = 2050 + years_B
    metrics["B"]["History_Mass"] = mass_B
    
    # 3. Scenario C: Quadratic (Hybrid)
    # Same growth g (from rockets only), but higher initial C0
    t_C = metrics["C"]["Time"]
    C0_C = metrics["C"]["AnnualCapacity"]
    g_C = g_B # Only rockets contribute to growth
    
    years_C = np.linspace(0, float(t_C), 100)
    mass_C = (g_C / 2.0) * years_C**2 + (C0_C - g_C / 2.0) * years_C
    
    metrics["C"]["History_Years"] = 2050 + years_C
    metrics["C"]["History_Mass"] = mass_C
    
    return metrics

def main():
    # Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.dirname(current_dir) 
    if not os.path.exists(output_dir):
        output_dir = current_dir
    
    print("--- Starting Task 1 Analysis (Final Config) ---")
    
    # 1. Calculation (Scalar Metrics)
    print("Step 1: Calculating Metrics with Analytical Solutions...")
    metrics = calc.calculate_metrics()
    
    # 2. Data Reconstruction for Plotting
    print("Step 2: Reconstructing Time-Series Data for Visualization...")
    metrics = reconstruct_timeline_data(metrics)
    
    # 3. AHP & TOPSIS
    print("Step 3: Running AHP and TOPSIS Evaluation...")
    w_c, w_t = calc.run_ahp(a=5)
    topsis_res = calc.run_topsis(metrics, w_c, w_t)
    
    # 4. Visualization
    print("Step 4: Generating Timeline Plot...")
    try:
        viz.plot_cumulative_mass_timeline(metrics, output_dir)
        print("   -> Plot saved successfully.")
    except Exception as e:
        print(f"   -> Warning: Visualization failed. {e}")
    
    # 5. Result Logging
    print("Step 5: Writing detailed report to Result.txt...")
    result_file = os.path.join(output_dir, "Result.txt")
    
    with open(result_file, "w") as f:
        f.write("# Task 1 Analysis Results (Final Model)\n")
        f.write("=======================================\n\n")
        
        f.write("## 1. Key Assumptions\n")
        f.write("1. **Payload Capacity**: Strictly Fixed at 150 tons per launch (Constant).\n")
        f.write("2. **Launch Frequency**: Increasing linearly. Each of the 10 sites adds 1 launch per year.\n")
        f.write("3. **Base Capacity**: 2050 baseline = Historical Average + 20 launches/year.\n")
        f.write("4. **Apex Logistics**: Includes transfer cost ($100/kg) from Apex Anchor to Moon surface.\n")
        f.write(f"5. **Preference**: AHP Parameter a=5 (Time weight: {w_t:.4f}, Cost weight: {w_c:.4f})\n\n")
        
        f.write("## 2. Calculated Metrics\n")
        for s in ["A", "B", "C"]:
            c_val = metrics[s]["Cost"]
            t_val = metrics[s]["Time"]
            capacity = metrics[s]["AnnualCapacity"]
            
            f.write(f"Scenario {s} ({metrics[s]['Label']}):\n")
            f.write(f"  - Total Cost: ${c_val:,.2f} ({c_val/1e12:.2f} Trillion)\n")
            f.write(f"  - Completion Time: {t_val:.2f} Years (Finish: {2050+t_val:.1f})\n")
            f.write(f"  - Initial Annual Capacity: {capacity:,.0f} tons/year\n")
            if s == "B" or s == "C":
                f.write(f"  - Growth Model: Quadratic (Driven by frequency increase only)\n")
            f.write("\n")
        
        f.write("## 3. TOPSIS Ranking Results\n")
        f.write(f"{'Rank':<6} | {'Scenario':<10} | {'Score':<10} | {'Norm Cost':<12} | {'Norm Time':<12}\n")
        f.write("-" * 60 + "\n")
        
        sorted_scenarios = sorted(topsis_res.items(), key=lambda x: x[1]['Score'], reverse=True)
        
        for rank, (scenario, res) in enumerate(sorted_scenarios, 1):
            f.write(f"{rank:<6} | {scenario:<10} | {res['Score']:.4f}     | {res['Norm_Cost']:.4f}       | {res['Norm_Time']:.4f}\n")
            
        f.write("\n## 4. Conclusion\n")
        best_scenario = sorted_scenarios[0][0]
        f.write(f"The optimal strategy is **Scenario {best_scenario}**.\n")
        if best_scenario == "A":
            f.write("Despite Scenario C being slightly faster, the massive cost reduction of the Space Elevator (A) makes it the superior choice.\n")
        elif best_scenario == "C":
            f.write("The synchronized Hybrid model offers the best balance of speed and feasibility.\n")
            
    print(f"--- Analysis Complete. Results saved to {result_file} ---")

if __name__ == "__main__":
    main()