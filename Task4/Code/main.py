"""
Task 4: Multi-Objective Optimization for Moon Colony Transportation
====================================================================
Purpose: Orchestrate optimization workflow and generate analysis results
Output Files:
    - Result.txt: Summary of optimization results and key findings
    - fig1_ahp_weights.pdf: AHP weight comparison visualization
    - fig2_multi_objective_radar.pdf: Multi-objective radar chart
    - fig3_environmental_heatmap.pdf: Environmental impact breakdown
    - fig4_pareto_cost_time.pdf: Pareto frontier analysis
    - fig5_comprehensive_scores.pdf: Comprehensive score comparison

Author: Task 4 Analysis Team
Date: 2026-02-01
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parameters import *
from optimization import optimize_scenario_a, optimize_scenario_b, optimize_scenario_c
from visualization import generate_all_figures

# ============================================================================
# USER ADJUSTMENT GUIDE
# ============================================================================
# [Line 40-60]   Main execution workflow
# [Line 80-120]  Result.txt formatting
# [Line 150-200] Results analysis and interpretation
# ============================================================================


def format_large_number(num):
    """Format large numbers with scientific notation"""
    if num >= 1e12:
        return f"{num/1e12:.2f} trillion"
    elif num >= 1e9:
        return f"{num/1e9:.2f} billion"
    elif num >= 1e6:
        return f"{num/1e6:.2f} million"
    else:
        return f"{num:.2f}"


def write_results_summary(results, output_path):
    """
    Write comprehensive results summary to Result.txt

    Args:
        results: Dict containing optimization results for scenarios A, B, C
        output_path: Path object for output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("Task 4: Multi-Objective Optimization Results\n")
        f.write("Moon Colony Transportation System Analysis\n")
        f.write("="*80 + "\n\n")

        # AHP Weights Summary
        f.write("AHP WEIGHT CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write("Main Objective Weights:\n")
        f.write(f"  Cost (w1):          {w_cost:.4f} ({w_cost*100:.2f}%)\n")
        f.write(f"  Time (w2):          {w_time:.4f} ({w_time*100:.2f}%)\n")
        f.write(f"  Reliability (w3):   {w_reliability:.4f} ({w_reliability*100:.2f}%)\n")
        f.write(f"  Environment (w4):   {w_environment:.4f} ({w_environment*100:.2f}%)\n\n")

        f.write("Environmental Sub-Weights:\n")
        f.write(f"  PM2.5 (w_PM2.5):    {w_PM25:.4f} ({w_PM25*100:.2f}%)\n")
        f.write(f"  CO2 (w_CO2):        {w_CO2:.4f} ({w_CO2*100:.2f}%)\n")
        f.write(f"  Ecology (w_eco):    {w_eco:.4f} ({w_eco*100:.2f}%)\n")
        f.write(f"  Resource (w_reso):  {w_reso:.4f} ({w_reso*100:.2f}%)\n\n")

        # Scenario Results
        for scenario_key in ['A', 'B', 'C']:
            result = results[scenario_key]

            f.write("="*80 + "\n")
            f.write(f"SCENARIO {scenario_key}: ")

            if scenario_key == 'A':
                f.write("PURE SPACE ELEVATOR SYSTEM\n")
            elif scenario_key == 'B':
                f.write("PURE ROCKET SYSTEM\n")
            else:
                f.write("HYBRID SYSTEM (ELEVATOR + ROCKET)\n")

            f.write("="*80 + "\n\n")

            # Decision Variables
            f.write("Optimal Decision Variables:\n")
            f.write("-"*80 + "\n")

            if scenario_key == 'A':
                s_E_elec = result['decision_vars']['s_E_elec']
                m_E = result['decision_vars']['m_E']
                f.write(f"  Renewable energy ratio (s_E_elec): {s_E_elec:.4f} ({s_E_elec*100:.2f}%)\n")
                f.write(f"  Maintenance frequency (m_E):       {m_E} times/harbour/year\n")

            elif scenario_key == 'B':
                num_sites = result['decision_vars']['num_sites']
                selected_sites = result['decision_vars']['selected_sites']
                f.write(f"  Number of launch sites selected:   {num_sites}\n")
                f.write(f"  Selected sites:\n")
                for site in selected_sites:
                    f.write(f"    - {site}\n")

            else:  # Scenario C
                alpha = result['decision_vars']['alpha']
                s_E_elec = result['decision_vars']['s_E_elec']
                m_E = result['decision_vars']['m_E']
                num_sites = int(np.sum(result['decision_vars']['y']))

                f.write(f"  Elevator mass fraction (alpha):     {alpha:.4f} ({alpha*100:.2f}%)\n")
                f.write(f"  Rocket mass fraction (1-alpha):     {1-alpha:.4f} ({(1-alpha)*100:.2f}%)\n")
                f.write(f"  Renewable energy ratio (s_E_elec): {s_E_elec:.4f} ({s_E_elec*100:.2f}%)\n")
                f.write(f"  Maintenance frequency (m_E):       {m_E} times/harbour/year\n")
                f.write(f"  Number of launch sites:            {num_sites}\n")

            f.write("\n")

            # Objectives (Raw Values)
            f.write("Objective Values:\n")
            f.write("-"*80 + "\n")
            raw = result['objectives_raw']

            f.write(f"  Total Cost:           ${format_large_number(raw['cost'])} USD\n")
            f.write(f"                        (${raw['cost']:.2e} USD)\n")
            f.write(f"  Transport Time:       {raw['time_years']:.2f} years\n")
            f.write(f"                        ({raw['time_days']:.0f} days)\n")
            f.write(f"  System Reliability:   {raw['reliability']:.6f}\n")
            f.write(f"                        ({raw['reliability']*100:.4f}%)\n")
            f.write(f"  Environmental Impact: {raw['environmental_impact']:.6f}\n")
            f.write(f"                        (normalized index, 0-1 scale)\n\n")

            # Normalized Objectives
            f.write("Normalized Objectives (0-1 scale):\n")
            f.write("-"*80 + "\n")
            norm = result['objectives_normalized']
            f.write(f"  F1 (Cost):          {norm[0]:.6f}\n")
            f.write(f"  F2 (Time):          {norm[1]:.6f}\n")
            f.write(f"  F3 (Unreliability): {norm[2]:.6f}\n")
            f.write(f"  F4 (Environment):   {norm[3]:.6f}\n\n")

            # Comprehensive Score
            f.write("Comprehensive Performance:\n")
            f.write("-"*80 + "\n")
            score = result['comprehensive_score']
            f.write(f"  Weighted Score: {score:.6f}\n")
            f.write(f"  Formula: F = {w_cost:.4f}*F1 + {w_time:.4f}*F2 + {w_reliability:.4f}*F3 + {w_environment:.4f}*F4\n\n")

        # Comparative Analysis
        f.write("="*80 + "\n")
        f.write("COMPARATIVE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        scores = [results['A']['comprehensive_score'],
                  results['B']['comprehensive_score'],
                  results['C']['comprehensive_score']]
        scenario_names = ['A (Elevator)', 'B (Rocket)', 'C (Hybrid)']

        # Ranking
        sorted_indices = np.argsort(scores)
        f.write("Ranking by Comprehensive Score (lower is better):\n")
        f.write("-"*80 + "\n")
        for rank, idx in enumerate(sorted_indices, 1):
            f.write(f"  {rank}. Scenario {scenario_names[idx]}: {scores[idx]:.6f}\n")

        f.write("\n")

        # Best scenario
        best_scenario_idx = sorted_indices[0]
        best_scenario_key = ['A', 'B', 'C'][best_scenario_idx]
        best_result = results[best_scenario_key]

        f.write("RECOMMENDED SOLUTION:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Optimal Scenario: {scenario_names[best_scenario_idx]}\n")
        f.write(f"  Comprehensive Score: {scores[best_scenario_idx]:.6f}\n\n")

        f.write("Key Advantages:\n")

        if best_scenario_key == 'A':
            f.write("  - Lowest environmental impact (near-zero emissions)\n")
            f.write("  - Highest long-term cost efficiency\n")
            f.write("  - Sustainable operation with renewable energy\n")
            f.write("  - No atmospheric pollution\n")

        elif best_scenario_key == 'B':
            f.write("  - Shortest transport time\n")
            f.write("  - Proven technology with mature infrastructure\n")
            f.write("  - Flexible launch schedule\n")
            f.write("  - Redundancy through multiple launch sites\n")

        else:  # Scenario C
            f.write("  - Balanced performance across all objectives\n")
            f.write("  - Risk diversification through dual systems\n")
            f.write("  - Moderate environmental impact\n")
            f.write("  - Adaptive to demand fluctuations\n")

        f.write("\n")

        # Environmental Comparison
        f.write("Environmental Impact Comparison:\n")
        f.write("-"*80 + "\n")
        ei_a = results['A']['objectives_raw']['environmental_impact']
        ei_b = results['B']['objectives_raw']['environmental_impact']
        ei_c = results['C']['objectives_raw']['environmental_impact']

        f.write(f"  Scenario A: {ei_a:.6f} (baseline)\n")
        f.write(f"  Scenario B: {ei_b:.6f} ({ei_b/ei_a:.2f}x higher than A)\n")
        f.write(f"  Scenario C: {ei_c:.6f} ({ei_c/ei_a:.2f}x higher than A)\n\n")

        f.write(f"  Environmental Impact Reduction (B->C): {(ei_b-ei_c)/ei_b*100:.2f}%\n")
        f.write(f"  Environmental Impact Reduction (B->A): {(ei_b-ei_a)/ei_b*100:.2f}%\n\n")

        # Cost-Time Tradeoff
        f.write("Cost-Time Tradeoff Analysis:\n")
        f.write("-"*80 + "\n")
        cost_a = results['A']['objectives_raw']['cost']
        cost_b = results['B']['objectives_raw']['cost']
        cost_c = results['C']['objectives_raw']['cost']
        time_a = results['A']['objectives_raw']['time_years']
        time_b = results['B']['objectives_raw']['time_years']
        time_c = results['C']['objectives_raw']['time_years']

        f.write(f"  Scenario A: ${format_large_number(cost_a)} USD, {time_a:.2f} years\n")
        f.write(f"  Scenario B: ${format_large_number(cost_b)} USD, {time_b:.2f} years\n")
        f.write(f"  Scenario C: ${format_large_number(cost_c)} USD, {time_c:.2f} years\n\n")

        f.write(f"  Cost savings (A vs B): ${format_large_number(cost_b-cost_a)} USD ({(cost_b-cost_a)/cost_b*100:.2f}%)\n")
        f.write(f"  Time savings (B vs A): {time_a-time_b:.2f} years ({(time_a-time_b)/time_a*100:.2f}%)\n\n")

        # Footer
        f.write("="*80 + "\n")
        f.write("VISUALIZATION OUTPUTS\n")
        f.write("="*80 + "\n\n")
        f.write("Generated figures:\n")
        f.write("  1. fig1_ahp_weights.pdf - AHP weight comparison\n")
        f.write("  2. fig2_multi_objective_radar.pdf - Multi-objective performance\n")
        f.write("  3. fig3_environmental_heatmap.pdf - Environmental impact breakdown\n")
        f.write("  4. fig4_pareto_cost_time.pdf - Cost-time Pareto frontier\n")
        f.write("  5. fig5_comprehensive_scores.pdf - Comprehensive score comparison\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


def main():
    """
    Main execution workflow
    """
    print("="*80)
    print("Task 4: Multi-Objective Optimization for Moon Colony Transportation")
    print("="*80)
    print()

    # Step 1: Optimize Scenario A (Pure Elevator)
    print("Step 1: Optimizing Scenario A (Pure Space Elevator System)...")
    result_a = optimize_scenario_a()
    print(f"  Completed! Comprehensive Score: {result_a['comprehensive_score']:.6f}")
    print()

    # Step 2: Optimize Scenario B (Pure Rocket)
    print("Step 2: Optimizing Scenario B (Pure Rocket System)...")
    result_b = optimize_scenario_b()
    print(f"  Completed! Comprehensive Score: {result_b['comprehensive_score']:.6f}")
    print()

    # Step 3: Optimize Scenario C (Hybrid)
    print("Step 3: Optimizing Scenario C (Hybrid System)...")
    result_c = optimize_scenario_c()
    print(f"  Completed! Comprehensive Score: {result_c['comprehensive_score']:.6f}")
    print()

    # Compile results
    results = {
        'A': result_a,
        'B': result_b,
        'C': result_c
    }

    # Step 4: Generate visualizations
    print("Step 4: Generating publication-quality visualizations...")
    generate_all_figures(results)
    print()

    # Step 5: Write results summary
    print("Step 5: Writing results summary...")
    output_dir = Path(__file__).parent.parent.resolve()
    result_file = output_dir / "Result.txt"
    write_results_summary(results, result_file)
    print(f"  Saved: {result_file}")
    print()

    # Summary
    print("="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print()
    print("Optimal Scenario:")
    scores = [result_a['comprehensive_score'], result_b['comprehensive_score'], result_c['comprehensive_score']]
    scenario_names = ['A (Pure Elevator)', 'B (Pure Rocket)', 'C (Hybrid)']
    best_idx = np.argmin(scores)
    print(f"  {scenario_names[best_idx]}")
    print(f"  Comprehensive Score: {scores[best_idx]:.6f}")
    print()
    print("Output files saved to:")
    print(f"  {output_dir}")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
