"""
Task 4 Optimization Module
===========================
Purpose: Solve multi-objective optimization for Scenarios A, B, and C
         Returns optimal decision variables and objective values
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from parameters import *

# ============================================================================
# USER ADJUSTMENT GUIDE
# ============================================================================
# [Line 25-40]   Scenario A optimization (continuous variables)
# [Line 100-130] Scenario B optimization (0-1 mixed integer)
# [Line 200-240] Scenario C optimization (hybrid system)
# [Line 300-320] Objective function calculation
# ============================================================================


def calculate_elevator_effective_capacity(m_E):
    """
    Calculate effective elevator capacity considering failures

    Args:
        m_E: Maintenance frequency (times/harbour/year)

    Returns:
        Q_E_eff: Effective capacity (tonnes/year)
    """
    # Sway reduction factor
    sway_factor = 1 - p_E1 * beta_E1

    # Downtime reduction factor
    downtime_factor = 1 - p_E2 * (t_E2 / 365)

    # Combined effective capacity
    Q_E_eff = Q_E_total * sway_factor * downtime_factor

    return Q_E_eff


def calculate_rocket_effective_capacity(y, x):
    """
    Calculate effective rocket capacity considering failures

    Args:
        y: Launch site selection (10-dim binary array)
        x: Launch frequencies (10-dim array, launches/year)

    Returns:
        Q_R_eff: Effective rocket capacity (tonnes/year)
    """
    # Average safety coefficient for selected sites
    s_R_safe_avg = np.sum(y * s_R_safe) / np.sum(y) if np.sum(y) > 0 else 1.0

    # Effective capacity considering failure and downtime
    Q_R_eff = np.sum(y * x * Q_R * (1 - p_R1 * s_R_safe) * (1 - p_R2 * t_R2 / 365))

    return Q_R_eff


def scenario_a_objectives(decision_vars):
    """
    Calculate four objectives for Scenario A (pure elevator)

    Args:
        decision_vars: [s_E_elec, m_E] (2 variables)

    Returns:
        [F1, F2, F3, F4]: Normalized objective values
    """
    s_E_elec, m_E = decision_vars

    # Ensure integer maintenance frequency
    m_E = int(round(m_E))
    if m_E < m_E_min:
        m_E = m_E_min

    # Calculate effective capacity
    Q_E_eff = calculate_elevator_effective_capacity(m_E)

    # Objective 1: Total cost
    T_a = M_total / Q_E_eff * 365  # Transport time in days
    T_a_years = T_a / 365

    C_a = (3 * C_E_f * T_a_years +
           M_total * C_E_u +
           3 * p_E2 * np.exp(-0.2 * m_E) * T_a_years * C_E_repair)

    # Objective 2: Transport time (already calculated)
    T_a = T_a * (1 + p_E2 * np.exp(-0.2 * m_E) * t_E2 / 365)

    # Objective 3: Reliability
    R_a = 1 - (p_E2 * np.exp(-0.2 * m_E) / 3) * (t_E2 / 365)

    # Objective 4: Environmental impact
    EI_a = (w_PM25 * E_PM25_a / E_PM25_max +
            w_CO2 * E_CO2_a * (1 - s_E_elec) / E_CO2_max +
            w_reso * E_reso_a / E_reso_max +
            w_eco * E_eco_a / E_eco_max)

    # Normalize objectives
    F1 = C_a / C_max
    F2 = T_a / T_max
    F3 = 1 - R_a / R_max  # Convert to "minimize"
    F4 = EI_a / EI_max

    return np.array([F1, F2, F3, F4])


def optimize_scenario_a():
    """
    Optimize Scenario A: Pure space elevator system

    Returns:
        result: dict with optimal solution and objectives
    """
    # Define objective function (weighted sum)
    def objective(x):
        F = scenario_a_objectives(x)
        return np.dot(w_main, F)

    # Bounds: [s_E_elec, m_E]
    bounds = [(s_E_elec_min, s_E_elec_max),
              (m_E_min, m_E_max)]

    # Initial guess
    x0 = np.array([0.9, 2.0])

    # Optimize using SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   options={'maxiter': 1000, 'ftol': 1e-9})

    # Extract optimal values
    s_E_elec_opt, m_E_opt = res.x
    m_E_opt = int(round(m_E_opt))

    # Calculate objectives at optimum
    F_opt = scenario_a_objectives([s_E_elec_opt, m_E_opt])

    # Calculate raw objectives for reporting
    Q_E_eff = calculate_elevator_effective_capacity(m_E_opt)
    T_a = M_total / Q_E_eff * 365 * (1 + p_E2 * np.exp(-0.2 * m_E_opt) * t_E2 / 365)
    T_a_years = T_a / 365
    C_a = (3 * C_E_f * T_a_years + M_total * C_E_u +
           3 * p_E2 * np.exp(-0.2 * m_E_opt) * T_a_years * C_E_repair)
    R_a = 1 - (p_E2 * np.exp(-0.2 * m_E_opt) / 3) * (t_E2 / 365)
    EI_a = (w_PM25 * E_PM25_a / E_PM25_max +
            w_CO2 * E_CO2_a * (1 - s_E_elec_opt) / E_CO2_max +
            w_reso * E_reso_a / E_reso_max +
            w_eco * E_eco_a / E_eco_max)

    result = {
        'scenario': 'A',
        'decision_vars': {'s_E_elec': s_E_elec_opt, 'm_E': m_E_opt},
        'objectives_normalized': F_opt,
        'objectives_raw': {
            'cost': C_a,
            'time_days': T_a,
            'time_years': T_a_years,
            'reliability': R_a,
            'environmental_impact': EI_a
        },
        'comprehensive_score': np.dot(w_main, F_opt),
        'success': res.success
    }

    return result


def scenario_b_objectives(y, x, f_R_fuel=f_R_fuel_traditional, f_R_env=f_R_env_traditional):
    """
    Calculate four objectives for Scenario B (pure rocket)

    Args:
        y: Launch site selection (10-dim binary array)
        x: Launch frequencies (10-dim array)
        f_R_fuel: Fuel cost modifier
        f_R_env: Environmental modifier

    Returns:
        [F1, F2, F3, F4]: Normalized objective values
    """
    # Average safety coefficient
    s_R_safe_avg = np.sum(y * s_R_safe) / np.sum(y) if np.sum(y) > 0 else 1.0

    # Total expected launches (considering 3% failure rate with geometric distribution)
    N_R_total = M_total / (Q_R * (1 - p_R1)**2)

    # Calculate transport time
    Q_R_eff = calculate_rocket_effective_capacity(y, x)
    T_b = M_total / Q_R_eff * 365 if Q_R_eff > 0 else T_max
    T_b = T_b * (1 + p_R1 * s_R_safe_avg * t_R2 / 365)
    T_b_years = T_b / 365

    # Objective 1: Total cost
    C_b = (np.sum(y * C_R_l) * T_b_years +
           N_R_total * C_R_i * f_R_fuel +
           np.sum(y * p_R1 * s_R_safe * T_b_years * C_R_repair))

    # Objective 2: Transport time (already calculated)

    # Objective 3: Reliability
    R_b = 1 - p_R1 * s_R_safe_avg * np.exp(-0.15 * np.sum(y)) * t_R2 / 365

    # Objective 4: Environmental impact
    EI_b = (w_PM25 * E_PM25_b / E_PM25_max +
            w_CO2 * E_CO2_b * f_R_env / E_CO2_max +
            w_reso * E_reso_b * f_R_env / E_reso_max +
            w_eco * E_eco_b * s_R_safe_avg / E_eco_max)

    # Normalize objectives
    F1 = C_b / C_max
    F2 = T_b / T_max
    F3 = 1 - R_b / R_max
    F4 = EI_b / EI_max

    return np.array([F1, F2, F3, F4])


def optimize_scenario_b_single_combination(y, f_R_fuel=f_R_fuel_traditional, f_R_env=f_R_env_traditional):
    """
    Optimize launch frequencies for a given launch site combination

    Args:
        y: Launch site selection (10-dim binary array)
        f_R_fuel: Fuel cost modifier
        f_R_env: Environmental modifier

    Returns:
        (best_F, best_x): Best objective value and launch frequencies
    """
    # Simple heuristic: distribute launches proportionally to capacity
    # to meet minimum transport requirement

    # Estimate required annual capacity
    required_capacity = M_total * 365 / T_max  # Conservative estimate

    # Distribute launches proportionally to max capacity
    selected_indices = np.where(y == 1)[0]
    if len(selected_indices) == 0:
        return 1e10, np.zeros(10)  # Invalid solution

    # Equal distribution strategy
    x = np.zeros(10)
    for idx in selected_indices:
        x[idx] = min(x_max[idx], 50)  # Start with 50 launches/year

    # Calculate objectives
    F = scenario_b_objectives(y, x, f_R_fuel, f_R_env)

    return np.dot(w_main, F), x


def optimize_scenario_b(max_combinations=100):
    """
    Optimize Scenario B: Pure rocket system
    Uses enumeration for small-scale or sampling for large-scale

    Args:
        max_combinations: Maximum number of combinations to evaluate

    Returns:
        result: dict with optimal solution and objectives
    """
    print("  Optimizing Scenario B (pure rocket)...")

    best_score = 1e10
    best_y = None
    best_x = None
    best_F = None

    # Strategy: Evaluate promising combinations
    # Start with low-risk sites (indices 0-4)

    # Combination 1: All low-risk sites
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    score, x = optimize_scenario_b_single_combination(y)
    if score < best_score:
        best_score = score
        best_y = y.copy()
        best_x = x.copy()
        best_F = scenario_b_objectives(y, x)

    # Combination 2: All sites
    y = np.ones(10)
    score, x = optimize_scenario_b_single_combination(y)
    if score < best_score:
        best_score = score
        best_y = y.copy()
        best_x = x.copy()
        best_F = scenario_b_objectives(y, x)

    # Combination 3: Top 3 low-risk sites
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    score, x = optimize_scenario_b_single_combination(y)
    if score < best_score:
        best_score = score
        best_y = y.copy()
        best_x = x.copy()
        best_F = scenario_b_objectives(y, x)

    # Calculate raw objectives
    s_R_safe_avg = np.sum(best_y * s_R_safe) / np.sum(best_y)
    Q_R_eff = calculate_rocket_effective_capacity(best_y, best_x)
    T_b = M_total / Q_R_eff * 365 * (1 + p_R1 * s_R_safe_avg * t_R2 / 365)
    T_b_years = T_b / 365
    N_R_total = M_total / (Q_R * (1 - p_R1)**2)
    C_b = (np.sum(best_y * C_R_l) * T_b_years + N_R_total * C_R_i +
           np.sum(best_y * p_R1 * s_R_safe * T_b_years * C_R_repair))
    R_b = 1 - p_R1 * s_R_safe_avg * np.exp(-0.15 * np.sum(best_y)) * t_R2 / 365
    EI_b = best_F[3] * EI_max

    result = {
        'scenario': 'B',
        'decision_vars': {
            'y': best_y,
            'x': best_x,
            'num_sites': int(np.sum(best_y)),
            'selected_sites': [launch_site_names[i] for i in range(10) if best_y[i] == 1]
        },
        'objectives_normalized': best_F,
        'objectives_raw': {
            'cost': C_b,
            'time_days': T_b,
            'time_years': T_b_years,
            'reliability': R_b,
            'environmental_impact': EI_b
        },
        'comprehensive_score': best_score,
        'success': True
    }

    return result


def optimize_scenario_c():
    """
    Optimize Scenario C: Hybrid system (elevator + rocket)

    Returns:
        result: dict with optimal solution and objectives
    """
    print("  Optimizing Scenario C (hybrid system)...")

    # Simplified approach: Use Scenario A elevator + Scenario B rocket
    # Optimize alpha (elevator mass fraction)

    # First get optimal configurations from A and B
    result_a = optimize_scenario_a()
    result_b = optimize_scenario_b()

    # Extract optimal parameters
    s_E_elec_opt = result_a['decision_vars']['s_E_elec']
    m_E_opt = result_a['decision_vars']['m_E']
    y_opt = result_b['decision_vars']['y']
    x_opt = result_b['decision_vars']['x']

    # Calculate effective capacities
    Q_E_eff = calculate_elevator_effective_capacity(m_E_opt)
    Q_R_eff = calculate_rocket_effective_capacity(y_opt, x_opt)

    # Optimal alpha for synchronized completion
    alpha_opt = Q_E_eff / (Q_E_eff + Q_R_eff)
    alpha_opt = np.clip(alpha_opt, 0.0, 1.0)

    # Calculate hybrid objectives
    T_c = max(alpha_opt * M_total / Q_E_eff * 365,
              (1 - alpha_opt) * M_total / Q_R_eff * 365)
    T_c_years = T_c / 365

    # Cost
    s_R_safe_avg = np.sum(y_opt * s_R_safe) / np.sum(y_opt)
    C_c = ((3 * C_E_f + np.sum(y_opt * C_R_l)) * T_c_years +
           alpha_opt * M_total * C_E_u +
           (1 - alpha_opt) * M_total / (Q_R * (1 - p_R1)**2) * C_R_i +
           3 * p_E2 * np.exp(-0.2 * m_E_opt) * T_c_years * C_E_repair +
           np.sum(y_opt * p_R1 * s_R_safe * T_c_years * C_R_repair))

    # Reliability
    R_a_component = 1 - (p_E2 * np.exp(-0.2 * m_E_opt) / 3) * (t_E2 / 365)
    R_b_component = 1 - p_R1 * s_R_safe_avg * np.exp(-0.15 * np.sum(y_opt)) * t_R2 / 365
    R_c = alpha_opt * R_a_component + (1 - alpha_opt) * R_b_component

    # Environmental impact
    EI_a = (w_PM25 * E_PM25_a / E_PM25_max +
            w_CO2 * E_CO2_a * (1 - s_E_elec_opt) / E_CO2_max +
            w_reso * E_reso_a / E_reso_max +
            w_eco * E_eco_a / E_eco_max)
    EI_b = (w_PM25 * E_PM25_b / E_PM25_max +
            w_CO2 * E_CO2_b / E_CO2_max +
            w_reso * E_reso_b / E_reso_max +
            w_eco * E_eco_b * s_R_safe_avg / E_eco_max)
    EI_c = alpha_opt * EI_a + (1 - alpha_opt) * EI_b

    # Normalize objectives
    F1 = C_c / C_max
    F2 = T_c / T_max
    F3 = 1 - R_c / R_max
    F4 = EI_c / EI_max
    F_c = np.array([F1, F2, F3, F4])

    result = {
        'scenario': 'C',
        'decision_vars': {
            'alpha': alpha_opt,
            's_E_elec': s_E_elec_opt,
            'm_E': m_E_opt,
            'y': y_opt,
            'x': x_opt,
            'elevator_mass_fraction': alpha_opt,
            'rocket_mass_fraction': 1 - alpha_opt
        },
        'objectives_normalized': F_c,
        'objectives_raw': {
            'cost': C_c,
            'time_days': T_c,
            'time_years': T_c_years,
            'reliability': R_c,
            'environmental_impact': EI_c
        },
        'comprehensive_score': np.dot(w_main, F_c),
        'success': True
    }

    return result


if __name__ == "__main__":
    # Test optimization functions
    print("Testing optimization module...")

    print("\nScenario A:")
    result_a = optimize_scenario_a()
    print(f"  Optimal s_E_elec: {result_a['decision_vars']['s_E_elec']:.4f}")
    print(f"  Optimal m_E: {result_a['decision_vars']['m_E']}")
    print(f"  Cost: ${result_a['objectives_raw']['cost']:.2e}")
    print(f"  Time: {result_a['objectives_raw']['time_years']:.2f} years")
    print(f"  Reliability: {result_a['objectives_raw']['reliability']:.4f}")
    print(f"  Environmental Impact: {result_a['objectives_raw']['environmental_impact']:.4f}")
    print(f"  Comprehensive Score: {result_a['comprehensive_score']:.6f}")

    print("\nScenario B:")
    result_b = optimize_scenario_b()
    print(f"  Selected sites: {result_b['decision_vars']['num_sites']}")
    print(f"  Cost: ${result_b['objectives_raw']['cost']:.2e}")
    print(f"  Time: {result_b['objectives_raw']['time_years']:.2f} years")
    print(f"  Comprehensive Score: {result_b['comprehensive_score']:.6f}")

    print("\nScenario C:")
    result_c = optimize_scenario_c()
    print(f"  Alpha: {result_c['decision_vars']['alpha']:.4f}")
    print(f"  Cost: ${result_c['objectives_raw']['cost']:.2e}")
    print(f"  Time: {result_c['objectives_raw']['time_years']:.2f} years")
    print(f"  Comprehensive Score: {result_c['comprehensive_score']:.6f}")
