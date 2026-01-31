"""
Task 1 Calculation Module
=========================
Purpose: Calculate cost (C) and time (T) for three transportation scenarios (A, B, C)
         with 0-1 programming optimization for rocket base selection.

Key Features:
- Scenario A: Space Elevator only
- Scenario B: Rockets only (optimized base selection via enumeration)
- Scenario C: Hybrid (synchronized completion)
- Piecewise growth model with 200 launches/year capacity constraint
- AHP weighting and TOPSIS evaluation

Author: Generated for MCM 2026 Problem B
"""

import numpy as np
import pandas as pd
from itertools import combinations

# ==========================================
# Global Parameters
# ==========================================

# Launch capacity constraint
LAUNCH_CAP_PER_BASE = 200  # Maximum launches per year per base

# Baseline offset for 2050
ADDITION = 20  # Each base adds 20 launches to historical average by 2050

# Rocket payload (fixed)
Q_R_PER_LAUNCH = 150  # tons per launch

# Normalization parameters for 0-1 programming
C_MAX = 1.5e14  # 150 trillion USD (maximum expected cost)
T_MAX = 350     # 350 years (maximum expected timeline)

# ==========================================
# Cost Parameters (USD/ton)
# ==========================================

# Elevator route: Earth Port -> Apex Anchor -> Moon
# - Earth Port -> Apex: negligible (electricity cost ignored)
# - Apex -> Moon transfer rocket: 92.7 USD/kg
C_E_TRANSFER_PER_TON = 92.7 * 1000  # 92,700 USD/ton

# Rocket route: Direct launch from Earth to Moon
# - Falcon Heavy cost: 1500 USD/kg
C_R_DIRECT_PER_TON = 1500 * 1000  # 1,500,000 USD/ton


def get_rocket_bases_data():
    """
    Returns rocket base data as defined in Solution1.md
    Columns: Location, x_max (initial launches/year in 2050), fixed_cost (USD/year)
    """
    data = [
        {"Location": "Florida", "x_max": 73 + ADDITION, "fixed_cost": 6e8},
        {"Location": "California", "x_max": 34 + ADDITION, "fixed_cost": 3e8},
        {"Location": "Taiyuan (China)", "x_max": 11 + ADDITION, "fixed_cost": 0.02e8},
        {"Location": "Mahia (NZ)", "x_max": 11 + ADDITION, "fixed_cost": 0.02e8},
        {"Location": "Kazakhstan", "x_max": 9 + ADDITION, "fixed_cost": 0.68e8},
        {"Location": "French Guiana", "x_max": 5 + ADDITION, "fixed_cost": 4e8},
        {"Location": "India", "x_max": 5 + ADDITION, "fixed_cost": 0.2e8},
        {"Location": "Virginia", "x_max": 2 + ADDITION, "fixed_cost": 0.04e8},
        {"Location": "Texas", "x_max": 2 + ADDITION, "fixed_cost": 0.6e8},
        {"Location": "Alaska", "x_max": 1 + ADDITION, "fixed_cost": 0.02e8},
    ]
    return pd.DataFrame(data)


def calculate_piecewise_time(bases_subset, target_mass):
    """
    Calculates completion time for a given subset of rocket bases using piecewise growth model.

    Model:
    - Phase 1: All bases in linear growth (quadratic cumulative mass)
    - Phase 2+: Bases reaching 200 launches/year cap sequentially

    Args:
        bases_subset: DataFrame with columns ['x_max', 'fixed_cost']
        target_mass: Target cumulative mass (tons)

    Returns:
        T: Completion time (years)
    """
    n_bases = len(bases_subset)
    if n_bases == 0:
        return float('inf')

    # Initial capacity and growth rate
    x_maxs = bases_subset['x_max'].values
    Q_R0 = np.sum(x_maxs) * Q_R_PER_LAUNCH  # Initial annual capacity
    g_R = n_bases * Q_R_PER_LAUNCH          # Growth rate (all bases contribute)

    # Calculate when each base reaches capacity
    t_caps = LAUNCH_CAP_PER_BASE - x_maxs  # Time to reach cap for each base
    t_cap_min = np.min(t_caps)

    # Check if task completes in Phase 1 (pure quadratic growth)
    # S(t) = Q_R0 * t + (g_R/2) * t^2
    M1 = Q_R0 * t_cap_min + (g_R / 2.0) * t_cap_min**2

    if M1 >= target_mass:
        # Solve quadratic: (g_R/2)*T^2 + Q_R0*T - target_mass = 0
        a = g_R / 2.0
        b = Q_R0
        c = -target_mass
        delta = b**2 - 4*a*c
        if delta < 0:
            return float('inf')
        T = (-b + np.sqrt(delta)) / (2*a)
        return T

    # Phase 2+: Piecewise calculation
    # Sort bases by time to reach capacity
    sorted_indices = np.argsort(t_caps)
    sorted_x_maxs = x_maxs[sorted_indices]
    sorted_t_caps = t_caps[sorted_indices]

    cumulative_mass = 0.0
    t_current = 0.0
    active_bases = n_bases
    Q_current = Q_R0

    for i in range(n_bases):
        t_next_cap = sorted_t_caps[i]
        dt = t_next_cap - t_current

        # Growth rate for current active bases
        g_current = active_bases * Q_R_PER_LAUNCH

        # Mass accumulated in this interval
        dM = Q_current * dt + (g_current / 2.0) * dt**2

        if cumulative_mass + dM >= target_mass:
            # Task completes in this interval
            # Solve: Q_current * dt + (g_current/2) * dt^2 = remaining_mass
            remaining = target_mass - cumulative_mass
            a = g_current / 2.0
            b = Q_current
            c = -remaining
            delta = b**2 - 4*a*c
            if delta < 0:
                return t_current + remaining / Q_current  # Fallback to linear
            dt_final = (-b + np.sqrt(delta)) / (2*a)
            return t_current + dt_final

        # Update for next phase
        cumulative_mass += dM
        t_current = t_next_cap

        # Base i reaches capacity, update Q_current
        # New capacity = old capacity + growth from previous phase - contribution from capped base
        Q_current = Q_current + g_current * dt - sorted_x_maxs[i] * Q_R_PER_LAUNCH
        # After capping, the base contributes constant 200 launches/year
        Q_current += LAUNCH_CAP_PER_BASE * Q_R_PER_LAUNCH
        active_bases -= 1

    # All bases at capacity - linear growth from here
    # Final capacity (all bases at 200 launches/year)
    Q_final = n_bases * LAUNCH_CAP_PER_BASE * Q_R_PER_LAUNCH
    remaining = target_mass - cumulative_mass
    if Q_final > 0:
        T_final = t_current + remaining / Q_final
        return T_final
    else:
        return float('inf')


def optimize_rocket_bases(df_bases, w_c, w_t, target_mass):
    """
    0-1 programming via enumeration: find optimal rocket base combination.

    Objective: Minimize normalized weighted objective
        Obj = w_c * (C / C_MAX) + w_t * (T / T_MAX)

    Args:
        df_bases: DataFrame with all 10 bases
        w_c, w_t: AHP weights for cost and time
        target_mass: Target mass (100 million tons)

    Returns:
        best_combination: List of base indices
        best_cost: Total cost (USD)
        best_time: Completion time (years)
        best_obj: Objective value
    """
    n_total = len(df_bases)
    best_obj = float('inf')
    best_combination = None
    best_cost = None
    best_time = None

    # Enumerate all non-empty subsets (2^10 - 1 = 1023 combinations)
    for r in range(1, n_total + 1):
        for combo in combinations(range(n_total), r):
            subset = df_bases.iloc[list(combo)]

            # Calculate time
            T_B = calculate_piecewise_time(subset, target_mass)

            if T_B == float('inf'):
                continue

            # Calculate cost
            fixed_cost_total = subset['fixed_cost'].sum()
            C_B_fixed = fixed_cost_total * T_B
            C_B_variable = target_mass * C_R_DIRECT_PER_TON
            C_B = C_B_fixed + C_B_variable

            # Normalized objective
            C_norm = C_B / C_MAX
            T_norm = T_B / T_MAX
            obj = w_c * C_norm + w_t * T_norm

            if obj < best_obj:
                best_obj = obj
                best_combination = list(combo)
                best_cost = C_B
                best_time = T_B

    return best_combination, best_cost, best_time, best_obj


def calculate_metrics():
    """
    Calculates Cost (C) and Time (T) for Scenarios A, B, and C.

    Returns:
        Dictionary with metrics for scenarios A, B, C and metadata
    """
    M = 1e8  # 100 million metric tons

    # ==========================================
    # Scenario A: Space Elevator Alone
    # ==========================================
    N_E = 3  # 3 Galactic Harbours
    Q_E_yearly = 1.79e5  # tons/year per harbour
    C_E_fixed_yearly = 3e9  # 3 billion USD/year per harbour

    Q_E_total_yearly = N_E * Q_E_yearly
    T_A = M / Q_E_total_yearly

    # Cost = Fixed maintenance + Transfer cost (Apex -> Moon)
    C_A_fixed = N_E * C_E_fixed_yearly * T_A
    C_A_variable = M * C_E_TRANSFER_PER_TON
    C_A = C_A_fixed + C_A_variable

    # ==========================================
    # Scenario B: Rockets Alone (Optimized)
    # ==========================================
    df_bases = get_rocket_bases_data()

    # Get AHP weights
    w_c, w_t = run_ahp(a=5)

    # Optimize base selection
    best_combo, C_B, T_B, obj_B = optimize_rocket_bases(df_bases, w_c, w_t, M)

    # Store selected bases info
    selected_bases = df_bases.iloc[best_combo]
    Q_R0_optimal = selected_bases['x_max'].sum() * Q_R_PER_LAUNCH

    # ==========================================
    # Scenario C: Hybrid (Synchronized)
    # ==========================================
    # Use optimal base combination from Scenario B
    Q_total_0 = Q_E_total_yearly + Q_R0_optimal

    # Calculate time for hybrid system
    # Combined system: elevator (constant) + rockets (growing)
    # Simplified: treat elevator as additional constant capacity
    # Create a virtual combined system

    # Approach: Solve for T_C such that:
    # - Elevator contributes: M_E = Q_E_total * T_C
    # - Rockets contribute: M_R = M - M_E (using piecewise model)

    # Iterative solution (binary search)
    T_min = 0.0
    T_max = T_B  # Must be faster than rockets alone

    for _ in range(50):  # Binary search iterations
        T_mid = (T_min + T_max) / 2.0
        M_E = Q_E_total_yearly * T_mid
        M_R_needed = M - M_E

        if M_R_needed <= 0:
            T_max = T_mid
            continue

        # Calculate time for rockets to deliver M_R_needed
        T_R_actual = calculate_piecewise_time(selected_bases, M_R_needed)

        if T_R_actual > T_mid:
            T_min = T_mid  # Need more time
        else:
            T_max = T_mid  # Can finish earlier

    T_C = T_max
    M_E = Q_E_total_yearly * T_C
    M_R = M - M_E

    # Cost C
    fixed_cost_bases = selected_bases['fixed_cost'].sum()
    C_C_fixed = (N_E * C_E_fixed_yearly + fixed_cost_bases) * T_C
    C_C_var_E = M_E * C_E_TRANSFER_PER_TON
    C_C_var_R = M_R * C_R_DIRECT_PER_TON
    C_C = C_C_fixed + C_C_var_E + C_C_var_R

    # ==========================================
    # Return Results
    # ==========================================
    return {
        "Meta": {"Mass": M},
        "A": {
            "Cost": C_A,
            "Time": T_A,
            "Label": "Space Elevator Only",
            "AnnualCapacity": Q_E_total_yearly
        },
        "B": {
            "Cost": C_B,
            "Time": T_B,
            "Label": "Rockets Only (Optimized)",
            "AnnualCapacity": Q_R0_optimal,
            "SelectedBases": best_combo,
            "BaseNames": selected_bases['Location'].tolist()
        },
        "C": {
            "Cost": C_C,
            "Time": T_C,
            "Label": "Hybrid (Synchronized)",
            "AnnualCapacity": Q_total_0,
            "ElevatorMass": M_E,
            "RocketMass": M_R
        }
    }


def run_ahp(a=5):
    """
    AHP weighting for cost vs time.

    Args:
        a: Preference parameter (time is a times more important than cost)

    Returns:
        w_c: Cost weight
        w_t: Time weight
    """
    w1 = 1 / (1 + a)  # Cost weight
    w2 = a / (1 + a)  # Time weight
    return w1, w2


def run_topsis(metrics, w_c, w_t):
    """
    TOPSIS evaluation for scenarios A, B, C.

    Args:
        metrics: Dictionary from calculate_metrics()
        w_c, w_t: Weights from AHP

    Returns:
        Dictionary with TOPSIS scores for each scenario
    """
    scenarios = ["A", "B", "C"]

    costs = np.array([metrics[s]["Cost"] for s in scenarios])
    times = np.array([metrics[s]["Time"] for s in scenarios])

    # Vector normalization
    norm_c = costs / np.sqrt(np.sum(costs**2))
    norm_t = times / np.sqrt(np.sum(times**2))

    # Weighted normalized matrix
    v_c = norm_c * w_c
    v_t = norm_t * w_t

    # Ideal solutions (both are "smaller is better")
    pis_c = np.min(v_c)  # Positive ideal (minimum cost)
    pis_t = np.min(v_t)  # Positive ideal (minimum time)

    nis_c = np.max(v_c)  # Negative ideal (maximum cost)
    nis_t = np.max(v_t)  # Negative ideal (maximum time)

    # Distances
    d_pos = np.sqrt((v_c - pis_c)**2 + (v_t - pis_t)**2)
    d_neg = np.sqrt((v_c - nis_c)**2 + (v_t - nis_t)**2)

    # Closeness coefficient
    denominator = d_pos + d_neg
    scores = np.where(denominator == 0, 0, d_neg / denominator)

    results = {}
    for i, s in enumerate(scenarios):
        results[s] = {
            "Score": scores[i],
            "Norm_Cost": norm_c[i],
            "Norm_Time": norm_t[i],
            "Weighted_Cost": v_c[i],
            "Weighted_Time": v_t[i]
        }

    return results
