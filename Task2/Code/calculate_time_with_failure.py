"""
calculate_time_with_failure.py

Purpose: Calculate mission cost and time for three scenarios (A, B, C) under non-perfect
         operational conditions with failure probabilities.

Key Features:
- Scenario A: Space elevator only (with swaying and climber failures)
- Scenario B: Rockets only (with launch failures and maintenance downtime)
- Scenario C: Hybrid system (synchronized elevator + rockets)
- Piecewise growth model from Task 1 with failure adjustments
- Geometric distribution for launch backup modeling

Output: Metrics dictionary with cost, time, and auxiliary variables for each scenario.

User Adjustment Guide:
- Line 21-35: Failure parameters (p_E1, p_E2, p_R1, p_R2, beta_E1, etc.)
- Line 38-48: Cost parameters from Task 1
- Line 51-53: Normalization bounds
"""

import numpy as np
import pandas as pd
from itertools import combinations

# ============================================================================
# Failure Parameters (Section 3.2 of Solution2.md)
# ============================================================================
P_E1 = 0.05           # Tether swaying probability per year (5%)
BETA_E1 = 0.3         # Capacity reduction factor when swaying occurs (reduces to 70%)
P_E2 = 0.03           # Climber breakdown probability per harbour per year (3%)
T_E2 = 30.0           # Climber breakdown duration (days)
C_E_REPAIR = 50e6     # Repair cost per climber breakdown (50 million USD)

P_R1 = 0.03           # Launch failure probability per launch (3%)
P_R2 = 0.04           # Launch site maintenance probability per site per year (4%)
T_R2 = 60.0           # Launch site maintenance duration (days)

# ============================================================================
# Cost and Capacity Parameters (inherited from Task 1)
# ============================================================================
C_E_FIXED = 3.0e9     # Elevator fixed annual cost per harbour (USD/year)
C_E_TRANSFER = 92.7 * 1000  # Elevator transfer cost apex->Moon (92,700 USD/ton)
Q_E_TOTAL = 537000.0  # Nominal elevator capacity, 3 harbours (tons/year)

C_R_DIRECT = 1500 * 1000    # Rocket direct cost (1,500,000 USD/ton)
Q_R_PER_LAUNCH = 150.0      # Payload per launch (tons)
LAUNCH_CAP_PER_BASE = 200   # Max launches per base per year

# ============================================================================
# Normalization Bounds (from Task 1)
# ============================================================================
C_MAX = 1.5e14        # Max cost: 150 trillion USD
T_MAX = 350.0         # Max time: 350 years

# ============================================================================
# Mission Parameters
# ============================================================================
TARGET_MASS = 1.0e8   # Total mass to Moon: 100 million tons

# ============================================================================
# Effective Capacity Calculation with Failure Adjustments
# ============================================================================

def calculate_effective_elevator_capacity():
    """
    Calculate effective elevator capacity accounting for failures.

    Formula (Section 4.2):
        Q_E,eff = Q_E,total * (1 - p_E1 * beta_E1) * (1 - p_E2 * t_E2/365)

    Returns:
        float: Effective elevator capacity (tons/year)
    """
    swaying_factor = 1.0 - P_E1 * BETA_E1
    downtime_factor = 1.0 - P_E2 * (T_E2 / 365.0)
    Q_E_eff = Q_E_TOTAL * swaying_factor * downtime_factor
    return Q_E_eff


def calculate_effective_rocket_capacity_factor():
    """
    Calculate the capacity reduction factor for rockets due to failures.

    Formula (Section 5.2.2):
        Factor = (1 - p_R1) * (1 - p_R2 * t_R2/365)

    Returns:
        float: Capacity reduction factor
    """
    delivery_rate = 1.0 - P_R1
    availability_factor = 1.0 - P_R2 * (T_R2 / 365.0)
    return delivery_rate * availability_factor


# ============================================================================
# Piecewise Growth Model with Failure Adjustment (from Task 1)
# ============================================================================

def calculate_piecewise_time_with_failure(bases_subset, target_mass):
    """
    Calculate time required using piecewise quadratic-linear growth model
    with failure adjustments.

    Args:
        bases_subset (pd.DataFrame): Selected rocket bases
        target_mass (float): Total mass to deliver (tons)

    Returns:
        float: Mission duration (years)
    """
    n_bases = len(bases_subset)
    if n_bases == 0:
        return np.inf

    # Initial launch frequency (2050) for each base
    x_0 = bases_subset['Initial_Frequency'].values

    # Growth rate per year
    g = bases_subset['Growth_Rate'].values

    # Failure-adjusted capacity factor
    failure_factor = calculate_effective_rocket_capacity_factor()

    # Total initial capacity
    Q_R0 = np.sum(x_0) * Q_R_PER_LAUNCH * failure_factor

    # Total growth rate
    g_R = np.sum(g) * Q_R_PER_LAUNCH * failure_factor

    # Calculate when each base reaches cap (200 launches/year)
    t_cap = []
    for i in range(n_bases):
        if g[i] > 0:
            t_i = (LAUNCH_CAP_PER_BASE - x_0[i]) / g[i]
            t_cap.append(t_i)
        else:
            t_cap.append(np.inf)

    t_cap = np.array(t_cap)
    t_cap_min = np.min(t_cap)

    # Phase 1: Pure quadratic growth (before first base caps)
    # M(t) = Q_R0 * t + (g_R / 2) * t^2
    M1 = Q_R0 * t_cap_min + (g_R / 2.0) * t_cap_min**2

    if M1 >= target_mass:
        # Solve quadratic: (g_R/2)*t^2 + Q_R0*t - target_mass = 0
        a = g_R / 2.0
        b = Q_R0
        c = -target_mass
        if a > 0:
            t_solution = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            return t_solution
        else:
            return target_mass / Q_R0

    # Phase 2+: Sequential base capping
    M_cumulative = M1
    t_current = t_cap_min
    active_bases = n_bases
    capped_bases = 0

    sorted_cap_indices = np.argsort(t_cap)

    for i in range(n_bases):
        if i == 0:
            continue

        idx = sorted_cap_indices[i]
        t_next = t_cap[idx]

        if M_cumulative >= target_mass:
            break

        # Current capacity (after previous bases capped)
        Q_current = (active_bases * LAUNCH_CAP_PER_BASE +
                     (n_bases - active_bases) * LAUNCH_CAP_PER_BASE) * Q_R_PER_LAUNCH * failure_factor

        # Growth rate (from uncapped bases)
        g_current = (active_bases - capped_bases) * g[sorted_cap_indices[i-1]] * Q_R_PER_LAUNCH * failure_factor

        # Mass in this segment
        dt = t_next - t_current
        M_segment = Q_current * dt + (g_current / 2.0) * dt**2

        if M_cumulative + M_segment >= target_mass:
            # Solve for time in this segment
            remaining_mass = target_mass - M_cumulative

            if g_current > 0:
                a = g_current / 2.0
                b = Q_current
                c = -remaining_mass
                dt_solution = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            else:
                dt_solution = remaining_mass / Q_current

            return t_current + dt_solution

        M_cumulative += M_segment
        t_current = t_next
        capped_bases += 1

    # Phase 3: All bases at max capacity (linear growth)
    Q_max = n_bases * LAUNCH_CAP_PER_BASE * Q_R_PER_LAUNCH * failure_factor
    remaining_mass = target_mass - M_cumulative
    t_final = t_current + remaining_mass / Q_max

    return t_final


# ============================================================================
# Expected Launch Count with Geometric Distribution (Section 5.2.1)
# ============================================================================

def calculate_expected_launches(mass_to_deliver):
    """
    Calculate expected number of launches accounting for failure probability.

    Formula (Section 5.2.1):
        N' = (M / (Q_R * (1-p_R1)^2))

    Geometric distribution: E[attempts per success] = 1 / (1-p_R1)

    Args:
        mass_to_deliver (float): Total mass to deliver by rockets (tons)

    Returns:
        float: Expected number of launches
    """
    geometric_factor = 1.0 / (1.0 - P_R1)
    expected_launches = (mass_to_deliver / (Q_R_PER_LAUNCH * (1.0 - P_R1))) * geometric_factor
    return expected_launches


# ============================================================================
# Scenario A: Elevator Only with Failures
# ============================================================================

def calculate_scenario_A():
    """
    Calculate metrics for Scenario A: Space elevator only.

    Returns:
        dict: Contains C_A', T_A', Q_E_eff, E[N_E2], Z_A'
    """
    Q_E_eff = calculate_effective_elevator_capacity()

    # Mission duration (Section 4.2)
    T_A_prime = TARGET_MASS / Q_E_eff

    # Expected breakdown count (Section 4.2)
    E_N_E2 = 3 * P_E2 * T_A_prime

    # Expected total cost (Section 4.2)
    C_A_prime = (3 * C_E_FIXED * T_A_prime +
                 TARGET_MASS * C_E_TRANSFER +
                 3 * C_E_REPAIR * E_N_E2)

    return {
        'cost': C_A_prime,
        'time': T_A_prime,
        'Q_E_eff': Q_E_eff,
        'E_N_E2': E_N_E2,
        'objective': None  # Will be set later with weights
    }


# ============================================================================
# Scenario B: Rockets Only with Failures and 0-1 Optimization
# ============================================================================

def optimize_rocket_bases(df_bases, w_c, w_t):
    """
    Optimize rocket base selection using enumeration-based 0-1 programming.

    Args:
        df_bases (pd.DataFrame): Rocket base data
        w_c (float): Cost weight
        w_t (float): Time weight

    Returns:
        dict: Best solution with cost, time, selected bases, objective
    """
    n_total = len(df_bases)
    best_obj = np.inf
    best_solution = None

    # Enumerate all non-empty subsets (2^10 - 1 = 1023 combinations)
    for r in range(1, n_total + 1):
        for combo in combinations(range(n_total), r):
            bases_subset = df_bases.iloc[list(combo)]

            # Calculate time with piecewise growth model
            T_B_prime = calculate_piecewise_time_with_failure(bases_subset, TARGET_MASS)

            if np.isinf(T_B_prime):
                continue

            # Calculate expected total launches with geometric distribution
            N_prime_total = calculate_expected_launches(TARGET_MASS)

            # Distribute launches proportionally among selected bases
            n_selected = len(bases_subset)
            N_prime_per_base = N_prime_total / n_selected

            # Calculate cost (Section 5.2)
            C_fixed = bases_subset['Fixed_Cost'].sum() * T_B_prime
            C_launch = N_prime_per_base * n_selected * Q_R_PER_LAUNCH * C_R_DIRECT
            C_B_prime = C_fixed + C_launch

            # Normalized objective
            C_norm = C_B_prime / C_MAX
            T_norm = T_B_prime / T_MAX
            obj = w_c * C_norm + w_t * T_norm

            if obj < best_obj:
                best_obj = obj
                best_solution = {
                    'cost': C_B_prime,
                    'time': T_B_prime,
                    'selected_bases': bases_subset.index.tolist(),
                    'n_selected': n_selected,
                    'N_prime_total': N_prime_total,
                    'objective': obj
                }

    return best_solution


# ============================================================================
# Scenario C: Hybrid System with Failures and Synchronization
# ============================================================================

def calculate_scenario_C_with_sync(df_bases, w_c, w_t, selected_bases_indices):
    """
    Calculate metrics for Scenario C: Hybrid system with time synchronization.

    Uses binary search to find mission duration where elevator and rocket
    capacities are balanced.

    Args:
        df_bases (pd.DataFrame): Rocket base data
        w_c (float): Cost weight
        w_t (float): Time weight
        selected_bases_indices (list): Pre-selected rocket bases from Scenario B

    Returns:
        dict: Contains C_C', T_C', alpha, selected bases, objective
    """
    bases_subset = df_bases.loc[selected_bases_indices]
    Q_E_eff = calculate_effective_elevator_capacity()

    # Binary search for synchronized time
    t_low, t_high = 1.0, T_MAX
    tolerance = 1e-6
    max_iter = 50

    for iteration in range(max_iter):
        t_mid = (t_low + t_high) / 2.0

        # Elevator mass delivered
        M_E = Q_E_eff * t_mid

        # Rocket mass delivered (need to integrate piecewise growth)
        M_R = calculate_rocket_mass_delivered(bases_subset, t_mid)

        M_total = M_E + M_R

        if abs(M_total - TARGET_MASS) < tolerance:
            T_C_prime = t_mid
            break
        elif M_total < TARGET_MASS:
            t_low = t_mid
        else:
            t_high = t_mid
    else:
        T_C_prime = t_mid

    # Calculate alpha (elevator fraction)
    M_E_final = Q_E_eff * T_C_prime
    M_R_final = calculate_rocket_mass_delivered(bases_subset, T_C_prime)
    alpha = M_E_final / TARGET_MASS

    # Expected launches for rocket portion
    N_prime_total = calculate_expected_launches(M_R_final)

    # Calculate cost (Section 6.2)
    C_fixed = (3 * C_E_FIXED + bases_subset['Fixed_Cost'].sum()) * T_C_prime
    C_elevator = alpha * TARGET_MASS * C_E_TRANSFER
    C_rocket = N_prime_total * Q_R_PER_LAUNCH * C_R_DIRECT
    C_repair = 3 * C_E_REPAIR * P_E2 * T_C_prime
    C_C_prime = C_fixed + C_elevator + C_rocket + C_repair

    return {
        'cost': C_C_prime,
        'time': T_C_prime,
        'alpha': alpha,
        'M_E': M_E_final,
        'M_R': M_R_final,
        'selected_bases': selected_bases_indices,
        'N_prime_total': N_prime_total,
        'objective': w_c * (C_C_prime / C_MAX) + w_t * (T_C_prime / T_MAX)
    }


def calculate_rocket_mass_delivered(bases_subset, time_duration):
    """
    Calculate total mass delivered by rockets over given time duration.
    Uses piecewise integration of growth model with failure adjustments.

    Args:
        bases_subset (pd.DataFrame): Selected rocket bases
        time_duration (float): Mission duration (years)

    Returns:
        float: Total mass delivered (tons)
    """
    n_bases = len(bases_subset)
    x_0 = bases_subset['Initial_Frequency'].values
    g = bases_subset['Growth_Rate'].values

    failure_factor = calculate_effective_rocket_capacity_factor()

    # Calculate capping times
    t_cap = []
    for i in range(n_bases):
        if g[i] > 0:
            t_i = (LAUNCH_CAP_PER_BASE - x_0[i]) / g[i]
            t_cap.append(min(t_i, time_duration))
        else:
            t_cap.append(time_duration)

    t_cap = np.array(t_cap)
    t_cap_min = np.min(t_cap)

    # Phase 1: Quadratic growth
    if time_duration <= t_cap_min:
        # All bases in quadratic phase
        M_total = 0.0
        for i in range(n_bases):
            M_i = (x_0[i] * time_duration + (g[i] / 2.0) * time_duration**2) * Q_R_PER_LAUNCH * failure_factor
            M_total += M_i
        return M_total

    # Integrate piecewise
    M_cumulative = 0.0
    t_current = 0.0
    sorted_indices = np.argsort(t_cap)

    for i in range(n_bases):
        idx = sorted_indices[i]
        t_next = min(t_cap[idx], time_duration)

        if t_next <= t_current:
            continue

        dt = t_next - t_current

        # Count uncapped bases
        uncapped = np.sum(t_cap > t_current)
        capped = n_bases - uncapped

        # Current capacity
        Q_current = (np.sum(x_0[t_cap > t_current] + g[t_cap > t_current] * t_current) +
                     capped * LAUNCH_CAP_PER_BASE) * Q_R_PER_LAUNCH * failure_factor

        # Growth rate
        g_current = np.sum(g[t_cap > t_current]) * Q_R_PER_LAUNCH * failure_factor

        # Mass in segment
        M_segment = Q_current * dt + (g_current / 2.0) * dt**2
        M_cumulative += M_segment

        t_current = t_next

        if t_current >= time_duration:
            break

    # Final linear phase if time exceeds all caps
    if time_duration > np.max(t_cap):
        t_remaining = time_duration - np.max(t_cap)
        Q_max = n_bases * LAUNCH_CAP_PER_BASE * Q_R_PER_LAUNCH * failure_factor
        M_cumulative += Q_max * t_remaining

    return M_cumulative


# ============================================================================
# Main Calculation Function
# ============================================================================

def calculate_metrics_with_failure(df_bases, w_c=1/6, w_t=5/6, verbose=True):
    """
    Calculate all three scenarios with failure adjustments.

    Args:
        df_bases (pd.DataFrame): Rocket base data
        w_c (float): Cost weight from AHP
        w_t (float): Time weight from AHP
        verbose (bool): Whether to print progress messages

    Returns:
        dict: Results for scenarios A, B, C
    """
    if verbose:
        print("  [1/3] Scenario A (Elevator only)...", end=" ")
    results_A = calculate_scenario_A()
    results_A['objective'] = w_c * (results_A['cost'] / C_MAX) + w_t * (results_A['time'] / T_MAX)
    if verbose:
        print("✓")

    if verbose:
        print("  [2/3] Scenario B (Rockets only, 0-1 optimization)...", end=" ")
    results_B = optimize_rocket_bases(df_bases, w_c, w_t)
    if verbose:
        print("✓")

    if verbose:
        print("  [3/3] Scenario C (Hybrid system with synchronization)...", end=" ")
    selected_bases = results_B['selected_bases']
    results_C = calculate_scenario_C_with_sync(df_bases, w_c, w_t, selected_bases)
    if verbose:
        print("✓\n")

    return {
        'A': results_A,
        'B': results_B,
        'C': results_C
    }


# ============================================================================
# Test Code (for standalone execution)
# ============================================================================

if __name__ == "__main__":
    # Create dummy rocket base data for testing
    np.random.seed(42)
    df_test = pd.DataFrame({
        'Initial_Frequency': np.random.uniform(10, 30, 10),
        'Growth_Rate': np.random.uniform(1, 3, 10),
        'Fixed_Cost': np.random.uniform(1e8, 5e8, 10)
    })

    results = calculate_metrics_with_failure(df_test)

    print("\n" + "="*60)
    print("SCENARIO A: Elevator Only (with failures)")
    print("="*60)
    print(f"Cost: ${results['A']['cost']/1e12:.2f} trillion")
    print(f"Time: {results['A']['time']:.2f} years")
    print(f"Effective elevator capacity: {results['A']['Q_E_eff']:.0f} tons/year")
    print(f"Expected breakdowns: {results['A']['E_N_E2']:.2f}")
    print(f"Normalized objective: {results['A']['objective']:.6f}")

    print("\n" + "="*60)
    print("SCENARIO B: Rockets Only (with failures)")
    print("="*60)
    print(f"Cost: ${results['B']['cost']/1e12:.2f} trillion")
    print(f"Time: {results['B']['time']:.2f} years")
    print(f"Selected bases: {results['B']['n_selected']} out of 10")
    print(f"Expected total launches: {results['B']['N_prime_total']:.0f}")
    print(f"Normalized objective: {results['B']['objective']:.6f}")

    print("\n" + "="*60)
    print("SCENARIO C: Hybrid System (with failures)")
    print("="*60)
    print(f"Cost: ${results['C']['cost']/1e12:.2f} trillion")
    print(f"Time: {results['C']['time']:.2f} years")
    print(f"Elevator fraction: {results['C']['alpha']*100:.1f}%")
    print(f"Elevator mass: {results['C']['M_E']/1e6:.2f} million tons")
    print(f"Rocket mass: {results['C']['M_R']/1e6:.2f} million tons")
    print(f"Normalized objective: {results['C']['objective']:.6f}")
