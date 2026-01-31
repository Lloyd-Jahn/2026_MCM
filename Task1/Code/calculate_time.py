import numpy as np
import pandas as pd

# ==========================================
# Task 1 Parameter Configuration
# ==========================================

# 1. Baseline Offset: 2050 assumption
# "Technological Advancement": Every site adds 20 launches to its current average immediately by 2050.
addition = 20 

# 2. Non-linear Growth Rate
# "Infrastructure Expansion": 
# Each of the 10 sites increases capacity by 1 launch every year.
# Global Frequency Growth = 10 sites * 1 launch/year = 10 launches/year^2
rocket_frequency_growth_rate = 10 

def get_rocket_bases_data():
    """
    Returns the rocket base data as defined in Solution1.md
    Columns: Location, Max Launches/Year (x_max), Fixed Cost (USD/Year)
    """
    data = [
        {"Location": "Florida", "x_max": 73 + addition, "fixed_cost": 6e8},
        {"Location": "California", "x_max": 34 + addition, "fixed_cost": 3e8},
        {"Location": "Taiyuan (China)", "x_max": 11 + addition, "fixed_cost": 0.02e8},
        {"Location": "Mahia (NZ)", "x_max": 11 + addition, "fixed_cost": 0.02e8},
        {"Location": "Kazakhstan", "x_max": 9 + addition, "fixed_cost": 0.68e8},
        {"Location": "French Guiana", "x_max": 5 + addition, "fixed_cost": 4e8},
        {"Location": "India", "x_max": 5 + addition, "fixed_cost": 0.2e8},
        {"Location": "Virginia", "x_max": 2 + addition, "fixed_cost": 0.04e8},
        {"Location": "Texas", "x_max": 2 + addition, "fixed_cost": 0.6e8},
        {"Location": "Alaska", "x_max": 1 + addition, "fixed_cost": 0.02e8},
    ]
    return pd.DataFrame(data)

def solve_quadratic_time(initial_capacity, growth_capacity, target_mass):
    """
    Solves for Time T where Cumulative Transported Mass >= Target Mass.
    Equation: (g/2) * T^2 + (C_0 - g/2) * T - M = 0
    """
    if growth_capacity == 0:
        return target_mass / initial_capacity
        
    a = growth_capacity / 2.0
    b = initial_capacity - (growth_capacity / 2.0)
    c = -target_mass
    
    delta = b**2 - 4*a*c
    if delta < 0:
        return float('inf')
        
    T = (-b + np.sqrt(delta)) / (2*a)
    return T

def calculate_metrics():
    """
    Calculates Cost (C) and Time (T) for Scenarios A, B, and C.
    Explicitly handles Apex-to-Moon transfer costs.
    """
    M = 1e8  # 100 million metric tons
    
    # ------------------------------------------
    # Cost Parameters (Unit: USD/ton)
    # ------------------------------------------
    # 1. Elevator Route
    c_E_climber_per_ton = 150 * 1000  # $150/kg
    c_transfer_per_ton = 100 * 1000   # $100/kg (Apex->Moon)
    c_E_total_per_ton = c_E_climber_per_ton + c_transfer_per_ton # $250/kg
    
    # 2. Rocket Route
    c_R_direct_per_ton = 500 * 1000   # $500/kg
    
    # ==========================================
    # Scenario A: Space Elevator Alone
    # ==========================================
    N_E = 3 
    Q_E_yearly = 1.79e5 
    C_E_fixed_yearly = 3e9 
    
    Q_E_total_yearly = N_E * Q_E_yearly
    T_A = M / Q_E_total_yearly
    C_A = (N_E * C_E_fixed_yearly * T_A) + (M * c_E_total_per_ton)
    
    # ==========================================
    # Scenario B: Rockets Alone
    # ==========================================
    df_bases = get_rocket_bases_data()
    total_launches_initial = df_bases['x_max'].sum()
    total_fixed_cost_yearly = df_bases['fixed_cost'].sum()
    
    # FIXED PAYLOAD: 150 tons (Constant)
    Q_R_per_launch = 150 
    
    # Initial Annual Capacity
    Q_R_initial_yearly = total_launches_initial * Q_R_per_launch
    
    # Growth in Capacity due to Frequency Increase ONLY
    # g = (10 launches/year^2) * 150 tons/launch = 1500 tons/year^2
    Q_R_growth_yearly = rocket_frequency_growth_rate * Q_R_per_launch
    
    # Time B
    T_B = solve_quadratic_time(Q_R_initial_yearly, Q_R_growth_yearly, M)
    
    # Cost B
    C_B_fixed = total_fixed_cost_yearly * T_B
    C_B_variable = M * c_R_direct_per_ton
    C_B = C_B_fixed + C_B_variable
    
    # ==========================================
    # Scenario C: Hybrid (Synchronized)
    # ==========================================
    Q_Total_initial = Q_E_total_yearly + Q_R_initial_yearly
    Q_Total_growth = Q_R_growth_yearly
    
    # Time C
    T_C = solve_quadratic_time(Q_Total_initial, Q_Total_growth, M)
    
    # Load Allocation
    M_E = Q_E_total_yearly * T_C
    M_R = M - M_E
    
    # Cost C
    C_C_fixed = (N_E * C_E_fixed_yearly * T_C) + (total_fixed_cost_yearly * T_C)
    C_C_var_E = M_E * c_E_total_per_ton
    C_C_var_R = M_R * c_R_direct_per_ton
    C_C = C_C_fixed + C_C_var_E + C_C_var_R

    return {
        "Meta": {"Mass": M},
        "A": {"Cost": C_A, "Time": T_A, "Label": "Scenario A", "AnnualCapacity": Q_E_total_yearly},
        "B": {"Cost": C_B, "Time": T_B, "Label": "Scenario B", "AnnualCapacity": Q_R_initial_yearly},
        "C": {"Cost": C_C, "Time": T_C, "Label": "Scenario C", "AnnualCapacity": Q_Total_initial}
    }

def run_ahp(a=5):
    w1 = 1 / (1 + a) # Cost weight
    w2 = a / (1 + a) # Time weight
    return w1, w2

def run_topsis(metrics, w_c, w_t):
    scenarios = ["A", "B", "C"]
    
    costs = np.array([metrics[s]["Cost"] for s in scenarios])
    times = np.array([metrics[s]["Time"] for s in scenarios])
    
    norm_c = costs / np.sqrt(np.sum(costs**2))
    norm_t = times / np.sqrt(np.sum(times**2))
    
    v_c = norm_c * w_c
    v_t = norm_t * w_t
    
    pis_c = np.min(v_c)
    pis_t = np.min(v_t)
    
    nis_c = np.max(v_c)
    nis_t = np.max(v_t)
    
    d_pos = np.sqrt((v_c - pis_c)**2 + (v_t - pis_t)**2)
    d_neg = np.sqrt((v_c - nis_c)**2 + (v_t - nis_t)**2)
    
    denominator = d_pos + d_neg
    if np.any(denominator == 0):
        scores = np.zeros(3)
    else:
        scores = d_neg / denominator
    
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