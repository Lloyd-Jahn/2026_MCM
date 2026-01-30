import numpy as np
import pandas as pd

def get_rocket_bases_data():
    """
    Returns the rocket base data as defined in Solution1.md
    Columns: Location, Max Launches/Year (x_max), Fixed Cost (USD/Year)
    """
    data = [
        {"Location": "Florida", "x_max": 73, "fixed_cost": 6e8},
        {"Location": "California", "x_max": 34, "fixed_cost": 3e8},
        {"Location": "Taiyuan (China)", "x_max": 11, "fixed_cost": 0.02e8},
        {"Location": "Mahia (NZ)", "x_max": 11, "fixed_cost": 0.02e8},
        {"Location": "Kazakhstan", "x_max": 9, "fixed_cost": 0.68e8},
        {"Location": "French Guiana", "x_max": 5, "fixed_cost": 4e8},
        {"Location": "India", "x_max": 5, "fixed_cost": 0.2e8},
        {"Location": "Virginia", "x_max": 2, "fixed_cost": 0.04e8},
        {"Location": "Texas", "x_max": 2, "fixed_cost": 0.6e8},
        {"Location": "Alaska", "x_max": 1, "fixed_cost": 0.02e8},
    ]
    return pd.DataFrame(data)

def calculate_metrics():
    """
    Calculates Cost (C) and Time (T) for Scenarios A, B, and C.
    M: Total Mass (metric tons)
    Also returns annual capacity for timeline visualization.
    """
    # Global Parameters
    M = 1e8  # 100 million metric tons
    
    # --- Scenario A: Space Elevator Alone ---
    # Parameters
    N_E = 3  # Number of Galactic Harbours
    Q_E_yearly = 1.79e5  # Capacity per harbour per year (tons)
    C_E_fixed_yearly = 3e9  # Fixed cost per harbour per year (USD)
    
    # Logic
    Q_E_total_yearly = N_E * Q_E_yearly
    T_A = M / Q_E_total_yearly
    C_A = N_E * C_E_fixed_yearly * T_A  # Variable cost assumed ~0
    
    # --- Scenario B: Rockets Alone ---
    # Parameters
    df_bases = get_rocket_bases_data()
    # Assume we use ALL bases to minimize time (since a=5 implies Time is priority)
    # x_i = x_max_i (using full capacity)
    total_launches_per_year = df_bases['x_max'].sum()
    total_fixed_cost_yearly = df_bases['fixed_cost'].sum()
    
    Q_R_per_launch = 150  # tons per launch
    c_R_per_ton = 1.5e6   # USD per ton (variable cost)
    
    # Logic
    Q_R_total_yearly = total_launches_per_year * Q_R_per_launch
    T_B = M / Q_R_total_yearly
    
    # Cost B = (Total Fixed Cost/Year * Years) + (Variable Cost * Mass)
    C_B_fixed = total_fixed_cost_yearly * T_B
    C_B_variable = M * c_R_per_ton
    C_B = C_B_fixed + C_B_variable
    
    # --- Scenario C: Combination (Synchronized) ---
    # We want time T_C such that both systems work to finish together.
    # Combined yearly capacity
    Q_Total_yearly = Q_E_total_yearly + Q_R_total_yearly
    
    # Logic
    T_C = M / Q_Total_yearly
    
    # Assume they run for T_C years. 
    # Cost = (Elevator Fixed * T_C) + (Rocket Fixed * T_C) + (Rocket Variable * Mass_Rocket)
    # Mass carried by Rocket (M_R) is proportional to its capacity share
    alpha = Q_E_total_yearly / Q_Total_yearly # Percentage carried by Elevator
    M_R = (1 - alpha) * M
    
    C_C_fixed_elevator = N_E * C_E_fixed_yearly * T_C
    C_C_fixed_rocket = total_fixed_cost_yearly * T_C
    C_C_variable_rocket = M_R * c_R_per_ton
    
    C_C = C_C_fixed_elevator + C_C_fixed_rocket + C_C_variable_rocket

    return {
        "Meta": {"Mass": M},
        "A": {"Cost": C_A, "Time": T_A, "Label": "Scenario A", "AnnualCapacity": Q_E_total_yearly},
        "B": {"Cost": C_B, "Time": T_B, "Label": "Scenario B", "AnnualCapacity": Q_R_total_yearly},
        "C": {"Cost": C_C, "Time": T_C, "Label": "Scenario C", "AnnualCapacity": Q_Total_yearly}
    }

def run_ahp(a=5):
    """
    Returns weights for [Cost, Time] based on comparison value a.
    Matrix: [[1, 1/a], [a, 1]]
    """
    A = np.array([[1, 1/a], [a, 1]])
    # Approximate weights calculation (Normalizing columns)
    # For 2x2 consistent matrix:
    w1 = 1 / (1 + a) # Cost weight
    w2 = a / (1 + a) # Time weight
    return w1, w2

def run_topsis(metrics, w_c, w_t):
    """
    Performs TOPSIS ranking.
    metrics: dict of scenarios
    w_c: weight for Cost
    w_t: weight for Time
    Both Cost and Time are 'Lower is Better' (Cost attributes)
    """
    scenarios = ["A", "B", "C"]
    
    # 1. Build Decision Matrix
    costs = np.array([metrics[s]["Cost"] for s in scenarios])
    times = np.array([metrics[s]["Time"] for s in scenarios])
    
    # 2. Vector Normalization
    norm_c = costs / np.sqrt(np.sum(costs**2))
    norm_t = times / np.sqrt(np.sum(times**2))
    
    # 3. Weighted Normalized Matrix
    v_c = norm_c * w_c
    v_t = norm_t * w_t
    
    # 4. Determine Ideal Solutions
    # Both are "min" criteria (smaller is better)
    # Positive Ideal (PIS): min(Cost), min(Time)
    # Negative Ideal (NIS): max(Cost), max(Time)
    
    pis_c = np.min(v_c)
    pis_t = np.min(v_t)
    
    nis_c = np.max(v_c)
    nis_t = np.max(v_t)
    
    # 5. Separation Measures
    d_pos = np.sqrt((v_c - pis_c)**2 + (v_t - pis_t)**2)
    d_neg = np.sqrt((v_c - nis_c)**2 + (v_t - nis_t)**2)
    
    # 6. Relative Closeness (Score)
    scores = d_neg / (d_pos + d_neg)
    
    # Package results
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
