"""
Task 4 Model Parameters
=======================
Purpose: Centralized parameter definitions for multi-objective optimization
         Includes costs, capacities, failure rates, environmental impacts, and AHP weights
"""

import numpy as np

# ============================================================================
# USER ADJUSTMENT GUIDE
# ============================================================================
# [Line 20-30]   Task parameters (total mass, water demand)
# [Line 35-50]   Space elevator system parameters
# [Line 55-75]   Rocket system parameters
# [Line 80-95]   Failure parameters
# [Line 100-125] Environmental impact baseline values
# [Line 130-145] AHP weights (updated from Solution4.md)
# [Line 150-165] Normalization bounds
# ============================================================================

# ============================================================================
# 1. Task Parameters
# ============================================================================

M = 1e8                      # Total construction material mass (tonnes)
M_w = 3.05125e5              # Annual water demand (tonnes)
M_total = M + M_w            # Total transportation mass (tonnes)
T_oper = 10 * 365            # Long-term operation cycle (days), 10 years as example

# ============================================================================
# 2. Space Elevator System Parameters
# ============================================================================

N_E = 3                      # Number of Galactic Harbours
Q_E = 1.79e5                 # Single harbour annual capacity (tonnes/year)
Q_E_total = N_E * Q_E        # Total nominal capacity (tonnes/year)

# Cost parameters (from Task 1)
C_E_f = 3.0e9                # Fixed annual cost per harbour (USD/year)
C_E_u = 92700                # Unit transfer cost (USD/tonne) - apex to moon
C_E_repair = 5.0e7           # Climber repair cost (USD/event)

# Decision variable ranges
s_E_elec_min = 0.8           # Min renewable energy ratio
s_E_elec_max = 1.0           # Max renewable energy ratio
m_E_min = 1                  # Min maintenance frequency (times/harbour/year)
m_E_max = 10                 # Max maintenance frequency

# ============================================================================
# 3. Rocket System Parameters
# ============================================================================

Q_R = 150                    # Single launch payload (tonnes)
c_R = 1500000                # Direct transport cost (USD/tonne)
C_R_i = Q_R * c_R            # Single launch cost (USD)
C_R_repair = 1.0e8           # Launch site repair cost (USD/event)

# Launch site fixed costs (USD/year) - from Task 1 Table 9.1
C_R_l = np.array([
    4.0e8,   # 1. French Guiana
    6.0e8,   # 2. Florida (Kennedy)
    0.2e8,   # 3. India (Satish Dhawan)
    0.02e8,  # 4. New Zealand (Mahia)
    0.6e8,   # 5. Texas (Boca Chica)
    0.02e8,  # 6. Alaska
    3.0e8,   # 7. California (Vandenberg)
    0.04e8,  # 8. Virginia (Wallops)
    0.68e8,  # 9. Kazakhstan
    0.02e8   # 10. China (Taiyuan)
])

# Launch site safety coefficients (from Solution4.md Table 7.2)
s_R_safe = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0,  # Low risk sites (1-5)
    1.5, 1.5, 1.5, 1.5, 1.5   # High risk sites (6-10)
])

# Max annual launches per site (from Task 1)
x_max = np.array([25, 93, 25, 31, 22, 21, 54, 22, 29, 31])

# Fuel type modifiers
f_R_fuel_traditional = 1.0   # Traditional kerosene cost modifier
f_R_fuel_green = 1.5         # Green fuel cost modifier (50% increase)
f_R_env_traditional = 1.0    # Traditional kerosene environmental modifier
f_R_env_green = 0.0          # Green fuel environmental modifier (zero emissions)

# ============================================================================
# 4. Failure Parameters (from Task 2)
# ============================================================================

# Space elevator failure modes
p_E1 = 0.05                  # Tether sway probability (annual)
beta_E1 = 0.3                # Capacity reduction coefficient
p_E2 = 0.03                  # Climber failure probability (annual/harbour)
t_E2 = 30                    # Climber failure downtime (days/event)

# Rocket system failure modes
p_R1 = 0.03                  # Launch failure rate (per launch)
p_R2 = 0.04                  # Launch site maintenance probability (annual/site)
t_R2 = 60                    # Launch site maintenance downtime (days/event)

# ============================================================================
# 5. Environmental Impact Baseline Values (from Solution4.md Table 5.1)
# ============================================================================

# Pure elevator values (E^a)
E_PM25_a = 0.0               # PM2.5 emission (kg/tonne)
E_CO2_a = 1.2                # CO2 emission (kg/tonne)
E_reso_a = 0.35              # Resource consumption (kg/tonne)
E_eco_a = 1.2                # Ecological impact index (0-10)

# Pure rocket baseline values (E^max = E^b)
E_PM25_max = 0.85            # PM2.5 emission (kg/tonne)
E_CO2_max = 1860             # CO2 emission (kg/tonne)
E_reso_max = 200             # Resource consumption (kg/tonne)
E_eco_max = 8.5              # Ecological impact index (0-10)

# Pure rocket values (same as max for traditional fuel)
E_PM25_b = E_PM25_max
E_CO2_b = E_CO2_max
E_reso_b = E_reso_max
E_eco_b = E_eco_max

# ============================================================================
# 6. AHP Weights (UPDATED from Solution4.md Section 4.3)
# ============================================================================

# Main objective weights (Cost, Time, Reliability, Environment)
w_main = np.array([0.2190, 0.1012, 0.0508, 0.6290])

# Environmental sub-weights (PM2.5, CO2, Ecology, Resource)
w_env = np.array([0.1792, 0.6608, 0.1087, 0.0513])

# Individual weight symbols for clarity
w_cost = w_main[0]           # 0.2190 (21.90%)
w_time = w_main[1]           # 0.1012 (10.12%)
w_reliability = w_main[2]    # 0.0508 (5.08%)
w_environment = w_main[3]    # 0.6290 (62.90%)

w_PM25 = w_env[0]            # 0.1792 (17.92%)
w_CO2 = w_env[1]             # 0.6608 (66.08%)
w_eco = w_env[2]             # 0.1087 (10.87%)
w_reso = w_env[3]            # 0.0513 (5.13%)

# ============================================================================
# 7. Normalization Bounds (from Task 1)
# ============================================================================

C_max = 1.5e14               # Maximum cost (USD)
T_max = 350 * 365            # Maximum time (days), 350 years
R_max = 1.0                  # Maximum reliability (perfect system)

# Environmental impact max (normalized to 1.0 using weighted sum)
# EI_max = sum(w_env * [1, 1, 1, 1]) = 1.0 when all E^k = E^max
EI_max = np.sum(w_env)       # Should be 1.0 (normalized)

# ============================================================================
# 8. Launch Site Names (for visualization)
# ============================================================================

launch_site_names = [
    "French Guiana",
    "Florida (USA)",
    "Satish Dhawan (India)",
    "Mahia (New Zealand)",
    "Texas (USA)",
    "Alaska (USA)",
    "California (USA)",
    "Virginia (USA)",
    "Kazakhstan",
    "Taiyuan (China)"
]

# ============================================================================
# 9. Professional Color Palette (for visualization)
# ============================================================================

# Nature-style professional colors (Hex codes)
COLOR_PALETTE = {
    'elevator': '#1f77b4',    # Blue
    'rocket': '#ff7f0e',      # Orange
    'hybrid': '#2ca02c',      # Green
    'cost': '#d62728',        # Red
    'time': '#9467bd',        # Purple
    'reliability': '#8c564b', # Brown
    'environment': '#17becf', # Cyan
    'PM25': '#bcbd22',        # Yellow-green
    'CO2': '#e377c2',         # Pink
    'ecology': '#7f7f7f',     # Gray
    'resource': '#bcbd22'     # Yellow-green
}
