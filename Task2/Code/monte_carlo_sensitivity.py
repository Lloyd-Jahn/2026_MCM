"""
monte_carlo_sensitivity.py

Purpose: Perform Monte Carlo simulation and Sobol sensitivity analysis for failure
         parameter uncertainty quantification.

Key Features:
- Latin Hypercube Sampling (LHS) for 7-dimensional parameter space
- 10,000 Monte Carlo samples
- Sobol variance decomposition (first-order and total-order indices)
- Statistical analysis: mean, median, std, CV, confidence intervals
- Uncertainty propagation from failure parameters to cost/time outputs

Output: Dictionary with MC results, statistics, and Sobol indices.

User Adjustment Guide:
- Line 28-35: Failure parameter sampling ranges
- Line 38-39: Monte Carlo settings (n_samples, random_seed)
- Line 145-155: Sobol analysis settings
"""

import numpy as np
import pandas as pd
from SALib.sample import latin as lhs_sampler
from SALib.analyze import sobol
import warnings
warnings.filterwarnings('ignore')

# Import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

# ============================================================================
# Sampling Ranges for Failure Parameters (Section 8.2, Step 1)
# ============================================================================
PARAM_BOUNDS = {
    'p_E1': [0.03, 0.07],       # Tether swaying probability: 3%-7%
    'beta_E1': [0.2, 0.4],      # Capacity reduction factor: 0.2-0.4
    'p_E2': [0.01, 0.05],       # Climber breakdown probability: 1%-5%
    't_E2': [20.0, 40.0],       # Climber breakdown duration: 20-40 days
    'p_R1': [0.02, 0.04],       # Launch failure probability: 2%-4%
    'p_R2': [0.02, 0.06],       # Launch site maintenance probability: 2%-6%
    't_R2': [45.0, 75.0]        # Launch site maintenance duration: 45-75 days
}

# Monte Carlo settings
N_SAMPLES = 10000             # Number of Monte Carlo samples
RANDOM_SEED = 42              # For reproducibility

# ============================================================================
# Latin Hypercube Sampling (LHS)
# ============================================================================

def generate_lhs_samples(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """
    Generate Latin Hypercube Samples for 7 failure parameters.

    Args:
        n_samples (int): Number of samples
        seed (int): Random seed

    Returns:
        pd.DataFrame: Sampled parameter values
    """
    np.random.seed(seed)

    # Create bounds array for SALib
    bounds = np.array(list(PARAM_BOUNDS.values()))

    # Define problem for SALib
    problem = {
        'num_vars': 7,
        'names': list(PARAM_BOUNDS.keys()),
        'bounds': bounds
    }

    # Generate LHS samples
    samples = lhs_sampler.sample(problem, n_samples, seed=seed)

    # Convert to DataFrame
    df_samples = pd.DataFrame(samples, columns=problem['names'])

    return df_samples, problem


# ============================================================================
# Monte Carlo Simulation Loop (Section 8.2, Step 2)
# ============================================================================

def run_monte_carlo_simulation(df_samples, df_bases, calculate_func, w_c=1/6, w_t=5/6):
    """
    Run Monte Carlo simulation for all scenarios.

    Args:
        df_samples (pd.DataFrame): LHS samples of failure parameters
        df_bases (pd.DataFrame): Rocket base data
        calculate_func (function): Calculation function from calculate_time_with_failure
        w_c (float): Cost weight
        w_t (float): Time weight

    Returns:
        pd.DataFrame: MC results with columns [C_A, T_A, C_B, T_B, C_C, T_C, Z_A, Z_B, Z_C]
    """
    n_samples = len(df_samples)
    results_list = []

    print(f"\n{'='*70}")
    print(f"Running Monte Carlo simulation with {n_samples:,} samples")
    print(f"{'='*70}")

    # Import necessary modules
    import calculate_time_with_failure as calc

    # Create progress bar
    if TQDM_AVAILABLE:
        iterator = tqdm(range(n_samples), desc="MC Simulation", unit="sample",
                       ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        iterator = range(n_samples)
        print(f"Processing {n_samples:,} samples (progress updates every 1000 samples)...")

    for idx in iterator:
        if not TQDM_AVAILABLE and (idx + 1) % 1000 == 0:
            print(f"  [{idx + 1:,}/{n_samples:,}] {(idx+1)/n_samples*100:.1f}% complete")

        # Extract sampled parameters
        sample = df_samples.iloc[idx]

        # Temporarily override global parameters in calc module
        original_params = {
            'P_E1': calc.P_E1,
            'BETA_E1': calc.BETA_E1,
            'P_E2': calc.P_E2,
            'T_E2': calc.T_E2,
            'P_R1': calc.P_R1,
            'P_R2': calc.P_R2,
            'T_R2': calc.T_R2
        }

        calc.P_E1 = sample['p_E1']
        calc.BETA_E1 = sample['beta_E1']
        calc.P_E2 = sample['p_E2']
        calc.T_E2 = sample['t_E2']
        calc.P_R1 = sample['p_R1']
        calc.P_R2 = sample['p_R2']
        calc.T_R2 = sample['t_R2']

        try:
            # Calculate all three scenarios (suppress verbose output)
            results = calculate_func(df_bases, w_c, w_t, verbose=False)

            results_list.append({
                'C_A': results['A']['cost'],
                'T_A': results['A']['time'],
                'Z_A': results['A']['objective'],
                'C_B': results['B']['cost'],
                'T_B': results['B']['time'],
                'Z_B': results['B']['objective'],
                'C_C': results['C']['cost'],
                'T_C': results['C']['time'],
                'Z_C': results['C']['objective']
            })

        except Exception as e:
            # If calculation fails, use NaN
            if not TQDM_AVAILABLE and (idx + 1) % 100 == 0:
                print(f"  Warning: Sample {idx} failed - {str(e)[:50]}")
            results_list.append({
                'C_A': np.nan, 'T_A': np.nan, 'Z_A': np.nan,
                'C_B': np.nan, 'T_B': np.nan, 'Z_B': np.nan,
                'C_C': np.nan, 'T_C': np.nan, 'Z_C': np.nan
            })

        # Restore original parameters
        for key, value in original_params.items():
            setattr(calc, key, value)

    df_results = pd.DataFrame(results_list)

    # Remove NaN rows
    n_failed = df_results.isna().any(axis=1).sum()
    if n_failed > 0:
        print(f"\n⚠ Removed {n_failed:,} failed samples ({n_failed/len(results_list)*100:.2f}%)")
        df_results = df_results.dropna()

    print(f"\n✓ Monte Carlo simulation completed successfully")
    print(f"  Valid samples: {len(df_results):,}/{len(results_list):,}")
    print(f"{'='*70}\n")

    return df_results


# ============================================================================
# Statistical Analysis (Section 8.2, Step 3)
# ============================================================================

def calculate_statistics(df_results):
    """
    Calculate statistical metrics for Monte Carlo results.

    Returns:
        dict: Statistics for each scenario (mean, median, std, CV, CI)
    """
    stats = {}

    for scenario in ['A', 'B', 'C']:
        C_col = f'C_{scenario}'
        T_col = f'T_{scenario}'
        Z_col = f'Z_{scenario}'

        stats[scenario] = {
            'cost': {
                'mean': df_results[C_col].mean(),
                'median': df_results[C_col].median(),
                'std': df_results[C_col].std(),
                'cv': df_results[C_col].std() / df_results[C_col].mean(),
                'ci_90': [df_results[C_col].quantile(0.05), df_results[C_col].quantile(0.95)],
                'ci_95': [df_results[C_col].quantile(0.025), df_results[C_col].quantile(0.975)]
            },
            'time': {
                'mean': df_results[T_col].mean(),
                'median': df_results[T_col].median(),
                'std': df_results[T_col].std(),
                'cv': df_results[T_col].std() / df_results[T_col].mean(),
                'ci_90': [df_results[T_col].quantile(0.05), df_results[T_col].quantile(0.95)],
                'ci_95': [df_results[T_col].quantile(0.025), df_results[T_col].quantile(0.975)]
            },
            'objective': {
                'mean': df_results[Z_col].mean(),
                'median': df_results[Z_col].median(),
                'std': df_results[Z_col].std(),
                'cv': df_results[Z_col].std() / df_results[Z_col].mean()
            }
        }

    return stats


# ============================================================================
# Sobol Sensitivity Analysis (Section 8.2, Step 4)
# ============================================================================

def generate_sobol_samples(n_samples=N_SAMPLES, seed=RANDOM_SEED):
    """
    Generate Sobol samples for sensitivity analysis.

    Sobol analysis requires N*(2D+2) samples where D=7 parameters.
    For accurate results, use n_samples = base sample size.

    Args:
        n_samples (int): Base sample size
        seed (int): Random seed

    Returns:
        tuple: (problem definition, sobol samples array)
    """
    np.random.seed(seed)

    bounds = np.array(list(PARAM_BOUNDS.values()))

    problem = {
        'num_vars': 7,
        'names': list(PARAM_BOUNDS.keys()),
        'bounds': bounds
    }

    # Generate Sobol samples (requires SALib)
    from SALib.sample import saltelli
    sobol_samples = saltelli.sample(problem, n_samples, calc_second_order=False)

    print(f"✓ Generated {len(sobol_samples):,} Sobol samples")
    print(f"  Formula: N × (2D + 2) = {n_samples:,} × (2×7 + 2) = {len(sobol_samples):,}")

    return problem, sobol_samples


def analyze_sobol_indices(problem, sobol_samples, output_values):
    """
    Calculate Sobol sensitivity indices.

    Args:
        problem (dict): Problem definition
        sobol_samples (np.array): Sobol sample matrix
        output_values (np.array): Output values (cost or time)

    Returns:
        dict: Sobol indices (S1: first-order, ST: total-order)
    """
    # Analyze using SALib
    Si = sobol.analyze(problem, output_values, calc_second_order=False)

    # Extract indices
    indices = {
        'S1': dict(zip(problem['names'], Si['S1'])),      # First-order indices
        'ST': dict(zip(problem['names'], Si['ST'])),      # Total-order indices
        'S1_conf': dict(zip(problem['names'], Si['S1_conf'])),  # Confidence intervals
        'ST_conf': dict(zip(problem['names'], Si['ST_conf']))
    }

    return indices


def run_sobol_analysis(df_bases, calculate_func, w_c=1/6, w_t=5/6, n_base=1000):
    """
    Run full Sobol sensitivity analysis.

    Args:
        df_bases (pd.DataFrame): Rocket base data
        calculate_func (function): Calculation function
        w_c (float): Cost weight
        w_t (float): Time weight
        n_base (int): Base sample size (total samples = n_base * (2*7+2) = n_base * 16)

    Returns:
        dict: Sobol indices for C_A, T_A, C_B, T_B, C_C, T_C
    """
    print(f"\n{'='*70}")
    print(f"Running Sobol sensitivity analysis (base n={n_base:,})")
    print(f"{'='*70}")

    problem, sobol_samples = generate_sobol_samples(n_base)

    # Convert to DataFrame
    df_sobol = pd.DataFrame(sobol_samples, columns=problem['names'])

    # Run simulation
    import calculate_time_with_failure as calc

    output_arrays = {
        'C_A': [], 'T_A': [],
        'C_B': [], 'T_B': [],
        'C_C': [], 'T_C': []
    }

    n_total = len(df_sobol)

    # Create progress bar
    if TQDM_AVAILABLE:
        iterator = tqdm(range(n_total), desc="Sobol Analysis", unit="sample",
                       ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        iterator = range(n_total)
        print(f"Processing {n_total:,} Sobol samples (progress updates every 1000 samples)...")

    for idx in iterator:
        if not TQDM_AVAILABLE and (idx + 1) % 1000 == 0:
            print(f"  [{idx + 1:,}/{n_total:,}] {(idx+1)/n_total*100:.1f}% complete")

        sample = df_sobol.iloc[idx]

        # Override parameters
        original_params = {
            'P_E1': calc.P_E1, 'BETA_E1': calc.BETA_E1,
            'P_E2': calc.P_E2, 'T_E2': calc.T_E2,
            'P_R1': calc.P_R1, 'P_R2': calc.P_R2, 'T_R2': calc.T_R2
        }

        calc.P_E1 = sample['p_E1']
        calc.BETA_E1 = sample['beta_E1']
        calc.P_E2 = sample['p_E2']
        calc.T_E2 = sample['t_E2']
        calc.P_R1 = sample['p_R1']
        calc.P_R2 = sample['p_R2']
        calc.T_R2 = sample['t_R2']

        try:
            results = calculate_func(df_bases, w_c, w_t, verbose=False)
            output_arrays['C_A'].append(results['A']['cost'])
            output_arrays['T_A'].append(results['A']['time'])
            output_arrays['C_B'].append(results['B']['cost'])
            output_arrays['T_B'].append(results['B']['time'])
            output_arrays['C_C'].append(results['C']['cost'])
            output_arrays['T_C'].append(results['C']['time'])

        except:
            # Use mean value for failed samples
            output_arrays['C_A'].append(np.nan)
            output_arrays['T_A'].append(np.nan)
            output_arrays['C_B'].append(np.nan)
            output_arrays['T_B'].append(np.nan)
            output_arrays['C_C'].append(np.nan)
            output_arrays['T_C'].append(np.nan)

        # Restore parameters
        for key, value in original_params.items():
            setattr(calc, key, value)

    # Convert to arrays and handle NaN
    for key in output_arrays:
        arr = np.array(output_arrays[key])
        # Replace NaN with median
        if np.any(np.isnan(arr)):
            median_val = np.nanmedian(arr)
            arr[np.isnan(arr)] = median_val
        output_arrays[key] = arr

    # Analyze Sobol indices
    sobol_results = {}
    print(f"\n{'='*70}")
    print(f"Computing Sobol indices for 6 output metrics...")
    print(f"{'='*70}\n")

    for output_name, output_vals in output_arrays.items():
        print(f"  ✓ Analyzing {output_name}...")
        sobol_results[output_name] = analyze_sobol_indices(problem, sobol_samples, output_vals)

    print(f"\n✓ Sobol sensitivity analysis completed")
    print(f"{'='*70}\n")

    return sobol_results


# ============================================================================
# Main Function
# ============================================================================

def perform_sensitivity_analysis(df_bases, calculate_func, w_c=1/6, w_t=5/6,
                                   run_mc=True, run_sobol=True,
                                   n_mc=10000, n_sobol_base=1000):
    """
    Perform complete sensitivity analysis (Monte Carlo + Sobol).

    Args:
        df_bases (pd.DataFrame): Rocket base data
        calculate_func (function): Calculation function
        w_c (float): Cost weight
        w_t (float): Time weight
        run_mc (bool): Whether to run Monte Carlo
        run_sobol (bool): Whether to run Sobol analysis
        n_mc (int): Monte Carlo sample size
        n_sobol_base (int): Sobol base sample size

    Returns:
        dict: Complete results including MC data, statistics, and Sobol indices
    """
    results = {}

    if run_mc:
        print("\n" + "="*70)
        print("MONTE CARLO SIMULATION")
        print("="*70)
        df_samples, problem = generate_lhs_samples(n_mc)
        df_mc_results = run_monte_carlo_simulation(df_samples, df_bases, calculate_func, w_c, w_t)
        stats = calculate_statistics(df_mc_results)

        results['mc_samples'] = df_samples
        results['mc_results'] = df_mc_results
        results['statistics'] = stats

    if run_sobol:
        print("\n" + "="*70)
        print("SOBOL SENSITIVITY ANALYSIS")
        print("="*70)
        sobol_indices = run_sobol_analysis(df_bases, calculate_func, w_c, w_t, n_sobol_base)
        results['sobol_indices'] = sobol_indices

    return results


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    # Create dummy data for testing
    import calculate_time_with_failure as calc

    np.random.seed(42)
    df_test = pd.DataFrame({
        'Initial_Frequency': np.random.uniform(10, 30, 10),
        'Growth_Rate': np.random.uniform(1, 3, 10),
        'Fixed_Cost': np.random.uniform(1e8, 5e8, 10)
    })

    # Run small test
    results = perform_sensitivity_analysis(
        df_test,
        calc.calculate_metrics_with_failure,
        run_mc=True,
        run_sobol=False,  # Skip Sobol for quick test
        n_mc=100
    )

    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)

    for scenario in ['A', 'B', 'C']:
        print(f"\nScenario {scenario}:")
        print(f"  Cost: ${results['statistics'][scenario]['cost']['mean']/1e12:.2f} ± "
              f"{results['statistics'][scenario]['cost']['std']/1e12:.2f} trillion")
        print(f"  Time: {results['statistics'][scenario]['time']['mean']:.2f} ± "
              f"{results['statistics'][scenario]['time']['std']:.2f} years")
        print(f"  CV (Cost): {results['statistics'][scenario]['cost']['cv']:.4f}")
        print(f"  CV (Time): {results['statistics'][scenario]['time']['cv']:.4f}")
