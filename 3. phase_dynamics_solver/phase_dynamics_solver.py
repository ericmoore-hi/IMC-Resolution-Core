"""
Description:
    Generates Figure 3 illustrating the spectral distinction between SAT (convergent)
    and UNSAT (limit cycle) regimes using numerical integration of the dynamical equations.
-------------------------------------------------------------------------------------------------
Copyright (c) 2025-2026 Eric Moore. All rights reserved.
Author: Eric Moore (orcid.org/0009-0002-1826-0815)
Project: Inertial Manifold Computing (IMC)
-------------------------------------------------------------------------------------------------
This code is part of the research manuscript:
"Deterministic 3-SAT Resolution via Continuous Physical Dynamics in Inertial Manifold 
Computing and its Empirical Robustness to Finite Precision Limits"
-------------------------------------------------------------------------------------------------
LICENSE: For academic and non-commercial research use only.
Commercial use or redistribution prohibited without written permission.
-------------------------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint

# --- Configuration & Constants ---
TIME_MAX = 20.0
STEPS = 1000
NOISE_STD = 0.05
RANDOM_SEED = 42


def sat_potential_gradient(phi, t):
    """
    Dynamics for a satisfiable clause (SAT regime).
    Represents gradient descent on a simple potential V = 1 - cos(phi).
    The system naturally converges to the fixed point (attractor).
    """
    gamma = 1.0  # Damping/relaxation rate
    dphi_dt = -gamma * np.sin(phi)
    return dphi_dt


def unsat_frustrated_dynamics(phi, t):
    """
    Dynamics for an unsatisfiable loop (UNSAT regime).
    Models a 3-node frustrated cycle where no global fixed point exists.
    phi: Vector of phase variables [phi1, phi2, phi3].
    """
    # Coupling strength and self-decay
    k = 1.0
    decay = 0.5

    p1, p2, p3 = phi

    # Cyclic frustration logic (resembling Rock-Paper-Scissors topology)
    dp1 = -k * np.sin(p1 - p2 + np.pi) - decay * p1
    dp2 = -k * np.sin(p2 - p3 + np.pi) - decay * p2
    dp3 = -k * np.sin(p3 - p1 + np.pi) - decay * p3

    return [dp1, dp2, dp3]


def generate_plot():
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Time vector
    t = np.linspace(0, TIME_MAX, STEPS)

    # --- 1. Simulation: SAT Case ---
    phi0_sat = [3.0]  # Initial phase away from equilibrium
    sol_sat = odeint(sat_potential_gradient, phi0_sat, t)

    # Use cosine metric to visualize energy minimization (Proxy for V)
    signal_sat = np.cos(sol_sat[:, 0])

    # Add thermal noise floor
    signal_sat += np.random.normal(0, NOISE_STD, len(t))

    # --- 2. Simulation: UNSAT Case ---
    phi0_unsat = [0.1, 2.0, 4.0]  # Arbitrary initial states
    sol_unsat = odeint(unsat_frustrated_dynamics, phi0_unsat, t)

    # Vizualize the oscillation of the first node
    signal_unsat = np.sin(sol_unsat[:, 0] * 2.0)

    # Add thermal noise floor
    signal_unsat += np.random.normal(0, NOISE_STD, len(t))

    # --- Plotting ---
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 11
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Plot A: SAT
    ax1.plot(t, signal_sat, color='#2ca02c', linewidth=2.0, label='System State')
    ax1.set_title('(a) SAT Regime: Stable Resonance (DC) - Solution Converged')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.4, label='Target Attractor')

    # Annotation for SAT
    ax1.text(12, 0.5, "Converged State\n(Fixed Point)",
             fontsize=10, verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#2ca02c', boxstyle='round,pad=0.5'))
    ax1.legend(loc='lower right', frameon=True)
    ax1.set_ylim(-1.5, 1.5)

    # Plot B: UNSAT
    ax2.plot(t, signal_unsat, color='#d62728', linewidth=2.0, label='System State')
    ax2.set_title('(b) UNSAT Regime: Limit Cycle Oscillation (AC) - No Solution')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Amplitude (a.u.)')

    # Annotation for UNSAT
    ax2.text(12, 0.0, "Persistent Oscillation\n(Topological Limit Cycle)",
             fontsize=10, verticalalignment='center',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='#d62728', boxstyle='round,pad=0.5'))
    ax2.legend(loc='lower right', frameon=True)
    ax2.set_ylim(-1.5, 1.5)

    plt.tight_layout()

    # Save output
    output_filename = "3. imc_resonance_signature_v1.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure generated successfully: {output_filename}")


if __name__ == "__main__":
    generate_plot()
