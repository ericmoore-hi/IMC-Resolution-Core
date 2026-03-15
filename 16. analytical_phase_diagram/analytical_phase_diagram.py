"""
Script: Analytical Phase Transition Analysis
-------------------------------------------------------------------------------------------------
Description: This script visualizes the theoretical comparison between
classical phase transition (Sharp Freeze) and the proposed IMC
tunneling-based dynamics.

Note: This is an analytical model used to visualize the smoothing effect
of inertia in the Hard-SAT region as predicted by IMC equations.
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

import matplotlib.pyplot as plt
import numpy as np

def plot_theoretical_phase_transition():
    # 1. Parameter range (Constraint density alpha = M/N)
    alpha_values = np.linspace(3.0, 6.0, 100)

    # 2. MATHEMATICAL MODELS

    # Classical Model: Sharp first-order phase transition
    # Modeled using a high-gradient sigmoid to represent the algorithmic "freeze"
    p_classical = 1.0 / (1.0 + np.exp(12 * (alpha_values - 4.26)))

    # IMC Model: Smoothed transition due to inertial tunneling
    # Represents the prediction that inertial dynamics bypasses the sharp
    # complexity barrier near alpha_c.
    p_imc = 1.0 / (1.0 + np.exp(3.5 * (alpha_values - 4.8)))

    # 3. Visualization
    plt.figure(figsize=(10, 6))

    # Plot Classical Baseline
    plt.plot(alpha_values, p_classical, 'k--', linewidth=2, alpha=0.6,
             label='Classical Limit (Sharp Freeze)')

    # Plot IMC Theoretical Dynamics
    plt.plot(alpha_values, p_imc, 'b-', linewidth=3,
             label='IMC Theoretical Prediction (Tunneling)')

    # Critical Threshold Reference Line
    plt.axvline(x=4.26, color='r', linestyle=':', label=r'Critical Threshold $\alpha_c \approx 4.26$')

    # Highlight Hard-SAT Region
    plt.fill_between(alpha_values, 0, 1,
                     where=((alpha_values >= 4.0) & (alpha_values <= 4.5)),
                     color='red', alpha=0.1)
    plt.text(4.28, 0.6, 'Hard-SAT Region', color='red', rotation=90, fontweight='bold')

    # Plot Aesthetics
    plt.xlabel(r'Constraint Density $\alpha = M/N$', fontsize=12, fontweight='bold')
    plt.ylabel(r'Theoretical Probability of Solution $P(SAT)$', fontsize=12, fontweight='bold')
    plt.title('Analytical Comparison: Classical vs. IMC Dynamics', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.3)

    # Export output
    output_file = 'Phase_Transition_Diagram.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Analytical phase transition plot generated: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_theoretical_phase_transition()
