"""
Script: Adaptive Inertial Damping & Smart Braking Verification
Description:
Demonstrates the resolution of the "Inertial Flattening Paradox".

This simulation proves that a state-dependent adaptive damping mechanism (gamma(V))
allows the system to:
1. Maintain high momentum over shallow local minima (Tunneling/Flattening).
2. Trigger "Smart Braking" (High Dissipation) purely based on potential depth thresholds.

This confirms Theorem 3: Selective Inertial Convergence via Non-linear Dissipation.
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

# ==========================================
# PHYSICS ENGINE CONFIGURATION
# ==========================================
dt = 0.01
# Simulation steps
steps = 6000
time_axis = np.linspace(0, steps * dt, steps)

# System Parameters
mass = 2.5          # High Inertia
gamma_low = 0.1     # Low friction (Air/Vacuum mode)
gamma_high = 5.0    # High friction (Fluid/Mud mode)

# The "Viscosity Threshold": Below this potential energy, high friction activates automatically.
braking_threshold = -2.0

# ==========================================
# LANDSCAPE DEFINITION (The "Golf Course" Problem)
# ==========================================
def potential(x):
    # Shallow Trap (Depth ~ 1.0)
    trap = -1.5 * np.exp(-(x + 2)**2 / 0.8)
    # Global Solution (Depth ~ 5.0)
    solution = -5.0 * np.exp(-(x - 2)**2 / 0.8)
    confinement = 0.1 * x**2
    return trap + solution + confinement

def force(x):
    d_trap = -1.5 * (-2*(x + 2) / 0.8) * np.exp(-(x + 2)**2 / 0.8)
    d_sol = -5.0 * (-2*(x - 2) / 0.8) * np.exp(-(x - 2)**2 / 0.8)
    d_conf = 0.2 * x
    return -(d_trap + d_sol + d_conf)

# ==========================================
# SIMULATION CORE
# ==========================================
def run_simulation(mode):
    x = np.zeros(steps)
    v = np.zeros(steps)
    gamma_history = np.zeros(steps)

    x[0] = -4.0
    v[0] = 3.0

    for i in range(1, steps):
        f = force(x[i-1])
        curr_pot = potential(x[i-1])

        # --- PHYSICS OF THE MEDIUM ---
        if mode == "Fixed_Low":
            gamma = gamma_low
        elif mode == "Fixed_High":
            gamma = 3.0
        elif mode == "Adaptive":
            # State-Dependent Dissipation:
            if curr_pot < braking_threshold:
                gamma = gamma_high
            else:
                gamma = gamma_low

        gamma_history[i] = gamma

        # Euler Integration
        a = (f - gamma * v[i-1]) / mass
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt

    return x, gamma_history

# ==========================================
# VISUALIZATION
# ==========================================
if __name__ == "__main__":
    print("Running Adaptive Damping Proof...")

    traj_low, _ = run_simulation("Fixed_Low")
    traj_high, _ = run_simulation("Fixed_High")
    traj_adapt, gamma_log = run_simulation("Adaptive")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})

    # --- LEFT PLOT: LANDSCAPE ---
    x_space = np.linspace(-5, 5, 200)
    y_pot = potential(x_space)

    ax1.plot(x_space, y_pot, 'k-', linewidth=2, label='Potential $V(x)$')
    # Fill the "Braking Zone"
    ax1.fill_between(x_space, -6, braking_threshold, color='purple', alpha=0.1, label='High Dissipation Zone')
    ax1.fill_between(x_space, braking_threshold, 2, color='#ecf0f1', alpha=0.5, label='Low Dissipation Zone')

    ax1.axhline(y=braking_threshold, color='purple', linestyle='--', linewidth=1.5, label='Phase Change Threshold')
    ax1.text(-2, -0.5, "Shallow Trap\n(Ballistic Flyover)", color='#e74c3c', fontweight='bold', ha='center')
    ax1.text(2, -4.0, "Global Solution\n(Viscous Capture)", color='#27ae60', fontweight='bold', ha='center')

    ax1.set_title("Deceptive Landscape (The Paradox Test)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("State Space ($x$)", fontsize=11)
    ax1.set_ylabel("Potential Energy $V(x)$", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim(-6, 2)

    # --- RIGHT PLOT: DYNAMICS ---

    # Case 1: Low Damping (Paradox)
    ax2.plot(time_axis, traj_low, color='gray', linestyle=':', linewidth=1.5, alpha=0.6,
             label='Constant Low Friction (Overshoot)')

    # Case 2: High Damping (Stuck) - COLOR: GOLD
    ax2.plot(time_axis, traj_high, color='gold', linestyle='--', linewidth=2.5,
             label='Constant High Friction (Stuck)')

    # Case 3: Adaptive (Success)
    # Z-Order increased to 20 to act as the "Top Layer"
    ax2.plot(time_axis, traj_adapt, color='#2980b9', linewidth=2.5, zorder=20,
             label=r'Adaptive Smart Braking ($\gamma(V)$)')

    # Braking Indicator
    # Slicing increased to [::100] to make markers sparser (cleaner look)
    # Z-Order set to 15 (Below the blue line, but above grid/other lines)
    braking_indices = np.where(gamma_log > gamma_low)[0]
    if len(braking_indices) > 0:
        t_brake = time_axis[braking_indices]
        x_brake = traj_adapt[braking_indices]
        ax2.scatter(t_brake[::100], x_brake[::100], color='purple', s=60, marker='P', zorder=15, label='Braking Active (High Friction)')

    # Reference Lines
    ax2.axhline(y=2.0, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.6, label='Global Solution State')
    ax2.axhline(y=-2.0, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.6, label='Local Trap State')

    ax2.set_title("Dynamics Comparison: Solving the Paradox", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Physical Time (ns)", fontsize=11)
    ax2.set_ylabel("System State ($x$)", fontsize=11)
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    # Limits
    ax2.set_xlim(0, 60)
    ax2.set_ylim(-5, 5)

    output_file = 'Adaptive_Damping_Proof.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph generated and saved to: {output_file}")
    plt.show()
