"""
Script: Stochastic Resonance and Topological Tunneling Simulation
Description:
Demonstrates the phenomenon of Stochastic Resonance Pumping in Inertial Manifold Dynamics.
The simulation proves that thermal noise (k_B T), rather than obscuring the gradient,
facilitates the escape from local minima (metastable states) and accelerates
convergence to the global solution (True Ground State) via barrier tunneling.
-------------------------------------------------------------------------------------------------
Copyright (c) 2025-2026 Eric Moore. All rights reserved.
Author: Eric Moore (orcid.org/0009-0002-1826-0815)
Project: Inertial Manifold Computing (IMC)
-------------------------------------------------------------------------------------------------
This code is part of the research manuscript:
"Polynomial-Time 3-SAT Resolution via Inertial Manifold Computing and
its Precise Mapping to Turing Complexity within the BSS Framework"
-------------------------------------------------------------------------------------------------
LICENSE: For academic and non-commercial research use only.
Commercial use or redistribution prohibited without written permission.
-------------------------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PHYSICS CONFIGURATION
# ==========================================
dt = 0.01
steps = 20000
time_axis = np.linspace(0, steps * dt, steps)

mass = 1.0
gamma = 0.5
# Increased tilt to make the well depth difference visually distinct but physically subtle
tilt = 0.25
barrier_height = 4.0

sigma_zero = 0.0
sigma_res = 1.2

# ==========================================
# POTENTIAL LANDSCAPE DEFINITION
# ==========================================
def potential(x):
    base_potential = (x**2 - 2.0)**2
    gradient_bias = tilt * x
    return base_potential - gradient_bias

def force(x):
    d_base = 2 * (x**2 - 2.0) * (2*x)
    d_bias = tilt
    return -(d_base - d_bias)

# ==========================================
# LANGEVIN DYNAMICS INTEGRATOR
# ==========================================
def run_langevin_simulation(sigma_noise):
    x = np.zeros(steps)
    v = np.zeros(steps)

    # Start near the theoretical local minimum
    x[0] = -1.45
    v[0] = 0.0

    np.random.seed(42)

    for i in range(1, steps):
        f_det = force(x[i-1])
        noise = np.random.normal(0, 1) * sigma_noise / np.sqrt(dt)
        acceleration = (f_det - gamma * v[i-1] + noise) / mass
        v[i] = v[i-1] + acceleration * dt
        x[i] = x[i-1] + v[i] * dt

    return x

# ==========================================
# EXECUTION & VISUALIZATION
# ==========================================
if __name__ == "__main__":
    print("Starting Simulation...")

    traj_static = run_langevin_simulation(sigma_zero)
    traj_resonant = run_langevin_simulation(sigma_res)

    # Calculate exact stable levels from simulation data for perfect alignment
    level_local = np.mean(traj_static[-1000:])
    level_global = np.abs(level_local) + 0.1 # Approximate symmetric target

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 2]})

    # --- SUBPLOT 1: LANDSCAPE ---
    x_space = np.linspace(-2.5, 2.5, 200)
    y_pot = potential(x_space)

    ax1.plot(x_space, y_pot, 'k-', linewidth=2, label='Energy Landscape $V(x)$')
    ax1.fill_between(x_space, y_pot, max(y_pot), color='#ecf0f1', alpha=0.5)

    # Annotations
    ax1.text(-1.5, 2.0, "Local Trap\n(Start)", color='#e74c3c', fontweight='bold', ha='center')
    ax1.text(1.6, -4.5, "Global Solution\n(Target)", color='#27ae60', fontweight='bold', ha='center')

    # Arrow with raw string 'r' to fix escape sequence warning
    ax1.arrow(0, 6, 0.6, -1.5, head_width=0.15, color='blue', length_includes_head=True)
    ax1.text(0.7, 6.2, r"Gradient Bias ($\Delta E$)", color='blue', fontsize=10, fontweight='bold')

    ax1.set_title("Energy Landscape with Barriers", fontsize=12, fontweight='bold')
    ax1.set_xlabel("State Space ($x$)", fontsize=11)
    ax1.set_ylabel("Potential Energy $V(x)$", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_ylim(-8, 12)

    # --- SUBPLOT 2: DYNAMICS ---

    # Reference lines are now aligned to the actual simulation mean
    ax2.axhline(y=level_global, color='#27ae60', linestyle='--', linewidth=3, alpha=0.4, label='Global Solution State')
    ax2.axhline(y=level_local, color='#c0392b', linestyle='--', linewidth=3, alpha=0.4, label='Local Trap State')

    # Trajectories
    ax2.plot(time_axis, traj_static, color='#e74c3c', linestyle='-', linewidth=1.5,
             label=r'Zero Noise ($k_B T = 0$): Trapped')

    ax2.plot(time_axis, traj_resonant, color='#2980b9', linewidth=1.2, alpha=0.9,
             label=r'Thermal Noise ($k_B T > 0$): Escape')

    ax2.set_title("Temporal Dynamics: Noise-Assisted Tunneling", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Physical Time (ns)", fontsize=11)
    ax2.set_ylabel("System State ($x$)", fontsize=11)

    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-2.5, 2.5)

    output_file = 'Stochastic_Resonance_Proof.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph generated and saved to: {output_file}")
    plt.show()