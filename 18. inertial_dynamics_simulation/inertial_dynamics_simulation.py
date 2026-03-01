"""
Script: Inertial Dynamics Simulation
Description:
Demonstrates the effect of inertial mass on barrier crossing in an adversarial
potential landscape.
- High Mass (Red): Preserves momentum to overcome potential barriers.
- Low Mass (Blue): Dissipates momentum quickly and becomes trapped in local minima.
- Simulation uses identical initial conditions (v0) for both particles.
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

# ==============================
# PHYSICS ENGINE & POTENTIAL
# ==============================
def adversarial_potential(x):
    """
    Defines the potential energy surface V(x).
    Combines a global quadratic well with localized Gaussian traps and barriers.
    """
    global_well = 1.5 * x ** 2
    local_trap = -6.0 * np.exp(-((x + 2.8) ** 2) / 0.12)
    barrier = 2.5 * np.exp(-((x + 1.2) ** 2) / 0.2)
    return global_well + local_trap + barrier


def gradient(x):
    """
    Calculates the gradient of the potential (Force = -dV/dx).
    """
    return (
            3.0 * x
            + 100.0 * (x + 2.8) * np.exp(-((x + 2.8) ** 2) / 0.12)
            - 25.0 * (x + 1.2) * np.exp(-((x + 1.2) ** 2) / 0.2)
    )


def run_simulation():
    # 1. SIMULATION PARAMETERS
    dt = 0.001          # Time step
    steps = 100000      # Duration allows for full convergence/relaxation
    gamma = 0.8         # Damping coefficient for stabilization

    # 2. INITIAL CONDITIONS
    # Both particles start with the same velocity to ensure a fair comparison based on mass.
    v0_common = 4.0

    masses = [0.15, 6.0]
    colors = ['#3498db', '#e74c3c']
    labels = [
        r'Low Inertia ($M=0.15$) — Trapped',
        r'High Inertia ($M=6.0$) — Escapes & Converges'
    ]

    plt.figure(figsize=(10, 6))

    # 3. DYNAMICS LOOP
    max_time = steps * dt

    for M, col, lab in zip(masses, colors, labels):
        x = -2.8        # Identical starting position
        v = v0_common   # Identical starting velocity

        trajectory = []

        for _ in range(steps):
            # Langevin dynamics (deterministic part): F = -grad(V) - gamma*v
            force = -gradient(x) - gamma * v
            a = force / M

            # Euler integration
            v += a * dt
            x += v * dt
            trajectory.append(x)

        # Time axis generation
        t = np.arange(len(trajectory)) * dt
        plt.plot(t, trajectory, color=col, linewidth=3, label=lab)

    # 4. REFERENCE: GLOBAL SOLUTION
    # Dashed line indicates the theoretical target (x=0)
    plt.plot([0, max_time], [0, 0], color='#27ae60', linestyle='--',
             linewidth=3, label='Global Solution ($x=0$)')

    # 5. VISUALIZATION SETTINGS
    plt.title('Inertial Manifold Dynamics: Momentum-Driven Escape',
              fontsize=14, fontweight='bold')
    plt.xlabel('Physical Time (ns)', fontsize=12, fontweight='bold')
    plt.ylabel('System State Vector ($x$)', fontsize=12, fontweight='bold')

    # Annotations for clarity
    plt.text(5, -2.6, 'Stuck (Low Inertia)', color='#3498db',
             fontsize=11, fontweight='bold')
    plt.text(40, 0.2, 'Converged (High Inertia)', color='#e74c3c',
             fontsize=11, fontweight='bold')

    plt.legend(loc='center right', framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Limits
    plt.ylim(-3.5, 1.5)
    plt.xlim(0, max_time)

    # Save output
    output_filename = 'Inertial_Manifold_SUCCESS.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph successfully generated: {output_filename}")
    plt.show()


if __name__ == "__main__":
    run_simulation()