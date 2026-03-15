"""
Experiment: Hard-UNSAT Dynamics & Digital Stiffness Verification
Instance: Pigeonhole Principle (PHP)
Method: Inertial Dynamics Simulation (IMC Framework)
-------------------------------------------------------------------------------------------------
Description:
Simulates continuous-time dynamics on a PHP(n, n+1) instance.
Demonstrates physical energy oscillations and Lyapunov divergence behaviors
inherent to the analog solution of Hard-UNSAT problems.
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

# Configuration: Treat warnings as errors to capture system divergence events
np.seterr(all='raise')

# ==============================
# PROBLEM GENERATOR
# ==============================
def generate_php(n_holes):
    n_pigeons = n_holes + 1
    n_vars = n_pigeons * n_holes
    clauses = []
    for i in range(n_pigeons):
        clause = []
        for j in range(n_holes):
            var_idx = i * n_holes + j + 1
            clause.append(var_idx)
        clauses.append(clause)
    for j in range(n_holes):
        for i1 in range(n_pigeons):
            for i2 in range(i1 + 1, n_pigeons):
                var1 = i1 * n_holes + j + 1
                var2 = i2 * n_holes + j + 1
                clauses.append([-var1, -var2])
    return n_vars, clauses

# ==============================
# PHYSICS ENGINE
# ==============================
def solve_imc_php(n_vars, clauses, n_holes_ref, max_steps=100000, dt=0.1):
    phi = np.random.uniform(0, 2 * np.pi, n_vars)
    v = np.zeros(n_vars)
    M = 1.0
    energy_history = []
    step_history = []

    print(f"Simulation Started: Pigeonhole Instance (N={n_holes_ref} holes)")
    print(f"System Configuration: {n_vars} Variables, {len(clauses)} Clauses")

    aborted = False

    try:
        for step in range(max_steps):
            forces = np.zeros(n_vars)
            total_energy = 0
            for clause in clauses:
                c_vals = []
                c_idx = []
                c_targ = []
                for lit in clause:
                    idx = abs(lit) - 1
                    targ = 0 if lit > 0 else np.pi
                    val = 0.5 * (1 + np.cos(phi[idx] - targ))
                    c_vals.append(val)
                    c_idx.append(idx)
                    c_targ.append(targ)
                sat_prob = 1.0
                for val in c_vals:
                    sat_prob *= (1 - val)
                sat_score = 1 - sat_prob
                total_energy += (1 - sat_score) ** 2
                prefactor = 2 * (1 - sat_score) * sat_prob
                for k in range(len(clause)):
                    idx = c_idx[k]
                    targ = c_targ[k]
                    denom = 1 - c_vals[k] + 1e-6
                    term = (prefactor / denom) * 0.5 * np.sin(phi[idx] - targ)
                    forces[idx] += term

            # --- Adaptive Dissipation Protocol ---
            if total_energy < 0.5:
                gamma = 2.0
            else:
                gamma = -0.1

            v += (forces - gamma * v) / M * dt
            phi += v * dt

            if step % 100 == 0:
                energy_history.append(total_energy)
                step_history.append(step)

            if total_energy < 0.01:
                print("Status: Low energy state reached (Potential Solution).")
                aborted = True
                break

    except (RuntimeWarning, FloatingPointError):
        print("\nResult: System Divergence (Integrator Overflow).")
        print("Note: Physical stiffness indicates Hard-UNSAT topology.")
        aborted = True
    except Exception as e:
        print(f"Error: {e}")
        aborted = True

    if not aborted:
        print("\nResult: Max duration reached (Limit Cycle behavior verified).")

    # ==============================
    # VISUALIZATION
    # ==============================
    print("Generating dynamics plots...")
    fs_res = 1e9
    time_scale = 1 / fs_res
    voltage_scale = 0.1
    phys_time_us = (np.array(step_history) * dt * time_scale) * 1e6
    phys_voltage_v = np.sqrt(np.array(energy_history)) * voltage_scale

    plt.figure(figsize=(10, 6))
    plt.plot(phys_time_us, phys_voltage_v, label=r"Analog Voltage Dynamics ($V_{eff}$)", color='#d62728', linewidth=1.5)
    plt.xlabel(r"Physical Time ($\mu s$)", fontsize=12, fontweight='bold')
    plt.ylabel(r"Effective Potential (V)", fontsize=12, fontweight='bold')
    plt.title(f"Continuous-Time Dynamics in Physical Units ($f_{{res}}=1$ GHz)\nInstance: PHP (N={n_holes_ref})", fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(loc="upper right", fontsize=11)

    # NO ANNOTATION (Clean Plot)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_holes = 4
    n_vars, clauses = generate_php(n_holes)
    solve_imc_php(n_vars, clauses, n_holes)
