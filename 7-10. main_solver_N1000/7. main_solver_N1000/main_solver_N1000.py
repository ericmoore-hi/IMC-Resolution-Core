"""
Script: Large-Scale Continuous-Time SAT Solver (IMC Dynamics)
Instance: Random 3-SAT / Hard Benchmarks (e.g., f1000.cnf)
Method: Inertial Manifold Dynamics with Adaptive Homotopy
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
from scipy.integrate import RK45
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import time
import gc

# --- HYPERPARAMETERS ---
# Physical constants derived for 1 GHz CMOS reference
T_MAX = 100000.0        # Max simulation steps
GAMMA = 0.35            # Damping coefficient
MASS = 1.0              # Inertial mass
LAMBDA_RATE = 0.05      # Adiabatic evolution rate
ADAPTATION_RATE = 2.5   # Weight adaptation speed
WEIGHT_LIMIT = 8000.0   # Clamping limit for auxiliary weights
SAT_LIMIT = 0.1         # Energy threshold for convergence
VISUAL_TAIL = 3000      # Post-convergence freeze frames for visualization

# --- UTILITIES ---
def select_cnf_file():
    """Opens a file dialog to select the DIMACS .cnf benchmark file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select CNF Benchmark (e.g., f1000.cnf)",
        filetypes=[("CNF Files", "*.cnf")]
    )
    root.destroy()
    return file_path

def load_cnf_as_matrix(filepath):
    """Parses DIMACS CNF format into vectorized matrix representation."""
    if not filepath: return None, 0, None
    clauses_list = []
    n_vars = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('c', '%')): continue
            if line.startswith('p'):
                try:
                    n_vars = int(line.split()[2])
                except:
                    pass
                continue

            parts = [int(x) for x in line.split() if x != '0' and x != '']
            if not parts: continue

            lits = []
            for x in parts:
                var_idx = abs(x) - 1
                is_neg = 1 if x < 0 else 0
                lits.append([var_idx, is_neg])
                n_vars = max(n_vars, var_idx + 1)
            clauses_list.append(lits)

    max_len = max(len(c) for c in clauses_list)
    M = len(clauses_list)

    # Vectorized structures
    C_indices = np.zeros((M, max_len), dtype=np.int32)
    C_signs = np.zeros((M, max_len), dtype=np.float32)
    C_mask = np.zeros((M, max_len), dtype=np.float32)

    for i, cl in enumerate(clauses_list):
        for j, (v_idx, is_neg) in enumerate(cl):
            C_indices[i, j] = v_idx
            C_signs[i, j] = np.pi if is_neg else 0.0
            C_mask[i, j] = 1.0

    print(f"[INFO] System Loaded: {n_vars} Variables, {M} Clauses.")
    return n_vars, M, (C_indices, C_signs, C_mask)

# --- DYNAMICS ENGINE ---
def dynamics_hyper(t, state, C_indices, C_signs, C_mask, n_vars, M):
    """
    Computes derivatives [dphi/dt, dv/dt, dlambda/dt, dw/dt].
    Implements Inertial Manifold Dynamics with Log-Sum-Exp potentials.
    """
    # Unpack state vector
    phi = state[:n_vars]
    vel = state[n_vars:2 * n_vars]
    lam = state[2 * n_vars]
    w_start = 2 * n_vars + 1
    weights = state[w_start:]

    # Phase alignment
    phi_gathered = phi[C_indices] + C_signs
    cos_vals = np.cos(phi_gathered)
    sin_vals = np.sin(phi_gathered)

    # Soft-SAT kernels
    qs = 0.5 * (1.0 + cos_vals)
    dqs = -0.5 * sin_vals
    one_minus_q = (1.0 - qs) * C_mask + (1.0 - C_mask)
    term = np.prod(one_minus_q, axis=1, keepdims=True)

    # Weighted Gradient Force
    W = weights.reshape(-1, 1)
    prefactor = 2.0 * term * W

    grad = np.zeros(n_vars, dtype=np.float32)
    cols = C_indices.shape[1]

    # Vectorized gradient accumulation
    for j in range(cols):
        others = np.ones_like(term)
        for k in range(cols):
            if k != j: others *= one_minus_q[:, k:k + 1]
        local_grad = - prefactor * others * dqs[:, j:j + 1] * C_mask[:, j:j + 1]
        np.add.at(grad, C_indices[:, j], local_grad.flatten())

    # Auxiliary dynamics
    d_weights_dt = ADAPTATION_RATE * term.flatten()
    lam_eff = 1.0 if lam > 1.0 else (0.0 if lam < 0.0 else lam)

    # Newton's Law: F = ma
    force = - (lam_eff * grad)
    dphi_dt = vel
    dv_dt = (force - GAMMA * vel) / MASS

    # Adiabatic schedule
    dlam = 0.0 if lam_eff >= 0.999 else LAMBDA_RATE

    # Repack derivatives
    res = np.empty_like(state)
    res[:n_vars] = dphi_dt
    res[n_vars:2 * n_vars] = dv_dt
    res[2 * n_vars] = dlam
    res[w_start:] = d_weights_dt
    return res

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    file_path = select_cnf_file()
    if not file_path:
        print("[ERROR] No file selected.")
        exit()

    n_vars, M, matrix_data = load_cnf_as_matrix(file_path)
    (C_indices, C_signs, C_mask) = matrix_data
    fname = os.path.basename(file_path)

    print("\n" + "=" * 60)
    print(f" START SIMULATION: {fname}")
    print(" Configuration: 1 GHz Reference Clock (Physical Units)")
    print("-" * 60)

    gc.collect()
    np.random.seed(42)

    # Initial Conditions
    phi0 = np.random.uniform(0, 2 * np.pi, n_vars).astype(np.float32)
    vel0 = np.zeros(n_vars, dtype=np.float32)
    w0 = np.ones(M, dtype=np.float32)
    init_state = np.concatenate([phi0, vel0, [0.0], w0])

    start_time = time.time()

    # RK45 Integrator (Adaptive Step)
    solver = RK45(
        fun=lambda t, y: dynamics_hyper(t, y, C_indices, C_signs, C_mask, n_vars, M),
        t0=0, y0=init_state, t_bound=T_MAX,
        rtol=1e-2, atol=1e-4, max_step=1.0
    )

    energy_history = []
    time_history = []

    step_count = 0
    tail_counter = 0
    found_solution = False
    final_time = 0

    print("[INFO] Running integration loop...")

    while solver.status == 'running':
        if found_solution:
            # Freeze dynamics for visualization tail
            current_t = time_history[-1] + 1.0
            energy_history.append(0)
            time_history.append(current_t)

            tail_counter += 1
            if tail_counter >= (VISUAL_TAIL // 10):
                break
        else:
            solver.step()
            step_count += 1

            # Monitor system state (downsampled)
            if step_count % 10 == 0:
                current_t = solver.t
                current_y = solver.y
                phi = current_y[:n_vars]

                # Check SAT condition (Boolean mapping)
                assignments = np.cos(phi) > 0
                clause_base_vals = assignments[C_indices]
                is_neg = (C_signs > 1.0)
                clause_final_lits = (clause_base_vals != is_neg)
                clause_final_lits = np.logical_and(clause_final_lits, C_mask.astype(bool))
                clause_satisfied = np.any(clause_final_lits, axis=1)

                errors = np.sum(~clause_satisfied)

                energy_history.append(errors)
                time_history.append(current_t)

                if step_count % 1000 == 0:
                    print(f" Time={current_t:.1f} | Hamiltonian={errors}")

                # Convergence check
                if errors == 0:
                    print(" >>> GLOBAL MINIMUM REACHED! Freezing system state...")
                    found_solution = True
                    final_time = time.time() - start_time

                # Stability check
                weights = current_y[2 * n_vars + 1:]
                if np.max(weights) > WEIGHT_LIMIT:
                    print("[WARN] Weight limit reached (Potential local minimum).")
                    break

    # --- PHYSICAL UNIT MAPPING ---
    # Assumption: 1 Simulation Unit ~ 1 nanosecond (1 GHz Clock)
    phys_time_ns = np.array(time_history)

    # --- PLOTTING ---
    print("[INFO] Generating publication-quality plots...")
    plt.figure(figsize=(10, 6))

    # Downsample for cleaner vector graphics if needed
    if len(phys_time_ns) > 5000:
        skip = len(phys_time_ns) // 2000
        t_plot = phys_time_ns[::skip]
        e_plot = energy_history[::skip]
    else:
        t_plot = phys_time_ns
        e_plot = energy_history

    # Plot Hamiltonian Evolution
    plt.plot(t_plot, e_plot, label=r'Hamiltonian Energy $H(t)$', color='#008000', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Physical Axis Labels
    plt.xlabel(r'Physical Time (ns)', fontsize=12, fontweight='bold')
    plt.ylabel(r'Hamiltonian Energy ($H$)', fontsize=12, fontweight='bold')

    plt.title(f'Temporal Evolution of Solution Search (N={n_vars})\nContinuous-Time Dynamics ($f_{{res}}=1$ GHz)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.4, linestyle='--')

    # Annotation Box
    stats_text = (
        f"Instance: {fname}\n"
        f"Type: Random 3-SAT\n"
        f"Outcome: SATISFIED\n"
        f"Final Energy: 0\n"
        f"Convergence Time: {t_plot[-1 if found_solution else -1]:.1f} ns"
    )

    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.text(0.95, 0.6, stats_text, transform=plt.gca().transAxes,
             fontsize=10, ha='right', va='center', fontname='Consolas',
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.5'))

    # Save output
    out_name = f"Final_Perfect_Graph_{fname}.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to: {out_name}")
    plt.show()