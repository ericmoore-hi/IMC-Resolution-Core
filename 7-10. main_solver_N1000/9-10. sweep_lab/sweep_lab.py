"""
Script: Consolidated Precision-Complexity Analysis for SAT Instances
Description: Automated batch processing of CNF instances to map bit-depth
             resonance and Turing-equivalent complexity using dynamic timeouts.
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
import pandas as pd
import time
import os
import tkinter as tk
from tkinter import filedialog
from scipy.integrate import RK45

# --- MATHEMATICAL ENGINE ---

def apply_finite_precision(data, bits):
    """Simulates hardware-level finite precision via quantization."""
    if bits is None or bits >= 64:
        return data
    scaling_factor = 2**bits
    return np.round(data * scaling_factor) / scaling_factor

def load_cnf_as_matrix(filepath):
    """Vectorized parser for DIMACS CNF files into matrix representation."""
    clauses_list = []
    n_vars = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('c', '%', 'p')): continue
            parts = [int(x) for x in line.split() if x != '0' and x != '']
            if not parts: continue
            lits = []
            for x in parts:
                var_idx = abs(x) - 1
                lits.append([var_idx, 1 if x < 0 else 0])
                n_vars = max(n_vars, var_idx + 1)
            clauses_list.append(lits)

    M = len(clauses_list)
    max_len = max(len(c) for c in clauses_list)
    C_indices = np.zeros((M, max_len), dtype=np.int32)
    C_signs = np.zeros((M, max_len), dtype=np.float32)
    C_mask = np.zeros((M, max_len), dtype=np.float32)
    for i, cl in enumerate(clauses_list):
        for j, (v_idx, is_neg) in enumerate(cl):
            C_indices[i, j] = v_idx
            C_signs[i, j] = np.pi if is_neg else 0.0
            C_mask[i, j] = 1.0
    return n_vars, M, (C_indices, C_signs, C_mask)

def compute_dynamics(t, state, C_indices, C_signs, C_mask, n_vars, M, bits):
    """Computes state derivatives with explicit precision quantization."""
    GAMMA, MASS, ADAPTATION_RATE = 0.35, 1.0, 2.5

    state_q = apply_finite_precision(state, bits)
    phi, vel, lam = state_q[:n_vars], state_q[n_vars:2*n_vars], state_q[2*n_vars]
    weights = state_q[2*n_vars+1:]

    phi_gathered = phi[C_indices] + C_signs
    qs = 0.5 * (1.0 + np.cos(phi_gathered))
    one_minus_q = (1.0 - qs) * C_mask + (1.0 - C_mask)
    clause_terms = np.prod(one_minus_q, axis=1, keepdims=True)

    prefactor = 2.0 * clause_terms * weights.reshape(-1, 1)
    grad = np.zeros(n_vars, dtype=np.float32)
    sin_vals = np.sin(phi_gathered)

    for j in range(C_indices.shape[1]):
        others = np.ones_like(clause_terms)
        for k in range(C_indices.shape[1]):
            if k != j: others *= one_minus_q[:, k:k+1]
        local_grad = prefactor * others * (0.5 * sin_vals[:, j:j+1]) * C_mask[:, j:j+1]
        np.add.at(grad, C_indices[:, j], local_grad.flatten())

    accel = (-(lam * grad) - GAMMA * vel) / MASS
    d_weights = ADAPTATION_RATE * clause_terms.flatten()
    d_lam = 0.05 if lam < 1.0 else 0.0

    res = np.empty_like(state)
    res[:n_vars] = vel
    res[n_vars:2*n_vars] = accel
    res[2*n_vars] = d_lam
    res[2*n_vars+1:] = d_weights
    return apply_finite_precision(res, bits)

# --- EXECUTION ENGINE ---

def run_batch_analysis():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Multiple CNF Files for Batch Analysis",
        filetypes=[("CNF Files", "*.cnf")]
    )
    root.destroy()

    if not file_paths:
        return

    precision_levels = [1, 2, 4, 8, 16, 32]
    global_registry = []

    header = f"{'INSTANCE':<18} | {'BITS':<5} | {'STATUS':<10} | {'PHYS_TIME':<10} | {'COMPLEXITY'}"
    print(f"\n{header}")
    print("-" * len(header))

    for cnf_path in file_paths:
        instance_name = os.path.basename(cnf_path)
        n_vars, M, matrix_data = load_cnf_as_matrix(cnf_path)

        # Dynamic timeout formula: base_time + (variable_count * scaling_factor)
        # Provides sufficient window for large-scale instances (e.g., N=600)
        instance_timeout = 20.0 + (n_vars * 0.8)

        for b in precision_levels:
            np.random.seed(42)
            phi0 = np.random.uniform(0, 2*np.pi, n_vars).astype(np.float32)
            init_state = np.concatenate([phi0, np.zeros(n_vars), [0.0], np.ones(M)])

            solver = RK45(fun=lambda t, y: compute_dynamics(t, y, *matrix_data, n_vars, M, b),
                          t0=0, y0=init_state, t_bound=200000, rtol=1e-2, atol=1e-4)

            start_wall, bit_ops, status, final_t = time.time(), 0.0, "TIMEOUT", 0.0

            while solver.status == 'running':
                if (time.time() - start_wall) > instance_timeout:
                    break
                solver.step()
                bit_ops += M * 3 * b * solver.step_size

                # Convergence Verification (Boolean Assignment Mapping)
                phi = solver.y[:n_vars]
                satisfied = np.any(((np.cos(phi)[matrix_data[0]] > 0) != (matrix_data[1] > 1.0)) & matrix_data[2].astype(bool), axis=1)
                if np.sum(~satisfied) == 0:
                    status = "SUCCESS"
                    final_t = solver.t
                    break

            res = {
                "Instance": instance_name,
                "Variables": n_vars,
                "Precision_Bits": b,
                "Status": status,
                "Phys_Time_ns": round(final_t, 2) if status == "SUCCESS" else "N/A",
                "Complexity_BitOps": f"{bit_ops:.2e}"
            }
            global_registry.append(res)
            print(f"{instance_name[:18]:<18} | {b:<5} | {status:<10} | {res['Phys_Time_ns']:<10} | {res['Complexity_BitOps']}")

    # Export consolidated report
    pd.DataFrame(global_registry).to_csv("Consolidated_Complexity_Report.csv", index=False)
    print(f"\n[COMPLETE] Statistical report saved to: Consolidated_Complexity_Report.csv")

if __name__ == "__main__":
    run_batch_analysis()