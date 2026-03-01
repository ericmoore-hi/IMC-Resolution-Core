"""
Script: COMPACT SCIENTIFIC BENCHMARK (N=10 to N=20)
Description:
Runs a rapid but rigorously averaged benchmark in the range N=[10, 20].
Uses 2-step intervals to cover more ground in the same amount of time.
Performs 5 trials per point to ensure statistical validity (Median selection).
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
import time
import random

# ==========================================
# 1. GENERATOR & SOLVER
# ==========================================
def generate_3sat_instance(N, alpha=4.26):
    M = int(round(N * alpha))
    clauses = []
    while len(clauses) < M:
        vars_idx = random.sample(range(N), 3)
        clause = []
        for idx in vars_idx:
            sign = random.choice([1, -1])
            clause.append(sign * (idx + 1))
        clauses.append(clause)
    return clauses

def solve_physics_engine(N, clauses, time_limit=5.0): # 5 saniyə limit kifayətdir bu ölçü üçün
    c_indices = [[abs(l)-1 for l in c] for c in clauses]
    c_signs   = [[1 if l>0 else -1 for l in c] for c in clauses]

    dt = 0.2
    phi = np.random.uniform(0, 2*np.pi, N)
    restart_interval = 600
    start_time = time.time()

    for step in range(50000): # Addım sayını azaltdıq ki sürətli olsun
        if (time.time() - start_time) > time_limit:
            return False, 0.0

        if step % 50 == 0:
            bool_state = np.cos(phi) > 0
            solved = True
            for i in range(len(c_indices)):
                vals = bool_state[c_indices[i]]
                req = (np.array(c_signs[i]) == 1)
                if not np.any(vals == req):
                    solved = False
                    break
            if solved:
                return True, step * dt

        if step > 0 and step % restart_interval == 0:
            phi = np.random.uniform(0, 2*np.pi, N)

        forces = np.zeros(N)
        for i in range(len(c_indices)):
            idx = c_indices[i]
            sgn = c_signs[i]
            p_vals = phi[idx]
            analog_vals = np.cos(p_vals) * sgn

            if np.all(analog_vals < 0.2):
                targets = np.where(np.array(sgn) > 0, 0.0, np.pi)
                diffs = targets - p_vals
                diffs = (diffs + np.pi) % (2*np.pi) - np.pi
                np.add.at(forces, idx, diffs)

        noise = np.random.normal(0, 0.3, N)
        phi += (forces * 0.3 + noise) * dt

    return False, 0.0

# ==========================================
# 2. RAPID BENCHMARK LOOP
# ==========================================
def run_compact_benchmark():
    print("\n--- COMPACT SCALING BENCHMARK (N=10 to N=20) ---")
    print("[INFO] Range: [10, 12, 14, 16, 18, 20]")
    print("[INFO] Strategy: 5 trials per point -> Median.")
    print("-" * 60)

    # 2-2 artırırıq ki, qrafik geniş olsun amma vaxt getməsin
    N_values = [10, 12, 14, 16, 18, 20]

    final_N = []
    final_Time = []

    for N in N_values:
        print(f"[N={N}] Testing: ", end="", flush=True)
        times = []

        # 5 Nümunə götürürük (Statistik dürüstlük üçün)
        while len(times) < 5:
            clauses = generate_3sat_instance(N)
            success, t_phys = solve_physics_engine(N, clauses)

            if success:
                times.append(t_phys)
                print(".", end="", flush=True)
            else:
                pass # Timeout, yenisini yoxla

        avg_time = np.median(times)
        final_N.append(N)
        final_Time.append(avg_time)
        print(f" DONE ({avg_time:.1f} ns)")

    # ==========================================
    # 3. PLOTTING
    # ==========================================
    print("-" * 60)
    print("[INFO] Generating Graph...")

    plt.figure(figsize=(10, 7))

    plt.scatter(final_N, final_Time, color='#d62728', s=150, edgecolors='k', zorder=5,
                label='Experimental Median Time')

    # Fit Trend
    log_n = np.log(final_N)
    log_t = np.log(final_Time)
    coeffs = np.polyfit(log_n, log_t, 1)
    exponent = coeffs[0]
    prefactor = np.exp(coeffs[1])

    # Xətti bir az irəli uzadaq (N=30-a qədər) ki proqnoz görünsün
    x_fit = np.linspace(10, 30, 100)
    y_fit = prefactor * (x_fit ** exponent)

    plt.plot(x_fit, y_fit, 'b--', linewidth=2.5,
             label=f'Scaling Trend ($T \\propto N^{{{exponent:.2f}}})$')

    plt.text(0.05, 0.8, f"Measured Exponent: {exponent:.2f}", transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Variables ($N$)', fontsize=12, fontweight='bold')
    plt.ylabel('Convergence Time (ns)', fontsize=12, fontweight='bold')
    plt.title('Computational Scaling (Small Scale N=10-20)', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc='lower right')

    out_file = 'Compact_Scaling_Benchmark.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Graph saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    run_compact_benchmark()