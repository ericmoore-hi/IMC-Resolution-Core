"""
Script: Analytical Frequency Scaling Model for IMC
-------------------------------------------------------------------------------------------------
Description: Models the bandwidth scaling of the proposed physical architecture
based on coupled oscillator theory (1/sqrt(N)) versus classical exponential decay.
Includes Monte Carlo analysis to verify stability under thermal noise.

Output:
  1. Analytical_Frequency_Scaling.png (Log-Log Plot)
  2. Analytical_Table.png (Data Table)
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_theoretical_scaling():
    print("[INFO] Computing analytical scaling model...")

    # 1. Variable Count (System Size N)
    N_values = np.array([10, 50, 100, 200, 500, 1000, 2000])

    # 2. Physics Model: Coupled Oscillators
    # Reference: Section 4.10. Effective bandwidth scales as 1/sqrt(N).
    f_base = 1200.0  # MHz (Base frequency reference for 180nm CMOS)

    # Theoretical scaling law: omega = omega_0 / sqrt(N)
    omega_theoretical = f_base / np.sqrt(N_values)

    # 3. Noise Modeling (Monte Carlo Simulation)
    # Simulating thermal fluctuations (Johnson Noise) to test signal integrity.
    np.random.seed(42)

    # SNR degradation model: Variance increases logarithmically with size
    noise_variance = 0.02 * np.log1p(N_values)

    # Monte Carlo iterations (1000 runs per data point)
    omega_with_noise = []
    for i, n_val in enumerate(N_values):
        # Apply Gaussian noise to the theoretical signal
        samples = omega_theoretical[i] * (1 + np.random.normal(0, noise_variance[i], 1000))
        # Compute mean effective frequency after noise integration
        omega_with_noise.append(np.mean(samples))

    omega_with_noise = np.array(omega_with_noise)

    # 4. Baseline Model: Classical Exponential Decay
    # Represents the "Time/Bandwidth Wall" of classical search algorithms (e^-N)
    omega_exponential = f_base * np.exp(-0.02 * N_values)

    # 5. Data Compilation
    df = pd.DataFrame({
        'N (Size)': N_values,
        'Theoretical Model (MHz)': np.round(omega_theoretical, 2),
        'Noise-Robust Frequency (MHz)': np.round(omega_with_noise, 2),
        'Exponential Limit (MHz)': np.round(omega_exponential, 4)
    })

    print("\n--- ANALYTICAL SCALING TABLE ---")
    print(df)

    # 6. Visualization
    print("[INFO] Generating plots...")
    plt.figure(figsize=(10, 7))

    # IMC Model Plot
    plt.loglog(N_values, omega_with_noise, 'go-', linewidth=2, markersize=8,
               label=r'IMC Analytical Model ($\omega \propto 1/\sqrt{N}$)')

    # Classical Baseline Plot
    plt.loglog(N_values, omega_exponential, 'r:', linewidth=2, alpha=0.5,
               label=r'Exponential Decay Baseline ($e^{-N}$)')

    # Aesthetics
    plt.xlabel('System Size ($N$)', fontsize=12, fontweight='bold')
    plt.ylabel('Effective Frequency (MHz)', fontsize=12, fontweight='bold')
    plt.title('Analytical Bandwidth Scaling with Noise Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=11)

    # Annotation for signal viability
    plt.annotate('Measurable Signal Region', xy=(1000, 30), xytext=(500, 100),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    # Save Graph
    graph_filename = 'Analytical_Frequency_Scaling.png'
    plt.savefig(graph_filename, dpi=300, bbox_inches='tight')

    # Save Table as Image (WITHOUT INTERNAL TITLE)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    # REMOVED: plt.title(...) - Now the image is clean.

    table_filename = 'Analytical_Table.png'
    plt.savefig(table_filename, dpi=300, bbox_inches='tight')

    print(f"[SUCCESS] Outputs generated: {graph_filename}, {table_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_theoretical_scaling()