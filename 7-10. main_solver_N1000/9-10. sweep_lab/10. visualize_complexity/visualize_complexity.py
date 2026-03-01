"""
Script: IMC Complexity Scaling Visualizer
Description: Scientific tool to map Inertial Manifold Computing (IMC)
             trajectories to Turing-equivalent bit-operations.
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

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog
import os

def generate_scaling_plot():
    # Initialize file selection dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Consolidated Complexity Report",
        filetypes=[("CSV Files", "*.csv")]
    )
    root.destroy()

    if not file_path:
        return

    # Academic plotting configuration
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "figure.dpi": 300
    })

    # Data ingestion and preprocessing
    df = pd.read_csv(file_path)
    df_success = df[df['Status'] == 'SUCCESS'].copy()
    df_success['Complexity_BitOps'] = df_success['Complexity_BitOps'].astype(float)

    # Filtering for optimal resonance points (minimum complexity per N)
    df_optimal = df_success.groupby('Variables')['Complexity_BitOps'].min().reset_index()

    # Visualization
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.lineplot(
        data=df_optimal, x='Variables', y='Complexity_BitOps',
        marker='o', markersize=7, color='#1f77b4', linewidth=2, label='Optimal IMC Trajectory'
    )

    # Logarithmic transformation for complexity scaling analysis
    ax.set_yscale('log')

    # Metadata and Axis Labeling
    ax.set_title("Computational Complexity Scaling Analysis", pad=15)
    ax.set_xlabel("Problem Size (Number of Variables, N)")
    ax.set_ylabel("Turing-Equivalent Complexity (Bit-Operations, Log Scale)")

    # Grid and Aesthetic Refinement
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc='upper left')

    # Value annotations for data verification
    for x, y in zip(df_optimal['Variables'], df_optimal['Complexity_BitOps']):
        ax.annotate(f'{y:.1e}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # High-resolution export
    output_img = "complexity_scaling_analysis.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Plot successfully exported to: {output_img}")

if __name__ == "__main__":
    generate_scaling_plot()