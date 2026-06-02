import os
import numpy as np
import torch

from src.sweep import run_sweep
from src.io_utils import save_results_csv
from src.plots import (
    plot_capacity_vs_N,
    plot_capacity_ideal_vs_turbulence,
    plot_optimal_beta_vs_N,
    plot_optimal_prior_vs_N,
)

def pick_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def main():
    os.makedirs("outputs", exist_ok=True)

    N_vals = np.linspace(1e-6, 2.0, 200)

    device = pick_device(prefer_cuda=True)
    print("Using device:", device)

    run_ideal = True
    run_turbulence = True

    alpha_g = 4.0
    beta_g = 2.0
    K_turb = 256
    turb_seed = 0

    results = run_sweep(
        N_vals,
        device=device,
        null_enabled=True,
        ml_enabled=True,
        ml_restarts=6, # initially set to 6
        ml_steps=1800,
        ml_lr=1e-4,
        ml_seed=1234,
        ml_patience=400,
        progress_every=25,

        run_ideal=run_ideal,
        run_turbulence=run_turbulence,
        alpha_g=alpha_g,
        beta_g=beta_g,
        K_turb=K_turb,
        turb_seed=turb_seed,
    )

    save_results_csv(results, "outputs/kennedy_capacity_results.csv")

    plot_capacity_vs_N(results, "outputs/capacity_vs_N.pdf")
    plot_capacity_ideal_vs_turbulence(results, "outputs/capacity_ideal_vs_turbulence.pdf")
    plot_optimal_beta_vs_N(results, "outputs/optimal_beta_vs_N.pdf")
    plot_optimal_prior_vs_N(results, "outputs/optimal_prior_vs_N.pdf")

    print("Saved outputs.")

if __name__ == "__main__":
    main()