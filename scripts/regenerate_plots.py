import os
import pandas as pd

from src.plots import (
    plot_capacity_vs_N,
    plot_optimal_beta_vs_N,
    plot_optimal_prior_vs_N,
)

def main():
    os.makedirs("figures", exist_ok=True)

    df = pd.read_csv("outputs/kennedy_capacity_results.csv")
    results = {col: df[col].to_numpy() for col in df.columns}

    plot_capacity_vs_N(results, "figures/capacity_vs_N.pdf")
    plot_optimal_beta_vs_N(results, "figures/optimal_beta_vs_N.pdf")
    plot_optimal_prior_vs_N(results, "figures/optimal_prior_vs_N.pdf")

    print("Regenerated figures in figures/")

if __name__ == "__main__":
    main()
