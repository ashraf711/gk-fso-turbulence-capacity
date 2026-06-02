import csv
import numpy as np

def save_results_csv(results, filename):
    """
    Writes a CSV using keys if present.
    Supports both ideal and Gamma–Gamma (turbulence) outputs.

    Ideal keys (if present):
    N, exp(-2N), Holevo_Cinf, Dolinar_C1,
    Kennedy_null, Kennedy_optML,
    p_null, beta_star, p_star

    Turbulence keys (if present):
    GG_alpha, GG_beta, GG_K,
    Kennedy_null_GG, p_null_GG,
    Kennedy_optML_GG, beta_star_GG, p_star_GG
    """
    keys = [
        # Common
        "N", "exp(-2N)", "Holevo_Cinf", "Dolinar_C1",

        # Ideal baselines / opt
        "Kennedy_null", "Kennedy_optML",
        "p_null", "beta_star", "p_star",

        # Turbulence metadata (you now store as arrays)
        "GG_alpha", "GG_beta", "GG_K",

        # Turbulence curves
        "Kennedy_null_GG", "p_null_GG",
        "Kennedy_optML_GG", "beta_star_GG", "p_star_GG",
    ]

    # Only keep existing keys
    keys = [k for k in keys if k in results]

    N = np.asarray(results["N"])
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(N)):
            row = []
            for k in keys:
                v = np.asarray(results[k])
                # v should be a vector (same length as N) now, including GG_*.
                # Still safe if someone accidentally passes a scalar:
                if v.ndim == 0:
                    row.append(float(v))
                else:
                    row.append(float(v[i]))
            w.writerow(row)
