import numpy as np
import torch
from .benchmarks import holevo_C_infty, dolinar_C1
from .kennedy_channel import (
    I_kennedy_np,
    I_kennedy_np_avg_gamma_gamma,
)
from .turbulence import sample_gamma_gamma_np
from .optim_p_newton import (
    optimize_p_for_fixed_beta,
    optimize_p_for_fixed_beta_turbulence,
)
from .optim_joint_adam import maximize_I_ml

def run_sweep(
    N_vals,
    device="cpu",
    null_enabled=True,
    ml_enabled=True,
    ml_restarts=6,
    ml_steps=1800,
    ml_lr=1e-4,
    ml_seed=1234,
    ml_patience=300,
    progress_every=25,

    run_ideal=True,
    run_turbulence = True,
    alpha_g=4.0,
    beta_g=2.0,
    K_turb=256,
    turb_seed=0,
):
    """
    Runs:
      - Holevo C∞
      - Dolinar/Helstrom C1
      - Kennedy nulling baseline: beta = sqrt(N), ideal
      - Kennedy with (beta,p) optimized via Adam (ideal)
    Optionally (if turbulence_enabled=True):
      - Kennedy nulling baseline: beta = sqrt(N), under Gamma-Gamma turbulence
      - Kennedy with (beta,p) optimized under Gamma–Gamma turbulence
    Returns dict of arrays.
    """
    N_vals = np.asarray(N_vals, dtype=float)

    results = {}
    results["N"] = N_vals
    if run_turbulence:
        results["GG_alpha"] = np.full_like(N_vals, float(alpha_g))
        results["GG_beta"] = np.full_like(N_vals, float(beta_g))
        results["GG_K"] = np.full_like(N_vals, int(K_turb), dtype=int)

    results["exp(-2N)"] = np.exp(-2 * N_vals)

    results["Holevo_Cinf"] = holevo_C_infty(N_vals)
    results["Dolinar_C1"] = dolinar_C1(N_vals)
    T_fixed_np = None
    T_fixed_torch = None

    if run_turbulence:
        T_fixed_np = sample_gamma_gamma_np(alpha_g, beta_g, K_turb, seed=turb_seed)
        T_fixed_torch = torch.tensor(T_fixed_np, dtype=torch.double, device=device)
    

    # Kennedy nulling baseline: beta = sqrt(N), ideal
    if null_enabled and run_ideal:
        C_null = np.empty_like(N_vals)
        p_null = np.empty_like(N_vals)

        for i, N in enumerate(N_vals):
            beta_null = np.sqrt(N)
            p_opt = optimize_p_for_fixed_beta(N, beta=beta_null)
            C_null[i] = I_kennedy_np(N, beta=beta_null, p=p_opt)
            p_null[i] = p_opt

        results["Kennedy_null"] = C_null
        results["p_null"] = p_null

    # Kennedy nulling baseline under Gamma-Gamma turbulence
    if null_enabled and run_turbulence:
        C_null_GG = np.empty_like(N_vals)
        p_null_GG = np.empty_like(N_vals)

        T_fixed = T_fixed_np

        for i, N in enumerate(N_vals):
            beta_null = np.sqrt(N)
            p_opt = optimize_p_for_fixed_beta_turbulence(
                N, beta=beta_null, T_samples=T_fixed
            )
            C_null_GG[i] = I_kennedy_np_avg_gamma_gamma(
                N, beta=beta_null, p=p_opt, T_samples=T_fixed
            )
            p_null_GG[i] = p_opt

        results["Kennedy_null_GG"] = C_null_GG
        results["p_null_GG"] = p_null_GG

    # Ideal Kennedy/GK joint optimization over beta and p
    if ml_enabled and run_ideal:
        C_ml = np.empty_like(N_vals)
        beta_star = np.empty_like(N_vals)
        p_star = np.empty_like(N_vals)

        init = None
        for i, N in enumerate(N_vals):
            best_I, best_beta, best_p = maximize_I_ml(
                float(N),
                restarts=ml_restarts,
                steps=ml_steps,
                lr=ml_lr,
                seed=ml_seed,
                device=device,
                init=init,
                patience=ml_patience,
                verbose=False,
                use_gamma_gamma=False,
            )

            C_ml[i] = best_I
            beta_star[i] = best_beta
            p_star[i] = best_p
            init = {"beta": best_beta, "p": best_p}

            if progress_every and (i % progress_every == 0):
                print(f"Ideal progress: {i}/{len(N_vals)} (N={N:.6g})")

        results["Kennedy_optML"] = C_ml
        results["beta_star"] = beta_star
        results["p_star"] = p_star

    # Gamma-Gamma turbulence Kennedy/GK joint optimization over beta and p
    if ml_enabled and run_turbulence:
        C_ml_GG = np.empty_like(N_vals)
        beta_star_GG = np.empty_like(N_vals)
        p_star_GG = np.empty_like(N_vals)

        init = None
        for i, N in enumerate(N_vals):
            best_I, best_beta, best_p = maximize_I_ml(
                float(N),
                restarts=ml_restarts,
                steps=ml_steps,
                lr=ml_lr,
                seed=ml_seed,
                device=device,
                init=init,
                patience=ml_patience,
                verbose=False,
                use_gamma_gamma=True,
                alpha_g=alpha_g,
                beta_g=beta_g,
                K_turb=K_turb,
                T_samples=T_fixed_torch,
                common_random_numbers=True,
            )

            C_ml_GG[i] = best_I
            beta_star_GG[i] = best_beta
            p_star_GG[i] = best_p
            init = {"beta": best_beta, "p": best_p}

            if progress_every and (i % progress_every == 0):
                print(f"GG progress: {i}/{len(N_vals)} (N={N:.6g})")

        results["Kennedy_optML_GG"] = C_ml_GG
        results["beta_star_GG"] = beta_star_GG
        results["p_star_GG"] = p_star_GG

    return results
