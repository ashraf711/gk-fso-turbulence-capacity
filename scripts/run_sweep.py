import torch
from .kennedy_channel import I_kennedy_t,  I_kennedy_t_avg_gamma_gamma
from .turbulence import sample_gamma_gamma

def maximize_I_ml(
    N,
    restarts=5,
    steps=1500,
    lr=1e-4,
    seed=0,
    device="cpu",
    init=None,
    patience=300,
    verbose=False,
    use_gamma_gamma=False,
    alpha_g=4.0,
    beta_g=2.0,
    K_turb=256,
    T_samples=None,
    common_random_numbers=True,
):
    """
    For a single N, maximize I(X;Y) over real beta and p∈(0,1) using Torch/Adam.
    p = sigmoid(theta_p), beta is unconstrained.
    Returns: best_I, best_beta, best_p
    """
    torch.manual_seed(seed)
    N_t = torch.tensor(float(N), dtype=torch.double, device=device)

    best_I = -1.0
    best_beta = 0.0
    best_p = 0.5
    
    T_fixed = None

    if use_gamma_gamma:
        if T_samples is not None:
            T_fixed = T_samples.to(device=device, dtype=torch.double)
        elif common_random_numbers:
            T_fixed = sample_gamma_gamma(alpha_g, beta_g, K_turb, device=device, dtype=torch.double)

    for r in range(restarts):
        # Init
        if init is not None and r == 0:
            beta = torch.tensor(init["beta"], dtype=torch.double, device=device)
            p0 = torch.tensor(init["p"], dtype=torch.double, device=device).clamp(1e-12, 1-1e-12)
            theta_p = torch.logit(p0)

        else:
            beta = torch.randn((), dtype=torch.double, device=device) * 0.5
            theta_p = torch.randn((), dtype=torch.double, device=device) * 0.5

        beta.requires_grad_(True)
        theta_p.requires_grad_(True)

        opt = torch.optim.Adam([beta, theta_p], lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)

        best_I_r = -1.0
        best_beta_r = float(beta.item())
        best_p_r = float(torch.sigmoid(theta_p).item())
        no_improve = 0

        for t in range(steps):
            p = torch.sigmoid(theta_p)
            p = torch.clamp(p, 1e-12, 1 - 1e-12)

            if use_gamma_gamma:
                # Use fixed samples if provided, otherwise resample each call
                I = I_kennedy_t_avg_gamma_gamma(
                    N_t, beta, p, alpha_g, beta_g, K=K_turb,
                    T_samples=T_fixed
                )
            else:
                I = I_kennedy_t(N_t, beta, p)
            
            loss = -I

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (t + 1) % 50 == 0:
                scheduler.step()

            I_val = float(I.item())
            if I_val > best_I_r:
                best_I_r = I_val
                best_beta_r = float(beta.item())
                best_p_r = float(p.item())
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (t + 1) % 300 == 0:
                tag = "GG" if use_gamma_gamma else "ideal"
                print(f"[N={N:.4f} r={r+1}] t={t+1} beta={beta.item():.5f} p={p.item():.5f} I={I_val:.7f}")

            if no_improve > patience:
                break

        if best_I_r > best_I:
            best_I = best_I_r
            best_beta = best_beta_r
            best_p = best_p_r

    return best_I, best_beta, best_p
