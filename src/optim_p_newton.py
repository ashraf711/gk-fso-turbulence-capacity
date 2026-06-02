import numpy as np
from .entropy import h2_np

def optimize_p_for_fixed_beta(N, beta, tol=1e-12, maxit=200):
    """
    Optimize p in [0,1] for fixed beta using Newton's method (concave in p).
    """
    alpha = np.sqrt(N)
    q0 = np.exp(-((beta - alpha)**2))
    q1 = np.exp(-((beta + alpha)**2))

    p = 0.5
    for _ in range(maxit):
        qbar = p*q0 + (1 - p)*q1

        eps = 1e-15
        qbar = np.clip(qbar, eps, 1 - eps)

        # dI/dp = h2'(qbar)*(q0 - q1) - h2(q0) + h2(q1)
        # h2'(x) = log2((1-x)/x)
        d1 = np.log2((1 - qbar)/qbar) * (q0 - q1) - h2_np(q0) + h2_np(q1)

        # d2 = h2''(qbar)*(q0-q1)^2,  h2''(x)=-(1/ln2)*1/(x(1-x))
        d2 = -(q0 - q1)**2 / (np.log(2) * qbar * (1 - qbar))

        step = d1 / (d2 + 1e-30)
        p_new = np.clip(p - step, 0.0, 1.0)

        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new

    return float(p)


def optimize_p_for_fixed_beta_turbulence(N, beta, T_samples, tol=1e-12, maxit=200):
    """
    Optimize p in [0,1] for fixed beta under turbulence using Newton's method.
    Uses a Monte Carlo average over provided T_samples (common random numbers).
    """
    alpha = np.sqrt(N)
    sqrt_T = np.sqrt(T_samples)

    q0 = np.exp(-((beta - sqrt_T * alpha)**2))
    q1 = np.exp(-((beta + sqrt_T * alpha)**2))

    # safety
    q0 = np.clip(q0, 1e-15, 1 - 1e-15)
    q1 = np.clip(q1, 1e-15, 1 - 1e-15)

    p = 0.5
    for _ in range(maxit):
        qbar = p*q0 + (1 - p)*q1
        qbar = np.clip(qbar, 1e-15, 1 - 1e-15)

        # per-sample derivatives, then average
        d1_k = np.log2((1 - qbar)/qbar) * (q0 - q1) - h2_np(q0) + h2_np(q1)
        d2_k = -(q0 - q1)**2 / (np.log(2) * qbar * (1 - qbar))

        d1 = np.mean(d1_k)
        d2 = np.mean(d2_k)

        step = d1 / (d2 + 1e-30)
        p_new = np.clip(p - step, 0.0, 1.0)

        if abs(p_new - p) < tol:
            p = p_new
            break
        p = p_new

    return float(p)
