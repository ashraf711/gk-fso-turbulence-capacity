import numpy as np
import torch
from .entropy import h2_np, h2_t
from .turbulence import sample_gamma_gamma

def I_kennedy_np(N, beta, p):
    alpha = np.sqrt(N)

    q0 = np.exp(-((beta - alpha)**2))
    q1 = np.exp(-((beta + alpha)**2))

    qbar = p * q0 + (1 - p) * q1  # p is prior for X=|−α>, 1-p for X=|+α>
    return h2_np(qbar) - p*h2_np(q0) - (1-p)*h2_np(q1)

def I_kennedy_np_given_T(N, beta, p, T):
    """
    Mutual information conditioned on turbulence realization(s) T.
    Physical model: |±alpha> -> |±sqrt(T) alpha>, then local displacement beta.
    """
    alpha = np.sqrt(N)

    q0 = np.exp(-((beta - np.sqrt(T)*alpha)**2))  # no-click given X = |−α>
    q1 = np.exp(-((beta + np.sqrt(T)*alpha)**2))  # no-click given X = |+α>

    qbar = p * q0 + (1 - p) * q1
    return h2_np(qbar) - p*h2_np(q0) - (1-p)*h2_np(q1)


def I_kennedy_np_avg_gamma_gamma(N, beta, p, T_samples):
    """
    Monte Carlo estimate of E_T[ I(X;Y | T) ] using provided T_samples (NumPy array).
    """
    I_samples = I_kennedy_np_given_T(N, beta, p, T_samples)
    return float(np.mean(I_samples))


def I_kennedy_t(N_t, beta_t, p_t):
    """Torch version for autograd, physical symmetric displacement convention."""
    alpha_t = torch.sqrt(N_t)

    q0 = torch.exp(-((beta_t - alpha_t)**2))
    q1 = torch.exp(-((beta_t + alpha_t)**2))

    qbar = p_t * q0 + (1 - p_t) * q1
    return h2_t(qbar) - p_t * h2_t(q0) - (1 - p_t) * h2_t(q1)



def I_kennedy_t_given_T(N_t, beta_t, p_t, T_t):
    """
    Mutual information conditioned on turbulence realization(s) T_t.
    Physical model: |±alpha> -> |±sqrt(T) alpha>, then local displacement beta.
    """
    alpha_t = torch.sqrt(N_t)
    sqrt_T = torch.sqrt(T_t)

    q0 = torch.exp(-((beta_t - sqrt_T * alpha_t)**2))
    q1 = torch.exp(-((beta_t + sqrt_T * alpha_t)**2))

    qbar = p_t * q0 + (1 - p_t) * q1
    return h2_t(qbar) - p_t * h2_t(q0) - (1 - p_t) * h2_t(q1)


def I_kennedy_t_avg_gamma_gamma(N_t, beta_t, p_t, alpha_g, beta_g, K=256, T_samples=None):
    """
    Monte Carlo estimate of E_T[ I(X;Y | T) ] under Gamma–Gamma turbulence.
    Returns a scalar tensor (objective for Adam).
    """
    if T_samples is None:
        T_samples = sample_gamma_gamma(alpha_g, beta_g, K, device=N_t.device, dtype=N_t.dtype)

    I_samples = I_kennedy_t_given_T(N_t, beta_t, p_t, T_t=T_samples)
    return I_samples.mean()
