import torch
import numpy as np

def sample_gamma_gamma_np(alpha_g, beta_g, K, seed=None):
    """
    Mean-one Gamma–Gamma via product of two Gammas (NumPy).
    X ~ Gamma(shape=alpha_g, scale=1/alpha_g)
    Y ~ Gamma(shape=beta_g,  scale=1/beta_g)
    """
    rng = np.random.default_rng(seed)

    X = rng.gamma(shape=alpha_g, scale=1.0/alpha_g, size=K)  # mean 1
    Y = rng.gamma(shape=beta_g,  scale=1.0/beta_g,  size=K)  # mean 1
    return X * Y

def sample_gamma_gamma(alpha_g, beta_g, K, device, dtype=torch.double):
    """
    Mean-one Gamma–Gamma samples via product of two Gamma RVs:
      X ~ Gamma(alpha_g, rate=alpha_g), Y ~ Gamma(beta_g, rate=beta_g), T=XY
    """
    a = torch.as_tensor(alpha_g, dtype=dtype, device=device)
    b = torch.as_tensor(beta_g, dtype=dtype, device=device)

    X = torch.distributions.Gamma(concentration=a, rate=a).sample((K,))  # mean 1
    Y = torch.distributions.Gamma(concentration=b, rate=b).sample((K,))  # mean 1
    return X * Y


