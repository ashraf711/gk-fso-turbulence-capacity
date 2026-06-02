import numpy as np
import torch

def h2_np(q):
    """Binary entropy in bits (NumPy), vectorized, stable at 0/1."""
    q = np.clip(q, 1e-15, 1 - 1e-15)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

def h2_t(q):
    """Binary entropy in bits (Torch), elementwise, stable at 0/1."""
    eps = 1e-12
    q = torch.clamp(q, eps, 1 - eps)
    return -(q * torch.log2(q) + (1 - q) * torch.log2(1 - q))
