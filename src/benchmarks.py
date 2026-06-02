import numpy as np
from .entropy import h2_np

def holevo_C_infty(N):
    """Holevo capacity for coherent-state BPSK: C∞(N) = h2((1 - e^{-2N})/2)."""
    x = (1 - np.exp(-2 * N)) / 2.0
    return h2_np(x)

def dolinar_C1(N):
    """Symbol-by-symbol optimal capacity via Helstrom error (equal priors)."""
    Pe = 0.5 * (1 - np.sqrt(1 - np.exp(-4 * N)))
    return 1.0 - h2_np(Pe)
