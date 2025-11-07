# ddfs/utils/numeric.py

from __future__ import annotations
import numpy as np # pyright: ignore[reportMissingImports]

def spectral_norm(M: np.ndarray) -> float:
    """
    Robust spectral norm of a matrix.
    """
    if M.size == 0:
        return 0.0
    return float(np.linalg.norm(M, ord=2))

def deg2rad(x):
    """
    Vectorized degree to radian conversion.
    """
    return np.deg2rad(x)

def rand_unit(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Random unit vector in n-dimensional space.
    """
    v = rng.standard_normal(n)
    nrm = np.linalg.norm(v)
    if nrm == 0:
        v[0] = 1.0
        nrm = 1.0
    return v / nrm