# nominal/init/warmstart.py
from __future__ import annotations

from typing import Tuple

import numpy as np  # pyright: ignore[reportMissingImports]


def warmstart_fixed(model, K: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Allocate and warm-start X, U for fixed-final-time runs. Returns (X, U, sigma_guess).
    """
    X = np.empty((model.n_x, K), dtype=float)
    U = np.empty((model.n_u, K), dtype=float)
    X, U = model.initialize_trajectory(X, U)
    sigma_guess = float(getattr(model, "t_f_guess", 1.0))
    return X, U, sigma_guess


def warmstart_free(model, K: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Allocate and warm-start X, U, and return initial sigma for free-final-time runs.
    """
    X = np.empty((model.n_x, K), dtype=float)
    U = np.empty((model.n_u, K), dtype=float)
    X, U = model.initialize_trajectory(X, U)
    sigma_guess = float(getattr(model, "t_f_guess", 1.0))
    return X, U, sigma_guess
