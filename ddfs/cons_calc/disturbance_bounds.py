# ddfs/cons_calc/disturbance_bounds.py

from __future__ import annotations
import numpy as np  # pyright: ignore[reportMissingImports]
from typing import Sequence, Tuple

def compute_beta_i(
    H_i: np.ndarray,
    Xi_i: np.ndarray,
    k_indices: Sequence[int],
    k_i: int,
    C: float,
    L_J: float,
    gamma: float,
) -> float:
    """
    Implements Eq. (β_i) in paper:
    β_i = sum_k (C|k-k_i| ||z(k)|| + gamma + 0.5 L_J ||z(k)||^2)^2,
    with z(k) = [eta(k); xi(k)], using stacked H_i, Xi_i.
    """
    assert H_i.shape[1] == Xi_i.shape[1] == len(k_indices), "Column/time mismatch"
    z = np.vstack([H_i, Xi_i])
    z_norm = np.linalg.norm(z, axis=0)
    offs = np.abs(np.asarray(k_indices) - int(k_i))
    terms = C * offs * z_norm + float(gamma) + 0.5 * float(L_J) * z_norm**2
    return float(np.sum(terms**2))

def empirical_beta_lower_bound(H_plus: np.ndarray, Z: np.ndarray) -> float:
    """
    Optional: compute ||W_emp||_2^2 from least-squares fit to set a floor for β_i.
    """
    # [Ahat Bhat] = H_plus Z^T (Z Z^T)^dagger
    ZZt = Z @ Z.T
    AhatBhat = H_plus @ Z.T @ np.linalg.inv(ZZt)
    W_emp = H_plus - AhatBhat @ Z
    smax = np.linalg.svd(W_emp, compute_uv=False)[0] if W_emp.size else 0.0
    return float(smax**2)

def inflate_beta_if_needed(beta_i: float, H_plus: np.ndarray, Z: np.ndarray, tau: float = 1.1) -> float:
    """
    Ensure beta_i >= tau ||W_emp||_2^2 from empirical fit.
    """
    need = empirical_beta_lower_bound(H_plus, Z)
    return max(float(beta_i), float(tau * need))
