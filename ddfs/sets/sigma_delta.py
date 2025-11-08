# ddfs/sets/sigma_delta.py
from __future__ import annotations

import numpy as np # type: ignore

def make_N2(C:float, tilde_T_i:int | float, n:int, m:int) -> np.ndarray:
    r"""
    Construct the variation QMI block N2 given C = L_J * v and \tilde T_i.

    Paper form:
        N2 = E · blkdiag(C^2 \tilde T_i^2 I_n, -I_n, -I_m) · E^T,
    where E selects the (I_n), (ΔA), (ΔB) blocks in the lifted vector.

    Compact equivalent used here (same as N1's reduced shape):
        Use a 2-block ordering [y; w], with y ∈ R^n and w ∈ R^{n+m} collecting the
        ΔA/ΔB parts. Then

            N2 = [[ C^2 * \tilde T_i^2 * I_n,     0            ],
                  [          0              ,  -I_{n+m}        ]]

    Args:
        C:         Lipschitz*increment constant (L_J * v).
        tilde_T:   \tilde T_i (e.g., 2*|T_i| - 1).
        n:         state dimension.
        m:         input dimension.

    Returns:
        N2: (n + n + m, n + n + m) = (n + (n+m)) square matrix.
    """
    c2t2 = float(C) * float(tilde_T_i) * float(C) * float(tilde_T_i)

    TL = c2t2 * np.eye(n, dtype=float)
    BR = -np.eye(n+m, dtype=float)

    N2 = np.block([
        [TL, np.zeros((n, n+m), dtype=float)],
        [np.zeros((n+m, n), dtype=float), BR]
    ])

    return 0.5 * (N2 + N2.T)

def pad_top_left(N: np.ndarray, total_dim: int) -> np.ndarray:
    """
    Pad N into the top-left of a total_dim x total_dim zero matrix.
    Useful to build \tilde N_2 = [[N2, 0], [0, 0]] to match S(P,L,nu) size.

    Args:
        N:         (r, r) square block to embed
        total_dim: final dimension (>= r)

    Returns:
        Nt: (total_dim, total_dim)
    """
    N = np.asarray(N, dtype=float)
    if N.ndim != 2 or N.shape[0] != N.shape[1]:
        raise ValueError("N must be a square 2D array.")
    r = N.shape[0]
    if total_dim < r:
        raise ValueError(f"total_dim ({total_dim}) must be >= N.shape[0] ({r}).")
    if total_dim == r:
        return N.copy()

    Nt = np.zeros((total_dim, total_dim), dtype=float)
    Nt[:r, :r] = N
    return Nt