# ddfs/sets/sigma_data.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np # type: ignore

def build_Z(H: np.ndarray, Xi: np.ndarray) -> np.ndarray:
    """
    Stack Z = [H; Xi] along rows.

    Args:
        H:  (n, L) state deviations stack over the window
        Xi: (m, L) input deviations stack over the window
    
    Returns:
        Z: (n+m, L) stacked deviations
    """
    H = np.asarray(H, dtype=float)
    Xi = np.asarray(Xi, dtype=float)
    if H.ndim != 2 or Xi.ndim != 2:
        raise ValueError("H and Xi must be 2D arrays")
    if H.shape[1] != Xi.shape[1]:
        raise ValueError(f"Column mismatch: H has {H.shape[1]} columns, Xi has {Xi.shape[1]} columns")

    return np.vstack([H, Xi])

def make_N1(H_plus: np.ndarray, Z: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Construct the data QMI block N1:

        N1 =
        [I_n  H^+; 0  -Z] · blkdiag(β I_n, -I_L) · [I_n  H^+; 0  -Z]^T

    Efficient equivalent used here (avoids building big blocks):
        Let Hp = H^+,  B = [Hp; -Z].
        Then N1 = β · diag(I_n, 0) - B B^T
                = [[ βI_n - HpHp^T,   Hp Z^T ],
                   [  Z Hp^T,        -Z Z^T ]]

    Shapes:
        H_plus: (n, L)
        Z:      (n+m, L)
        N1:     (n + (n+m), n + (n+m)) = (n+p, n+p) with p = n + m
    """

    Hp = np.asarray(H_plus, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if Hp.ndim != 2 or Z.ndim != 2:
        raise ValueError("H_plus and Z must be 2D arrays")
    
    n, L1 = Hp.shape
    p, L2 = Z.shape

    if L1 != L2:
        raise ValueError(f"Time-length mismatch: H_plus has {L1} columns, Z has {L2} columns")
    
    if L1 == 0:
        return np.zeros((n+p, n+p), dtype=float)
    
    HpHpT = Hp @ Hp.T
    HZt = Hp @ Z.T
    ZZt = Z @ Z.T

    TL = beta * np.eye(n) - HpHpT
    TR = HZt
    BL = HZt.T
    BR = -ZZt

    N1 = np.block([[TL, TR], [BL, BR]])

    return 0.5 * (N1 + N1.T)

def pad_top_left(N: np.ndarray, total_dim: int) -> np.ndarray:
    """
    Pad a matrix N into top-left corner of a total_dim x total_dim zero matrix.
    Useful to form tilde N_1 = [[N1 0], [0 0]] when embedding N1 in the LMI.

    Args:
        N: (r, r) block to embed
        total_dim: final dimension (>= r)
    
    Returns:
        Nt: (total_dim, total_dim) with Nt[:r,:r] = N
    """

    N = np.asarray(N, dtype=float)
    r = N.shape[0]
    if N.ndim != 2 or N.shape[0] != N.shape[1]:
        raise ValueError("N must be a square 2D array.")
    if total_dim < r:
        raise ValueError(f"total dim ({total_dim}) must be >= N size ({r})")
    if total_dim == r:
        return N.copy()
    
    Nt = np.zeros((total_dim, total_dim), dtype=float)
    Nt[:r,:r] = N
    return Nt

def make_Z_and_N1(H: np.ndarray, Xi: np.ndarray, H_plus: np.ndarray, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    Z = build_Z(H, Xi)
    N1 = make_N1(H_plus, Z, beta)
    return Z, N1
