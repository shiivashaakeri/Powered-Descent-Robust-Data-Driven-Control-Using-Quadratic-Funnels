# ddfs/sdp/slemma_blocks.py
from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]
import cvxpy as cp  # type: ignore


def s_dim(n: int, m: int) -> int:
    """
    Dimension of S(P,L,nu) in the theorem's 6-block layout.
    Block sizes are [n, n, m, n, m, n] => total = 4n + 2m.
    """
    return 4 * int(n) + 2 * int(m)


def S_block(P, L, nu, alpha: float = 0.99):
    r"""
    Build the big S(P_i, L_i, \nu) block used in the S-lemma certificate:

        S(P,L,nu) =
        [ alphaP-nuI   0      0      0      0      0  ]
        [   0    -P    -Lᵀ     -P    -Lᵀ     0  ]
        [   0    -L      0     -L      0      L ]
        [   0    -P    -Lᵀ     -P    -Lᵀ     0  ]
        [   0    -L      0     -L      0      L ]
        [   0     0      Lᵀ     0      Lᵀ     P ]

    where P ∈ Sⁿ₊₊, L ∈ R^{m x n}, alpha∈(0,1), nu>0.

    Shapes:
      - P: (n,n)
      - L: (m,n)
      - returns: (4n+2m, 4n+2m)
    """
    # Check if inputs are CVXPY expressions
    is_cvxpy = isinstance(P, cp.Expression) or isinstance(L, cp.Expression) or isinstance(nu, cp.Expression)
    
    if is_cvxpy:
        # CVXPY version - work with expressions
        n = P.shape[0]
        m = L.shape[0]
        
        # Create zero matrices as CVXPY constants
        Znn = cp.Constant(np.zeros((n, n), dtype=float))
        Znm = cp.Constant(np.zeros((n, m), dtype=float))
        Zmn = cp.Constant(np.zeros((m, n), dtype=float))
        Zmm = cp.Constant(np.zeros((m, m), dtype=float))
        
        # Convert nu to expression if needed
        if not isinstance(nu, cp.Expression):
            nu_expr = cp.Constant(float(nu))
        else:
            nu_expr = nu
        
        # Build block matrix using CVXPY bmat
        A11 = alpha * P - nu_expr * cp.Constant(np.eye(n, dtype=float))
        
        S = cp.bmat([
            [A11,  Znn,   Znm,   Znn,   Znm,   Znn],
            [Znn,  -P,   -L.T,   -P,   -L.T,   Znn],
            [Zmn,  -L,    Zmm,   -L,    Zmm,    L ],
            [Znn,  -P,   -L.T,   -P,   -L.T,   Znn],
            [Zmn,  -L,    Zmm,   -L,    Zmm,    L ],
            [Znn,  Znn,   L.T,   Znn,   L.T,     P ],
        ])
        
        # Symmetrize for numerical hygiene
        return 0.5 * (S + S.T)
    else:
        # NumPy version - original implementation
        P = np.asarray(P, dtype=float)
        L = np.asarray(L, dtype=float)
        n = P.shape[0]
        if P.ndim != 2 or P.shape[1] != n:
            raise ValueError("P must be square (n x n).")
        if L.ndim != 2 or L.shape[1] != n:
            raise ValueError("L must have shape (m x n).")
        m = L.shape[0]

        Znn = np.zeros((n, n), dtype=float)
        Znm = np.zeros((n, m), dtype=float)
        Zmn = np.zeros((m, n), dtype=float)
        Zmm = np.zeros((m, m), dtype=float)

        A11 = alpha * P - float(nu) * np.eye(n, dtype=float)

        S = np.block([
            [A11,  Znn,   Znm,   Znn,   Znm,   Znn],
            [Znn,  -P,   -L.T,   -P,   -L.T,   Znn],
            [Zmn,  -L,    Zmm,   -L,    Zmm,    L ],
            [Znn,  -P,   -L.T,   -P,   -L.T,   Znn],
            [Zmn,  -L,    Zmm,   -L,    Zmm,    L ],
            [Znn,  Znn,   L.T,   Znn,   L.T,     P ],
        ])

        # Numerical symmetrization for hygiene
        return 0.5 * (S + S.T)


def pad_N(N: np.ndarray, total_dim: int) -> np.ndarray:
    """
    Create \tilde N = [[N, 0], [0, 0]] of size (total_dim x total_dim),
    placing N in the top-left corner.

    Args:
        N:          (r, r) matrix (e.g., N1 or N2 from sigma_data / sigma_delta)
        total_dim:  desired final dimension (>= r), typically S.shape[0]

    Returns:
        Nt: (total_dim, total_dim) array
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


def pad_N_for_S(N: np.ndarray, P: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Convenience: pad N into the top-left of a zero matrix sized to S(P,L,nu).

    Args:
        N: (r, r) — the compact QMI block (e.g., from make_N1 / make_N2)
        P: (n, n)
        L: (m, n)

    Returns:
        \tilde N with shape (4n+2m, 4n+2m)
    """
    n = np.asarray(P).shape[0]
    m = np.asarray(L).shape[0]
    return pad_N(N, total_dim=s_dim(n, m))


# Optional thin wrappers (handy for keeping call-sites tidy)
def build_tilde_N1(N1: np.ndarray, P: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return \tilde N_1 padded to the S-block size."""
    return pad_N_for_S(N1, P, L)


def build_tilde_N2(N2: np.ndarray, P: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return \tilde N_2 padded to the S-block size."""
    return pad_N_for_S(N2, P, L)