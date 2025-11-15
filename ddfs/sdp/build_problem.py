# ddfs/sdp/build_problem.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import cvxpy as cp  # type: ignore

from ddfs.sdp.slemma_blocks import S_block, pad_N
from ddfs.sdp.add_constraints import feasibility_constraints


def _as_const(M: np.ndarray, name: str = "") -> cp.Expression:
    if not isinstance(M, np.ndarray):
        raise TypeError(f"{name or 'matrix'} must be a numpy array.")
    # Symmetrize numerically (helps tiny asymmetries from upstream ops)
    M = 0.5 * (M + M.T)
    return cp.Constant(M, name=name)


def build_problem(
    *,
    n: int,
    m: int,
    alpha: float,
    P_min_i: np.ndarray,
    R_max_i: np.ndarray,
    N1: np.ndarray,
    N2: np.ndarray,
    eps_psd: float = 1e-9,
) -> Tuple[cp.Problem, Dict[str, cp.Expression]]:
    """
    Build the LMI program:

        maximize   logdet(P)
        subject to S(P,L,nu) - λ1 * ~N1 - λ2 * ~N2 ⪰ 0
                   P ⪰ P_min_i
                   [[R_max_i, L],
                    [Lᵀ,     P]] ⪰ 0

    Args:
        n, m:        state and input dimensions.
        alpha:       decay factor in (0,1) for V_{k+1} - α V_k ≤ 0.
        P_min_i:     (n×n) segment state envelope (lower bound).
        R_max_i:     (m×m) segment input envelope (upper bound).
        N1:          data QMI block from Sigma_data (shape (2n+m)×(2n+m)).
        N2:          variation QMI block from Sigma_delta (shape (2n+m)×(2n+m)).
        eps_psd:     tiny PSD slack for numerical robustness.

    Returns:
        (problem, vars) where vars contains {"P","L","lam1","lam2","nu"}.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    # Decision variables
    P = cp.Variable((n, n), PSD=True, name="P")          # Lyapunov
    L = cp.Variable((m, n), name="L")                    # L = K P (implicit)
    lam1 = cp.Variable(nonneg=True, name="lambda1")      # S-lemma multipliers
    lam2 = cp.Variable(nonneg=True, name="lambda2")
    nu = cp.Variable(nonneg=True, name="nu")             # Schur/strictness scalar

    # Core S-lemma block
    S = S_block(P=P, L=L, nu=nu, alpha=alpha)            # shape: (4n+2m)×(4n+2m)
    dim_S = int(S.shape[0])

    # Pad N1, N2 up to S's dimension
    N1p = _as_const(pad_N(N1, total_dim=dim_S), name="N1_tilde")
    N2p = _as_const(pad_N(N2, total_dim=dim_S), name="N2_tilde")

    # Feasibility LMIs
    cons = []
    cons += feasibility_constraints(P, L, P_min_i=P_min_i, R_max_i=R_max_i, eps=eps_psd)

    # Robust decay LMI (add tiny eps to help numerics)
    cons.append(S - lam1 * N1p - lam2 * N2p >> eps_psd * np.eye(dim_S))

    # Objective: maximize logdet(P) (implicitly pushes P ≻ 0)
    obj = cp.Maximize(cp.log_det(P))

    prob = cp.Problem(obj, cons)

    vars_dict = {"P": P, "L": L, "lam1": lam1, "lam2": lam2, "nu": nu}
    return prob, vars_dict