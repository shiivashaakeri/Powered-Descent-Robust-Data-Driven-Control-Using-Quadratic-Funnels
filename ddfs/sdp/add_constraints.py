# ddfs/sdp/add_constraints.py
from __future__ import annotations

from typing import Iterable, List, Union

import numpy as np  # pyright: ignore[reportMissingImports]
import cvxpy as cp  # type: ignore


NDArray = np.ndarray
CVXConstLike = Union[NDArray, cp.Expression]


def _sym(A: CVXConstLike) -> CVXConstLike:
    """Numerically symmetrize a constant/array; leave CVX variables/expressions alone."""
    if isinstance(A, np.ndarray):
        return 0.5 * (A + A.T)
    # If it's a CVX expression, assume caller provided symmetric input
    return A


def _to_const(X: CVXConstLike, name: str = "") -> cp.Expression:
    """
    Convert numpy array to a CVXPY Constant; pass through CVX expressions.
    """
    if isinstance(X, np.ndarray):
        return cp.Constant(X, name=name)  # Constant is fine; PSD not enforced here
    return X


def feasibility_constraints(
    P: cp.Variable,
    L: cp.Variable,
    P_min_i: CVXConstLike,
    R_max_i: CVXConstLike,
    eps: float = 0.0,
) -> List[cp.Constraint]:
    r"""
    Build feasibility LMIs for the quadratic funnel:

      1)  P ⪰ P_min_i
      2)  [ R_max_i   L  ] ⪰ 0
          [   Lᵀ      P  ]

    Args:
        P:         (n, n) symmetric variable (Lyapunov matrix).
        L:         (m, n) variable with L = K P (design variable in SDP).
        P_min_i:   (n, n) numpy array or CVX expression, segment envelope lower bound.
        R_max_i:   (m, m) numpy array or CVX expression, segment envelope upper bound for inputs.
        eps:       Optional small nonnegative slack for numerical robustness. If > 0, enforces
                   P - P_min_i ⪰ eps*I and block ⪰ eps*I.

    Returns:
        List of CVXPY constraints.
    """
    # Shapes
    n = int(P.shape[0])
    m = int(L.shape[0])

    if P.shape != (n, n):
        raise ValueError("P must be square (n x n).")
    if L.shape[1] != n:
        raise ValueError("L must have shape (m x n).")

    # Convert constants / symmetrize numerically
    Pmin = _to_const(_sym(np.asarray(P_min_i, dtype=float) if isinstance(P_min_i, np.ndarray) else P_min_i), name="Pmin")
    Rmax = _to_const(_sym(np.asarray(R_max_i, dtype=float) if isinstance(R_max_i, np.ndarray) else R_max_i), name="Rmax")

    I_n = np.eye(n, dtype=float)
    I_nm = np.eye(n + m, dtype=float)

    cons: List[cp.Constraint] = []

    # 1) State-feasibility:  P ⪰ P_min_i  (optionally tightened by eps*I)
    if eps > 0.0:
        cons.append(P - Pmin >> eps * I_n)
    else:
        cons.append(P >> Pmin)

    # 2) Input-feasibility via Schur block LMI:
    #    [ R_max_i  L ]
    #    [  Lᵀ      P ]  ⪰ 0
    block = cp.bmat([[Rmax, L],
                     [L.T,  P]])
    if eps > 0.0:
        cons.append(block >> eps * I_nm)
    else:
        cons.append(block >> 0)

    return cons


# Convenience splitters if you prefer adding them separately
def state_feasibility_constraint(P: cp.Variable, P_min_i: CVXConstLike, eps: float = 0.0) -> cp.Constraint:
    n = int(P.shape[0])
    Pmin = _to_const(_sym(np.asarray(P_min_i, dtype=float) if isinstance(P_min_i, np.ndarray) else P_min_i), name="Pmin")
    return (P - Pmin >> eps * np.eye(n)) if eps > 0.0 else (P >> Pmin)


def input_feasibility_constraint(P: cp.Variable, L: cp.Variable, R_max_i: CVXConstLike, eps: float = 0.0) -> cp.Constraint:
    n = int(P.shape[0]); m = int(L.shape[0])
    Rmax = _to_const(_sym(np.asarray(R_max_i, dtype=float) if isinstance(R_max_i, np.ndarray) else R_max_i), name="Rmax")
    blk = cp.bmat([[Rmax, L],
                   [L.T,  P]])
    return (blk >> eps * np.eye(n + m)) if eps > 0.0 else (blk >> 0)