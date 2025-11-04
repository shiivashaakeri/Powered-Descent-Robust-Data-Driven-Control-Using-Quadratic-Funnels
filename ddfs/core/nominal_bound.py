# ddfs/core/nominal_bound.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np  # pyright: ignore[reportMissingImports]


def _align_quat(q_next: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    # Antipodal handling for quaternions (q ≡ -q)
    return -q_next if float(np.dot(q_next, q_cur)) < 0.0 else q_next


def _disc_bound(X: np.ndarray, U: np.ndarray, quat_slice: Optional[slice]) -> float:
    """
    v_disc = max_k || [X[:,k+1]-X[:,k] ; U[:,k+1]-U[:,k]] ||_2
    """
    K = X.shape[1]
    if K < 2:
        return 0.0

    norms = np.empty(K - 1, dtype=float)
    for k in range(K - 1):
        xk = X[:, k].copy()
        xkp1 = X[:, k + 1].copy()
        if quat_slice is not None:
            qk = xk[quat_slice]
            qk1 = xkp1[quat_slice]
            xkp1[quat_slice] = _align_quat(qk1, qk)
        dx = xkp1 - xk
        du = U[:, k + 1] - U[:, k]
        norms[k] = float(np.linalg.norm(np.concatenate([dx, du], axis=0)))
    return float(np.max(norms))


def _ct_bound(f, X: np.ndarray, U: np.ndarray, dt: float) -> float:  # noqa: C901, PLR0912
    """
    v_ct_bound = dt * max_k || [f(X_k, U_k) ; dU/dt(k)] ||_2
    where dU/dt via finite difference on the nominal control.
    """
    K = X.shape[1]
    if K == 0:
        return 0.0

    # u_dot (central, with one-sided ends)
    u_dot = np.zeros_like(U)
    if K >= 3:
        u_dot[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / (2.0 * dt)
    if K >= 2:
        u_dot[:, 0] = (U[:, 1] - U[:, 0]) / dt
        u_dot[:, -1] = (U[:, -1] - U[:, -2]) / dt

    sup_norm = 0.0
    for k in range(K):
        xk = X[:, k].astype(float)
        uk = U[:, k].astype(float)
        # f expects (n,1) column vectors; returns (n,1) or (n,) - flatten to ensure 1D
        # Handle sympy lambdify output that may have weird shapes
        # Try to evaluate and convert manually if np.asarray fails
        try:
            xdot_raw = f(xk.reshape(-1, 1), uk.reshape(-1, 1))
            # Convert to list first to handle inhomogeneous arrays from sympy
            if hasattr(xdot_raw, 'tolist'):
                xdot_list = xdot_raw.tolist()
            elif hasattr(xdot_raw, '__iter__') and not isinstance(xdot_raw, (str, bytes)):
                xdot_list = list(xdot_raw)
            else:
                xdot_list = [xdot_raw]
            # Flatten nested lists and convert to float array
            xdot = np.array([float(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v) for v in xdot_list], dtype=float)  # noqa: E501
        except (ValueError, TypeError, IndexError):
            # Last resort: manually evaluate each element
            xdot = np.zeros(X.shape[0], dtype=float)
            xk_col = xk.reshape(-1, 1)
            uk_col = uk.reshape(-1, 1)
            for i in range(X.shape[0]):
                try:
                    # Try calling f and extracting just element i
                    result = f(xk_col, uk_col)
                    if hasattr(result, '__getitem__'):
                        try:
                            xdot[i] = float(result[i, 0])
                        except (IndexError, TypeError):
                            try:
                                xdot[i] = float(result[i])
                            except (IndexError, TypeError):
                                xdot[i] = 0.0
                    else:
                        xdot[i] = 0.0
                except Exception:
                    xdot[i] = 0.0
        z = np.concatenate([xdot, u_dot[:, k]], axis=0)
        sup_norm = max(sup_norm, float(np.linalg.norm(z)))
    return float(dt * sup_norm)


def nominal_increment_bounds(
    f_twin, X: np.ndarray, U: np.ndarray, dt: float, quat_slice: Optional[slice]
) -> Dict[str, float]:
    """
    Compute both discrete and CT-based bounds on the nominal increments.
    Assumes X,U are already in the same (non-dimensional) units as the twin.
    """
    v_disc = _disc_bound(X, U, quat_slice)
    v_ct = _ct_bound(f_twin, X, U, dt)
    return {"v_disc": v_disc, "v_ct_bound": v_ct}
