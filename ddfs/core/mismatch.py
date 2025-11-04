# ddfs/core/mismatch.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]

from models.discretization import integrate_one_step


def deltas_and_gamma(
    f, X_nom: np.ndarray, U_nom: np.ndarray, dt: float, quat_slice: Optional[slice] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    n, K = X_nom.shape
    deltas = np.empty((n, K - 1), dtype=float)
    norms = np.empty((K - 1,), dtype=float)

    for k in range(K - 1):
        xk = X_nom[:, k].astype(float)
        uk = U_nom[:, k].astype(float)

        xkp1_pred = integrate_one_step(f, xk, uk, float(dt), quat_slice)
        xkp1_ref = X_nom[:, k + 1].astype(float)

        # Keep reference quaternion unit-norm for the diff too
        if quat_slice is not None:
            q = xkp1_ref[quat_slice]
            nq = float(np.linalg.norm(q))
            if nq > 0.0:
                xkp1_ref[quat_slice] = q / nq

        dk = xkp1_pred - xkp1_ref

        # Optional: replace raw quaternion vec diff by rotation-vector error
        # (uncomment if you prefer geodesic metric for orientation)
        # if quat_slice is not None:
        #     q_pred = xkp1_pred[quat_slice]
        #     q_ref = xkp1_ref[quat_slice]
        #     q_pred /= np.linalg.norm(q_pred) + 1e-12
        #     q_ref /= np.linalg.norm(q_ref) + 1e-12
        #     dot = np.clip(np.dot(q_ref, q_pred), -1.0, 1.0)
        #     angle = 2.0 * np.arccos(abs(dot))
        #     axis = q_pred - dot * q_ref
        #     axis /= np.linalg.norm(axis) + 1e-12
        #     rotvec = angle * axis
        #     dk[quat_slice] = rotvec  # 3 numbers if you replace slice handling

        deltas[:, k] = dk
        norms[k] = float(np.linalg.norm(dk, ord=2))

        if not np.isfinite(norms[k]):
            raise FloatingPointError(f"Non-finite Δ at k={k}: ||Δ||={norms[k]}")

    gamma = float(np.max(norms)) if norms.size else 0.0
    return deltas, norms, gamma
