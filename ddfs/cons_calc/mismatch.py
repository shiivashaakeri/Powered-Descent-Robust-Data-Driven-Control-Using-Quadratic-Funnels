from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np  # pyright: ignore[reportMissingImports]

from models.discretization import integrate_one_step  # uses rk4 + optional quat renorm

NormT = Literal["l2", "linf"]


@dataclass(frozen=True)
class MismatchResult:
    deltas: np.ndarray  # shape: (n_x, K-1)
    norms: np.ndarray  # shape: (K-1,)
    gamma: float  # max(norms)


class MismatchCalculator:
    """
    Compute Δ_k and gamma along a nominal trajectory:
        Δ_k = x_phys_next(k) - x_nom(k+1),
        gamma = max_k ||Δ_k||.

    Assumes X_nom, U_nom, and f are in the *same units* (we expect non-dimensional if the nominal is).
    """

    def __init__(
        self,
        f_continuous,
        dt: float,
        quat_slice: Optional[slice] = None,
        norm: NormT = "l2",
    ) -> None:
        self.f = f_continuous
        self.dt = float(dt)
        self.quat_slice = quat_slice
        if norm not in ("l2", "linf"):
            raise ValueError(f"Unsupported norm: {norm}")
        self.norm = norm

    def _vec_norm(self, v: np.ndarray) -> float:
        if self.norm == "l2":
            return float(np.linalg.norm(v, ord=2))
        else:
            return float(np.linalg.norm(v, ord=np.inf))

    def compute(self, X_nom: np.ndarray, U_nom: np.ndarray) -> MismatchResult:
        """
        X_nom: (n_x, K)
        U_nom: (n_u, K)
        Returns MismatchResult with per-step deltas and norms.
        """
        X = np.asarray(X_nom)
        U = np.asarray(U_nom)
        assert X.ndim == 2 and U.ndim == 2, "X_nom,U_nom shapes must be (n_x,K),(n_u,K)"
        n_x, K = X.shape
        deltas = np.zeros((n_x, K - 1), dtype=float)
        norms = np.zeros((K - 1,), dtype=float)

        for k in range(K - 1):
            xk = X[:, k]
            uk = U[:, k]
            # propagate one step with PHYSICAL model continuous dynamics
            xk1_phys = integrate_one_step(self.f, xk, uk, self.dt, quat_slice=self.quat_slice)
            xk1_ref = X[:, k + 1].copy()

            # Keep reference quaternion unit-norm for the diff too
            if self.quat_slice is not None:
                q = xk1_ref[self.quat_slice]
                nq = float(np.linalg.norm(q))
                if nq > 0.0:
                    xk1_ref[self.quat_slice] = q / nq

            dk = xk1_phys - xk1_ref
            deltas[:, k] = dk
            norms[k] = self._vec_norm(dk)

            if not np.isfinite(norms[k]):
                raise FloatingPointError(f"Non-finite Δ at k={k}: ||Δ||={norms[k]}")

        gamma = float(np.max(norms)) if norms.size else 0.0
        return MismatchResult(deltas=deltas, norms=norms, gamma=gamma)


# --- Backwards-compatible function wrapper (deprecated) ---
def deltas_and_gamma(f_continuous, X_nom: np.ndarray, U_nom: np.ndarray, dt: float, quat_slice: Optional[slice] = None):
    """
    DEPRECATED: Use MismatchCalculator(f, dt, quat_slice).compute(X, U) instead.
    """
    calc = MismatchCalculator(f_continuous, dt=dt, quat_slice=quat_slice, norm="l2")
    res = calc.compute(X_nom, U_nom)
    return res.deltas, res.norms, res.gamma
