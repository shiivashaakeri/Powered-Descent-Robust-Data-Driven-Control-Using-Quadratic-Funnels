from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np  # pyright: ignore[reportMissingImports]

NormT = Literal["l2", "linf"]


def _align_quat(q_next: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    # Antipodal handling for quaternions (q ≡ -q)
    return -q_next if float(np.dot(q_next, q_cur)) < 0.0 else q_next


@dataclass(frozen=True)
class NominalBoundResult:
    increments: np.ndarray  # per-step vector increments ||[Δx; Δu]||
    v_max: float  # max increment
    v_p95: float  # 95th percentile (diagnostic)
    rate_sup: Optional[float] = None  # optional rate-based bound (Δt * sup ||[ẋ; u̇]||)


class NominalIncrementBoundCalculator:
    """
    Bound the nominal sequence increments:
      v = max_k || (x_nom(k+1), u_nom(k+1)) - (x_nom(k), u_nom(k)) ||.

    Optionally also compute a rate-based bound:
      v_rate ≤ Δt * sup_k || (ẋ_nom(t_k), u̇_nom(t_k)) ||,
    if f_nom (continuous-time nominal dynamics) and u̇_nom are available.
    """

    def __init__(
        self,
        dt: float,
        norm: NormT = "l2",
        quat_slice: Optional[slice] = None,
    ) -> None:
        self.dt = float(dt)
        if norm not in ("l2", "linf"):
            raise ValueError(f"Unsupported norm: {norm}")
        self.norm = norm
        self.quat_slice = quat_slice

    def _vec_norm(self, v: np.ndarray) -> float:
        if self.norm == "l2":
            return float(np.linalg.norm(v, ord=2))
        else:
            return float(np.linalg.norm(v, ord=np.inf))

    def from_increments(self, X_nom: np.ndarray, U_nom: np.ndarray) -> NominalBoundResult:
        X = np.asarray(X_nom)
        U = np.asarray(U_nom)
        _, K = X.shape
        # concatenate [x; u] increments
        inc = np.zeros((K - 1,), dtype=float)
        for k in range(K - 1):
            xk = X[:, k].copy()
            xkp1 = X[:, k + 1].copy()
            if self.quat_slice is not None:
                qk = xk[self.quat_slice]
                qk1 = xkp1[self.quat_slice]
                xkp1[self.quat_slice] = _align_quat(qk1, qk)
            dx = xkp1 - xk
            du = U[:, k + 1] - U[:, k]
            inc[k] = self._vec_norm(np.concatenate([dx, du], axis=0))
        v_max = float(np.max(inc)) if inc.size else 0.0
        v_p95 = float(np.percentile(inc, 95)) if inc.size else 0.0
        return NominalBoundResult(increments=inc, v_max=v_max, v_p95=v_p95, rate_sup=None)

    def from_rates(  # noqa: C901, PLR0912
        self,
        X_nom: np.ndarray,
        U_nom: np.ndarray,
        f_nom: Optional[Callable] = None,
        u_dot: Optional[np.ndarray] = None,
    ) -> NominalBoundResult:
        """
        Estimate a rate-based bound using ẋ = f_nom(x,u) and optional u̇.
        If u̇ not given, approximate with forward differences.
        """
        X = np.asarray(X_nom)
        U = np.asarray(U_nom)
        n_x, K = X.shape

        if f_nom is None:
            # Fall back to increments only
            return self.from_increments(X, U)

        xdot = np.zeros((n_x, K), dtype=float)
        for k in range(K):
            # Handle sympy lambdify output that may have weird shapes
            try:
                xdot_raw = f_nom(X[:, k].reshape(-1, 1), U[:, k].reshape(-1, 1))
                # Convert to list first to handle inhomogeneous arrays from sympy
                if hasattr(xdot_raw, "tolist"):
                    xdot_list = xdot_raw.tolist()
                elif hasattr(xdot_raw, "__iter__") and not isinstance(xdot_raw, (str, bytes)):
                    xdot_list = list(xdot_raw)
                else:
                    xdot_list = [xdot_raw]
                # Flatten nested lists and convert to float array
                xdot_arr = np.array(
                    [float(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v) for v in xdot_list], dtype=float
                )
                xdot[:, k] = xdot_arr.flatten()
            except (ValueError, TypeError, IndexError):
                # Last resort: manually evaluate each element
                xk_col = X[:, k].reshape(-1, 1)
                uk_col = U[:, k].reshape(-1, 1)
                for i in range(n_x):
                    try:
                        result = f_nom(xk_col, uk_col)
                        if hasattr(result, "__getitem__"):
                            try:
                                xdot[i, k] = float(result[i, 0])
                            except (IndexError, TypeError):
                                try:
                                    xdot[i, k] = float(result[i])
                                except (IndexError, TypeError):
                                    xdot[i, k] = 0.0
                        else:
                            xdot[i, k] = 0.0
                    except Exception:
                        xdot[i, k] = 0.0

        if u_dot is None:
            # forward diff for u (last sample: repeat previous rate)
            u_dot = np.zeros_like(U)
            u_dot[:, :-1] = (U[:, 1:] - U[:, :-1]) / self.dt
            u_dot[:, -1] = u_dot[:, -2] if K > 1 else 0.0

        # sup_k || [ẋ; u̇] ||
        sup_rates = 0.0
        for k in range(K):
            sup_rates = max(sup_rates, self._vec_norm(np.concatenate([xdot[:, k], u_dot[:, k]])))
        v_rate = self.dt * sup_rates

        # Also compute increments diagnostic
        base = self.from_increments(X, U)
        return NominalBoundResult(
            increments=base.increments, v_max=base.v_max, v_p95=base.v_p95, rate_sup=float(v_rate)
        )


# --- Backwards-compatible function wrapper (deprecated) ---
def nominal_increment_bounds(
    f_twin,
    X_nom: np.ndarray,
    U_nom: np.ndarray,
    dt: float,
    quat_slice: Optional[slice] = None,
) -> dict:
    """
    DEPRECATED: Use NominalIncrementBoundCalculator(dt, quat_slice).from_increments(...)
    or .from_rates(...) instead.

    Returns dict with keys: "v_disc", "v_ct_bound"
    """
    calc = NominalIncrementBoundCalculator(dt=dt, quat_slice=quat_slice, norm="l2")
    # Compute both increments and rates
    res_inc = calc.from_increments(X_nom, U_nom)
    res_rate = calc.from_rates(X_nom, U_nom, f_nom=f_twin, u_dot=None)
    return {
        "v_disc": res_inc.v_max,
        "v_ct_bound": res_rate.rate_sup if res_rate.rate_sup is not None else 0.0,
    }
