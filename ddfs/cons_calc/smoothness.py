# ddfs/cons_calc/smoothness.py

from __future__ import annotations
from dataclasses import dataclass

from typing import Callable, List, Optional, Tuple, Dict, Any

import numpy as np # pyright: ignore[reportMissingImports]
import sympy as sp # pyright: ignore[reportMissingImports]

from ddfs.utils.numeric import deg2rad, rand_unit, spectral_norm
from ddfs.utils.quat import quat_mul, quat_from_euler_xyz
from ddfs.utils.sampling import rng_box, rng_box_scalar

from models.sym_rocket2d import build_symbolics_rocket2d
from models.sym_rocket6dof import build_symbolics_rocket6dof

@dataclass
class SmoothnessDiagnostics:
    L_J: float
    method: str
    samples_evaluated: int
    per_point_scores: List[float]
    notes: str = ""

class LipschitzJacobianCalculator:
    """
    Compute L_J (default: Hessian aggregation). Fallback: finite-difference on Jacobians.
    All in nondimensional coordinates.
    """
    def __init__(
        self,
        model_name: str,
        model_params_nd: Dict[str, Any],
        quat_slice: Optional[slice] = None,
        rng_seed: int = 123
    ):
        self.model_name = model_name.lower()
        self.params = model_params_nd
        self.quat_slice = quat_slice
        self.rng = np.random.default_rng(rng_seed)

        if self.model_name == "rocket2d":
            self._x_sym, self._u_sym, self._f_sym, self._H_funcs = build_symbolics_rocket2d(self.params)
        elif self.model_name == "rocket6dof":
            self._x_sym, self._u_sym, self._f_sym, self._H_funcs = build_symbolics_rocket6dof(self.params)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
    
    # Public methods
    def via_hessians(
        self,
        X_nom: np.ndarray,
        U_nom: np.ndarray,
        S_tube_nd: Dict[str, Any],
        n_times: int = 32,
        n_points_per_time: int = 4,
    ) -> SmoothnessDiagnostics:
        idxs = self._pick_time_indices(X_nom.shape[1], n_times)
        per_point: List[float] = []
        for k in idxs:
            x0, u0 = X_nom[:, k], U_nom[:, k]
            for _ in range(n_points_per_time):
                x_s, u_s = self._sample_in_tube(x0, u0, S_tube_nd)
                per_point.append(self._hessian_agg_norm_at(x_s, u_s))
        LJ = float(np.max(per_point)) if per_point else 0.0
        return SmoothnessDiagnostics(LJ, "hessian_agg", len(per_point), per_point)
    
    def via_finite_diff(
        self,
        A_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        B_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        X_nom: np.ndarray,
        U_nom: np.ndarray,
        S_tube_nd: Dict[str, Any],
        n_times: int = 32,
        n_dirs: int = 8,
        eps_rel: float = 1e-3,
    ) -> SmoothnessDiagnostics:

        n_x, n_u = X_nom.shape[0], U_nom.shape[0]
        n_z = n_x + n_u

        idxs = self._pick_time_indices(X_nom.shape[1], n_times)
        per_point: List[float] = []
        for k in idxs:
            x0, u0 = X_nom[:, k], U_nom[:, k]
            J0 = np.hstack((np.asarray(A_func(x0, u0), float), np.asarray(B_func(x0, u0), float)))
            radii = self._tube_radii_vec(S_tube_nd, n_x, n_u)
            eps = eps_rel * max(1e-8, float(np.linalg.norm(radii, 2)))
            for _ in range(n_dirs):
                d = rand_unit(self.rng, n_z)
                dx, du = d[:n_x] * eps, d[n_x:] * eps
                x_s, u_s = self._apply_perturbation(x0, u0, dx, du)
                J1 = np.hstack((np.asarray(A_func(x_s, u_s), float), np.asarray(B_func(x_s, u_s), float)))
                per_point.append(spectral_norm(J1 - J0)/ eps)
        LJ = float(np.max(per_point)) if per_point else 0.0
        return SmoothnessDiagnostics(LJ, "fd_jacobian", len(per_point), per_point)

    # Private methods
    def _hessian_agg_norm_at(self, s: np.ndarray, u: np.ndarray) -> float:
        """
        inputs: s: state R^n_x, u: control R^n_u
        output: Lipschitz constant of the Jacobian of f at (s,u)
        """
        s2 = 0.0
        for Hf in self._H_funcs:
            H  = np.asarray(Hf(s, u), float)
            s2 += spectral_norm(H)**2
        return float(np.sqrt(s2))
    
    @staticmethod
    def _pick_time_indices(K: int, n_times: int) -> List[int]:
        """
        inputs: K ∈ N: number of time steps, n_times ∈ N: number of time steps to sample
        output: list of n_times indices in [0, K-1]
        """
        if K <= 0 or n_times <= 0:
            return []
        if n_times >= K:
            return list(range(K))
        return [int(round(i * (K-1) / (n_times-1))) for i in range(n_times)]
    
    def _sample_in_tube(self, x0: np.ndarray, u0: np.ndarray, S_tube_nd: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        inputs: x0: state R^n_x, u0: control R^n_u, S_tube_nd: tube radii
        output: x_s: state R^n_x, u_s: control R^n_u
        """
        n_x, n_u = x0.shape[0], u0.shape[0]
        dx, du = self._random_box_delta(S_tube_nd, n_x, n_u)
        return self._apply_perturbation(x0, u0, dx, du)
    
    def _apply_perturbation(self, x0: np.ndarray, u0: np.ndarray, dx: np.ndarray, du: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        inputs: x0: state R^n_x, u0: control R^n_u, dx: perturbation R^n_x, du: perturbation R^n_u
        output: x_s: state R^n_x, u_s: control R^n_u
        """
        x, u = x0.copy(), u0.copy()
        if self.model_name == "rocket2d":
            x[:4] += dx[:4]
            x[4] += dx[4]
            x[5] += dx[5]
            u[0] += du[0]
            u[1] += du[1]
            return x, u
        
        # 6-DoF: x = [m, r_I(3), v_I(3), q_BI(4), w_B(3)], u = [T_B(3)]
        elif self.model_name == "rocket6dof":
            x[0] += dx[0]
            x[1:4] += dx[1:4]
            x[4:7] += dx[4:7]
            dq_euler = dx[7:10]
            q_nom = x[7:11].copy()
            q_delta = quat_from_euler_xyz(dq_euler)
            q_new = quat_mul(q_delta, q_nom)
            n = float(np.linalg.norm(q_new))
            if n > 0:
                q_new /= n
            x[7:11] = q_new
            x[11:14] += dx[10:13]
            return x, u
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _random_box_delta(self, S_tube_nd: Dict[str, Any], n_x: int, n_u: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_name == "rocket2d":
            dr = np.array(S_tube_nd["state"]["dr_max_nd"], float)
            dv = np.array(S_tube_nd["state"]["dv_max_nd"], float)
            dth = float(S_tube_nd["state"]["dtheta_max_rad"])
            dw  = float(S_tube_nd["state"]["domega_max_radps"])
            dT  = float(S_tube_nd["input"]["dT_max_nd"])
            dg  = float(S_tube_nd["input"]["dgimbal_max_rad"])
            dx = np.zeros(n_x, float)
            dx[0:2] = rng_box(self.rng, dr)
            dx[2:4] = rng_box(self.rng, dv)
            dx[4]   = rng_box_scalar(self.rng, dth)
            dx[5]   = rng_box_scalar(self.rng, dw)
            du = np.zeros(n_u, float)
            du[0] = rng_box_scalar(self.rng, dg)
            du[1] = rng_box_scalar(self.rng, dT)
            return dx, du

        dr = np.array(S_tube_nd["state"]["dr_max_nd"], float)
        dv = np.array(S_tube_nd["state"]["dv_max_nd"], float)
        de = np.array(S_tube_nd["state"]["deuler_max_rad"], float)
        dw = np.array(S_tube_nd["state"]["domega_max_radps"], float)
        dm = float(S_tube_nd["state"].get("dm_nd", 0.0))
        dT = float(S_tube_nd["input"]["dT_max_nd"])
        dx = np.zeros(n_x, float)
        dx[0]     = rng_box_scalar(self.rng, dm)
        dx[1:4]   = rng_box(self.rng, dr)
        dx[4:7]   = rng_box(self.rng, dv)
        dx[7:10]  = rng_box(self.rng, de)   # euler packed, applied later
        dx[10:13] = rng_box(self.rng, dw)
        du = rng_box(self.rng, np.array([dT, dT, dT], float))
        return dx, du

    def _tube_radii_vec(self, S_tube_nd: Dict[str, Any], n_x: int, n_u: int) -> np.ndarray:  # noqa: ARG002
        if self.model_name == "rocket2d":
            rx = np.array(S_tube_nd["state"]["dr_max_nd"] + S_tube_nd["state"]["dv_max_nd"], float)
            rest = np.array([S_tube_nd["state"]["dtheta_max_rad"], S_tube_nd["state"]["domega_max_radps"]], float)
            inp = np.array([S_tube_nd["input"]["dgimbal_max_rad"], S_tube_nd["input"]["dT_max_nd"]], float)
            return np.concatenate([rx, rest, inp], axis=0)

        dr = np.array(S_tube_nd["state"]["dr_max_nd"], float)
        dv = np.array(S_tube_nd["state"]["dv_max_nd"], float)
        de = np.array(S_tube_nd["state"]["deuler_max_rad"], float)
        dw = np.array(S_tube_nd["state"]["domega_max_radps"], float)
        dm = float(S_tube_nd["state"].get("dm_nd", 0.0))
        inp = np.array([S_tube_nd["input"]["dT_max_nd"]]*3, float)
        return np.concatenate([[dm], dr, dv, de, dw, inp], axis=0)


def convert_S_tube_to_nondim(
    model_name: str,
    S_tube_physical: Dict[str, Any],
    r_scale: float,
    m_scale: float,
) -> Dict[str, Any]:
    """Physical → non-dimensional tube widths, matching model scaling."""
    if model_name.lower() == "rocket2d":
        st, ip = S_tube_physical["state"], S_tube_physical["input"]
        return {
            "state": {
                "dr_max_nd":      (np.asarray(st["dr_max_m"], float) / r_scale).tolist(),
                "dv_max_nd":      (np.asarray(st["dv_max_mps"], float) / r_scale).tolist(),
                "dtheta_max_rad": float(deg2rad(st["dtheta_max_deg"])),
                "domega_max_radps": float(deg2rad(st["domega_max_degps"])),
            },
            "input": {
                "dT_max_nd":       float(ip["dT_max_N"]) / (m_scale * r_scale),
                "dgimbal_max_rad": float(deg2rad(ip["dgimbal_max_deg"])),
            },
        }

    st, ip = S_tube_physical["state"], S_tube_physical["input"]
    return {
        "state": {
            "dr_max_nd":        (np.asarray(st["dr_max_m"], float) / r_scale).tolist(),
            "dv_max_nd":        (np.asarray(st["dv_max_mps"], float) / r_scale).tolist(),
            "deuler_max_rad":   (deg2rad(np.asarray(st["deuler_max_deg"], float))).tolist(),
            "domega_max_radps": (deg2rad(np.asarray(st["domega_max_degps"], float))).tolist(),
            "dm_nd":            float(st.get("dm_kg", 0.0)) / m_scale,
        },
        "input": {
            "dT_max_nd": float(ip["dT_max_N"]) / (m_scale * r_scale),
        },
    }
        