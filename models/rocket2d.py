# models/rocket2d.py
from __future__ import annotations

from typing import List, Tuple

import cvxpy as cvx  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import sympy as sp  # pyright: ignore[reportMissingImports]

from .base import ModelBase


class Rocket2D(ModelBase):
    """
    2D planar rocket landing (Up-East plane).
    State x = [rx, ry, vx, vy, theta, omega]  (6)
    Control u = [gimbal, T]                    (2)
    """

    n_x, n_u = 6, 2

    # Physical parameters (pre-scaling)
    m = 2.0
    I = 1e-2
    g = 1.0
    r_T = 1e-2  # noqa: N815
    T_max = 5.0
    T_min = 0.4 * T_max
    t_max = np.deg2rad(60.0)
    w_max = np.deg2rad(60.0)
    max_gimbal = np.deg2rad(7.0)

    # Boundary conditions (pre-scaling)
    r_init = np.array([4.0, 4.0])
    v_init = np.array([-2.0, -1.0])
    t_init = np.array([0.0])
    w_init = np.array([0.0])

    r_final = np.array([0.0, 0.0])
    v_final = np.array([0.0, 0.0])
    t_final = np.array([0.0])
    w_final = np.array([0.0])

    # Warm start
    t_f_guess = 10.0  # fixed final time [s]

    def __init__(self) -> None:
        super().__init__()
        # Scales
        self.r_scale = float(np.linalg.norm(self.r_init))
        self.m_scale = float(self.m)

        # Boundary state vectors
        self.x_init = np.concatenate([self.r_init, self.v_init, self.t_init, self.w_init]).astype(float)
        self.x_final = np.concatenate([self.r_final, self.v_final, self.t_final, self.w_final]).astype(float)

    # ---------- Scaling ----------
    def nondimensionalize(self) -> None:
        # Parameter scaling
        self.r_T /= self.r_scale
        self.g /= self.r_scale
        self.I /= self.m_scale * self.r_scale**2
        self.m /= self.m_scale
        self.T_min /= self.m_scale * self.r_scale
        self.T_max /= self.m_scale * self.r_scale

        # States
        self.x_init = self.x_nondim(self.x_init.copy())
        self.x_final = self.x_nondim(self.x_final.copy())

    def redimensionalize(self) -> None:
        self.r_T *= self.r_scale
        self.g *= self.r_scale
        self.I *= self.m_scale * self.r_scale**2
        self.m *= self.m_scale
        self.T_min *= self.m_scale * self.r_scale
        self.T_max *= self.m_scale * self.r_scale

        self.x_init = self.x_redim(self.x_init.copy())
        self.x_final = self.x_redim(self.x_final.copy())

    def x_nondim(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[0:4] /= self.r_scale
        return x

    def x_redim(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[0:4] *= self.r_scale
        return x

    def u_nondim(self, u: np.ndarray) -> np.ndarray:
        u = u.copy()
        u[1, :] /= self.m_scale * self.r_scale
        return u

    def u_redim(self, u: np.ndarray) -> np.ndarray:
        u = u.copy()
        u[1, :] *= self.m_scale * self.r_scale
        return u

    # ---------- Dynamics ----------
    def get_equations(self):
        f = sp.zeros(self.n_x, 1)

        rx, ry, vx, vy, th, om = sp.symbols("rx ry vx vy t w", real=True)
        gimbal, T = sp.symbols("gimbal T", real=True)

        x = sp.Matrix([rx, ry, vx, vy, th, om])
        u = sp.Matrix([gimbal, T])

        m = sp.Float(self.m)
        I = sp.Float(self.I)
        g = sp.Float(self.g)
        r_T = sp.Float(self.r_T)

        f[0, 0] = vx
        f[1, 0] = vy
        f[2, 0] = (1.0 / m) * sp.sin(th + gimbal) * T
        f[3, 0] = (1.0 / m) * (sp.cos(th + gimbal) * T - m * g)
        f[4, 0] = om
        f[5, 0] = (1.0 / I) * (-sp.sin(gimbal) * T * r_T)

        f = sp.simplify(f)
        A = sp.simplify(f.jacobian(x))
        B = sp.simplify(f.jacobian(u))

        f_func = sp.lambdify((x, u), f, "numpy")
        A_func = sp.lambdify((x, u), A, "numpy")
        B_func = sp.lambdify((x, u), B, "numpy")
        return f_func, A_func, B_func

    # ---------- Warm start ----------
    def initialize_trajectory(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K = X.shape[1]
        for k in range(K):
            a1 = (K - k) / K
            a2 = k / K
            X[:, k] = a1 * self.x_init + a2 * self.x_final

        # Controls: zero gimbal, mid-thrust
        U[0, :] = 0.0
        U[1, :] = 0.5 * (self.T_min + self.T_max)
        return X, U

    # ---------- SCVX pieces ----------
    def get_constraints(
        self,
        X_v: cvx.Expression,
        U_v: cvx.Expression,
        X_last_p: cvx.Expression,  # noqa: ARG002
        U_last_p: cvx.Expression,  # noqa: ARG002
    ) -> List[cvx.Constraint]:
        K = X_v.shape[1]  # noqa: F841

        cons: List[cvx.Constraint] = [
            # Boundary conditions (full state at end, partial at start)
            X_v[0:2, 0] == self.x_init[0:2],
            X_v[2:4, 0] == self.x_init[2:4],
            X_v[4, 0] == self.x_init[4],
            X_v[5, 0] == self.x_init[5],
            X_v[:, -1] == self.x_final,
            # State limits
            cvx.abs(X_v[4, :]) <= self.t_max,  # |theta|
            cvx.abs(X_v[5, :]) <= self.w_max,  # |omega|
            X_v[1, :] >= 0.0,  # altitude nonnegative
            # Control limits
            cvx.abs(U_v[0, :]) <= self.max_gimbal,  # gimbal
            U_v[1, :] >= self.T_min,  # thrust bounds
            U_v[1, :] <= self.T_max,
        ]
        return cons

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):  # noqa: ARG002
        # No model-specific penalty for the 2D nominal case
        return None
