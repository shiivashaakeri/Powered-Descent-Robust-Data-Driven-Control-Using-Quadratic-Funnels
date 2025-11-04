# models/rocket6dof.py
from __future__ import annotations

from typing import List, Optional, Tuple

import cvxpy as cvx  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import sympy as sp  # pyright: ignore[reportMissingImports]

from .base import ModelBase
from .frames import dcm_from_quat_sym, euler_to_quat, omega_sym, skew_sym


class Rocket6DoF(ModelBase):
    """
    6-DoF rocket landing (inertial frame: East, North, Up).
    State x = [m, r_I(3), v_I(3), q_BI(4), w_B(3)]     -> (14,)
    Control u = T_B (body-frame thrust vector)         -> (3,)
    """

    n_x, n_u = 14, 3

    # Mass
    m_wet = 30000.0
    m_dry = 22000.0

    # Time guess (free-final-time runs)
    t_f_guess = 15.0

    # Limits (angles in degrees here, converted to radians immediately)
    max_gimbal_deg = 7.0
    max_angle_deg = 70.0  # tilt
    glide_slope_deg = 20.0
    max_body_rate_deg = 90.0

    # Thrust limits
    T_max = 800000.0
    T_min = 0.4 * T_max

    # Inertia (diagonal)
    J_B = np.diag([4_000_000.0, 4_000_000.0, 100_000.0])

    # Gravity
    g_I = np.array([0.0, 0.0, -9.81])  # noqa: N815

    # Fuel constant alpha_m = 1/(Isp*g0)
    alpha_m = 1.0 / (282.0 * 9.81)

    # Vector from CoM to thrust point (body)
    r_T_B = np.array([0.0, 0.0, -14.0])  # noqa: N815

    # Boundary (defaults; may be randomized in __init__)
    r_I_init = np.array([0.0, 200.0, 200.0])  # noqa: N815
    v_I_init = np.array([-50.0, -100.0, -50.0])  # noqa: N815
    q_B_I_init = euler_to_quat((0.0, 0.0, 0.0))  # noqa: N815
    w_B_init = np.deg2rad(np.array([0.0, 0.0, 0.0]))  # noqa: N815

    r_I_final = np.array([0.0, 0.0, 0.0])  # noqa: N815
    v_I_final = np.array([0.0, 0.0, -5.0])  # noqa: N815
    q_B_I_final = euler_to_quat((0.0, 0.0, 0.0))  # noqa: N815
    w_B_final = np.deg2rad(np.array([0.0, 0.0, 0.0]))  # noqa: N815

    # Lazily-created slack for lower-thrust linearization
    _s_prime: Optional[cvx.Variable] = None

    def __init__(self) -> None:
        super().__init__()

        # Convert angular limits to radians & derived constants
        self.tan_delta_max = float(np.tan(np.deg2rad(self.max_gimbal_deg)))
        self.cos_theta_max = float(np.cos(np.deg2rad(self.max_angle_deg)))
        self.tan_gamma_gs = float(np.tan(np.deg2rad(self.glide_slope_deg)))
        self.w_B_max = float(np.deg2rad(self.max_body_rate_deg))

        # Scales
        self.r_scale = float(np.linalg.norm(self.r_I_init))
        self.m_scale = float(self.m_wet)

        # Construct boundary states
        self.x_init = np.concatenate(
            [
                np.array([self.m_wet]),
                self.r_I_init,
                self.v_I_init,
                self.q_B_I_init,
                self.w_B_init,
            ]
        ).astype(float)

        self.x_final = np.concatenate(
            [
                np.array([self.m_dry]),
                self.r_I_final,
                self.v_I_final,
                self.q_B_I_final,
                self.w_B_final,
            ]
        ).astype(float)

    # ---------- Scaling ----------
    def nondimensionalize(self) -> None:
        # Parameters
        self.alpha_m *= self.r_scale  # s
        self.r_T_B /= self.r_scale
        self.g_I /= self.r_scale
        self.J_B = self.J_B / (self.m_scale * self.r_scale**2)

        # States
        self.x_init = self.x_nondim(self.x_init.copy())
        self.x_final = self.x_nondim(self.x_final.copy())

        # Thrust
        self.T_max = self.u_nondim_scalar(self.T_max)
        self.T_min = self.u_nondim_scalar(self.T_min)

        # Masses
        self.m_wet /= self.m_scale
        self.m_dry /= self.m_scale

    def redimensionalize(self) -> None:
        self.alpha_m /= self.r_scale
        self.r_T_B *= self.r_scale
        self.g_I *= self.r_scale
        self.J_B = self.J_B * (self.m_scale * self.r_scale**2)

        self.T_max = self.u_redim_scalar(self.T_max)
        self.T_min = self.u_redim_scalar(self.T_min)

        self.m_wet *= self.m_scale
        self.m_dry *= self.m_scale

    def x_nondim(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[0] /= self.m_scale  # mass
        x[1:4] /= self.r_scale  # r
        x[4:7] /= self.r_scale  # v
        # q unchanged
        # w unchanged (1/s)  (time scaling not applied here)
        return x

    def x_redim(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x[0] *= self.m_scale
        x[1:4] *= self.r_scale
        x[4:7] *= self.r_scale
        return x

    def u_nondim(self, u: np.ndarray) -> np.ndarray:
        """For vectorized controls (3,K)."""
        return u / (self.m_scale * self.r_scale)

    def u_redim(self, u: np.ndarray) -> np.ndarray:
        return u * (self.m_scale * self.r_scale)

    def u_nondim_scalar(self, val: float) -> float:
        return val / (self.m_scale * self.r_scale)

    def u_redim_scalar(self, val: float) -> float:
        return val * (self.m_scale * self.r_scale)

    # ---------- Dynamics ----------
    def get_equations(self):
        f = sp.zeros(self.n_x, 1)

        # Symbols
        m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = sp.symbols(
            "m rx ry rz vx vy vz q0 q1 q2 q3 wx wy wz", real=True
        )
        ux, uy, uz = sp.symbols("ux uy uz", real=True)

        x = sp.Matrix([m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz])
        u = sp.Matrix([ux, uy, uz])

        g_I = sp.Matrix(self.g_I.tolist())
        r_T_B = sp.Matrix(self.r_T_B.tolist())
        J_B = sp.Matrix(self.J_B.tolist())

        C_BI = dcm_from_quat_sym(x[7:11, 0])  # body->inertial
        C_IB = C_BI.T  # inertial->body

        # Dynamics
        f[0, 0] = -sp.Float(self.alpha_m) * u.norm()  # mdot
        f[1:4, 0] = x[4:7, 0]  # rdot = v
        f[4:7, 0] = (1.0 / x[0, 0]) * C_IB * u + g_I  # vdot = (1/m) C_IB u + g
        f[7:11, 0] = sp.Rational(1, 2) * omega_sym(x[11:14, 0]) * x[7:11, 0]  # qdot
        f[11:14, 0] = J_B**-1 * (skew_sym(r_T_B) * u) - skew_sym(x[11:14, 0]) * x[11:14, 0]  # wdot

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
            m_k = a1 * self.x_init[0] + a2 * self.x_final[0]
            r_k = a1 * self.x_init[1:4] + a2 * self.x_final[1:4]
            v_k = a1 * self.x_init[4:7] + a2 * self.x_final[4:7]
            q_k = np.array([1.0, 0.0, 0.0, 0.0])  # identity (keeps tilt constraints simple)
            w_k = a1 * self.x_init[11:14] + a2 * self.x_final[11:14]
            X[:, k] = np.concatenate([[m_k], r_k, v_k, q_k, w_k])

        # Thrust initial guess: mid between bounds, along +body z
        U[:, :] = (0.5 * (self.T_min + self.T_max)) * np.array([[0.0], [0.0], [1.0]])
        return X, U

    # ---------- SCVX pieces ----------
    def _ensure_s_prime(self, K: int) -> cvx.Variable:
        if self._s_prime is None or self._s_prime.shape != (K, 1):
            self._s_prime = cvx.Variable((K, 1), nonneg=True)
        return self._s_prime

    def get_constraints(
        self,
        X_v: cvx.Expression,
        U_v: cvx.Expression,
        X_last_p: cvx.Expression,  # noqa: ARG002
        U_last_p: cvx.Expression,
    ) -> List[cvx.Constraint]:
        K = X_v.shape[1]
        s_prime = self._ensure_s_prime(K)

        cons: List[cvx.Constraint] = [
            # Boundary conditions
            X_v[0, 0] == self.x_init[0],
            X_v[1:4, 0] == self.x_init[1:4],
            X_v[4:7, 0] == self.x_init[4:7],
            X_v[7:11, 0] == self.x_init[7:11],
            X_v[11:14, 0] == self.x_init[11:14],
            # Final (mass free)
            X_v[1:, -1] == self.x_final[1:],
        ]

        # State constraints
        cons += [
            X_v[0, :] >= self.m_dry,  # min mass
            cvx.norm(X_v[1:3, :], axis=0) <= X_v[3, :] / self.tan_gamma_gs,  # glideslope
            cvx.norm(X_v[8:10, :], axis=0) <= np.sqrt((1.0 - self.cos_theta_max) / 2.0),  # tilt
            cvx.norm(X_v[11:14, :], axis=0) <= self.w_B_max,  # angular rate
        ]

        # Control constraints
        cons += [
            cvx.norm(U_v[0:2, :], axis=0) <= self.tan_delta_max * U_v[2, :],  # gimbal cone
            cvx.norm(U_v, axis=0) <= self.T_max,  # upper thrust bound
        ]

        # Linearized lower-thrust bound (lossless convexification, scalar projection)
        eps = 1e-8
        proj = [
            cvx.sum(cvx.multiply(U_last_p[:, k], U_v[:, k])) / (cvx.norm(U_last_p[:, k]) + eps)
            for k in range(K)
        ]
        cons += [self.T_min - cvx.vstack(proj) <= s_prime]

        return cons

    def get_objective(self, X_v, U_v, X_last_p, U_last_p):  # noqa: ARG002
        # Penalize slack for (linearized) lower-thrust convexification
        return cvx.Minimize(1e5 * cvx.sum(self._s_prime))

    def get_linear_cost(self) -> float:
        return (
            float(np.sum(self._s_prime.value))
            if (self._s_prime is not None and self._s_prime.value is not None)
            else 0.0
        )

    def get_nonlinear_cost(self, X: Optional[np.ndarray] = None, U: Optional[np.ndarray] = None) -> float:  # noqa: ARG002
        if U is None:
            return 0.0
        mag = np.linalg.norm(U, axis=0)
        violation = np.maximum(0.0, self.T_min - mag)
        return float(np.sum(violation))
