# nominal/dynamics/integrators/foh_free.py
from __future__ import annotations

from typing import Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from scipy.integrate import odeint  # pyright: ignore[reportMissingImports]


class FirstOrderHold:
    """
    First-Order-Hold discretization for FREE final time.
    Time is normalized to [0,1]; sigma scales the dynamics.
    Produces (Ā, B̄, C̄, S̄, z̄) and supports piecewise nonlinear integration.

    Shapes
    ------
    X: (n_x, K)
    U: (n_u, K)
    A_bar: (n_x*n_x, K-1)   stored column-flattened (Fortran order)
    B_bar: (n_x*n_u, K-1)
    C_bar: (n_x*n_u, K-1)
    S_bar: (n_x, K-1)
    z_bar: (n_x, K-1)
    """

    def __init__(self, model, K: int):
        self.m = model
        self.K = int(K)
        self.n_x = int(model.n_x)
        self.n_u = int(model.n_u)

        # Callables
        self.f, self.A, self.B = model.get_equations()

        # Storage
        self.A_bar = np.zeros((self.n_x * self.n_x, self.K - 1))
        self.B_bar = np.zeros((self.n_x * self.n_u, self.K - 1))
        self.C_bar = np.zeros((self.n_x * self.n_u, self.K - 1))
        self.S_bar = np.zeros((self.n_x, self.K - 1))
        self.z_bar = np.zeros((self.n_x, self.K - 1))

        # Slices in augmented state
        x_end = self.n_x
        A_end = x_end + self.n_x * self.n_x
        B_end = A_end + self.n_x * self.n_u
        C_end = B_end + self.n_x * self.n_u
        S_end = C_end + self.n_x
        Z_end = S_end + self.n_x
        self._x_sl = slice(0, x_end)
        self._A_sl = slice(x_end, A_end)
        self._B_sl = slice(A_end, B_end)
        self._C_sl = slice(B_end, C_end)
        self._S_sl = slice(C_end, S_end)
        self._Z_sl = slice(S_end, Z_end)

        # Initial augmented state
        self.V0 = np.zeros((Z_end,))
        self.V0[self._A_sl] = np.eye(self.n_x).reshape(-1, order="F")

        # Normalized time step (over [0,1])
        self.dt = 1.0 / (self.K - 1)

    # --------- Public API --------- #
    def calculate_discretization(
        self, X: np.ndarray, U: np.ndarray, sigma: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Ā,B̄,C̄,S̄,z̄ for each interval with normalized time and dilation sigma.
        """
        for k in range(self.K - 1):
            self.V0[self._x_sl] = X[:, k]
            V1 = odeint(self._ode_dVdt, self.V0, (0.0, self.dt), args=(U[:, k], U[:, k + 1], sigma))[1, :]

            Phi = V1[self._A_sl].reshape((self.n_x, self.n_x), order="F")

            self.A_bar[:, k] = Phi.reshape(-1, order="F")
            self.B_bar[:, k] = (Phi @ V1[self._B_sl].reshape((self.n_x, self.n_u), order="F")).reshape(-1, order="F")
            self.C_bar[:, k] = (Phi @ V1[self._C_sl].reshape((self.n_x, self.n_u), order="F")).reshape(-1, order="F")
            self.S_bar[:, k] = Phi @ V1[self._S_sl]
            self.z_bar[:, k] = Phi @ V1[self._Z_sl]
        return self.A_bar, self.B_bar, self.C_bar, self.S_bar, self.z_bar

    def integrate_nonlinear_piecewise(self, X_l: np.ndarray, U: np.ndarray, sigma: float) -> np.ndarray:
        """
        Piecewise integration over [0, sigma] using FOH on U.
        """
        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]
        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dx, X_l[:, k], (0.0, self.dt * sigma), args=(U[:, k], U[:, k + 1], sigma))[
                1, :
            ]
        return X_nl

    def integrate_nonlinear_full(self, x0: np.ndarray, U: np.ndarray, sigma: float) -> np.ndarray:
        """
        Integrate the full nonlinear dynamics from x0 across the entire horizon with sigma scaling.
        """
        X_nl = np.zeros((self.n_x, self.K))
        X_nl[:, 0] = x0
        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k], (0.0, self.dt * sigma), args=(U[:, k], U[:, k + 1], sigma))[
                1, :
            ]
        return X_nl

    # --------- ODEs in augmented space --------- #
    def _ode_dVdt(self, V: np.ndarray, t: float, u0: np.ndarray, u1: np.ndarray, sigma: float) -> np.ndarray:
        """
        Augmented ODE on normalized time tau ∈ [0, dt], with dynamics scaled by sigma.
          V = [x, vec(Phi_A), vec(B_bar), vec(C_bar), S_bar, z_bar]
        """
        alpha = (self.dt - t) / self.dt
        x = V[self._x_sl]
        u = u0 + (t / self.dt) * (u1 - u0)

        # Transition for A(·) on normalized time; use Phi_A(τ_{k+1}, ξ)
        Phi_A = V[self._A_sl].reshape((self.n_x, self.n_x), order="F")
        Phi_A_xi = np.linalg.inv(Phi_A)

        A = sigma * self.A(x, u)
        B = sigma * self.B(x, u)
        f = self.f(x, u).reshape(-1)

        dV = np.zeros_like(V)
        dV[self._x_sl] = sigma * f
        dV[self._A_sl] = (A @ Phi_A).reshape(-1, order="F")
        dV[self._B_sl] = (Phi_A_xi @ B).reshape(-1, order="F") * alpha
        dV[self._C_sl] = (Phi_A_xi @ B).reshape(-1, order="F") * (1.0 - alpha)
        dV[self._S_sl] = Phi_A_xi @ f
        z_t = -(A @ x + B @ u)
        dV[self._Z_sl] = Phi_A_xi @ z_t
        return dV

    def _dx(self, x: np.ndarray, t: float, u0: np.ndarray, u1: np.ndarray, sigma: float) -> np.ndarray:
        u = u0 + (t / (self.dt * sigma)) * (u1 - u0)
        return self.f(x, u).reshape(-1)
