# nominal/dynamics/integrators/foh_fixed.py
from __future__ import annotations

from typing import Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
from scipy.integrate import odeint  # pyright: ignore[reportMissingImports]


class FirstOrderHold:
    """
    First-Order-Hold discretization for FIXED final time (sigma is fixed).
    Produces (Ā, B̄, C̄, z̄) for each interval and supports piecewise nonlinear integration.

    Shapes
    ------
    X: (n_x, K)
    U: (n_u, K)
    A_bar: (n_x*n_x, K-1)   stored column-flattened (Fortran order) per interval
    B_bar: (n_x*n_u, K-1)
    C_bar: (n_x*n_u, K-1)
    z_bar: (n_x, K-1)
    """

    def __init__(self, model, K: int, sigma: float):
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
        self.z_bar = np.zeros((self.n_x, self.K - 1))

        # Slices in the augmented ODE state V
        x_end = self.n_x
        A_end = x_end + self.n_x * self.n_x
        B_end = A_end + self.n_x * self.n_u
        C_end = B_end + self.n_x * self.n_u
        z_end = C_end + self.n_x
        self._x_sl = slice(0, x_end)
        self._A_sl = slice(x_end, A_end)
        self._B_sl = slice(A_end, B_end)
        self._C_sl = slice(B_end, C_end)
        self._z_sl = slice(C_end, z_end)

        # Initial condition in augmented space
        self.V0 = np.zeros((z_end,))
        self.V0[self._A_sl] = np.eye(self.n_x).reshape(-1, order="F")

        # Timing
        self.sigma = float(sigma)
        self.dt = self.sigma / (self.K - 1)

    # --------- Public API --------- #
    def calculate_discretization(
        self, X: np.ndarray, U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Ā,B̄,C̄,z̄ for each interval k = 0..K-2 using FOH on u.
        """
        for k in range(self.K - 1):
            self.V0[self._x_sl] = X[:, k]
            V1 = odeint(self._ode_dVdt, self.V0, (0.0, self.dt), args=(U[:, k], U[:, k + 1]))[1, :]

            # State transition from τ_k to τ_{k+1}
            Phi = V1[self._A_sl].reshape((self.n_x, self.n_x), order="F")

            self.A_bar[:, k] = Phi.reshape(-1, order="F")
            self.B_bar[:, k] = (Phi @ V1[self._B_sl].reshape((self.n_x, self.n_u), order="F")).reshape(-1, order="F")
            self.C_bar[:, k] = (Phi @ V1[self._C_sl].reshape((self.n_x, self.n_u), order="F")).reshape(-1, order="F")
            self.z_bar[:, k] = Phi @ V1[self._z_sl]
        return self.A_bar, self.B_bar, self.C_bar, self.z_bar

    def integrate_nonlinear_piecewise(self, X_l: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Integrate the full nonlinear dynamics piecewise over each interval,
        starting from X_l[:,k] and using FOH on U.
        """
        X_nl = np.zeros_like(X_l)
        X_nl[:, 0] = X_l[:, 0]
        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dx, X_l[:, k], (0.0, self.dt), args=(U[:, k], U[:, k + 1]))[1, :]
        return X_nl

    def integrate_nonlinear_full(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Integrate the full nonlinear dynamics from initial state x0 across the entire horizon.
        """
        X_nl = np.zeros((self.n_x, self.K))
        X_nl[:, 0] = x0
        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(self._dx, X_nl[:, k], (0.0, self.dt), args=(U[:, k], U[:, k + 1]))[1, :]
        return X_nl

    # --------- ODEs in augmented space --------- #
    def _ode_dVdt(self, V: np.ndarray, t: float, u0: np.ndarray, u1: np.ndarray) -> np.ndarray:
        """
        Augmented ODE for computing Ā,B̄,C̄,z̄ on [0, dt].
          V = [x, vec(Phi_A), vec(B_bar), vec(C_bar), z_bar]
        """
        alpha = (self.dt - t) / self.dt
        # beta = t / self.dt  -> not needed explicitly; we use (1 - alpha)
        x = V[self._x_sl]
        u = u0 + (t / self.dt) * (u1 - u0)

        # Pre-compute
        Phi_A = V[self._A_sl].reshape((self.n_x, self.n_x), order="F")
        Phi_A_xi = np.linalg.inv(Phi_A)

        A = self.A(x, u)
        B = self.B(x, u)
        f = self.f(x, u).reshape(-1)

        dV = np.zeros_like(V)
        dV[self._x_sl] = f
        dV[self._A_sl] = (A @ Phi_A).reshape(-1, order="F")
        dV[self._B_sl] = (Phi_A_xi @ B).reshape(-1, order="F") * alpha
        dV[self._C_sl] = (Phi_A_xi @ B).reshape(-1, order="F") * (1.0 - alpha)
        z_t = f - A @ x - B @ u
        dV[self._z_sl] = Phi_A_xi @ z_t
        return dV

    def _dx(self, x: np.ndarray, t: float, u0: np.ndarray, u1: np.ndarray) -> np.ndarray:
        u = u0 + (t / self.dt) * (u1 - u0)
        return self.f(x, u).reshape(-1)
