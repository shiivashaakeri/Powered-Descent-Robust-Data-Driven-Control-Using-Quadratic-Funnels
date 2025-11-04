# system_dynamics/linearize.py
from __future__ import annotations

from typing import Tuple

import numpy as np

from .params import VehicleEnvParams
from .quaternions import q_mul, q_normalize
from .typing import Control, FContinuous, State, deriv_to_vec


def _apply_tangent_delta(x: State,
                         delta: np.ndarray) -> State:
    """
    Apply a 13-dim tangent increment:
      δξ = [δm, δr(3), δv(3), δθ(3), δω(3)]
    Quaternion is perturbed left-multiplicatively by δq ≈ [1, 0.5 δθ].
    """
    delta_m = float(delta[0])
    delta_r = delta[1:4]
    delta_v = delta[4:7]
    delta_theta = delta[7:10]
    delta_omega = delta[10:13]

    x_new = State(
        m=x.m + delta_m,
        r_I=x.r_I + delta_r,
        v_I=x.v_I + delta_v,
        q_BI=q_normalize(q_mul(np.concatenate(([1.0], 0.5 * delta_theta)), x.q_BI)),
        omega_B=x.omega_B + delta_omega,
    )
    return x_new

def _control_delta(u: Control, dU: np.ndarray) -> Control:
    return Control(T_B=u.T_B + dU)

def finite_difference_A_B(f: FContinuous,
                              x: State,
                              u: Control,
                              params: VehicleEnvParams,
                              eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finite-difference linearization:
          δξ = [δm, δr(3), δv(3), δθ(3), δω(3)]  (13-dim tangent)
          A ≈ ∂f/∂ξ  ∈ R^{14,13},   B ≈ ∂f/∂u ∈ R^{14,3}
        Rows ordered as derivative vector [ṁ, ṙ(3), v̇(3), q̇(4), ω̇(3)].
        """
        f0 = f(x, u, params)
        y0 = deriv_to_vec(f0)  # (14,)

        # --- A matrix ---
        A = np.zeros((14, 13), dtype=float)
        for i in range(13):
            delta_xi = np.zeros(13, dtype=float)
            delta_xi[i] = eps
            x_pert = _apply_tangent_delta(x, delta_xi)
            y = deriv_to_vec(f(x_pert, u, params))
            A[:, i] = (y - y0) / eps

        # --- B matrix (control is T_B in R^3) ---
        B = np.zeros((14, 3), dtype=float)
        for j in range(3):
            du = np.zeros(3, dtype=float)
            du[j] = eps
            u_pert = _control_delta(u, du)
            y = deriv_to_vec(f(x, u_pert, params))
            B[:, j] = (y - y0) / eps

        return A, B
