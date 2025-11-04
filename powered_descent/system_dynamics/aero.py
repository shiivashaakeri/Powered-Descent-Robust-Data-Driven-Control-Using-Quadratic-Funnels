# system_dynamics/aero.py
from __future__ import annotations

import numpy as np

from .frames import unit
from .params import VehicleEnvParams
from .quaternions import rotate_B_to_I, rotate_I_to_B


def aerodynamic_force_B(v_I: np.ndarray, q_BI: np.ndarray, params: VehicleEnvParams) -> np.ndarray:
    """
    A_B = -0.5 * rho * ||v_I||^2 * S_A * C_A * v̂_B
    - Ellipsoidal model if ca_x != ca_yz (lift-like components allowed)
    - Spherical model if ca_x == ca_yz (A_B anti-parallel to v_B)
    """
    speed = float(np.linalg.norm(v_I))
    if speed <= 1e-12:
        return np.zeros(3, dtype=float)

    v_B = rotate_I_to_B(q_BI, v_I)
    vhat_B = unit(v_B)

    # Spherical if coefficients equal; otherwise ellipsoidal via C_A
    if abs(params.ca_x - params.ca_yz) < 1e-12:  # noqa: SIM108
        CAv = vhat_B  # acts like identity scaling (direction only)
    else:
        CAv = params.C_A @ vhat_B

    A_B = -0.5 * params.rho * (speed ** 2) * params.S_A * CAv
    return A_B

def aerodynamic_force_I(v_I: np.ndarray, q_BI: np.ndarray, params: VehicleEnvParams) -> np.ndarray:
    """A_I = C_{I<-B} A_B."""
    A_B = aerodynamic_force_B(v_I, q_BI, params)
    return rotate_B_to_I(q_BI, A_B)
