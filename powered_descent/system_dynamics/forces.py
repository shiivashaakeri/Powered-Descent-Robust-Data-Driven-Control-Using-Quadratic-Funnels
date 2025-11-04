# system_dynamics/forces.py
from __future__ import annotations

import numpy as np

from .aero import aerodynamic_force_I
from .params import VehicleEnvParams
from .quaternions import rotate_B_to_I


def thrust_force_I(T_B: np.ndarray, q_BI: np.ndarray) -> np.ndarray:
    """F_T,I = C_{I<-B} T_B."""
    return rotate_B_to_I(q_BI, T_B)

def net_force_I(T_B: np.ndarray, v_I: np.ndarray, q_BI: np.ndarray, params: VehicleEnvParams) -> np.ndarray:
    """
    F_I = C_{I<-B} T_B + A_I
    (Paper separates gravity as an external constant g_I; include it later in dynamics.)
    """
    F_T_I = thrust_force_I(T_B, q_BI)
    A_I = aerodynamic_force_I(v_I, q_BI, params)
    return F_T_I + A_I
