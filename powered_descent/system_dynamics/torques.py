# system_dynamics/torques.py
from __future__ import annotations

import numpy as np

from .aero import aerodynamic_force_B
from .frames import cross
from .params import VehicleEnvParams


def net_torque_B(T_B: np.ndarray, v_I: np.ndarray, q_BI: np.ndarray, params: VehicleEnvParams) -> np.ndarray:
    """
    M_B = r_{T,B} x T_B + r_{cp,B} x A_B
    (Uses aerodynamic force computed in body frame.)
    """
    A_B = aerodynamic_force_B(v_I, q_BI, params)
    M_T = cross(params.r_T_B, T_B)
    M_A = cross(params.r_cp_B, A_B)
    return M_T + M_A
