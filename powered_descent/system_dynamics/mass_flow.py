# system_dynamics/mass_flow.py
from __future__ import annotations

import numpy as np

from .params import VehicleEnvParams


def mdot(T_B: np.ndarray, params: VehicleEnvParams) -> float:
    """
    dot m = -alpha_{mdot} ||T_B||_2 - beta_{mdot}

    Parameters
    ----------
    T_B : array-like, shape (..., 3)
        Thrust vector in body frame.
    params : VehicleEnvParams
        Parameters carrying alpha_{mdot}=1/(Isp*g0), beta_{mdot}=alpha_{mdot} P_amb A_noz.

    Returns
    -------
    scalar mass rate [UM/UT] in ND units (negative during burn).
    """
    Tmag = float(np.linalg.norm(T_B))
    return -(params.alpha_mdot * Tmag + params.beta_mdot)
