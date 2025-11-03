from __future__ import annotations

import numpy as np

from .euler_321 import C_IB, T_Theta, skew
from .params import EnvParams, VehicleParams

Array = np.ndarray


# State layout helpers
def state_unpack(x: Array):
    """
    x: (13,) -> (m, r_I(3), v_I(3), Theta(3), omega_B(3))
    """
    m = x[0]
    r_I = x[1:4]
    v_I = x[4:7]
    Theta = x[7:10]
    omega_B = x[10:13]
    return m, r_I, v_I, Theta, omega_B


def state_pack(m, r_I, v_I, Theta, omega_B) -> Array:
    return np.concatenate([[m], r_I, v_I, Theta, omega_B])


def f_continuous(x: Array, u: Array, veh: VehicleParams, env: EnvParams) -> Array:
    """
    Implements (6-DoF) dynamics:
        mdot = -alpha_mdot * ||F_B||
        rdot_I = v_I
        vdot_I = (1/m) C_IB(Theta) F_B + g_I
        Theta_dot = T(Theta) @ omega_B
        omega_B_dot = J^{-1} (tau_B + r_F^x F_B - omega_B^x J omega_B)
    Inputs:
        x: (13,), u: (6,)
        u = [F_B(3), tau_B(3)]
    """
    m, r_I, v_I, Theta, omega_B = state_unpack(x)
    F_B = u[:3]
    tau_B = u[3:]

    # Mass rate
    mdot = -veh.alpha_mdot * np.linalg.norm(F_B)

    # Kinematics
    rdot_I = v_I

    # Translational dynamics
    C = C_IB(Theta)
    vdot_I = (C @ F_B) / max(m, 1e-9) + env.g_I

    # Attitude kinematics
    Theta_dot = T_Theta(Theta) @ omega_B

    # Rotational dynamics
    J = veh.J
    omega_hat = skew(omega_B)
    tau_eff = tau_B + skew(veh.r_F_B) @ F_B - omega_hat @ (J @ omega_B)
    omega_dot = np.linalg.solve(J, tau_eff)

    return state_pack(mdot, rdot_I, vdot_I, Theta_dot, omega_dot)


# Alias for backward compatibility with tests
f = f_continuous
