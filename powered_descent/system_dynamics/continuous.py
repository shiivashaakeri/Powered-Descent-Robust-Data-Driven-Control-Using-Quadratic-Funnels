# system_dynamics/continuous.py
from __future__ import annotations

import numpy as np

from .forces import net_force_I
from .mass_flow import mdot as mdot_fn
from .params import VehicleEnvParams
from .quaternions import omega_matrix
from .torques import net_torque_B
from .typing import Control, Deriv, State


def f(x: State, u: Control, params: VehicleEnvParams) -> Deriv:
    """
    Continuous-time dynamics dot x = f(x,u):
      - dot m = -alpha_{mdot} ||T_B|| - beta_{mdot}
      - dot r_I = v_I
      - dot v_I = (1/m) F_I + g_I, with F_I = C_{I<-B} T_B + A_I(v,q)
      - dot q   = 0.5 * Omega(omega_B) * q   (q is scalar-first)
      - J_B dot omega_B = M_B - omega x (J_B omega)

    Side-effect free; quaternion normalization deferred to integrator.

    Parameters
    ----------
    x : State
        Current state.
    u : Control
        Current control.
    params : VehicleEnvParams
        Parameters.
    """
    # mass rate
    m_dot = mdot_fn(u.T_B, params)

    # translation
    r_dot = x.v_I
    F_I = net_force_I(u.T_B, x.v_I, x.q_BI, params)
    v_dot = (F_I / x.m) + params.g_I

    # quaternion kinematics
    q_dot = 0.5 * (omega_matrix(x.omega_B) @ x.q_BI)

    # rigid-body rotational dynamics
    J = params.J_B
    Jw = J @ x.omega_B
    M_B = net_torque_B(u.T_B, x.v_I, x.q_BI, params)
    omega_dot = np.linalg.solve(J, M_B - np.cross(x.omega_B, Jw))

    return Deriv(
        mdot=m_dot,
        rdot_I=r_dot,
        vdot_I=v_dot,
        qdot=q_dot,
        omegadot_B=omega_dot,
    )
