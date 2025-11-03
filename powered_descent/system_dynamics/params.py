# This file contains the parameters for the system dynamics.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class VehicleParams:
    # Inertia (kg m^2)
    J: Array  # shape (3, 3)

    # Location of engine force application in body frame (m)
    r_F_B: Array  # shape (3,)  # noqa: N815

    # Mas flow proportionality (s/m)
    alpha_mdot: float

    # Thrust magnitude bounds (N)
    F_min: float
    F_max: float

    # RCS torque bound (Nm) - infinity norm
    T_max: float

    # Max gimbal half angle (rad)
    delta_max: float


@dataclass(frozen=True)
class EnvParams:
    # Interial gravity vector (m/s^2)
    g_I: Array  # shape (3,)  # noqa: N815


@dataclass(frozen=True)
class CaseParams:
    # State hard bounds (absolute box)
    x_lb: Array  # shape (13,)
    x_ub: Array  # shape (13,)

    # Deviation  box around nominal
    dx_lb: Array  # shape (13,)
    dx_ub: Array  # shape (13,)

    # Input hard bounds (component-wise box)
    u_lb: Array  # shape (6,)
    u_ub: Array  # shape (6,)

    # Terminal ellipsoid (Q_f^{1/2}), s.t. (x-x_nom)^T Q_f (x-x_nom) <= 1
    Qf_max_sqrt: Array  # shape (13, 13)


def make_default_params() -> tuple[VehicleParams, EnvParams, CaseParams]:
    J = np.diag([13600.0, 13600.0, 19150.0])
    r_F_B = np.array([0.0, 0.0, -0.25])
    alpha_mdot = 4.5324e-4
    F_min, F_max = 5400.0, 24750.0
    T_max = 150.0
    delta_max = np.deg2rad(25.0)

    g_I = np.array([0.0, 0.0, -1.62])

    x_lb = -np.array([-2100, 150, 150, 0, 40, 40, 30, np.pi, np.pi / 2, np.pi, 0.5, 0.5, 0.5], dtype=float)
    x_ub = np.array([3737.7, 350, 300, 500, 30, 30, 5, np.pi, np.pi / 2, np.pi, 0.5, 0.5, 0.5], dtype=float)

    dx_angles = (2.0 / 9.0) * np.pi
    dx_lb = -np.array(
        [
            np.inf,
            100,
            100,
            100,
            np.inf,
            np.inf,
            np.inf,
            dx_angles,
            dx_angles,
            dx_angles,
            dx_angles,
            dx_angles,
            dx_angles,
        ],
        dtype=float,
    )
    dx_ub = np.array(
        [
            np.inf,
            100,
            100,
            100,
            np.inf,
            np.inf,
            np.inf,
            dx_angles,
            dx_angles,
            dx_angles,
            dx_angles,
            dx_angles,
            dx_angles,
        ],
        dtype=float,
    )

    u_lb = -np.array([7695.5, 7695.5, -5400, 150, 150, 150], dtype=float)
    u_ub = np.array([7695.5, 7695.5, 24750, 150, 150, 150], dtype=float)

    Qf_vec = np.array(
        [
            1673.0,
            0.5,
            0.5,
            0.5,
            0.25,
            0.25,
            0.25,
            np.pi / 60,
            np.pi / 60,
            np.pi / 60,
            np.pi / 60,
            np.pi / 60,
            np.pi / 60,
        ],
        dtype=float,
    )
    Qf_max_sqrt = np.diag(Qf_vec)  # (13,13)

    veh = VehicleParams(
        J=J, r_F_B=r_F_B, alpha_mdot=alpha_mdot, F_min=F_min, F_max=F_max, T_max=T_max, delta_max=delta_max
    )
    env = EnvParams(g_I=g_I)
    case = CaseParams(x_lb=x_lb, x_ub=x_ub, dx_lb=dx_lb, dx_ub=dx_ub, u_lb=u_lb, u_ub=u_ub, Qf_max_sqrt=Qf_max_sqrt)

    return veh, env, case
