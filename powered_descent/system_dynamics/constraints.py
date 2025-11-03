from __future__ import annotations

import numpy as np

from .params import CaseParams, VehicleParams

Array = np.ndarray


def _thrust_gimbal_ok(F_B: Array, veh: VehicleParams) -> bool:
    """
    Gimbal cone: ||F_B|| <= sec(delta_max) * z_B^T F_B
    where z_B = [0, 0, 1] in body frame.
    """
    nF = np.linalg.norm(F_B)
    rhs = (1.0 / np.cos(veh.delta_max)) * F_B[2]
    return nF <= rhs + 1e-9


def _thrust_mag_ok(F_B: Array, veh: VehicleParams) -> bool:
    nF = np.linalg.norm(F_B)
    return (veh.F_min - 1e-9) <= nF <= (veh.F_max + 1e-9)


def _torque_inf_ok(tau_B: Array, veh: VehicleParams) -> bool:
    return np.max(np.abs(tau_B)) <= veh.T_max + 1e-9


def U_constraints_ok(u: Array, veh: VehicleParams) -> bool:
    F_B = u[:3]
    tau_B = u[3:]
    return _thrust_mag_ok(F_B, veh) and _torque_inf_ok(tau_B, veh) and _thrust_gimbal_ok(F_B, veh)


def X_box_ok(x: Array, case: CaseParams) -> bool:
    return np.all(x >= case.x_lb - 1e-12) and np.all(x <= case.x_ub + 1e-12)


def delta_X_box_ok(x: Array, x_nom: Array, case: CaseParams) -> bool:
    dx = x - x_nom
    return np.all(dx >= case.dx_lb - 1e-12) and np.all(dx <= case.dx_ub + 1e-12)


def Xf_ellipsoid_ok(xT: Array, xT_nom: Array, case: CaseParams) -> bool:
    S = case.Qf_max_sqrt
    d = S @ (xT - xT_nom)
    return float(d.T @ d) <= 1.0 + 1e-9


def bound_project(v: Array, lb: Array, ub: Array) -> Array:
    return np.minimum(np.maximum(v, lb), ub)
