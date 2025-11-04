# system_dynamics/constraints.py
from __future__ import annotations

import numpy as np

from .frames import H_gamma, H_theta, e1
from .params import VehicleEnvParams
from .typing import ConstraintResult, Control, State


def mass_lb(x: State, params: VehicleEnvParams) -> ConstraintResult:
    residual = x.m - params.m_dry                 # want >= 0
    return ConstraintResult("mass_lower_bound", residual >= 0.0, residual)

def glide_slope(x: State, params: VehicleEnvParams) -> ConstraintResult:
    lhs = float(e1 @ x.r_I)
    rhs = float(np.tan(params.gamma_gs_rad) * np.linalg.norm(H_gamma @ x.r_I))
    residual = lhs - rhs                           # want >= 0
    return ConstraintResult("glide_slope", residual >= 0.0, residual, {"lhs": lhs, "rhs": rhs})

def tilt(x: State, params: VehicleEnvParams) -> ConstraintResult:
    # cos(theta_max) <= 1 - 2||H_theta q||_2   => residual := (1 - 2||Hθ q||) - cos θ_max
    cos_th_max = float(np.cos(params.theta_max_rad))
    term = 1.0 - 2.0 * np.linalg.norm(H_theta @ x.q_BI)
    residual = term - cos_th_max                  # want >= 0
    return ConstraintResult("tilt", residual >= 0.0, residual, {"term": term, "cos_theta_max": cos_th_max})

def omega_ub(x: State, params: VehicleEnvParams) -> ConstraintResult:
    normw = float(np.linalg.norm(x.omega_B))
    residual = float(params.omega_max_rad_per_UT) - normw    # want >= 0 (||ω|| ≤ ω_max)
    return ConstraintResult("omega_rate_bound", residual >= 0.0, residual, {"||omega||": normw})

def thrust_bounds(u: Control, params: VehicleEnvParams) -> tuple[ConstraintResult, ConstraintResult]:
    Tmag = float(np.linalg.norm(u.T_B))
    low = Tmag - params.T_min                     # want >= 0
    high = params.T_max - Tmag                    # want >= 0
    return (
        ConstraintResult("thrust_min", low >= 0.0, low, {"||T||": Tmag, "T_min": params.T_min}),
        ConstraintResult("thrust_max", high >= 0.0, high, {"||T||": Tmag, "T_max": params.T_max}),
    )

def gimbal(u: Control, params: VehicleEnvParams) -> ConstraintResult:
    T = u.T_B
    Tmag = float(np.linalg.norm(T))
    if Tmag <= 1e-12:
        # Degenerate: treat as violated relative to min thrust constraint elsewhere
        return ConstraintResult("gimbal", False, -np.inf, {"||T||": Tmag})

    cos_delta = float(np.cos(params.delta_max_rad))
    lhs = float((T @ e1))
    rhs = cos_delta * Tmag
    residual = lhs - rhs                           # want >= 0  (e1^T T ≥ cos δ_max ||T||)
    return ConstraintResult("gimbal", residual >= 0.0, residual, {"lhs": lhs, "rhs": rhs})

def all_state_constraints(x: State, params: VehicleEnvParams) -> list[ConstraintResult]:
    return [mass_lb(x, params), glide_slope(x, params), tilt(x, params), omega_ub(x, params)]

def all_control_constraints(u: Control, params: VehicleEnvParams) -> list[ConstraintResult]:
    lo, hi = thrust_bounds(u, params)
    return [lo, hi, gimbal(u, params)]
