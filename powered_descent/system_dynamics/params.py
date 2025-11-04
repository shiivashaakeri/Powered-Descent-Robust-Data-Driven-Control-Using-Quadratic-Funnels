# system_dynamics/params.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------
# helpers / conventions
# ---------------------------------------
def deg2rad(d: float) -> float:
    return np.pi * d / 180.0


# canonical basis in R^3 (I-frame axes)
e1 = np.array([1.0, 0.0, 0.0], dtype=float)
e2 = np.array([0.0, 1.0, 0.0], dtype=float)
e3 = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass(frozen=True)
class VehicleEnvParams:
    # --- propulsion / mass flow ---
    Isp: float  # [UT], specific impulse (non-dimensionalized)
    g0: float  # [UL/UT^2], standard gravity (set by ND scheme)
    P_amb: float  # [UM/(UT^2*UL)], ambient pressure
    A_noz: float  # [UL^2], nozzle exit area

    # --- thrust limits & gimbal ---
    T_min: float  # [UM*UL/UT^2]
    T_max: float  # [UM*UL/UT^2]
    delta_max_rad: float  # [rad], max gimbal angle (relative to body x-axis)

    # --- aero / atmosphere ---
    rho: float  # [UM/UL^3]
    S_A: float  # [UL^2], reference area (choose 1.0 if ND)
    # aerodynamic coefficient matrix C_A = diag(ca_x, ca_yz, ca_yz)
    ca_x: float
    ca_yz: float

    # --- rigid body / geometry ---
    J_B: np.ndarray  # [UM*UL^2] 3x3 inertia (about CoM, body frame)
    r_T_B: np.ndarray  # [UL] 3x, engine gimbal pivot (lever arm)  # noqa: N815
    r_cp_B: np.ndarray  # [UL] 3x, aerodynamic center of pressure  # noqa: N815

    # --- environment constants ---
    g_I: np.ndarray  # [UL/UT^2] 3x, constant gravity (inertial)  # noqa: N815

    # --- state limits ---
    m_dry: float  # [UM]
    m_ig: float  # [UM], ignition mass
    theta_max_rad: float  # [rad], tilt limit
    omega_max_rad_per_UT: float  # [rad/UT], body-rate limit  # noqa: N815
    gamma_gs_rad: float  # [rad], glide-slope half-angle

    # --- STC-related knobs (angle-of-attack / speed gate, etc.) ---
    V_alpha: float  # [UL/UT], speed scale for triggers
    alpha_max_rad: float  # [rad], max angle (e.g., AoA cap used by cSTC)

    # --- model selector ---
    aero_model: str = "ellipsoidal"  # "ellipsoidal" | "spherical"

    # -------- derived, convenience --------
    @property
    def alpha_mdot(self) -> float:
        """alpha_{ṁ} = 1 / (Isp * g0)"""
        return 1.0 / (self.Isp * self.g0)

    @property
    def beta_mdot(self) -> float:
        """beta_{ṁ} = alpha_{ṁ} * P_amb * A_noz"""
        return self.alpha_mdot * self.P_amb * self.A_noz

    @property
    def C_A(self) -> np.ndarray:
        """Aerodynamic coefficient SPD matrix in body frame."""
        return np.diag([self.ca_x, self.ca_yz, self.ca_yz])


def default_params() -> VehicleEnvParams:
    """
    Non-dimensional baseline matching the provided spec.
    Notes:
      - g0 is set to 1.0 [UL/UT^2] under the ND scheme (adjust if your ND uses a different g0).
      - S_A and (ca_x, ca_yz) are not specified in the snippet; using benign defaults (1.0).
        Override as needed when validating aero.
    """
    # inertia: 0.168 * diag([2e-2, 1, 1]) UM*UL^2
    J_B = 0.168 * np.diag([2e-2, 1.0, 1.0])

    return VehicleEnvParams(
        # propulsion / mass flow
        Isp=30.0,  # UT
        g0=1.0,  # UL/UT^2 (ND convention)
        P_amb=0.0,  # UM/(UT^2*UL)
        A_noz=0.0,  # UL^2
        # thrust & gimbal
        T_min=1.5,  # UM*UL/UT^2
        T_max=6.5,  # UM*UL/UT^2
        delta_max_rad=deg2rad(20.0),
        # aero / atmosphere
        rho=1.0,  # UM/UL^3
        S_A=1.0,  # UL^2 (choose per ND; 1.0 is a common placeholder)
        ca_x=1.0,  # set >0; adjust if you want drag-min along body x
        ca_yz=1.0,  # set >= ca_x; >ca_x enables lift-like component
        # rigid body / geometry
        J_B=J_B,
        r_T_B=(-0.25) * e1,
        r_cp_B=(+0.05) * e1,
        # environment
        g_I=(-1.0) * e1,  # UL/UT^2 along -x_I
        # state limits
        m_dry=1.0,  # UM
        m_ig=2.0,  # UM
        theta_max_rad=deg2rad(90.0),
        omega_max_rad_per_UT=deg2rad(28.6),
        gamma_gs_rad=deg2rad(75.0),
        # STC knobs
        V_alpha=2.0,  # UL/UT
        alpha_max_rad=deg2rad(3.0),
        # model selector
        aero_model="ellipsoidal",
    )
