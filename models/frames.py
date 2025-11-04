# nominal/models/frames.py
from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]
import sympy as sp  # pyright: ignore[reportMissingImports]


# ---------------------- SymPy helpers (symbolic) ---------------------- #
def skew_sym(v: sp.Matrix) -> sp.Matrix:
    """Skew-symmetric matrix for cross product (symbolic). v shape: (3,1) or (3,)"""
    return sp.Matrix(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def omega_sym(w: sp.Matrix) -> sp.Matrix:
    """Quaternion kinematics matrix (symbolic), qdot = 0.5*Omega(w)*q ."""
    return sp.Matrix(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ]
    )


def dcm_from_quat_sym(q: sp.Matrix) -> sp.Matrix:
    """
    Direction cosine matrix from unit quaternion q = [q0,q1,q2,q3] (w,x,y,z).
    Returns C_BI: body->inertial.
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return sp.Matrix(
        [
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        ]
    )


# ---------------------- Numeric helpers (numpy) ---------------------- #
def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion [w,x,y,z]."""
    n = np.linalg.norm(q)
    return q if n == 0 else q / n


def euler_to_quat(euler_deg: tuple[float, float, float] | list[float]) -> np.ndarray:
    """
    Euler (roll, pitch, yaw) in DEGREES → quaternion [w,x,y,z].
    Convention: intrinsic rotations about x (roll), y (pitch), z (yaw), applied in that order.
    """
    roll, pitch, yaw = np.deg2rad(euler_deg)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)

    q = np.zeros(4)
    q[0] = cy * cr * cp + sy * sr * sp  # w
    q[1] = cy * sr * cp - sy * cr * sp  # x
    q[2] = sy * cr * cp - cy * sr * sp  # y
    q[3] = cy * cr * sp + sy * sr * cp  # z
    return normalize_quat(q)
