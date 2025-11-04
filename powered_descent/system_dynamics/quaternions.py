# system_dynamics/quaternions.py
from __future__ import annotations

import numpy as np

# Convention:
# - Scalar-first Hamilton quaternion q = [q0, q1, q2, q3]^T
# - q_{B<-I}: rotates a vector from I to B as v_B = C_{B<-I}(q) v_I
# - C_{I<-B}(q) = C_{B<-I}(q)^T

def q_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def q_mul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Quaternion product (Hamilton)."""
    q0, qv = q[0], q[1:]
    p0, pv = p[0], p[1:]
    s = q0 * p0 - np.dot(qv, pv)
    v = q0 * pv + p0 * qv + np.cross(qv, pv)
    return np.concatenate(([s], v))

def omega_matrix(omega_B: np.ndarray) -> np.ndarray:
    """
    Omega(ω) such that qdot = 0.5 * Omega(ω_B) * q, for q_{B<-I}.
    """
    wx, wy, wz = omega_B
    return np.array([
        [ 0.0, -wx, -wy, -wz],
        [ wx,  0.0,  wz, -wy],
        [ wy, -wz,  0.0,  wx],
        [ wz,  wy, -wx,  0.0]
    ], dtype=float)

def C_B_from_I(q_BI: np.ndarray) -> np.ndarray:
    """
    Rotation matrix C_{B<-I}(q): maps inertial vectors to body.
    """
    q = q_normalize(q_BI)
    q0, q1, q2, q3 = q
    # Standard scalar-first SO(3) from quaternion
    q00 = q0*q0
    q11 = q1*q1
    q22 = q2*q2
    q33 = q3*q3
    q01 = q0*q1
    q02 = q0*q2
    q03 = q0*q3
    q12 = q1*q2
    q13 = q1*q3
    q23 = q2*q3
    C = np.array([
        [q00 + q11 - q22 - q33,     2*(q12 - q03),         2*(q13 + q02)],
        [    2*(q12 + q03),     q00 - q11 + q22 - q33,     2*(q23 - q01)],
        [    2*(q13 - q02),         2*(q23 + q01),     q00 - q11 - q22 + q33]
    ], dtype=float)
    return C

def C_I_from_B(q_BI: np.ndarray) -> np.ndarray:
    """C_{I<-B} = C_{B<-I}^T."""
    C_BI = C_B_from_I(q_BI)
    return C_BI.T

def rotate_I_to_B(q_BI: np.ndarray, v_I: np.ndarray) -> np.ndarray:
    return C_B_from_I(q_BI) @ v_I

def rotate_B_to_I(q_BI: np.ndarray, v_B: np.ndarray) -> np.ndarray:
    return C_I_from_B(q_BI) @ v_B
