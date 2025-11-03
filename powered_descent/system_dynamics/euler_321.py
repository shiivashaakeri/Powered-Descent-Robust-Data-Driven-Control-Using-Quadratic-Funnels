from __future__ import annotations

import numpy as np

Array = np.ndarray


def Rx(phi: float) -> Array:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(psi: float) -> Array:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def C_IB(theta_vec: Array) -> Array:
    """
    Rotation from body to inertial frame for 3-2-1 Euler angles.
    Theta = (phi, theta, psi), with roll phi about x, pitch theta about y, and yaw psi about z.
    C_IB = Rz(psi) @ Ry(theta) @ Rx(phi)
    """
    phi, theta, psi = theta_vec
    return Rz(psi) @ Ry(theta) @ Rx(phi)


def T_Theta(theta_vec: Array, eps: float = 1e-8) -> Array:
    """
    Mapping omega_B -> Theta_dot for 3-2-1 Euler angles:
    [phi_dot, theta_dot, psi_dot]^T = T(Theta) @ [p, q, r]^T

    T(Theta) = [
        1   sin(phi)tan(theta)   cos(phi)tan(theta)
        0   cos(phi)            -sin(phi)
        0   sin(phi) sec(theta) cos(phi) sec(theta)
    ]
    gaurd near cos(theta) = 0 by eps
    """

    phi, theta, _ = theta_vec
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cth = np.sign(cth) * max(abs(cth), eps)
    tan_th = sth / cth
    sec_th = 1.0 / cth

    return np.array(
        [
            [1.0, sphi * tan_th, cphi * tan_th],
            [0.0, cphi, -sphi],
            [0.0, sphi * sec_th, cphi * sec_th],
        ]
    )


def skew(v: Array) -> Array:
    x, y, z = v
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


# shorthands
hat = skew


def vee(S: Array) -> Array:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


# Aliases for 3-2-1 Euler angle transformation
T_321 = T_Theta


def T_321_inv(theta_vec: Array, eps: float = 1e-8) -> Array:
    """
    Inverse mapping Theta_dot -> omega_B for 3-2-1 Euler angles:
    [p, q, r]^T = T_inv(Theta) @ [phi_dot, theta_dot, psi_dot]^T
    """
    T = T_Theta(theta_vec, eps)
    return np.linalg.inv(T)
