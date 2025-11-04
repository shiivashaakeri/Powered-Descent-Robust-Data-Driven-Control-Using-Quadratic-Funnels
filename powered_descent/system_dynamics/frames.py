# system_dynamics/frames.py
from __future__ import annotations

import numpy as np

# Canonical basis in R^3
e1 = np.array([1.0, 0.0, 0.0], dtype=float)
e2 = np.array([0.0, 1.0, 0.0], dtype=float)
e3 = np.array([0.0, 0.0, 1.0], dtype=float)

def skew(v: np.ndarray) -> np.ndarray:
    """Return [v]x so that [v]x w = v x w."""
    vx, vy, vz = v
    return np.array([[0.0, -vz,  vy],
                     [vz,  0.0, -vx],
                     [-vy, vx,  0.0]], dtype=float)

def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe cross with float output."""
    return np.cross(a, b)

# Constraint helper matrices (paper notation)
# H_gamma := [e2^T; e3^T] ∈ R^{2x3}
H_gamma = np.vstack((e2, e3))

# For scalar-first quaternion q = [q0, q1, q2, q3]^T,
# H_θ := [e3, e4] picks the last two components (q2, q3) ∈ R^{2x4}
H_theta = np.array([[0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]], dtype=float)

def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n
