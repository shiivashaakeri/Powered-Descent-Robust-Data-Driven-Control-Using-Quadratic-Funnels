# models/discretization.py
from __future__ import annotations

from typing import Callable, Optional

import numpy as np  # pyright: ignore[reportMissingImports]


def rk4_step(f: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Runge-Kutta 4th order step for a single integration step.

    Args:
        f: The function to integrate.
        x: The current state.
        u: The current control input.
        dt: The time step.
    """
    # Ensure x and u are column vectors for sympy lambdified functions
    x = np.asarray(x).reshape(-1, 1) if x.ndim == 1 else np.asarray(x)
    u = np.asarray(u).reshape(-1, 1) if u.ndim == 1 else np.asarray(u)

    k1 = np.asarray(f(x, u)).flatten()
    k2 = np.asarray(f(x + 0.5 * dt * k1.reshape(-1, 1), u)).flatten()
    k3 = np.asarray(f(x + 0.5 * dt * k2.reshape(-1, 1), u)).flatten()
    k4 = np.asarray(f(x + dt * k3.reshape(-1, 1), u)).flatten()
    return x.flatten() + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_one_step(f, x: np.ndarray, u: np.ndarray, dt: float, quat_slice: Optional[slice] = None) -> np.ndarray:
    """
    Integrate one step of the system.

    Args:
        f: The function to integrate.
        x: The current state.
        u: The current control input.
        dt: The time step.
        quat_slice: The slice of the state vector that contains the quaternion.
    """
    x1 = rk4_step(f, x, u, dt)
    if quat_slice is not None:
        q = x1[quat_slice]
        n = float(np.linalg.norm(q))
        if n > 0.0:
            x1[quat_slice] = q / n
    return x1
