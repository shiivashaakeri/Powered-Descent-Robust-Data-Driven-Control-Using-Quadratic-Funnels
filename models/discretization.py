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
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


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
