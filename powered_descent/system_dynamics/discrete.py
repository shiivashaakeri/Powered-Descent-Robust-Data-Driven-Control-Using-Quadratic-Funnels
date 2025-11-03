from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .continuous import f_continuous
from .params import EnvParams, VehicleParams

Array = np.ndarray


def rk4_step(x: Array, u: Array, dt: float, f: Callable[[Array, Array], Array]) -> Array:
    """
    Runge-Kutta 4th order step for discrete-time dynamics.
    """
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def sample(x: Array, u: Array, dt: float, veh: VehicleParams, env: EnvParams) -> Array:
    """
    One RK4 sample with bound checks optional upstream.
    """
    return rk4_step(x, u, dt, lambda xx, uu: f_continuous(xx, uu, veh, env))


def rollout(x0: Array, U: Iterable[Array], dt: float, veh: VehicleParams, env: EnvParams, record: bool = True):
    """
    Rollout a control sequence U = [u_0, u_1, ..., u_{N-1}] over time dt.
    Returns X (N+1, 13)
    """
    x = np.array(x0, dtype=float).copy()
    traj = [x.copy()] if record else None
    for u in U:
        x = sample(x, np.asarray(u, dtype=float), dt, veh, env)
        if record:
            traj.append(x.copy())
    return np.stack(traj, axis=0) if record else x
