# system_dynamics/typing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np

from powered_descent.system_dynamics.params import VehicleEnvParams

ArrayLike = Union[np.ndarray, Sequence[float], float]


@dataclass
class State:
    m: float  # mass
    r_I: np.ndarray  # (3,)  # noqa: N815
    v_I: np.ndarray  # (3,)  # noqa: N815
    q_BI: np.ndarray  # (4,) scalar-first quaternion q_{B<-I}  # noqa: N815
    omega_B: np.ndarray  # (3,)  # noqa: N815


@dataclass
class Control:
    T_B: np.ndarray  # (3,) thrust vector in body frame


@dataclass
class Deriv:
    mdot: float
    rdot_I: np.ndarray  # (3,)  # noqa: N815
    vdot_I: np.ndarray  # (3,)  # noqa: N815
    qdot: np.ndarray  # (4,)
    omegadot_B: np.ndarray  # (3,)  # noqa: N815


@dataclass
class IntegrateOptions:
    method: str = "rk4"  # 'rk4' (default)
    renorm_quat_substages: bool = True
    clip_mass_to_dry: bool = True  # clip after the step, not inside f


@dataclass
class ConstraintResult:
    name: str
    satisfied: bool
    residual: float  # >= 0 means satisfied for our conventions
    extras: Dict[str, Any] = field(default_factory=dict)


# ---------- helpers to safely copy / combine ----------
def copy_state(x: State) -> State:
    return State(
        m=float(x.m),
        r_I=np.array(x.r_I, dtype=float),
        v_I=np.array(x.v_I, dtype=float),
        q_BI=np.array(x.q_BI, dtype=float),
        omega_B=np.array(x.omega_B, dtype=float),
    )


def add_state(x: State, dx: Deriv, alpha: float) -> State:
    return State(
        m=x.m + alpha * dx.mdot,
        r_I=x.r_I + alpha * dx.rdot_I,
        v_I=x.v_I + alpha * dx.vdot_I,
        q_BI=x.q_BI + alpha * dx.qdot,
        omega_B=x.omega_B + alpha * dx.omegadot_B,
    )


def state_to_vec(x: State) -> np.ndarray:
    return np.concatenate(
        (
            np.array([x.m], dtype=float),
            x.r_I.reshape(3),
            x.v_I.reshape(3),
            x.q_BI.reshape(4),
            x.omega_B.reshape(3),
        )
    )


def deriv_to_vec(dx: Deriv) -> np.ndarray:
    return np.concatenate(
        (
            np.array([dx.mdot], dtype=float),
            dx.rdot_I.reshape(3),
            dx.vdot_I.reshape(3),
            dx.qdot.reshape(4),
            dx.omegadot_B.reshape(3),
        )
    )


def vec_to_state(v: np.ndarray) -> State:
    v = np.asarray(v, dtype=float).reshape(-1)
    assert v.size == 13, "Expected 13 elements: [m, r(3), v(3), q(4), omega(3)]"
    m = v[0]
    r_I = v[1:4]
    v_I = v[4:7]
    q_BI = v[7:11]
    omega_B = v[11:14]
    return State(m=m, r_I=r_I, v_I=v_I, q_BI=q_BI, omega_B=omega_B)


# type aliases for user callbacks
UCallback = Callable[[float, State], Control]
FContinuous = Callable[["State", "Control", "VehicleEnvParams"], "Deriv"]
