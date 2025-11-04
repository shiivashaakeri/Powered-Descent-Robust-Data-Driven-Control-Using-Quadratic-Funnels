# system_dynamics/boundary.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import e1
from .params import VehicleEnvParams
from .quaternions import q_normalize
from .typing import ConstraintResult, State


@dataclass
class FinalTargets:
    r_I: np.ndarray          # target position (3,)  # noqa: N815
    v_I: np.ndarray          # target velocity (3,)  # noqa: N815
    q_BI: np.ndarray         # target attitude quaternion (4,)  # noqa: N815
    omega_B: np.ndarray      # target body rates (3,)  # noqa: N815

def ignition_baseline(params: VehicleEnvParams,
                      r_I_ig: np.ndarray,
                      v_I_ig: np.ndarray) -> State:
    """Baseline ignition state with m=m_ig, ω=0, q identity by default."""
    # Use identity quaternion if q_id is not provided in params
    q_id = getattr(params, 'q_id', np.array([1.0, 0.0, 0.0, 0.0]))
    return State(
        m=float(params.m_ig),
        r_I=np.array(r_I_ig, dtype=float),
        v_I=np.array(v_I_ig, dtype=float),
        q_BI=np.array(q_id, dtype=float),   # identity or provided in params
        omega_B=np.zeros(3, dtype=float),
    )

def final_targets_baseline(params: VehicleEnvParams, v_d: float) -> FinalTargets:
    """Final landing targets: r=0, v=-v_d e1, q=q_id, ω=0."""
    # Use identity quaternion if q_id is not provided in params
    q_id = getattr(params, 'q_id', np.array([1.0, 0.0, 0.0, 0.0]))
    return FinalTargets(
        r_I=np.zeros(3, dtype=float),
        v_I=-float(v_d) * e1,
        q_BI=np.array(q_id, dtype=float),
        omega_B=np.zeros(3, dtype=float),
    )

def poly_ignition_r_v(t_c: float,
                      r_I_in: np.ndarray,
                      v_I_in: np.ndarray,
                      g_I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Free-ignition polynomials (engine-off propagation):
      p_r,ig(t_c) = r_in + v_in t_c + 0.5 g_I t_c^2
      p_v,ig(t_c) = v_in + g_I t_c
    """
    r_I_in = np.asarray(r_I_in, dtype=float)
    v_I_in = np.asarray(v_I_in, dtype=float)
    g_I = np.asarray(g_I, dtype=float)

    r = r_I_in + v_I_in * t_c + 0.5 * g_I * (t_c**2)
    v = v_I_in + g_I * t_c
    return r, v

def check_final(x: State, tgt: FinalTargets) -> list[ConstraintResult]:
    dr = x.r_I - tgt.r_I
    dv = x.v_I - tgt.v_I
    dq = float(1.0 - abs(np.dot(q_normalize(x.q_BI), q_normalize(tgt.q_BI))))  # 0 if identical up to sign
    dw = x.omega_B - tgt.omega_B
    return [
        ConstraintResult("final_r", np.linalg.norm(dr) <= 1e-9, -np.linalg.norm(dr), {"||dr||": np.linalg.norm(dr)}),
        ConstraintResult("final_v", np.linalg.norm(dv) <= 1e-9, -np.linalg.norm(dv), {"||dv||": np.linalg.norm(dv)}),
        ConstraintResult("final_q", dq <= 1e-9, -dq, {"1-|<q,q*>|": dq}),
        ConstraintResult("final_omega", np.linalg.norm(dw) <= 1e-9, -np.linalg.norm(dw), {"||dω||": np.linalg.norm(dw)})
    ]
