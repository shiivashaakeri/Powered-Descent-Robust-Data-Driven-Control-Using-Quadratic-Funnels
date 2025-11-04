# system_dynamics/discrete.py
from __future__ import annotations

from typing import Optional, Sequence, Union

from .params import VehicleEnvParams
from .quaternions import q_normalize
from .typing import Control, Deriv, FContinuous, IntegrateOptions, State, UCallback, add_state, copy_state


def _renorm_quat_inplace(x: State) -> None:
    x.q_BI[:] = q_normalize(x.q_BI)

def sample(f: FContinuous, x: State, u: Control, dt: float,
           params: VehicleEnvParams,
           opts: Optional[IntegrateOptions] = None) -> State:
    """
    One fixed-step integration with RK4. Quaternion is renormalized at each
    sub-stage (if opts.renorm_quat_substages). Mass is clipped to m_dry only
    AFTER the step (never inside f).
    """
    if opts is None:
        opts = IntegrateOptions()

    # stage 1
    k1 = f(x, u, params)
    x2 = add_state(x, k1, dt * 0.5)
    if opts.renorm_quat_substages:
        _renorm_quat_inplace(x2)

    # stage 2
    k2 = f(x2, u, params)
    x3 = add_state(x, k2, dt * 0.5)
    if opts.renorm_quat_substages:
        _renorm_quat_inplace(x3)

    # stage 3
    k3 = f(x3, u, params)
    x4 = add_state(x, k3, dt)
    if opts.renorm_quat_substages:
        _renorm_quat_inplace(x4)

    # stage 4
    k4 = f(x4, u, params)

    # combine
    x_next = copy_state(x)
    x_next = add_state(
        x_next,
        Deriv(
            mdot=(k1.mdot + 2*k2.mdot + 2*k3.mdot + k4.mdot) / 6.0,
            rdot_I=(k1.rdot_I + 2*k2.rdot_I + 2*k3.rdot_I + k4.rdot_I) / 6.0,
            vdot_I=(k1.vdot_I + 2*k2.vdot_I + 2*k3.vdot_I + k4.vdot_I) / 6.0,
            qdot=(k1.qdot + 2*k2.qdot + 2*k3.qdot + k4.qdot) / 6.0,
            omegadot_B=(k1.omegadot_B + 2*k2.omegadot_B + 2*k3.omegadot_B + k4.omegadot_B) / 6.0,
        ),
        alpha=dt
    )

    # final quaternion renorm
    _renorm_quat_inplace(x_next)

    # mass positivity guard outside ODE
    if opts.clip_mass_to_dry:
        x_next.m = max(x_next.m, params.m_dry)

    return x_next

def rollout(f: FContinuous,
            x0: State,
            U: Union[Sequence[Control], UCallback],
            dt: float,
            N: int,
            params: VehicleEnvParams,
            opts: Optional[IntegrateOptions] = None,
            t0: float = 0.0) -> Sequence[State]:
    """
    Simulate for N steps. U can be:
      - a sequence of length >= N
      - a callback U(t, x) -> Control
    """
    if opts is None:
        opts = IntegrateOptions()

    xs = [copy_state(x0)]
    t = t0
    x = copy_state(x0)

    is_callback = callable(U)

    for k in range(N):
        u = U(t, x) if is_callback else U[k]
        x = sample(f, x, u, dt, params, opts)
        xs.append(copy_state(x))
        t += dt

    return xs
