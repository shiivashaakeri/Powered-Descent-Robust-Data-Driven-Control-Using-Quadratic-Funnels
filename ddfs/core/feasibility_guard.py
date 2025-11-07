# ddfs/core/feasibility_guard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]


# -------------------------------------------------------------------
# Config & Result containers
# -------------------------------------------------------------------
@dataclass
class FeasibilityConfig:
    # thresholds for deciding to suspend excitation
    margin_thresh_state: float = 1e-6
    margin_thresh_input: float = 1e-6
    # subtract from available margins before scaling (extra safety)
    safety_margin: float = 5e-4
    # cap on overall excitation scaling
    max_scale: float = 1.0
    # whether to enforce MVIE ellipsoids if provided
    use_mvie_state: bool = True
    use_mvie_input: bool = True
    # optional shrinking of MVIE budget (<=1.0 keeps inside ellipsoid)
    mvie_shrink_state: float = 1.0
    mvie_shrink_input: float = 1.0
    # when True, if state margin is too small, suspend excitation regardless of input room
    suspend_on_low_state_margin: bool = True


@dataclass
class GuardResult:
    eps_clamped: np.ndarray          # final excitation to apply (possibly zero)
    scale_halfspaces: float          # alpha from halfspace constraints (inputs)
    scale_mvie: float                # beta from R_max (inputs)
    min_state_margin: float          # min_i (b_x - A_x @ x)_i
    min_input_margin: float          # min_j (b_u - A_u @ u_base)_j   (before eps)
    mvie_state_ok: bool              # True if eta within Q (if Q provided/enabled)
    mvie_input_budget: float         # sqrt( target_radius^2 / (eps^T R eps) ) raw (inf if eps=0)
    suspended: bool                  # True if excitation was zeroed
    reason: str                      # brief reason string


# -------------------------------------------------------------------
# Halfspace builders (model-agnostic interface)
# Each returns (A, b) such that feasible set ≈ { z | A z <= b }
# -------------------------------------------------------------------
HalfspaceBuilder = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


# ---------- 2D state/input linearizations ----------
def _state_halfspaces_2d(x_nom: np.ndarray, *, t_max: float, w_max: float) -> Tuple[np.ndarray, np.ndarray]:
    n = 6
    A, b = [], []
    # |theta| <= t_max
    e = np.zeros(n); e[4] = 1.0; A += [ e.copy(), -e.copy() ]; b += [ t_max, t_max ]
    # |omega| <= w_max
    e = np.zeros(n); e[5] = 1.0; A += [ e.copy(), -e.copy() ]; b += [ w_max, w_max ]
    # altitude: ry >= 0  -> -ry <= 0
    e = np.zeros(n); e[1] = -1.0; A.append(e.copy()); b.append(0.0)
    return (np.vstack(A), np.asarray(b, float)) if A else (np.zeros((0, n)), np.zeros((0,), float))


def _input_halfspaces_2d(u_nom: np.ndarray, *, max_gimbal: float, T_min: float, T_max: float) -> Tuple[np.ndarray, np.ndarray]:
    n = 2
    A, b = [], []
    # |gimbal| <= max_gimbal
    e = np.zeros(n); e[0] = 1.0; A += [ e.copy(), -e.copy() ]; b += [ max_gimbal, max_gimbal ]
    # thrust bounds: +T <= T_max ; -T <= -T_min
    e = np.zeros(n); e[1] = 1.0;  A.append(e.copy());   b.append(T_max)
    e[1] = -1.0;                A.append(e.copy());   b.append(-T_min)
    return (np.vstack(A), np.asarray(b, float)) if A else (np.zeros((0, n)), np.zeros((0,), float))


# ---------- 6DoF helpers ----------
def _linearize_soc_state_glideslope(x_nom: np.ndarray, tan_gamma: float) -> Tuple[np.ndarray, float]:
    # x = [m, rE, rN, rU, vE, vN, vU, q0,q1,q2,q3, wBx,wBy,wBz]
    rE, rN, rU = x_nom[1], x_nom[2], x_nom[3]
    rn = np.array([rE, rN], float)
    nr = float(np.linalg.norm(rn))
    if nr < 1e-12:
        # fallback direction
        grad = np.array([1.0, 0.0])
        rn_norm = 0.0
    else:
        grad = rn / nr
        rn_norm = nr
    a = np.zeros(14); a[1], a[2], a[3] = grad[0], grad[1], -1.0 / tan_gamma
    h0 = rn_norm - rU / tan_gamma                      # h(r) ≈ ||r_xy|| - rU/tanγ
    b = float(a @ x_nom - h0)                          # a^T x_nom - h(x_nom)
    return a, b


def _linearize_soc_norm_cap(x0: np.ndarray, idxs: Sequence[int], cap: float, n: int) -> Tuple[np.ndarray, float]:
    z0 = x0[list(idxs)].astype(float)
    nz = float(np.linalg.norm(z0))
    a = np.zeros(n)
    if nz < 1e-12:
        a[idxs[0]] = 1.0
        return a, cap
    g = z0 / nz
    for c, idx in zip(g, idxs):
        a[idx] = c
    h0 = nz - cap
    b = float(a @ x0 - h0)
    return a, b


def _state_halfspaces_6dof(x_nom: np.ndarray, *, m_dry: float, tan_gamma: float,
                           cos_theta_max: float, w_B_max: float) -> Tuple[np.ndarray, np.ndarray]:
    n = 14
    A, b = [], []

    # mass >= m_dry  -> -m <= -m_dry
    e = np.zeros(n); e[0] = -1.0; A.append(e.copy()); b.append(-m_dry)

    # glide slope
    a_gs, b_gs = _linearize_soc_state_glideslope(x_nom, tan_gamma)
    A.append(a_gs); b.append(b_gs)

    # tilt: ||q_vec|| <= sqrt((1 - cosθ_max)/2)
    tilt_cap = float(np.sqrt(max(0.0, (1.0 - cos_theta_max) / 2.0)))
    a_tilt, b_tilt = _linearize_soc_norm_cap(x_nom, idxs=[8, 9, 10], cap=tilt_cap, n=n)
    A.append(a_tilt); b.append(b_tilt)

    # body rate: ||w_B|| <= w_B_max
    a_w, b_w = _linearize_soc_norm_cap(x_nom, idxs=[11, 12, 13], cap=float(w_B_max), n=n)
    A.append(a_w); b.append(b_w)

    return np.vstack(A), np.asarray(b, float)


def _input_halfspaces_6dof(u_nom: np.ndarray, *, tan_delta_max: float, T_max: float) -> Tuple[np.ndarray, np.ndarray]:
    n = 3
    A, b = [], []

    # gimbal cone: ||u_xy|| - tan_delta * uz <= 0  (linearize at u_nom)
    ux, uy, uz = u_nom
    rxy = np.linalg.norm([ux, uy])
    if rxy < 1e-12:
        e = np.zeros(n); e[0] = 1.0; e[2] = -tan_delta_max
        A.append(e.copy()); b.append(0.0)
    else:
        gxy = np.array([ux, uy]) / rxy
        a = np.zeros(n); a[0], a[1], a[2] = gxy[0], gxy[1], -tan_delta_max
        h0 = rxy - tan_delta_max * uz
        b.append(float(a @ u_nom - h0)); A.append(a.copy())

    # thrust cap: ||u|| <= T_max  (linearize at u_nom)
    nu = float(np.linalg.norm(u_nom))
    if nu < 1e-12:
        e = np.zeros(n); e[2] = 1.0
        A.append(e.copy()); b.append(T_max)
    else:
        gu = u_nom / nu
        a = gu.copy()
        h0 = nu - T_max
        b.append(float(a @ u_nom - h0)); A.append(a.copy())

    return np.vstack(A), np.asarray(b, float)


# -------------------------------------------------------------------
# Utility: scale factor for keeping u_base + alpha*eps inside {Au z <= bu}
#   Given margins m = b - A u_base, find max alpha >= 0 with A (u_base + alpha*eps) <= b
#   For each i with A_i @ eps > 0: alpha_i = (m_i - safety_margin) / (A_i @ eps)
#   Take alpha = min_i alpha_i ; if no positive denominators, alpha = +inf
# -------------------------------------------------------------------
def _scale_to_halfspaces(
    eps: np.ndarray,
    u_base: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    *,
    safety_margin: float = 0.0,
) -> float:
    if eps.size == 0 or A.size == 0:
        return float("inf")
    margins = b - A @ u_base
    # Only constraints that tighten with +eps
    denom = A @ eps
    pos = denom > 1e-12
    if not np.any(pos):
        return float("inf")
    alphas = (margins[pos] - safety_margin) / denom[pos]
    if alphas.size == 0:
        return float("inf")
    return float(np.maximum(0.0, np.min(alphas)))


# -------------------------------------------------------------------
# MVIE checks/scales
#   State feasibility: eta^T Q eta <= (mvie_shrink_state)^2
#   Input budget:     (alpha*eps)^T R (alpha*eps) <= (mvie_shrink_input)^2
#                     => alpha <= sqrt(r^2 / (eps^T R eps))
# -------------------------------------------------------------------
def _mvie_state_ok(eta: np.ndarray, Q: Optional[np.ndarray], shrink: float) -> bool:
    if Q is None:
        return True
    val = float(eta.T @ Q @ eta)
    return val <= (shrink ** 2) + 1e-12


def _mvie_input_scale(eps: np.ndarray, R: Optional[np.ndarray], shrink: float) -> float:
    if R is None or eps.ndim == 0 or np.allclose(eps, 0.0):
        return float("inf")
    num = shrink ** 2
    den = float(eps.T @ R @ eps)
    if den <= 1e-18:
        return float("inf")
    return float(np.sqrt(max(0.0, num / den)))


# -------------------------------------------------------------------
# FeasibilityGuard
# -------------------------------------------------------------------
class FeasibilityGuard:
    """
    Given builders for linearized feasible sets and (optional) MVIE ellipsoids,
    clamp/suspend a proposed excitation so that:
       x = x_nom + eta remains feasible (state halfspaces & Q_max),
       u = u_nom + K eta + eps remains feasible (input halfspaces & R_max).
    """

    def __init__(
        self,
        *,
        model_name: str,
        build_state_halfspaces: HalfspaceBuilder,
        build_input_halfspaces: HalfspaceBuilder,
        Q_all: Optional[np.ndarray] = None,  # (n_x, n_x, K) or None
        R_all: Optional[np.ndarray] = None,  # (n_u, n_u, K) or None
        cfg: Optional[FeasibilityConfig] = None,
    ):
        self.model_name = model_name
        self.build_state = build_state_halfspaces
        self.build_input = build_input_halfspaces
        self.Q_all = Q_all
        self.R_all = R_all
        self.cfg = cfg or FeasibilityConfig()

    def guard(
        self,
        k: int,
        x_nom_k: np.ndarray,
        u_nom_k: np.ndarray,
        eta_k: np.ndarray,
        K_i: np.ndarray,
        eps_proposed: np.ndarray,
    ) -> GuardResult:
        x_applied = x_nom_k + eta_k
        u_base = u_nom_k + K_i @ eta_k

        # Linearized halfspaces at the nominal point (as per design)
        Ax, bx = self.build_state(x_nom_k)
        Au, bu = self.build_input(u_nom_k)

        # State margins
        margins_x = bx - Ax @ x_applied if Ax.size else np.array([np.inf])
        min_margin_x = float(np.min(margins_x)) if margins_x.size else float("inf")

        # Optional state MVIE
        Qk = self.Q_all[:, :, k] if (self.Q_all is not None) else None
        mvie_state_ok = (not self.cfg.use_mvie_state) or _mvie_state_ok(eta_k, Qk, self.cfg.mvie_shrink_state)

        # If state is too tight, suspend excitation if policy says so
        if self.cfg.suspend_on_low_state_margin and min_margin_x <= self.cfg.margin_thresh_state:
            return GuardResult(
                eps_clamped=np.zeros_like(eps_proposed),
                scale_halfspaces=0.0,
                scale_mvie=0.0,
                min_state_margin=min_margin_x,
                min_input_margin=float("inf"),
                mvie_state_ok=mvie_state_ok,
                mvie_input_budget=float("inf"),
                suspended=True,
                reason="low_state_margin",
            )
        if not mvie_state_ok:
            return GuardResult(
                eps_clamped=np.zeros_like(eps_proposed),
                scale_halfspaces=0.0,
                scale_mvie=0.0,
                min_state_margin=min_margin_x,
                min_input_margin=float("inf"),
                mvie_state_ok=False,
                mvie_input_budget=float("inf"),
                suspended=True,
                reason="outside_Qmax",
            )

        # Input margins (before excitation)
        margins_u_base = bu - Au @ u_base if Au.size else np.array([np.inf])
        min_margin_u = float(np.min(margins_u_base)) if margins_u_base.size else float("inf")

        if min_margin_u <= self.cfg.margin_thresh_input:
            return GuardResult(
                eps_clamped=np.zeros_like(eps_proposed),
                scale_halfspaces=0.0,
                scale_mvie=0.0,
                min_state_margin=min_margin_x,
                min_input_margin=min_margin_u,
                mvie_state_ok=mvie_state_ok,
                mvie_input_budget=float("inf"),
                suspended=True,
                reason="low_input_margin",
            )

        # Compute allowable scales
        alpha_hs = _scale_to_halfspaces(
            eps_proposed, u_base, Au, bu, safety_margin=self.cfg.safety_margin
        )
        Rk = self.R_all[:, :, k] if (self.R_all is not None) else None
        beta_mvie = (
            _mvie_input_scale(eps_proposed, Rk, self.cfg.mvie_shrink_input)
            if self.cfg.use_mvie_input
            else float("inf")
        )

        # Final scale
        scale = min(self.cfg.max_scale, alpha_hs, beta_mvie)
        if not np.isfinite(scale) or scale <= 0.0:
            return GuardResult(
                eps_clamped=np.zeros_like(eps_proposed),
                scale_halfspaces=float(alpha_hs),
                scale_mvie=float(beta_mvie),
                min_state_margin=min_margin_x,
                min_input_margin=min_margin_u,
                mvie_state_ok=mvie_state_ok,
                mvie_input_budget=float(beta_mvie),
                suspended=True,
                reason="no_feasible_scale",
            )

        eps_final = scale * eps_proposed
        return GuardResult(
            eps_clamped=eps_final,
            scale_halfspaces=float(alpha_hs),
            scale_mvie=float(beta_mvie),
            min_state_margin=min_margin_x,
            min_input_margin=min_margin_u,
            mvie_state_ok=mvie_state_ok,
            mvie_input_budget=float(beta_mvie),
            suspended=False,
            reason="ok",
        )


# -------------------------------------------------------------------
# Convenience factories for your two models
#   Pass in the *model instance* so we can read its limits.
#   These return a ready-to-use FeasibilityGuard with default builders.
# -------------------------------------------------------------------
def make_guard_rocket2d(
    *,
    model,                     # Rocket2D (already nondimensionalized)
    Q_all: Optional[np.ndarray] = None,
    R_all: Optional[np.ndarray] = None,
    cfg: Optional[FeasibilityConfig] = None,
) -> FeasibilityGuard:
    def build_state(x_nom: np.ndarray):
        return _state_halfspaces_2d(x_nom, t_max=float(model.t_max), w_max=float(model.w_max))

    def build_input(u_nom: np.ndarray):
        return _input_halfspaces_2d(
            u_nom, max_gimbal=float(model.max_gimbal), T_min=float(model.T_min), T_max=float(model.T_max)
        )

    return FeasibilityGuard(
        model_name="rocket2d",
        build_state_halfspaces=build_state,
        build_input_halfspaces=build_input,
        Q_all=Q_all,
        R_all=R_all,
        cfg=cfg,
    )


def make_guard_rocket6dof(
    *,
    model,                     # Rocket6DoF (already nondimensionalized)
    Q_all: Optional[np.ndarray] = None,
    R_all: Optional[np.ndarray] = None,
    cfg: Optional[FeasibilityConfig] = None,
) -> FeasibilityGuard:
    def build_state(x_nom: np.ndarray):
        return _state_halfspaces_6dof(
            x_nom,
            m_dry=float(model.m_dry),
            tan_gamma=float(model.tan_gamma_gs),
            cos_theta_max=float(model.cos_theta_max),
            w_B_max=float(model.w_B_max),
        )

    def build_input(u_nom: np.ndarray):
        return _input_halfspaces_6dof(
            u_nom,
            tan_delta_max=float(model.tan_delta_max),
            T_max=float(model.T_max),
        )

    return FeasibilityGuard(
        model_name="rocket6dof",
        build_state_halfspaces=build_state,
        build_input_halfspaces=build_input,
        Q_all=Q_all,
        R_all=R_all,
        cfg=cfg,
    )