# ddfs/envelopes/max_ellipsoids.py
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Optional, Tuple, Dict, List
import numpy as np # type: ignore
import cvxpy as cp # type: ignore

# Core MVIE solver (centered at x_nom)
@dataclass
class MVIESettings:
    shrink: float = 0.98             # shrink margins to keep strict interior
    eps_margin: float = 1e-9         # drop constraints with nonpositive margin
    solver: str = "ECOS"             # solver for CVXPY
    verbose: bool = False            # verbose output

def mvie_centered(
    A: np.ndarray,
    b: np.ndarray,
    x_nom: np.ndarray,
    radius_cap: Optional[np.ndarray | float] = None,
    settings: MVIESettings = MVIESettings(),
) -> np.ndarray:

    """
    Solve: max logdet(P) s.t. a_i^T P a_i <= (margin_i)^2, P>=0, 
    and P <= (radius_cap)^2 I
    where margin_i = shrink * (b_i - a_i^T x_nom).
    Returns PSD matrix P (ellipsoid E={eta: eta^T P^{-1} eta <= 1})
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x_nom = np.asarray(x_nom, dtype=float).reshape(-1)
    assert A.shape[0] == b.shape[0], "A and b must have the same number of rows"
    n = A.shape[1]

    # Compute margins
    raw = b - A @ x_nom
    margins = settings.shrink * raw
    keep = margins > settings.eps_margin
    A = A[keep]
    margins = margins[keep]
    if A.shape[0] == 0:
        # no valid halfspaces, return tiny ellipsoid
        return 1e-6 * np.eye(n)
    
    P = cp.Variable((n, n), PSD=True)
    constraints =[]
    for i in range(A.shape[0]):
        ai = A[i, :].reshape(-1, 1)
        constraints += [cp.quad_form(ai, P) <= (margins[i])**2]
    
    # optinal cap: 0 <= P <= (cap)^2 I or diag(cap)^2
    if radius_cap is not None:
        if np.isscalar(radius_cap):
            cap2 = float(radius_cap)**2
            constraints += [P <= cap2 * np.eye(n)]
        else:
            r = np.asarray(radius_cap, dtype=float).reshape(-1)
            assert r.size == n, "radius_cap must be a scalar or have n elements"
            constraints += [P <= np.diag(r**2)]
    
    prob = cp.Problem(cp.Maximize(cp.log_det(P)), constraints)
    prob.solve(solver=settings.solver, verbose=settings.verbose)
    if P.value is None:
        return 1e-6 * np.eye(n)
    return 0.5 * (P.value + P.value.T)

# Half space builders
def _append(A: List[np.ndarray], b: List[float], a: np.ndarray, beta: float) -> None:
    A.append(a.astype(float).reshape(-1, 1))
    b.append(float(beta))

def _stack(A_list: List[np.ndarray], b_list: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    if not A_list:
        return np.zeros((0, 0)), np.zeros((0,))
    A = np.vstack(A_list)
    b = np.asarray(b_list, dtype=float)
    return A, b

# Rocket2D
# state: [rx, ry, vx, vy, theta, omega], input: [gimbal, T]
def halfspaces_rocket2d_state(x_nom: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    rx, ry, vx, vy, th, om = range(6)
    A, b = [], []

    # ry >= 0, -ry <= 0
    a = np.zeros(6)
    a[ry] = -1.0
    _append(A, b, a, 0)
    
    # |theta| <= t_max
    t_max = float(params["t_max"])
    a = np.zeros(6)
    a[th] = 1.0
    _append(A, b, a, t_max)
    a = np.zeros(6)
    a[th] = -1.0
    _append(A, b, a, t_max)

    # ||omega|| <= w_max
    w_max = float(params["w_max"])
    a = np.zeros(6)
    a[om] = 1.0
    _append(A, b, a, w_max)
    a = np.zeros(6)
    a[om] = -1.0
    _append(A, b, a, w_max)

    return _stack(A, b)

def halfspaces_rocket2d_input(x_nom: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    gim, T = range(2)
    A, b = [], []

    # |gimbal| <= gimbal_max
    max_gim = float(params["max_gimbal"])
    a = np.zeros(2)
    a[gim] = 1.0
    _append(A, b, a, max_gim)
    a = np.zeros(2)
    a[gim] = -1.0
    _append(A, b, a, max_gim)

    # T <= T_max, T >= T_min
    T_max = float(params["T_max"])
    T_min = float(params["T_min"])
    a = np.zeros(2)
    a[T] = 1.0
    _append(A, b, a, T_max)
    a = np.zeros(2)
    a[T] = -1.0
    _append(A, b, a, -T_min)

    return _stack(A, b)


# ---------------- Rocket6DoF ----------------
# State x = [m, r(3), v(3), q(4), w(3)],  Input u = [ux, uy, uz]
def halfspaces_rocket6dof_state(x_nom: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    # indices
    m = 0; r = slice(1, 4); v = slice(4, 7); q = slice(7, 11); w = slice(11, 14)
    rE, rN, rU = 1, 2, 3
    q1, q2 = 8, 9  # q = [q0,q1,q2,q3]
    A, b = [], []
    # Mass lower bound: m >= m_dry  =>  -m <= -m_dry
    m_dry = float(params["m_dry"])
    a = np.zeros(14); a[m] = -1.0; _append(A, b, a, -m_dry)
    # Altitude nonnegative: r_U >= 0  => -r_U <= 0
    a = np.zeros(14); a[rU] = -1.0; _append(A, b, a, 0.0)
    # Glide-slope: ||r_EN|| <= s * r_U  (inner box): |rE| <= s rU / √2 ; |rN| <= s rU / √2
    s = 1.0 / float(params["tan_gamma_gs"])  # s = 1/tan(gamma)
    coef = s / np.sqrt(2.0)
    a = np.zeros(14); a[rE] = +1.0; a[rU] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(14); a[rE] = -1.0; a[rU] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(14); a[rN] = +1.0; a[rU] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(14); a[rN] = -1.0; a[rU] += -coef; _append(A, b, a, 0.0)
    # Tilt: ||[q1,q2]|| <= R  → inner box |q1|,|q2| <= R/√2
    R_tilt = np.sqrt((1.0 - float(params["cos_theta_max"])) / 2.0)
    qcap = R_tilt / np.sqrt(2.0)
    a = np.zeros(14); a[q1] = +1.0; _append(A, b, a,  qcap)
    a = np.zeros(14); a[q1] = -1.0; _append(A, b, a,  qcap)
    a = np.zeros(14); a[q2] = +1.0; _append(A, b, a,  qcap)
    a = np.zeros(14); a[q2] = -1.0; _append(A, b, a,  qcap)
    # Body-rate: ||w|| <= w_max  → inner box |w_i| <= w_max/√3
    w_cap = float(params["w_B_max"]) / np.sqrt(3.0)
    for idx in (11, 12, 13):
        a = np.zeros(14); a[idx] = +1.0; _append(A, b, a,  w_cap)
        a = np.zeros(14); a[idx] = -1.0; _append(A, b, a,  w_cap)
    return _stack(A, b)

def halfspaces_rocket6dof_input(u_nom: np.ndarray, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    ux, uy, uz = 0, 1, 2
    A, b = [], []
    # Gimbal cone: ||u_xy|| <= tan_delta * uz, uz >= 0
    tan_delta = float(params["tan_delta_max"])
    coef = tan_delta / np.sqrt(2.0)
    a = np.zeros(3); a[ux] = +1.0; a[uz] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(3); a[ux] = -1.0; a[uz] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(3); a[uy] = +1.0; a[uz] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(3); a[uy] = -1.0; a[uz] += -coef; _append(A, b, a, 0.0)
    a = np.zeros(3); a[uz] = -1.0; _append(A, b, a, 0.0)  # uz >= 0
    # Upper thrust bound: ||u|| <= T_max → inner box |u_i| <= T_max/√3
    T_max = float(params["T_max"])
    box = T_max / np.sqrt(3.0)
    for idx in (0, 1, 2):
        a = np.zeros(3); a[idx] = +1.0; _append(A, b, a,  box)
        a = np.zeros(3); a[idx] = -1.0; _append(A, b, a,  box)
    # Lower thrust bound: ||u|| >= T_min → inner halfspace  n^T u >= T_min with n = u_nom/‖u_nom‖
    T_min = float(params["T_min"])
    n = np.asarray(u_nom, float).reshape(-1)
    n_norm = np.linalg.norm(n)
    if n_norm > 0:
        n = n / n_norm
        a = -n  # -n^T u <= -T_min
        _append(A, b, a, -T_min)
    return _stack(A, b)

# --------- Convenience wrappers to get Q_max, R_max at one time step ----------
def qmax_from_constraints(model_name: str,
                          x_nom: np.ndarray,
                          params_state: Dict[str, float],
                          x_cap: Optional[np.ndarray | float] = None,
                          settings: MVIESettings = MVIESettings()) -> np.ndarray:
    if model_name == "rocket2d":
        A, b = halfspaces_rocket2d_state(x_nom, params_state)
    elif model_name == "rocket6dof":
        A, b = halfspaces_rocket6dof_state(x_nom, params_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return mvie_centered(A, b, x_nom, x_cap, settings)

def rmax_from_constraints(model_name: str,
                          u_nom: np.ndarray,
                          params_input: Dict[str, float],
                          u_cap: Optional[np.ndarray | float] = None,
                          settings: MVIESettings = MVIESettings()) -> np.ndarray:
    if model_name == "rocket2d":
        A, b = halfspaces_rocket2d_input(u_nom, params_input)
    elif model_name == "rocket6dof":
        A, b = halfspaces_rocket6dof_input(u_nom, params_input)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return mvie_centered(A, b, u_nom, u_cap, settings)

# --------- Terminal set support (X_f) ----------
def apply_terminal_halfspaces(A: np.ndarray, b: np.ndarray,
                              A_term: Optional[np.ndarray], b_term: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if A_term is None or b_term is None or A_term.size == 0:
        return A, b
    A2 = np.vstack([A, A_term])
    b2 = np.concatenate([b, b_term])
    return A2, b2
    
    
    