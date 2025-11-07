# ddfs/core/excitation.py
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Value
from typing import Optional, Sequence, Tuple, Literal, Dict

import numpy as np # type: ignore

ExciteType = Literal["prbs", "rademacher", "gaussian"]

# ----------------------
# Linearized input halfspace
# ----------------------
def _halfspaces_input_2d(
    u_nom: np.ndarray,
    max_gimbal: float,
    T_min: float,
    T_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    A = np.array(
        [
            [+1.0, 0.0], # gimbal <= max_gimbal
            [-1.0, 0.0], # gimbal >= -max_gimbal
            [0.0, +1.0], # T <= T_max
            [0.0, -1.0], # -T <= -T_min
        ],
        dtype=float,
    )
    b = np.array(
        [
            max_gimbal,
            max_gimbal,
            T_max,
            T_min,
        ],
        dtype=float,
    )
    return A, b

def _halfspaces_input_6dof(
    u_nom: np.ndarray,
    tan_delta_max: float,
    T_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    """
    ux, uy, uz = float(u_nom[0]), float(u_nom[1]), float(u_nom[2])

    A_list, b_list = [], []

    # (1) gimbal cone: g(u) := ||[ux, uy]|| - tan * uz <= 0
    n_xy = np.hypot(ux, uy)
    if n_xy < 1e-12:
        a = np.array([1.0, 0.0, -tan_delta_max], dtype=float)
        g0 = -tan_delta_max * uz
    else:
        a = np.array([ux/n_xy, uy/n_xy, -tan_delta_max], dtype=float)
        g0 = n_xy - tan_delta_max * uz
    
    # linearization at u_nom: a^T (u - u_nom) + g0 <= 0
    b = float(a @ u_nom - g0)
    A_list.append(a)
    b_list.append(b)

    # (2) thrust cap: h(u) := ||u|| - T_max <= 0
    n_u = np.linalg.norm(u_nom)
    if n_u < 1e-12:
        a2 = np.array([0.0, 0.0, 1.0], dtype=float)
        h0 = 0.0
    else:
        a2 = u_nom / n_u
        h0 = n_u
    b2 = float(a2 @ u_nom - (h0 - T_max))
    A_list.append(a2)
    b_list.append(b2)

    return np.vstack(A_list), np.asarray(b_list, dtype=float)

def _scale_to_halfspaces(
    eps: np.ndarray,
    u_nom: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    safety_margin: float = 0.0,
) -> float:
    """
    """
    margins = b - A @ u_nom - safety_margin
    if np.any(margins < 0.0):
        return 0.0
    
    Ae = A @ eps
    ub_list = []
    for mi, aei in zip(margins, Ae):
        if aei > 1e-16:
            ub_list.append(mi / aei)
    if not ub_list:
        return 1.0
    alpha = float(np.clip(np.min(ub_list), 0.0, 1.0))
    return alpha

# ----------------------
# Ellipsoid shaping utilities
# ----------------------
def _shape_to_ellipsoid(
    eps_raw: np.ndarray,
    R: Optional[np.ndarray],
    rho: float,
) -> np.ndarray:
    """
    """
    if R is None:
        nrm = np.linalg.norm(eps_raw)
        if nrm < 1e-12:
            return np.zeros_like(eps_raw)
        return (rho / nrm) * eps_raw
    
    q = float(eps_raw.T @ R @ eps_raw)
    if q < 1e-24:
        return np.zeros_like(eps_raw)
    return float(rho / np.sqrt(q)) * eps_raw

@dataclass
class ExcitationConfig:
    kind: ExciteType = "rademacher"             # "prbs" | "rademacher" | "gaussian"
    rho: float = 0.05                           # amplitude (L2 or ellipsoidal radius)
    hold: int = 1                               # piecewise-constant span in steps
    margin_thresh: float = 0.0                  # suspend if min margin <= this
    safety_margin: float = 0.0                  # shrink each halfspace by this
    active_dims: Optional[Sequence[int]] = None # indices of inputs to excite
    guassian_std: float = 1.0                   # raw std before normalization (guassian)

class ExcitationPolicy:
    """
    """
    def __init__(
        self,
        m: int,
        cfg: Optional[ExcitationConfig] = None,
    ) -> None:
        if m <= 0:
            raise ValueError("Input dimension m must be positive.")
        self.m = int(m)
        self.cfg = cfg or ExcitationConfig()
        self._cached_vec: Optional[np.ndarray] = None
        self._cached_block: Tuple[int, int] = (-1, -1)
    
        if self.cfg.active_dims is None:
            self._mask = np.ones((self.m,), dtype=bool)
        else:
            mask = np.zeros((self.m,), dtype=bool)
            mask[np.asarray(self.cfg.active_dims, dtype=int)] = True
            self._mask = mask
    
    # -------- public API --------
    def next(
        self,
        k: int,
        u_nom: np.ndarray,
        model_name: str,
        model_params: Dict[str, float],
        R_k: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        """
        rng = rng or np.random.default_rng()
        u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
        if u_nom.size != self.m:
            raise ValueError(f"u_nom has dimension {u_nom.size}, expected {self.m}")
        
        # 1) draw raw excitation 
        eps_raw = self._draw_raw(k, rng)

        # apply channel mask before shaping
        eps_raw = eps_raw * self._mask.astype(float)

        # 2) ellipsoid (or L2) shaping to meet amplitude budget rho
        eps_shaped = _shape_to_ellipsoid(eps_raw, R_k, self.cfg.rho)

        # 3) half-space scaling to remain feasible (linearized at u_nom)
        A, b = self._build_input_halfspaces(model_name, u_nom, model_params)

        # if margings are small, suspend excitation
        margins = b - A @ u_nom
        if np.min(margins) <= self.cfg.margin_thresh:
            return np.zeros(self.m, dtype=float)

        alpha = _scale_to_halfspaces(
            eps_shaped, u_nom, A, b, safety_margin=self.cfg.safety_margin
        )

        # 4) apply scaling and return
        return alpha * eps_shaped
    
    # -------- internals --------
    def _draw_raw(self, k: int, rng: np.random.Generator) -> np.ndarray:
        """
        """
        block_idx = k // max(1, self.cfg.hold)
        if self._cached_vec is not None and block_idx == self._cached_block[0]:
            return self._cached_vec.copy()
        
        kind = self.cfg.kind
        if kind == "prbs":
            signs = rng.choice([-1.0, 1.0], size=self.m)
            vec = signs
        elif kind == "rademacher":
            vec = rng.choice([-1.0, 1.0], size=self.m)
        elif kind == "gaussian":
            vec = rng.normal(loc=0.0, scale=self.cfg.guassian_std, size=self.m)
        else:
            raise ValueError(f"Invalid excitation type: {kind}")
        
        self._cached_vec = vec.astype(float)
        self._cached_block = (block_idx, max(1, self.cfg.hold))
        return self._cached_vec.copy()

    def _build_input_halfspaces(
        self,
        model_name: str,
        u_nom: np.ndarray,
        params: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:

        if model_name == "rocket2d":
            try:
                A, b = _halfspaces_input_2d(
                    u_nom=u_nom,
                    max_gimbal=float(params["max_gimbal"]),
                    T_min=float(params["T_min"]),
                    T_max=float(params["T_max"]),
                )
            except KeyError as e:
                raise KeyError(f"Missing required parameter for {model_name}: {e}") from None
            return A, b
        
        if model_name == "rocket6dof":
            try:
                A, b = _halfspaces_input_6dof(
                    u_nom=u_nom,
                    tan_delta_max=float(params["tan_delta_max"]),
                    T_max=float(params["T_max"]),
                )
            except KeyError as e:
                raise KeyError(f"Missing required parameter for {model_name}: {e}") from None
            return A, b
        
        raise ValueError(f"Unsupported model: {model_name}")
