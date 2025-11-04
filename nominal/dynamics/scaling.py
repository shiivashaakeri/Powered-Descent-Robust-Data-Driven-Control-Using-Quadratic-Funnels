# nominal/dynamics/scaling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np  # pyright: ignore[reportMissingImports]


class ScalableModel(Protocol):
    r_scale: float
    m_scale: float
    x_init: np.ndarray
    x_final: np.ndarray

    def nondimensionalize(self) -> None: ...
    def redimensionalize(self) -> None: ...
    def x_nondim(self, x: np.ndarray) -> np.ndarray: ...
    def x_redim(self, x: np.ndarray) -> np.ndarray: ...
    def u_nondim(self, u): ...
    def u_redim(self, u): ...


@dataclass
class Scales:
    r_scale: float
    m_scale: float


def compute_default_scales_from_state(x_init: np.ndarray, m_wet: float | None = None) -> Scales:
    """
    Heuristic: r_scale = ||position_init||_2 (or 1 if absent), m_scale = m_wet if provided else 1.
    Works for both 2D and 6-DoF layouts (pos commonly starts at indices 0:2 or 1:4).
    """
    # Try to infer a position segment robustly
    if x_init.size >= 4:
        # 2D model typically: [rx, ry, vx, vy, ...]
        r_guess = np.linalg.norm(x_init[0:2])
    elif x_init.size >= 7:
        # 6-DoF model typically: [m, r(3), v(3), ...]
        r_guess = np.linalg.norm(x_init[1:4])
    else:
        r_guess = 1.0
    r_scale = float(r_guess if r_guess > 0 else 1.0)
    m_scale = float(m_wet) if m_wet and m_wet > 0 else 1.0
    return Scales(r_scale=r_scale, m_scale=m_scale)


def apply_nondimensionalize(model: ScalableModel) -> None:
    """
    Call the model's own nondimensionalization (preferred - preserves parameter semantics).
    """
    model.nondimensionalize()


def apply_redimensionalize(model: ScalableModel) -> None:
    """
    Call the model's own redimensionalization.
    """
    model.redimensionalize()
