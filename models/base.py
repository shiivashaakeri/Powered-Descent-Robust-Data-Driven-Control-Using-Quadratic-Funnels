# nominal/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

import cvxpy as cvx  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

FType = Callable[[np.ndarray, np.ndarray], np.ndarray]
JType = Callable[[np.ndarray, np.ndarray], np.ndarray]


class ModelBase(ABC):
    """
    Abstract base for nominal vehicle models used with SCVX.

    Conventions
    ----------
    - State trajectory X has shape (n_x, K)
    - Control trajectory U has shape (n_u, K)
    - All angles are assumed in radians within this class (any deg->rad handled by config loader).
    - All units can be non-dimensionalized using r_scale and m_scale.
    """

    n_x: int
    n_u: int

    # Scales (default: no scaling)
    r_scale: float = 1.0
    m_scale: float = 1.0

    # Problem boundary conditions (set by concrete models)
    x_init: np.ndarray
    x_final: np.ndarray

    # Gravity etc. may be set in concrete models
    def __init__(self) -> None:
        super().__init__()
        # Placeholders for lambdified dynamics
        self._f: Optional[FType] = None
        self._A: Optional[JType] = None
        self._B: Optional[JType] = None

    # ---------- Scaling API ----------
    def nondimensionalize(self) -> None:
        """Non-dimensionalize parameters and boundaries (override to adjust constants)."""
        # Default: only states/controls via helpers; concrete models will update parameters.
        self.x_init = self.x_nondim(self.x_init.copy())
        self.x_final = self.x_nondim(self.x_final.copy())

    def redimensionalize(self) -> None:
        """Re-dimensionalize parameters (override to adjust constants)."""
        # Default: only states/controls via helpers; concrete models will update parameters.
        self.x_init = self.x_redim(self.x_init.copy())
        self.x_final = self.x_redim(self.x_final.copy())

    def x_nondim(self, x: np.ndarray) -> np.ndarray:
        """Nondimensionalize state vector (override for model-specific layout)."""
        return x

    def u_nondim(self, u: np.ndarray | float) -> np.ndarray | float:
        """Nondimensionalize control (override for model-specific layout)."""
        return u

    def x_redim(self, x: np.ndarray) -> np.ndarray:
        """Redimensionalize state vector (override for model-specific layout)."""
        return x

    def u_redim(self, u: np.ndarray | float) -> np.ndarray | float:
        """Redimensionalize control."""
        return u

    # ---------- Dynamics ----------
    @abstractmethod
    def get_equations(self) -> Tuple[FType, JType, JType]:
        """
        Returns lambdified functions (f, A, B):
          f(x, u) -> (n_x, 1)
          A(x, u) -> (n_x, n_x)
          B(x, u) -> (n_x, n_u)
        """
        ...

    # ---------- Initialization ----------
    @abstractmethod
    def initialize_trajectory(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fill X, U with a warm start (dynamically inconsistent is fine)."""
        ...

    # ---------- Model-specific SCVX pieces ----------
    @abstractmethod
    def get_constraints(
        self,
        X_v: cvx.Expression,
        U_v: cvx.Expression,
        X_last_p: cvx.Expression,
        U_last_p: cvx.Expression,
    ) -> List[cvx.Constraint]:
        """Return model constraints as CVXPY constraints."""
        ...

    def get_objective(
        self,
        X_v: cvx.Expression,  # noqa: ARG002
        U_v: cvx.Expression,  # noqa: ARG002
        X_last_p: cvx.Expression,  # noqa: ARG002
        U_last_p: cvx.Expression,  # noqa: ARG002
    ) -> Optional[cvx.Expression]:
        """Return a CVXPY objective term (cvx.Minimize(expr)) or None."""
        return None

    def get_linear_cost(self) -> float:
        """Return scalar linearized constraint cost for reporting (default 0)."""
        return 0.0

    def get_nonlinear_cost(self, X: Optional[np.ndarray] = None, U: Optional[np.ndarray] = None) -> float:  # noqa: ARG002
        """Return scalar nonlinear constraint cost for reporting (default 0)."""
        return 0.0
