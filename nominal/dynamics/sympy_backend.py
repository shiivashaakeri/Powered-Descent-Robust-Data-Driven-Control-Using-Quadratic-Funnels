# nominal/dynamics/sympy_backend.py
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import sympy as sp  # pyright: ignore[reportMissingImports]

FType = Callable[[np.ndarray, np.ndarray], np.ndarray]
JType = Callable[[np.ndarray, np.ndarray], np.ndarray]


def lambdify_dynamics(f: sp.Matrix, x: sp.Matrix, u: sp.Matrix) -> Tuple[FType, JType, JType]:
    """
    Given symbolic dynamics f(x,u), return numpy-callables for:
      f(x,u) in R^{n_x}, A(x,u)=∂f/∂x in R^{n_x x n_x}, B(x,u)=∂f/∂u in R^{n_x x n_u}.
    """
    f = sp.simplify(f)
    A = sp.simplify(f.jacobian(x))
    B = sp.simplify(f.jacobian(u))

    f_func = sp.lambdify((x, u), f, "numpy")
    A_func = sp.lambdify((x, u), A, "numpy")
    B_func = sp.lambdify((x, u), B, "numpy")
    return f_func, A_func, B_func


def assert_shapes(f_func: FType, A_func: JType, B_func: JType, n_x: int, n_u: int) -> None:
    """
    Quick runtime sanity check on shapes.
    """
    x0 = np.zeros((n_x,))
    u0 = np.zeros((n_u,))
    f0 = np.asarray(f_func(x0, u0)).reshape(-1)
    A0 = np.asarray(A_func(x0, u0))
    B0 = np.asarray(B_func(x0, u0))
    assert f0.shape == (n_x,), f"f has wrong shape {f0.shape}"
    assert A0.shape == (n_x, n_x), f"A has wrong shape {A0.shape}"
    assert B0.shape == (n_x, n_u), f"B has wrong shape {B0.shape}"
