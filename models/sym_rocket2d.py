# models/sym_rocket2d.py

from __future__ import annotations
from typing import Callable, List, Optional, Tuple
import numpy as np # pyright: ignore[reportMissingImports]
import sympy as sp # pyright: ignore[reportMissingImports]

def build_symbolics_rocket2d(params_nd: dict) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, List[Callable]]:
    """
    Returns (x, u, f, H_funcs) where H_funcs[i] is a callable (x,u)-> Hessian(f_i) w.r.t z=[x;u].
    All quantities are in nondimensional form.
    """
    rx, ry, vx, vy, th, om = sp.symbols("rx ry vx vy th om", real=True)
    gimbal, T = sp.symbols("gimbal T", real=True)

    x = sp.Matrix([rx, ry, vx, vy, th, om]) # n_x = 6
    u = sp.Matrix([gimbal, T]) # n_u = 2
    z = sp.Matrix.vstack(x, u) # n_z = 8

    m = sp.Float(params_nd["m"])
    I = sp.Float(params_nd["I"])
    g = sp.Float(params_nd["g"])
    r_T = sp.Float(params_nd["r_T"])

    f = sp.Matrix.zeros(6, 1)
    f[0, 0] = vx
    f[1, 0] = vy
    f[2, 0] = (1.0 / m) * sp.sin(th + gimbal) * T
    f[3, 0] = (1.0 / m) * (sp.cos(th + gimbal) * T - m * g)
    f[4, 0] = om
    f[5, 0] = (1.0 / I) * (-sp.sin(gimbal) * T * r_T)

    H_funcs: List[Callable] = []
    for i in range(f.shape[0]):
        H_i = sp.hessian(f[i, 0], z)
        H_funcs.append(sp.lambdify((x, u), H_i, "numpy"))
    
    return x, u, f, H_funcs