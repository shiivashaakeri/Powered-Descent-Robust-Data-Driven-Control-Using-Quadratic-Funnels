# models/sym_rocket6dof.py

from __future__ import annotations
from typing import Callable, List, Tuple
import sympy as sp # pyright: ignore[reportMissingImports]

from models.frames import dcm_from_quat_sym, euler_to_quat, omega_sym, skew_sym

def build_symbolics_rocket6dof(params_nd: dict) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, List[Callable]]:
    """
    Returns (x, u, f, H_funcs) in nondimensional coordinates.
    State x = [m, r_I(3), v_I(3), q_BI(4), w_B(3)]     -> (14,)
    Control u = T_B (body-frame thrust vector)         -> (3,)
    """

    # Symbols
    m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = sp.symbols("m rx ry rz vx vy vz q0 q1 q2 q3 wx wy wz", real=True)
    ux, uy, uz = sp.symbols("ux uy uz", real=True)

    x = sp.Matrix([m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]) # n_x = 14
    u = sp.Matrix([ux, uy, uz]) # n_u = 3
    z = sp.Matrix.vstack(x, u) # n_z = 17

    # Params
    g_I = sp.Matrix(params_nd["g_I"])
    r_T_B = sp.Matrix(params_nd["r_T_B"])
    J_B = sp.Matrix(params_nd["J_B"])
    alpha_m = sp.Float(params_nd["alpha_m"])

    C_BI = dcm_from_quat_sym(x[7:11, 0]) # body->inertial
    C_IB = C_BI.T # inertial->body

    f = sp.Matrix.zeros(14, 1)
    f[0, 0] = -alpha_m * u.norm()                      # mdot
    f[1:4, 0] = x[4:7, 0]                              # rdot = v
    f[4:7, 0] = (1.0 / x[0, 0]) * C_IB * u + g_I      # vdot
    f[7:11, 0] = sp.Rational(1, 2) * omega_sym(x[11:14, 0]) * x[7:11, 0]
    f[11:14, 0] = J_B**-1 * (skew_sym(r_T_B) * u) - skew_sym(x[11:14, 0]) * x[11:14, 0]

    H_funcs: List[Callable] = []
    for i in range(f.shape[0]):
        H_i = sp.hessian(f[i, 0], z)
        H_funcs.append(sp.lambdify((x, u), H_i, "numpy"))
    
    return x, u, f, H_funcs