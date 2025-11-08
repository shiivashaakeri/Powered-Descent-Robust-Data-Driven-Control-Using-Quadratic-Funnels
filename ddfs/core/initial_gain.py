# ddfs/core/initial_gain.py
import numpy as np  # pyright: ignore[reportMissingImports]
from ddfs.utils.la import dare  # you already have this

def lqr_gain_discrete(A, B, Q, R, xi_equals_K_eta=True):
    """
    Discrete-time LQR gain: solves DARE for P, returns K such that:
       u = -K_lqr x  (classic form)
    If your deviation law is xi = K eta (no minus), we flip the sign so:
       K = -K_lqr
    """
    P = dare(A, B, Q, R)
    K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return -K_lqr if xi_equals_K_eta else K_lqr

def finite_diff_jacobian(f_dt, x, u, eps=1e-6):
    """
    Central-difference Jacobian of *discrete* map f_dt(x,u) that returns x_{k+1}.
    """
    n = x.size
    m = u.size
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        dx = np.zeros_like(x); dx[i] = eps
        A[:, i] = (f_dt(x + dx, u) - f_dt(x - dx, u)) / (2 * eps)
    for j in range(m):
        du = np.zeros_like(u); du[j] = eps
        B[:, j] = (f_dt(x, u + du) - f_dt(x, u - du)) / (2 * eps)
    return A, B