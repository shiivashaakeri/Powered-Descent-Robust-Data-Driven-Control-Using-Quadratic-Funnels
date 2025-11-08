 # Block builders, Schur helpers, spectral norms, safe vstack
import numpy as np  # pyright: ignore[reportMissingImports]
try:
    from scipy.linalg import solve_discrete_are  # pyright: ignore[reportMissingImports]
except Exception:
    solve_discrete_are = None

def dare(A, B, Q, R, iters=500, eps=1e-9):
    """
    Solve the discrete-time Riccati equation A'PA - A'PB(R+B'PB)^-1 B'PA + Q = P.
    Returns P, K_lqr with u = -K_lqr x.
    Uses SciPy if available; otherwise a simple fixed-point iteration.
    """
    n = A.shape[0]
    if solve_discrete_are is not None:
        P = solve_discrete_are(A, B, Q, R)
    else:
        P = Q.copy()
        for _ in range(iters):
            RB = R + B.T @ P @ B
            K = np.linalg.solve(RB, B.T @ P @ A)
            Pn = A.T @ P @ A - A.T @ P @ B @ K + Q
            if np.linalg.norm(Pn - P, ord='fro') <= eps * np.linalg.norm(P, ord='fro'):
                P = Pn
                break
            P = Pn
    K_lqr = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)  # note: u = -K_lqr x
    return P, K_lqr