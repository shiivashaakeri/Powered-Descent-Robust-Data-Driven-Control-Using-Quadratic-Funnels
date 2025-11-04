# nominal/solvers/cvxpy_solve.py
from __future__ import annotations

from typing import Optional

import cvxpy as cvx  # pyright: ignore[reportMissingImports]


def solve(
    prob: cvx.Problem,
    solver: str = "ECOS",
    verbose: bool = False,
    max_iters: Optional[int] = None,
    retries: int = 0,
) -> bool:
    """
    Return True on failure, False on success. Treats non-optimal statuses as failure.
    Supports ECOS and MOSEK, with optional retries on SolverError.
    """
    solver = (solver or "ECOS").upper()
    opts = {"verbose": verbose}

    if solver == "ECOS":
        if max_iters is not None:
            opts["max_iters"] = int(max_iters)
        solve_kwargs = dict(solver=cvx.ECOS, **opts)
    elif solver == "MOSEK":
        solve_kwargs = dict(solver=cvx.MOSEK, **opts)
    else:
        raise ValueError(f"Unsupported solver '{solver}'. Use 'ECOS' or 'MOSEK'.")

    attempt = 0
    while attempt <= max(0, retries):
        try:
            prob.solve(**solve_kwargs)
        except cvx.SolverError:
            attempt += 1
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            return False
        attempt += 1
    return True
