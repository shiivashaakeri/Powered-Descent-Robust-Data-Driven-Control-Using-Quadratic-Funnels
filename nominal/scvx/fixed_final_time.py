# nominal/scvx/fixed_final_time.py
from __future__ import annotations

from typing import List

import cvxpy as cvx  # pyright: ignore[reportMissingImports]

from .problem_base import ProblemBase


class SCProblem(ProblemBase):
    """
    SCVX subproblem for FIXED final time (sigma fixed).
    Dynamics:
        X_{k+1} = Ā_k X_k + B̄_k U_k + C̄_k U_{k+1} + z̄_k + nu_k
    Trust region (L1):
        ||X - X_last||_1 + ||U - U_last||_1 ≤ r_tr
    Objective:
        weight_nu * ||nu||_1 + (optional model objective)
    """

    def __init__(self, model, K: int):
        super().__init__(model, K)

        X = self.var["X"]
        U = self.var["U"]
        nu = self.var["nu"]

        A_bar = self.par["A_bar"]
        B_bar = self.par["B_bar"]
        C_bar = self.par["C_bar"]
        z_bar = self.par["z_bar"]

        X_last = self.par["X_last"]
        U_last = self.par["U_last"]

        weight_nu = self.par["weight_nu"]
        tr_radius = self.par["tr_radius"]

        constraints: List[cvx.Constraint] = []

        # Model constraints (bounds, boundary conditions, etc.)
        constraints += self.m.get_constraints(X, U, X_last, U_last)

        # Linear time-varying FOH dynamics
        constraints += [
            X[:, k + 1]
            == cvx.reshape(A_bar[:, k], (self.m.n_x, self.m.n_x)) @ X[:, k]
            + cvx.reshape(B_bar[:, k], (self.m.n_x, self.m.n_u)) @ U[:, k]
            + cvx.reshape(C_bar[:, k], (self.m.n_x, self.m.n_u)) @ U[:, k + 1]
            + z_bar[:, k]
            + nu[:, k]
            for k in range(self.K - 1)
        ]

        # Trust region
        dx = X - X_last
        du = U - U_last
        constraints += [cvx.norm(dx, 1) + cvx.norm(du, 1) <= tr_radius]

        # Objective (additive with model objective if any)
        sc_obj = weight_nu * cvx.norm(nu, 1)
        model_obj = self.m.get_objective(X, U, X_last, U_last)
        objective = cvx.Minimize(sc_obj) if model_obj is None else cvx.Minimize(sc_obj + model_obj.args[0])

        self.prob = cvx.Problem(objective, constraints)
