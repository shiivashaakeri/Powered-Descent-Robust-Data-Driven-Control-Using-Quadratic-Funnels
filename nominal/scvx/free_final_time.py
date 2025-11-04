# nominal/scvx/free_final_time.py
from __future__ import annotations

from typing import List

import cvxpy as cvx  # pyright: ignore[reportMissingImports]

from .problem_base import ProblemBase


class SCProblem(ProblemBase):
    """
    SCVX subproblem for FREE final time.
    Normalized-time FOH dynamics:
        X_{k+1} = Ā_k X_k + B̄_k U_k + C̄_k U_{k+1} + S̄_k * sigma + z̄_k + nu_k
    Trust region (L1):
        ||X - X_last||_1 + ||U - U_last||_1 + |sigma - sigma_last| ≤ r_tr
    Objective:
        weight_sigma * sigma + weight_nu * ||nu||_1 + (optional model objective)
    """

    def __init__(self, model, K: int):
        super().__init__(model, K)

        # Extra variable and parameters for free-final-time
        self.var["sigma"] = cvx.Variable(nonneg=True)
        self.par["S_bar"] = cvx.Parameter((self.m.n_x, self.K - 1))
        self.par["sigma_last"] = cvx.Parameter(nonneg=True)
        self.par["weight_sigma"] = cvx.Parameter(nonneg=True)

        X = self.var["X"]
        U = self.var["U"]
        nu = self.var["nu"]
        sigma = self.var["sigma"]

        A_bar = self.par["A_bar"]
        B_bar = self.par["B_bar"]
        C_bar = self.par["C_bar"]
        S_bar = self.par["S_bar"]
        z_bar = self.par["z_bar"]

        X_last = self.par["X_last"]
        U_last = self.par["U_last"]
        sigma_last = self.par["sigma_last"]

        weight_nu = self.par["weight_nu"]
        weight_sigma = self.par["weight_sigma"]
        tr_radius = self.par["tr_radius"]

        constraints: List[cvx.Constraint] = []

        # Model constraints
        constraints += self.m.get_constraints(X, U, X_last, U_last)

        # Normalized-time FOH dynamics with sigma terms
        constraints += [
            X[:, k + 1]
            == cvx.reshape(A_bar[:, k], (self.m.n_x, self.m.n_x)) @ X[:, k]
            + cvx.reshape(B_bar[:, k], (self.m.n_x, self.m.n_u)) @ U[:, k]
            + cvx.reshape(C_bar[:, k], (self.m.n_x, self.m.n_u)) @ U[:, k + 1]
            + S_bar[:, k] * sigma
            + z_bar[:, k]
            + nu[:, k]
            for k in range(self.K - 1)
        ]

        # Trust region
        dx = X - X_last
        du = U - U_last
        ds = sigma - sigma_last
        constraints += [cvx.norm(dx, 1) + cvx.norm(du, 1) + cvx.norm(ds, 1) <= tr_radius]

        # Objective
        sc_obj = weight_sigma * sigma + weight_nu * cvx.norm(nu, 1)
        model_obj = self.m.get_objective(X, U, X_last, U_last)
        objective = cvx.Minimize(sc_obj) if model_obj is None else cvx.Minimize(sc_obj + model_obj.args[0])

        self.prob = cvx.Problem(objective, constraints)
