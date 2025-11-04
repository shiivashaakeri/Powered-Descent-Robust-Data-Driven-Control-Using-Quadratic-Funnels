# nominal/scvx/problem_base.py
from __future__ import annotations

from typing import Dict, Optional

import cvxpy as cvx  # pyright: ignore[reportMissingImports]


class ProblemBase:
    """
    Common scaffolding for SCVX subproblems.

    Subclasses are responsible for:
      - adding model constraints
      - adding dynamics constraints
      - adding trust-region constraint
      - composing the objective
      - constructing `self.prob = cvx.Problem(objective, constraints)`
    """

    def __init__(self, model, K: int):
        self.m = model
        self.K = int(K)

        # Variables (common)
        self.var: Dict[str, cvx.Expression] = {
            "X": cvx.Variable((self.m.n_x, self.K)),  # state trajectory
            "U": cvx.Variable((self.m.n_u, self.K)),  # control trajectory
            "nu": cvx.Variable((self.m.n_x, self.K - 1)),  # virtual control
        }

        # Parameters (common)
        self.par: Dict[str, cvx.Expression] = {
            "A_bar": cvx.Parameter((self.m.n_x * self.m.n_x, self.K - 1)),
            "B_bar": cvx.Parameter((self.m.n_x * self.m.n_u, self.K - 1)),
            "C_bar": cvx.Parameter((self.m.n_x * self.m.n_u, self.K - 1)),
            "z_bar": cvx.Parameter((self.m.n_x, self.K - 1)),
            "X_last": cvx.Parameter((self.m.n_x, self.K)),
            "U_last": cvx.Parameter((self.m.n_u, self.K)),
            "weight_nu": cvx.Parameter(nonneg=True),
            "tr_radius": cvx.Parameter(nonneg=True),
        }

        self.prob: Optional[cvx.Problem] = None

    # ---------- Utilities ----------
    def set_parameters(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k not in self.par:
                # allow subclasses to extend; ignore unknown keys silently here
                continue
            self.par[k].value = v

    def print_available_parameters(self) -> None:
        print("Parameter names:")
        for k in sorted(self.par.keys()):
            print(f"  - {k}")

    def print_available_variables(self) -> None:
        print("Variable names:")
        for k in sorted(self.var.keys()):
            print(f"  - {k}")

    def get_variable(self, name: str):
        if name not in self.var:
            raise KeyError(f"Variable '{name}' does not exist.")
        return self.var[name].value

    def solve(self, **kwargs) -> bool:
        """
        Returns: error flag (False on success, True on solver error)
        """
        if self.prob is None:
            raise RuntimeError("Problem not constructed. Instantiate subclass which builds cvx.Problem.")
        try:
            self.prob.solve(**kwargs)
            return False
        except cvx.SolverError:
            return True
