# ddfs/sdp/solve.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import cvxpy as cp  # type: ignore

from ddfs.sdp.build_problem import build_problem


@dataclass(frozen=True)
class SolveResult:
    status: str
    solver_used: str
    obj_value: Optional[float]
    P: Optional[np.ndarray]
    L: Optional[np.ndarray]
    K: Optional[np.ndarray]
    lambda1: Optional[float]
    lambda2: Optional[float]
    nu: Optional[float]
    info: Dict[str, float]


def _pick_solver(preferred: Optional[str] = None) -> str:
    installed = set(cp.installed_solvers())
    if preferred and preferred in installed:
        return preferred
    if "MOSEK" in installed:
        return "MOSEK"
    if "SCS" in installed:
        return "SCS"
    raise RuntimeError(
        f"No supported solver available. Installed: {sorted(installed)}. "
        "Please install MOSEK or SCS."
    )


def _recover_K(L: np.ndarray, P: np.ndarray) -> np.ndarray:
    # Symmetrize P for numerical stability
    P = 0.5 * (P + P.T)
    n = P.shape[0]
    # Try a stable solve for P^{-1}
    try:
        invP = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse (regularized)
        invP = np.linalg.pinv(P, rcond=1e-9)
    K = L @ invP
    # Clean tiny imaginary or noise
    K = np.real_if_close(K, tol=1e-12)
    return K


def solve_funnel_sdp(
    *,
    n: int,
    m: int,
    alpha: float,
    P_min_i: np.ndarray,
    R_max_i: np.ndarray,
    N1: np.ndarray,
    N2: np.ndarray,
    preferred_solver: Optional[str] = None,
    solver_opts: Optional[Dict] = None,
    eps_psd: float = 1e-9,
    verbose: bool = False,
) -> SolveResult:
    """
    Build and solve the quadratic-funnel SDP:
        maximize   logdet(P)
        s.t.       S(P,L,nu) - λ1*~N1 - λ2*~N2 ⪰ 0
                   P ⪰ P_min_i
                   [[R_max_i, L], [Lᵀ, P]] ⪰ 0

    Returns a SolveResult with (P, L, K, λ1, λ2, ν) and metadata.
    """
    prob, V = build_problem(
        n=n, m=m, alpha=alpha,
        P_min_i=P_min_i, R_max_i=R_max_i,
        N1=N1, N2=N2, eps_psd=eps_psd,
    )

    solver = _pick_solver(preferred_solver)
    opts = dict(solver_opts or {})
    # Reasonable defaults if not provided
    if solver == "MOSEK":
        opts.setdefault("verbose", verbose)
        # You can add mosek_params here if you like:
        # opts.setdefault("mosek_params", {"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8})
    elif solver == "SCS":
        opts.setdefault("eps", 1e-6)
        opts.setdefault("max_iters", 50_000)
        opts.setdefault("verbose", verbose)

    try:
        prob.solve(solver=solver, **opts)
    except Exception as e:
        # Fallback: try SCS if MOSEK failed or vice-versa
        alt = "SCS" if solver != "SCS" and "SCS" in cp.installed_solvers() else None
        if alt is not None:
            if verbose:
                print(f"[solve] Primary solver {solver} failed ({e}). Trying fallback {alt}…")
            solver = alt
            alt_opts = {"eps": 1e-6, "max_iters": 50_000, "verbose": verbose}
            alt_opts.update(solver_opts or {})
            prob.solve(solver=solver, **alt_opts)
        else:
            raise

    status = prob.status
    obj_value = None if prob.value is None else float(prob.value)

    # Pull variable values (may be None if infeasible/unbounded)
    P = V["P"].value
    L = V["L"].value
    lam1 = V["lam1"].value
    lam2 = V["lam2"].value
    nu = V["nu"].value

    # Convert scalars safely
    lam1_f = None if lam1 is None else float(np.real_if_close(lam1))
    lam2_f = None if lam2 is None else float(np.real_if_close(lam2))
    nu_f = None if nu is None else float(np.real_if_close(nu))

    # Recover K if possible
    K = None
    if P is not None and L is not None:
        try:
            K = _recover_K(np.asarray(L, dtype=float), np.asarray(P, dtype=float))
        except Exception:
            K = None

    # Clean arrays
    def _clean(M):
        if M is None:
            return None
        M = np.asarray(M, dtype=float)
        M = np.real_if_close(M, tol=1e-12)
        return M

    P = _clean(P)
    L = _clean(L)
    K = _clean(K)

    info = {
        "primal_obj": obj_value if obj_value is not None else np.nan,
        "alpha": float(alpha),
        "eps_psd": float(eps_psd),
    }

    return SolveResult(
        status=status,
        solver_used=solver,
        obj_value=obj_value,
        P=P,
        L=L,
        K=K,
        lambda1=lam1_f,
        lambda2=lam2_f,
        nu=nu_f,
        info=info,
    )


# Optional: tiny CLI for smoke tests
if __name__ == "__main__":
    import json

    # Minimal synthetic demo (random PSD envelopes & blocks)
    n, m = 4, 2
    rng = np.random.default_rng(0)
    def rand_psd(d): 
        A = rng.standard_normal((d, d)); 
        return A @ A.T + 1e-2 * np.eye(d)

    P_min_i = rand_psd(n)
    R_max_i = rand_psd(m)

    dN = 2 * n + m
    N1 = rand_psd(dN)
    N2 = rand_psd(dN)

    res = solve_funnel_sdp(
        n=n, m=m, alpha=0.9,
        P_min_i=P_min_i, R_max_i=R_max_i,
        N1=N1, N2=N2,
        preferred_solver=None,  # auto-pick
        verbose=True,
    )
    print(json.dumps({
        "status": res.status,
        "solver": res.solver_used,
        "obj": res.obj_value,
        "have_P": res.P is not None,
        "have_K": res.K is not None,
    }, indent=2))