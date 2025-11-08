#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np  # type: ignore
import yaml  # pyright: ignore[reportMissingModuleSource]
import cvxpy as cvx  # type: ignore

from ddfs.io.load_nominal import load_nominal
from models.rocket2d import Rocket2D
from models.rocket6dof import Rocket6DoF

try:
    from ddfs.viz.max_ellipsoids import (
        save_summary_plot,   # required
        save_timesweep_plots,  # optional
        default_pairs,         # optional
    )
except Exception:  # pragma: no cover
    from ddfs.viz.max_ellipsoids import save_summary_plot  # type: ignore
    save_timesweep_plots = None  # type: ignore
    default_pairs = None  # type: ignore


# ---------------- helpers: build halfspaces ----------------
def _add_abs_bound(A_list, b_list, dim: int, bound: float, n: int) -> None:
    a = np.zeros((n,), dtype=float)
    a[dim] = +1.0
    A_list.append(a.copy())
    b_list.append(float(bound))
    a[dim] = -1.0
    A_list.append(a.copy())
    b_list.append(float(bound))


def _stack(A_list: List[np.ndarray], b_list: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    if not A_list:
        return np.zeros((0, 0)), np.zeros((0,))
    return np.vstack(A_list), np.asarray(b_list, dtype=float)


# ---- 2D state/input halfspaces
def build_state_halfspaces_2d(model: Rocket2D, x_nom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = 6
    A_list: List[np.ndarray] = []
    b_list: List[float] = []
    _add_abs_bound(A_list, b_list, 4, float(model.t_max), n)  # |theta| <= t_max
    _add_abs_bound(A_list, b_list, 5, float(model.w_max), n)  # |omega| <= w_max
    a = np.zeros((n,))
    a[1] = -1.0
    A_list.append(a)
    b_list.append(0.0)  # ry >= 0
    return _stack(A_list, b_list)


def build_input_halfspaces_2d(model: Rocket2D, u_nom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = 2
    A_list: List[np.ndarray] = []
    b_list: List[float] = []
    _add_abs_bound(A_list, b_list, 0, float(model.max_gimbal), n)  # |gimbal| <= max
    a = np.zeros((n,))
    a[1] = +1.0
    A_list.append(a.copy())
    b_list.append(float(model.T_max))  # T <= T_max
    a[1] = -1.0
    A_list.append(a.copy())
    b_list.append(float(-model.T_min))  # -T <= -T_min  (i.e., T >= T_min)
    return _stack(A_list, b_list)


# ---- 6DoF helpers
def _linearize_soc_norm_cap(x0: np.ndarray, idxs, cap: float, n: int) -> Tuple[np.ndarray, float]:
    z0 = x0[list(idxs)].astype(float)
    nz = float(np.linalg.norm(z0))
    if nz < 1e-12:
        a = np.zeros((n,))
        a[idxs[0]] = 1.0
        return a, cap
    g = z0 / nz
    a = np.zeros((n,))
    for c, idx in zip(g, idxs):
        a[idx] = c
    h0 = nz - cap
    b = float(a @ x0 - h0)
    return a, b


def _linearize_glideslope(x_nom: np.ndarray, tan_gamma: float) -> Tuple[np.ndarray, float]:
    rE, rN, rU = x_nom[1], x_nom[2], x_nom[3]
    rn = np.array([rE, rN])
    nrm = float(np.linalg.norm(rn))
    grad_rn = (rn / nrm) if nrm > 1e-12 else np.array([1.0, 0.0])
    a = np.zeros((14,))
    a[1], a[2], a[3] = grad_rn[0], grad_rn[1], -1.0 / tan_gamma
    h0 = (nrm if nrm > 1e-12 else 0.0) - rU / tan_gamma
    b = float(a @ x_nom - h0)
    return a, b


def build_state_halfspaces_6dof(model: Rocket6DoF, x_nom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = 14
    A_list: List[np.ndarray] = []
    b_list: List[float] = []
    a = np.zeros((n,))
    a[0] = -1.0
    A_list.append(a.copy())
    b_list.append(float(-model.m_dry))  # m >= m_dry
    a_gs, b_gs = _linearize_glideslope(x_nom, model.tan_gamma_gs)
    A_list.append(a_gs)
    b_list.append(b_gs)
    tilt_cap = float(np.sqrt((1.0 - model.cos_theta_max) / 2.0))
    a_tilt, b_tilt = _linearize_soc_norm_cap(x_nom, [8, 9, 10], tilt_cap, n)
    A_list.append(a_tilt)
    b_list.append(b_tilt)
    a_w, b_w = _linearize_soc_norm_cap(x_nom, [11, 12, 13], float(model.w_B_max), n)
    A_list.append(a_w)
    b_list.append(b_w)
    return _stack(A_list, b_list)


def build_input_halfspaces_6dof(model: Rocket6DoF, u_nom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = 3
    A_list: List[np.ndarray] = []
    b_list: List[float] = []
    # gimbal cone (linearized)
    z0 = u_nom[[0, 1]]
    nz = float(np.linalg.norm(z0))
    if nz < 1e-12:
        a = np.zeros((n,))
        a[0] = 1.0
        a[2] = -float(model.tan_delta_max)
        A_list.append(a.copy())
        b_list.append(0.0)
    else:
        gxy = z0 / nz
        a = np.zeros((n,))
        a[0], a[1], a[2] = gxy[0], gxy[1], -float(model.tan_delta_max)
        h0 = nz - float(model.tan_delta_max) * u_nom[2]
        b = float(a @ u_nom - h0)
        A_list.append(a.copy())
        b_list.append(b)
    # thrust magnitude cap (linearized)
    nu = float(np.linalg.norm(u_nom))
    if nu < 1e-12:
        a = np.zeros((n,))
        a[2] = 1.0
        A_list.append(a.copy())
        b_list.append(float(model.T_max))
    else:
        gu = u_nom / nu
        a = gu.copy()
        h0 = nu - float(model.T_max)
        b = float(a @ u_nom - h0)
        A_list.append(a.copy())
        b_list.append(b)
    return _stack(A_list, b_list)


# ---------------- MVIE via log-det(P) ----------------
def maximum_volume_ellipsoid(
    A: np.ndarray,
    b: np.ndarray,
    center: np.ndarray,
    dim: int,
    cap: Optional[float] = None,
    diag_only: bool = False,
    solver: str = "SCS",
) -> np.ndarray:
    """
    Find P ⪰ 0 maximizing log det P subject to ∥A_i P^{1/2}∥_2 ≤ (b_i - a_i^T center).
    Returns SPD-ish matrix (with small jitter if infeasible/empty).
    """
    if A.size == 0:
        return 1e-9 * np.eye(dim)

    margins = b - A @ center
    keep = margins > 1e-10
    if not np.any(keep):
        return 1e-9 * np.eye(dim)
    A_ = A[keep]
    m_ = margins[keep]

    if diag_only:
        p = cvx.Variable(dim)  # diagonal entries
        cons = [p >= 0]
        for i in range(A_.shape[0]):
            ai = A_[i : i + 1].T  # (dim,1)
            cons.append(cvx.quad_form(ai, cvx.diag(p)) <= float(m_[i]) ** 2)
        if cap is not None and cap > 0:
            cons.append(cvx.diag(p) << (cap**2) * np.eye(dim))
        obj = cvx.Maximize(cvx.sum(cvx.log(p + 1e-12)))
        prob = cvx.Problem(obj, cons)
    else:
        P = cvx.Variable((dim, dim), PSD=True)
        cons = [cvx.quad_form(A_[i : i + 1].T, P) <= float(m_[i]) ** 2 for i in range(A_.shape[0])]
        if cap is not None and cap > 0:
            cons.append(P << (cap**2) * np.eye(dim))
        prob = cvx.Problem(cvx.Maximize(cvx.log_det(P)), cons)

    try:
        prob.solve(solver=getattr(cvx, solver), eps=1e-5, max_iters=5000, verbose=False)
    except Exception:
        # fallback attempt
        prob.solve(solver=cvx.SCS, eps=2e-5, max_iters=8000, verbose=False)

    if diag_only:
        if p.value is None:
            return 1e-9 * np.eye(dim)
        P_val = np.diag(np.maximum(np.asarray(p.value, dtype=float), 0.0))
    else:
        if prob.status not in ("optimal", "optimal_inaccurate") or prob.value is None:
            return 1e-9 * np.eye(dim)
        P_val = np.asarray(prob.variables()[0].value, dtype=float)

    # symmetrize + jitter for numerical safety
    P_val = 0.5 * (P_val + P_val.T)
    P_val = P_val + 1e-9 * np.eye(dim)
    return P_val


# ---------------- main ----------------
def _derive_model_from_inherits(plant_cfg: dict) -> str:
    inherits = ((plant_cfg.get("meta") or {}).get("inherits_nominal")) or ""
    if inherits:
        name = Path(inherits).stem.lower()
        if "2d" in name:
            return "rocket2d"
        if "6dof" in name:
            return "rocket6dof"
    # fallback: try explicit model.name
    name = (plant_cfg.get("model") or {}).get("name", "")
    name = str(name).lower()
    if name in ("rocket2d", "rocket6dof"):
        return name
    raise ValueError("Could not infer model (check meta.inherits_nominal or model.name).")


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-time MVIEs for state/input constraints; save legacy plots and cache P_time/R_time."
    )
    ap.add_argument("plant_yaml", help="DDFS plant config (uses execution.dt if present)")
    ap.add_argument("--nominal-dir", default=None, help="Optional override for nominal_trajectories/<model>")
    ap.add_argument("--run-root", required=True, help="Run root where artifacts/ellipsoids/... will be written")
    ap.add_argument("--diag", action="store_true", help="Use diagonal MVIEs (more conservative, faster)")
    ap.add_argument("--solver", default="SCS", help="CVXPY solver to use (default: SCS)")
    ap.add_argument("--cap-state", type=float, default=None, help="Optional spectral cap for state ellipsoids")
    ap.add_argument("--cap-input", type=float, default=None, help="Optional spectral cap for input ellipsoids")
    ap.add_argument("--no-legacy-plots", action="store_true", help="Skip PNG plots under nominal_dir/max_ellipsoids")
    args = ap.parse_args()

    plant_cfg = yaml.safe_load(Path(args.plant_yaml).read_text())
    model_name = _derive_model_from_inherits(plant_cfg)

    # dt override
    dt_override = (plant_cfg.get("execution") or {}).get("dt", None)
    dt_override = float(dt_override) if dt_override is not None else None

    # nominal dir (resamples if dt_override)
    nom_dir = (
        Path(args.nominal_dir)
        if args.nominal_dir
        else Path(__file__).resolve().parents[2] / "nominal_trajectories" / model_name
    )
    X, U, dt, K, r_scale, m_scale = load_nominal(
        nom_dir=nom_dir, model_name=model_name, dt_override=dt_override
    )

    # models (nondimensional for constraints)
    if model_name == "rocket2d":
        M = Rocket2D()
        M.nondimensionalize()
        build_Ax = lambda x: build_state_halfspaces_2d(M, x)
        build_Au = lambda u: build_input_halfspaces_2d(M, u)
        n_x, n_u = 6, 2
    else:
        M = Rocket6DoF()
        M.nondimensionalize()
        build_Ax = lambda x: build_state_halfspaces_6dof(M, x)
        build_Au = lambda u: build_input_halfspaces_6dof(M, u)
        n_x, n_u = 14, 3

    # compute per-time MVIEs
    Q = np.zeros((n_x, n_x, K))
    R = np.zeros((n_u, n_u, K))
    for k in range(K):
        Ax, bx = build_Ax(X[:, k])
        Au, bu = build_Au(U[:, k])
        Q[:, :, k] = maximum_volume_ellipsoid(
            Ax, bx, center=X[:, k], dim=n_x, cap=args.cap_state, diag_only=args.diag, solver=args.solver
        )
        R[:, :, k] = maximum_volume_ellipsoid(
            Au, bu, center=U[:, k], dim=n_u, cap=args.cap_input, diag_only=args.diag, solver=args.solver
        )

    # ---------- legacy outputs (npz + plots) ----------
    if not args.no_legacy_plots:
        out_dir = nom_dir / "max_ellipsoids"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "results_allsteps.npz"
        np.savez(npz_path, Q=Q, R=R, X=X, U=U, dt=dt, K=K, model=model_name)

        pairs_val = default_pairs(model_name) if callable(default_pairs) else (
            {"state": [(0, 1)], "input": [(0, 1)]}
            if model_name == "rocket2d"
            else {"state": [(1, 2)], "input": [(0, 2)]}
        )

        meta = {
            "plant_yaml": str(Path(args.plant_yaml).resolve()),
            "nominal_dir": str(nom_dir.resolve()),
            "output_dir": str(out_dir.resolve()),
            "model": model_name,
            "dt": float(dt),
            "K": int(K),
            "pairs": pairs_val,
            "diag": bool(args.diag),
            "cap_state": args.cap_state,
            "cap_input": args.cap_input,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[mvie] saved {npz_path} and meta.json")

        summary_png = out_dir / "summary.png"
        save_summary_plot(
            out_png=summary_png,
            model_name=model_name,
            X=X,
            U=U,
            Q=Q,
            R=R,
            build_Ax_fn=build_Ax,
            build_Au_fn=build_Au,
            steps=None,
            title=f"{model_name} • MVIE vs linearized feasible sets",
        )
        print(f"[viz] saved {summary_png}")

        if callable(save_timesweep_plots):
            sweep_pngs = save_timesweep_plots(
                out_dir=out_dir,
                model_name=model_name,
                X=X,
                U=U,
                Q=Q,
                R=R,
                build_Ax_fn=build_Ax,
                build_Au_fn=build_Au,
                state_pairs=None,
                input_pairs=None,
                step_stride=1,
            )
            print(f"[viz] saved {len(sweep_pngs)} time-sweep plots in {out_dir}")

    # ---------- standardized cache for synthesizer ----------
    # Expect shapes: P_time: (K, n_x, n_x), R_time: (K, n_u, n_u)
    P_time = np.transpose(Q, (2, 0, 1))
    R_time = np.transpose(R, (2, 0, 1))

    # Symmetrize + jitter
    P_time = 0.5 * (P_time + np.transpose(P_time, (0, 2, 1))) + 1e-9 * np.eye(n_x)[None, :, :]
    R_time = 0.5 * (R_time + np.transpose(R_time, (0, 2, 1))) + 1e-9 * np.eye(n_u)[None, :, :]

    run_root = Path(args.run_root).resolve()
    cache_base = run_root / "artifacts" / "ellipsoids" / model_name
    (cache_base / "state").mkdir(parents=True, exist_ok=True)
    (cache_base / "input").mkdir(parents=True, exist_ok=True)

    np.save(cache_base / "state" / "P_time.npy", P_time)
    np.save(cache_base / "input" / "R_time.npy", R_time)

    cache_meta = {
        "model": model_name,
        "dt": float(dt),
        "K": int(K),
        "n_x": int(n_x),
        "n_u": int(n_u),
        "diag": bool(args.diag),
        "cap_state": args.cap_state,
        "cap_input": args.cap_input,
        "plant_yaml": str(Path(args.plant_yaml).resolve()),
        "nominal_dir": str(nom_dir.resolve()),
    }
    (cache_base / "meta.json").write_text(json.dumps(cache_meta, indent=2))

    print(f"[cache] wrote {cache_base}")
    print(f"  -> state envelopes: {cache_base/'state'/'P_time.npy'}  (shape {tuple(P_time.shape)})")
    print(f"  -> input  envelopes: {cache_base/'input'/'R_time.npy'}  (shape {tuple(R_time.shape)})")


if __name__ == "__main__":
    main()