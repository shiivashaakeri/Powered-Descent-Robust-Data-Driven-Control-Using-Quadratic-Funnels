#!/usr/bin/env python3
from __future__ import annotations
"""
Data collection orchestrator

Orchestrates: load config → load nominal → load Q/R (MVIE) → build deviation model →
segment timeline → per-window loop: (measure) → propose excitation → feasibility guard →
log (H, H_plus, Xi) → write datasets (per segment + manifest/summary/aggregate).

Supports two measurement modes:
  1) --meas-X/--meas-U: replay measured arrays (shape X:(n,Kx), U:(m,Ku))
  2) default "nominal proxy": uses nominal X,U as measurements (dry-run)

K-schedule options:
  - --K-schedule FILE: npy/npz containing either (m,n) for a constant gain or (S,m,n)
    where S = #segments; we'll map seg i→K[i].
  - Otherwise, defaults to zeros((m,n)).

Notes on sizes:
  - We assume X_nom and U_nom have length K.
  - H_i^+ needs η(k_next), i.e., η at the first index AFTER the window. If k_next == K
    and your measurement doesn't include X[:, K], we duplicate the last column (best-effort)
    while warning.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np  # type: ignore
import yaml  # type: ignore

from ddfs.io.load_nominal import load_nominal
from ddfs.io.paths import RunPaths
from ddfs.io.datasets import RunInfo, DatasetWriter

from ddfs.core.segment_manager import SegmentTimeline, SegmentSpec
from ddfs.core.deviation_model import build_deviation_model
from ddfs.core.data_logger import DataWindowLogger, WindowSpec
from ddfs.core.excitation import ExcitationPolicy, ExcitationConfig
from ddfs.core.feasibility_guard import (
    FeasibilityGuard,
    FeasibilityConfig,
    make_guard_rocket2d,
    make_guard_rocket6dof,
)

from models.rocket2d import Rocket2D
from models.rocket6dof import Rocket6DoF

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _infer_model_from_yaml(plant_yaml: Path) -> str:
    cfg = yaml.safe_load(Path(plant_yaml).read_text())
    name = (cfg.get("model") or {}).get("name")
    if name in {"rocket2d", "rocket6dof"}:
        return str(name)
    # Fallback: try meta.inherits_nominal
    inherits = ((cfg.get("meta") or {}).get("inherits_nominal")) or ""
    low = Path(inherits).stem.lower()
    if "6dof" in low:
        return "rocket6dof"
    if "2d" in low:
        return "rocket2d"
    raise ValueError("Could not infer model name from plant YAML.")


def _build_model_from_yaml(model_name: str, plant_yaml: Path, r_scale: float, m_scale: float):
    """Instantiate model and set physical constants (if present), then nondimensionalize."""
    cfg = yaml.safe_load(Path(plant_yaml).read_text())
    if model_name == "rocket2d":
        M = Rocket2D()
        M.r_scale = float(r_scale)
        M.m_scale = float(m_scale)
        consts = (cfg.get("model") or {}).get("constants") or {}
        # Physical constants (converted to nd in nondimensionalize)
        for key in ["m", "I", "g", "r_T", "T_min", "T_max"]:
            if key in consts:
                setattr(M, key, float(consts[key]))
        if "max_gimbal_deg" in consts:
            M.max_gimbal = np.deg2rad(float(consts["max_gimbal_deg"]))
        if "theta_max_deg" in consts:
            M.t_max = np.deg2rad(float(consts["theta_max_deg"]))
        if "w_max_deg" in consts:
            M.w_max = np.deg2rad(float(consts["w_max_deg"]))
        M.nondimensionalize()
        return M

    if model_name == "rocket6dof":
        M = Rocket6DoF()
        M.r_scale = float(r_scale)
        M.m_scale = float(m_scale)
        msec = (cfg.get("model") or {}).get("mass") or {}
        if "m_wet" in msec:
            M.m_wet = float(msec["m_wet"])
        if "m_dry" in msec:
            M.m_dry = float(msec["m_dry"])
        isec = (cfg.get("model") or {}).get("inertia") or {}
        if "J_B_diag" in isec:
            M.J_B = np.diag(np.array(isec["J_B_diag"], dtype=float))
        esec = (cfg.get("model") or {}).get("environment") or {}
        if "g_I" in esec:
            M.g_I = np.array(esec["g_I"], dtype=float)
        if "alpha_m" in esec:
            M.alpha_m = float(esec["alpha_m"])
        if "r_T_B" in esec:
            M.r_T_B = np.array(esec["r_T_B"], dtype=float)
        tsec = (cfg.get("model") or {}).get("thrust") or {}
        if "T_max" in tsec:
            M.T_max = float(tsec["T_max"])
        if "T_min" in tsec:
            M.T_min = float(tsec["T_min"])  # if present
        if "max_gimbal_deg" in tsec:
            M.max_gimbal_deg = float(tsec["max_gimbal_deg"])  # radians computed in nondim
        asec = (cfg.get("model") or {}).get("attitude_limits") or {}
        if "max_tilt_deg" in asec:
            M.max_angle_deg = float(asec["max_tilt_deg"])  # radians computed in nondim
        if "max_body_rate_deg" in asec:
            M.max_body_rate_deg = float(asec["max_body_rate_deg"])  # rad/s computed in nondim
        gsec = (cfg.get("model") or {}).get("glide_slope") or {}
        if "angle_deg" in gsec:
            M.glide_slope_deg = float(gsec["angle_deg"])  # used to form tan_gamma_gs in nondim
        M.nondimensionalize()
        return M

    raise ValueError(f"Unsupported model: {model_name}")


def _load_qr(npz_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not npz_path.exists():
        print(f"[collect] MVIE file not found at {npz_path} — proceeding without Q/R.")
        return None, None
    data = np.load(npz_path)
    Q = data.get("Q")
    R = data.get("R")
    if Q is None or R is None:
        print(f"[collect] MVIE file missing Q or R arrays — proceeding without Q/R.")
        return None, None
    return Q, R


def _prepare_guard(model_name: str, model, Q: Optional[np.ndarray], R: Optional[np.ndarray], cfg: FeasibilityConfig) -> FeasibilityGuard:
    if model_name == "rocket2d":
        return make_guard_rocket2d(model=model, Q_all=Q, R_all=R, cfg=cfg)
    return make_guard_rocket6dof(model=model, Q_all=Q, R_all=R, cfg=cfg)


def _load_K_schedule(path: Optional[Path], m: int, n: int, S: int) -> np.ndarray:
    if path is None:
        print("[collect] No K-schedule provided — using zeros.")
        return np.zeros((S, m, n), dtype=float)
    arr: Optional[np.ndarray] = None
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        if "K" in z:
            arr = z["K"]
    if arr is None:
        arr = np.load(p)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape == (m, n):
        K_sched = np.repeat(arr[None, :, :], S, axis=0)
        print(f"[collect] Loaded constant K schedule from {p}")
        return K_sched
    if arr.ndim == 3 and arr.shape[0] == S and arr.shape[1:] == (m, n):
        print(f"[collect] Loaded per-segment K schedule from {p}")
        return arr
    raise ValueError(f"K schedule shape mismatch: got {arr.shape}, expected (m,n) or (S,m,n)")


# ------------------------------------------------------------
# Measurement providers
# ------------------------------------------------------------
@dataclass
class MeasurementProvider:
    """Abstracts how we get x(k). In replay mode we ignore commanded u."""
    X: np.ndarray  # (n, Kx)

    def get_x(self, k: int) -> np.ndarray:
        if k < self.X.shape[1]:
            return self.X[:, k]
        # Graceful tail duplication
        return self.X[:, -1]


# ------------------------------------------------------------
# Main collection loop
# ------------------------------------------------------------

def collect(
    *,
    plant_yaml: Path,
    nominal_dir: Optional[Path],
    out_base: Path,
    run_id: Optional[str],
    T: int,
    L: int,
    excitation_cfg: ExcitationConfig,
    guard_cfg: FeasibilityConfig,
    K_schedule_path: Optional[Path],
    meas_X_path: Optional[Path],
    meas_U_path: Optional[Path],  # accepted but unused in replay (we compute xi via K,eps)
    dt_override: Optional[float],
    seed: Optional[int],
    aggregate: bool,
) -> str:
    rng = np.random.default_rng(seed)

    # --- load model name + nominal ---
    model_name = _infer_model_from_yaml(plant_yaml)
    nom_dir = nominal_dir or (Path("nominal_trajectories") / model_name)
    X_nom, U_nom, dt_eff, K_total, r_scale, m_scale = load_nominal(
        nom_dir=nom_dir, model_name=model_name, dt_override=dt_override
    )

    # --- instantiate model (nondimensional) for feasibility/halfspaces ---
    model = _build_model_from_yaml(model_name, plant_yaml, r_scale=r_scale, m_scale=m_scale)

    # --- MVIE Q/R ---
    qr_npz = nom_dir / "max_ellipsoids" / "results_allsteps.npz"
    Q_all, R_all = _load_qr(qr_npz)

    # --- deviation model (η, ξ) ---
    dev = build_deviation_model(model_name=model_name, X_nom=X_nom, U_nom=U_nom, quat_mode="tangent")
    n, m = dev.n, dev.m

    # --- segment timeline ---
    tl = SegmentTimeline(dt=float(dt_eff), T=int(T), L=int(L), K=int(K_total), strict_L=False)
    print(tl.summary())

    # --- K schedule ---
    K_sched = _load_K_schedule(K_schedule_path, m=m, n=n, S=len(tl))

    # --- excite/guard ---
    excite = ExcitationPolicy(m=m, cfg=excitation_cfg)
    guard = _prepare_guard(model_name, model, Q_all, R_all, guard_cfg)

    # --- measurements (replay or nominal proxy) ---
    if meas_X_path is not None:
        X_meas = np.asarray(np.load(meas_X_path), dtype=float)
        if X_meas.shape[0] != X_nom.shape[0]:
            raise ValueError(f"meas X shape {X_meas.shape} incompatible with nominal {X_nom.shape}")
    else:
        X_meas = X_nom.copy()
    meas = MeasurementProvider(X=X_meas)

    # --- output wiring ---
    run_id = run_id or time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    paths = RunPaths(base=out_base, model=model_name, run_id=run_id)
    run_info = RunInfo(
        run_id=run_id,
        model=model_name,
        n=n,
        m=m,
        dt=float(dt_eff),
        T=int(T),
        L=int(L),
        K_total=int(K_total),
        note="collect_data",
        extras={"plant_yaml": str(plant_yaml), "nominal_dir": str(nom_dir)},
    )
    writer = DatasetWriter(paths, run_info)

    # --- per segment: build window and log ---
    for seg in tl:  # type: SegmentSpec
        kD, k_next = seg.kD_start, seg.k_end_excl
        Lw = k_next - kD
        K_i = K_sched[seg.idx]

        logger = DataWindowLogger(n=n, m=m)
        spec = WindowSpec(
            seg_idx=seg.idx,
            kD=kD,
            k_next=k_next,
            L=Lw,
            dt=seg.dt,
            model=model_name,
            note=f"segment_{seg.idx}",
        )
        logger.begin(spec, K_i=K_i)

        for k in range(kD, k_next):
            # Measurements
            xk = meas.get_x(k)
            xk1 = meas.get_x(k + 1)  # may duplicate last if beyond available
            eta_k = dev.eta_k(xk, k)
            eta_kp1 = dev.eta_k(xk1, min(k + 1, X_nom.shape[1] - 1))

            # Base input and proposed excitation
            u_nom_k = U_nom[:, k]
            eps_prop = excite.next(
                k=k,
                u_nom=u_nom_k,
                model_name=model_name,
                model_params=(
                    {
                        "max_gimbal": float(getattr(model, "max_gimbal", 0.0)),
                        "T_min": float(getattr(model, "T_min", 0.0)),
                        "T_max": float(getattr(model, "T_max", 0.0)),
                    }
                    if model_name == "rocket2d"
                    else {
                        "tan_delta_max": float(getattr(model, "tan_delta_max", getattr(model, "tan_delta", 0.0))),
                        "T_max": float(getattr(model, "T_max", 0.0)),
                    }
                ),
                R_k=(R_all[:, :, k] if R_all is not None else None),
                rng=rng,
            )

            # Guard (state/input margins + MVIE budgets)
            gr = guard.guard(
                k=k,
                x_nom_k=X_nom[:, k],
                u_nom_k=u_nom_k,
                eta_k=eta_k,
                K_i=K_i,
                eps_proposed=eps_prop,
            )

            xi_k = K_i @ eta_k + gr.eps_clamped

            # Log triplet
            # Optionally, attach extras (margins, scales) per column for diagnostics
            extras = {
                "scale_hs": gr.scale_halfspaces,
                "scale_mvie": gr.scale_mvie,
                "min_state_margin": gr.min_state_margin,
                "min_input_margin": gr.min_input_margin,
                "mvie_state_ok": gr.mvie_state_ok,
                "mvie_input_budget": gr.mvie_input_budget,
                "suspended": gr.suspended,
                "reason": gr.reason,
            }
            tail = {"eta_tail": eta_kp1} if (k == k_next - 1) else None
            logger.log_transition(k=k, eta_k=eta_k, xi_k=xi_k, eta_kplus1=eta_kp1, extras=extras, extras_tail=tail)

        # Done with window → write
        H, H_plus, Xi, meta, extras = logger.finalize()
        writer.write_segment_window(seg.idx, H, H_plus, Xi, meta, extras, do_pe_check=True)

    # Finalize + optional aggregate
    extra_summary = {"seed": seed, "excitation": vars(excitation_cfg), "guard_cfg": vars(guard_cfg)}
    writer.finalize_run(write_summary=True, extra_summary=extra_summary)
    if aggregate:
        writer.save_aggregate()

    print(f"[collect] Finished run_id={run_id} at {paths.root}")
    return str(paths.root)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Collect deviation-data windows with excitation+guard.")
    ap.add_argument("plant_yaml", type=Path, help="DDFS plant config YAML (e.g., ddfs/configs/mars_2d_physical.yaml)")

    ap.add_argument("--nominal-dir", type=Path, default=None, help="Override nominal dir (X.npy,U.npy,metadata.json)")
    ap.add_argument("--out-base", type=Path, default=Path("."), help="Repo base for outputs (default: .)")
    ap.add_argument("--run-id", type=str, default=None, help="Run id (default: timestamp)")

    ap.add_argument("--T", type=int, default=40, help="Segment length in steps")
    ap.add_argument("--L", type=int, default=None, help="Data window length (default: n+m, clamped to T)")
    ap.add_argument("--dt-override", type=float, default=None, help="Resample nominal to this dt (seconds)")

    ap.add_argument("--K-schedule", type=Path, default=None, help="Path to (m,n) or (S,m,n) K gains (npy/npz)")
    ap.add_argument("--meas-X", type=Path, default=None, help="Optional measured X array to replay (n,Kx)")
    ap.add_argument("--meas-U", type=Path, default=None, help="Optional measured U array to replay (unused)")

    # Excitation
    ap.add_argument("--excitation-kind", type=str, choices=["prbs", "rademacher", "gaussian"], default="rademacher")
    ap.add_argument("--rho", type=float, default=0.05, help="Excitation radius (L2 or R-ellipsoid)")
    ap.add_argument("--hold", type=int, default=1, help="Piecewise-constant span for excitation")
    ap.add_argument("--exc-active-dims", type=int, nargs="*", default=None, help="List of input indices to excite")

    # Guard
    ap.add_argument("--margin-thresh-state", type=float, default=1e-6)
    ap.add_argument("--margin-thresh-input", type=float, default=1e-6)
    ap.add_argument("--safety-margin", type=float, default=5e-4)
    ap.add_argument("--max-scale", type=float, default=1.0)
    ap.add_argument("--use-mvie-state", action="store_true")
    ap.add_argument("--use-mvie-input", action="store_true")
    ap.add_argument("--mvie-shrink-state", type=float, default=1.0)
    ap.add_argument("--mvie-shrink-input", type=float, default=1.0)
    ap.add_argument("--suspend-on-low-state-margin", action="store_true")

    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true", help="Write aggregate.npz across segments")
    return ap


def main() -> None:
    ap = _build_argparser()
    args = ap.parse_args()

    # Build configs
    exc_cfg = ExcitationConfig(
        kind=args.excitation_kind,
        rho=float(args.rho),
        hold=int(args.hold),
        active_dims=args.exc_active_dims,
    )
    guard_cfg = FeasibilityConfig(
        margin_thresh_state=float(args.margin_thresh_state),
        margin_thresh_input=float(args.margin_thresh_input),
        safety_margin=float(args.safety_margin),
        max_scale=float(args.max_scale),
        use_mvie_state=bool(args.use_mvie_state),
        use_mvie_input=bool(args.use_mvie_input),
        mvie_shrink_state=float(args.mvie_shrink_state),
        mvie_shrink_input=float(args.mvie_shrink_input),
        suspend_on_low_state_margin=bool(args.suspend_on_low_state_margin),
    )

    # Decide L default after we know n+m.
    # To get n+m cheaply, peek nominal briefly.
    model_name = _infer_model_from_yaml(args.plant_yaml)
    nom_dir = args.nominal_dir or (Path("nominal_trajectories") / model_name)
    X_nom, U_nom, dt_eff, K_total, r_scale, m_scale = load_nominal(
        nom_dir=nom_dir, model_name=model_name, dt_override=args.dt_override
    )
    n = X_nom.shape[0] - 1 if model_name == "rocket6dof" else X_nom.shape[0]  # dev.n computed later, this is a safe upper bound
    m = U_nom.shape[0]
    L = int(args.L) if args.L is not None else int(min(args.T, n + m))

    collect(
        plant_yaml=args.plant_yaml,
        nominal_dir=args.nominal_dir,
        out_base=args.out_base,
        run_id=args.run_id,
        T=int(args.T),
        L=L,
        excitation_cfg=exc_cfg,
        guard_cfg=guard_cfg,
        K_schedule_path=args.K_schedule,
        meas_X_path=args.meas_X,
        meas_U_path=args.meas_U,
        dt_override=args.dt_override,
        seed=args.seed,
        aggregate=bool(args.aggregate),
    )


if __name__ == "__main__":
    main()
