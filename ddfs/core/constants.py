# ddfs/core/constants.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np  # pyright: ignore[reportMissingImports]
import yaml  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

from ddfs.cons_calc import MismatchCalculator, NominalIncrementBoundCalculator
from ddfs.cons_calc.smoothness import LipschitzJacobianCalculator, convert_S_tube_to_nondim
from ddfs.io.load_nominal import load_nominal
from models.rocket2d import Rocket2D
from models.rocket6dof import Rocket6DoF


def _build_twin_model(model_name: str, r_scale: float, m_scale: float):
    if model_name == "rocket2d":
        twin = Rocket2D()
        twin.r_scale = float(r_scale)
        twin.m_scale = float(m_scale)
        twin.nondimensionalize()
        f_twin, _, _ = twin.get_equations()
        quat_slice = None
    elif model_name == "rocket6dof":
        twin = Rocket6DoF()
        twin.r_scale = float(r_scale)
        twin.m_scale = float(m_scale)
        twin.nondimensionalize()
        f_twin, _, _ = twin.get_equations()
        quat_slice = slice(7, 11)
    else:
        raise ValueError(f"Unknown model name for twin: {model_name}")
    return f_twin, quat_slice


def _derive_nominal_dir_from_inherits(inherits_nominal: str) -> Path:
    p = Path(inherits_nominal).resolve()
    name = p.stem.lower()
    if "2d" in name:
        model_name = "rocket2d"
    elif "6dof" in name:
        model_name = "rocket6dof"
    else:
        raise ValueError(f"Could not infer model name from config path: {inherits_nominal}")
    return p.parent.parent.parent / "nominal_trajectories" / model_name


def _apply_2d_constants_physical(phys: Rocket2D, cfg: Dict[str, Any]) -> None:
    consts = (cfg.get("model") or {}).get("constants") or {}
    # All are PHYSICAL units here; scaling happens in phys.nondimensionalize()
    for key in ["m", "I", "g", "r_T", "T_min", "T_max"]:
        if key in consts:
            setattr(phys, key, float(consts[key]))
    lims = consts  # same dict
    if "max_gimbal_deg" in lims:
        phys.max_gimbal = np.deg2rad(float(lims["max_gimbal_deg"]))
    if "theta_max_deg" in lims:
        phys.t_max = np.deg2rad(float(lims["theta_max_deg"]))
    if "w_max_deg" in lims:
        phys.w_max = np.deg2rad(float(lims["w_max_deg"]))


def _apply_6dof_constants_physical(phys: Rocket6DoF, cfg: Dict[str, Any]) -> None:  # noqa: C901
    msec = (cfg.get("model") or {}).get("mass") or {}
    if "m_wet" in msec:
        phys.m_wet = float(msec["m_wet"])
    if "m_dry" in msec:
        phys.m_dry = float(msec["m_dry"])

    isec = (cfg.get("model") or {}).get("inertia") or {}
    if "J_B_diag" in isec:
        phys.J_B = np.diag(np.array(isec["J_B_diag"], dtype=float))

    esec = (cfg.get("model") or {}).get("environment") or {}
    if "g_I" in esec:
        phys.g_I = np.array(esec["g_I"], dtype=float)
    if "alpha_m" in esec:
        phys.alpha_m = float(esec["alpha_m"])
    if "r_T_B" in esec:
        phys.r_T_B = np.array(esec["r_T_B"], dtype=float)

    tsec = (cfg.get("model") or {}).get("thrust") or {}
    if "T_max" in tsec:
        phys.T_max = float(tsec["T_max"])
    if "T_min" in tsec:
        phys.T_min = float(tsec["T_min"])
    if "max_gimbal_deg" in tsec:
        phys.max_gimbal_deg = float(tsec["max_gimbal_deg"])

    asec = (cfg.get("model") or {}).get("attitude_limits") or {}
    if "max_tilt_deg" in asec:
        phys.max_angle_deg = float(asec["max_tilt_deg"])
    if "max_body_rate_deg" in asec:
        phys.max_body_rate_deg = float(asec["max_body_rate_deg"])

    gsec = (cfg.get("model") or {}).get("glide_slope") or {}
    if "angle_deg" in gsec:
        phys.glide_slope_deg = float(gsec["angle_deg"])


def _build_phys_model(cfg: Dict[str, Any], r_scale: float, m_scale: float):
    name = (cfg.get("model") or {}).get("name")
    if name == "rocket2d":
        phys = Rocket2D()
        phys.r_scale = float(r_scale)
        phys.m_scale = float(m_scale)
        _apply_2d_constants_physical(phys, cfg)  # << set PHYSICAL constants first
        phys.nondimensionalize()  # << then scale once
        f_phys, _, _ = phys.get_equations()
        quat_slice = None
    elif name == "rocket6dof":
        phys = Rocket6DoF()
        phys.r_scale = float(r_scale)
        phys.m_scale = float(m_scale)
        _apply_6dof_constants_physical(phys, cfg)  # << set PHYSICAL constants first
        phys.nondimensionalize()  # << then scale once
        f_phys, _, _ = phys.get_equations()
        quat_slice = slice(7, 11)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return phys, f_phys, quat_slice, name


def main():
    ap = argparse.ArgumentParser(description="Compute Δ, gamma along (resampled) nominal and write constants.yaml")
    ap.add_argument("plant_yaml", help="Path to ddfs plant config YAML (with meta.inherits_nominal)")
    ap.add_argument("--nominal-dir", default=None, help="Dir with X.npy, U.npy, metadata.json")
    ap.add_argument("--out", default=None, help="Output YAML (default: ddfs/configs/constants.yaml)")
    args = ap.parse_args()

    plant_cfg = yaml.safe_load(Path(args.plant_yaml).read_text())
    inherits = ((plant_cfg.get("meta") or {}).get("inherits_nominal")) or ""
    nom_dir = Path(args.nominal_dir) if args.nominal_dir else _derive_nominal_dir_from_inherits(inherits)

    if not (nom_dir / "metadata.json").exists():
        raise FileNotFoundError(f"metadata.json not found under {nom_dir}")

    model_name = (plant_cfg.get("model") or {}).get("name")
    dt_override = (plant_cfg.get("execution") or {}).get("dt", None)
    dt_override = float(dt_override) if dt_override is not None else None

    # Load + (optionally) resample nominal to dt_override
    X_nom, U_nom, dt_eff, K_new, r_scale, m_scale = load_nominal(
        nom_dir=nom_dir, model_name=model_name, dt_override=dt_override
    )

    # Diagnostic prints (one run only)
    if X_nom.shape[0] >= 7:
        print(
            f"[units] max|r|={np.abs(X_nom[1:4]).max():.3g}  "
            f"max|v|={np.abs(X_nom[4:7]).max():.3g}"
        )
    else:
        print(f"[units] max|rx,ry,vx,vy|={np.abs(X_nom[0:4]).max():.3g}")
    print(f"[units] max‖u‖={np.linalg.norm(U_nom, axis=0).max():.3g}")
    if model_name == "rocket6dof":
        q_norms = np.linalg.norm(X_nom[7:11], axis=0)
        print(f"[q] mean|‖q‖-1|={(np.abs(q_norms - 1)).mean():.3g}")

    # Build plant model for Δ/gamma
    phys, f_phys, quat_slice_phys, model_name = _build_phys_model(plant_cfg, r_scale, m_scale)

    # --- Δ and gamma ---
    gamma_calc = MismatchCalculator(f_phys, dt=dt_eff, quat_slice=quat_slice_phys, norm="l2")
    gamma_res = gamma_calc.compute(X_nom, U_nom)
    gamma = gamma_res.gamma
    norms = gamma_res.norms

    # Nominal increment bound v
    quat_slice_twin = quat_slice_phys  # same quat slice for both
    v_calc = NominalIncrementBoundCalculator(dt=dt_eff, quat_slice=quat_slice_twin, norm="l2")
    v_res = v_calc.from_increments(X_nom, U_nom)
    v = v_res.v_max

    # Optionally compute rate-based diagnostic (not persisted unless you want to)
    f_twin, _ = _build_twin_model(model_name, r_scale, m_scale)
    v_rate_res = v_calc.from_rates(X_nom, U_nom, f_nom=f_twin, u_dot=None)
    v_rate = v_rate_res.rate_sup

    # --- L_J: Lipschitz bound on Jacobian via Hessian aggregation ---
    # Convert the S_tube (physical) to non-dimensional, if provided; otherwise use a tiny fallback tube.
    S_tube_phys = ((plant_cfg.get("assumptions") or {}).get("S_tube"))
    S_tube_nd = None
    if S_tube_phys is not None:
        try:
            S_tube_nd = convert_S_tube_to_nondim(model_name, S_tube_phys, r_scale, m_scale)
        except (ValueError, TypeError, KeyError):
            # Fall back to default if conversion fails (e.g., expression strings in YAML)
            S_tube_nd = None
    
    if S_tube_nd is None:
        if model_name == "rocket2d":
            S_tube_nd = {
                "state": {
                    "dr_max_nd": [1e-2, 1e-2],
                    "dv_max_nd": [1e-2, 1e-2],
                    "dtheta_max_rad": 1e-2,
                    "domega_max_radps": 1e-2,
                },
                "input": {
                    "dT_max_nd": 1e-2,
                    "dgimbal_max_rad": 1e-2,
                },
            }
        else:  # rocket6dof
            S_tube_nd = {
                "state": {
                    "dr_max_nd": [1e-2, 1e-2, 1e-2],
                    "dv_max_nd": [1e-2, 1e-2, 1e-2],
                    "deuler_max_rad": [1e-2, 1e-2, 1e-2],
                    "domega_max_radps": [1e-2, 1e-2, 1e-2],
                    "dm_nd": 0.0,
                },
                "input": {
                    "dT_max_nd": 1e-2,
                },
            }

    # Build the non-dimensional parameter dict for the symbolic builders
    if model_name == "rocket2d":
        params_nd = {
            "m": float(phys.m),
            "I": float(phys.I),
            "g": float(phys.g),
            "r_T": float(phys.r_T),
        }
    else:  # rocket6dof
        params_nd = {
            "g_I": np.asarray(phys.g_I, dtype=float).tolist(),
            "r_T_B": np.asarray(phys.r_T_B, dtype=float).tolist(),
            "J_B": np.asarray(phys.J_B, dtype=float).tolist(),
            "alpha_m": float(phys.alpha_m),
        }

    lj_calc = LipschitzJacobianCalculator(model_name, params_nd, quat_slice=quat_slice_phys)
    lj_diag = lj_calc.via_hessians(X_nom, U_nom, S_tube_nd, n_times=32, n_points_per_time=4)
    L_J = float(lj_diag.L_J)

    # Merge into constants.yaml under the model key
    out_item = {
        "model": model_name,
        "gamma": float(gamma),
        "gamma_norm": "l2",
        "K": int(K_new),
        "dt": float(dt_eff),
        "nominal_dir": str(nom_dir),
        "v": float(v),
        "v_p95": float(v_res.v_p95),
        "L_J": float(L_J),
        "L_J_method": lj_diag.method,
        "L_J_samples": int(lj_diag.samples_evaluated),
        # "v_rate": float(v_rate) if v_rate is not None else None,  # uncomment if desired
    }
    out_path = Path(args.out) if args.out else Path("ddfs/configs/constants.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = yaml.safe_load(out_path.read_text()) if out_path.exists() else {}
    if not isinstance(existing, dict):
        existing = {}
    existing[model_name] = out_item
    out_path.write_text(yaml.safe_dump(existing, sort_keys=False))

    print(f"[constants] wrote {out_path}")
    print(f"  model: {model_name}")
    print(f"  dt: {dt_eff:.6g}  (K={K_new})")
    print(f"  gamma (max ||Δ||₂): {gamma:.6g}")
    print(f"  median ||Δ||: {float(np.median(norms)):.6g}, 95th: {float(np.percentile(norms, 95)):.6g}")
    print(f"  v (max nominal increment): {v:.6g}  (p95: {v_res.v_p95:.6g})")
    if v_rate is not None:
        print(f"  v_rate (Δt·max ||[ẋ;u̇]||₂): {v_rate:.6g}")
    print(f"  L_J (Jacobian Lipschitz bound): {L_J:.6g}  via {lj_diag.method}  (samples: {lj_diag.samples_evaluated})")


if __name__ == "__main__":
    main()
