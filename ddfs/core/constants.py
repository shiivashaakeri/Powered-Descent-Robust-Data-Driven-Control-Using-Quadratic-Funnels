# ddfs/core/constants.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np  # pyright: ignore[reportMissingImports]
import yaml  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

from ddfs.core.mismatch import deltas_and_gamma
from ddfs.io.load_nominal import load_nominal
from models.rocket2d import Rocket2D
from models.rocket6dof import Rocket6DoF


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

    # Build plant model in the same (non-dimensional) units as nominal
    _, f_phys, quat_slice, model_name = _build_phys_model(plant_cfg, r_scale, m_scale)

    # Compute deltas and gamma at chosen dt

    _, norms, gamma = deltas_and_gamma(f_phys, X_nom, U_nom, dt_eff, quat_slice)

    # Merge into constants.yaml under the model key
    out_item = {
        "model": model_name,
        "gamma": float(gamma),
        "gamma_norm": "l2",
        "K": int(K_new),
        "dt": float(dt_eff),
        "nominal_dir": str(nom_dir),
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
    if norms.size:
        q95 = float(np.percentile(norms, 95))
        med = float(np.median(norms))
        print(f"  median ||Δ||: {med:.6g}, 95th: {q95:.6g}")


if __name__ == "__main__":
    main()
