# ddfs/io/load_nominal.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]


def _infer_units_and_convert(
    X: np.ndarray, U: np.ndarray, meta: dict, model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Heuristic: if thrust scale is >> 10, treat as physical and convert to non-dimensional."""
    r_scale = float(meta["model"]["r_scale"])
    m_scale = float(meta["model"]["m_scale"])

    Xc, Uc = X.copy(), U.copy()

    if model_name == "rocket6dof":
        # u is thrust vector (3,)
        max_u = float(np.linalg.norm(Uc, axis=0).max())
        # physical thrusts ~ 1e5..1e6; nondim typically <~ 1
        if max_u > 1e3:
            Uc = Uc / (m_scale * r_scale)
        # states: mass, r, v, q, w -> nominal usually already nondim if produced by our SCVX;
        # but if positions look huge, scale them down.
        # If max |r| > ~ 50 (way bigger than a few r_scale multiples), assume physical.
        if float(np.abs(Xc[1:4]).max()) > 50.0:
            Xc[1:4] /= r_scale
            Xc[4:7] /= r_scale  # velocities
            Xc[0:1] /= m_scale  # mass

        # Normalize quaternion just in case
        q = Xc[7:11]
        n = np.linalg.norm(q, axis=0) + 1e-12
        Xc[7:11] = q / n
        return Xc, Uc

    if model_name == "rocket2d":
        # Use metadata to detect physical vs nondim thrust robustly
        try:
            Tmax_phys = float(meta["config"]["model"]["constants"]["T_max"])
        except Exception:
            Tmax_phys = None

        # If thrust channel looks physical (close to T_max in N), convert
        max_T = float(np.abs(Uc[1]).max())
        if Tmax_phys is not None:
            T_nd_expected = Tmax_phys / (m_scale * r_scale)  # ~0.4-0.5
            physical_u = max_T > 2.0 * T_nd_expected
        else:
            physical_u = max_T > 1.5  # conservative fallback

        if physical_u:
            Uc[1] = Uc[1] / (m_scale * r_scale)

        # State position/velocity look physical if magnitudes > ~1-2
        if (np.abs(Xc[0:2]).max() > 1.5) or (np.abs(Xc[2:4]).max() > 1.5):
            Xc[0:4] = Xc[0:4] / r_scale

        # Ensure gimbal is in radians (rare, but cheap safety)
        if np.abs(Uc[0]).max() > np.pi + 1e-3:
            Uc[0] = np.deg2rad(Uc[0])

        return Xc, Uc

    return Xc, Uc


def _slerp(q0: np.ndarray, q1: np.ndarray, tau: float) -> np.ndarray:
    """SLERP between two unit quaternions q0,q1 at fraction tau in [0,1]."""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = q0 + tau * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta0 = np.arccos(dot)
    s0 = np.sin((1 - tau) * theta0)
    s1 = np.sin(tau * theta0)
    s = np.sin(theta0) + 1e-12
    q = (s0 * q0 + s1 * q1) / s
    return q / (np.linalg.norm(q) + 1e-12)


def _resample(
    X: np.ndarray, U: np.ndarray, t_old: np.ndarray, t_new: np.ndarray, model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample onto t_new. Zero-order hold for U, linear for X except quaternion SLERP on 6DoF."""
    n_x, n_u = X.shape[0], U.shape[0]
    Xr = np.empty((n_x, len(t_new)), dtype=float)
    Ur = np.empty((n_u, len(t_new)), dtype=float)

    # Precompute piecewise-constant indices for U (ZOH)
    idx_u = np.searchsorted(t_old, t_new, side="right") - 1
    idx_u = np.clip(idx_u, 0, len(t_old) - 1)
    Ur[:] = U[:, idx_u]

    # Linear for X except quaternion
    quat_slice = slice(7, 11) if model_name == "rocket6dof" else None

    # For each state dim or block
    for k, tk in enumerate(t_new):
        i1 = np.searchsorted(t_old, tk, side="right") - 1
        i1 = int(np.clip(i1, 0, len(t_old) - 2))
        i2 = i1 + 1
        t1, t2 = t_old[i1], t_old[i2]
        tau = 0.0 if t2 == t1 else (tk - t1) / (t2 - t1)

        if quat_slice is None:
            Xr[:, k] = (1 - tau) * X[:, i1] + tau * X[:, i2]
        else:
            # copy all linear dims
            Xr[:, k] = (1 - tau) * X[:, i1] + tau * X[:, i2]
            # overwrite quaternion by SLERP
            q0 = X[quat_slice, i1]
            q1 = X[quat_slice, i2]
            Xr[quat_slice, k] = _slerp(q0 / (np.linalg.norm(q0) + 1e-12), q1 / (np.linalg.norm(q1) + 1e-12), float(tau))
    # Ensure quaternions are unit after resample
    if quat_slice is not None:
        q = Xr[quat_slice]
        n = np.linalg.norm(q, axis=0) + 1e-12
        Xr[quat_slice] = q / n
    return Xr, Ur


def load_nominal(
    nom_dir: Path, model_name: str, dt_override: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float, int, float, float]:
    X = np.load(nom_dir / "X.npy")
    U = np.load(nom_dir / "U.npy")
    if X.ndim == 3:
        X = X[-1]
    if U.ndim == 3:
        U = U[-1]

    meta = json.loads((nom_dir / "metadata.json").read_text())
    K = int(meta["discretization"]["K"])
    tf = float(meta["final_sigma"])
    t_old = np.linspace(0.0, tf, K)

    # Default: keep original grid
    if dt_override is None:
        t_new = t_old
    else:
        # Make sure we hit tf exactly (inclusive)
        K_new = int(np.floor(tf / dt_override + 1e-9)) + 1
        t_new = np.linspace(0.0, tf, K_new)

    # Resample if needed
    if len(t_new) != len(t_old) or np.max(np.abs(t_new - t_old)) > 1e-12:
        X, U = _resample(X, U, t_old, t_new, model_name)
        K = len(t_new)
        tf_eff = t_new[-1]
        dt_eff = tf_eff / max(K - 1, 1)
    else:
        dt_eff = t_old[1] - t_old[0] if len(t_old) > 1 else 0.0
        K = len(t_old)

    # Convert units to model's nondimensional convention if necessary
    X, U = _infer_units_and_convert(X, U, meta, model_name)

    # Final quaternion hygiene (6DoF)
    if model_name == "rocket6dof":
        q = X[7:11]
        n = np.linalg.norm(q, axis=0) + 1e-12
        X[7:11] = q / n

    r_scale = float(meta["model"]["r_scale"])
    m_scale = float(meta["model"]["m_scale"])
    return X, U, float(dt_eff), int(K), r_scale, m_scale
