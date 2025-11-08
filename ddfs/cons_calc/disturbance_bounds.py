# ddfs/cons_calc/disturbance_bounds.py
from __future__ import annotations
from typing import Sequence, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import numpy as np  # pyright: ignore[reportMissingImports]

def compute_beta_i(
    H_i: np.ndarray,
    Xi_i: np.ndarray,
    k_indices: Sequence[int],
    k_i: int,
    C: float,
    L_J: float,
    gamma: float,
) -> float:
    """
    Implements β_i aggregation over the window:
      β_i = sum_k ( C |k - k_i| ||z(k)|| + gamma + 0.5 L_J ||z(k)||^2 )^2,
    where z(k) = [η(k); ξ(k)] with columns pulled from H_i, Xi_i.

    Shapes:
      H_i: (n, Lw), Xi_i: (m, Lw), len(k_indices) == Lw
    """
    assert H_i.shape[1] == Xi_i.shape[1] == len(k_indices), "Column/time mismatch"
    z = np.vstack([H_i, Xi_i])                    # (n+m, Lw)
    # Euclidean norms per time
    z_norm = np.linalg.norm(z, axis=0)            # (Lw,)
    # time offsets in *steps*
    offs = np.abs(np.asarray(k_indices, dtype=int) - int(k_i)).astype(float)  # (Lw,)
    terms = C * offs * z_norm + float(gamma) + 0.5 * float(L_J) * (z_norm**2)
    beta = float(np.sum(terms**2))
    # ensure nonnegative (guard tiny negatives from FP error)
    return max(beta, 0.0)

def empirical_beta_lower_bound(H_plus: np.ndarray, Z: np.ndarray, ridge: float = 1e-9) -> float:
    """
    Lower bound via residual of LS fit H_plus ≈ [A B] Z.
    Returns ||W_emp||_2^2. Uses Tikhonov ridge for stability if ZZ^T is ill-conditioned.
    """
    if H_plus.size == 0 or Z.size == 0:
        return 0.0
    ZZt = Z @ Z.T
    # solve H_plus Z^T (ZZ^T + λI)^{-1} instead of inv(ZZ^T)
    n = ZZt.shape[0]
    AhatBhat = H_plus @ Z.T @ np.linalg.pinv(ZZt + ridge * np.eye(n))
    W_emp = H_plus - AhatBhat @ Z
    if W_emp.size == 0:
        return 0.0
    smax = np.linalg.svd(W_emp, compute_uv=False)[0]
    return float(smax**2)

def inflate_beta_if_needed(
    beta_i: float, H_plus: np.ndarray, Z: np.ndarray, tau: float = 1.1, ridge: float = 1e-9
) -> float:
    """
    Ensure β_i ≥ τ * ||W_emp||_2^2 to avoid underestimation.
    """
    need = empirical_beta_lower_bound(H_plus, Z, ridge=ridge)
    return max(float(beta_i), float(tau * need))

def persist_beta_window(
    out_dir: Path | str,
    beta_i: float,
    k_indices: Sequence[int],
    k_i: int,
    C: float,
    L_J: float,
    gamma: float,
    z_norm: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save β_i and lightweight window summary alongside optional diagnostics.
    Layout:
      out_dir/beta.json
      out_dir/z_norm.npy   (optional)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "beta_i": float(beta_i),
        "k_i": int(k_i),
        "k_indices": [int(k) for k in k_indices],
        "C": float(C),
        "L_J": float(L_J),
        "gamma": float(gamma),
    }
    if extra_meta:
        meta.update(extra_meta)

    (out_dir / "beta.json").write_text(json.dumps(meta, indent=2))
    if z_norm is not None:
        np.save(out_dir / "z_norm.npy", np.asarray(z_norm, dtype=float))

    return out_dir

# Convenience: compute → optional inflate → persist (one call)
def compute_inflate_persist_beta(
    H_i: np.ndarray,
    Xi_i: np.ndarray,
    H_plus: Optional[np.ndarray],
    Z: Optional[np.ndarray],
    k_indices: Sequence[int],
    k_i: int,
    C: float,
    L_J: float,
    gamma: float,
    out_dir: Path | str,
    tau: float = 1.1,
    ridge: float = 1e-9,
) -> float:
    z = np.vstack([H_i, Xi_i])
    z_norm = np.linalg.norm(z, axis=0)
    beta = compute_beta_i(H_i, Xi_i, k_indices, k_i, C, L_J, gamma)
    if H_plus is not None and Z is not None:
        beta = inflate_beta_if_needed(beta, H_plus, Z, tau=tau, ridge=ridge)
    persist_beta_window(out_dir, beta, k_indices, k_i, C, L_J, gamma, z_norm=z_norm)
    return beta