# ddfs/viz/timeline.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


@dataclass
class _SegMeta:
    seg_idx: int
    kD: int
    k_next: int
    L: int
    dt: float


# -------------------------
# Loading helpers
# -------------------------

def _load_manifest(run_root: Path) -> Dict:
    mf = run_root / "artifacts" / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"manifest.json not found under {mf}")
    return json.loads(mf.read_text())


def _iter_segments(run_root: Path) -> Iterable[int]:
    segs_dir = run_root / "segments"
    if not segs_dir.exists():
        return []
    for p in sorted(segs_dir.glob("seg_*/")):
        try:
            yield int(p.name.split("_")[-1])
        except Exception:
            continue


def _load_seg_meta(run_root: Path, seg_idx: int) -> _SegMeta:
    seg_dir = run_root / "segments" / f"seg_{seg_idx:04d}"
    meta_p = seg_dir / "segment_meta.json"
    if not meta_p.exists():
        raise FileNotFoundError(f"segment_meta.json missing for seg {seg_idx}")
    m = json.loads(meta_p.read_text())
    return _SegMeta(
        seg_idx=int(m["seg_idx"]),
        kD=int(m["kD"]),
        k_next=int(m["k_next"]),
        L=int(m["L"]),
        dt=float(m["dt"]),
    )


def _load_window_arrays(run_root: Path, seg_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    seg_dir = run_root / "segments" / f"seg_{seg_idx:04d}"
    # Prefer raw_window.npz (has extras). Fall back to individual npys if needed.
    npz_p = seg_dir / "raw_window.npz"
    if npz_p.exists():
        d = np.load(npz_p, allow_pickle=True)
        H = d["H"]
        H_plus = d["H_plus"]
        Xi = d["Xi"]
        extras: Dict[str, np.ndarray] = {}
        for k in d.files:
            if k in {"H", "H_plus", "Xi", "meta"}:
                continue
            extras[k] = d[k]
        return H, H_plus, Xi, extras
    # fallback
    H = np.load(seg_dir / "H.npy")
    H_plus = np.load(seg_dir / "H_plus.npy")
    Xi = np.load(seg_dir / "Xi.npy")
    return H, H_plus, Xi, {}


# -------------------------
# Core plots
# -------------------------

def plot_timeline(run_root: str | Path, *, show: bool = False, outfile: Optional[str | Path] = None) -> Path:
    """
    Visual timeline of segments and their data windows T_i^D.
    - Horizontal axis: time (s)
    - Bars: segments T_i
    - Highlighted spans: data windows T_i^D
    """
    run_root = Path(run_root)
    mani = _load_manifest(run_root)
    run_info = mani.get("run_info", {})
    dt = float(run_info.get("dt", 1.0))

    segs = list(_iter_segments(run_root))
    metas = [_load_seg_meta(run_root, s) for s in segs]
    if not metas:
        raise RuntimeError("No segments present to plot.")

    # Build figure
    fig, ax = plt.subplots(figsize=(12, 1.5 + 0.35 * len(metas)))

    ygap = 0.6
    for r, sm in enumerate(metas[::-1]):  # reverse so seg_0000 at top
        y = r * ygap
        # Full segment bar
        t0 = sm.kD - (sm.L - (sm.k_next - sm.kD))  # not used; just keep index idea handy
        k_start = sm.k_next - sm.L - (sm.kD - (sm.k_next - sm.L))  # but we don't store k_start; reconstruct from neighbors
        # Safer: compute using k_next and L for window; for T_i use [k_next - (segment length), k_next)
        k_end = sm.k_next
        k_start_guess = k_end - max(sm.L, 1)  # minimal placeholder if true T unknown
        # If previous meta exists, align contiguous T_i
        # Draw T_i as a light bar spanning [k_start_guess, k_end)
        ax.barh(y=y, width=(k_end - k_start_guess) * dt, left=k_start_guess * dt,
                height=0.28, color="#d9d9d9", edgecolor="none")
        # Data window span [kD, k_next)
        ax.barh(y=y, width=(sm.k_next - sm.kD) * dt, left=sm.kD * dt,
                height=0.28, color="#7db8ff", edgecolor="#2a6ee8")
        ax.text((sm.kD + 0.1) * dt, y, f"seg {sm.seg_idx}  D=[{sm.kD},{sm.k_next})", va="center", fontsize=8)

    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.set_title("Segment timeline with data windows $\\mathcal{T}_i^D$")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8)

    out = Path(outfile) if outfile else (run_root / "artifacts" / "viz_timeline.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_excitation_and_suspensions(
    run_root: str | Path,
    *,
    show: bool = False,
    outfile: Optional[str | Path] = None,
) -> Path:
    """
    Plot per-step excitation magnitude (||eps||) over data windows, and mark guard suspensions.
    Requires extras keys in raw_window.npz if present:
      - 'eps_applied' or per-column 'eps_norm' (fallback: use ||Xi|| as proxy)
      - 'suspended' per-column boolean (optional)
    """
    run_root = Path(run_root)

    times: List[float] = []
    eps_mag: List[float] = []
    susp: List[bool] = []

    for s in _iter_segments(run_root):
        meta = _load_seg_meta(run_root, s)
        H, Hp, Xi, extras = _load_window_arrays(run_root, s)
        L = meta.L
        t0 = meta.kD * meta.dt
        t = t0 + np.arange(L) * meta.dt

        # Try to get excitation magnitude
        if "eps_applied" in extras:
            E = np.asarray(extras["eps_applied"])  # shape (m,L) or (L,m)
            if E.ndim == 2 and E.shape[0] <= E.shape[1]:
                mag = np.linalg.norm(E, axis=0)
            else:
                mag = np.linalg.norm(E, axis=1)
        elif "eps_norm" in extras:
            mag = np.asarray(extras["eps_norm"]).reshape(-1)
        else:
            # Fallback: proxy via ||Xi|| (u deviation)
            mag = np.linalg.norm(Xi, axis=0)

        if "suspended" in extras:
            sus = np.asarray(extras["suspended"]).astype(bool).reshape(-1)
            if len(sus) != L:
                sus = np.resize(sus, L)
        else:
            sus = np.zeros(L, dtype=bool)

        times.append(t)
        eps_mag.append(mag)
        susp.append(sus)

    if not times:
        raise RuntimeError("No window arrays found.")

    T = np.concatenate(times)
    M = np.concatenate(eps_mag)
    S = np.concatenate(susp)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(T, M, linewidth=1.2)
    if np.any(S):
        ax.scatter(T[S], M[S], s=20, marker="x", color="red", label="suspended")
        ax.legend()
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\lVert\epsilon\rVert$")
    ax.set_title("Excitation magnitude over data windows (x = suspended)")
    ax.grid(True, linestyle=":", linewidth=0.8)

    out = Path(outfile) if outfile else (run_root / "artifacts" / "viz_excitation.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Timeline & excitation plots for a data-collection run.")
    ap.add_argument("run_root", type=str, help="Path to run root (…/data_runs/<run_id>)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_root)
    plot_timeline(root, show=args.show)
    plot_excitation_and_suspensions(root, show=args.show)

