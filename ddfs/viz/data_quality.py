# ddfs/viz/data_quality.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


def _load_summary(run_root: Path) -> Dict:
    summ = run_root / "artifacts" / "summary.json"
    if not summ.exists():
        # try manifest and derive minimal info
        mf = run_root / "artifacts" / "manifest.json"
        if not mf.exists():
            raise FileNotFoundError("Neither summary.json nor manifest.json found.")
        man = json.loads(mf.read_text())
        return {"pe_checks": [], "run_info": man.get("run_info", {})}
    return json.loads(summ.read_text())


def _iter_segments(run_root: Path):
    segs_dir = run_root / "segments"
    for p in sorted(segs_dir.glob("seg_*/")):
        try:
            yield int(p.name.split("_")[-1])
        except Exception:
            continue


def _load_seg_meta(run_root: Path, seg_idx: int) -> Dict:
    p = run_root / "segments" / f"seg_{seg_idx:04d}" / "segment_meta.json"
    return json.loads(p.read_text())


def _load_window(run_root: Path, seg_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    seg_dir = run_root / "segments" / f"seg_{seg_idx:04d}"
    npz = seg_dir / "raw_window.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        H, H_plus, Xi = d["H"], d["H_plus"], d["Xi"]
        extras: Dict[str, np.ndarray] = {}
        for k in d.files:
            if k in {"H", "H_plus", "Xi", "meta"}:
                continue
            extras[k] = d[k]
        return H, H_plus, Xi, extras
    return np.load(seg_dir / "H.npy"), np.load(seg_dir / "H_plus.npy"), np.load(seg_dir / "Xi.npy"), {}


# -------------------------
# PE rank bars
# -------------------------

def plot_pe_ranks(run_root: str | Path, *, show: bool = False, outfile: Optional[str | Path] = None) -> Path:
    """Bar chart of rank([H; Xi]) vs required (n+m) per segment."""
    run_root = Path(run_root)
    summary = _load_summary(run_root)
    pe = summary.get("pe_checks", [])

    if not pe:  # fallback: recompute quickly from files
        vals = []
        for s in _iter_segments(run_root):
            H = np.load(run_root / "segments" / f"seg_{s:04d}" / "H.npy")
            Xi = np.load(run_root / "segments" / f"seg_{s:04d}" / "Xi.npy")
            rank = int(np.linalg.matrix_rank(np.vstack([H, Xi])))
            vals.append({"seg_idx": s, "rank_H_Xi": rank})
        pe = vals

    segs = [int(x["seg_idx"]) for x in pe]
    ranks = [int(x.get("rank_H_Xi", 0)) for x in pe]
    req = summary.get("run_info", {}).get("n", 0) + summary.get("run_info", {}).get("m", 0)

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(segs)), 3.2))
    ax.bar(segs, ranks, color="#7db8ff", edgecolor="#2a6ee8")
    ax.axhline(req, color="black", linewidth=1.2, linestyle="--", label=f"n+m={req}")
    ax.set_xlabel("segment i")
    ax.set_ylabel("rank [H; Xi]")
    ax.set_title("Persistence-of-Excitation rank per segment")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8)
    ax.legend()

    out = Path(outfile) if outfile else (run_root / "artifacts" / "viz_pe_ranks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# -------------------------
# Margin histograms
# -------------------------

def plot_margin_histograms(
    run_root: str | Path,
    *,
    show: bool = False,
    outfile: Optional[str | Path] = None,
) -> Path:
    """
    Aggregate and plot histograms of min state/input margins across all windows.
    Expects per-column extras keys (if present): 'min_state_margin', 'min_input_margin'.
    Gracefully skips if not available.
    """
    run_root = Path(run_root)
    ms: List[float] = []
    mu: List[float] = []

    for s in _iter_segments(run_root):
        _, _, _, extras = _load_window(run_root, s)
        if "min_state_margin" in extras:
            ms.extend(np.asarray(extras["min_state_margin"]).ravel().tolist())
        if "min_input_margin" in extras:
            mu.extend(np.asarray(extras["min_input_margin"]).ravel().tolist())

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    if ms:
        axs[0].hist(ms, bins=40, alpha=0.9)
    axs[0].set_title("Min state margin")
    axs[0].set_xlabel("b_x - A_x x")
    axs[0].set_ylabel("count")
    axs[0].grid(True, linestyle=":", linewidth=0.8)

    if mu:
        axs[1].hist(mu, bins=40, alpha=0.9)
    axs[1].set_title("Min input margin (pre-excitation)")
    axs[1].set_xlabel("b_u - A_u u_base")
    axs[1].grid(True, linestyle=":", linewidth=0.8)

    out = Path(outfile) if outfile else (run_root / "artifacts" / "viz_margin_hists.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# -------------------------
# \|eta\| / \|xi\| vs MVIE radii
# -------------------------

def plot_norms_vs_ellipsoids(
    run_root: str | Path,
    mvie_npz: str | Path,
    *,
    show: bool = False,
    outfile: Optional[str | Path] = None,
) -> Path:
    """
    For each step k in each window, compute
      r_eta(k) = sqrt(eta(k)^T Q(k) eta(k))
      r_xi(k)  = sqrt(xi(k)^T  R(k) xi(k))
    and plot time-series ratios ||eta||2 vs r_eta, ||xi||2 vs r_xi.
    Points with r<=1 lie within MVIE (for shrink=1.0).
    """
    run_root = Path(run_root)
    mvie_npz = Path(mvie_npz)

    if not mvie_npz.exists():
        raise FileNotFoundError(f"MVIE npz not found: {mvie_npz}")
    D = np.load(mvie_npz)
    Q = D["Q"]  # (n_x,n_x,K)
    R = D["R"]  # (n_u,n_u,K)
    K_total = int(D["K"]) if "K" in D.files else Q.shape[-1]
    dt = float(D["dt"]) if "dt" in D.files else 1.0

    t_all: List[float] = []
    eta_l2: List[float] = []
    xi_l2: List[float] = []
    r_eta: List[float] = []
    r_xi: List[float] = []

    for s in _iter_segments(run_root):
        meta = _load_seg_meta(run_root, s)
        H, Hp, Xi, _ = _load_window(run_root, s)
        # columns j in [0,L-1] map to k = kD + j
        ks = meta["kD"] + np.arange(meta["L"], dtype=int)
        ks = ks[ks < K_total]
        if ks.size == 0:
            continue
        L = ks.size

        # Align shapes
        E = H[:, :L]
        U = Xi[:, :L]

        for j, k in enumerate(ks):
            e = E[:, j]
            u = U[:, j]
            Qi = Q[:, :, k]
            Ri = R[:, :, k]
            eta_l2.append(float(np.linalg.norm(e)))
            xi_l2.append(float(np.linalg.norm(u)))
            r_eta.append(float(np.sqrt(max(0.0, e.T @ Qi @ e))))
            r_xi.append(float(np.sqrt(max(0.0, u.T @ Ri @ u))))
            t_all.append((k) * dt)

    if not t_all:
        raise RuntimeError("No overlapping k indices between windows and MVIE data.")

    T = np.asarray(t_all)
    eta_l2 = np.asarray(eta_l2)
    xi_l2 = np.asarray(xi_l2)
    r_eta = np.asarray(r_eta)
    r_xi = np.asarray(r_xi)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    axs[0].plot(T, eta_l2, label=r"$\lVert\eta\rVert_2$")
    axs[0].plot(T, r_eta, label=r"$\sqrt{\eta^T Q \eta}$")
    axs[0].set_ylabel("state deviation / radius")
    axs[0].set_title(r"State: $\lVert\eta\rVert_2$ vs $\sqrt{\eta^T Q \eta}$")
    axs[0].grid(True, linestyle=":", linewidth=0.8); axs[0].legend()

    axs[1].plot(T, xi_l2, label=r"$\lVert\xi\rVert_2$")
    axs[1].plot(T, r_xi, label=r"$\sqrt{\xi^T R \xi}$")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("input deviation / radius")
    axs[1].set_title(r"Input: $\lVert\xi\rVert_2$ vs $\sqrt{\xi^T R \xi}$")
    axs[1].grid(True, linestyle=":", linewidth=0.8); axs[1].legend()

    out = Path(outfile) if outfile else (run_root / "artifacts" / "viz_norms_vs_mvie.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Data quality visuals: PE ranks, margins, norm-vs-ellipsoid plots.")
    ap.add_argument("run_root", type=str, help="Path to run root (…/data_runs/<run_id>)")
    ap.add_argument("--mvie-npz", type=str, default=None, help="Path to MVIE results_allsteps.npz")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = Path(args.run_root)
    plot_pe_ranks(root, show=args.show)
    plot_margin_histograms(root, show=args.show)
    if args.mvie_npz:
        plot_norms_vs_ellipsoids(root, args.mvie_npz, show=args.show)
