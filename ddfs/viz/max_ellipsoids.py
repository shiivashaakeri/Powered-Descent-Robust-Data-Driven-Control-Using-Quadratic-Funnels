# ddfs/viz/max_ellipsoids.py
from __future__ import annotations

from os import PathLike
from typing import Callable, Optional, Sequence, Tuple, Dict, List

import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]


# -----------------------------
# linear-algebra helpers
# -----------------------------
def principal_block(M: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return the 2x2 principal block [[i,j],[i,j]] from a 3D stack or 2D matrix."""
    if M.ndim == 2:
        return M[np.ix_([i, j], [i, j])]
    # if M is (n,n,K), caller should slice K outside
    raise ValueError("principal_block expects a 2D matrix (slice K outside).")


def ellipse_boundary(Q2: np.ndarray, n_pts: int = 256) -> np.ndarray:
    """
    Boundary of the ellipsoid { z : z^T Q2 z <= 1 } centered at the origin,
    returned as a (2,n_pts) array.
    """
    # For PSD Q2, boundary can be sampled via whitening transform
    V, S, _ = np.linalg.svd(Q2)
    M = V @ np.diag(1.0 / np.sqrt(np.maximum(S, 1e-12)))
    ang = np.linspace(0, 2 * np.pi, n_pts)
    circ = np.vstack([np.cos(ang), np.sin(ang)])
    return M @ circ


def feasible_star(A2: np.ndarray, margins: np.ndarray, n_pts: int = 512) -> np.ndarray:
    """
    Star-shaped polygon approximating the intersection of 2D halfspaces
    { z : a_i^T z <= m_i } around the origin. Returns a closed curve (2, n_pts+1).
    """
    if A2.size == 0 or margins.size == 0:
        th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        circ = np.vstack([np.cos(th), np.sin(th)]) * 1e-6
        return np.hstack([circ, circ[:, :1]])

    th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    d = np.vstack([np.cos(th), np.sin(th)])                   # (2, n_pts)
    denom = A2 @ d                                           # (m, n_pts)
    denom_pos = np.where(denom > 1e-12, denom, np.inf)
    r = np.min(margins[:, None] / denom_pos, axis=0)         # (n_pts,)
    r = np.clip(r, 0.0, 1e6)
    curve = d * r
    return np.hstack([curve, curve[:, :1]])


def cap_circle(cap: float, n_pts: int = 360) -> np.ndarray:
    """Circle of radius `cap`, centered at origin, returned as (2,n_pts)."""
    ang = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return np.vstack([np.cos(ang), np.sin(ang)]) * float(cap)


def project_halfspaces(
    A: np.ndarray, b: np.ndarray, center: np.ndarray, pair: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project the halfspaces (A, b) for the full space onto a 2D slice given by `pair`,
    after shifting by the local center to compute margins.
    Returns (A2, margins) where A2 is (m,2) and margins is (m,).
    """
    margins = b - A @ center
    keep = margins > 1e-10
    return A[keep][:, list(pair)], margins[keep]


# -----------------------------
# drawing helpers
# -----------------------------
def shade_feasible(
    ax,
    A2: np.ndarray,
    margins: np.ndarray,
    *,
    cap: Optional[float] = None,
    face: str = "tab:blue",
    edge: str = "tab:blue",
    alpha: float = 0.12,
    edge_alpha: float = 0.7,
    label: str = "feasible (lin.)",
) -> None:
    """Fill & outline the star-shaped feasible set; optionally draw ‖·‖≤cap circle."""
    poly = feasible_star(A2, margins)
    ax.fill(poly[0], poly[1], color=face, alpha=alpha, linewidth=0)
    ax.plot(poly[0], poly[1], color=edge, alpha=edge_alpha, lw=1.5, label=label)
    if cap is not None and cap > 0:
        circ = cap_circle(cap)
        ax.plot(circ[0], circ[1], color=edge, alpha=0.35, lw=1.0, ls="--", label="‖·‖≤cap")


def default_pairs(model_name: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Which 2D slices to visualize by default.
    You can override by passing custom pairs from your run script if you want.
    """
    if model_name == "rocket2d":
        # positions, velocities, attitude block; and the (gimbal, T) input
        return {
            "state": [(0, 1), (2, 3), (4, 5)],
            "input": [(0, 1)],
        }
    else:
        # a reasonable 6DoF subset (positions, velocities, body-rate), and inputs
        return {
            "state": [(1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (11, 12)],
            "input": [(0, 1), (0, 2), (1, 2)],
        }


# -----------------------------
# primary figure: a few steps (k) but full red trajectory
# -----------------------------
def save_summary_plot(
    out_png: "PathLike[str] | str",
    model_name: str,
    X: np.ndarray,                 # (n_x, K)
    U: np.ndarray,                 # (n_u, K)
    Q: np.ndarray,                 # (n_x, n_x, K)
    R: np.ndarray,                 # (n_u, n_u, K)
    build_Ax_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    build_Au_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    cap_x: Optional[float] = None,
    cap_u: Optional[float] = None,
    steps: Sequence[int] | None = None,
    title: str | None = None,
) -> str:
    """
    Make a compact figure for a few steps (e.g., first/middle/last).
    For each panel:
      • shaded feasible set (linearized at the local center),
      • MVIE ellipsoid slice centered at that step,
      • **full nominal trajectory** in that 2D slice plotted in red (all time steps),
      • a small black dot marking the current step k.
    """
    from pathlib import Path

    if steps is None:
        steps = [0, max(0, (X.shape[1] - 1) // 2), X.shape[1] - 1]

    pairs = default_pairs(model_name)
    nrows = len(steps)
    ncols = len(pairs["state"]) + len(pairs["input"])
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    # Precompute 2D trajectories for speed
    def traj2D(data: np.ndarray, i: int, j: int) -> np.ndarray:
        return data[[i, j], :]  # (2, K)

    for r, k in enumerate(steps):
        Ax, bx = build_Ax_fn(X[:, k])
        Au, bu = build_Au_fn(U[:, k])

        col = 0
        # ---- state panels
        for (i, j) in pairs["state"]:
            A2, m2 = project_halfspaces(Ax, bx, X[:, k], (i, j))
            Q2 = principal_block(Q[:, :, k], i, j)
            ell = ellipse_boundary(Q2)

            x_traj = traj2D(X, i, j)  # (2, K)

            ax = axs[r, col]; col += 1
            shade_feasible(ax, A2, m2, cap=cap_x)
            ax.plot(ell[0] + X[i, k], ell[1] + X[j, k], lw=2, color="tab:orange", label="ellipsoid")

            # full nominal trajectory in red
            ax.plot(x_traj[0], x_traj[1], color="crimson", lw=1.6, alpha=0.9, label="nominal (traj)")

            # mark the current step k as a black dot on top
            ax.scatter([X[i, k]], [X[j, k]], s=18, zorder=6, color="black")

            ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3)
            ax.set_title(f"k={k} • state({i},{j})")

        # ---- input panels
        for (i, j) in pairs["input"]:
            A2, m2 = project_halfspaces(Au, bu, U[:, k], (i, j))
            R2 = principal_block(R[:, :, k], i, j)
            ell = ellipse_boundary(R2)

            u_traj = traj2D(U, i, j)  # (2, K)

            ax = axs[r, col]; col += 1
            shade_feasible(ax, A2, m2, cap=cap_u)
            ax.plot(ell[0] + U[i, k], ell[1] + U[j, k], lw=2, color="tab:orange", label="ellipsoid")

            # full input trajectory in red
            ax.plot(u_traj[0], u_traj[1], color="crimson", lw=1.6, alpha=0.9, label="nominal (traj)")

            # mark the current step k as a black dot on top
            ax.scatter([U[i, k]], [U[j, k]], s=18, zorder=6, color="black")

            ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3)
            ax.set_title(f"k={k} • input({i},{j})")

    # single legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    if title:
        fig.suptitle(title, y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return str(out_png)


def save_time_sweep_plot(
    out_png: "PathLike[str] | str",
    centers: np.ndarray,   # (2, K) slice of X or U
    blocks: np.ndarray,    # (2,2,K) corresponding Q or R principal blocks
    title: str,
    k_ref: Optional[int] = None,
    cap: Optional[float] = None,
) -> str:
    """
    Draw every ellipsoid over time with fading color; plot the FULL center trajectory;
    mark k_ref (default: last step) with a black dot.
    """
    from pathlib import Path
    K = centers.shape[1]
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7, 5.2))

    # full center trajectory
    ax.plot(centers[0], centers[1], color="crimson", lw=1.8, alpha=0.95, label="nominal (traj)")

    for k in range(K):
        col = cmap(k / max(K - 1, 1))
        e = ellipse_boundary(blocks[:, :, k])
        ax.plot(e[0] + centers[0, k], e[1] + centers[1, k], color=col, alpha=0.75, lw=1.2)

    if k_ref is None:
        k_ref = K - 1
    ax.scatter([centers[0, k_ref]], [centers[1, k_ref]], color="black", s=26, zorder=6)

    if cap is not None and cap > 0:
        circ = cap_circle(cap)
        ax.plot(circ[0], circ[1], color="gray", lw=1.0, ls="--", alpha=0.5, label="‖·‖≤cap")

    ax.set_aspect("equal", "box"); ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper right")
    out = Path(out_png); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    return str(out)

def _blocks2x2(stack: np.ndarray, i: int, j: int) -> np.ndarray:
    """Extract (2,2,K) principal blocks from a (n,n,K) stack."""
    K = stack.shape[2]
    return stack[np.ix_([i, j], [i, j], np.arange(K))]  # (2,2,K)


def save_state_time_sweeps(
    out_dir: "PathLike[str] | str",
    model_name: str,
    X: np.ndarray,          # (n_x, K)
    Q: np.ndarray,          # (n_x, n_x, K)
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    cap_x: Optional[float] = None,
) -> List[str]:
    """
    For each (i,j) state pair, make a time-sweep plot with the full state trajectory.
    Returns list of file paths saved.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    if pairs is None:
        pairs = default_pairs(model_name)["state"]

    paths: List[str] = []
    for (i, j) in pairs:
        centers = X[[i, j], :]                   # (2,K)
        blocks = _blocks2x2(Q, i, j)            # (2,2,K)
        p = out_dir / f"timesweep_state_{i}_{j}.png"
        paths.append(
            save_time_sweep_plot(
                p, centers, blocks, title=f"state({i},{j}) • time-sweep", k_ref=None, cap=cap_x
            )
        )
    return paths


def save_input_time_sweeps(
    out_dir: "PathLike[str] | str",
    model_name: str,
    U: np.ndarray,          # (n_u, K)
    R: np.ndarray,          # (n_u, n_u, K)
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    cap_u: Optional[float] = None,
) -> List[str]:
    """
    For each (i,j) input pair, make a time-sweep plot with the full input trajectory.
    Returns list of file paths saved.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    if pairs is None:
        pairs = default_pairs(model_name)["input"]

    paths: List[str] = []
    for (i, j) in pairs:
        centers = U[[i, j], :]                   # (2,K)
        blocks = _blocks2x2(R, i, j)            # (2,2,K)
        p = out_dir / f"timesweep_input_{i}_{j}.png"
        paths.append(
            save_time_sweep_plot(
                p, centers, blocks, title=f"input({i},{j}) • time-sweep", k_ref=None, cap=cap_u
            )
        )
    return paths