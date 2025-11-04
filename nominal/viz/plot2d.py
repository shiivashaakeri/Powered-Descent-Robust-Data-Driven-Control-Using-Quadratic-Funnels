# nominal/viz/plot2d.py
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]


class IterBrowser2D:
    """
    Interactive browser for 2D nominal solutions.
    Assumes state layout: [rx, ry, vx, vy, theta, omega], control: [gimbal, T].
    East = x, Up = y.
    """

    def __init__(self, X_all: np.ndarray, U_all: np.ndarray, att_scale: float = 15.0, thrust_scale: float = 100.0):
        """
        X_all: (N_iter, n_x, K)
        U_all: (N_iter, n_u, K)
        """
        assert X_all.ndim == 3 and U_all.ndim == 3, "Provide stacked iterations (N, nx, K)."
        self.X_all = X_all
        self.U_all = U_all
        self.N = X_all.shape[0]
        self.i = self.N - 1

        self.att_scale = att_scale
        self.thrust_scale = thrust_scale

    def _draw(self, ax):
        ax.clear()
        Xi = self.X_all[self.i]
        Ui = self.U_all[self.i]
        K = Xi.shape[1]  # noqa: F841

        rx, ry = Xi[0], Xi[1]
        theta = Xi[4]

        # Attitude (body z-axis projection in 2D)
        dx = np.sin(theta)
        dy = np.cos(theta)

        # Thrust (body thrust aligned with +z_body, engine points -z_body in world)
        Fx = -np.sin(theta + Ui[0]) * Ui[1]
        Fy = -np.cos(theta + Ui[0]) * Ui[1]

        ax.set_xlabel("East")
        ax.set_ylabel("Up")
        ax.plot(rx, ry, color="lightgrey", zorder=0)

        ax.quiver(
            rx,
            ry,
            dx,
            dy,
            color="blue",
            width=0.003,
            scale=self.att_scale,
            headwidth=1,
            headlength=0,
            angles="xy",
            scale_units="xy",
        )
        ax.quiver(
            rx,
            ry,
            Fx,
            Fy,
            color="red",
            width=0.002,
            scale=self.thrust_scale,
            headwidth=1,
            headlength=0,
            angles="xy",
            scale_units="xy",
        )

        ax.set_aspect("equal")
        ax.set_title(f"Iteration {self.i}")

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(event.canvas.figure)
            return
        if event.key == "right":
            self.i = (self.i + 1) % self.N
        elif event.key == "left":
            self.i = (self.i - 1) % self.N
        self._draw(event.canvas.figure.axes[0])
        event.canvas.draw_idle()

    def save(self, path):
        """Save the final iteration plot to a file."""
        fig, ax = plt.subplots(figsize=(8, 9))
        self._draw(ax)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def show(self):
        fig, ax = plt.subplots(figsize=(8, 9))
        self._draw(ax)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show()


def plot2d(
    X_all: np.ndarray,
    U_all: np.ndarray,
    att_scale: float = 15.0,
    thrust_scale: float = 100.0,
    save_path: str | None = None,
):
    browser = IterBrowser2D(X_all, U_all, att_scale, thrust_scale)
    if save_path:
        browser.save(save_path)
    else:
        browser.show()


def plot2d_time_series(X: np.ndarray, U: np.ndarray, t: np.ndarray | None = None):
    """
    Plot trajectories of all states and inputs over time as subplots.
    X: (n_x, K), U: (n_u, K)
    """
    assert X.ndim == 2 and U.ndim == 2
    n_x, K = X.shape
    n_u = U.shape[0]
    if t is None:
        t = np.arange(K)

    state_labels = ["rx", "ry", "vx", "vy", "theta", "omega"]
    input_labels = ["gimbal", "T"]

    # States
    ncols = 3
    nrows = ceil(n_x / ncols)
    fig_s, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    for i in range(n_x):
        r, c = divmod(i, ncols)
        axs[r][c].plot(t, X[i])
        lbl = state_labels[i] if i < len(state_labels) else f"x[{i}]"
        axs[r][c].set_title(lbl)
        axs[r][c].set_xlabel("k")
        axs[r][c].set_ylabel("value")
    for j in range(n_x, nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r][c].axis("off")
    fig_s.suptitle("states of rocket")
    fig_s.tight_layout()

    # Inputs
    ncols_u = 2
    nrows_u = ceil(n_u / ncols_u)
    fig_u, axs_u = plt.subplots(nrows_u, ncols_u, figsize=(4 * ncols_u, 2.8 * nrows_u), squeeze=False)
    for i in range(n_u):
        r, c = divmod(i, ncols_u)
        axs_u[r][c].plot(t, U[i])
        lbl = input_labels[i] if i < len(input_labels) else f"u[{i}]"
        axs_u[r][c].set_title(lbl)
        axs_u[r][c].set_xlabel("k")
        axs_u[r][c].set_ylabel("value")
    for j in range(n_u, nrows_u * ncols_u):
        r, c = divmod(j, ncols_u)
        axs_u[r][c].axis("off")
    fig_u.suptitle("inputs of rocket")
    fig_u.tight_layout()

    return fig_s, fig_u
