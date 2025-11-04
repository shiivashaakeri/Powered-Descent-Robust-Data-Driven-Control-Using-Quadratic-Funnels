# nominal/viz/plot3d.py
from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from mpl_toolkits.mplot3d import art3d  # pyright: ignore[reportMissingImports]

from ..utils.quats import dcm_from_quat, normalize_quat


class IterBrowser3D:
    """
    Interactive 3D browser for 6-DoF nominal solutions.
    State: [m, r(3), v(3), q(4)=(w,x,y,z), w(3)], Control: U=thrust in body (3,)
    """

    def __init__(self, X_all: np.ndarray, U_all: np.ndarray, att_len: float = 20.0, thrust_len: float = 2e-5):
        """
        X_all: (N_iter, n_x, K)
        U_all: (N_iter, n_u, K)
        """
        assert X_all.ndim == 3 and U_all.ndim == 3
        self.X_all = X_all
        self.U_all = U_all
        self.N = X_all.shape[0]
        self.i = self.N - 1
        self.att_len = att_len
        self.thrust_len = thrust_len

    def _draw(self, ax):
        ax.clear()
        Xi = self.X_all[self.i]
        Ui = self.U_all[self.i]
        K = Xi.shape[1]

        r = Xi[1:4]  # (3,K)
        q = Xi[7:11]  # (4,K) (w,x,y,z)

        # Path
        ax.plot(r[0], r[1], r[2], color="lightgrey", zorder=0)

        # Attitude & thrust quivers
        for k in range(K):
            rw = r[:, k]
            qwxyz = normalize_quat(q[:, k])
            C_BI = dcm_from_quat(qwxyz)  # body->inertial
            C_IB = C_BI.T  # noqa: F841

            # Attitude: body +z axis in inertial
            ez_b = np.array([0.0, 0.0, 1.0])
            ez_i = C_BI @ ez_b

            # Thrust: U is in body frame; show -U in inertial as a vector
            u_b = Ui[:, k]
            F_i = -(C_BI @ u_b)

            ax.quiver(
                rw[0],
                rw[1],
                rw[2],
                ez_i[0],
                ez_i[1],
                ez_i[2],
                length=self.att_len,
                arrow_length_ratio=0.0,
                color="blue",
            )
            ax.quiver(
                rw[0],
                rw[1],
                rw[2],
                F_i[0],
                F_i[1],
                F_i[2],
                length=self.thrust_len,
                arrow_length_ratio=0.0,
                color="red",
            )

        # Ground pad
        pad = plt.Circle((0, 0), 20, color="lightgray")
        ax.add_patch(pad)
        art3d.pathpatch_2d_to_3d(pad, z=0, zdir="z")

        # Axes & labels
        ax.set_xlabel("East (x)")
        ax.set_ylabel("North (y)")
        ax.set_zlabel("Up (z)")
        ax.set_title(f"Iteration {self.i}")

        # Scale view to fit entire trajectory
        r_x_min, r_x_max = float(np.min(r[0])), float(np.max(r[0]))
        r_y_min, r_y_max = float(np.min(r[1])), float(np.max(r[1]))
        r_z_min, r_z_max = float(np.min(r[2])), float(np.max(r[2]))
        # Add padding
        pad_x = max(50.0, (r_x_max - r_x_min) * 0.1)
        pad_y = max(50.0, (r_y_max - r_y_min) * 0.1)
        pad_z = max(50.0, (r_z_max - r_z_min) * 0.1)
        ax.set_xlim([r_x_min - pad_x, r_x_max + pad_x])
        ax.set_ylim([r_y_min - pad_y, r_y_max + pad_y])
        ax.set_zlim([max(0.0, r_z_min - pad_z), r_z_max + pad_z])

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
        fig = plt.figure(figsize=(9, 10))
        ax = fig.add_subplot(111, projection="3d")
        self._draw(ax)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def show(self):
        fig = plt.figure(figsize=(9, 10))
        ax = fig.add_subplot(111, projection="3d")
        self._draw(ax)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show()


def plot3d(
    X_all: np.ndarray,
    U_all: np.ndarray,
    att_len: float = 20.0,
    thrust_len: float = 2e-5,
    save_path: str | None = None,
):
    browser = IterBrowser3D(X_all, U_all, att_len, thrust_len)
    if save_path:
        browser.save(save_path)
    else:
        browser.show()


def plot3d_time_series(X: np.ndarray, U: np.ndarray, t: np.ndarray | None = None):
    """
    Plot trajectories of all 6-DoF states and inputs over time as subplots.
    X: (14, K) with ordering [m, r(3), v(3), q(4)=(w,x,y,z), w(3)]
    U: (3, K) body-frame thrust
    """
    assert X.ndim == 2 and U.ndim == 2
    n_x, K = X.shape
    n_u = U.shape[0]
    if t is None:
        t = np.arange(K)

    state_labels = [
        "m",
        "r_x", "r_y", "r_z",
        "v_x", "v_y", "v_z",
        "q0", "q1", "q2", "q3",
        "w_x", "w_y", "w_z",
    ]
    input_labels = ["T_x", "T_y", "T_z"]

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
    ncols_u = 3
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
