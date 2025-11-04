# nominal/run/run_free.py
from __future__ import annotations

import argparse
import time
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource, reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import yaml  # pyright: ignore[reportMissingModuleSource]

from ..dynamics.integrators.foh_free import FirstOrderHold as FOHFree
from ..init.warmstart import warmstart_free
from ..io.save import save_arrays, save_metadata
from ..models.rocket2d import Rocket2D
from ..models.rocket6dof import Rocket6DoF
from ..scvx.free_final_time import SCProblem
from ..solvers.cvxpy_solve import solve as cvx_solve
from ..utils.formatting import format_line, hr
from ..utils.seeds import set_seed
from ..viz.plot2d import plot2d
from ..viz.plot3d import plot3d


def _build_model(name: str):
    name = (name or "").lower()
    if name in ("rocket2d", "2d", "mars2d"):
        return Rocket2D()
    if name in ("rocket6dof", "6dof", "mars6dof"):
        return Rocket6DoF()
    raise ValueError(f"Unknown model name '{name}'")


def _load_cfg(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _meta_float(x):
    try:
        return float(x)
    except Exception:
        return x


def run(cfg_path: str) -> Tuple[Path, Dict[str, np.ndarray]]:  # noqa: C901, PLR0915, PLR0912
    cfg = _load_cfg(Path(cfg_path))

    # --- seed & I/O ---
    seed = int(cfg.get("run", {}).get("seed", 0))
    seed = set_seed(seed)
    output_dir = Path(cfg.get("run", {}).get("output_dir", "nominal_trajectories"))
    # Normalize legacy defaults
    try:
        output_str = str(output_dir)
        if output_str == "output/trajectory" or output_str.startswith("output/") or output_str in {"output", "trajectory"} or output_str.startswith("nominal/"):  # noqa: E501
            output_dir = Path("nominal_trajectories")
    except Exception:
        pass
    do_plot = bool(cfg.get("run", {}).get("plot", False))

    # --- discretization / solver params ---
    disc = cfg.get("discretization", {})
    K = int(disc.get("K", 30))
    iterations = int(disc.get("iterations", 50))
    solver = str(disc.get("solver", "ECOS"))
    verbose_solver = bool(disc.get("verbose_solver", False))

    tr = cfg.get("trust_region", {})
    w_nu = float(tr.get("w_nu", 1e5))
    w_sigma = float(tr.get("w_sigma", 10.0))
    tr_radius = float(tr.get("tr_radius_init", 5.0))
    rho_0 = float(tr.get("rho_0", 0.0))
    rho_1 = float(tr.get("rho_1", 0.25))
    rho_2 = float(tr.get("rho_2", 0.9))
    alpha = float(tr.get("alpha", 2.0))
    beta = float(tr.get("beta", 3.2))

    # --- model ---
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "rocket6dof")
    model = _build_model(model_name)
    if bool(model_cfg.get("nondimensionalize", True)):
        model.nondimensionalize()

    # --- warm start ---
    X, U, sigma = warmstart_free(model, K)

    # --- SCVX objects ---
    integrator = FOHFree(model, K)
    problem = SCProblem(model, K)

    # --- loop state ---
    all_X = [model.x_redim(X.copy())]
    all_U = [model.u_redim(U.copy())]
    all_sigma = [float(sigma)]
    converged = False
    last_nl_cost = None
    it_done = 0

    for it in range(iterations):
        print(hr())
        print(f"------ Iteration {it + 1:02d} ------")
        t_it = time.time()

        # Discretize about current (X,U,sigma)
        t0 = time.time()
        A_bar, B_bar, C_bar, S_bar, z_bar = integrator.calculate_discretization(X, U, sigma)
        print(format_line("Time for transition matrices", time.time() - t0, " s"))

        # Fill parameters
        problem.set_parameters(
            A_bar=A_bar,
            B_bar=B_bar,
            C_bar=C_bar,
            S_bar=S_bar,
            z_bar=z_bar,
            X_last=X,
            U_last=U,
            sigma_last=sigma,
            weight_nu=w_nu,
            weight_sigma=w_sigma,
            tr_radius=tr_radius,
        )

        # Inner accept/reject loop
        while True:
            err = cvx_solve(problem.prob, solver=solver, verbose=verbose_solver, max_iters=200, retries=0)
            print(format_line("Solver error", err))

            new_X = problem.get_variable("X")
            new_U = problem.get_variable("U")
            if err or new_X is None or new_U is None:
                tr_radius /= alpha
                print(f"Solver failed (status={problem.prob.status}). Retrying with radius={tr_radius:.4g}")
                problem.set_parameters(tr_radius=tr_radius)
                if tr_radius < 1e-8:
                    raise RuntimeError("Trust region shrank too small without a feasible/optimal solve.")
                continue
            new_sigma_val = problem.get_variable("sigma")
            new_sigma = float(new_sigma_val)

            # Nonlinear roll-out (for accuracy metric)
            X_nl = integrator.integrate_nonlinear_piecewise(new_X, new_U, new_sigma)

            lin_cost_dyn = np.linalg.norm(problem.get_variable("nu"), 1)
            nl_cost_dyn = np.linalg.norm(new_X - X_nl, 1)
            lin_cost_con = model.get_linear_cost()
            nl_cost_con = model.get_nonlinear_cost(X=new_X, U=new_U)

            L = nl_cost_dyn + nl_cost_con
            J = lin_cost_dyn + lin_cost_con

            print(format_line("Final time (sigma)", new_sigma))

            if last_nl_cost is None:
                last_nl_cost = L
                X, U, sigma = new_X, new_U, new_sigma
                break

            actual_change = last_nl_cost - L
            predicted_change = last_nl_cost - J

            print(format_line("Virtual control cost", lin_cost_dyn))
            print(format_line("Constraint cost", lin_cost_con))
            print(format_line("Actual change", actual_change))
            print(format_line("Predicted change", predicted_change))

            if abs(predicted_change) < 1e-4:
                converged = True
                break

            rho = actual_change / predicted_change
            if rho < rho_0:
                tr_radius /= alpha
                print(f"Trust region too large. Retrying with radius={tr_radius:.4g}")
            else:
                X, U, sigma = new_X, new_U, new_sigma
                print("Solution accepted.")
                if rho < rho_1:
                    tr_radius /= alpha
                    print("Decreasing trust region.")
                elif rho >= rho_2:
                    tr_radius *= beta
                    print("Increasing trust region.")
                last_nl_cost = L
                break

            problem.set_parameters(tr_radius=tr_radius)

        print(format_line("Time for iteration", time.time() - t_it, " s"))
        all_X.append(model.x_redim(X.copy()))
        all_U.append(model.u_redim(U.copy()))
        all_sigma.append(float(sigma))
        it_done = it + 1
        if converged:
            print(f"Converged after {it_done} iterations.")
            break

    if not converged:
        print("Maximum iterations reached without convergence.")

    all_X = np.stack(all_X, axis=0)
    all_U = np.stack(all_U, axis=0)
    all_sigma = np.array(all_sigma, dtype=float)

    # --- Save arrays + metadata ---
    model_name = model.__class__.__name__.lower()
    run_dir = save_arrays(output_dir / model_name,
                          {"X": all_X, "U": all_U, "sigma": all_sigma},
                          use_numeric_subdir=False)
    meta = {
        "config_path": str(cfg_path),
        "config": cfg,
        "seed": int(seed),
        "model": {
            "name": model.__class__.__name__,
            "n_x": int(model.n_x),
            "n_u": int(model.n_u),
            "r_scale": _meta_float(getattr(model, "r_scale", np.nan)),
            "m_scale": _meta_float(getattr(model, "m_scale", np.nan)),
        },
        "discretization": {"K": int(K), "iterations_requested": int(iterations), "iterations_done": int(it_done)},
        "solver": {"name": solver, "verbose": bool(verbose_solver)},
        "trust_region": {
            "w_nu": _meta_float(w_nu),
            "w_sigma": _meta_float(w_sigma),
            "rho_0": _meta_float(rho_0),
            "rho_1": _meta_float(rho_1),
            "rho_2": _meta_float(rho_2),
            "alpha": _meta_float(alpha),
            "beta": _meta_float(beta),
        },
        "converged": bool(converged),
        "final_sigma": _meta_float(all_sigma[-1] if len(all_sigma) else np.nan),
    }
    save_metadata(run_dir, meta)
    print(f"Saved arrays + metadata to {run_dir}")

    # --- Save figures: states and inputs ---
    Xf = all_X[-1]
    Uf = all_U[-1]
    K = Xf.shape[1]
    t = np.arange(K)

    if isinstance(model, Rocket6DoF):
        state_labels = [
            "m",
            "r_x","r_y","r_z",
            "v_x","v_y","v_z",
            "q0","q1","q2","q3",
            "w_x","w_y","w_z",
        ]
        input_labels = ["T_x","T_y","T_z"]
    else:
        state_labels = ["rx","ry","vx","vy","theta","omega"]
        input_labels = ["gimbal","T"]

    n_x = Xf.shape[0]
    ncols = 3
    nrows = ceil(n_x / ncols)
    fig_states, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), squeeze=False)
    for i in range(n_x):
        r, c = divmod(i, ncols)
        axs[r][c].plot(t, Xf[i])
        axs[r][c].set_title(state_labels[i] if i < len(state_labels) else f"x[{i}]")
        axs[r][c].set_xlabel("k")
        axs[r][c].set_ylabel("value")
    for j in range(n_x, nrows*ncols):
        r, c = divmod(j, ncols)
        axs[r][c].axis('off')
    fig_states.suptitle("states of rocket")
    fig_states.tight_layout()
    fig_states.savefig(run_dir / "states.png", dpi=150)
    plt.close(fig_states)

    n_u = Uf.shape[0]
    ncols_u = 3
    nrows_u = ceil(n_u / ncols_u)
    fig_inputs, axs_u = plt.subplots(nrows_u, ncols_u, figsize=(4*ncols_u, 2.8*nrows_u), squeeze=False)
    for i in range(n_u):
        r, c = divmod(i, ncols_u)
        axs_u[r][c].plot(t, Uf[i])
        axs_u[r][c].set_title(input_labels[i] if i < len(input_labels) else f"u[{i}]")
        axs_u[r][c].set_xlabel("k")
        axs_u[r][c].set_ylabel("value")
    for j in range(n_u, nrows_u*ncols_u):
        r, c = divmod(j, ncols_u)
        axs_u[r][c].axis('off')
    fig_inputs.suptitle("inputs of rocket")
    fig_inputs.tight_layout()
    fig_inputs.savefig(run_dir / "inputs.png", dpi=150)
    plt.close(fig_inputs)

    # --- Save interactive trajectory plot ---
    traj_plot_path = run_dir / "trajectory.png"
    if isinstance(model, Rocket6DoF):
        plot3d(all_X, all_U, save_path=str(traj_plot_path))
    else:
        plot2d(all_X, all_U, save_path=str(traj_plot_path))
    print(f"Saved trajectory plot to {traj_plot_path}")

    # --- Optional interactive plotting ---
    if do_plot:
        if isinstance(model, Rocket6DoF):
            plot3d(all_X, all_U)
        else:
            plot2d(all_X, all_U)

    return run_dir, {"X": all_X, "U": all_U, "sigma": all_sigma}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config (e.g., configs/mars_6dof_free.yaml)")
    args = parser.parse_args()
    run(args.config)
