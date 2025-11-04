# Nominal Powered-Descent (Successive Convexification)

This is the **nominal** component of the larger project
`Powered-Descent-Robust-Data-Driven-Control-Using-Quadratic-Funnels`.
It implements a clean, modular version of the **Successive Convexification (SCVX)** pipeline
for powered-descent trajectory optimization, with both **fixed** and **free-final-time** variants.

> **Scope:** Only the nominal (deterministic) dynamics, discretization, and optimizer loops.
> Ignore “Robust” and “Data-Driven” for now.

---

## Mapping to Paper Sections

- **Dynamics & Linearization:** `models/`, `dynamics/sympy_backend.py`
- **FOH Discretization:** `dynamics/integrators/foh_fixed.py`, `dynamics/integrators/foh_free.py`
- **Convex Subproblems (SCVX):** `scvx/`
- **Initialization:** `init/warmstart.py`
- **Solvers:** `solvers/cvxpy_solve.py`
- **Orchestrators:** `run/run_fixed.py`, `run/run_free.py`
- **I/O & Viz:** `io/`, `viz/`, `utils/`

---

## Quickstart

1. **Install**
   ```bash
   uv venv && source .venv/bin/activate  # or your preferred environment
   pip install -e .