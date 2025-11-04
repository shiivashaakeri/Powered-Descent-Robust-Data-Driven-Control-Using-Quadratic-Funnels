# nominal/io/load.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]


def _resolve_run_dir(output_dir: Path, run_id: str | None) -> Path:
    if run_id is None or run_id == "latest":
        nums = sorted([int(p.name) for p in output_dir.iterdir() if p.is_dir() and p.name.isdigit()])
        if not nums:
            raise FileNotFoundError(f"No runs under {output_dir}")
        return output_dir / f"{nums[-1]:03d}"
    d = output_dir / run_id
    if not d.exists():
        raise FileNotFoundError(f"Run directory does not exist: {d}")
    return d


def load_run(output_dir: str, run_id: str | None = "latest") -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load arrays (all *.npy) and metadata.json from a run directory.
    """
    odir = Path(output_dir)
    rdir = _resolve_run_dir(odir, run_id)

    arrays: Dict[str, np.ndarray] = {}
    for npy in rdir.glob("*.npy"):
        arrays[npy.stem] = np.load(npy)

    meta_path = rdir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open("r") as f:
            meta = json.load(f)
    return arrays, meta
