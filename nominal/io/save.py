# nominal/io/save.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np  # pyright: ignore[reportMissingImports]


def _next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    run_id = f"{(max(nums) + 1) if nums else 0:03d}"
    out = base / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_arrays(output_dir: str | os.PathLike,
                arrays: Dict[str, np.ndarray],
                *,
                use_numeric_subdir: bool = True) -> Path:
    """
    Save a dict of numpy arrays as *.npy into an auto-incremented folder under output_dir.
    Returns the created run directory path.
    """
    base = Path(output_dir)
    if use_numeric_subdir:
        out = _next_run_dir(base)
    else:
        out = base
        out.mkdir(parents=True, exist_ok=True)
    for k, v in arrays.items():
        np.save(str(out / f"{k}.npy"), v)
    return out


def save_metadata(run_dir: str | os.PathLike, metadata: Dict[str, Any]) -> Path:
    """
    Save metadata.json alongside arrays.
    """
    run_dir = Path(run_dir)
    meta_path = run_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return meta_path
