# ddfs/io/datasets.py
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]

from ddfs.core.data_logger import WindowMeta
from ddfs.io.paths import RunPaths


@dataclass
class RunInfo:
    """Metadata describing this data-collection run."""
    run_id: str                 # unique id (timestamp, seed, etc.)
    model: str                  # "rocket2d" | "rocket6dof"
    n: int
    m: int
    dt: float
    T: int                      # segment length in steps
    L: int                      # window length in steps
    K_total: int                # total nominal length (optional but useful)
    note: str = ""              # free-form note
    extras: Optional[Dict[str, object]] = None


class DatasetWriter:
    """Writes per-segment window arrays and a run-level manifest/summary.

    Layout:
      {base}/nominal_trajectories/{model}/data_runs/{run_id}/
        artifacts/
          manifest.json
          summary.json              (optional PE checks)
        segments/
          seg_0000/
            H.npy
            H_plus.npy
            Xi.npy
            raw_window.npz          (H,H_plus,Xi,extras,...)
            segment_meta.json
          seg_0001/
            ...
        aggregate/
          aggregate.npz             (optional: stacked H/H+/Xi across segments)
    """

    def __init__(self, paths: RunPaths, run_info: RunInfo) -> None:
        self.paths = paths
        self.run_info = run_info
        # ensure dirs
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.segments_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # bookkeeping
        self._segments_written: List[int] = []
        self._per_seg_shapes: List[Dict[str, Tuple[int, int]]] = []
        self._pe_flags: List[Dict[str, object]] = []  # per-seg PE checks

        # write manifest stub (filled/touched now, updated on finalize)
        self._write_manifest(initial=True)

    # -----------------------------
    # Per-segment writers
    # -----------------------------
    def write_segment_window(
        self,
        seg_idx: int,
        H: np.ndarray,
        H_plus: np.ndarray,
        Xi: np.ndarray,
        meta: WindowMeta,
        extras: Optional[Dict[str, np.ndarray]] = None,
        do_pe_check: bool = True,
    ) -> str:
        """Write per-segment arrays + metadata + raw npz. Returns segment dir."""
        seg_dir = self.paths.seg_dir(seg_idx)
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Basic shape checks
        n, L = H.shape
        n2, L2 = H_plus.shape
        m, L3 = Xi.shape
        if n != self.run_info.n or m != self.run_info.m or L != meta.L or L2 != meta.L or L3 != meta.L or n2 != n:
            raise ValueError(
                f"Shape mismatch: "
                f"H{H.shape}, H_plus{H_plus.shape}, Xi{Xi.shape}, meta.L={meta.L}, run(n={self.run_info.n},m={self.run_info.m})"
            )

        # Save simple .npy files
        np.save(seg_dir / "H.npy", H)
        np.save(seg_dir / "H_plus.npy", H_plus)
        np.save(seg_dir / "Xi.npy", Xi)

        # Save raw window npz (includes extras + meta blob)
        payload = dict(H=H, H_plus=H_plus, Xi=Xi, meta=np.array([asdict(meta)], dtype=object))
        if extras:
            for k, v in extras.items():
                payload[k] = v
        np.savez(seg_dir / "raw_window.npz", **payload)

        # Save segment_meta.json (lightweight JSON)
        seg_meta = asdict(meta)
        with open(seg_dir / "segment_meta.json", "w") as f:
            json.dump(seg_meta, f, indent=2)

        # PE check (optional)
        pe_info = {}
        if do_pe_check:
            # rank([H; Xi]) should be n+m if L >= n+m (informative window)
            stacked = np.vstack([H, Xi])
            rank = int(np.linalg.matrix_rank(stacked))
            pe_ok = bool(rank == (self.run_info.n + self.run_info.m))
            pe_info = {"rank_H_Xi": rank, "pe_required": self.run_info.n + self.run_info.m, "pe_ok": pe_ok}
            with open(seg_dir / "pe_check.json", "w") as f:
                json.dump(pe_info, f, indent=2)

        # bookkeep
        self._segments_written.append(seg_idx)
        self._per_seg_shapes.append(
            {"H": H.shape, "H_plus": H_plus.shape, "Xi": Xi.shape, "kD": meta.kD, "k_next": meta.k_next}
        )
        if pe_info:
            self._pe_flags.append({"seg_idx": seg_idx, **pe_info})

        return str(seg_dir)

    # -----------------------------
    # Run-level artifacts
    # -----------------------------
    def _manifest_path(self) -> Path:
        return self.paths.artifacts_dir / "manifest.json"

    def _summary_path(self) -> Path:
        return self.paths.artifacts_dir / "summary.json"

    def _write_manifest(self, initial: bool = False) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest = {
            "created_at_utc": now if initial else None,
            "updated_at_utc": now,
            "run_info": asdict(self.run_info),
            "paths": {
                "root": str(self.paths.root),
                "segments_dir": str(self.paths.segments_dir),
                "artifacts_dir": str(self.paths.artifacts_dir),
                "aggregate_dir": str(self.paths.aggregate_dir),
            },
            "segments_written": sorted(self._segments_written),
            "per_segment_shapes": self._per_seg_shapes,
        }
        with open(self._manifest_path(), "w") as f:
            json.dump(manifest, f, indent=2)

    def finalize_run(
        self,
        write_summary: bool = True,
        extra_summary: Optional[Dict[str, object]] = None,
    ) -> str:
        """Finish the run: write manifest (final) and optional summary with PE checks."""
        if write_summary:
            summary = {
                "model": self.run_info.model,
                "run_id": self.run_info.run_id,
                "n": self.run_info.n,
                "m": self.run_info.m,
                "dt": self.run_info.dt,
                "T": self.run_info.T,
                "L": self.run_info.L,
                "segments": sorted(self._segments_written),
                "num_segments": len(self._segments_written),
                "pe_checks": self._pe_flags,  # list of {seg_idx, rank_H_Xi, pe_required, pe_ok}
            }
            if extra_summary:
                summary.update(extra_summary)
            with open(self._summary_path(), "w") as f:
                json.dump(summary, f, indent=2)

        # touch (update) manifest with final state
        self._write_manifest(initial=False)
        return str(self._summary_path())

    # -----------------------------
    # Optional: aggregated dataset
    # -----------------------------
    def save_aggregate(
        self,
        segments: Optional[Iterable[int]] = None,
        filename: str = "aggregate.npz",
        include_index: bool = True,
    ) -> str:
        """Stack H/H_plus/Xi across segments and save one NPZ."""
        segs = sorted(segments) if segments is not None else sorted(self._segments_written)
        if not segs:
            raise RuntimeError("No segments to aggregate.")
        self.paths.aggregate_dir.mkdir(parents=True, exist_ok=True)

        Hs, Hps, Xis, index = [], [], [], []
        for s in segs:
            seg_dir = self.paths.seg_dir(s)
            Hs.append(np.load(seg_dir / "H.npy"))
            Hps.append(np.load(seg_dir / "H_plus.npy"))
            Xis.append(np.load(seg_dir / "Xi.npy"))
            if include_index:
                index.append({"seg_idx": s, "dir": str(seg_dir)})

        H_all = np.concatenate(Hs, axis=1)
        H_plus_all = np.concatenate(Hps, axis=1)
        Xi_all = np.concatenate(Xis, axis=1)

        out = self.paths.aggregate_dir / filename
        payload = dict(H=H_all, H_plus=H_plus_all, Xi=Xi_all, run_info=np.array([asdict(self.run_info)], dtype=object))
        if include_index:
            payload["index"] = np.array(index, dtype=object)
        np.savez(out, **payload)
        return str(out)