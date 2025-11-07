# ddfs/io/paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RUNS_DIRNAME = "data_runs"

@dataclass(frozen=True)
class RunPaths:
    base: Path           # repo base or where you want to store runs
    model: str           # e.g., "rocket2d" | "rocket6dof"
    run_id: str          # e.g., "2025-11-07T12-34-56Z-seed42"

    @property
    def root(self) -> Path:
        # {base}/nominal_trajectories/{model}/data_runs/{run_id}/
        return self.base / "nominal_trajectories" / self.model / DEFAULT_RUNS_DIRNAME / self.run_id

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def aggregate_dir(self) -> Path:
        return self.root / "aggregate"

    def seg_dir(self, seg_idx: int) -> Path:
        return self.segments_dir / f"seg_{seg_idx:04d}"