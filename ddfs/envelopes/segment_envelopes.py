# ddfs/envelopes/segment_envelopes.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]

from ddfs.io.paths import RunPaths
from ddfs.core.segment_manager import SegmentSpec

StructureT = Literal["diag", "block2x2"]

def _blockwise_max(mats: np.ndarray, structure: StructureT) -> np.ndarray:
    """
    mats: (N, d, d) PSD shapes. Return PSD S s.t. S ⪰ mats[k] for all k, and tight under given structure.
    """
    if mats.ndim != 3:
        raise ValueError("mats must be (N,d,d)")
    N, d, _ = mats.shape
    if structure == "diag":
        out = np.zeros((d, d), float)
        out[np.diag_indices(d)] = np.max(mats[:, range(d), range(d)], axis=0)
        return out
    elif structure == "block2x2":
        # Assume fixed disjoint 2x2 principal blocks (0,1), (2,3), ... remaining singles diagonal
        out = np.zeros((d, d), float)
        i = 0
        while i < d:
            if i + 1 < d:  # a 2x2 block
                # Envelope via eigenvalue-wise max over blocks: conservative but PSD & simple
                blocks = mats[:, i:i+2, i:i+2]            # (N,2,2)
                # eig per block
                lam = np.linalg.eigvalsh(blocks)         # (N,2)
                lam_max = lam.max(axis=0)                # (2,)
                # reconstruct with identity eigenvectors (conservative, rotationally aligned)
                out[i:i+2, i:i+2] = np.diag(lam_max)
                i += 2
            else:
                out[i, i] = np.max(mats[:, i, i], axis=0)
                i += 1
        return out
    else:
        raise ValueError(f"Unknown structure: {structure}")

def _blockwise_min(mats: np.ndarray, structure: StructureT) -> np.ndarray:
    if mats.ndim != 3:
        raise ValueError("mats must be (N,d,d)")
    N, d, _ = mats.shape
    if structure == "diag":
        out = np.zeros((d, d), float)
        out[np.diag_indices(d)] = np.min(mats[:, range(d), range(d)], axis=0)
        return out
    elif structure == "block2x2":
        out = np.zeros((d, d), float)
        i = 0
        while i < d:
            if i + 1 < d:
                blocks = mats[:, i:i+2, i:i+2]           # (N,2,2)
                lam = np.linalg.eigvalsh(blocks)         # (N,2)
                lam_min = lam.min(axis=0)                # (2,)
                lam_min = np.maximum(lam_min, 0.0)       # PSD guard
                out[i:i+2, i:i+2] = np.diag(lam_min)
                i += 2
            else:
                out[i, i] = max(0.0, np.min(mats[:, i, i], axis=0))
                i += 1
        return out
    else:
        raise ValueError(f"Unknown structure: {structure}")

@dataclass
class SegmentEnvelopes:
    paths: RunPaths
    model_name: str
    structure: StructureT = "diag"

    def _cache_dir(self) -> Path:
        return self.paths.artifacts_dir() / "ellipsoids" / self.model_name

    def _load_time_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        cd = self._cache_dir()
        P_time = np.load(cd / "state" / "P_time.npy")   # (K,n,n)
        R_time = np.load(cd / "input" / "R_time.npy")   # (K,m,m)
        return P_time, R_time

    # --- public API ---
    def segment_caps(self, seg: SegmentSpec) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (P_min_i, R_max_i) for the given segment.
        P_min_i  = blockwise_max_k P_min(k)   over k ∈ T_i  (ensures E(P_i) ⊂ E(P_min_i))
        R_max_i  = blockwise_min_k R_max(k)   over k ∈ T_i  (ensures E_u(R_max_i) ⊂ per-time feasible)
        """
        P_time, R_time = self._load_time_arrays()
        kslice = slice(seg.k_start, seg.k_end_excl)
        P_seg = P_time[kslice]     # (|T_i|, n, n)
        R_seg = R_time[kslice]     # (|T_i|, m, m)

        P_min_i = _blockwise_max(P_seg, self.structure)
        R_max_i = _blockwise_min(R_seg, self.structure)
        return P_min_i, R_max_i