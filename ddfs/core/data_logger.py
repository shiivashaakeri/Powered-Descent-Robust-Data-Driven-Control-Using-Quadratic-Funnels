# ddfs/core/data_logger.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from multiprocessing import Value
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np  # pyright: ignore[reportMissingImports]

@dataclass
class WindowSpec:
    """
    Data window spec for segment i.
    T_i^D = [kD, k_next) with length L = k_next - kD.
    We must also log eta(k_next) to build H_i^+.
    """
    seg_idx: int
    kD: int                    # start index of data window
    k_next: int                # first index AFTER the window (segment boundary)
    L: int                     # length of the data window
    dt: float                  # time step (seconds)
    model: str                 # model name
    note: str = ""              # optional note

@dataclass
class WindowMeta:
    """ Returned alongside the matrices for bookkeeping. """
    seg_idx: int
    kD: int
    k_next: int
    L: int
    n: int
    m: int
    dt: float
    model: str
    note: str = ""

class DataWindowLogger:
    """ Assembles the deviation data matrices for one segment window:

        H_i      = [ eta(kD) ... eta(k_next-1) ]           ∈ R^{nxL}
        H_i^+    = [ eta(kD+1) ... eta(k_next) ]           ∈ R^{nxL}
        Xi_i     = [  xi(kD) ...  xi(k_next-1) ]           ∈ R^{mxL}

    Usage:
        logger = DataWindowLogger(n, m)
        logger.begin(window_spec, K_i=K)
        for k in range(kD, k_next):
            logger.log_transition(k, eta_k, xi_k, eta_kplus1, extras=...)
        H, H_plus, Xi, meta, extras = logger.finalize()
    """
    def __init__(self, n: int, m: int) -> None:
        self.n = int(n)
        self.m = int(m)
        self._spec: Optional[WindowSpec] = None
        self._H: Optional[np.ndarray] = None                # (n, L) (eta(k...k+L-1))
        self._H_plus: Optional[np.ndarray] = None           # (n, L) (eta(k+1...k+L))
        self._Xi: Optional[np.ndarray] = None               # (m, L) (xi(k...k+L-1))
        self._idx: int = 0       
        self._K_i: Optional[int] = None                    # (m, n) gain for this segment
        self._extras_cols: Dict[str, int] = {}             # column offsets for extras dict
        self._extras_tail: Dict[str, List] = {}

    # ----------------------
    # Lifescycle
    # ----------------------
    def begin(self, spec: WindowSpec, K_i: Optional[np.ndarray] = None) -> None:
        """ Initialize buffers for a new window. """
        if spec.L <= 0 or spec.k_next - spec.kD != spec.L:
            raise ValueError("Invalid window spec: L must be equal k_next - kD and be positive.")

        self._spec = spec
        self._H = np.zeros((self.n, spec.L), dtype=float)
        self._H_plus = np.zeros((self.n, spec.L), dtype=float)
        self._Xi = np.zeros((self.m, spec.L), dtype=float)
        self._idx = 0
        self._K_i = None if K_i is None else np.asarray(K_i, dtype=float)
        self._extras_cols.clear()
        self._extras_tail.clear()

    def log_transition(
        self,
        k: int,
        eta_k: np.ndarray,
        xi_k: np.ndarray,
        eta_kplus1: np.ndarray,
        extras: Optional[Dict[str, object]] = None,
        extras_tail: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Log one transition at time k in {kD, ..., k_next-1}

        Notes:
         - We write eta_k to column idx of H, eta_{k+1} to column idx of H_plus, and xi_k to column idx of Xi.
         - 'extras' are per-column (aligned to k...k_next-1).
         - 'extras_tail' if supplied at the last call, is aligned to k_next.
        """
        if self._spec is None or self._H is None or self._H_plus is None or self._Xi is None:
            raise RuntimeError("Logger not initialized. Call begin() first.")

        # bounds check
        if not (self._spec.kD <= k <= self._spec.k_next - 1):
            raise IndexError(f"k={k} outside window [{self._spec.kD}, {self._spec.k_next - 1}]")
        if self._idx >= self._spec.L:
            raise RuntimeError("All L steps are already logged for this window.")
        
        # shape check + write
        eta_k = np.asarray(eta_k, dtype=float).reshape(self.n)
        xi_k = np.asarray(xi_k, dtype=float).reshape(self.m)
        eta_k1 = np.asarray(eta_kplus1, dtype=float).reshape(self.n)

        self._H[:, self._idx] = eta_k
        self._H_plus[:, self._idx] = eta_k1
        self._Xi[:, self._idx] = xi_k

        # store extras (per-column)
        if extras:
            for key, val in extras.items():
                self._extras_cols.setdefault(key, []).append(val)
        
        # capture tail extras aligned with eta(k_next) on the final call
        if extras_tail:
            # we expect tail data only when this is the last column (idx == L-1)
            if self._idx == self._spec.L - 1:
                for key, val in extras_tail.items():
                    self._extras_tail.setdefault(key, []).append(val)
        
        self._idx += 1
    
    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, WindowMeta, Dict[str, np.ndarray]]:
        """
        Finish window and return (H, H_plus, Xi, meta, extras)

        Returns:
            H           : (n, L)
            H_plus      : (n, L)
            Xi          : (m, L)
            meta        : WindowMeta with size/indices
            extras      : Dict[str, np.ndarray]
        """

        if self._spec is None or self._H is None or self._H_plus is None or self._Xi is None:
            raise RuntimeError("Logger not initialized. Call begin() first.")

        if self._idx != self._spec.L:
            raise RuntimeError(f"Window not complete. logged {self._idx} of {self._spec.L} steps.")
        
        # build meta
        meta = WindowMeta(
            seg_idx=self._spec.seg_idx,
            kD=self._spec.kD,
            k_next=self._spec.k_next,
            L=self._spec.L,
            n=self.n,
            m=self.m,
            dt=self._spec.dt,
            model=self._spec.model,
            note=self._spec.note,
        )

        # materialize extras as arrays (ragged entires are allowed but discouraged)
        extras_np: Dict[str, np.ndarray] = {}
        for k, vlist in self._extras_cols.items():
            try:
                extras_np[k] = np.asarray(vlist)
            except Exception:
                extras_np[k] = np.array(vlist, dtype=object)
        
        # tail extras (aligned to k_next). We suffix with '_tail' for clarity.
        for k, vlist in self._extras_tail.items():
            try:
                extras_np[f'{k}_tail'] = np.asarray(vlist)
            except Exception:
                extras_np[f'{k}_tail'] = np.array(vlist, dtype=object)
        
        if self._K_i is not None:
            extras_np['K_i'] = self._K_i.copy()
        
        return self._H.copy(), self._H_plus.copy(), self._Xi.copy(), meta, extras_np


# -------------------------
# Persistence helpers
# -------------------------
@staticmethod
def save_npz(
    out_npz: Path | str,
    H: np.ndarray,
    H_plus: np.ndarray,
    Xi: np.ndarray,
    meta: WindowMeta,
    extras: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    Save a single window to NPZ (+ JSON-ish meta inside the NPZ)
    """
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(H=H, H_plus=H_plus, Xi=Xi, meta=np.array(asdict(meta), dtype=object))
    if extras:
        for k, v in extras.items():
            payload[k] = v
    np.savez_compressed(out_npz, **payload)
    return str(out_npz)

@staticmethod
def stack_windows(windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Concatenate multiple windows into a single H, H_plus, Xi. """
    Hs, Hps, Xis = zip(*windows) if windows else ([], [], [])
    if not Hs:
        return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))
    H = np.concatenate(Hs, axis=1)
    H_plus = np.concatenate(Hps, axis=1)
    Xi = np.concatenate(Xis, axis=1)
    return H, H_plus, Xi