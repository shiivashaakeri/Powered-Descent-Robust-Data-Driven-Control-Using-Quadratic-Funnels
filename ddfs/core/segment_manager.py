# ddfs/core/segment_manager.py
from __future__ import annotations

from dataclasses import dataclass
from turtle import pen
from typing import Iterable, List, Optional

@dataclass(frozen=True)
class SegmentSpec:
    """
    One segment i with step-range T_i = [k_start, k_end_excl)
    and data window T_i^D = [kD_start, k_end_excl).
    All indices are 0-based; *_excl is exclusive.
    """
    idx: int
    k_start: int
    k_end_excl: int
    kD_start: int
    dt: float

    @property
    def T_slice(self) -> slice:
        """ Slice for the full segment steps. """
        return slice(self.k_start, self.k_end_excl)
    
    def D_slice(self) -> slice:
        """ Slice for the data window steps (used for H_i, Xi_i). """
        return slice(self.kD_start, self.k_end_excl)
    
    @property
    def len_T(self) -> int:
        """ |T_i| = number of steps in the segment. """
        return self.k_end_excl - self.k_start
    
    @property
    def len_D(self) -> int:
        """ |T_i^D| = number of steps in the data window. """
        return self.k_end_excl - self.kD_start
    
    @property
    def t_start(self) -> float:
        """ Start time (seconds) of the segment. """
        return self.k_start * self.dt
    
    @property
    def t_end(self) -> float:
        """ End time (seconds) of the segment. """
        return self.k_end_excl * self.dt
    
    @property
    def tD_start(self) -> float:
        """ Start time (seconds) of the data window. """
        return self.kD_start * self.dt

    @property
    def has_full_data(self) -> bool:
        """ True if |T_i^D| = the requested L. """
        return True

class SegmentTimeline:
    """
    """

    def __init__(
        self,
        dt: float,
        T: int,
        L: int,
        K: int,
        strict_L: bool = False,
    ) -> None:
        if not (isinstance(T, int) and isinstance(L, int) and isinstance(K, int)):
            raise TypeError("T, L, and K must be integers.")
        
        if T <= 0 or L <= 0:
            raise ValueError("T and L must be positive.")
        if L > T:
            raise ValueError("L cannot be greater than T.")
        if K <= 1:
            raise ValueError("K must be at least 2.")
        
        self.dt = float(dt)
        self.T = int(T)
        self.L = int(L)
        self.K = int(K)
        self.strict_L = bool(strict_L)

        self.segments: List[SegmentSpec] = []
        self._build()

    # --------- building --------
    def _build(self) -> None:
        i = 0
        while True:
            k_start = i * self.T
            if k_start >= self.K:
                break
            k_end_excl = min((i+1) * self.T, self.K)
            seg_len = k_end_excl - k_start

            # Data window anchored at end: kD_start = k_end_excl - L, but never before k_start.
            kD_start = max(k_end_excl - self.L, k_start)

            # In strict mode, enforce full-length data window for all FULL segments (length = T)
            if self.strict_L and seg_len == self.T:
                if (k_end_excl - kD_start) != self.L:
                    raise ValueError("Strict-L violation while building segments.")
            
            self.segments.append(SegmentSpec(
                idx=i,
                k_start=k_start,
                k_end_excl=k_end_excl,
                kD_start=kD_start,
                dt=self.dt,
            ))
            i += 1
    
    # --------- queries --------
    def num_segments(self) -> int:
        return len(self.segments)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __iter__(self) -> Iterable[SegmentSpec]:
        return iter(self.segments)
    
    def segment_of_step(self, k: int) -> int:
        """ Return the segment index i such that k ∈ T_i. """
        if not (0 <= k < self.K):
            raise IndexError(f"Step {k} out of range [0, {self.K-1})")
        
        i = k // self.T
        return min(i, len(self.segments) - 1)
    
    def has_full_data(self, i: int) -> bool:
        seg = self.segments[i]
        return (seg.k_end_excl - seg.kD_start) == min(seg.len_T, self.L)
    
    def data_length(self, i: int) -> int:
        """ |T_i^D| for segment i. """
        seg = self.segments[i]
        return seg.len_D
    
    def summary(self) -> str:
        lines = [
            f"[segments] K={self.K} steps, T={self.T}, L={self.L}, dt={self.dt:.3f}s, S={len(self.segments)} segments",
        ]
        for s in self.segments:
            full = "full" if self.has_full_data(s.idx) else "partial"
            lines.append(
                f"  - i={s.idx:2d} : T_i=[{s.k_start},{s.k_end_excl}) "
                f"(len={s.len_T}),  T_i^D=[{s.kD_start},{s.k_end_excl}) "
                f"(len={s.len_D}, full={full})"
            )
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        print(self.summary())

    


