# ddfs/utils/sampling.py

from __future__ import annotations
import numpy as np # pyright: ignore[reportMissingImports]

def rng_box(rng: np.random.Generator, halfwidth: np.ndarray) -> np.ndarray:
    """
    Uniform sample in component-wise box [-h_i, h_i] for each i.
    """
    return (rng.random(halfwidth.shape) * 2.0 - 1.0) * halfwidth

def rng_box_scalar(rng: np.random.Generator, h: float) -> float:
    """
    Uniform sample in box [-h, h].
    """
    return float((rng.random() * 2.0 - 1.0) * h)