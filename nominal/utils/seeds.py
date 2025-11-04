# nominal/utils/seeds.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np  # pyright: ignore[reportMissingImports]


def set_seed(seed: Optional[int]) -> int:
    """
    Seed Python, NumPy (and returns the used seed).
    """
    if seed is None:
        # derive a seed from environment entropy but keep it printable
        seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return int(seed)
