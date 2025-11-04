# nominal/utils/formatting.py
from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]


def format_line(name: str, value, unit: str = "") -> str:
    """
    Formats a line like:
        {Name:}           {value}{unit}
    """
    name = f"{name}:"
    v = f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
    return f"{name.ljust(36)}{v}{unit}"


def hr(char: str = "-", n: int = 54) -> str:
    return char * n
