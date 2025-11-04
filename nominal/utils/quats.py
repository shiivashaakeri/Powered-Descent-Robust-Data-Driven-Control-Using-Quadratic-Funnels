# nominal/utils/quats.py
from __future__ import annotations

import numpy as np  # pyright: ignore[reportMissingImports]


def euler_to_quat(deg_xyz: tuple[float, float, float]) -> np.ndarray:
    """XYZ intrinsic rotations in degrees → quaternion (w, x, y, z)."""
    a = np.deg2rad(np.array(deg_xyz, dtype=float))
    cx, sx = np.cos(a[0] * 0.5), np.sin(a[0] * 0.5)
    cy, sy = np.cos(a[1] * 0.5), np.sin(a[1] * 0.5)
    cz, sz = np.cos(a[2] * 0.5), np.sin(a[2] * 0.5)

    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx
    return np.array([w, x, y, z], dtype=float)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    return q / (n if n > 0 else 1.0)


def dcm_from_quat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) → DCM C_BI (body→inertial)."""
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
            [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_from_dcm(C: np.ndarray) -> np.ndarray:
    """DCM → quaternion (w,x,y,z) with a numerically stable branch."""
    C = np.asarray(C, dtype=float).reshape(3, 3)
    tr = np.trace(C)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (C[2, 1] - C[1, 2]) / s
        y = (C[0, 2] - C[2, 0]) / s
        z = (C[1, 0] - C[0, 1]) / s
    else:
        i = np.argmax([C[0, 0], C[1, 1], C[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2
            w = (C[2, 1] - C[1, 2]) / s
            x = 0.25 * s
            y = (C[0, 1] + C[1, 0]) / s
            z = (C[0, 2] + C[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - C[0, 0] + C[1, 1] - C[2, 2]) * 2
            w = (C[0, 2] - C[2, 0]) / s
            x = (C[0, 1] + C[1, 0]) / s
            y = 0.25 * s
            z = (C[1, 2] + C[2, 1]) / s
        else:
            s = np.sqrt(1.0 - C[0, 0] - C[1, 1] + C[2, 2]) * 2
            w = (C[1, 0] - C[0, 1]) / s
            x = (C[0, 2] + C[2, 0]) / s
            y = (C[1, 2] + C[2, 1]) / s
            z = 0.25 * s
    return normalize_quat(np.array([w, x, y, z], dtype=float))
