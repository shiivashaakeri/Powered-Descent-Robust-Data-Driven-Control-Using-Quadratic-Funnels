# ddfs/utils/quat.py

from __future__ import annotations
import numpy as np # pyright: ignore[reportMissingImports]

try:
    from models.frames import euler_to_quat
except Exception:
    euler_to_quat = None

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions.
    shapes: q1 (4,), q2 (4,), out (4,)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)

def quat_from_euler_xyz(rad_xyz: np.ndarray) -> np.ndarray:
    """
    XYZ intrinsic rotations in radians → quaternion (w,x,y,z).
    """
    if euler_to_quat is not None:
        q = euler_to_quat(tuple(map(float, rad_xyz)))
        return np.array(q, dtype=float)
    
    cr, sr = np.cos(rad_xyz[0] * 0.5), np.sin(rad_xyz[0] * 0.5)
    cp, sp = np.cos(rad_xyz[1] * 0.5), np.sin(rad_xyz[1] * 0.5)
    cy, sy = np.cos(rad_xyz[2] * 0.5), np.sin(rad_xyz[2] * 0.5)
    w = cr*cp*cy - sr*sp*sy
    x = sr*cp*cy + cr*sp*sy
    y = cr*sp*cy - sr*cp*sy
    z = cr*cp*sy + sr*sp*cy
    return np.array([w, x, y, z], dtype=float)