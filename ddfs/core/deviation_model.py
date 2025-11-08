# ddfs/core/deviation_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np # type: ignore

QuatMode = Literal["tangent", "component"]

# ----------------------
# small math helpers
# ----------------------
def _wrap_to_pi(a: float | np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return (a + np.pi) % (2 * np.pi) - np.pi

def _q_norm(q: np.ndarray) -> float:
    return float(np.linalg.norm(q))

def _q_conj(q: np.ndarray) -> np.ndarray:
    # scalar-first [w, x, y, z] -> [w, -x, -y, -z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def _q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)

def _q_unit(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def _q_align_hemisphere(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    # flip sign to keep dot >= 0
    return q if float(np.dot(q, q_ref)) >= 0.0 else -q

def _quat_to_rotvec(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Log map: unit quaternion -> R^3 rotation vector r = theta * u
    q = [w, v], theta in [0, pi], u = v / ||v||.
    for small angles, use first order approximation r ~ q - 1/2 * q x q.

    inputs: q: unit quaternion [w, x, y, z]
    outputs: r: rotation vector [theta, x, y, z]
    """
    q = _q_unit(q)
    w = float(q[0])
    v = q[1:4]
    nv = float(np.linalg.norm(v))
    # clamp numerical drift
    w = np.clip(w, -1.0, 1.0)

    if nv < 1e-10:
        # small-angle: theta ~ nv, use theta/nv as scaling
        return 2.0 * v
    theta = 2.0 * np.arctan2(nv, w if w > 0.0 else -w)
    theta = float(theta)
    u = v / nv
    return theta * u

def _rotvec_to_quat(r: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    """
    th = float(np.linalg.norm(r))
    if th < 1e-10:
        # small-angle: cos(th/2) ~ 1 - th^2/8, sin(th/2) ~ th/2
        half = 0.5 * th
        w = 1.0 - (half**2) / 2.0
        v = (0.5 * r) if th > eps else 0.5 * r
        return _q_unit(np.array([w, v[0], v[1], v[2]], dtype=float))
    u = r / th
    w = np.cos(0.5 * th)
    s = np.sin(0.5 * th)
    return _q_unit(np.array([w, u[0] * s, u[1] * s, u[2] * s], dtype=float))

# ----------------------
# main deviation class
# ----------------------

@dataclass
class DeviationModel:
    """
    Maps actual (x,u) and nominal (x_nom, u_nom) to deviation (eta, xi)
    and back, with special handling for attitude (quaternions).

    model_name:
        "rocket2d" (state n_x=6, input n_u=2) or "rocket6dof" (state n_x=14, input n_u=3)
    
    quat_mode:
        - "tangent" (default): 6-DoF attitude deviation is a 3-vector rotation vector from q_nom to q (n_eta=13)
        - "component": raw 4D quaternion difference (n_eta=14)
    
    angle_wrap_2d:
        Wraps theta-differences for rocket2d to (-pi, pi].
    """
    model_name: Literal["rocket2d", "rocket6dof"]
    X_nom: np.ndarray # (n_x, K)
    U_nom: np.ndarray # (n_u, K)
    quat_mode: QuatMode = "tangent"
    angle_wrap_2d: bool = True

    def __post_init__(self) -> None:
        self.X_nom = np.asarray(self.X_nom, dtype=float)
        self.U_nom = np.asarray(self.U_nom, dtype=float)

        if self.model_name == "rocket2d":
            self.n_x, self.n_u = 6, 2
            self.theta_idx = 4
            self.quat_slice = None
        elif self.model_name == "rocket6dof":
            self.n_x, self.n_u = 14, 3
            self.quat_slice = slice(7, 11)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        if self.model_name == "rocket2d" and self.quat_mode == "tangent":
            self.n_eta = self.n_x - 1 # replace 4D quat by 3D rotvec
        else:
            self.n_eta = self.n_x
        
        # basic shape checks
        if self.X_nom.shape[0] != self.n_x:
            raise ValueError(f"X_nom has wrong shape: {self.X_nom.shape}, expected ({self.n_x}, K)")
        if self.U_nom.shape[0] != self.n_u:
            raise ValueError(f"U_nom has wrong shape: {self.U_nom.shape}, expected ({self.n_u}, K)")
    
    # ------- single step mapping -------
    def eta_k(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Deviation eta at step k from actual x and nominal X_nom[:, k]
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        x_nom = self.X_nom[:, k].reshape(-1)

        if self.model_name == "rocket2d":
            e = x - x_nom
            if self.angle_wrap_2d:
                e[self.theta_idx] = float(_wrap_to_pi(x[self.theta_idx] - x_nom[self.theta_idx]))
            return e
        
        if self.quat_mode == "component":
            e = x - x_nom
            q = _q_align_hemisphere(_q_unit(x[self.quat_slice]), _q_unit(x_nom[self.quat_slice]))
            e[self.quat_slice] = q - x_nom[self.quat_slice]
            return e
        
        # tangent mode
        e = np.empty((self.n_eta,), dtype=float)
        e[0:7] = x[0:7] - x_nom[0:7]
        q = _q_unit(x[self.quat_slice])
        q_nom = _q_unit(self.X_nom[self.quat_slice, k])
        q = _q_align_hemisphere(q, q_nom)
        q_rel = _q_mul(q, _q_conj(q_nom))
        e[7:10] = _quat_to_rotvec(q_rel)
        e[10:13] = x[11:14] - x_nom[11:14]
        return e
    
    def xi_k(self, u: np.ndarray, k: int) -> np.ndarray:
        """
        Deviation xi at step k from actual u and nominal U_nom[:, k]
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        return u - self.U_nom[:, k].reshape(-1)
    
    # ------- batch trajectory mapping -------
    def eta_traj(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape != self.X_nom.shape:
            raise ValueError(f"X has wrong shape: {X.shape}, expected {self.X_nom.shape}")
        K = X.shape[1]
        E = np.zeros((self.n_eta, K), dtype=float)
        for k in range(K):
            E[:, k] = self.eta_k(X[:, k], k)
        return E
    
    def xi_traj(self, U: np.ndarray) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        if U.shape != self.U_nom.shape:
            raise ValueError(f"U has wrong shape: {U.shape}, expected {self.U_nom.shape}")
        return U - self.U_nom
    
    # -------- inverse mapping -------
    def x_from_eta_k(self, eta: np.ndarray, k: int) -> np.ndarray:
        """
        Compose actual k from eta and nominal at step k
        """
        x_nom = self.X_nom[:, k].copy()
        eta = np.asarray(eta, dtype=float).reshape(-1)

        if self.model_name == "rocket2d":
            x = x_nom.copy()
            x += eta
            if self.angle_wrap_2d:
                x[self.theta_idx] = float(_wrap_to_pi(x[self.theta_idx]))
            return x
        
        if self.quat_mode == "component":
            x = x_nom.copy()
            x += eta
            x[self.quat_slice] = _q_unit(x[self.quat_slice])
            return x
        
        # tangent mode
        x = x_nom.copy()
        # m, r, v
        x[0:7] = x_nom[0:7] + eta[0:7]
        # quat
        q_nom = _q_unit(x_nom[self.quat_slice])
        r = eta[7:10]
        dq = _rotvec_to_quat(r)
        q = _q_mul(dq, q_nom)
        x[self.quat_slice] = _q_unit(q)
        # body rates
        x[11:14] = x_nom[11:14] + eta[10:13]
        return x
    
    def u_from_xi_k(self, xi: np.ndarray, k: int) -> np.ndarray:
        return self.U_nom[:, k] + np.asarray(xi, dtype=float).reshape(-1)
    
    # --------- shape ---------
    @property
    def n(self) -> int:
        """ Dimensions of eta """
        return self.n_eta
    
    @property
    def m(self) -> int:
        """ Dimensions of xi """
        return self.n_u
    
    # -------- utilities --------
    def update_nominal(self, X_nom: np.ndarray, U_nom: np.ndarray) -> None:
        X_nom = np.asarray(X_nom, dtype=float)
        U_nom = np.asarray(U_nom, dtype=float)
        if X_nom.shape[0] != self.n_x or U_nom.shape[0] != self.n_u:
            raise ValueError("Nominal shapes do not match model dimensions.")
        self.X_nom = X_nom
        self.U_nom = U_nom
    
    def slices(self) -> Tuple[Optional[slice], Optional[int]]:
        """
        Returns (quat_slice, theta_idx) for convenience in callers.
        """
        return self.quat_slice, getattr(self, "theta_idx", None)

# ----------------------
# module-level convenience
# ----------------------

def build_deviation_model(
    model_name: Literal["rocket2d", "rocket6dof"],
    X_nom: np.ndarray,
    U_nom: np.ndarray,
    quat_mode: QuatMode = "tangent",
    angle_wrap_2d: bool = True
) -> DeviationModel:
    """
    Shortcut to build a DeviationModel with defaults.
    """
    return DeviationModel(
        model_name=model_name,
        X_nom=X_nom,
        U_nom=U_nom,
        quat_mode=quat_mode,
        angle_wrap_2d=angle_wrap_2d
    )