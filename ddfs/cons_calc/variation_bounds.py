# ddfs/cons_calc/variation_bounds.py

from dataclasses import dataclass

@dataclass(frozen=True)
class VariationConstants:
    L_J: float
    v: float
    C: float  # C = L_J * v

def build_variation_constants(L_J: float, v: float) -> VariationConstants:
    C = float(L_J) * float(v)
    return VariationConstants(L_J=float(L_J), v=float(v), C=C)

def delta_ball_radius(C: float, tilde_T_i: int) -> float:
    """‖[ΔA ΔB]‖₂ ≤ C · ẐT_i ⇒ return the radius C·ẐT_i."""
    return float(C) * float(tilde_T_i)