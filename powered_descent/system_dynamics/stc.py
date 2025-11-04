# system_dynamics/stc.py
from __future__ import annotations

from typing import Any, Callable

# We express cSTC as:
#   h(z) = -min(g(z), 0) * c(z) = 0
# where g is the trigger, c is the constraint (can be equality or inequality via slack).

def cstc_eq(g: Callable[[Any], float],
            c: Callable[[Any], float]) -> Callable[[Any], float]:
    """
    Build h(z) = -min(g(z), 0) * c(z). Root-finding or penalty methods can enforce h(z)=0.
    """
    def h(z: Any) -> float:
        gz = float(g(z))
        cz = float(c(z))
        return -min(gz, 0.0) * cz
    return h

def cstc_with_slack(g: Callable[[Any], float],
                    c_le_zero: Callable[[Any], float]) -> Callable[[Any], float]:
    """
    Inequality version: want c(z) <= 0 only when g(z) < 0.
    Use h(z) = -min(g(z),0) * max(c(z), 0).  (Equivalent to c(z)+s=0, s>=0.)
    """
    def h(z: Any) -> float:
        gz = float(g(z))
        cz = float(c_le_zero(z))
        return -min(gz, 0.0) * max(cz, 0.0)
    return h
