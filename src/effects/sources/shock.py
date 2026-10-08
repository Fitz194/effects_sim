"""Shock-front kinematics for visualisation.

Shock speed from the Rankine-Hugoniot relation for an ideal gas:
    U = c0 * sqrt(1 + (gamma + 1) / (2 gamma) * dp / p0)

Ground arrival time is integrated along the ground using the local peak overpressure:
    t(r) = t_gz + integral_0^r dr' / U(dp(r'))
with t_gz = h / U(dp(0)) for an air burst. This is an approximation (it ignores the curved
path of the incident/Mach front near ground zero) and is used for animation, not for damage.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

C0_SEA_LEVEL = 340.3  # m/s
GAMMA_AIR = 1.4


def shock_speed(dp, p0: float = 101_325.0, c0: float = C0_SEA_LEVEL, gamma: float = GAMMA_AIR):
    dp = np.maximum(np.asarray(dp, dtype=float), 0.0)
    return c0 * np.sqrt(1.0 + (gamma + 1.0) / (2.0 * gamma) * dp / p0)


def arrival_time_table(blast, r_max: float, n: int = 4000, p0: float = 101_325.0):
    """(ranges, times) arrays for ground arrival time, suitable for np.interp."""
    r = np.linspace(0.0, r_max, n)
    u = shock_speed(blast.overpressure(r), p0)
    t_gz = getattr(blast, "burst_height_m", 0.0) / u[0]
    t = t_gz + cumulative_trapezoid(1.0 / u, r, initial=0.0)
    return r, t
