"""Thermal radiation fluence from a point source.

    Q(R) = f_th * Y * tau(R) / (4 pi R^2)

f_th: thermal partition (~0.35 for air bursts, lower for surface bursts; Glasstone & Dolan ch. 7).
tau:  atmospheric transmittance, here a single-parameter exponential exp(-R / L) with L set from
      visibility. This is deliberately simple: real transmittance includes scattered-light buildup
      and depends on wavelength and burst height. Treat thermal results as order-of-magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from effects.units import KT_JOULES


@dataclass(frozen=True)
class ThermalSource:
    yield_kt: float
    burst_height_m: float = 0.0
    thermal_fraction: float = 0.35
    visibility_m: float = 20_000.0
    attenuation_per_visibility: float = 1.0  # L = visibility / this

    def slant_range(self, ground_range_m):
        return np.hypot(np.asarray(ground_range_m, dtype=float), self.burst_height_m)

    def transmittance(self, slant_m):
        length = self.visibility_m / self.attenuation_per_visibility
        return np.exp(-np.asarray(slant_m) / length)

    def fluence(self, ground_range_m):
        """Fluence at the target (J/m^2), unshielded line of sight."""
        r = np.maximum(self.slant_range(ground_range_m), 1.0)
        energy = self.thermal_fraction * self.yield_kt * KT_JOULES
        return energy * self.transmittance(r) / (4.0 * np.pi * r**2)

    def range_for_fluence(self, q_j_m2: float) -> float:
        if self.fluence(0.0) < q_j_m2:
            return 0.0
        hi = 10.0
        while self.fluence(hi) > q_j_m2:
            hi *= 2.0
        return float(brentq(lambda r: self.fluence(r) - q_j_m2, 0.0, hi, xtol=0.1))
