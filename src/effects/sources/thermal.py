"""Thermal radiation fluence from a point source.

    Q(R) = f_th * Y * tau(R) / (4 pi R^2)

f_th: thermal partition, 0.35 for air bursts (Glasstone & Dolan 1977, ch. 7; lower for surface
      bursts).
tau:  atmospheric transmittance, a single exponential exp(-R / L) with L = visibility / 2.
      DERIVED: L = V/2 was fitted (RMS error 4 % in range, worst case 12 %) to 22 points read from
      Fig. 7.42 (p. 291), slant range of 3-50 cal/cm^2 versus yield for 1.5 kt - 9 Mt, 12-mile
      visibility, burst height 200 W^0.4 ft. The same fit with f_th free gives f_th = 0.33,
      L = 11 km, consistent with 0.35. ASSUMED: L scales linearly with visibility (the book's
      Sec. 7.93 ff. procedures are not implemented). The exponential absorbs scattered-light
      buildup, so it is not a physical transmittance and is not valid far outside 0.2-20 miles or
      for burst heights well above 200 W^0.4 ft.
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
    attenuation_per_visibility: float = 2.0  # L = visibility / this, fitted to Fig. 7.42

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
