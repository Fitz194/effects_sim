"""Air-blast source models.

Free-air TNT blast uses the Kinney & Graham analytic fits:
    G. F. Kinney and K. J. Graham, *Explosive Shocks in Air*, 2nd ed., Springer, 1985.
Scaled distance Z = R / W^(1/3), R in m, W in kg TNT. Valid roughly 0.05 < Z < 500 m/kg^(1/3).

Large (nuclear-scale) sources are mapped onto TNT via cube-root (Hopkinson-Cranz) scaling:
    W_eff = yield_kt * 1e6 kg * blast_fraction * reflection_factor
- blast_fraction ~0.5: share of yield appearing as blast (Glasstone & Dolan 1977, ch. 1).
- reflection_factor ~1.8: hemispherical ground reflection for near-surface bursts
  (the common "yield doubling" approximation, reduced for imperfect reflection).

Known limitations (documented, addressed by the Build 3 solver):
- No Mach-stem / height-of-burst curves; burst height only enters through slant range.
- Flat ground, no terrain or building shielding, sea-level-like ambient (p0 is a parameter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from effects.units import KT_TNT_KG, P0_SEA_LEVEL

Z_MIN, Z_MAX = 0.05, 500.0


def _z(z):
    return np.clip(np.asarray(z, dtype=float), Z_MIN, Z_MAX)


def kg_overpressure(z, p0: float = P0_SEA_LEVEL):
    """Peak incident overpressure (Pa) from Kinney-Graham, scaled distance z (m/kg^1/3)."""
    z = _z(z)
    num = 808.0 * (1.0 + (z / 4.5) ** 2)
    den = (
        np.sqrt(1.0 + (z / 0.048) ** 2)
        * np.sqrt(1.0 + (z / 0.32) ** 2)
        * np.sqrt(1.0 + (z / 1.35) ** 2)
    )
    return p0 * num / den


def kg_impulse(z, w_kg):
    """Positive-phase incident impulse (Pa*s) from Kinney-Graham."""
    z = _z(z)
    per_cbrt = 0.067 * np.sqrt(1.0 + (z / 0.23) ** 4) / (z**2 * np.cbrt(1.0 + (z / 1.55) ** 3))
    return per_cbrt * np.cbrt(w_kg) * 100.0  # bar*ms -> Pa*s


def kg_duration(z, w_kg):
    """Positive-phase duration (s) from Kinney-Graham."""
    z = _z(z)
    per_cbrt = (
        980.0
        * (1.0 + (z / 0.54) ** 10)
        / ((1.0 + (z / 0.02) ** 3) * (1.0 + (z / 0.74) ** 6) * np.sqrt(1.0 + (z / 6.9) ** 2))
    )
    return per_cbrt * np.cbrt(w_kg) * 1e-3


@dataclass(frozen=True)
class NuclearBlast:
    """Blast from a large point-source release, via TNT equivalence."""

    yield_kt: float
    burst_height_m: float = 0.0
    blast_fraction: float = 0.5
    reflection_factor: float = 1.8
    p0: float = P0_SEA_LEVEL

    @property
    def w_eff_kg(self) -> float:
        return self.yield_kt * KT_TNT_KG * self.blast_fraction * self.reflection_factor

    def slant_range(self, ground_range_m):
        return np.hypot(np.asarray(ground_range_m, dtype=float), self.burst_height_m)

    def scaled_distance(self, ground_range_m):
        return self.slant_range(ground_range_m) / np.cbrt(self.w_eff_kg)

    def overpressure(self, ground_range_m):
        return kg_overpressure(self.scaled_distance(ground_range_m), self.p0)

    def impulse(self, ground_range_m):
        return kg_impulse(self.scaled_distance(ground_range_m), self.w_eff_kg)

    def duration(self, ground_range_m):
        return kg_duration(self.scaled_distance(ground_range_m), self.w_eff_kg)

    def range_for_overpressure(self, p_pa: float) -> float:
        """Ground range (m) at which overpressure falls to p_pa; 0 if never reached on ground."""
        if self.overpressure(0.0) < p_pa:
            return 0.0
        hi = 10.0
        while self.overpressure(hi) > p_pa:
            hi *= 2.0
            if hi > 1e8:
                raise ValueError("overpressure threshold outside model range")
        return float(brentq(lambda r: float(self.overpressure(r)) - p_pa, 0.0, hi, xtol=0.1))


def impact_energy_kt(diameter_m: float, density_kg_m3: float, velocity_m_s: float) -> float:
    """Kinetic energy of a spherical impactor, in kt TNT equivalent.

    Converter only: atmospheric entry, airburst altitude and cratering are not modelled
    (see Collins, Melosh & Marcus 2005 for a full treatment).
    """
    from effects.units import KT_JOULES

    mass = density_kg_m3 * np.pi * diameter_m**3 / 6.0
    return 0.5 * mass * velocity_m_s**2 / KT_JOULES
