"""Nuclear-specific airblast fits.

Brode height-of-burst fit
    H. L. Brode, *Airblast From Nuclear Bursts - Analytic Approximations*, Pacific-Sierra
    Research Corp., 1986, pp. 60-71. Peak overpressure (psi) on an ideal surface for a 1 kt burst,
    in scaled kilofeet; cube-root scaled to other yields. Includes Mach-stem enhancement.
    Stated accuracy ~10%, sea-level ambient. Constants transcribed from the MIT-licensed
    `glasstone` library (github.com/GOFAI/glasstone, overpressure.py), which cites the same source.

    The fit loses physical meaning at low overpressure (it goes negative for air bursts below
    roughly 0.1 psi), so below `p_switch` we continue it with the Kinney-Graham TNT-equivalent
    curve, rescaled to match Brode at the switch range. The result is continuous and decays
    physically.

DNA 1 kt free-air standard
    Defense Nuclear Agency standard curve (as reproduced in `glasstone`):
    P(Pa) = 3.04e11/r^3 + 1.13e9/r^2 + 7.9e6 / (r sqrt(ln(r/445.42 + 3 exp(-sqrt(r/445.42)/3))))
    r in metres for 1 kt, sea level. Used here for validation of the TNT-equivalence model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from effects.sources.blast import NuclearBlast
from effects.units import P0_SEA_LEVEL, PSI

FT = 0.3048


def brode_1kt_psi(z, r, y):
    """Brode fit. z = burst height / ground range; r, y = scaled slant range and height (kft)."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        a = 1.22 - (3.908 * z**2) / (1 + 810.2 * z**5)
        b = (
            2.321
            + (6.195 * z**18) / (1 + 1.113 * z**18)
            - (0.03831 * z**17) / (1 + 0.02415 * z**17)
            + 0.6692 / (1 + 4164 * z**8)
        )
        c = 4.153 - (1.149 * z**18) / (1 + 1.641 * z**18) - 1.1 / (1 + 2.771 * z**2.5)
        d = -4.166 + (25.76 * z**1.75) / (1 + 1.382 * z**18) + (8.257 * z) / (1 + 3.219 * z)
        e = 1 - (0.004642 * z**18) / (1 + 0.003886 * z**18)
        f = 0.6096 + (2.879 * z**9.25) / (1 + 2.359 * z**14.5) - (17.5 * z**2) / (1 + 71.66 * z**3)
        g = 1.83 + (5.361 * z**2) / (1 + 0.3139 * z**6)
        h = (
            (8.808 * z**1.5) / (1 + 154.5 * z**3.5)
            - (0.2905 + 64.67 * z**5) / (1 + 441.5 * z**5)
            - (1.389 * z) / (1 + 49.03 * z**5)
            + (1.094 * r**2) / ((781.2 - 123.4 * r + 37.98 * r**1.5 + r**2) * (1 + 2 * y))
        )
        j = (0.000629 * y**4) / (3.493e-9 + y**4) - (2.67 * y**2) / (1 + 1e7 * y**4.3)
        k = 5.18 + (0.2803 * y**3.5) / (3.788e-6 + y**4)
        return 10.47 / r**a + b / r**c + (d * e) / (1 + f * r**g) + h + j / r**k


def dna_free_air_1kt_pa(r_m):
    """DNA 1 kt free-air peak overpressure (Pa) at slant range r_m (m), sea level."""
    r = np.asarray(r_m, dtype=float)
    return (
        3.04e11 / r**3
        + 1.13e9 / r**2
        + 7.9e6 / (r * np.sqrt(np.log(r / 445.42 + 3 * np.exp(-np.sqrt(r / 445.42) / 3))))
    )


def outer_range(fn, p_target: float, r_max: float = 1e7) -> float:
    """Outermost ground range where fn(r) >= p_target (air bursts can be non-monotone near GZ)."""
    r = np.concatenate([[0.0], np.logspace(0, np.log10(r_max), 4000)])
    v = fn(r)
    above = np.nonzero(v >= p_target)[0]
    if len(above) == 0:
        return 0.0
    i = above[-1]
    if i == len(r) - 1:
        raise ValueError("threshold beyond model range")
    return float(brentq(lambda x: float(fn(x)) - p_target, r[i], r[i + 1], xtol=0.1))


@dataclass(frozen=True)
class BrodeBlast:
    """Nuclear airblast with height-of-burst effects (Brode 1986) and a Kinney-Graham tail."""

    yield_kt: float
    burst_height_m: float = 0.0
    p0: float = P0_SEA_LEVEL
    p_switch_pa: float = 2.0 * PSI

    def _brode_pa(self, ground_range_m):
        gr = np.maximum(np.asarray(ground_range_m, dtype=float), 1.0)
        s = np.cbrt(self.yield_kt)
        x = gr / FT / 1000.0 / s
        y = self.burst_height_m / FT / 1000.0 / s
        z = self.burst_height_m / gr
        return brode_1kt_psi(z, np.hypot(x, y), y) * PSI * (self.p0 / P0_SEA_LEVEL)

    def _tail(self):
        """(switch range, KG reference blast, scale) computed once per instance."""
        cached = self.__dict__.get("_tail_cache")
        if cached is None:
            ref = NuclearBlast(self.yield_kt, self.burst_height_m, 0.5, 1.8, self.p0)
            r_s = outer_range(self._brode_pa, self.p_switch_pa, 1e6)
            scale = float(self._brode_pa(r_s) / ref.overpressure(r_s)) if r_s > 0 else 1.0
            cached = (r_s, ref, scale)
            object.__setattr__(self, "_tail_cache", cached)
        return cached

    def overpressure(self, ground_range_m):
        gr = np.asarray(ground_range_m, dtype=float)
        r_s, ref, scale = self._tail()
        return np.where(gr <= r_s, self._brode_pa(gr), ref.overpressure(gr) * scale)

    def range_for_overpressure(self, p_pa: float) -> float:
        return outer_range(self.overpressure, p_pa)
