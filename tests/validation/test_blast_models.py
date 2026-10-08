"""Cross-checks between independent published blast models, plus one literature reference point.

1. Kinney-Graham TNT-equivalence (surface burst, blast fraction 0.5, reflection 1.8) vs the
   Brode (1986) nuclear fit at HOB = 0. Measured: within 10% at 5-20 psi, KG ~17% low at
   1-2 psi. Tolerance 20%.
2. Kinney-Graham free-air (blast fraction 0.5, no reflection) vs the DNA 1 kt free-air standard:
   within 30% from 100 m to 5 km.
3. Optimum burst height for the 5 psi radius of a 15 kt burst: ~2,500 ft per Glasstone & Dolan's
   height-of-burst curves, as cited by A. Wellerstein, "The trouble with airbursts",
   Restricted Data blog, 6 Dec 2013 (1,960 ft maximises 5 psi for 7 kt -> cube-root scales to
   ~2,530 ft for 15 kt). Brode should land within 15%.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from effects.sources import BrodeBlast, NuclearBlast, dna_free_air_1kt_pa
from effects.units import psi_to_pa

pytestmark = pytest.mark.validation
FT = 0.3048


@pytest.mark.parametrize("yield_kt", [1, 15, 1000])
@pytest.mark.parametrize("psi", [1, 2, 5, 10, 20])
def test_kg_surface_matches_brode(yield_kt, psi):
    kg = NuclearBlast(yield_kt, 0.0)
    r = kg.range_for_overpressure(psi_to_pa(psi))
    assert BrodeBlast(yield_kt, 0.0).overpressure(r) == pytest.approx(psi_to_pa(psi), rel=0.20)


@pytest.mark.parametrize("r_m", [100, 200, 500, 1000, 2000, 5000])
def test_kg_free_air_matches_dna_standard(r_m):
    kg = NuclearBlast(1.0, 0.0, blast_fraction=0.5, reflection_factor=1.0)
    assert kg.overpressure(r_m) == pytest.approx(dna_free_air_1kt_pa(r_m), rel=0.30)


def test_brode_optimum_hob_5psi_15kt():
    def neg_range(h):
        return -BrodeBlast(15.0, h).range_for_overpressure(psi_to_pa(5))

    opt = minimize_scalar(neg_range, bounds=(100, 1500), method="bounded", options={"xatol": 5})
    assert opt.x / FT == pytest.approx(2530, rel=0.15)


def test_airburst_extends_5psi_beyond_surface_burst():
    surface = BrodeBlast(15.0, 0.0).range_for_overpressure(psi_to_pa(5))
    air = BrodeBlast(15.0, 700.0).range_for_overpressure(psi_to_pa(5))
    assert air > 1.3 * surface


def test_brode_tail_continuous_and_positive():
    b = BrodeBlast(15.0, 600.0)
    r_s = b._tail()[0]
    eps = 0.5
    lo, hi = b.overpressure(r_s - eps), b.overpressure(r_s + eps)
    assert hi == pytest.approx(lo, rel=0.01)
    r = np.linspace(0, 50_000, 5001)
    assert np.all(b.overpressure(r) > 0)
