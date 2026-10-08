"""Thermal model against points read from Glasstone & Dolan (1977), Fig. 7.42 (p. 291).

Fig. 7.42 gives slant ranges for specified radiant exposures versus yield, for air bursts up to
15,000 ft and a 12-mile visibility (burst height 200 W^0.4 ft). Points were read from the figure
(about 3 % reading error). The book's own worked example (Sec. 7.43, p. 290) is also included:
newspaper ignites at 8 cal/cm^2, which for 1 Mt is a slant range of "about 7 miles".
"""

import pytest

from effects.sources import ThermalSource
from effects.units import cal_cm2_to_j_m2

pytestmark = pytest.mark.validation

MILE = 1609.344
V12MI = 12 * MILE

# (yield_kt, cal_cm2, slant_range_miles, source)
POINTS = [
    (1000.0, 8.0, 7.0, "Sec. 7.43 worked example, p. 290 ('about 7 miles')"),
    (1.5, 8.0, 0.42, "Fig. 7.42"),
    (1.5, 50.0, 0.17, "Fig. 7.42"),
    (10.0, 3.0, 1.72, "Fig. 7.42"),
    (10.0, 25.0, 0.60, "Fig. 7.42"),
    (1000.0, 3.0, 9.27, "Fig. 7.42"),
    (1000.0, 12.0, 5.76, "Fig. 7.42"),
    (9000.0, 3.0, 17.55, "Fig. 7.42"),
    (9000.0, 50.0, 7.78, "Fig. 7.42"),
]


@pytest.mark.parametrize("y,q,miles,src", POINTS, ids=[f"{p[0]:g}kt_{p[1]:g}cal" for p in POINTS])
def test_slant_range_matches_figure(y, q, miles, src):
    # Burst height only enters through slant range; use a burst at ground level so ground
    # range equals slant range.
    t = ThermalSource(y, 0.0, 0.35, V12MI)
    got = t.range_for_fluence(cal_cm2_to_j_m2(q)) / MILE
    assert got == pytest.approx(miles, rel=0.15), src
