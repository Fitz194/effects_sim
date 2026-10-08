"""Brode blast model against points read from Glasstone & Dolan (1977), Fig. 3.73 (pp. 111-112).

Fig. 3.73a gives ground-level peak overpressure contours for a 1 kt burst versus height of burst
and ground distance; the surface-burst (height 0) intercepts below were read from the printed
figure (reading error about +/-3 ft). The 50 psi point is the book's worked example on p. 112.
The tolerance is 20 %, the book's own stated accuracy for its curves.
"""

import pytest

from effects.sources import make_blast
from effects.units import psi_to_pa

pytestmark = pytest.mark.validation

FT = 0.3048

# (yield_kt, burst_height_ft, ground_distance_ft, book_psi, source)
POINTS = [
    (100.0, 2320.0, 1860.0, 50.0, "Fig 3.73b worked example, p. 112"),
    (1.0, 0.0, 340.0, 100.0, "Fig 3.73a surface intercept, p. 111"),
    (1.0, 0.0, 260.0, 200.0, "Fig 3.73a surface intercept, p. 111"),
    (1.0, 0.0, 192.0, 500.0, "Fig 3.73a surface intercept, p. 111"),
    (1.0, 0.0, 150.0, 1000.0, "Fig 3.73a surface intercept, p. 111"),
]


@pytest.mark.parametrize("y,h,d,psi,src", POINTS, ids=[f"{p[0]:g}kt_{p[3]:g}psi" for p in POINTS])
def test_brode_matches_book(y, h, d, psi, src):
    b = make_blast("brode", y, h * FT)
    got = float(b.overpressure(d * FT)) / psi_to_pa(1.0)
    assert got == pytest.approx(psi, rel=0.20), src
