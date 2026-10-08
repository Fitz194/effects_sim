"""Comparison against published reference points.

Add rows to tests/validation/data/reference_points.csv, transcribed from a primary source
(e.g. Glasstone & Dolan 1977 figures/tables for surface bursts, or UFC 3-340-02 free-air curves).
Each row gives a yield, burst height, overpressure and the published ground range, with a
tolerance. Rows with an empty `source` column are ignored so placeholders never pass silently.
"""

from pathlib import Path

import pandas as pd
import pytest

from effects.sources import NuclearBlast
from effects.units import psi_to_pa

pytestmark = pytest.mark.validation

DATA = Path(__file__).parent / "data" / "reference_points.csv"


def _rows():
    if not DATA.exists():
        return []
    df = pd.read_csv(DATA, comment="#").dropna(subset=["source"])
    return [r for _, r in df.iterrows()]


ROWS = _rows()


@pytest.mark.skipif(not ROWS, reason="no published reference points entered yet")
@pytest.mark.parametrize("row", ROWS, ids=lambda r: f"{r.yield_kt}kt_{r.overpressure_psi}psi")
def test_range_matches_reference(row):
    b = NuclearBlast(row.yield_kt, row.burst_height_m)
    r = b.range_for_overpressure(psi_to_pa(row.overpressure_psi))
    assert r == pytest.approx(row.ground_range_m, rel=row.rel_tol), row.source
