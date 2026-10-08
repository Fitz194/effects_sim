import numpy as np
import pytest

from effects.sources import BrodeBlast
from effects.sources.shock import C0_SEA_LEVEL, arrival_time_table, shock_speed


def test_weak_shock_travels_at_sound_speed():
    assert shock_speed(0.0) == pytest.approx(C0_SEA_LEVEL)


def test_shock_speed_increases_with_pressure():
    p = np.logspace(2, 7, 50)
    assert np.all(np.diff(shock_speed(p)) > 0)


def test_arrival_time_monotone_and_far_field_slope():
    r, t = arrival_time_table(BrodeBlast(15.0, 0.0), 20_000.0)
    assert np.all(np.diff(t) > 0)
    slope = (t[-1] - t[-200]) / (r[-1] - r[-200])
    assert slope == pytest.approx(1.0 / C0_SEA_LEVEL, rel=0.02)


def test_airburst_arrival_at_ground_zero_is_positive():
    r, t = arrival_time_table(BrodeBlast(15.0, 600.0), 5_000.0)
    assert t[0] > 0
