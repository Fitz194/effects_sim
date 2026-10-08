import numpy as np
import pytest

from effects.fragility import CasualtyModel, LognormalCurve, load_fragility_library


@pytest.fixture(scope="module")
def lib():
    return load_fragility_library()


def test_median_gives_half():
    c = LognormalCurve(20e3, 0.3)
    assert c.prob(20e3) == pytest.approx(0.5)


def test_library_loads_with_expected_states(lib):
    assert "masonry" in lib
    assert lib["masonry"].states == ("light", "moderate", "severe", "collapse")


def test_state_probs_sum_to_one_and_nonnegative(lib):
    p = np.logspace(2, 6, 300)
    for fs in lib.values():
        sp = fs.state_probs(p)
        assert np.allclose(sp.sum(axis=-1), 1.0)
        assert np.all(sp >= -1e-12)


def test_exceedance_monotone_in_severity(lib):
    p = np.logspace(2, 6, 300)
    for fs in lib.values():
        ex = fs.exceedance(p)
        assert np.all(np.diff(ex, axis=-1) <= 1e-12)


def test_median_scale_shifts_curve():
    c = LognormalCurve(20e3, 0.3)
    assert c.prob(40e3, median_scale=2.0) == pytest.approx(0.5)


def test_casualty_rates_bounded(lib):
    m = CasualtyModel()
    p = np.logspace(2, 6, 100)
    q = np.logspace(2, 7, 100)
    sp = lib["masonry"].state_probs(p)
    f, i = m.rates(sp, p, q, 0.85)
    assert np.all((f >= 0) & (f <= 1))
    assert np.all((i >= 0) & (f + i <= 1 + 1e-12))


def test_no_effect_no_casualties(lib):
    m = CasualtyModel()
    sp = lib["masonry"].state_probs(np.array([1.0]))
    f, i = m.rates(sp, np.array([1.0]), np.array([1.0]), 0.85)
    assert f[0] == pytest.approx(0.0, abs=1e-9)
    assert i[0] == pytest.approx(0.0, abs=1e-9)
