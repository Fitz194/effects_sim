"""Checks that the default fragility and casualty parameters match the primary source.

Source: S. Glasstone and P. J. Dolan, *The Effects of Nuclear Weapons*, 3rd ed. (1977). Page
numbers are the book's printed pages. These tests guard the packaged defaults only; a
user-supplied library is not checked. Tolerances follow the book's own stated accuracy of
+/-20 % in distance (Sec. 5.142, p. 220), so they are deliberately loose.
"""

from importlib import resources

import numpy as np
import pytest
import yaml

from effects.fragility import CasualtyModel, load_fragility_library
from effects.sources import make_blast
from effects.units import cal_cm2_to_j_m2, psi_to_pa

pytestmark = pytest.mark.validation

FT = 0.3048
DIFFRACTION = ("wood_frame", "masonry")
DRAG = ("reinforced_concrete", "steel_frame", "light_industrial")


@pytest.fixture(scope="module")
def lib():
    return load_fragility_library()


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(
        resources.files("effects.fragility").joinpath("default_fragility.yaml").read_text()
    )


def exceed(fs, state, psi):
    idx = fs.states.index(state)
    return float(fs.exceedance(psi_to_pa(psi))[..., idx])


def optimum_hob_range_ft(p_pa, yield_kt):
    """Largest ground range (ft) at which p_pa is reached, over burst heights."""
    best = 0.0
    for h in np.linspace(0.0, 4000.0 * (yield_kt / 1000.0) ** (1 / 3), 81):
        best = max(best, make_blast("brode", yield_kt, h).range_for_overpressure(p_pa))
    return best / FT


def test_every_class_declares_its_basis(lib, raw):
    for name, spec in raw["classes"].items():
        assert spec.get("basis") in {"derived", "placeholder"}, name
        if spec["basis"] == "derived":
            assert spec.get("reference"), f"{name} is derived but has no reference"


def test_drag_sensitive_classes_are_flagged(lib):
    for name in DRAG:
        assert lib[name].drag_sensitive and lib[name].reference_yield_kt, name
    for name in DIFFRACTION:
        assert not lib[name].drag_sensitive, name


@pytest.mark.parametrize("name", DIFFRACTION + DRAG)
def test_light_damage_is_one_psi(lib, name):
    """Sec. 5.143 (p. 220): light damage at 1 psi for all but blast-resistant structures."""
    assert exceed(lib[name], "light", 1.0) == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize("state,book_ft", [("collapse", 29_000), ("moderate", 33_000)])
def test_wood_frame_reproduces_book_worked_example(lib, state, book_ft):
    """Sec. 5.141 example (p. 218): Type 5, 1 Mt, optimum burst height -> 29,000 ft severe,
    33,000 ft moderate. The packaged medians must give back those distances to within 15 %."""
    median_pa = lib["wood_frame"].curves[lib["wood_frame"].states.index(state)].median
    assert optimum_hob_range_ft(median_pa, 1000.0) == pytest.approx(book_ft, rel=0.15)


def test_class_ordering(lib):
    med = {n: lib[n].curves[-1].median for n in lib}
    assert med["wood_frame"] < med["masonry"] < med["reinforced_concrete"] < med["steel_frame"]
    assert med["light_industrial"] < med["steel_frame"]


def test_houses_collapse_near_book_values(lib):
    """Book (Nagasaki/Nevada, Sec. 5.140 data and Fig. 5.140): wood houses about 3-3.5 psi,
    brick about 5 psi for severe damage."""
    assert exceed(lib["wood_frame"], "collapse", 3.5) == pytest.approx(0.5, abs=0.1)
    assert exceed(lib["masonry"], "collapse", 5.0) == pytest.approx(0.5, abs=0.1)
    assert exceed(lib["wood_frame"], "collapse", 8.0) > 0.99
    assert exceed(lib["wood_frame"], "collapse", 1.0) < 0.01


def test_casualty_indoor_rates_are_table_12_21():
    """Table 12.21 (p. 547): killed 8/14/88 %, serious injury 14/18/11 % (light/moderate/severe)."""
    m = CasualtyModel()
    assert m.indoor_fatal_rate[:4] == (0.0, 0.08, 0.14, 0.88)
    assert m.indoor_injury_rate[:4] == (0.0, 0.14, 0.18, 0.11)


def test_blast_lethality_is_table_12_38():
    """Table 12.38 (p. 551): lethality 50 % at 62 psi; threshold 40 psi, 100 % at 92 psi."""
    c = CasualtyModel().blast_fatal
    assert float(c.prob(psi_to_pa(62.0))) == pytest.approx(0.5, abs=1e-3)
    assert float(c.prob(psi_to_pa(40.0))) == pytest.approx(0.01, abs=0.01)
    assert float(c.prob(psi_to_pa(92.0))) == pytest.approx(0.99, abs=0.01)


@pytest.mark.parametrize(
    "yield_kt,second,third",
    [(1.0, 4.0, 6.0), (10.0, 4.7, 7.2), (100.0, 5.4, 8.4), (10_000.0, 6.85, 10.85)],
)
def test_burn_medians_match_figure_12_65(yield_kt, second, third):
    """Fig. 12.65 (p. 565): 50 % second- and third-degree burn lines, read at four yields."""
    m = CasualtyModel()
    assert m.burn_injury.median_j_m2(yield_kt) == pytest.approx(cal_cm2_to_j_m2(second), rel=0.04)
    assert m.burn_fatal.median_j_m2(yield_kt) == pytest.approx(cal_cm2_to_j_m2(third), rel=0.04)
