"""Literature anchors for the default building fragility library.

These are coarse published statements, not curve fits, so tolerances are loose on purpose.
They guard the packaged default library only; a user-supplied library is not checked here.

Sources and how far each was verified:
  * US OTA (1979), "The Effects of Nuclear War", ch. 2: a typical residence collapses at about
    5 psi; reinforced concrete structures are levelled at about 20 psi. Read directly.
  * About 8 psi destroys wooden and brick houses (Japan damage surveys). Read only as relayed by
    Wikipedia, "Effects of nuclear explosions" (citing the AFSWP Operation CASTLE report).
    The primary source has not been checked.
"""

import pytest

from effects.fragility import load_fragility_library
from effects.units import psi_to_pa

pytestmark = pytest.mark.validation

RESIDENTIAL = ("wood_frame", "masonry")


@pytest.fixture(scope="module")
def lib():
    return load_fragility_library()


def collapse_prob(fs, psi):
    # exceedance() returns (..., n_states) with the last state being collapse
    return float(fs.exceedance(psi_to_pa(psi))[..., -1])


@pytest.mark.parametrize("name", RESIDENTIAL)
def test_typical_residence_collapses_near_5_psi(lib, name):
    """OTA 1979: typical residence collapses at about 5 psi -> about even odds at 5 psi."""
    assert collapse_prob(lib[name], 5.0) == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("name", RESIDENTIAL)
def test_nearly_all_residences_destroyed_by_8_psi(lib, name):
    """About 8 psi destroys wooden and brick houses (Japan surveys, secondary source)."""
    assert collapse_prob(lib[name], 8.0) >= 0.9


@pytest.mark.parametrize("name", RESIDENTIAL)
def test_residences_mostly_stand_at_1_psi(lib, name):
    """1 psi breaks windows and does light damage, not collapse."""
    assert collapse_prob(lib[name], 1.0) < 0.01


def test_heavy_construction_outlasts_houses(lib):
    """OTA Table 3: houses are destroyed at 5 psi, reinforced concrete only near 20 psi."""
    rc = lib["reinforced_concrete"]
    for name in RESIDENTIAL:
        assert collapse_prob(rc, 8.0) < 0.1 < collapse_prob(lib[name], 8.0)


def test_every_class_declares_its_basis(lib):
    """No class may pass as referenced without saying so. See default_fragility.yaml."""
    import yaml
    from importlib import resources

    raw = yaml.safe_load(
        resources.files("effects.fragility").joinpath("default_fragility.yaml").read_text()
    )
    for name, spec in raw["classes"].items():
        assert spec.get("basis") in {"anchored", "placeholder"}, name
        if spec["basis"] == "anchored":
            assert spec.get("reference"), f"{name} is anchored but has no reference"
