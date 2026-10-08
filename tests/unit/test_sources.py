import numpy as np
import pytest

from effects.sources import NuclearBlast, ThermalSource, impact_energy_kt, kg_overpressure
from effects.sources.blast import kg_duration, kg_impulse
from effects.units import KT_JOULES, psi_to_pa


def test_kg_overpressure_monotonic_decreasing():
    z = np.logspace(-1, 2, 200)
    p = kg_overpressure(z)
    assert np.all(np.diff(p) < 0)


def test_kg_far_field_tends_to_acoustic_decay():
    # At large Z the fit -> ~1/Z (times a slowly varying factor): check slope approx -1.1..-1.4
    z = np.array([100.0, 200.0])
    p = kg_overpressure(z)
    slope = np.log(p[1] / p[0]) / np.log(2.0)
    assert -1.4 < slope < -1.0


def test_kg_impulse_and_duration_positive():
    z = np.logspace(-1, 2, 50)
    assert np.all(kg_impulse(z, 1000.0) > 0)
    assert np.all(kg_duration(z, 1000.0) > 0)


def test_cube_root_scaling_invariance():
    """Same scaled distance -> same overpressure, any yield."""
    a = NuclearBlast(1.0)
    b = NuclearBlast(1000.0)
    r = 500.0
    assert a.overpressure(r) == pytest.approx(b.overpressure(r * 10.0), rel=1e-12)


def test_impulse_scales_with_cube_root_at_same_z():
    a, b = NuclearBlast(1.0), NuclearBlast(1000.0)
    assert b.impulse(5000.0) / a.impulse(500.0) == pytest.approx(10.0, rel=1e-9)


def test_range_solver_inverts_overpressure():
    b = NuclearBlast(15.0)
    for psi in (1, 5, 20):
        r = b.range_for_overpressure(psi_to_pa(psi))
        assert b.overpressure(r) == pytest.approx(psi_to_pa(psi), rel=1e-3)


def test_airburst_lower_ground_zero_pressure_than_surface():
    assert NuclearBlast(15, 600).overpressure(0.0) < NuclearBlast(15, 0).overpressure(0.0)


def test_thermal_inverse_square_without_attenuation():
    t = ThermalSource(10.0, 0.0, 0.35, visibility_m=1e12)
    assert t.fluence(1000.0) / t.fluence(2000.0) == pytest.approx(4.0, rel=1e-6)


def test_thermal_energy_conservation_on_sphere():
    t = ThermalSource(1.0, 0.0, 0.35, visibility_m=1e12)
    r = 3000.0
    assert t.fluence(r) * 4 * np.pi * r**2 == pytest.approx(0.35 * KT_JOULES, rel=1e-6)


def test_impact_energy():
    # 100 m stony body at 20 km/s: m = 3000*pi*100^3/6, E = m v^2 / 2
    mass = 3000.0 * np.pi * 100.0**3 / 6.0
    expected = 0.5 * mass * 20_000.0**2 / KT_JOULES
    assert impact_energy_kt(100.0, 3000.0, 20_000.0) == pytest.approx(expected)
    assert expected == pytest.approx(75_086, rel=1e-3)


def test_brode_scaling_invariance():
    from effects.sources import BrodeBlast

    a, b = BrodeBlast(1.0, 100.0), BrodeBlast(1000.0, 1000.0)
    assert a.overpressure(300.0) == pytest.approx(b.overpressure(3000.0), rel=1e-6)


def test_make_blast_selects_model():
    from effects.sources import BrodeBlast, make_blast

    assert isinstance(make_blast("brode", 10, 0), BrodeBlast)
    assert isinstance(make_blast("kinney_graham", 10, 0), NuclearBlast)
    with pytest.raises(ValueError):
        make_blast("nope", 10, 0)
