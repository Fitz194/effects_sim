"""Monte Carlo against closed-form answers.

If a fragility curve has median theta and dispersion beta, and the median carries an independent
lognormal epistemic factor exp(sigma * eps), the mean exceedance probability at fixed demand p is

    E[P] = Phi( ln(p / theta) / sqrt(beta^2 + sigma^2) )

We switch off every other uncertainty so the demand on each building is fixed, run the full engine,
and compare the mean exceedance it reports against that formula.
"""

import numpy as np
import pytest
from scipy.special import ndtr

from effects.montecarlo import run
from effects.scenario import Scenario, build_environment, nominal_sources

pytestmark = pytest.mark.validation

SIGMA = 0.3


def scenario(n):
    return Scenario.model_validate(
        {
            "name": "mc_analytic",
            "source": {"yield_kt": 20},
            "environment": {
                "half_width_m": 4000,
                "cell_m": 200,
                "synthetic": {"total_population": 10_000, "n_buildings": 1500, "seed": 7},
            },
            "montecarlo": {
                "n_samples": n,
                "seed": 123,
                "uncertainty": {
                    "yield_sigma_ln": 0.0,
                    "blast_fraction_range": None,
                    "reflection_factor_range": None,
                    "thermal_fraction_range": None,
                    "visibility_range_m": None,
                    "fragility_median_sigma_ln": SIGMA,
                    "population_sigma_ln": 0.0,
                    "indoor_fraction_range": None,
                    "blast_model_sigma_ln": 0.0,
                },
            },
        }
    )


def test_mean_exceedance_matches_closed_form():
    n = 4000
    sc = scenario(n)
    env, lib = build_environment(sc)
    res = run(sc, env, lib)
    blast, _ = nominal_sources(sc)
    b = env.buildings
    p = blast.overpressure(np.hypot(b["x"], b["y"]).to_numpy())

    mc_ex = 1.0 - np.cumsum(res.building_state_probs, axis=1)[:, :-1]  # P(DS >= state)
    worst = 0.0
    for cls_name, fs in lib.items():
        sel = (b["cls"] == cls_name).to_numpy()
        if not sel.any():
            continue
        for j, curve in enumerate(fs.curves):
            analytic = ndtr(np.log(p[sel] / curve.median) / np.hypot(curve.beta, SIGMA))
            # Monotone enforcement in FragilitySet only bites if curves cross; default library
            # curves don't cross, so the closed form applies state by state.
            worst = max(worst, np.max(np.abs(mc_ex[sel, j] - analytic)))
    # Standard error of a Bernoulli-like mean with n=4000 is <= 0.008; allow ~4 SE.
    assert worst < 0.03


def test_total_fatalities_converge():
    """Mean of totals is stable between independent seeds (law of large numbers)."""
    sc_a, sc_b = scenario(1500), scenario(1500)
    sc_b.montecarlo.seed = 999
    env, lib = build_environment(sc_a)
    a, b = run(sc_a, env, lib), run(sc_b, env, lib)
    assert a.fatalities.mean() == pytest.approx(b.fatalities.mean(), rel=0.03)
