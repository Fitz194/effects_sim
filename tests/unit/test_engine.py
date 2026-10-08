import json
from pathlib import Path

import numpy as np
import pytest

from effects.cli import main
from effects.data import Grid
from effects.montecarlo import run
from effects.scenario import Scenario, build_environment

ROOT = Path(__file__).resolve().parents[2]


def small_scenario(**mc):
    return Scenario.model_validate(
        {
            "name": "test",
            "source": {"yield_kt": 10},
            "environment": {
                "half_width_m": 3000,
                "cell_m": 100,
                "synthetic": {"total_population": 50_000, "n_buildings": 2000, "seed": 3},
            },
            "montecarlo": {"n_samples": 20, "seed": 1, **mc},
        }
    )


def test_grid_geometry():
    g = Grid(1000, 100)
    assert g.n == 20
    assert g.centres[0] == pytest.approx(-950)
    r, c, ok = g.cell_index(np.array([-999, 0, 2000]), np.array([-999, 0, 0]))
    assert list(ok) == [True, True, False]
    assert (r[0], c[0]) == (0, 0)


def test_synthetic_population_total():
    sc = small_scenario()
    env, _ = build_environment(sc)
    assert env.population.sum() == pytest.approx(50_000)
    mix = env.class_mix()
    assert np.allclose(mix.sum(axis=-1), 1.0)


def test_engine_runs_and_is_reproducible():
    sc = small_scenario()
    env, lib = build_environment(sc)
    a = run(sc, env, lib)
    b = run(sc, env, lib)
    assert np.array_equal(a.fatalities, b.fatalities)
    assert 0 < a.fatalities.mean() < env.population.sum()
    assert np.allclose(a.building_state_probs.sum(axis=1), 1.0)


def test_bigger_yield_more_harm():
    sc_small, sc_big = small_scenario(), small_scenario()
    sc_big.source.yield_kt = 100
    env, lib = build_environment(sc_small)
    assert run(sc_big, env, lib).fatalities.mean() > run(sc_small, env, lib).fatalities.mean()


def test_scenario_files_validate():
    for f in (ROOT / "scenarios").glob("*.yaml"):
        Scenario.from_yaml(f)


def test_cli_run_writes_outputs(tmp_path):
    main(
        [
            "run",
            str(ROOT / "scenarios" / "synthetic_city.yaml"),
            "--samples",
            "5",
            "--out",
            str(tmp_path),
        ]
    )
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["n_samples"] == 5
    for name in ("buildings.csv", "fields.npz", "fatalities.png", "buildings.png"):
        assert (tmp_path / name).exists()
